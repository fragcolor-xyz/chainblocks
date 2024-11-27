/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include "wires.hpp"
#include <cassert>
#include <shards/core/async.hpp>
#include <shards/core/taskflow.hpp>
#include <shards/core/platform.hpp>
#include <shards/core/brancher.hpp>
#include <shards/core/module.hpp>
#include <shards/common_types.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/ops_internal.hpp>
#include <shards/core/wire_doppelganger_pool.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <shards/inlined.hpp>

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
// Remove define from winspool.h
#ifdef MAX_PRIORITY
#undef MAX_PRIORITY
#endif
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#endif

namespace shards {
void WireBase::resetComposition() {
  if (wire) {
    if (wire->composeResult) {
      shards::arrayFree(wire->composeResult->requiredInfo);
      shards::arrayFree(wire->composeResult->exposedInfo);
      wire->composeResult.reset();
    }
  }
}
std::unordered_set<const SHWire *> &WireBase::gatheringWires() {
#ifdef WIN32
  // TODO FIX THIS
  // we have to leak.. or windows tls emulation will crash at process end
  thread_local std::unordered_set<const SHWire *> *wires = new std::unordered_set<const SHWire *>();
  return *wires;
#else
  thread_local std::unordered_set<const SHWire *> wires;
  return wires;
#endif
}

void WireBase::verifyAlreadyComposed(const SHInstanceData &data, IterableExposedInfo &shared) {
  SHLOG_TRACE("WireBase::verifyAlreadyComposed, source: {} composing: {} inputType: {}", data.wire ? data.wire->name : nullptr,
              wire->name, data.inputType);
  // verify input type
  if (!passthrough && data.inputType != wire->inputType && !wire->ignoreInputTypeCheck) {
    throw ComposeError(fmt::format(
        "Attempted to call an already composed wire with a different input type! wire: {}, old type: {}, new type: {}",
        wire->name, (SHTypeInfo)wire->inputType, data.inputType));
  }

  // ensure requirements match our input data
  for (auto &req : wire->composeResult->requiredInfo) {
    // find each in shared
    auto name = req.name;
    auto res = std::find_if(shared.begin(), shared.end(), [name](SHExposedTypeInfo &x) {
      std::string_view xNameView(x.name);
      return name == xNameView;
    });
    if (res == shared.end()) {
      throw ComposeError(
          fmt::format("Attempted to call an already composed wire ({}) with a missing required variable: {}", wire->name, name));
    }
  }
}

void WireBase::resolveWire() {
  // Actualize the wire here, if we are deserialized
  // wire might already be populated!
  if (!wire) {
    if (wireref->valueType == SHType::Wire) {
      wire = SHWire::sharedFromRef(wireref->payload.wireValue);
    } else if (wireref->valueType == SHType::String) {
      auto sv = SHSTRVIEW((*wireref));
      std::string s(sv);
      SHLOG_DEBUG("WireBase: Resolving wire {}", sv);
      wire = GetGlobals().GlobalWires[s];
    } else {
      wire = IntoWire{} //
                 .defaultWireName("inline-shards")
                 .defaultLooped(false)
                 .var(wireref);
    }
  }
}

SHTypeInfo WireBase::compose(const SHInstanceData &data) {
  resolveWire();

  // Easy case, no wire...
  if (!wire)
    return data.inputType;

  assert(data.wire);

  if (wire.get() == data.wire) {
    SHLOG_DEBUG("WireBase::compose early return, data.wire == wire, name: {}", wire->name);
    return data.inputType; // we don't know yet...
  }

  auto mesh = data.wire->mesh.lock();
  shassert(mesh && "Mesh should be set");

  checkWireMesh(data.wire->name, mesh.get());

  auto currentMesh = wire->mesh.lock();
  if (currentMesh && currentMesh != mesh)
    throw ComposeError(fmt::format("Attempted to compose a wire ({}) that is already part of another mesh!", wire->name));

  wire->mesh = data.wire->mesh;
  wire->id = data.wire->id;

  shassert(data.privateContext && "Visited wires should be set");
  auto composeCtx = reinterpret_cast<CompositionContext *>(data.privateContext);
  shassert(composeCtx && "Composition context should be set");
  auto visitedIt = composeCtx->visitedWires.find(wire.get()); // should be race free
  if (visitedIt != composeCtx->visitedWires.end()) {
    // but visited does not mean composed...
    if (wire->composeResult && activating) {
      IterableExposedInfo shared(data.shared);
      verifyAlreadyComposed(data, shared);
    }
    return visitedIt->second;
  }

  // avoid stack-overflow
  if (wire->isRoot || gatheringWires().count(wire.get())) {
    SHLOG_DEBUG("WireBase::compose early return, wire is being visited, name: {}", wire->name);
    return data.inputType; // we don't know yet...
  }

  SHLOG_TRACE("WireBase::compose, source: {} composing: {} inputType: {}", data.wire->name, wire->name, data.inputType);

  // we can add early in this case!
  // useful for Resume/Start
  if (passthrough) {
    auto [_, done] = composeCtx->visitedWires.emplace(wire.get(), CoreInfo::AnyType);
    if (done) {
      SHLOG_TRACE("Pre-Marking as composed: {} ptr: {}", wire->name, (void *)wire.get());
    }
  } else if (mode == Stepped) {
    auto [_, done] = composeCtx->visitedWires.emplace(wire.get(), CoreInfo::AnyType);
    if (done) {
      SHLOG_TRACE("Pre-Marking as composed: {} ptr: {}", wire->name, (void *)wire.get());
    }
  }

  // and the subject here
  gatheringWires().insert(wire.get());
  DEFER(gatheringWires().erase(wire.get()));

  auto dataCopy = data;
  dataCopy.wire = wire.get();
  IterableExposedInfo shared(data.shared);
  IterableExposedInfo sharedCopy;
  if (!wire->pure) {
    if (mode == RunWireMode::Async && !capturing) {
      // keep only globals
      auto end = std::remove_if(shared.begin(), shared.end(), [](const SHExposedTypeInfo &x) { return !x.global; });
      sharedCopy = IterableExposedInfo(shared.begin(), end);
    } else {
      // we allow Detached but they need to be referenced during warmup
      sharedCopy = shared;
    }
  }

  dataCopy.shared = sharedCopy;

  // make sure to compose only once...
  if (!wire->composeResult) {
    SHLOG_TRACE("Running {} compose, pure: {}", wire->name, wire->pure);

    if (data.requiredVariables) {
      wire->requirements.clear();
    }

    wire->composeResult = composeWire(wire.get(), dataCopy);

    IterableExposedInfo exposing(wire->composeResult->exposedInfo);
    // keep only globals
    exposedInfo = IterableExposedInfo(
        exposing.begin(), std::remove_if(exposing.begin(), exposing.end(), [](SHExposedTypeInfo &x) { return !x.global; }));

    // Notice we DON'T need here to merge the required info here even if we had data.requiredVariables non null

    SHLOG_TRACE("Wire {} composed", wire->name);
  } else {
    SHLOG_TRACE("Skipping {} compose", wire->name);

    verifyAlreadyComposed(data, shared);
  }

  // write output type
  SHTypeInfo wireOutput = wire->outputType;
  composeCtx->visitedWires.insert_or_assign(wire.get(), wireOutput);
  SHLOG_TRACE("Marking as composed: {} ptr: {} inputType: {} outputType: {}", wire->name, (void *)wire.get(), *wire->inputType,
              wire->outputType);

#if SH_CORO_NEED_STACK_MEM
  // Propagate stack size
  data.wire->stackSize = std::max<size_t>(data.wire->stackSize, wire->stackSize);
#endif

  auto outputType = data.inputType;

  if (!passthrough) {
    if (mode == Inline)
      outputType = wireOutput;
    else if (mode == Stepped)
      outputType = CoreInfo::AnyType; // unpredictable
    else
      outputType = data.inputType;
  }

  return outputType;
}

struct Wait : public WireBase {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }

  void setup() {
    activating = false;  // this is needed to pass validation in compose
    passthrough = false; // also need this to have proper compose output type
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnoredOrPass; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("If Passthrough is true, this shard outputs the input value, passed through unchanged. Otherwise, it "
                   "outputs the output of the Wire it waited for.");
  }

  SHOptionalString help() {
    return SHCCSTR("Waits for the specified Wire to complete before resuming "
                   "execution of the current Wire.");
  }

  SHExposedTypeInfo _requiredWire{};
  ParamVar _timeout{};

  static SHParametersInfo parameters() { return waitParamsInfo; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    case 1:
      passthrough = value.payload.boolValue;
      break;
    case 2:
      _timeout = value;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    case 1:
      return Var(passthrough);
    case 2:
      return _timeout;
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (wireref.isVariable()) {
      _requiredWire = SHExposedTypeInfo{wireref.variableName(), SHCCSTR("The wire to run."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  void warmup(SHContext *ctx) {
    WireBase::warmup(ctx);
    _timeout.warmup(ctx);
  }

  void cleanup(SHContext *context) {
    _timeout.cleanup();
    if (wireref.isVariable())
      wire = nullptr;
    WireBase::cleanup(context);
  }

  const SHVar &activate(SHContext *context, const SHVar &input) {
    ensureWire();

    if (unlikely(!wire)) {
      SHLOG_WARNING("Wait's wire is void");
      return input;
    } else {
      auto timeout = _timeout.get();
      if (timeout.valueType == SHType::Float) {
        SHTime start = SHClock::now();
        auto dTimeout = SHDuration(timeout.payload.floatValue);
        while (wire->context != context && isRunning(wire.get())) {
          SH_SUSPEND(context, 0);

          // Deal with timeout
          SHTime now = SHClock::now();
          if ((now - start) > dTimeout) {
            SHLOG_ERROR("Wait: Wire {} timed out", wire->name);
            throw ActivationError("Wait: Wire timed out");
          }
        }
      } else {
        // Make sure to actually wait only if the wire is running on another context.
        while (wire->context != context && isRunning(wire.get())) {
          SH_SUSPEND(context, 0);
        }
      }

      if (!wire->finishedError.empty() || wire->state == SHWire::State::Failed) {
        SHLOG_TRACE("Waiting wire: {} failed with error: {}", wire->name, wire->finishedError);
        // if the wire has errors we need to propagate them
        // we can avoid interruption using Maybe shards
        throw ActivationError(wire->finishedError);
      }

      if (passthrough) {
        return input;
      } else {
        // Check for potential issues with object types returned by the wire
        if (wire->finishedOutput.valueType == SHType::Object) {
          if ((wire->finishedOutput.flags & SHVAR_FLAGS_USES_OBJINFO) != SHVAR_FLAGS_USES_OBJINFO) {
            SHLOG_ERROR("Object returned by wire '{}' does not have SHVAR_FLAGS_USES_OBJINFO flag set. This may lead to "
                        "premature destruction.",
                        wire->name);
            throw ActivationError(fmt::format("Object returned by wire '{}' lacks reference counting", wire->name));
          }
          if (!wire->finishedOutput.objectInfo || !wire->finishedOutput.objectInfo->reference) {
            SHLOG_ERROR("Object returned by wire '{}' has no reference counting. This will lead to premature destruction.",
                        wire->name);
            throw ActivationError(fmt::format("Object returned by wire '{}' lacks reference counting", wire->name));
          }
        }
        return wire->finishedOutput;
      }
    }
  }
};

struct IsRunning : public WireBase {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }

  void setup() {
    activating = false;  // this is needed to pass validation in compose
    passthrough = false; // also need this to have proper compose output type
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("This shard will either return true if the specified Wire is still running, or false if it has ended.");
  }

  SHOptionalString help() {
    return SHCCSTR("Checks if a Wire is running and outputs true if it is, false if otherwise. (Note that a looped Wire will "
                   "always be running and thus will always return true)");
  }

  SHExposedTypeInfo _requiredWire{};

  static inline Parameters params{{"Wire", SHCCSTR("The wire to check."), {WireVarTypes}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (wireref.isVariable()) {
      _requiredWire = SHExposedTypeInfo{wireref.variableName(), SHCCSTR("The wire to check."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    resolveWire();
    if (wire) {
      auto currentMesh = wire->mesh.lock();
      checkWireMesh(data.wire->name, currentMesh.get());
    }
    return CoreInfo::BoolType;
  }

  void warmup(SHContext *ctx) { WireBase::warmup(ctx); }

  void cleanup(SHContext *context) {
    if (wireref.isVariable())
      wire = nullptr;
    WireBase::cleanup(context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    ensureWire();

    if (unlikely(!wire)) {
      return Var::False;
    } else {
      // Make sure to actually wait only if the wire is running on another context.
      return Var(isRunning(wire.get()));
    }
  }
};

struct Peek : public WireBase {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }

  void setup() {
    activating = false;  // this is needed to pass validation in compose
    passthrough = false; // also need this to have proper compose output type
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHOptionalString outputHelp() {
    return SHCCSTR(
        "This shard either outputs none if the peeked Wire is still in progress, or the peeked Wire's output if it has ended.");
  }

  SHOptionalString help() {
    return SHCCSTR("Checks if another Wire has ended (Note that a looped Wire will never end). Outputs the Wire's output if it "
                   "has ended, or none if it is still in progress.");
  }

  SHExposedTypeInfo _requiredWire{};

  static inline Parameters params{{"Wire", SHCCSTR("The wire to check."), {WireVarTypes}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (wireref.isVariable()) {
      _requiredWire = SHExposedTypeInfo{wireref.variableName(), SHCCSTR("The wire to check."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    WireBase::compose(data);
    return CoreInfo::AnyType;
  }

  void warmup(SHContext *ctx) { WireBase::warmup(ctx); }

  void cleanup(SHContext *context) {
    if (wireref.isVariable())
      wire = nullptr;
    WireBase::cleanup(context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    ensureWire();

    if (unlikely(!wire)) {
      return Var::Empty;
    } else {
      // Make sure to actually wait only if the wire is running on another context.
      if (wire->context == context || isRunning(wire.get())) {
        return Var::Empty;
      }

      if (!wire->finishedError.empty() || wire->state == SHWire::State::Failed) {
        SHLOG_TRACE("Waiting wire: {} failed with error: {}", wire->name, wire->finishedError);
        // if the wire has errors we need to propagate them
        // we can avoid interruption using Maybe shards
        throw ActivationError(wire->finishedError);
      }

      // Check for potential issues with object types returned by the wire
      if (wire->finishedOutput.valueType == SHType::Object) {
        if ((wire->finishedOutput.flags & SHVAR_FLAGS_USES_OBJINFO) != SHVAR_FLAGS_USES_OBJINFO) {
          SHLOG_ERROR("Object returned by wire '{}' does not have SHVAR_FLAGS_USES_OBJINFO flag set. This may lead to premature "
                      "destruction.",
                      wire->name);
          throw ActivationError(fmt::format("Object returned by wire '{}' lacks reference counting", wire->name));
        }
        if (!wire->finishedOutput.objectInfo || !wire->finishedOutput.objectInfo->reference) {
          SHLOG_ERROR("Object returned by wire '{}' has no reference counting. This will lead to premature destruction.",
                      wire->name);
          throw ActivationError(fmt::format("Object returned by wire '{}' lacks reference counting", wire->name));
        }
      }
      return wire->finishedOutput;
    }
  }
};

struct StopWire : public WireBase {
  SHOptionalString help() { return SHCCSTR("Either stops the execution of a specified Wire or the current Wire."); }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnoredOrPass; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Depending on what is specified in the Passthrough parameter, this shard either outputs the input value, "
                   "passed through unchanged or it outputs the output of the stopped Wire. Note that if it outputs the output of "
                   "the stopped wire, it will be of Type::Any and thus should be checked or converted to the appropriate Type.");
  }

  void setup() { passthrough = true; }

  OwnedVar _output{};
  SHExposedTypeInfo _requiredWire{};

  SHTypeInfo _inputType{};

  std::weak_ptr<SHWire> _wire;

  SHTypeInfo compose(SHInstanceData &data) {
    assert(data.wire);

    if (wireref->valueType == SHType::None) {
      _inputType = data.inputType;
      data.wire->getComposeData().outputTypes.push_back(data.inputType);
    } else {
      resolveWire();
      if (wire) {
        auto currentMesh = wire->mesh.lock();
        checkWireMesh(data.wire->name, currentMesh.get());
      }
    }

    if (passthrough) {
      return data.inputType;
    } else {
      return CoreInfo::AnyType;
    }
  }

  void cleanup(SHContext *context) {
    if (wireref.isVariable())
      wire = nullptr;
    WireBase::cleanup(context);
  }

  static SHParametersInfo parameters() { return stopWireParamsInfo; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    case 1:
      passthrough = value.payload.boolValue;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    case 1:
      return Var(passthrough);
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (wireref.isVariable()) {
      _requiredWire = SHExposedTypeInfo{wireref.variableName(), SHCCSTR("The wire to stop."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    ensureWire();

    if (!wire || wire.get() == context->currentWire()) {
      // in this case we stop the current wire
      context->stopFlow(input);
      return input;
    } else {
      shards::stop(wire.get(), nullptr, context);
      if (passthrough) {
        return input;
      } else {
        _output = wire->finishedOutput;
        return _output;
      }
    }
  }
};

struct SuspendWire : public WireBase {
  void setup() {
    activating = false; // this is needed to pass validation in compose
    passthrough = true; // also need this to have proper compose output type
  }

  SHOptionalString help() {
    return SHCCSTR("Pauses a specified Wire's execution. If no Wire is specified, pauses the current wire.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  SHExposedTypeInfo _requiredWire{};

  void cleanup(SHContext *context) {
    if (wireref.isVariable())
      wire = nullptr;
    WireBase::cleanup(context);
  }

  static inline Parameters params{{"Wire", SHCCSTR("The wire to pause."), {WireVarTypes}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (wireref.isVariable()) {
      _requiredWire = SHExposedTypeInfo{wireref.variableName(), SHCCSTR("The wire to suspend."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    resolveWire();
    if (wire) {
      auto currentMesh = wire->mesh.lock();
      checkWireMesh(data.wire->name, currentMesh.get());
    }
    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    ensureWire();

    if (unlikely(!wire)) {
      // in this case we pause the current wire
      context->currentWire()->paused = true;
    } else {
      // pause the wire directly
      wire->paused = true;
    }

    return input;
  }
};

struct ResumeWire : public WireBase {
  void setup() {
    activating = false; // this is needed to pass validation in compose
    passthrough = true; // also need this to have proper compose output type
  }

  SHOptionalString help() { return SHCCSTR("Resumes another Wire (previously paused using Suspend)."); }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  SHExposedTypeInfo _requiredWire{};

  void cleanup(SHContext *context) {
    if (wireref.isVariable())
      wire = nullptr;
    WireBase::cleanup(context);
  }

  static inline Parameters params{{"Wire", SHCCSTR("The Wire to resume."), {WireVarTypes}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (wireref.isVariable()) {
      _requiredWire = SHExposedTypeInfo{wireref.variableName(), SHCCSTR("The wire to resume."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    resolveWire();
    if (wire) {
      auto currentMesh = wire->mesh.lock();
      checkWireMesh(data.wire->name, currentMesh.get());
    }
    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    ensureWire();

    wire->paused = false;

    return input;
  }
};

struct SwitchTo : public WireBase {
  std::deque<ParamVar> _vars;
  bool fromStart{false};
  bool stopOnCleanup{false};

  static SHOptionalString help() { return SHCCSTR("Suspends the current Wire and switches execution to the specified Wire."); }

  static SHOptionalString inputHelp() { return InputBaseWire; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output of this shard is the output of the Wire that execution was switched to, upon switching back to "
                   "the parent Wire.");
  }

  void setup() {
    // we use those during WireBase::compose
    passthrough = true;
    mode = RunWireMode::Async;
    capturing = true;
  }

  static inline Parameters params{
      {"Wire",
       SHCCSTR("The Wire to switch to. If none is provided, the Wire is switched back to the parent Wire it was switched "
               "from (should a parent Wire exist)."),
       {WireTypes}},
      {"Restart",
       SHCCSTR("If the Wire should always restart from the beginning instead of resuming from its previous state."),
       {CoreInfo::BoolType}},
      {"StopOnCleanup",
       SHCCSTR("If the child Wire should be stopped when the parent Wire is Destroyed or Stopped."),
       {CoreInfo::BoolType}}};

  static SHParametersInfo parameters() { return params; }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  SHExposedTypesInfo _mergedReqs;
  void destroy() { arrayFree(_mergedReqs); }
  SHExposedTypesInfo requiredVariables() { return _mergedReqs; }

  SHTypeInfo compose(const SHInstanceData &data) {
    // SwitchTo need to capture all it needs, so we need deeper informations
    // this is triggered by populating requiredVariables variable
    resolveWire();
    if (wire) {
      if (data.wire == wire.get()) {
        SHLOG_ERROR("SwitchTo: wire {} cannot switch to itself", wire->name);
        throw ComposeError("SwitchTo: wire cannot switch to itself");
      }

      auto dataCopy = data;
      dataCopy.requiredVariables = &wire->requirements;
      for (auto &req : dataCopy.shared) {
        if (!req.global)
          req.tracked = false;
      }

      WireBase::compose(dataCopy);

      // build the list of variables to capture and inject into spawned chain
      _vars.clear();
      arrayResize(_mergedReqs, 0);
      for (auto &avail : data.shared) {
        auto it = wire->requirements.find(avail.name);
        if (it != wire->requirements.end()) {
          if (!avail.global) {
            // Capture if not global as we need to copy it!
            SHLOG_TRACE("SwitchTo: adding variable to requirements: {}, wire {}", avail.name, wire->name);
            SHVar ctxVar{};
            ctxVar.valueType = SHType::ContextVar;
            ctxVar.payload.stringValue = avail.name;
            ctxVar.payload.stringLen = strlen(avail.name);
            auto &p = _vars.emplace_back();
            p = ctxVar;
          }

          arrayPush(_mergedReqs, it->second);
        }
      }
    } else {
      WireBase::compose(data); // we still need this to have proper state likely
    }

    return CoreInfo::AnyType; // can be anything
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    case 1:
      fromStart = value.payload.boolValue;
      break;
    case 2:
      stopOnCleanup = value.payload.boolValue;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    case 1:
      return Var(fromStart);
    case 2:
      return Var(stopOnCleanup);
    default:
      return Var::Empty;
    }
  }

  // NO cleanup, other wires reference this wire but should not stop it
  // An arbitrary wire should be able to resume it!
  // Cleanup mechanics still to figure, for now ref count of the actual wire
  // symbol, TODO maybe use SHVar refcount!

  void warmup(SHContext *context) {
    WireBase::warmup(context);

    for (auto &v : _vars) {
      SHLOG_TRACE("SwitchTo: warming up variable: {}", v.variableName());
      v.warmup(context);
    }

    // Fixup the wire id, since compose may be skipped, but id changed
    SHWire *currentWire = context->currentWire();
    if (wire && currentWire) {
      wire->id = currentWire->id;
    }
  }

  void cleanup(SHContext *context) {
    // Notice we don't stop wire normally!
    // it does not make sense cos maybe another wire wants to pick it up and resume it!
    // we should not be the owners of the wire, we just want to switch to it!
    // think like a state machine, we are just switching to another state
    if (stopOnCleanup) {
      if (wire) {
        SHLOG_TRACE("SwitchTo: stopping wire: {}", wire->name);
        shards::stop(wire.get());
      }
    }

    for (auto &v : _vars) {
      v.cleanup();
    }

    WireBase::cleanup(context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHWire *previousWithCoro = nullptr;
    for (auto it = context->wireStack.rbegin(); it != context->wireStack.rend(); ++it) {
      if (coroutineValid((*it)->coro)) {
        previousWithCoro = *it;
        break;
      }
    }

    if (!previousWithCoro) {
      throw ActivationError("SwitchTo, no previous wire with coroutine found.");
    }

    auto pWire = [&] {
      if (!wire) {
        if (previousWithCoro->resumer) {
          SHLOG_TRACE("SwitchTo, wire not found, using resumer: {}", previousWithCoro->resumer->name);
          return previousWithCoro->resumer;
        } else {
          throw ActivationError("SwitchTo, wire not found.");
        }
      } else {
        return wire.get();
      }
    }();

    // Allow to re run wires, or simply restart
    if (fromStart || shards::hasEnded(pWire)) {
      shards::stop(pWire);
    }

    // capture variables
    for (auto &v : _vars) {
      auto &var = v.get();
      std::string_view name = v.variableNameView();
      auto &v_ref = pWire->getVariable(toSWL(name));
      cloneVar(v_ref, var);
    }

    // Prepare if no callc was called
    if (!coroutineValid(pWire->coro)) {
      pWire->mesh = context->main->mesh.lock();
      shards::prepare(pWire);

      // handle early failure
      if (pWire->state == SHWire::State::Failed) {
        // destroy fresh cloned variables
        for (auto &v : _vars) {
          std::string_view name = v.variableNameView();
          destroyVar(pWire->getVariable(toSWL(name)));
        }
        SHLOG_ERROR("Wire {} failed to start.", pWire->name);
        throw ActivationError("Wire failed to start.");
      }
    }

    // we should be valid as this shard should be dependent on current
    // do this here as stop/prepare might overwrite
    if (pWire->resumer == nullptr)
      pWire->resumer = previousWithCoro;

    if (pWire != previousWithCoro && pWire->childWire != previousWithCoro) {
      // all we need to do is set pWire as childWire of the previousWireCoro
      shassert(previousWithCoro->childWire == nullptr && "SwitchTo: previousWireCoro already has a childWire");
      previousWithCoro->childWire = pWire;
    } else {
      // we are switching to ourselves, we need to set the childWire to nullptr
      pWire->childWire = nullptr;
    }

    // Start it if not started
    if (!shards::isRunning(pWire)) {
      shards::start(pWire, input);
    }

    // And normally we just delegate the Mesh
    // the following will suspend this current wire
    // and in mesh tick when re-evaluated tick will
    // resume with the wire we just set above!
    shards::suspend(context, 0);

    // We will end here when we get resumed!
    // When we are here, pWire is suspended, (could be in the middle of a loop, anywhere!)

    // reset resumer and childWire
    previousWithCoro->childWire = nullptr;
    pWire->resumer = nullptr;

    // return the last or final output of pWire
    if (pWire->state > SHWire::State::IterationEnded) { // failed or ended
      return pWire->finishedOutput;
    } else {
      return pWire->previousOutput;
    }
  }
};

struct Recur : public WireBase {
  std::weak_ptr<SHWire> _wwire;
  SHWire *_wire;
  std::deque<ParamVar> _vars;
  std::vector<SHSeq> _storage;

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR(
        "After the first cycle of Recur, the output of the Wire that calls Recur will be fed back as input for the next cycle.");
  }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output of Recur will be the output of the Wire that calls it."); }

  SHOptionalString help() {
    return SHCCSTR("The Recur shard executes the Wire that calls it recursively, using the output of the Wire as input again, "
                   "until the base cases are "
                   "reached. It then combines the results to produce the final result. For the shard not to Recur endlessly, a "
                   "base case needs to be defined, usually through a When or If shard.");
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    // set current wire as `wire`
    _wwire = data.wire->shared_from_this();

    // find all variables to store in current wire
    // use vector in the end.. cos slightly faster
    _vars.clear();
    for (auto &shared : data.shared) {
      if (!shared.global) {
        SHVar ctxVar{};
        ctxVar.valueType = SHType::ContextVar;
        ctxVar.payload.stringValue = shared.name;
        ctxVar.payload.stringLen = strlen(shared.name);
        auto &p = _vars.emplace_back();
        p = ctxVar;
      }
    }
    return WireBase::compose(data);
  }

  void warmup(SHContext *ctx) {
    _storage.resize(_vars.size());
    for (auto &v : _vars) {
      v.warmup(ctx);
    }
    auto swire = _wwire.lock();
    assert(swire);
    _wire = swire.get();
  }

  void cleanup(SHContext *context) {
    for (auto &v : _vars) {
      v.cleanup();
    }
    // force releasing resources
    for (size_t i = 0; i < _storage.size(); i++) {
      // must release on capacity
      for (uint32_t j = 0; j < _storage[i].cap; j++) {
        destroyVar(_storage[i].elements[j]);
      }
      arrayFree(_storage[i]);
    }
    _storage.resize(0);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    // store _vars
    for (size_t i = 0; i < _vars.size(); i++) {
      const auto len = _storage[i].len;
      arrayResize(_storage[i], len + 1);
      cloneVar(_storage[i].elements[len], _vars[i].get());
    }

    // (Do self)
    // Run within the root flow
    auto runRes = runSubWire(_wire, context, input);
    if (unlikely(runRes.state == SHRunWireOutputState::Failed)) {
      return Var::Empty;
    }

    // restore _vars
    for (size_t i = 0; i < _vars.size(); i++) {
      auto pops = arrayPop<SHSeq, SHVar>(_storage[i]);
      cloneVar(_vars[i].get(), pops);
    }

    return runRes.output;
  }
};

struct WireNotFound : public ActivationError {
  WireNotFound() : ActivationError("Could not find a wire to run") {}
};

template <class T> struct BaseLoader : public BaseRunner {
  SHTypeInfo _inputTypeCopy{};
  IterableExposedInfo _sharedCopy;
  bool _onWorkerThread = false;

  SHTypeInfo compose(const SHInstanceData &data) {
    WireBase::compose(data); // notice we skip BaseRunner::compose

    _onWorkerThread = data.onWorkerThread;

    _inputTypeCopy = data.inputType;
    IterableExposedInfo sharedStb(data.shared);

    if (mode == RunWireMode::Async) {
      // keep only globals
      auto end = std::remove_if(sharedStb.begin(), sharedStb.end(), [](const SHExposedTypeInfo &x) { return !x.global; });
      _sharedCopy = IterableExposedInfo(sharedStb.begin(), end);
    } else {
      // full copy
      _sharedCopy = IterableExposedInfo(sharedStb.begin(), sharedStb.end()); // notice to make full copy
    }

    if (mode == RunWireMode::Inline || mode == RunWireMode::Stepped) {
      // If inline allow wires to receive a result
      return CoreInfo::AnyType;
    } else {
      return data.inputType;
    }
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 1:
      mode = RunWireMode(value.payload.enumValue);
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 1:
      return Var::Enum(mode, CoreCC, 'runC');
    default:
      break;
    }
    return Var::Empty;
  }

  void cleanup(SHContext *context) { BaseRunner::cleanup(context); }

  SHVar activateWire(SHContext *context, const SHVar &input) {
    if (unlikely(!wire))
      throw WireNotFound();

    if (mode == RunWireMode::Async) {
      activateDetached(context, input);
    } else if (mode == RunWireMode::Stepped) {
      activateStepMode(context, input);
      return wire->previousOutput;
    } else {
      if (!wire->warmedUp) {
        // ok this can happen if Stop was called on the wire
        // and we are running it again
        // Stop is a legit way to clear the wire state
        wire->warmup(context);
      }
      // Run within the root flow
      const auto runRes = runSubWire(wire.get(), context, input);
      if (likely(runRes.state != SHRunWireOutputState::Failed)) {
        return runRes.output;
      }
    }

    return input;
  }
};

struct WireRunner : public BaseLoader<WireRunner> {
  static inline Parameters params{
      {"Wire", SHCCSTR("The wire variable to compose and run."), {CoreInfo::WireType, CoreInfo::WireVarType}},
      {"Mode",
       SHCCSTR("The way to run the wire. Inline: will run the sub wire "
               "inline within the root wire, a pause in the child wire will "
               "pause the root too; Detached: will run the wire separately in "
               "the same mesh, a pause in this wire will not pause the root; "
               "Stepped: the wire will run as a child, the root will tick the "
               "wire every activation of this shard and so a child pause "
               "won't pause the root."),
       {RunWireModeEnumInfo::Type}}};

  static SHParametersInfo parameters() { return params; }

  ParamVar _wire{};
  SHVar _wireHash{};
  SHWire *_wirePtr = nullptr;
  SHExposedTypeInfo _requiredWire{};

  void setParam(int index, const SHVar &value) {
    if (index == 0) {
      _wire = value;
    } else {
      BaseLoader<WireRunner>::setParam(index, value);
    }
  }

  SHVar getParam(int index) {
    if (index == 0) {
      return _wire;
    } else {
      return BaseLoader<WireRunner>::getParam(index);
    }
  }

  void cleanup(SHContext *context) {
    BaseLoader<WireRunner>::cleanup(context);
    _wire.cleanup();
    _wirePtr = nullptr;
  }

  void warmup(SHContext *context) {
    BaseLoader<WireRunner>::warmup(context);
    _wire.warmup(context);
  }

  SHExposedTypesInfo requiredVariables() {
    if (_wire.isVariable()) {
      _requiredWire = SHExposedTypeInfo{_wire.variableName(), SHCCSTR("The wire to run."), CoreInfo::WireType};
      return {&_requiredWire, 1, 0};
    } else {
      return {};
    }
  }

  void deferredCompose(SHContext *context) {
    SHInstanceData data{};
    data.inputType = _inputTypeCopy;
    data.shared = _sharedCopy;
    data.wire = wire.get();
    wire->mesh = context->main->mesh;

    // avoid stack-overflow
    if (gatheringWires().count(wire.get()))
      return; // we don't know yet...

    gatheringWires().insert(wire.get());
    DEFER(gatheringWires().erase(wire.get()));

    // We need to validate the sub wire to figure it out!
    auto res = composeWire(wire.get(), data);

    shards::arrayFree(res.exposedInfo);
    shards::arrayFree(res.requiredInfo);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto wireVar = _wire.get();
    wire = SHWire::sharedFromRef(wireVar.payload.wireValue);
    if (unlikely(!wire))
      return input;

    if (_wireHash.valueType == SHType::None || _wireHash != wire->composedHash || _wirePtr != wire.get()) {
      if (!_onWorkerThread) {
        // Compose and hash in a thread
        await(
            context,
            [this, context, wireVar]() {
              deferredCompose(context);
              wire->composedHash = shards::hash(wireVar);
            },
            [] {});
      } else {
        deferredCompose(context);
        wire->composedHash = shards::hash(wireVar);
      }

      _wireHash = wire->composedHash;
      _wirePtr = wire.get();
      doWarmup(context);
    }

    return BaseLoader<WireRunner>::activateWire(context, input);
  }
};

struct CapturingSpawners : public WireBase {
  SHExposedTypesInfo _mergedReqs;
  IterableExposedInfo _sharedCopy;
  std::deque<ParamVar> _vars;

  void destroy() { arrayFree(_mergedReqs); }

  SHExposedTypesInfo requiredVariables() { return _mergedReqs; }

  void compose(const SHInstanceData &data, const SHTypeInfo &inputType) {
    resolveWire();

    if (!wire) {
      throw ComposeError("CapturingSpawners: wire not found");
    }

    // Wire needs to capture all it needs, so we need deeper informations
    // this is triggered by populating requiredVariables variable
    auto dataCopy = data;
    dataCopy.requiredVariables = &wire->requirements;
    dataCopy.inputType = inputType;

    WireBase::compose(dataCopy); // discard the result, we do our thing here

    // build the list of variables to capture and inject into spawned chain
    _vars.clear();
    arrayResize(_mergedReqs, 0);
    for (auto &avail : data.shared) {
      auto it = wire->requirements.find(avail.name);
      if (it != wire->requirements.end()) {
        if (!avail.global) {
          // Capture if not global as we need to copy it!
          SHLOG_TRACE("CapturingSpawners: adding variable to requirements: {}, wire {}", avail.name, wire->name);
          SHVar ctxVar{};
          ctxVar.valueType = SHType::ContextVar;
          ctxVar.payload.stringValue = avail.name;
          ctxVar.payload.stringLen = strlen(avail.name);
          auto &p = _vars.emplace_back();
          p = ctxVar;
        }

        arrayPush(_mergedReqs, it->second);
      }
    }

    // copy shared
    const IterableExposedInfo shared(data.shared);
    _sharedCopy = shared;
  }
};
enum class WaitUntil { FirstSuccess, AllSuccess, SomeSuccess };
} // namespace shards

ENUM_HELP(WaitUntil, WaitUntil::FirstSuccess, SHCCSTR("Will wait until the first success and stop any other pending operation"));
ENUM_HELP(WaitUntil, WaitUntil::AllSuccess, SHCCSTR("Will wait until all complete, will stop and fail on any failure"));
ENUM_HELP(WaitUntil, WaitUntil::SomeSuccess, SHCCSTR("Will wait until all complete but won't fail if some of the wires failed"));

namespace shards {
struct ManyWire : public std::enable_shared_from_this<ManyWire> {
  uint32_t index;
  std::shared_ptr<SHWire> wire;
  std::shared_ptr<SHMesh> mesh; // used only if MT
  std::deque<SHVar *> injectedVariables;
  entt::scoped_connection _onCleanupConnection;
};

struct ParallelBase : public CapturingSpawners {
  DECL_ENUM_INFO(WaitUntil, WaitUntil,
                 "Policy for determining when to stop waiting in parallel operations. Defines the conditions under which "
                 "execution should proceed.",
                 'tryM');

  static inline Parameters _params{
      {"Wire", SHCCSTR("The Wire to copy and schedule."), IntoWire::RunnableTypes},
      {"Policy",
       SHCCSTR("The execution policy for the shard to abide by. A copied Wire is only deemed successful if it did not have an "
               "internal failure (eg.through Assert)"),
       {WaitUntilEnumInfo::Type}},
      {"Threads", SHCCSTR("The number of cpu threads to use. Number specified can not be lower than 1."), {CoreInfo::IntType}}};

  static SHParametersInfo parameters() { return _params; }

  ~ParallelBase() {
    // Make sure mesh outlives the wires, because of event handlers
    _pool.reset();
    _meshes.clear();
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    case 1:
      _policy = WaitUntil(value.payload.enumValue);
      break;
    case 2:
      _threads = std::max(int64_t(1), value.payload.intValue);
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    case 1:
      return Var::Enum(_policy, CoreCC, 'tryM');
    case 2:
      return Var(_threads);
    default:
      return Var::Empty;
    }
  }

  void compose(const SHInstanceData &data, const SHTypeInfo &inputType) {
    WireBase::resolveWire();

    if (_threads > 0) {
      mode = RunWireMode::Async;
      capturing = true;
    } else {
      mode = RunWireMode::Inline;
      capturing = false;
    }

    CapturingSpawners::compose(data, inputType);

    // wire should be populated now and such
    _pool.reset(new WireDoppelgangerPool<ManyWire>(SHWire::weakRef(wire)));
  }

  struct Composer {
    ParallelBase &server;
    bool onWorkerThread = true;

    void compose(SHWire *wire, SHMesh *mesh, bool recycling) {
      if (recycling)
        return;

      SHInstanceData data{};
      data.onWorkerThread = onWorkerThread;
      data.inputType = server._inputType;
      if (!wire->pure) {
        data.shared = server._sharedCopy;
      }
      data.wire = wire;
      wire->mesh = mesh->shared_from_this();
      auto res = composeWire(wire, data);
      arrayFree(res.exposedInfo);
      arrayFree(res.requiredInfo);
    }
  } _composer{*this};

  void warmup(SHContext *context) {
    if (capturing) {
      for (auto &v : _vars) {
        SHLOG_TRACE("ParallelBase: warming up variable: {}", v.variableName());
        v.warmup(context);
      }

      SHLOG_TRACE("ParallelBase: warmed up {} variables", _vars.size());
    }
  }

  void cleanup(SHContext *context) {
    if (capturing) {
      for (auto &v : _vars) {
        v.cleanup();
      }
    }

    for (auto &v : _outputs) {
      destroyVar(v);
    }
    _outputs.clear();

    // _wires seem to be used only in StepMany and DoMany
    for (auto &cref : _wires) {
      if (cref) {
        if (cref->mesh) {
          cref->mesh->terminate();
        }

        stop(cref->wire.get());

        if (capturing) {
          for (auto &v : _vars) {
            // notice, this should be already destroyed by the wire releaseVariable
            std::string_view name = v.variableNameView();
            destroyVar(cref->wire->getVariable(toSWL(name)));
          }
        }

        _pool->release(cref);
      }
    }
    _wires.clear();
  }

  virtual SHVar getInput(const SHVar &input, uint32_t index) = 0;

  virtual size_t getLength(const SHVar &input) = 0;

  static constexpr size_t MinWiresAdded = 4;

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_threads > 0);

    auto len = getLength(input);

    if (len == 0) {
      _outputs.resize(0);
      return Var(_outputs.data(), 0);
    }

    size_t succeeded = 0;
    size_t failed = 0;

    _successes.resize(len);
    std::fill(_successes.begin(), _successes.end(), false);

    _meshes.resize(len);

    _outputs.resize(len);

    // multithreaded
    tf::Taskflow flow;

    std::atomic_bool anySuccess = false;

    auto poolBatch = _pool->acquireBatch(len);

    // https://taskflow.github.io/taskflow/LimitTheMaximumConcurrency.html
    tf::Semaphore semaphore(std::max<size_t>(1, _threads));
    flow.for_each_index(size_t(0), len, size_t(1), [&](auto &idx) {
      if (_policy == WaitUntil::FirstSuccess && anySuccess) {
        // Early exit if FirstSuccess policy
        return;
      }

      SHLOG_TRACE("ParallelBase: activating wire {}", idx);

      auto &mesh = _meshes[idx];
      if (!mesh) {
        mesh = SHMesh::make();
        auto parentMesh = context->main->mesh.lock();
        mesh->parent = bool(parentMesh) ? parentMesh.get() : nullptr; // we need this for any storage
      }

      ManyWire *cref;
      try {
        cref = _pool->acquireFromBatch(poolBatch, idx, _composer, mesh.get());
      } catch (std::exception &e) {
        SHLOG_ERROR("Failed to acquire wire: {}", e.what());
        return;
      }

      cref->mesh = mesh;

#if SH_DEBUG_THREAD_NAMES
      pushThreadName(fmt::format("tf::Executor \"{}\" ({} idx: {})", cref->wire->name, context->currentWire()->name, idx));
      DEFER({ popThreadName(); });
#endif

      // setup captured variables as mesh externals
      std::deque<shards::OwnedVar> capturedVars;
      for (auto &v : _vars) {
        auto &var = v.get();
        auto &ref = capturedVars.emplace_back(var);
        // set external flag
        ref.flags |= SHVAR_FLAGS_EXTERNAL;
        std::string_view name = v.variableNameView();
        cref->mesh->addRef(toSWL(name), &ref);
      }

      bool success = true;
      cref->mesh->schedule(cref->wire, getInput(input, idx), false); // don't compose
      cref->wire->context->onWorkerThread = true;
      while (!cref->mesh->empty()) {
        if (!cref->mesh->tick() || (_policy == WaitUntil::FirstSuccess && anySuccess)) {
          success = false;
          break;
        }
      }

      _successes[idx] = success;

      if (success) {
        SHLOG_TRACE("ParallelBase: wire {} succeeded", idx);
        anySuccess = true;
        stop(cref->wire.get(), &_outputs[idx]);
      } else {
        SHLOG_TRACE("ParallelBase: wire {} failed", idx);
        _outputs[idx] = Var::Empty; // flag as empty to signal failure
        // ensure it's stopped anyway
        stop(cref->wire.get());
      }

      // if we throw we are screwed but thread should panic and terminate anyways
      mesh->terminate();
      _pool->release(cref);
    });

    auto future = TaskFlowInstance::instance().run(std::move(flow));

    // we done if we are here
    while (true) {
      auto suspend_state = shards::suspend(context, 0);
      if (unlikely(suspend_state != SHWireState::Continue)) {
        SHLOG_DEBUG("ParallelBase, interrupted!");
        anySuccess = true; // flags early stop as well
        future.get();      // wait for all to finish in any case
        return Var::Empty;
      } else if ((_policy == WaitUntil::FirstSuccess && anySuccess) ||
                 future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        future.get(); // wait for all to finish in any case
        break;
      }
    }

    for (size_t i = 0; i < len; i++) {
      auto success = _successes[i];
      if (success) {
        if (_policy == WaitUntil::FirstSuccess) {
          return _outputs[i]; // return the first success
        } else {
          SHLOG_TRACE("ParallelBase, reading succeeded wire {}", i);
          succeeded++;
        }
      } else {
        SHLOG_TRACE("ParallelBase, reading failed wire {}", i);
        failed++;
      }
    }

    if ((succeeded + failed) == len) {
      if (unlikely(succeeded == 0)) {
        throw ActivationError("ParallelBase, failed all wires!");
      } else {
        // all ended let's apply policy here
        if (_policy == WaitUntil::SomeSuccess) {
          return Var(_outputs.data(), len);
        } else {
          assert(_policy == WaitUntil::AllSuccess);
          if (len == succeeded) {
            return Var(_outputs.data(), len);
          } else {
            throw ActivationError("ParallelBase, failed some wires!");
          }
        }
      }
    }

    throw ActivationError("ParallelBase, failed to activate wires!");
  }

protected:
  WaitUntil _policy{WaitUntil::AllSuccess};
  std::unique_ptr<WireDoppelgangerPool<ManyWire>> _pool;
  Type _outputSeqType;
  Types _outputTypes;
  SHTypeInfo _inputType{};
  std::vector<OwnedVar> _outputs;
  std::vector<char>
      _successes; // don't use bool cos std lib uses bit vectors behind the scenes and won't work with multiple threads
  std::vector<std::shared_ptr<SHMesh>> _meshes;
  std::vector<ManyWire *> _wires;
  int64_t _threads{0};
};

struct TryMany : public ParallelBase {
  static SHTypesInfo inputTypes() { return CoreInfo::AnySeqType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnySeqType; }

  static SHOptionalString inputHelp() { return InputManyWire; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Depending on the Policy specified the shard will return a different output. WaitUntil::FirstSuccess will "
                   "return the output of the first successful Wire. "
                   "WaitUntil::SomeSuccess return a sequence with all the output from all the copies of the specified Wire. "
                   "WaitUntil::AllSuccess will either stop execution of the "
                   "current Wire if any of the copies fail or return a sequence with all the output from all the copies of the "
                   "specified Wire.");
  }

  SHOptionalString help() {
    return SHCCSTR(
        "This shard takes a sequence of values as input, schedules multiple copies of a specified Wire and executes them "
        "asynchronously. Each value from the sequence is provided as input to its corresponding copy of the scheduled Wire. The "
        "shard will then wait for all the scheduled Wires to end, and then, depending on "
        "the Policy specified, the shard will either return the output of the first successful Wire, return a sequence with all "
        "the output from all the copies of the specified Wire or stop execution of the current Wire if all the copies failed.");
  }

  static inline Parameters _params{
      {"Wire", SHCCSTR("The Wire to copy and schedule."), IntoWire::RunnableTypes},
      {"Policy",
       SHCCSTR("The execution policy for the shard to abide by. A copied Wire is only deemed successful if it did not have an "
               "internal "
               "failure (eg.through Assert)"),
       {WaitUntilEnumInfo::Type}},
      {"Threads", SHCCSTR("The number of cpu threads to use. Number specified can not be lower than 1."), {CoreInfo::IntType}}};

  static SHParametersInfo parameters() { return _params; }

  void setup() { _threads = 1; }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_threads < 1) {
      throw ComposeError("TryMany, threads must be >= 1");
    }

    if (data.inputType.seqTypes.len == 1) {
      // copy single input type
      _inputType = data.inputType.seqTypes.elements[0];
    } else {
      // else just mark as generic any
      _inputType = CoreInfo::AnyType;
    }

    ParallelBase::compose(data, _inputType);

    switch (_policy) {
    case WaitUntil::AllSuccess: {
      // seq result
      _outputTypes = Types({wire->outputType});
      _outputSeqType = Type::SeqOf(_outputTypes);
      return _outputSeqType;
    }
    case WaitUntil::SomeSuccess: {
      return CoreInfo::AnySeqType;
    }
    case WaitUntil::FirstSuccess: {
      // single result
      return wire->outputType;
    }
    }
    __builtin_unreachable();
  }

  SHVar getInput(const SHVar &input, uint32_t index) override { return input.payload.seqValue.elements[index]; }

  size_t getLength(const SHVar &input) override { return size_t(input.payload.seqValue.len); }
};

struct Expand : public ParallelBase {
  int64_t _width{10};

  void setup() { _threads = 1; }

  static SHOptionalString inputHelp() {
    return SHCCSTR("This shard takes a value of any type as input. This value is provided as input to every scheduled copy of "
                   "the specified Wire.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR(
        "Depending on the Policy specified the shard will return a different output. WaitUntil::FirstSuccess will return the "
        "output of the first successful Wire. WaitUntil::SomeSuccess return a sequence with all the output from all the copies "
        "of the specified Wire. WaitUntil::FirstSuccess will either stop execution of the current Wire if any of the copies fail "
        "or return a sequence with all the output from all the copies of the specified Wire.");
  }

  SHOptionalString help() {
    return SHCCSTR(
        "Schedules (n) number of copies of the specified Wire, where (n) is the number specified in the Size parameter. The "
        "parent "
        "Wire will wait until all the scheduled copies have ended and will either return a sequence of values outputs of"
        " all the copied Wires or the output of the first Wire that succeeds. Once done, it will continue with its own "
        "execution.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnySeqType; }

  static inline Parameters _params{
      {{"Size", SHCCSTR("The number of copies of the specified Wire to schedule."), {CoreInfo::IntType}}}, ParallelBase::_params};

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    if (index == 0) {
      _width = std::max(int64_t(1), value.payload.intValue);
    } else {
      ParallelBase::setParam(index - 1, value);
    }
  }

  SHVar getParam(int index) {
    if (index == 0) {
      return Var(_width);
    } else {
      return ParallelBase::getParam(index - 1);
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_threads < 1) {
      throw ComposeError("Expand, threads must be >= 1");
    }

    // input
    _inputType = data.inputType;

    ParallelBase::compose(data, _inputType);

    // output
    _outputTypes = Types({wire->outputType});
    _outputSeqType = Type::SeqOf(_outputTypes);
    return _outputSeqType;
  }

  SHVar getInput(const SHVar &input, uint32_t index) override { return input; }

  size_t getLength(const SHVar &input) override { return size_t(_width); }
};

struct Spawn : public CapturingSpawners {
  Spawn() {
    mode = RunWireMode::Async;
    capturing = true;
    passthrough = false;
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::WireType; }

  static SHOptionalString inputHelp() { return InputBaseWire; }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the specific copy of the Wire that was scheduled."); }

  SHOptionalString help() {
    return SHCCSTR("Schedules the specified Wire and runs it asynchronously. The current Wire will continue "
                   "its execution independently of the specified Wire. Unlike Detach, a copy of the specified Wire is scheduled "
                   "every time the shard is called.");
  }

  static inline Parameters _params{{"Wire", SHCCSTR("The Wire to schedule and run asynchronously"), IntoWire::RunnableTypes}};

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      wireref = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return wireref;
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    WireBase::resolveWire();

    if (!wire) {
      throw ComposeError("Spawn: wire not found");
    }

    CapturingSpawners::compose(data, data.inputType);

    _pool.reset(new WireDoppelgangerPool<ManyWire>(SHWire::weakRef(wire)));

    // copy input type
    _inputType = data.inputType;

    return CoreInfo::WireType;
  }

  struct Composer {
    Spawn &server;
    SHContext *context;

    void compose(SHWire *wire, SHContext *context, bool recycling) {
      if (recycling)
        return;

      SHLOG_TRACE("Spawn::Composer::compose {}", wire->name);
      SHInstanceData data{};
      data.inputType = server._inputType;
      if (!wire->pure) {
        data.shared = server._sharedCopy;
      }
      data.wire = wire;
      wire->mesh = context->main->mesh;
      auto res = composeWire(wire, data);
      arrayFree(res.exposedInfo);
      arrayFree(res.requiredInfo);
    }
  } _composer{*this};

  void warmup(SHContext *context) {
    WireBase::warmup(context);

    _composer.context = context;

    for (auto &v : _vars) {
      SHLOG_TRACE("Spawn: warming up variable: {}", v.variableName());
      v.warmup(context);
    }

    SHLOG_TRACE("Spawn: warmed up {} variables", _vars.size());
  }

  void cleanup(SHContext *context) {
    for (auto &v : _vars) {
      v.cleanup();
    }

    _composer.context = nullptr;

    WireBase::cleanup(context);
  }

  std::unordered_map<const SHWire *, ManyWire *> _wireContainers;

  void wireOnCleanup(const SHWire::OnCleanupEvent &e) {
    SHLOG_TRACE("Spawn::wireOnCleanup {}", e.wire->name);

    auto it = _wireContainers.find(e.wire);
    if (it != _wireContainers.end()) {
      auto container = it->second;
      for (auto &var : container->injectedVariables) {
        releaseVariable(var);
      }
      container->injectedVariables.clear();

      _pool->release(container);
      _wireContainers.erase(it);
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto mesh = context->main->mesh.lock();

    auto c = _pool->acquire(_composer, context);
    shassert(!_wireContainers.contains(c->wire.get()));
    _wireContainers[c->wire.get()] = c;

    // Connect the cleanup event
    if (!c->_onCleanupConnection) {
      c->_onCleanupConnection = c->wire->dispatcher.sink<SHWire::OnCleanupEvent>().connect<&Spawn::wireOnCleanup>(this);
    }

    shassert(c->injectedVariables.empty() && "Spawn: injected variables should be empty");
    for (auto &v : _vars) {
      SHVar *refVar = c->injectedVariables.emplace_back(referenceWireVariable(c->wire.get(), v.variableName()));
      cloneVar(*refVar, v.get());
    }

    mesh->schedule(c->wire, input, false);

    SHWire *rootWire = context->rootWire();
    mesh->dispatcher.trigger(SHWire::OnDetachedEvent{
        .wire = rootWire,
        .childWire = c->wire.get(),
    });

    return Var(c->wire); // notice this is "weak"
  }

  std::unique_ptr<WireDoppelgangerPool<ManyWire>> _pool;
  SHTypeInfo _inputType{};
};

struct WhenDone : Spawn {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  SeqVar _cache;
  bool _scheduled{false};
  bool _activated{false};

  void warmup(SHContext *context) {
    _scheduled = false;
    _activated = false;
    Spawn::warmup(context);
  }

  void cleanup(SHContext *context) {
    if (context && wire && !_scheduled && _activated) {
      _scheduled = true;

      auto mesh = context->main->mesh.lock();
      shassert(mesh && "Mesh is null");

      auto c = _pool->acquire(_composer, context);
      _wireContainers[c->wire.get()] = c;

      // Connect the cleanup event
      if (!c->_onCleanupConnection) {
        c->_onCleanupConnection = c->wire->dispatcher.sink<SHWire::OnCleanupEvent>().connect<&Spawn::wireOnCleanup>(this);
      }

      for (auto &v : _vars) {
        SHVar *refVar = c->injectedVariables.emplace_back(referenceWireVariable(c->wire.get(), v.variableName()));
        cloneVar(*refVar, v.get());
      }

      SHLOG_TRACE("WhenDone: scheduling {}, ptr: {}", c->wire->name, (void *)c->wire.get());
      mesh->schedule(c->wire, Var::Empty, false);

      SHWire *rootWire = context->rootWire();
      mesh->dispatcher.trigger(SHWire::OnDetachedEvent{
          .wire = rootWire,
          .childWire = c->wire.get(),
      });
    }

    for (auto &v : _vars) {
      v.cleanup();
    }

    _composer.context = nullptr;

    WireBase::cleanup(context);
  }

  void activate(SHContext *context, const SHVar &input) {
    _activated = true;

    // keep variables captured up to date
    _cache.clear();
    for (auto &v : _vars) {
      _cache.push_back(v.get());
    }
  }
};

struct DoMany : public TryMany {
  bool _composeSync{};

  void setup() { _threads = 0; } // we don't use threads

  static SHOptionalString inputHelp() { return InputManyWire; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("This shard outputs the output of all the scheduled copies in a sequence.");
  }

  SHOptionalString help() {
    return SHCCSTR(
        "This shard takes a sequence of values as input, schedules multiple copies of a specified Wire and executes them "
        "sequentially. Each value from the sequence is provided as input to its corresponding copy of the specified Wire. The "
        "shard then outputs a sequence of values containing the output of all copies of the specified Wire.");
  }

  static inline Types BoolTypes{CoreInfo::BoolType};
  static inline Parameters _params{
      {"Wire", SHCCSTR("The wire to run many times sequentially."), IntoWire::RunnableTypes},
      {"ComposeSync", SHCCSTR("Compose new wires synchronously."), BoolTypes},
  };

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    if (index == 1) {
      _composeSync = value.payload.boolValue;
    } else {
      TryMany::setParam(index, value);
    }
  }

  SHVar getParam(int index) {
    if (index == 1) {
      return Var(_composeSync);
    } else {
      return TryMany::getParam(index);
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _composer.onWorkerThread = false;

    WireBase::resolveWire();

    if (data.inputType.seqTypes.len == 1) {
      // copy single input type
      _inputType = data.inputType.seqTypes.elements[0];
    } else {
      // else just mark as generic any
      _inputType = CoreInfo::AnyType;
    }

    ParallelBase::compose(data, _inputType);

    switch (_policy) {
    case WaitUntil::AllSuccess: {
      // seq result
      _outputTypes = Types({wire->outputType});
      _outputSeqType = Type::SeqOf(_outputTypes);
      return _outputSeqType;
    }
    case WaitUntil::SomeSuccess: {
      return CoreInfo::AnySeqType;
    }
    case WaitUntil::FirstSuccess: {
      // single result
      return wire->outputType;
    }
    }
    __builtin_unreachable();
  }

  void warmup(SHContext *context) {
    TryMany::warmup(context);
    _meshes.clear();
    _meshes.push_back(context->main->mesh.lock());
  }

  void cleanup(SHContext *context) {
    for (auto &cref : _wires) {
      if (cref) {
        // ensure calling cleanup explicitly as stop might not be called fully
        cref->wire->cleanup(context);
      }
    }
    TryMany::cleanup(context);
    _meshes.clear();
  }

  std::vector<ManyWire *> _toWarmup;

  SHVar activate(SHContext *context, const SHVar &input) {
    auto len = getLength(input);
    // Compute required size
    size_t capacity = std::max(MinWiresAdded, len);
    size_t oldWiresSize = _wires.size();

    // Only ever grow
    capacity = std::max(_wires.size(), capacity);

    _outputs.resize(capacity);
    _wires.resize(capacity);

    if (_wires.size() > oldWiresSize) {
      int32_t numToAcquire = int32_t(_wires.size()) - int32_t(oldWiresSize);
      shassert(numToAcquire >= 0 && "DoMany: numToAcquire is negative");
      auto batch = _pool->acquireBatch(numToAcquire);
      if (_composeSync) {
        for (int32_t i = 0; i < numToAcquire; i++) {
          auto &cref = _wires[i + oldWiresSize];
          // compose on a worker thread!
          cref = _pool->acquireFromBatch(batch, i, _composer, _meshes[0].get());
        }
      } else {
        await(
            context,
            [&, this]() {
              for (int32_t i = 0; i < numToAcquire; i++) {
                auto &cref = _wires[i + oldWiresSize];
                // compose on a worker thread!
                cref = _pool->acquireFromBatch(batch, i, _composer, _meshes[0].get());
              }
            },
            [] {});
      }
    }

    for (size_t i = oldWiresSize; i < _wires.size(); i++) {
      auto &cref = _wires[i];
      cref->wire->warmup(context); // have to run this outside or:
      // Assertion failed: (context->currentWire() == currentWire), function suspend, file runtime.cpp, line 560.
    }

    for (uint32_t i = 0; i < len; i++) {
      auto &cref = _wires[i];
      const SHVar &inputElement = input.payload.seqValue.elements[i];

      // Run within the root flow
      auto runRes = runSubWire(cref->wire.get(), context, inputElement);
      if (unlikely(runRes.state == SHRunWireOutputState::Failed)) {
        // When an error happens during inline execution, propagate the error to the parent wire
        SHLOG_ERROR("Wire {} failed", wire->name);
        context->cancelFlow("Wire failed");
        return cref->wire->previousOutput; // doesn't matter actually
      } else {
        // we don't want to propagate a (Return)
        if (unlikely(runRes.state == SHRunWireOutputState::Returned)) {
          context->resetErrorStack();
          context->continueFlow();
        }
      }

      // this can be anything really...
      cloneVar(_outputs[i], cref->wire->previousOutput);
    }

    return Var(_outputs.data(), len);
  }
};

struct Branch {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static inline Types VarSeqTypes{CoreInfo::AnyVarType, CoreInfo::StringType};
  static inline Type VarSeqType = Type::SeqOf(VarSeqTypes);

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  SHOptionalString help() {
    return SHCCSTR("Creates a branch from the specified Behavior and schedules all the Wires specified. Every time this shard is "
                   "called, it will progress the state of all the Wires specified asynchronously and continue execution of the "
                   "current Wire. This shard is like a mass Step, where it Steps all the Wires specified.");
  }

  static SHParametersInfo parameters() {
    static Parameters params{
        {"Wires",
         SHCCSTR("The Wires to schedule and run on this branch."),
         {CoreInfo::WireType, CoreInfo::WireSeqType, CoreInfo::NoneType}},
        {"FailureBehavior",
         SHCCSTR("The policy to take when some of the Wires running on this branch fail."),
         {BranchFailureEnumInfo::Type}},
        {"CaptureAll",
         SHCCSTR(
             "If all of the existing variables should be copied onto the branch created, regardless if they are used or not."),
         {CoreInfo::BoolType}},
        {"Mesh",
         SHCCSTR("The Behavior to branch from. If none provided, it will branch from the Behavior that the current Wire is on."),
         {CoreInfo::NoneType, SHMesh::MeshType}},
    };
    return params;
  }

private:
  OwnedVar _wires{Var::Empty};
  Brancher _brancher;
  OwnedVar _capture;

public:
  ~Branch() {
    // Make sure that wires are destroyed before brancher in case references to branch/mesh are kept
    _capture.reset();
    _wires.reset();
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _wires = value;
      _brancher.setRunnables(_wires);
      break;
    case 1:
      _brancher.failureBehavior = BranchFailureBehavior(value.payload.enumValue);
      break;
    case 2:
      _brancher.captureAll = value.payload.boolValue;
      break;
    case 3:
      if (value.valueType == SHType::None) {
        _brancher.mesh = SHMesh::make();
      } else {
        auto sharedMesh = reinterpret_cast<std::shared_ptr<SHMesh> *>(value.payload.objectValue);
        _brancher.mesh = *sharedMesh;
      }
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _wires;
    case 1:
      return Var::Enum(_brancher.failureBehavior, BranchFailureEnumInfo::VendorId, BranchFailureEnumInfo::TypeId);
    case 2:
      return Var(_brancher.captureAll);
    case 3:
      shassert(_brancher.mesh && "Branch: mesh is null");
      return Var::Object(&_brancher.mesh, CoreCC, SHMesh::TypeId);
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    auto dataCopy = data;
    dataCopy.inputType = {}; // Branch doesn't have an input

    _brancher.compose(dataCopy);

    return SHMesh::MeshType;
  }

  SHExposedTypesInfo requiredVariables() { return _brancher.requiredVariables(); }

  void warmup(SHContext *context) { _brancher.warmup(context); }

  void cleanup(SHContext *context) { _brancher.cleanup(context); }

  void activate(SHContext *context, const SHVar &input) { _brancher.activate(); }
};

SHARDS_REGISTER_FN(wires) {
  REGISTER_ENUM(WireBase::RunWireModeEnumInfo);
  REGISTER_ENUM(ParallelBase::WaitUntilEnumInfo);
  REGISTER_ENUM(BranchFailureEnumInfo);

  using RunWireDo = RunWire<false, RunWireMode::Inline>;
  using RunWireDetach = RunWire<true, RunWireMode::Async>;
  using RunWireStep = RunWire<true, RunWireMode::Stepped>;
  REGISTER_SHARD("SwitchTo", SwitchTo);
  REGISTER_SHARD("Wait", Wait);
  REGISTER_SHARD("Stop", StopWire);
  REGISTER_SHARD("Do", RunWireDo);
  REGISTER_SHARD("Detach", RunWireDetach);
  REGISTER_SHARD("Step", RunWireStep);
  REGISTER_SHARD("WireRunner", WireRunner); // Forbidden for FBL
  REGISTER_SHARD("Recur", Recur);
  REGISTER_SHARD("TryMany", TryMany);
  REGISTER_SHARD("Spawn", Spawn);
  REGISTER_SHARD("Expand", Expand);
  REGISTER_SHARD("Branch", Branch);
  REGISTER_SHARD("DoMany", DoMany);
  REGISTER_SHARD("Peek", Peek);
  REGISTER_SHARD("IsRunning", IsRunning);
  REGISTER_SHARD("Suspend", SuspendWire);
  REGISTER_SHARD("Resume", ResumeWire);
  REGISTER_SHARD("WhenDone", WhenDone); // Forbidden for FBL for now. But to fix eventually
}
}; // namespace shards
