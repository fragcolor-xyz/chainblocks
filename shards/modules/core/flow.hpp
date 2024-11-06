#ifndef BA2BD435_CBCD_42CF_BCA0_DEB072B8CDCC
#define BA2BD435_CBCD_42CF_BCA0_DEB072B8CDCC

#include <shards/log/log.hpp>
#include <shards/shardwrapper.hpp>
#include <shards/shards.h>
#include <shards/core/foundation.hpp>
#include <shards/core/shared.hpp>
#include <shards/core/async.hpp>
#include <atomic>

namespace shards {
static inline Type condShardSeqs = Type::SeqOf(CoreInfo::ShardsOrNone);
static inline ParamsInfo condParamsInfo =
    ParamsInfo(ParamsInfo::Param("Wires",
                                 SHCCSTR("A sequence of shards, interleaving condition test predicate "
                                         "and action to execute if the condition matches."),
                                 condShardSeqs),
               ParamsInfo::Param("Passthrough", SHCCSTR("The output of this shard will be its input."), CoreInfo::BoolType),
               ParamsInfo::Param("Threading",
                                 SHCCSTR("Will not short circuit after the first true test expression. "
                                         "The threaded value gets used in only the action and not the "
                                         "test part of the clause."),
                                 CoreInfo::BoolType));

struct Cond {
  // WORKS but TODO refactor using newer abstracted types
  SHVar _wires{};
  std::vector<std::vector<Shard *>> _conditions;
  std::vector<std::vector<Shard *>> _actions;
  bool _passthrough = true;
  bool _threading = false;
  SHComposeResult _wireValidation{};

  static SHOptionalString help() {
    return SHCCSTR("Takes a sequence of conditions and predicates. "
                   "Evaluates each condition one by one and if one matches, "
                   "executes the associated action.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The value that will be passed to each predicate and action "
                   "to execute.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The input of the shard if `Passthrough` is `true`; otherwise, the "
                   "output of the action of the first matching condition.");
  }

  static SHParametersInfo parameters() { return SHParametersInfo(condParamsInfo); }

  void warmup(SHContext *ctx) {
    for (auto &blks : _conditions) {
      for (auto &blk : blks) {
        if (blk->warmup)
          blk->warmup(blk, ctx);
      }
    }
    for (auto &blks : _actions) {
      for (auto &blk : blks) {
        if (blk->warmup)
          blk->warmup(blk, ctx);
      }
    }
  }

  void cleanup(SHContext *context) {
    for (auto it = _conditions.rbegin(); it != _conditions.rend(); ++it) {
      auto &shards = *it;
      for (auto jt = shards.rbegin(); jt != shards.rend(); ++jt) {
        auto shard = *jt;
        shard->cleanup(shard, context);
      }
    }
    for (auto it = _actions.rbegin(); it != _actions.rend(); ++it) {
      auto &shards = *it;
      for (auto jt = shards.rbegin(); jt != shards.rend(); ++jt) {
        auto shard = *jt;
        shard->cleanup(shard, context);
      }
    }
  }

  void destroy() {
    cleanup(nullptr);
    destroyVar(_wires);
    shards::arrayFree(_wireValidation.exposedInfo);
    shards::arrayFree(_wireValidation.requiredInfo);
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0: {
      destroy();
      _conditions.clear();
      _actions.clear();
      cloneVar(_wires, value);
      if (value.valueType == SHType::Seq) {
        auto counter = value.payload.seqValue.len;
        if (counter % 2)
          throw SHException("Cond: first parameter must contain a sequence of "
                            "pairs [condition to check & action to perform if "
                            "check passed (true)].");
        _conditions.resize(counter / 2);
        _actions.resize(counter / 2);
        auto idx = 0;
        for (uint32_t i = 0; i < counter; i++) {
          auto val = value.payload.seqValue.elements[i];
          if (i % 2) { // action
            if (val.valueType == SHType::ShardRef) {
              assert(!val.payload.shardValue->owned);
              val.payload.shardValue->owned = true;
              _actions[idx].push_back(val.payload.shardValue);
            } else { // seq
              for (uint32_t y = 0; y < val.payload.seqValue.len; y++) {
                assert(val.payload.seqValue.elements[y].valueType == SHType::ShardRef);
                auto blk = val.payload.seqValue.elements[y].payload.shardValue;
                assert(!blk->owned);
                blk->owned = true;
                _actions[idx].push_back(blk);
              }
            }

            idx++;
          } else { // condition
            if (val.valueType == SHType::ShardRef) {
              assert(!val.payload.shardValue->owned);
              val.payload.shardValue->owned = true;
              _conditions[idx].push_back(val.payload.shardValue);
            } else { // seq
              for (uint32_t y = 0; y < val.payload.seqValue.len; y++) {
                assert(val.payload.seqValue.elements[y].valueType == SHType::ShardRef);
                auto blk = val.payload.seqValue.elements[y].payload.shardValue;
                assert(!blk->owned);
                blk->owned = true;
                _conditions[idx].push_back(blk);
              }
            }
          }
        }
      }
      break;
    }
    case 1:
      _passthrough = value.payload.boolValue;
      break;
    case 2:
      _threading = value.payload.boolValue;
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
      return Var(_passthrough);
    case 2:
      return Var(_threading);
    default:
      break;
    }
    return Var();
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    // Free any previous result!
    shards::arrayFree(_wireValidation.exposedInfo);
    shards::arrayFree(_wireValidation.requiredInfo);

    // Validate condition wires, altho they do not influence anything we need
    // to report errors
    for (const auto &action : _conditions) {
      auto validation = composeWire(action, data);
      if (validation.outputType.basicType != SHType::Bool) {
        throw ComposeError("Cond - expected Bool output from predicate shards");
      }
      shards::arrayFree(validation.exposedInfo);
      shards::arrayFree(validation.requiredInfo);
    }

    // Evaluate all actions, all must return the same type in order to be safe
    SHTypeInfo previousType{};
    auto idx = 0;
    auto first = true;
    auto exposing = true;
    for (const auto &action : _actions) {
      auto validation = composeWire(action, data);

      if (first) {
        // A first valid exposedInfo array is our gold
        _wireValidation = validation;
        first = false;
      } else {
        if (exposing) {
          auto curlen = _wireValidation.exposedInfo.len;
          auto newlen = validation.exposedInfo.len;
          if (curlen != newlen) {
            SHLOG_DEBUG("Cond: number of exposed variables between actions mismatch, "
                        "variables won't be exposed, flow is unpredictable!");
            exposing = false;
          }

          if (exposing) {
            for (uint32_t i = 0; i < curlen; i++) {
              if (_wireValidation.exposedInfo.elements[i] != validation.exposedInfo.elements[i]) {
                SHLOG_DEBUG("Cond: types of exposed variables between actions "
                            "mismatch, variables won't be exposed, flow is "
                            "unpredictable!");
                exposing = false;
                break;
              }
            }
          }
        }

        // free the exposed info part
        shards::arrayFree(validation.exposedInfo);
        shards::arrayFree(validation.requiredInfo);

        if (!exposing) {
          // make sure we expose nothing in this case!
          shards::arrayFree(_wireValidation.exposedInfo);
          shards::arrayFree(_wireValidation.requiredInfo);
        }
      }

      if (idx > 0 && !_passthrough && !validation.flowStopper) {
        if (validation.outputType.basicType != SHType::Any && validation.outputType != previousType) {
          validation.outputType = CoreInfo::AnyType;
          SHLOG_WARNING("Cond: Branches return different types, setting output type to Any!");
        }
      }

      idx++;
      previousType = validation.outputType;
    }

    return _passthrough ? data.inputType : previousType;
  }

  SHExposedTypesInfo exposedVariables() { return _wireValidation.exposedInfo; }

  shards::Var _output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    memset(&_output, 0, sizeof(SHVar));
    if (unlikely(_actions.size() == 0)) {
      if (_passthrough)
        return input;
      else
        return _output;
    }

    auto idx = 0;
    SHVar actionInput = input;
    SHVar finalOutput{};
    for (auto &cond : _conditions) {
      Shards shards{&cond[0], (uint32_t)cond.size(), 0};
      auto state = activateShards2(shards, context, input, _output);
      // conditional flow so we might have "returns" form (And) (Or)
      if (unlikely(state > SHWireState::Return))
        return _output;

      if (_output.payload.boolValue) {
        // Do the action if true!
        // And stop here
        memset(&_output, 0, sizeof(SHVar));
        Shards action{&_actions[idx][0], (uint32_t)_actions[idx].size(), 0};
        state = activateShards(action, context, actionInput, _output);
        if (state != SHWireState::Continue)
          return _output;
        if (_threading) {
          // set the output as the next action input (not cond tho!)
          finalOutput = _output;
          actionInput = finalOutput;
        } else {
          // not _threading so short circuit here
          return _passthrough ? input : _output;
        }
      }
      idx++;
    }

    return _passthrough ? input : _output;
  }
};

struct BaseSubFlow {
  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Must match the input types of the first shard in the sequence."); }

  SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Will match the output types of the first shard of the sequence."); }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) { _shards = value; }

  SHVar getParam(int index) { return _shards; }

  void cleanup(SHContext *context) {
    if (_shards)
      _shards.cleanup(context);
  }

  void warmup(SHContext *ctx) {
    if (_shards)
      _shards.warmup(ctx);
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_shards)
      _composition = _shards.compose(data);
    else
      _composition = {};

    return _composition.outputType;
  }

  SHExposedTypesInfo exposedVariables() { return _composition.exposedInfo; }

protected:
  ShardsVar _shards{};
  SHComposeResult _composition{};
  static inline Parameters _params{{"Shards", SHCCSTR("The shards to activate."), {CoreInfo::ShardsOrNone}}};
};

struct Maybe : public BaseSubFlow {
  static SHOptionalString help() {
    return SHCCSTR("Attempts to activate a shard or a sequence of shards. Upon "
                   "failure, activate another shard or sequence of shards.");
  }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _shards = value;
      break;
    case 1:
      _elseBlks = value;
      break;
    case 2:
      _silent = value.payload.boolValue;
      break;
    default:
      throw InvalidParameterIndex();
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _shards;
    case 1:
      return _elseBlks;
    case 2:
      return Var(_silent);
    default:
      throw InvalidParameterIndex();
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _self = data.shard;
    // exposed stuff should be balanced
    // and output should the same
    SHComposeResult elseComp{};
    if (_elseBlks) {
      elseComp = _elseBlks.compose(data);
    }

    if (_shards)
      _composition = _shards.compose(data);
    else
      _composition = {};

    const auto nextIsNone = data.outputTypes.len == 1 && data.outputTypes.elements[0].basicType == SHType::None;

    SHTypeInfo outputType = _composition.outputType;
    if (_elseBlks && !nextIsNone && !elseComp.flowStopper && _composition.outputType != elseComp.outputType) {
      outputType = CoreInfo::AnyType;
      SHLOG_WARNING("Maybe: Branches return different types, setting output type to Any!");
    }

    // Maybe won't expose
    _composition.exposedInfo = {};

    return !_elseBlks ? data.inputType : outputType;
  }

  void cleanup(SHContext *context) {
    if (_elseBlks)
      _elseBlks.cleanup(context);

    BaseSubFlow::cleanup(context);
  }

  void warmup(SHContext *ctx) {
    if (_elseBlks)
      _elseBlks.warmup(ctx);

    BaseSubFlow::warmup(ctx);
  }

  shards::Var _output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    if (likely(_shards)) {
      auto prev_level = logging::getSinkLevel();
      if (_silent) {
        logging::setSinkLevel(spdlog::level::off);
      }
      auto state = _shards.activate(context, input, _output);
      if (_silent) {
        logging::setSinkLevel(prev_level);
      }
      if (state == SHWireState::Error) {
        if (!_silent) {
          shassert(_self);
          auto currentWire = context->currentWire();
          SHLOG_WARNING("Maybe shard Ignored an error: {}, line: {}, column: {}, wire: {}", context->getErrorMessage(),
                        _self->line, _self->column, currentWire ? currentWire->name : "unknown");
        }
        if (likely(!context->onLastResume)) {
          context->resetErrorStack();
          context->continueFlow();
          if (_elseBlks) {
            _elseBlks.activate(context, input, _output);
            return _output;
          } else {
            return input;
          }
        } else {
          auto currentWire = context->currentWire();
          SHLOG_DEBUG("Maybe shard Ignored an error: {}, when on last resume, line: {}, column: {}, wire: {}",
                      context->getErrorMessage(), _self->line, _self->column, currentWire ? currentWire->name : "unknown");
          // Just continue as the wire is done
          return _output;
        }
      } else {
        if (!_elseBlks) {
          return input; // implicit pass through if no else blks
        } else {
          return _output;
        }
      }
    }
    return input;
  }

private:
  Shard *_self;
  ShardsVar _elseBlks{};
  bool _silent{false};
  static inline Parameters _params{BaseSubFlow::_params,
                                   {{"Else", SHCCSTR("The shards to activate on failure."), {CoreInfo::ShardsOrNone}},
                                    {"Silent",
                                     SHCCSTR("If logging should be disabled while running the shards (this "
                                             "will also disable (Log) and (Msg) shards) and no warning "
                                             "message should be printed on failure."),
                                     {CoreInfo::BoolType}}}};
};

struct Await : public BaseSubFlow {
  static SHOptionalString help() {
    return SHCCSTR("Executes a shard or a sequence of shards asynchronously "
                   "and awaits their completion.");
  }

  static SHParametersInfo parameters() { return _params; }

  OwnedVar _output;
  std::mutex mutex;
  bool _alreadyOnWorker{};

  void setParam(int index, const SHVar &value) { _shards = value; }

  SHVar getParam(int index) { return _shards; }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (data.onWorkerThread) {
      SHLOG_TRACE("Await shard is being used on a worker thread, ignoring, wire: {}", data.wire ? data.wire->name : "unknown");
      _alreadyOnWorker = true;
      OVERRIDE_ACTIVATE(data, activateSync);
    } else {
      _alreadyOnWorker = false;
    }
    auto dataCopy = data;
    // flag that we might use a worker
    dataCopy.onWorkerThread = true;
    return BaseSubFlow::compose(dataCopy);
  }

  std::optional<SHContext> _context;

  void warmup(SHContext *ctx) {
    BaseSubFlow::warmup(ctx);
    if (!_alreadyOnWorker) {
      _context.emplace(nullptr, ctx->currentWire(), ctx->flow);
      _context->wireStack = ctx->wireStack;
      _context->parent = ctx;
    }
  }

  void cleanup(SHContext *context) {
    BaseSubFlow::cleanup(context);
    if (_context.has_value()) {
      // this will trigger benign TSAN race condition warning
      // we cannot make the bool inside the context atomic, simple, but it's fine
      _context->stopFlow();
      // Don't .reset() _context, it's still running on the worker thread
    }
  }

  SHVar activateSync(SHContext *context, const SHVar &input) {
    SHVar output{};
    if (_shards.activate(context, input, output) == SHWireState::Error) {
      throw SHException("Await shard failed to activate the shards.");
    }
    return output;
  }

  const SHVar &activate(SHContext *context, const SHVar &input) {
    bool locked = false;
    DEFER({
      if (locked)
        mutex.unlock();
    });
    do {
      locked = mutex.try_lock();
      if (locked)
        break;
      else if (shards::suspend(context, 0) != SHWireState::Continue)
        return Var::Empty; // return as there is some error or so going on
    } while (true);

    // check if we should continue though!
    if (!context->shouldContinue()) {
      SHLOG_DEBUG("Await shard aborted before starting");
      return input;
    }

    // copy around to avoid race conditions
    OwnedVar inputCopy = input;
    _output = awaitne(
        context,
        [&] {
          // we cannot give the real context to the other thread or we will have race conditions
          // it's fine though cos awaitne will check the actual real context anyway!
          SHVar output{};
          _shards.activate(&*_context, inputCopy, output);
          return output;
        },
        [] {});

    // need to replicate things that happened in the context
    if (!_context->shouldContinue()) {
      SHLOG_DEBUG("Await shard stopped by context");
      context->mirror(&*_context);
    }

    return _output;
  }
};

// Not used actually
// I decided to not expose from When and If
// Exposing from flow stuff is dangerous
struct ExposerFlow {
  template <typename... Validations> void merge(Validations... vals) {
    constexpr int size = sizeof...(vals);
    std::array<SHComposeResult, size> v{vals...};
    IterableExposedInfo master(v[0].exposedInfo);
    for (size_t i = 1; i < v.size(); i++) {
      IterableExposedInfo sub(v[i].exposedInfo);
      if (master.size() != sub.size()) {
        throw ComposeError("Flow is unbalanced, exposing a different amount of variables.");
        _composition = {};
        return;
      }
      if (!std::equal(master.begin(), master.end(), sub.begin())) {
        throw ComposeError("Flow is unbalanced, type mismatch.");
        _composition = {};
        return;
      }
    }
    _composition = v[0];
  }

  SHExposedTypesInfo exposedVariables() { return _composition.exposedInfo; }
  SHComposeResult _composition{};
};

template <bool COND> struct When {
  static SHOptionalString help() {
    if constexpr (COND == true) {
      return SHCCSTR("Conditional shard that only executes the action if the "
                     "predicate is true.");
    } else {
      return SHCCSTR("Conditional shard that only executes the action if the "
                     "predicate is false.");
    }
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value that will be passed to the predicate."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() {
    if constexpr (COND == true) {
      return SHCCSTR("The input of the shard if `Passthrough` is `true`, or the "
                     "`Predicate` is `false`; otherwise, the output of the `Action`.");
    } else {
      return SHCCSTR("The input of the shard if `Passthrough` is `true`, or the "
                     "`Predicate` is `true`; otherwise, the output of the `Action`.");
    }
  }

  static inline Parameters _params{
      {"Predicate", SHCCSTR("The predicate to evaluate in order to trigger Action."), {CoreInfo::ShardsOrNone}},
      {"Action",
       SHCCSTR("The shards to activate on when Predicate is true for When and "
               "false for WhenNot."),
       {CoreInfo::ShardsOrNone}},
      {"Passthrough", SHCCSTR("The output of this shard will be its input."), {CoreInfo::BoolType}}};
  static SHParametersInfo parameters() { return _params; }

  ShardsVar _cond{};
  ShardsVar _action{};
  bool _passth = true;

  void setParam(int index, const SHVar &value) {
    if (index == 0)
      _cond = value;
    else if (index == 1)
      _action = value;
    else
      _passth = value.payload.boolValue;
  }

  SHVar getParam(int index) {
    if (index == 0)
      return _cond;
    else if (index == 1)
      return _action;
    else
      return Var(_passth);
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    // both not exposing!
    const auto cres = _cond.compose(data);
    if (cres.outputType.basicType != SHType::Bool) {
      throw ComposeError("When predicate should output a boolean value!");
    }

    auto ares = _action.compose(data);

    const auto nextIsNone = data.outputTypes.len == 1 && data.outputTypes.elements[0].basicType == SHType::None;

    if (!nextIsNone && !ares.flowStopper && !_passth) {
      if (ares.outputType != data.inputType) {
        SHLOG_ERROR("When Passthrough is false but action output type ({}) does not match input type ({}).", ares.outputType,
                    data.inputType);
        throw ComposeError("When Passthrough is false but action output type "
                           "does not match input type.");
      }
    }
    return data.inputType;
  }

  void cleanup(SHContext *context) {
    _cond.cleanup(context);
    _action.cleanup(context);
  }

  void warmup(SHContext *ctx) {
    _cond.warmup(ctx);
    _action.warmup(ctx);
  }

  shards::Var _output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    auto state = _cond.activate<true>(context, input, _output);
    if (unlikely(state > SHWireState::Return))
      return input;

    // type check in compose!
    if (_output.payload.boolValue == COND) {
      _action.activate(context, input, _output);
      if (!_passth)
        return _output;
    }

    return input;
  }
};

struct IfBlock {
  static SHOptionalString help() { return SHCCSTR("Evaluates a predicate and executes an action."); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value that will be passed to the predicate."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The input of the shard if `Passthrough` is `true`; otherwise, the "
                   "output of the action that was performed (i.e. `Then` or `Else`).");
  }

  static inline Parameters _params{
      {"Predicate",
       SHCCSTR("The predicate to evaluate in order to trigger `Then` (when "
               "`true`) or `Else` (when `false`)."),
       {CoreInfo::ShardsOrNone}},
      {"Then", SHCCSTR("The shards to activate when `Predicate` is `true`."), CoreInfo::ShardsOrNone},
      {"Else", SHCCSTR("The shards to activate when `Predicate` is `false`."), CoreInfo::ShardsOrNone},
      {"Passthrough", SHCCSTR("The output of this shard will be its input."), {CoreInfo::BoolType}}};
  static SHParametersInfo parameters() { return _params; }

  ShardsVar _cond{};
  ShardsVar _then{};
  ShardsVar _else{};
  bool _passth = false;

  void setParam(int index, const SHVar &value) {
    if (index == 0)
      _cond = value;
    else if (index == 1)
      _then = value;
    else if (index == 2)
      _else = value;
    else
      _passth = value.payload.boolValue;
  }

  SHVar getParam(int index) {
    if (index == 0)
      return _cond;
    else if (index == 1)
      return _then;
    else if (index == 2)
      return _else;
    else
      return Var(_passth);
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    // both not exposing!
    const auto cres = _cond.compose(data);
    if (cres.outputType.basicType != SHType::Bool) {
      throw ComposeError("If - predicate should output a boolean value!");
    }

    const auto tres = _then.compose(data);
    const auto eres = _else.compose(data);

    const auto nextIsNone = data.outputTypes.len == 1 && data.outputTypes.elements[0].basicType == SHType::None;

    SHTypeInfo outputType = tres.outputType;
    if (!nextIsNone && !tres.flowStopper && !eres.flowStopper && !_passth) {
      if (tres.outputType != eres.outputType) {
        outputType = CoreInfo::AnyType;
        SHLOG_WARNING("If: Branches return different types, setting output type to Any!");
      }
    }
    return _passth ? data.inputType : outputType;
  }

  void cleanup(SHContext *context) {
    _cond.cleanup(context);
    _then.cleanup(context);
    _else.cleanup(context);
  }

  void warmup(SHContext *ctx) {
    _cond.warmup(ctx);
    _then.warmup(ctx);
    _else.warmup(ctx);
  }

  shards::Var _output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    auto state = _cond.activate<true>(context, input, _output);
    if (unlikely(state > SHWireState::Return))
      return input;

    // type check in compose!
    if (_output.payload.boolValue) {
      _then.activate(context, input, _output);
    } else {
      _else.activate(context, input, _output);
    }

    if (!_passth)
      return _output;
    else
      return input;
  }
};

struct Match {
  static SHOptionalString help() {
    return SHCCSTR("Compares the input with the declared cases (in order of the declaration) and activates the shard of the "
                   "first matched case.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value that's compared with the declared cases."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("Same value as input if `:Passthrough` is `true` else the output of the matched case's shard if "
                   "`:Passthrough` is `false`.");
  }

  static inline Parameters params{
      {"Cases", SHCCSTR("Values to match against the input. A `none` case will match anything."), {CoreInfo::AnySeqType}},
      {"Passthrough",
       SHCCSTR("Parameter to control the shard's output. `true` allows the `Match` shard's input itself to appear as its output; "
               "`false` allows the matched shard's output to appear as `Match` shard's output."),
       {CoreInfo::BoolType}}};
  static SHParametersInfo parameters() { return params; }

  VariableResolver resolver;
  int _ncases = 0;
  std::vector<OwnedVar> _cases;
  std::vector<OwnedVar> _pcases;
  std::vector<ShardsVar> _actions;
  std::vector<SHVar> _full;
  bool _pass = true;

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0: {
      auto counter = value.payload.seqValue.len;
      if (counter % 2)
        throw SHException("Match: first parameter must contain a sequence of pairs [variable "
                          "to compare & action to perform if matches].");
      _ncases = int(counter / 2);
      _cases.resize(_ncases);
      _pcases.resize(_ncases);
      _actions.resize(_ncases);
      _full.resize(counter);
      auto idx = 0;
      for (uint32_t i = 0; i < counter; i += 2) {
        auto &matchItem = value.payload.seqValue.elements[i];
        _cases[idx] = matchItem;
        _pcases[idx] = matchItem;
        auto &actionItem = value.payload.seqValue.elements[i + 1];
        TypeInfo actionInfo(actionItem, SHInstanceData{});
        if (!matchTypes(actionInfo, CoreInfo::NoneType, true, true, true) &&
            !matchTypes(actionInfo, CoreInfo::ShardRefType, true, true, true) &&
            !matchTypes(actionInfo, CoreInfo::ShardRefSeqType, true, true, true)) {
          throw SHException(
              fmt::format("Match: action at index {} is invalid, it should be none, a shard or a sequence of shards.", idx));
        }
        _actions[idx] = actionItem;
        _full[i] = _pcases[idx];      // this cannot be matchItem, cos that will be gone after this call!!
        _full[i + 1] = _actions[idx]; // this cannot be actionItem, cos that will be gone after this call!!
        idx++;
      }
    } break;
    case 1:
      _pass = value.payload.boolValue;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_full);
    case 1:
      return Var(_pass);
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    SHTypeInfo outputType{};
    bool first = true;
    for (auto &action : _actions) {
      const auto cres = action.compose(data);
      if (!_pass) {
        // must evaluate output types and enforce they match between each case,
        // unless flow stopper
        if (first) {
          outputType = cres.outputType;
          first = false;
        } else {
          if (!cres.flowStopper) {
            if (outputType.basicType != SHType::Any && cres.outputType != outputType) {
              outputType = CoreInfo::AnyType;
              SHLOG_WARNING("Match: Branches return different types, setting output type to Any!");
            }
          }
        }
      }
    }

    return _pass ? data.inputType : outputType;
  }

  void warmup(SHContext *context) {
    const auto ncases = _cases.size();
    for (size_t i = 0; i < ncases; ++i) {
      resolver.warmup(_pcases[i], _cases[i], context);
    }

    for (auto &action : _actions) {
      action.warmup(context);
    }
  }

  void cleanup(SHContext *context) {
    for (auto &action : _actions) {
      action.cleanup(context);
    }

    resolver.cleanup(context);
  }

  shards::Var _output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    bool matched = false;
    resolver.reassign();
    for (auto i = 0; i < _ncases; i++) {
      auto &case_ = _cases[i];
      auto &action = _actions[i];
      if (case_ == input || case_.valueType == SHType::None) {
        action.activate(context, input, _output);
        matched = true;
        break;
      }
    }
    if (!matched) {
      std::string expected = "[ ";
      for (auto i = 0; i < _ncases; i++) {
        auto &case_ = _cases[i];
        expected += fmt::format("{}", case_);
        if ((i + 1) < _ncases)
          expected.append(" ");
      }
      expected += " ]";
      throw ActivationError(
          fmt::format("Failed to match input ({}), no matching case is present. Could be any of {}", input, expected));
    }
    return _pass ? input : _output;
  }
};

struct Sub {
  ShardsVar _shards{};
  SHComposeResult _composition{};

  static SHOptionalString help() {
    return SHCCSTR("Activates a shard or a sequence of shards independently, without consuming the input. I.e. the input of the "
                   "Sub flow will also be its output regardless of the shards activated in this Sub flow.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The input value passed to this Sub flow (and hence to the shard or sequence of shards in this Sub flow).");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output of this Sub flow (which is the same as its input)."); }

  static SHParametersInfo parameters() {
    static Parameters params{
        {"Shards", SHCCSTR("The shard or sequence of shards to execute in the Sub flow."), {CoreInfo::ShardsOrNone}}};
    return params;
  }

  void setParam(int index, const SHVar &value) { _shards = value; }

  SHVar getParam(int index) { return _shards; }

  SHTypeInfo compose(const SHInstanceData &data) {
    _composition = _shards.compose(data);
    return data.inputType;
  }

  SHExposedTypesInfo exposedVariables() { return _composition.exposedInfo; }

  void warmup(SHContext *ctx) { _shards.warmup(ctx); }

  void cleanup(SHContext *ctx) { _shards.cleanup(ctx); }

  void activate(SHContext *context, const SHVar &input) {
    SHVar output{}; // ignored but next call needs it
    _shards.activate(context, input, output);
  }
};
} // namespace shards

#endif /* BA2BD435_CBCD_42CF_BCA0_DEB072B8CDCC */
