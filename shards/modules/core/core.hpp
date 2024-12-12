/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#ifndef SH_CORE_SHARDS_CORE
#define SH_CORE_SHARDS_CORE

#include <shards/core/shards_macros.hpp>
#include <shards/core/foundation.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <shards/common_types.hpp>
#include <shards/number_types.hpp>
#include <shards/core/exposed_type_utils.hpp>
#include <shards/common_types.hpp>
#include <shards/inlined.hpp>
#include <shards/gfx/moving_average.hpp>
#include "time.hpp"
#include <cassert>
#include <cmath>
#include <optional>
#include <sstream>

using namespace std::chrono_literals;

namespace shards {
struct CoreInfo2 {
  DECL_ENUM_INFO(BasicTypes, Type,
                 "Fundamental data types supported by the system. Used for type checking and data manipulation.", 'type');
  static inline Type BasicTypesSeqType{{SHType::Seq, {.seqTypes = TypeEnumInfo::Type}}};
  static inline Types BasicTypesTypes{{TypeEnumInfo::Type, BasicTypesSeqType}, true};
};

// THis shard implements constants
// Constants are values like 0, ""
struct Const {
  static inline shards::ParamsInfo constParamsInfo = shards::ParamsInfo(
      shards::ParamsInfo::Param("Value", SHCCSTR("The constant value to insert in the wire."), CoreInfo::AnyType));

  OwnedVar _value{};
  OwnedVar _clone{};
  SHTypeInfo _innerInfo{};
  VariableResolver resolver;
  std::vector<SHExposedTypeInfo> _dependencies;

  void destroy() {
    // SHLOG_TRACE("Destroying Const (Last value: {}, clone: {})", _value, _clone);
    freeDerivedInfo(_innerInfo);
  }

  static SHOptionalString help() { return SHCCSTR("Declares an un-named constant value (of any data type)."); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The declared constant value."); }

  static SHParametersInfo parameters() { return SHParametersInfo(constParamsInfo); }

  void setParam(int index, const SHVar &value) { _value = value; }

  SHVar getParam(int index) { return _value; }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    freeDerivedInfo(_innerInfo);
    _dependencies.clear();
    _innerInfo = deriveTypeInfo(_value, data, &_dependencies);
    if (!_dependencies.empty()) {
      _clone = _value;
      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::NotInline;
    } else {
      _clone = shards::Var::Empty;
      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreConst;
    }
    return _innerInfo;
  }

  SHExposedTypesInfo requiredVariables() { return SHExposedTypesInfo{_dependencies.data(), uint32_t(_dependencies.size())}; }

  void cleanup(SHContext *context) { resolver.cleanup(context); }

  void warmup(SHContext *context) {
    if (_clone != shards::Var::Empty)
      resolver.warmup(_value, _clone, context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    // we need to reassign values every frame
    resolver.reassign();
    return _clone;
  }
};

static shards::ParamsInfo compareParamsInfo =
    shards::ParamsInfo(shards::ParamsInfo::Param("Value", SHCCSTR("The value to check against."), CoreInfo::AnyType));

static shards::ParamsInfo compareParamsInfo2 = shards::ParamsInfo(shards::ParamsInfo::Param(
    "Value",
    SHCCSTR("The value to check against. Note: This shard will not evaluate variables nested within "
            "sequences and tables. If you need to compare them against such variables, wrap them in "
            "parentheses, so that they are evaluated first. Example: Value:([your-variable]) instead of Value:your-variable."),
    CoreInfo::AnyType));

struct BaseOpsBin {
  ParamVar _operand{shards::Var(0)};
  ExposedInfo _requiredVariables;

  static SHOptionalString help() { return SHCCSTR("Compares the input value with a given value."); }

  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value to compare against the operand."); }

  SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The result of the comparison."); }

  static SHParametersInfo parameters() { return SHParametersInfo(compareParamsInfo); }

  SHExposedTypesInfo requiredVariables() {
    _requiredVariables.clear();
    mergeIntoExposedInfo(_requiredVariables, _operand, CoreInfo::AnyType);
    return SHExposedTypesInfo(_requiredVariables);
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _operand = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _operand;
    default:
      return shards::Var::Empty;
    }
  }

  void warmup(SHContext *context) { _operand.warmup(context); }
  void cleanup(SHContext *context) { _operand.cleanup(); }

  shards::Var _output;
};

#define LOGIC_OP(NAME, OP, HELP_TEXT, OUTPUT_HELP_TEXT)                                   \
  struct NAME : public BaseOpsBin {                                                       \
    FLATTEN ALWAYS_INLINE const SHVar &activate(SHContext *context, const SHVar &input) { \
      const auto &value = _operand.get();                                                 \
      if (input OP value) {                                                               \
        _output = shards::Var::True;                                                      \
        return _output;                                                                   \
      }                                                                                   \
      _output = shards::Var::False;                                                       \
      return _output;                                                                     \
    }                                                                                     \
    static SHOptionalString help() { return SHCCSTR(HELP_TEXT); }                         \
    static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyType; }     \
    static SHOptionalString outputHelp() { return SHCCSTR(OUTPUT_HELP_TEXT); }            \
  };                                                                                      \
  RUNTIME_CORE_SHARD_TYPE(NAME);

// Now use the updated macro with help text for each operation
LOGIC_OP(Is, ==, "Checks if the input is equal to the operand.",
         "Outputs true if the input is equal to the operand and false otherwise.");
LOGIC_OP(IsNot, !=, "Checks if the input is not equal to the operand.",
         "Outputs true if the input is not equal to the operand and false otherwise.");
LOGIC_OP(IsMore, >, "Checks if the input is greater than the operand.",
         "Outputs true if the input is greater than the operand and false otherwise.");
LOGIC_OP(IsLess, <, "Checks if the input is less than the operand.",
         "Outputs true if the input is less than the operand and false otherwise.");
LOGIC_OP(IsMoreEqual, >=, "Checks if the input is greater than or equal to the operand.",
         "Outputs true if the input is greater than or equal to the operand and false otherwise.");
LOGIC_OP(IsLessEqual, <=, "Checks if the input is less than or equal to the operand.",
         "Outputs true if the input is less than or equal to the operand and false otherwise.");

#define LOGIC_ANY_SEQ_OP(NAME, OP, HELP_TEXT, OUTPUT_HELP_TEXT)                           \
  struct NAME : public BaseOpsBin {                                                       \
    const SHVar &activate(SHContext *context, const SHVar &input) {                       \
      const auto &value = _operand.get();                                                 \
                                                                                          \
      if (input.valueType == SHType::Seq && value.valueType == SHType::Seq) {             \
        auto vlen = value.payload.seqValue.len;                                           \
        auto ilen = input.payload.seqValue.len;                                           \
                                                                                          \
        /* If both sequences are empty, return False (no elements satisfy ANY) */         \
        if (ilen == 0 && vlen == 0) {                                                     \
          _output = shards::Var::False;                                                   \
          return _output;                                                                 \
        } else if (ilen == 0 || vlen == 0) {                                              \
          /* If one is empty and the other is not, there are no pairs to satisfy ANY */   \
          _output = shards::Var::False;                                                   \
          return _output;                                                                 \
        }                                                                                 \
                                                                                          \
        if (ilen > vlen)                                                                  \
          throw ActivationError("Failed to compare, input len > operand len.");           \
                                                                                          \
        for (uint32_t i = 0; i < ilen; i++) {                                             \
          if (input.payload.seqValue.elements[i] OP value.payload.seqValue.elements[i]) { \
            _output = shards::Var::True;                                                  \
            return _output;                                                               \
          }                                                                               \
        }                                                                                 \
        _output = shards::Var::False;                                                     \
        return _output;                                                                   \
      } else if (input.valueType == SHType::Seq && value.valueType != SHType::Seq) {      \
        auto ilen = input.payload.seqValue.len;                                           \
                                                                                          \
        /* If input sequence is empty, return False (no elements satisfy ANY) */          \
        if (ilen == 0) {                                                                  \
          _output = shards::Var::False;                                                   \
          return _output;                                                                 \
        }                                                                                 \
                                                                                          \
        for (uint32_t i = 0; i < ilen; i++) {                                             \
          if (input.payload.seqValue.elements[i] OP value) {                              \
            _output = shards::Var::True;                                                  \
            return _output;                                                               \
          }                                                                               \
        }                                                                                 \
        _output = shards::Var::False;                                                     \
        return _output;                                                                   \
      } else if (input.valueType != SHType::Seq && value.valueType == SHType::Seq) {      \
        auto vlen = value.payload.seqValue.len;                                           \
                                                                                          \
        /* If operand sequence is empty, return False (no elements satisfy ANY) */        \
        if (vlen == 0) {                                                                  \
          _output = shards::Var::False;                                                   \
          return _output;                                                                 \
        }                                                                                 \
                                                                                          \
        for (uint32_t i = 0; i < vlen; i++) {                                             \
          if (input OP value.payload.seqValue.elements[i]) {                              \
            _output = shards::Var::True;                                                  \
            return _output;                                                               \
          }                                                                               \
        }                                                                                 \
        _output = shards::Var::False;                                                     \
        return _output;                                                                   \
      } else if (input.valueType != SHType::Seq && value.valueType != SHType::Seq) {      \
        /* Handle non-sequence types */                                                   \
        if (input OP value) {                                                             \
          _output = shards::Var::True;                                                    \
          return _output;                                                                 \
        }                                                                                 \
        _output = shards::Var::False;                                                     \
        return _output;                                                                   \
      }                                                                                   \
                                                                                          \
      /* Fallback case: if none of the above conditions are met */                        \
      _output = shards::Var::False;                                                       \
      return _output;                                                                     \
    }                                                                                     \
                                                                                          \
    static SHOptionalString help() { return SHCCSTR(HELP_TEXT); }                         \
    static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyButType; }  \
    static SHOptionalString outputHelp() { return SHCCSTR(OUTPUT_HELP_TEXT); }            \
    static SHParametersInfo parameters() { return SHParametersInfo(compareParamsInfo2); } \
  };                                                                                      \
  RUNTIME_CORE_SHARD_TYPE(NAME);

#define LOGIC_ALL_SEQ_OP(NAME, OP, HELP_TEXT, OUTPUT_HELP_TEXT)                              \
  struct NAME : public BaseOpsBin {                                                          \
    const SHVar &activate(SHContext *context, const SHVar &input) {                          \
      const auto &value = _operand.get();                                                    \
                                                                                             \
      if (input.valueType == SHType::Seq && value.valueType == SHType::Seq) {                \
        auto vlen = value.payload.seqValue.len;                                              \
        auto ilen = input.payload.seqValue.len;                                              \
                                                                                             \
        /* If both sequences are empty, return True (vacuously true for ALL) */              \
        if (ilen == 0 && vlen == 0) {                                                        \
          _output = shards::Var::True;                                                       \
          return _output;                                                                    \
        } else if (ilen == 0 || vlen == 0) {                                                 \
          _output = shards::Var::False;                                                      \
          return _output;                                                                    \
        }                                                                                    \
                                                                                             \
        if (ilen > vlen)                                                                     \
          throw ActivationError("Failed to compare, input len > operand len.");              \
                                                                                             \
        for (uint32_t i = 0; i < ilen; i++) {                                                \
          if (!(input.payload.seqValue.elements[i] OP value.payload.seqValue.elements[i])) { \
            _output = shards::Var::False;                                                    \
            return _output;                                                                  \
          }                                                                                  \
        }                                                                                    \
        _output = shards::Var::True;                                                         \
        return _output;                                                                      \
      } else if (input.valueType == SHType::Seq && value.valueType != SHType::Seq) {         \
        auto ilen = input.payload.seqValue.len;                                              \
                                                                                             \
        /* If input sequence is empty, return True (vacuously true for ALL) */               \
        if (ilen == 0) {                                                                     \
          _output = shards::Var::True;                                                       \
          return _output;                                                                    \
        }                                                                                    \
                                                                                             \
        for (uint32_t i = 0; i < ilen; i++) {                                                \
          if (!(input.payload.seqValue.elements[i] OP value)) {                              \
            _output = shards::Var::False;                                                    \
            return _output;                                                                  \
          }                                                                                  \
        }                                                                                    \
        _output = shards::Var::True;                                                         \
        return _output;                                                                      \
      } else if (input.valueType != SHType::Seq && value.valueType == SHType::Seq) {         \
        auto vlen = value.payload.seqValue.len;                                              \
                                                                                             \
        /* If operand sequence is empty, return True (vacuously true for ALL) */             \
        if (vlen == 0) {                                                                     \
          _output = shards::Var::True;                                                       \
          return _output;                                                                    \
        }                                                                                    \
                                                                                             \
        for (uint32_t i = 0; i < vlen; i++) {                                                \
          if (!(input OP value.payload.seqValue.elements[i])) {                              \
            _output = shards::Var::False;                                                    \
            return _output;                                                                  \
          }                                                                                  \
        }                                                                                    \
        _output = shards::Var::True;                                                         \
        return _output;                                                                      \
      } else if (input.valueType != SHType::Seq && value.valueType != SHType::Seq) {         \
        /* Handle non-sequence types */                                                      \
        if (!(input OP value)) {                                                             \
          _output = shards::Var::False;                                                      \
          return _output;                                                                    \
        }                                                                                    \
        _output = shards::Var::True;                                                         \
        return _output;                                                                      \
      }                                                                                      \
                                                                                             \
      /* Fallback case: if none of the above conditions are met */                           \
      _output = shards::Var::False;                                                          \
      return _output;                                                                        \
    }                                                                                        \
                                                                                             \
    static SHOptionalString help() { return SHCCSTR(HELP_TEXT); }                            \
    static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpAnyButType; }     \
    static SHOptionalString outputHelp() { return SHCCSTR(OUTPUT_HELP_TEXT); }               \
    static SHParametersInfo parameters() { return SHParametersInfo(compareParamsInfo2); }    \
  };                                                                                         \
  RUNTIME_CORE_SHARD_TYPE(NAME);

LOGIC_ANY_SEQ_OP(IsAny, ==,
                 "Checks if any element in the input is equal to the given value. It outputs true if any element is equal and "
                 "false otherwise.",
                 "Outputs true if any element in the input is equal to the specified value and false otherwise.");
LOGIC_ALL_SEQ_OP(IsAll, ==,
                 "Checks if all elements in the input are equal to the given value. It outputs true if all elements are equal "
                 "and false otherwise.",
                 "Outputs true if all elements in the input are equal to the specified value and false otherwise.");
LOGIC_ANY_SEQ_OP(IsAnyNot, !=,
                 "Checks if any element in the input is not equal to the given value. It outputs true if any element is not "
                 "equal and false otherwise.",
                 "Outputs true if any element in the input is not equal to the specified value and false otherwise.");
LOGIC_ALL_SEQ_OP(IsAllNot, !=,
                 "Checks if all elements in the input are not equal to the given value. It outputs true if all elements are "
                 "not equal and false otherwise.",
                 "Outputs true if all elements in the input are not equal to the specified value and false otherwise.");
LOGIC_ANY_SEQ_OP(IsAnyMore, >,
                 "Checks if any element in the input is greater than the given value. It outputs true if any element is "
                 "greater and false otherwise.",
                 "Outputs true if any element in the input is greater than the specified value and false otherwise.");
LOGIC_ALL_SEQ_OP(IsAllMore, >,
                 "Checks if all elements in the input are greater than the given value. It outputs true if all elements are "
                 "greater and false otherwise.",
                 "Outputs true if all elements in the input are greater than the specified value and false otherwise.");
LOGIC_ANY_SEQ_OP(IsAnyLess, <,
                 "Checks if any element in the input is less than the given value. It outputs true if any element is less and "
                 "false otherwise.",
                 "Outputs true if any element in the input is less than the specified value and false otherwise.");
LOGIC_ALL_SEQ_OP(IsAllLess, <,
                 "Checks if all elements in the input are less than the given value. It outputs true if all elements are less "
                 "and false otherwise.",
                 "Outputs true if all elements in the input are less than the specified value and false otherwise.");
LOGIC_ANY_SEQ_OP(
    IsAnyMoreEqual, >=,
    "Checks if any element in the input is greater than or equal to the given value. It outputs true if any element is "
    "greater or equal and false otherwise.",
    "Outputs true if any element in the input is greater than or equal to the specified value and false otherwise.");
LOGIC_ALL_SEQ_OP(
    IsAllMoreEqual, >=,
    "Checks if all elements in the input are greater than or equal to the given value. It outputs true if all elements are "
    "greater or equal and false otherwise.",
    "Outputs true if all elements in the input are greater than or equal to the specified value and false otherwise.");
LOGIC_ANY_SEQ_OP(IsAnyLessEqual, <=,
                 "Checks if any element in the input is less than or equal to the given value. It outputs true if any element "
                 "is less or equal and false otherwise.",
                 "Outputs true if any element in the input is less than or equal to the specified value and false otherwise.");
LOGIC_ALL_SEQ_OP(
    IsAllLessEqual, <=,
    "Checks if all elements in the input are less than or equal to the given value. It outputs true if all elements are less "
    "or equal and false otherwise.",
    "Outputs true if all elements in the input are less than or equal to the specified value and false otherwise.");

#define LOGIC_OP_DESC(NAME)              \
  RUNTIME_CORE_SHARD_FACTORY(NAME);      \
  RUNTIME_SHARD_cleanup(NAME);           \
  RUNTIME_SHARD_warmup(NAME);            \
  RUNTIME_SHARD_inputTypes(NAME);        \
  RUNTIME_SHARD_outputTypes(NAME);       \
  RUNTIME_SHARD_parameters(NAME);        \
  RUNTIME_SHARD_setParam(NAME);          \
  RUNTIME_SHARD_getParam(NAME);          \
  RUNTIME_SHARD_activate2(NAME);         \
  RUNTIME_SHARD_requiredVariables(NAME); \
  RUNTIME_SHARD_help(NAME);              \
  RUNTIME_SHARD_inputHelp(NAME);         \
  RUNTIME_SHARD_outputHelp(NAME);        \
  RUNTIME_SHARD_parameters(NAME);        \
  RUNTIME_SHARD_END(NAME);

struct Input {
  static SHOptionalString help() { return SHCCSTR("The input value of the wire."); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return SHCCSTR("No input is required."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input value of the wire."); }

  SHTypeInfo compose(const SHInstanceData &data) { return data.wire->inputType; }

  FLATTEN ALWAYS_INLINE const SHVar &activate(SHContext *context, const SHVar &input) {
    return context->wireStack.back()->currentInput;
  }
};

struct Pause {
  static SHOptionalString help() { return SHCCSTR("Pauses the wire for a given amount of time."); }

  ExposedInfo reqs{};
  static inline Parameters params{
      {"Time",
       SHCCSTR("The amount of time in seconds (can be fractional) to pause "
               "this wire."),
       {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::IntType, CoreInfo::FloatVarType, CoreInfo::IntVarType}}};

  ParamVar time{};

  static inline thread_local shards::Time::DoOnce doOnce{2s};

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Passes the input value through."); }

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) { time = value; }

  SHVar getParam(int index) { return time; }

  SHTypeInfo compose(const SHInstanceData &data) {
    reqs.clear();
    collectAllRequiredVariables(data, reqs, time);

    return data.inputType;
  }

  void warmup(SHContext *context) { time.warmup(context); }

  void cleanup(SHContext *context) { time.cleanup(); }

  SHExposedTypesInfo requiredVariables() { return SHExposedTypesInfo(reqs); }

  FLATTEN ALWAYS_INLINE SHVar activate(SHContext *context, const SHVar &input) {
    const auto &t = time.get();
    if (context->onWorkerThread) {
      doOnce([&]() {
        SHLOG_WARNING("Avoid using Pause on worker threads, it will block the worker thread. (wire: {})",
                      context->currentWire()->name);
      });
      if (t.valueType == SHType::None)
        std::this_thread::yield();
      else if (t.valueType == SHType::Int)
        std::this_thread::sleep_for(std::chrono::duration<SHInt, std::ratio<1>>(t.payload.intValue));
      else
        std::this_thread::sleep_for(std::chrono::duration<double, std::ratio<1>>(t.payload.floatValue));
      return input;
    } else {
      if (t.valueType == SHType::None)
        suspend(context, 0.0);
      else if (t.valueType == SHType::Int)
        suspend(context, double(t.payload.intValue));
      else
        suspend(context, t.payload.floatValue);
      return input;
    }
  }
};

struct PauseMs : public Pause {
  static inline Parameters params{{"Time",
                                   SHCCSTR("The amount of time in milliseconds to pause this wire."),
                                   {CoreInfo::NoneType, CoreInfo::IntType, CoreInfo::IntVarType}}};

  static SHParametersInfo parameters() { return params; }

  SHExposedTypesInfo requiredVariables() {
    if (time.isVariable()) {
      reqs = ExposedInfo(ExposedInfo::Variable(time.variableName(), SHCCSTR("The required variable"), CoreInfo::IntType));
    } else {
      reqs = {};
    }
    return SHExposedTypesInfo(reqs);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    const auto &t = time.get();
    if (t.valueType == SHType::None)
      suspend(context, 0.0);
    else
      suspend(context, double(t.payload.intValue) / 1000.0);
    return input;
  }
};

struct Comment {
  std::string _comment;

  static SHOptionalString help() { return SHCCSTR("A comment shard that does nothing."); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("Passes the input value through."); }

  static inline Parameters params{{"Text", SHCCSTR("The comment's text."), {CoreInfo::StringType}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) { _comment = SHSTRVIEW(value); }

  SHVar getParam(int index) { return shards::Var(_comment); }

  SHVar activate(SHContext *context, const SHVar &input) {
    // We are a NOOP shard
    SHLOG_FATAL("invalid state");
    return input;
  }
};

struct And {
  static SHOptionalString help() {
    return SHCCSTR("If the input of the preceding shard is true, the flow continues; otherwise, the flow stops. This shard is "
                   "typically used within conditional flows (e.g., If, When) to chain conditions. Note: Outside a conditional "
                   "flow, it might restart the current wire, which can be used as a trick in certain scenarios.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString inputHelp() { return SHCCSTR("If true, the flow continues; otherwise, it stops."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The output of this shard will be the input of the current conditional flow or wire.");
  }

  const SHVar &activate(SHContext *context, const SHVar &input) {
    if (input.payload.boolValue) {
      // Continue the flow
      context->rebaseFlow();
      return input;
    } else {
      // Stop the flow
      context->returnFlow(input);
      return input;
    }
  }
};

struct Or {
  static SHOptionalString help() {
    return SHCCSTR("Computes the logical OR between the input of this shard and the output of the next shard. If the input is "
                   "true, the flow stops and succeeds; if false, the flow continues with the next shard. Typically used within "
                   "conditional flows (e.g., If, When) to chain conditions. Note: Outside a conditional flow, it might restart "
                   "the current wire, which can be used as a trick in certain scenarios.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString inputHelp() { return SHCCSTR("If true, the flow stops and succeeds; otherwise, the flow continues."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The output of this shard will be the input of the current conditional flow or wire.");
  }

  const SHVar &activate(SHContext *context, const SHVar &input) {
    if (input.payload.boolValue) {
      // Stop the flow and succeed
      context->returnFlow(input);
      return input;
    } else {
      // Continue the flow with the initial input
      context->rebaseFlow();
      return input;
    }
  }
};

struct Not {
  static SHOptionalString help() { return SHCCSTR("Computes the logical negation of the input."); }

  static SHTypesInfo inputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value to be negated."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The negation of the input."); }

  shards::Var output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    output = shards::Var(!input.payload.boolValue);
    return output;
  }
};

struct IsNone {
  static SHOptionalString help() { return SHCCSTR("Gets whether the type of the input is `None`."); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value which type to check against."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() { return SHCCSTR("`true` is the type of input is `None`; otherwise, `false`."); }

  shards::Var output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    output = shards::Var(input.valueType == SHType::None);
    return output;
  }
};

struct IsNotNone {
  static SHOptionalString help() { return SHCCSTR("Gets whether the type of the input is different from `None`."); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value which type to check against."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("`true` is the type of input different from `None`; otherwise, `false`.");
  }

  shards::Var output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    output = shards::Var(input.valueType != SHType::None);
    return output;
  }
};

struct IsTrue {
  static SHOptionalString help() { return SHCCSTR("Gets whether the input is `true`."); }

  static SHTypesInfo inputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value to check against."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() { return SHCCSTR("`true` if the input is `true`; otherwise, `false`."); }

  shards::Var output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    output = shards::Var(input.payload.boolValue);
    return output;
  }
};

struct IsFalse {
  static SHOptionalString help() { return SHCCSTR("Gets whether the input is `false`."); }

  static SHTypesInfo inputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value to check against."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() { return SHCCSTR("`true` if the input is `false`; otherwise, `false`."); }

  shards::Var output;
  const SHVar &activate(SHContext *context, const SHVar &input) {
    output = shards::Var(!input.payload.boolValue);
    return output;
  }
};

struct Restart {
  // Ensures the input type matches the wire root input type.
  SHTypeInfo compose(const SHInstanceData &data) {
    if (data.wire->inputType->basicType != SHType::None && !matchTypes(data.inputType, data.wire->inputType, false, true, true)) {
      throw ComposeError("Restart input and wire input type mismatch, Restart "
                         "feeds back to the wire input, wire: " +
                         data.wire->name + " expected: " + type2Name(data.wire->inputType->basicType));
    }
    return data.inputType; // Actually, we are a flow stopper.
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    context->restartFlow(input);
    return input;
  }

  static SHOptionalString help() {
    return SHCCSTR(
        "Restarts the current flow with the provided input. This shard is used to restart the execution of the current wire from "
        "the beginning, using the same input. It ensures that the input type matches the wire's root input type. Note: This is a "
        "flow stopper and will not continue to subsequent shards in the current execution sequence.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input to restart the wire with. Must match the wire's root input type.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard does not produce an output as it restarts the flow."); }
};

struct Return {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    context->returnFlow(input);
    return input;
  }

  static SHOptionalString help() {
    return SHCCSTR("Stops the current flow and outputs the provided input. This shard is used to exit the execution of the "
                   "current wire early within loops or conditional flows, returning the specified input.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The input to return (when supported) and stop the flow."); }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard does not produce an output as it stops the flow."); }
};

struct Fail {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    context->cancelFlow(SHSTRVIEW(input));
    return input;
  }

  static SHOptionalString help() {
    return SHCCSTR("Stops the current flow and cancels the execution with the provided error message. This shard is used to "
                   "signal an error and halt the execution of the current wire.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The error message to cancel the flow with."); }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard does not produce an output as it cancels the flow."); }
};

struct IsValidNumber {
  static SHTypesInfo inputTypes() { return CoreInfo::FloatType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }

  SHVar activate(SHContext *context, const SHVar &input) { return shards::Var(std::isnormal(input.payload.floatValue)); }

  static SHOptionalString help() {
    return SHCCSTR("Checks if the input is a valid floating-point number (not zero, subnormal, infinity, or NaN). Outputs true "
                   "if the input is a normal floating-point number, otherwise outputs false.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The floating-point number to be checked."); }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs true if the input is a valid normal floating-point number, otherwise outputs false.");
  }
};

struct IsAlmost {
  SHOptionalString help() { return SHCCSTR("Checks whether the input is almost equal to a given value."); }

  static SHTypesInfo inputTypes() { return MathTypes; }
  static SHOptionalString inputHelp() { return SHCCSTR("The input can be of any number type or a sequence of such types."); }

  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("true if the input is almost equal to the given value; otherwise, false.");
  }

  SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &inValue) {
    switch (index) {
    case 0:
      destroyVar(_value);
      cloneVar(_value, inValue);
      break;
    case 1:
      if (inValue.valueType == SHType::Float)
        _threshold = inValue.payload.floatValue;
      else
        _threshold = double(inValue.payload.intValue);
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _value;
    case 1:
      return Var(_threshold);
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_value.valueType != data.inputType.basicType)
      throw ActivationError("Input and value types must match.");

    if (_threshold <= 0.0)
      throw ActivationError("Threshold must be greater than 0.");

    return CoreInfo::BoolType;
  }

  void destroy() { destroyVar(_value); }

  SHVar activate(SHContext *context, const SHVar &input) { return Var(_almostEqual(input, _value, _threshold)); }

private:
  static inline Types MathTypes{{
      CoreInfo::FloatType,
      CoreInfo::Float2Type,
      CoreInfo::Float3Type,
      CoreInfo::Float4Type,
      CoreInfo::IntType,
      CoreInfo::Int2Type,
      CoreInfo::Int3Type,
      CoreInfo::Int4Type,
      CoreInfo::Int8Type,
      CoreInfo::Int16Type,
      CoreInfo::AnySeqType,
  }};
  static inline Parameters _params = {{"Value", SHCCSTR("The value to test against for almost equality."), MathTypes},
                                      {"Threshold",
                                       SHCCSTR("The smallest difference to be considered equal. Should be greater than zero."),
                                       {CoreInfo::FloatType, CoreInfo::IntType}}};

  SHFloat _threshold{FLT_EPSILON};
  SHVar _value{};
};

struct NaNTo0 {
  static SHOptionalString help() {
    return SHCCSTR("Replaces NaN (Not a Number) values in the input with 0. This shard can handle both single float values and "
                   "sequences of float values.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("A float value or a sequence of float values to be checked for NaN."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The input with any NaN values replaced by 0."); }

  static SHTypesInfo inputTypes() { return CoreInfo::FloatOrFloatSeq; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatOrFloatSeq; }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (data.inputType.basicType == SHType::Float) {
      OVERRIDE_ACTIVATE(data, activateSingle);
    } else {
      OVERRIDE_ACTIVATE(data, activateSeq);
    }
    return data.inputType;
  }

  SHVar activateSingle(SHContext *context, const SHVar &input) {
    if (std::isnan(input.payload.floatValue)) {
      return shards::Var(0);
    } else {
      return input;
    }
  }

  SHVar activateSeq(SHContext *context, const SHVar &input) {
    // violate const.. edit in place
    auto violatedInput = const_cast<SHVar &>(input);
    for (uint32_t i = 0; i < input.payload.seqValue.len; i++) {
      auto val = violatedInput.payload.seqValue.elements[i];
      if (std::isnan(val.payload.floatValue)) {
        violatedInput.payload.seqValue.elements[i].payload.floatValue = 0.0;
      }
    }
    return input;
  }

  SHVar activate(SHContext *context, const SHVar &input) { throw ActivationError("Unexpected code path"); }
};

struct VariableBase {
  SHVar *_target{nullptr};
  SHVar *_cell{nullptr};
  std::string _name;
  ParamVar _key{};
  ExposedInfo _exposedInfo{};
  bool _isTable{false};
  bool _global{false};
  void *_tablePtr{nullptr};
  uint64_t _tableVersion{0};

  static inline Parameters getterParams{
      {"Name", SHCCSTR("The name of the variable."), {CoreInfo::StringOrAnyVar}},
      {"Key",
       SHCCSTR("The key of the value to read from the table (parameter applicable only if "
               "the target variable is a table)."),
       {CoreInfo::AnyType}},
      {"Global", SHCCSTR("If the variable is available to all of the wires in the same mesh."), {CoreInfo::BoolType}}};

  static inline Parameters setterParams{
      {"Name", SHCCSTR("The name of the variable."), {CoreInfo::StringOrAnyVar}, true}, // flagged as setter
      {"Key",
       SHCCSTR("The key of the value to write in the table (parameter applicable only if "
               "the target variable is a table)."),
       {CoreInfo::AnyType}},
      {"Global", SHCCSTR("If the variable is available to all of the wires in the same mesh."), {CoreInfo::BoolType}}};

  static constexpr int variableParamsInfoLen = 3;

  static SHParametersInfo parameters() { return getterParams; }

  void cleanup(SHContext *context) {
    if (_target) {
      releaseVariable(_target);
    }
    _key.cleanup();
    _target = nullptr;
    _cell = nullptr;
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0: // Name
      _name = SHSTRVIEW(value);
      break;
    case 1: // Key
      if (value.valueType == SHType::None) {
        _isTable = false;
      } else {
        _isTable = true;
      }
      _key = value;
      break;
    case 2: // Global
      _global = value.payload.boolValue;
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0: // Name
      return Var(_name);
    case 1: // Key
      return _key;
    case 2: // Global
      return Var(_global);
    default:
      throw InvalidParameterIndex();
    }
  }

  ALWAYS_INLINE void checkIfTableChanged() {
    if (_tablePtr != _target->payload.tableValue.opaque || _tableVersion != _target->version) {
      _tablePtr = _target->payload.tableValue.opaque;
      _cell = nullptr;
      _tableVersion = _target->version;
    }
  }
};

// Takes an original table type and either adds a new key/type pair or updates an existing one
// Returns the full new type info, the types/keys are owned by the typeInfoStorage and should be manually freed
//  keyToAdd can be null to indicate a variable key, which will convert this into an unkeyed table
static const SHTypeInfo &updateTableType(SHTypeInfo &typeInfoStorage, const SHVar *keyToAdd, const SHTypeInfo &typeToAdd,
                                         const SHTypeInfo *existingType = nullptr) {
  std::optional<size_t> replacedIndex;

  SHTypesInfo &typeStorage = typeInfoStorage.table.types;
  SHSeq &keyStorage = typeInfoStorage.table.keys;
  typeInfoStorage.basicType = SHType::Table;

  shards::arrayResize(typeStorage, 0);
  shards::arrayResize(keyStorage, 0);

  if (existingType) {
    auto &keys = existingType->table.keys;

    // Add keys, or leave empty
    // empty key means it's a variable
    if (keyToAdd) {
      for (size_t i = 0; i < keys.len; i++) {
        if (keyToAdd && keys.elements[i] == *keyToAdd) {
          replacedIndex = i;
        }

        shards::arrayPush(keyStorage, keys.elements[i]);
      }
    }

    for (size_t i = 0; i < existingType->table.types.len; i++) {
      if (replacedIndex && i == replacedIndex.value()) {
        // Update an existing key's matching type
        shards::arrayPush(typeStorage, typeToAdd);
      } else {
        shards::arrayPush(typeStorage, existingType->table.types.elements[i]);
      }
    }
  }

  // Add a new type
  if (!replacedIndex) {
    if (keyToAdd)
      shards::arrayPush(keyStorage, *keyToAdd);
    shards::arrayPush(typeStorage, typeToAdd);
  }

  return typeInfoStorage;
}

// Takes an original seq type and adds the new type to it
// Returns the full new type info, the types are owned by typeInfoStorage and should be manually freed
static const SHTypeInfo &updateSeqType(SHTypeInfo &typeInfoStorage, const SHTypeInfo &typeToAdd,
                                       const SHTypeInfo *existingType = nullptr) {
  std::optional<size_t> replacedIndex;

  shards::arrayResize(typeInfoStorage.seqTypes, 0);
  if (existingType) {
    for (size_t i = 0; i < existingType->seqTypes.len; i++) {
      // Already matching type in sequence type
      if (matchTypes(typeToAdd, existingType->seqTypes.elements[i], false, true, false)) {
        replacedIndex = i;
      }
      shards::arrayPush(typeInfoStorage.seqTypes, existingType->seqTypes.elements[i]);
    }
  }

  // Add new entry
  if (!replacedIndex) {
    shards::arrayPush(typeInfoStorage.seqTypes, typeToAdd);
  }

  typeInfoStorage.basicType = SHType::Seq;

  return typeInfoStorage;
}

struct SetBase : public VariableBase {
  Type _tableTypeInfo{};
  SHTypeInfo _tableContentInfo{};
  bool _isExposed{false}; // notice this is used in Update only

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The value to be set to the variable."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input value is passed through as the output."); }

  // Runs sanity checks on the target variable, returns the existing exposed type if any
  const SHExposedTypeInfo *setBaseCompose(const SHInstanceData &data, bool warnIfExists, bool failIfExists, bool overwrite) {
    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
    const SHExposedTypeInfo *existingExposedType = findExposedVariablePtr(inherited->inherited, _name);
    if (existingExposedType) {
      auto &reference = *existingExposedType;
      if (_isTable) {
        if (reference.exposedType.basicType != SHType::Table) {
          throw ComposeError(fmt::format("Set/Ref/Update, variable \"{}\" was not a table", _name));
        }
      } else {
        if (
            // need to check if this was just a any table definition {}
            !(reference.exposedType.basicType == SHType::Table && reference.exposedType.table.types.len == 0) &&
            data.inputType != reference.exposedType) {
          throw ComposeError(fmt::format("Set/Ref/Update, variable {} already set as another type: {} (new type: {})", _name,
                                         reference.exposedType, data.inputType));
        }
        if (failIfExists && !overwrite) {
          throw ComposeError(fmt::format("Ref, variable \"{}\" already exists", _name));
        } else if (warnIfExists && !overwrite) {
          SHLOG_INFO("Set - Warning: setting an already exposed variable \"{}\", use Update to avoid this warning.", _name);
        }
      }

      if (!overwrite && !reference.isMutable) {
        SHLOG_ERROR("Error with variable: {}", _name);
        throw ComposeError(fmt::format("Set/Ref/Update, attempted to write an immutable variable \"{}\".", _name));
      }
      if (reference.isProtected) {
        SHLOG_ERROR("Error with variable: {}", _name);
        throw ComposeError(fmt::format("Set/Ref/Update, attempted to write a protected variable \"{}\".", _name));
      }

      if (reference.tracked) {
        _isExposed = true;
      } else {
        _isExposed = false;
      }
    }

    return existingExposedType;
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceVariable(context, _name.c_str());
    _key.warmup(context);
  }
};

struct SetUpdateBase : public SetBase {
  Shard *_self{};
  entt::dispatcher *_dispatcherPtr{nullptr};

  void setupDispatcher(SHContext *context, bool isGlobal) { _dispatcherPtr = &context->main->mesh.lock()->dispatcher; }

  ALWAYS_INLINE const SHVar &activateTable(SHContext *context, const SHVar &input) noexcept {
    checkIfTableChanged();

    if (likely(_cell != nullptr)) {
      cloneVar(*_cell, input);
      return *_cell;
    }

    SHMap *table = nullptr;
    if (_target->valueType != SHType::Table) {
      if (_target->valueType != SHType::None)
        destroyVar(*_target);

      // Not initialized yet
      _target->valueType = SHType::Table;
      _target->payload.tableValue.api = &GetGlobals().TableInterface;
      table = _target->payload.tableValue.opaque = new SHMap();
    } else {
      table = static_cast<SHMap *>(_target->payload.tableValue.opaque);
    }

    if (input.valueType == SHType::Table && input.payload.tableValue.opaque == _target->payload.tableValue.opaque) {
      context->cancelFlow("Set/Update, attempted to set a variable to itself");
      return input;
    }

    auto &kv = _key.get();
    auto fk = shards::OwnedVar::Foreign(kv); // pass key copy on write
    // will use cloneVar and recycle memory under the hood
    auto &vRef = table->insert_or_assign(fk, input).first->second;

    // use fast cell from now
    // if variable we cannot do this tho!
    if (!_key.isVariable())
      _cell = &vRef;

    return vRef;
  }

  ALWAYS_INLINE const SHVar &activateRegular(SHContext *context, const SHVar &input) noexcept {
    // Clone will try to recycle memory and such
    cloneVar(*_target, input);
    return *_target;
  }

  SHVar activate(SHContext *context, const SHVar &input) { SHLOG_FATAL("Invalid code path, this should never be called"); }
};

struct Set : public SetUpdateBase {
  bool _tracked{false};
  entt::scoped_connection _onStartConnection{};
  struct OnStartHandler {
    Set *shard;
    SHWire *targetWire;
    void handle(SHWire::OnStartEvent &ev) {
      if (targetWire == ev.wire) {
        shard->onStart(ev);
      };
    }
  } _startHandler;

  SHTypeInfo _tableType{};

  static SHOptionalString help() { return SHCCSTR("Creates a mutable variable and assigns a value to it."); }

  static inline Parameters setParamsInfo{
      setterParams, {{"Tracked", SHCCSTR("If the variable should be marked as tracked."), {CoreInfo::BoolType}}}};

  static SHParametersInfo parameters() { return setParamsInfo; }

  void setParam(int index, const SHVar &value) {
    if (index < variableParamsInfoLen)
      VariableBase::setParam(index, value);
    else if (index == variableParamsInfoLen + 0) {
      _tracked = value.payload.boolValue;
    }
  }

  SHVar getParam(int index) {
    if (index < variableParamsInfoLen)
      return VariableBase::getParam(index);
    else if (index == variableParamsInfoLen + 0)
      return Var(_tracked);
    throw SHException("Param index out of range.");
  }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    _self = data.shard;

    const SHExposedTypeInfo *existingExposedType = setBaseCompose(data, true, false, false);

    bool global = _global;
    if (existingExposedType && existingExposedType->global) {
      global = true;
    }

    // bake exposed types
    if (_isTable) {
      // we are a table!
      _tableTypeInfo = updateTableType(_tableType, !_key.isVariable() ? &(SHVar &)_key : &Var::Empty, data.inputType,
                                       existingExposedType ? &existingExposedType->exposedType : nullptr);

      if (global) {
        _exposedInfo =
            ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed table."), _tableTypeInfo, true));
      } else {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed table."), _tableTypeInfo, true));
      }

      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreSetUpdateTable;
    } else {
      // just a variable!
      if (global) {
        _exposedInfo = ExposedInfo(
            ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed variable."), SHTypeInfo(data.inputType), true));
      } else {
        _exposedInfo =
            ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed variable."), SHTypeInfo(data.inputType), true));
      }

      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreSetUpdateRegular;
    }

    if (_tracked) {
      _exposedInfo._innerInfo.elements[0].tracked = true;
    } else {
      _exposedInfo._innerInfo.elements[0].tracked = false;
    }

    // always lift this limit in a Set/Update
    _exposedInfo._innerInfo.elements[0].exposedType.fixedSize = 0;

    // We declared this variable in this shard
    _exposedInfo._innerInfo.elements[0].declared = true;

    return data.inputType;
  }

  SHExposedTypesInfo exposedVariables() { return SHExposedTypesInfo(_exposedInfo); }

  void onStart(const SHWire::OnStartEvent &e) {
    if (_target->valueType != SHType::None) {
      // Ok so, at this point if we are exposed we might be also be Set!
      // if that is true we can disable this Shard completely from the graph !
      SHLOG_TRACE("Variable {} is exposed and already initialized, disabling shard", _name);
      shassert(_self && "Self should be valid at this point");
      const_cast<Shard *>(_self)->inlineShardId = InlineShard::NoopShard;
    }
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceWireVariable(context->currentWire(), _name.c_str());
    _key.warmup(context);

    shassert_extended(context, _self && "Self should be valid at this point");

    if (_tracked) {
      _target->flags |= SHVAR_FLAGS_TRACKED;

      // override shard default behavior
      const_cast<Shard *>(_self)->inlineShardId = InlineShard::NotInline;

      // need to defer the check to before we actually start running
      setupDispatcher(context, _global);
      _startHandler = OnStartHandler{this, context->currentWire()};
      _onStartConnection = context->main->dispatcher.sink<SHWire::OnStartEvent>().connect<&OnStartHandler::handle>(_startHandler);

      OnTrackedVarWarmup ev{context->main->id, _name, _key, _exposedInfo._innerInfo.elements[0], context->currentWire()};
      _dispatcherPtr->trigger(ev);
    } else {
      if (!_isTable) {
        if (_target->flags & SHVAR_FLAGS_TRACKED) {
          // something changed, we are no longer tracked
          // fixup activations and variable flags
          _target->flags &= ~SHVAR_FLAGS_TRACKED;
        }
      }

      // restore any possible deferred change here
      if (_isTable)
        const_cast<Shard *>(_self)->inlineShardId = InlineShard::CoreSetUpdateTable;
      else
        const_cast<Shard *>(_self)->inlineShardId = InlineShard::CoreSetUpdateRegular;
    }

    if (_global) {
      // need to add metadata to the global variable in the mesh
      mesh = context->main->mesh.lock();
      if (!mesh) {
        SHLOG_ERROR("Cannot add metadata to global variable {} because mesh is not available", _name);
        throw WarmupError("Cannot add metadata to global variable because mesh is not available");
      } else {
        mesh->setMetadata(_target, SHExposedTypeInfo(_exposedInfo._innerInfo.elements[0]));
      }
    }
  }

  std::shared_ptr<SHMesh> mesh;

  void cleanup(SHContext *context) {
    if (mesh) {
      // this is not perfect because will run only during Set,
      // but for now it's not an issue as we go thru all variables when composing
      // and then check if metadata is there or not
      mesh->releaseMetadata(_target);
      mesh.reset();
    }

    SetBase::cleanup(context);

    _onStartConnection.release();

    // for (size_t i = 0; i < _tableType.table.keys.len; i++) {
    //   shards::destroyVar(_tableType.table.keys.elements[i]);
    // }
    shards::arrayFree(_tableType.table.keys);
    shards::arrayFree(_tableType.table.types);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_tracked && "This shard should not be activated if variable not exposed");

    SHVar output;
    if (_isTable)
      output = activateTable(context, input);
    else
      output = activateRegular(context, input);

    assert(_dispatcherPtr != nullptr && "Dispatcher should be valid at this point");

    OnTrackedVarSet ev{context->main->id, _name, _key, *_target, _global, context->currentWire()};
    _dispatcherPtr->trigger(ev);

    return output;
  }
};

struct Ref : public SetBase {
  bool _overwrite{false};

  static SHOptionalString help() {
    return SHCCSTR("Creates an immutable reference variable. Once created this variable cannot be changed.");
  }

  static inline Parameters getParamsInfo{
      getterParams,
      {{"Overwrite", SHCCSTR("If the variable should be overwritten if it already exists."), {CoreInfo::BoolType}}}};

  static SHParametersInfo parameters() { return getParamsInfo; }

  void setParam(int index, const SHVar &value) {
    if (index < variableParamsInfoLen)
      VariableBase::setParam(index, value);
    else if (index == variableParamsInfoLen + 0) {
      _overwrite = value.payload.boolValue;
    }
  }

  SHVar getParam(int index) {
    if (index < variableParamsInfoLen)
      return VariableBase::getParam(index);
    else if (index == variableParamsInfoLen + 0)
      return Var(_overwrite);
    throw SHException("Param index out of range.");
  }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    setBaseCompose(data, false, true, _overwrite);

    // bake exposed types
    if (_isTable) {
      // we are a table!
      _tableContentInfo = data.inputType;
      if (_key.isVariable()) {
        // only add types info, we don't know keys
        _tableTypeInfo = SHTypeInfo{SHType::Table, {.table = {.types = {&_tableContentInfo, 1, 0}}}};
      } else {
        _tableTypeInfo = SHTypeInfo{SHType::Table, {.table = {.keys = {&(*_key), 1, 0}, .types = {&_tableContentInfo, 1, 0}}}};
      }
      _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed table."), _tableTypeInfo, false));

      // properly link the right call
      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreRefTable;
    } else {
      // just a variable!
      _exposedInfo =
          ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed variable."), SHTypeInfo(data.inputType), false));

      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreRefRegular;
    }

    // We declared this variable in this shard
    _exposedInfo._innerInfo.elements[0].declared = true;

    return data.inputType;
  }

  SHExposedTypesInfo exposedVariables() { return SHExposedTypesInfo(_exposedInfo); }

  void cleanup(SHContext *context) {
    if (_target) {
      // this is a special case
      // Ref will reference previous shard result..
      // Let's cleanup our storage so that release, if calls destroy
      // won't mangle the shard's variable memory
      const auto rc = _target->refcount;
      const auto flags = _target->flags;
      memset(_target, 0x0, sizeof(SHVar));
      _target->refcount = rc;
      _target->flags = flags;
      releaseVariable(_target);
    }
    _target = nullptr;

    _cell = nullptr;
    _key.cleanup();
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceWireVariable(context->currentWire(), _name.c_str());
    _key.warmup(context);
  }

  ALWAYS_INLINE SHVar activate(SHContext *context, const SHVar &input) {
    throw ActivationError("Invalid Ref activation function.");
  }

  ALWAYS_INLINE const SHVar &activateTable(SHContext *context, const SHVar &input) noexcept {
    checkIfTableChanged();

    if (likely(_cell != nullptr)) {
      memcpy(_cell, &input, sizeof(SHVar));
      return input;
    } else {
      if (_target->valueType != SHType::Table) {
        // Not initialized yet
        _target->valueType = SHType::Table;
        _target->payload.tableValue.api = &GetGlobals().TableInterface;
        _target->payload.tableValue.opaque = new SHMap();
      }

      auto &kv = _key.get();
      SHVar *vptr = _target->payload.tableValue.api->tableAt(_target->payload.tableValue, kv);

      // Notice, NO Cloning!
      memcpy(vptr, &input, sizeof(SHVar));

      // use fast cell from now
      if (!_key.isVariable())
        _cell = vptr;

      return input;
    }
  }

  ALWAYS_INLINE const SHVar &activateRegular(SHContext *context, const SHVar &input) noexcept {
    // must keep flags!
    assignVariableValue(*_target, input);
    return input;
  }
};

struct Update : public SetUpdateBase {
  bool _isGlobal{false};

  static SHOptionalString help() { return SHCCSTR("Modifies the value of an existing mutable variable."); }

  static SHTypeInfo *findTableNoneEntry(const SHTypeInfo *ti) {
    for (size_t i = 0; i < ti->table.keys.len; i++) {
      if (ti->table.keys.elements[i].valueType == SHType::None) {
        return &ti->table.types.elements[i];
      }
    }
    return nullptr;
  }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
    _self = data.shard;

    setBaseCompose(data, false, false, false);

    SHTypeInfo *originalTableType{};
    SHTypeInfo *tableInnerValueType{};

    // make sure we update to the same type
    if (_isTable) {
      // we are a table!
      _tableContentInfo = data.inputType;
      auto type = findExposedVariablePtr(inherited->inherited, _name);
      if (type && type->exposedType.basicType == SHType::Table) {
        originalTableType = const_cast<SHTypeInfo *>(&type->exposedType);

        auto &tableKeys = type->exposedType.table.keys;
        auto &tableTypes = type->exposedType.table.types;
        if (_key.isVariable()) {
          tableInnerValueType = findTableNoneEntry(originalTableType);
        } else {
          for (uint32_t y = 0; y < tableKeys.len; y++) {
            // if keys are populated they are not variables
            auto &key = tableKeys.elements[y];
            if (key == *_key) {
              tableInnerValueType = &tableTypes.elements[y];
            }
          }
          // Fallback to none type
          if (!tableInnerValueType) {
            tableInnerValueType = findTableNoneEntry(originalTableType);
          }
        }
        _isGlobal = type->global;
      }

      if (!originalTableType) {
        throw ComposeError("Update: error, original table type not found.");
      }

      if (!tableInnerValueType) {
        if (_key.isVariable()) {
          throw ComposeError(fmt::format(
              "Update: can not update table with variable key \"{}\" because it has no \"none\" type field information ({}).",
              *_key, *originalTableType));
        } else {
          throw ComposeError(fmt::format(
              "Update: can not update table with variable key \"{}\" because it is not present in the table type({}).", *_key,
              *originalTableType));
        }
      }

      if (!matchTypes(data.inputType, *tableInnerValueType, true, true, true, true)) {
        throw ComposeError(fmt::format("Update: error, update is changing table field for key {} from {} => {}", *_key,
                                       *tableInnerValueType, data.inputType));
      }

      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreSetUpdateTable;
    } else {
      auto type = findExposedVariablePtr(inherited->inherited, _name);
      if (type) {
        if (type->exposedType.basicType != SHType::Table && data.inputType != type->exposedType) {
          throw ComposeError("Update: error, update is changing the variable type.");
        }
        _isGlobal = type->global;
      } else {
        throw ComposeError(fmt::format("Update: error, variable {} is not exposed.", _name));
      }

      const_cast<Shard *>(data.shard)->inlineShardId = InlineShard::CoreSetUpdateRegular;
    }

    // bake exposed types
    if (_isTable) {
      // we are a table!
      _tableTypeInfo = *originalTableType;
      _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The updated table."), _tableTypeInfo, true));
    } else {
      // just a variable!
      _exposedInfo =
          ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The updated table."), SHTypeInfo(data.inputType), true));
    }

    // always lift this limit in a Set/Update
    _exposedInfo._innerInfo.elements[0].exposedType.fixedSize = 0;

    return data.inputType;
  }

  SHExposedTypesInfo requiredVariables() { return SHExposedTypesInfo(_exposedInfo); }

  void warmup(SHContext *context) {
    SetBase::warmup(context);

    shassert_extended(context, _self && "Self should be valid at this point");

    if (_isExposed) {
      if (!(_target->flags & SHVAR_FLAGS_TRACKED)) {
        throw WarmupError(fmt::format("Update: error, variable {} is not exposed.", _name));
      }

      const_cast<Shard *>(_self)->inlineShardId = InlineShard::NotInline;

      setupDispatcher(context, _isGlobal);
    } else {
      if (_target->flags & SHVAR_FLAGS_TRACKED) {
        throw WarmupError(fmt::format("Update: error, variable {} is exposed.", _name));
      }

      // restore any possible deferred change here
      if (_isTable)
        const_cast<Shard *>(_self)->inlineShardId = InlineShard::CoreSetUpdateTable;
      else
        const_cast<Shard *>(_self)->inlineShardId = InlineShard::CoreSetUpdateRegular;
    }
  }

  void cleanup(SHContext *context) { SetBase::cleanup(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    shassert_extended(context, _isExposed && "This shard should not be activated if variable not exposed");
    shassert_extended(context, _dispatcherPtr != nullptr && "Dispatcher should be valid at this point");

    SHVar output;
    if (_isTable) {
      output = activateTable(context, input);
      OnTrackedVarSet ev{context->main->id, _name, _key, *_cell, _isGlobal, context->currentWire()};
      _dispatcherPtr->trigger(ev);
    } else {
      output = activateRegular(context, input);
      OnTrackedVarSet ev{context->main->id, _name, _key, *_target, _isGlobal, context->currentWire()};
      _dispatcherPtr->trigger(ev);
    }

    return output;
  }
};

struct Get : public VariableBase {
  SHVar _defaultValue{};
  SHTypeInfo _defaultType{};
  std::vector<SHTypeInfo> _tableTypes{};
  std::vector<SHVar> _tableKeys{}; // should be fine not to be OwnedVar
  Shard *_shard{nullptr};

  static inline Parameters getParamsInfo{
      getterParams,
      {{"Default",
        SHCCSTR("The default value used if the variable is not set, the key is not present, or there is a type mismatch."),
        {CoreInfo::AnyType}}}};

  static SHParametersInfo parameters() { return getParamsInfo; }

  void setParam(int index, const SHVar &value) {
    if (index < variableParamsInfoLen)
      VariableBase::setParam(index, value);
    else if (index == variableParamsInfoLen + 0) {
      cloneVar(_defaultValue, value);
    }
  }

  SHVar getParam(int index) {
    if (index < variableParamsInfoLen)
      return VariableBase::getParam(index);
    else if (index == variableParamsInfoLen + 0)
      return _defaultValue;
    throw SHException("Param index out of range.");
  }

  void destroy() { freeDerivedInfo(_defaultType); }

  static SHOptionalString help() { return SHCCSTR("Reads the value of the specified variable."); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output is the value read from the specified variable."); }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);

    _shard = const_cast<Shard *>(data.shard);

    if (_defaultValue.valueType != SHType::None) {
      freeDerivedInfo(_defaultType);
      _defaultType = deriveTypeInfo(_defaultValue, data);
    }

    auto type = findExposedVariablePtr(inherited->inherited, _name);

    if (_isTable) {
      if (type) {
        if (type->exposedType.basicType != SHType::Table)
          throw ComposeError(fmt::format("Get: error, variable {} was not a table", _name));

        auto &tableKeys = type->exposedType.table.keys;
        auto &tableTypes = type->exposedType.table.types;
        if (tableKeys.len == tableTypes.len) {
          // if we have a name use it
          bool hasMagicNone = false; // we use none for any key such as @type({none: Type::String})
          std::optional<SHTypeInfo> magicNoneType;
          for (uint32_t y = 0; y < tableKeys.len; y++) {
            // if keys are populated they are not variables;
            auto &key = tableKeys.elements[y];
            if (key == _key) {
              return tableTypes.elements[y];
            } else if (key.valueType == SHType::None) {
              hasMagicNone = true;
              // only add once, if we got more types reset to Any
              if (!magicNoneType) {
                magicNoneType = tableTypes.elements[y];
              } else {
                if (*magicNoneType != tableTypes.elements[y])
                  magicNoneType = CoreInfo::AnyType;
              }
            }
          }

          if (hasMagicNone && _key.isVariable()) {
            // we got a variable key and we got a magic none
            // we can return the magic none type
            if (_defaultValue.valueType != SHType::None) {
              return _defaultType;
            } else {
              return *magicNoneType;
            }
          }
        } else {
          // we got no key names
          if (_defaultValue.valueType != SHType::None) {
            return _defaultType;
          } else if (tableTypes.len == 1) {
            // 1 type only so we assume we return that type
            return tableTypes.elements[0];
          } else {
            // multiple types...
            // return any, can still be used with ExpectX
            return CoreInfo::AnyType;
          }
        }

        if (type->isProtected) {
          throw ComposeError("Get (" + _name + "): Cannot Get, variable is protected.");
        }
      }

      if (_defaultValue.valueType != SHType::None) {
        return _defaultType;
      } else {
        if (_key.isVariable()) {
          throw ComposeError("Get (" + _name + ":" + std::string(_key.variableName()) +
                             "[variable]): Could not infer an output type, key not found "
                             "and no Default value provided.");
        } else {
          // refactored
          auto msg = fmt::format("Get ({}:{}): Could not infer an output type, key not found and no Default value provided.",
                                 _name, *_key);
          throw ComposeError(msg);
        }
      }
    } else {
      _tableTypes.clear();
      _tableKeys.clear();

      if (type) {
        if (type->isProtected) {
          throw ComposeError("Get (" + _name + "): Cannot Get, variable is protected.");
        }
        return type->exposedType;
      }

      // check if we can compose a table type
      if (_tableTypes.size() > 0) {
        auto outputTableType = SHTypeInfo(CoreInfo::AnyTableType);
        outputTableType.table.types = {&_tableTypes[0], uint32_t(_tableTypes.size()), 0};
        outputTableType.table.keys = {&_tableKeys[0], uint32_t(_tableKeys.size()), 0};
        return outputTableType;
      }
    }

    if (_defaultValue.valueType != SHType::None) {
      return _defaultType;
    } else {
      throw ComposeError("Get (" + _name + "): Could not infer an output type and no Default value provided.");
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (_defaultValue.valueType != SHType::None) {
      return {};
    } else {
      if (_isTable) {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The required table."), CoreInfo::AnyTableType));
        if (_key.isVariable()) {
          _exposedInfo.push_back(ExposedInfo::Variable(_key.variableName(), SHCCSTR("The required key."), CoreInfo::AnyType));
        }
      } else {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The required variable."), CoreInfo::AnyType));
      }
      return SHExposedTypesInfo(_exposedInfo);
    }
  }

  bool defaultTypeCheck(const SHVar &value) {
    if (value.valueType != _defaultValue.valueType) {
      return false;
    }

    if (value.valueType == SHType::Object && (value.payload.objectVendorId != _defaultValue.payload.objectVendorId ||
                                              value.payload.objectTypeId != _defaultValue.payload.objectTypeId))
      return false;

    if (value.valueType == SHType::Enum && (value.payload.enumVendorId != _defaultValue.payload.enumVendorId ||
                                            value.payload.enumTypeId != _defaultValue.payload.enumTypeId))
      return false;

    return true;
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceVariable(context, _name.c_str());

    _key.warmup(context);
  }

  void cleanup(SHContext *context) {
    // reset shard id
    if (_shard) {
      _shard->inlineShardId = InlineShard::NotInline;
    }
    VariableBase::cleanup(context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (_isTable) {
      checkIfTableChanged();
    }

    if (unlikely(_cell != nullptr)) {
      assert(_isTable);
      // This is used in the table case still
      return *_cell;
    } else {
      if (_isTable) {
        if (_target->valueType == SHType::Table) {
          auto &kv = _key.get();
          SHMap *table = static_cast<SHMap *>(_target->payload.tableValue.opaque);
          auto maybeValue = table->find(kv);
          if (maybeValue != table->end()) {
            auto &vRef = maybeValue->second;
            // Has it
            if (unlikely(_defaultValue.valueType != SHType::None && !defaultTypeCheck(vRef))) {
              return _defaultValue;
            } else {
              // Pin fast cell
              // skip if variable
              if (!_key.isVariable()) {
                _cell = &vRef;
              }
              return vRef;
            }
          } else {
            // No record
            if (_defaultType.basicType != SHType::None) {
              return _defaultValue;
            } else {
              throw ActivationError("Get - Key not found in table.");
            }
          }
        } else {
          if (_defaultType.basicType != SHType::None) {
            return _defaultValue;
          } else {
            throw ActivationError("Get - Table is empty or does not exist yet.");
          }
        }
      } else {
        auto &value = *_target;
        if (unlikely(_defaultValue.valueType != SHType::None && !defaultTypeCheck(value))) {
          return _defaultValue;
        } else {
          // Pin fast cell
          _cell = _target;
          // override shard internal id
          _shard->inlineShardId = InlineShard::CoreGet;
          return value;
        }
      }
    }
  }
};

struct Swap {
  static inline shards::ParamsInfo swapParamsInfo =
      shards::ParamsInfo(shards::ParamsInfo::Param("First", SHCCSTR("The name of first variable."), CoreInfo::AnyVarType),
                         shards::ParamsInfo::Param("Second", SHCCSTR("The name of second variable."), CoreInfo::AnyVarType));

  ParamVar _first;
  ParamVar _second;
  ExposedInfo _exposedInfo;

  void cleanup(SHContext *context) {
    _first.cleanup();
    _second.cleanup();
  }

  void warmup(SHContext *ctx) {
    _first.warmup(ctx);
    _second.warmup(ctx);
  }

  static SHOptionalString help() {
    return SHCCSTR("Swaps the values of the two variables passed to it via `First` and `Second` parameters.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input to this shard is passed through as its output."); }

  SHExposedTypesInfo requiredVariables() {
    _exposedInfo =
        ExposedInfo(ExposedInfo::Variable(_first.variableName(), SHCCSTR("The required variable."), CoreInfo::AnyType),
                    ExposedInfo::Variable(_second.variableName(), SHCCSTR("The required variable."), CoreInfo::AnyType));
    return SHExposedTypesInfo(_exposedInfo);
  }

  static SHParametersInfo parameters() { return SHParametersInfo(swapParamsInfo); }

  void setParam(int index, const SHVar &value) {
    if (index == 0)
      _first = value;
    else if (index == 1) {
      _second = value;
    } else {
      throw SHException("Param index out of range.");
    }
  }

  SHVar getParam(int index) {
    if (index == 0)
      return _first;
    else if (index == 1)
      return _second;
    throw SHException("Param index out of range.");
  }

  OwnedVar _cache;

  ALWAYS_INLINE const SHVar &activate(SHContext *context, const SHVar &input) {
    _cache = _first.get();
    cloneVar(_first.get(), _second.get());
    cloneVar(_second.get(), _cache);
    return input;
  }
};

struct SeqBase : public VariableBase {
  bool _clear = true;
  SHTypeInfo _tableInfo{};

  void initSeq() {
    if (_isTable) {
      SHMap *table = nullptr;
      if (_target->valueType != SHType::Table) {
        // Not initialized yet
        _target->valueType = SHType::Table;
        _target->payload.tableValue.api = &GetGlobals().TableInterface;
        table = _target->payload.tableValue.opaque = new SHMap();
      } else {
        table = static_cast<SHMap *>(_target->payload.tableValue.opaque);
      }

      if (!_key.isVariable()) {
        auto &kv = _key.get();
        auto fk = shards::OwnedVar::Foreign(kv);
        _cell = &(*table)[fk];

        auto &seq = *_cell;
        if (seq.valueType != SHType::Seq) {
          seq.valueType = SHType::Seq;
          seq.payload.seqValue = {};
        }
      } else {
        return; // we will check during activate
      }
    } else {
      if (_target->valueType != SHType::Seq) {
        _target->valueType = SHType::Seq;
        _target->payload.seqValue = {};
      }
      _cell = _target;
    }

    assert(_cell);
  }

  void fillVariableCell() {
    assert(_target->valueType == SHType::Table && "Expected a table");
    SHMap *table = static_cast<SHMap *>(_target->payload.tableValue.opaque);
    auto &kv = _key.get();
    auto fk = shards::OwnedVar::Foreign(kv);
    _cell = &(*table)[fk];

    auto &seq = *_cell;
    if (seq.valueType != SHType::Seq) {
      seq.valueType = SHType::Seq;
      seq.payload.seqValue = {};
    }
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceVariable(context, _name.c_str());
    _key.warmup(context);
    initSeq();
  }

  SHExposedTypesInfo exposedVariables() { return SHExposedTypesInfo(_exposedInfo); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  void destroy() {
    // for (size_t i = 0; i < _tableInfo.table.keys.len; i++) {
    //   shards::destroyVar(_tableInfo.table.keys.elements[i]);
    // }
    shards::arrayFree(_tableInfo.table.keys);
    shards::arrayFree(_tableInfo.table.types);
  }
};

struct Push : public SeqBase {
  bool _firstPush = false;
  SHTypeInfo _seqInfo{};

  void destroy() {
    SeqBase::destroy();
    shards::arrayFree(_seqInfo.seqTypes);
  }

  static SHOptionalString help() {
    return SHCCSTR("Pushes a new value into a sequence variable. If the variable does not exist, it will be created.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The value to push into the sequence."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The input to this shard is passed through as its output."); }

  static inline Parameters pushParams{setterParams,
                                      {{"Clear",
                                        SHCCSTR("Whether to clear this sequence at every wire iteration. This only works if it's "
                                                "the first push. The default is true."),
                                        {CoreInfo::BoolType}}}};

  static SHParametersInfo parameters() { return pushParams; }

  void setParam(int index, const SHVar &value) {
    if (index < variableParamsInfoLen)
      VariableBase::setParam(index, value);
    else if (index == variableParamsInfoLen + 0) {
      _clear = value.payload.boolValue;
    }
  }

  SHVar getParam(int index) {
    if (index < variableParamsInfoLen)
      return VariableBase::getParam(index);
    else if (index == variableParamsInfoLen + 0)
      return shards::Var(_clear);
    throw SHException("Param index out of range.");
  }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);

    // check if this type is already exposed
    auto type = findExposedVariablePtr(inherited->inherited, _name);
    auto global = _global || (type && type->global);

    const auto updateSeqInfo = [this, &data, global](const SHTypeInfo *existingSeqType = nullptr) {
      updateSeqType(_seqInfo, data.inputType, existingSeqType);

      if (global) {
        _exposedInfo = ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed sequence."), _seqInfo, true));
      } else {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed sequence."), _seqInfo, true));
      }
    };

    const auto updateTableInfo = [this, &data, global](bool firstPush, const SHTypeInfo *existingTableType = nullptr) {
      SHTypeInfo *existingSeqType{};
      if (existingTableType) {
        for (size_t i = 0; i < existingTableType->table.keys.len; i++) {
          if (existingTableType->table.keys.elements[i] == _key) {
            existingSeqType = &existingTableType->table.types.elements[i];
            break;
          }
        }
      }

      updateSeqType(_seqInfo, data.inputType, existingSeqType);
      updateTableType(_tableInfo, !_key.isVariable() ? &(SHVar &)_key : nullptr, _seqInfo, existingTableType);

      if (global) {
        _exposedInfo = ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed table."), _tableInfo, true));
      } else {
        _exposedInfo =
            ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed table."), SHTypeInfo(_tableInfo), true));
      }
    };

    if (_isTable) {
      if (type) {
        if (type->exposedType.basicType != SHType::Table) {
          throw ComposeError("Expected a table variable.");
        }

        if (type->tracked) {
          // cannot push into exposed variables
          throw ComposeError("Cannot push into exposed variables");
        }

        if (type->exposedType.table.types.elements) {
          auto &tableKeys = type->exposedType.table.keys;
          auto &tableTypes = type->exposedType.table.types;
          for (uint32_t y = 0; y < tableKeys.len; y++) {
            // if we got key it's not a variable
            if (_key == tableKeys.elements[y] && tableTypes.elements[y].basicType == SHType::Seq) {
              updateTableInfo(false, &type->exposedType);
              return data.inputType; // found lets escape
            }
          }
        }
      } else {
        // not found
        // implicitly initialize anyway
        updateTableInfo(true);
        _firstPush = true;
      }
    } else {
      if (type) {
        if (type->exposedType.basicType != SHType::Seq)
          throw ComposeError(fmt::format("Push: error, variable {} is not a sequence.", _name));
        // found, can we mutate it?
        if (!type->isMutable) {
          throw ComposeError(fmt::format("Cannot mutate a non-mutable variable: {}", _name));
        } else if (type->isProtected) {
          throw ComposeError(fmt::format("Cannot mutate a protected variable: {}", _name));
        }

        if (type->tracked) {
          // cannot push into exposed variables
          throw ComposeError(fmt::format("Cannot push into exposed variables: {}", _name));
        }

        // ok now update into
        updateSeqInfo(&type->exposedType);
        return data.inputType; // found lets escape
      } else {
        // not found
        // implicitly initialize anyway
        updateSeqInfo();
        _firstPush = true;
      }
    }

    if (_firstPush) {
      _exposedInfo._innerInfo.elements[0].declared = true;
    }

    return data.inputType;
  }

  ALWAYS_INLINE const SHVar &activate(SHContext *context, const SHVar &input) noexcept {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_clear && _firstPush) {
      shards::arrayResize(_cell->payload.seqValue, 0);
    }
    const auto len = _cell->payload.seqValue.len;
    shards::arrayResize(_cell->payload.seqValue, len + 1);
    cloneVar(_cell->payload.seqValue.elements[len], input);

    return input;
  }
};

struct Sequence : public SeqBase {
  OwnedVar _typeDesc{};
  SHTypeInfo _weakType{};

  static SHOptionalString help() {
    return SHCCSTR("Creates an empty sequence (or sequence in a table if a key is passed). Useful to declare and specify types.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The input to this shard is passed through as its output."); }

  static inline Parameters pushParams{
      setterParams,
      {{"Clear",
        SHCCSTR("If we should clear this sequence at every wire iteration; works only if this is the first push; default: true."),
        {CoreInfo::BoolType}},
       {"Type", SHCCSTR("The sequence type to forward declare."), {CoreInfo::NoneType, CoreInfo::TypeType}}}};

  static SHParametersInfo parameters() { return pushParams; }

  void setParam(int index, const SHVar &value) {
    if (index < variableParamsInfoLen)
      VariableBase::setParam(index, value);
    else if (index == variableParamsInfoLen + 0) {
      _clear = value.payload.boolValue;
    } else if (index == variableParamsInfoLen + 1) {
      _typeDesc = value;
    }
  }

  SHVar getParam(int index) {
    if (index < variableParamsInfoLen)
      return VariableBase::getParam(index);
    else if (index == variableParamsInfoLen + 0)
      return shards::Var(_clear);
    else if (index == variableParamsInfoLen + 1)
      return _typeDesc;
    throw SHException("Param index out of range.");
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    const auto updateTableInfo = [this] {
      _tableInfo.basicType = SHType::Table;
      if (_tableInfo.table.types.elements) {
        shards::arrayFree(_tableInfo.table.types);
      }
      if (_tableInfo.table.keys.elements) {
        shards::arrayFree(_tableInfo.table.keys);
      }
      shards::arrayPush(_tableInfo.table.types, _weakType);
      if (!_key.isVariable()) {
        shards::arrayPush(_tableInfo.table.keys, *_key);
      }
      if (_global) {
        _exposedInfo =
            ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed table."), SHTypeInfo(_tableInfo), true));
      } else {
        _exposedInfo =
            ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed table."), SHTypeInfo(_tableInfo), true));
      }
    };

    // Ensure variable did not exist
    if (!_isTable) {
      for (uint32_t i = 0; i < data.shared.len; i++) {
        auto &reference = data.shared.elements[i];
        if (strcmp(reference.name, _name.c_str()) == 0) {
          throw ComposeError("Sequence - Variable " + _name + " already exists.");
        }
      }
    } else {
      for (uint32_t i = 0; data.shared.len > i; i++) {
        if (data.shared.elements[i].name == _name && data.shared.elements[i].exposedType.table.types.elements) {
          auto &tableKeys = data.shared.elements[i].exposedType.table.keys;
          for (uint32_t y = 0; y < tableKeys.len; y++) {
            // if here, key is not variable
            if (*_key == tableKeys.elements[y]) {
              auto msg = fmt::format("Sequence - Variable {} in table {} already exists.", *_key, _name);
              throw ComposeError(msg);
            }
          }
        }
      }
    }

    // Process type to expose
    if (_typeDesc.valueType == SHType::None) {
      _weakType = CoreInfo::AnySeqType;
    } else {
      assert(_typeDesc.valueType == SHType::Type);
      auto typeDesc = _typeDesc.payload.typeValue;
      _weakType = *typeDesc;
    }

    if (!_isTable) {
      if (_global) {
        _exposedInfo = ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed sequence."), _weakType, true));
      } else {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed sequence."), _weakType, true));
      }
    } else {
      updateTableInfo();
    }

    if (_weakType.basicType != SHType::Seq) {
      throw ComposeError("Sequence - Type must be a sequence.");
    }

    // Ensure declared
    _exposedInfo._innerInfo.elements[0].declared = true;

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_clear) {
      auto &seq = *_cell;
      shards::arrayResize(seq.payload.seqValue, 0);
    }

    return input;
  }
};

struct TableDecl : public VariableBase {
  SHTypeInfo _tableInfo{};

  void initTable() {
    if (_isTable) {
      if (_target->valueType != SHType::Table) {
        // Not initialized yet
        _target->valueType = SHType::Table;
        _target->payload.tableValue.api = &GetGlobals().TableInterface;
        _target->payload.tableValue.opaque = new SHMap();
      }

      if (!_key.isVariable()) {
        auto &kv = _key.get();
        _cell = _target->payload.tableValue.api->tableAt(_target->payload.tableValue, kv);

        auto table = _cell;
        if (table->valueType != SHType::Table) {
          // Not initialized yet
          table->valueType = SHType::Table;
          table->payload.tableValue.api = &GetGlobals().TableInterface;
          table->payload.tableValue.opaque = new SHMap();
        }
      } else {
        return; // we will check during activate
      }
    } else {
      if (_target->valueType != SHType::Table) {
        _target->valueType = SHType::Table;
        _target->payload.tableValue.api = &GetGlobals().TableInterface;
        _target->payload.tableValue.opaque = new SHMap();
      }
      _cell = _target;
    }

    assert(_cell);
  }

  void fillTableCell() {
    auto &kv = _key.get();
    _cell = _target->payload.tableValue.api->tableAt(_target->payload.tableValue, kv);

    auto table = _cell;
    if (table->valueType != SHType::Table) {
      // Not initialized yet
      table->valueType = SHType::Table;
      table->payload.tableValue.api = &GetGlobals().TableInterface;
      table->payload.tableValue.opaque = new SHMap();
    }
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceVariable(context, _name.c_str());
    _key.warmup(context);
    initTable();
  }

  SHExposedTypesInfo exposedVariables() { return SHExposedTypesInfo(_exposedInfo); }

  static SHOptionalString help() { return SHCCSTR("Creates an empty table. Useful to declare and specify types."); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input to this shard is passed through as its output."); }

  void destroy() {
    shards::arrayFree(_tableInfo.table.keys);
    shards::arrayFree(_tableInfo.table.types);
  }

  OwnedVar _typeDesc{};
  SHTypeInfo _weakType{};

  static inline Parameters tableParams{
      setterParams, {{"Type", SHCCSTR("The table type to forward declare."), {CoreInfo::NoneType, CoreInfo::TypeType}}}};

  static SHParametersInfo parameters() { return tableParams; }

  void setParam(int index, const SHVar &value) {
    if (index < variableParamsInfoLen)
      VariableBase::setParam(index, value);
    else if (index == variableParamsInfoLen + 0) {
      _typeDesc = value;
    }
  }

  SHVar getParam(int index) {
    if (index < variableParamsInfoLen)
      return VariableBase::getParam(index);
    else if (index == variableParamsInfoLen + 0)
      return _typeDesc;
    throw SHException("Param index out of range.");
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    const auto updateTableInfo = [this] {
      _tableInfo.basicType = SHType::Table;
      shards::arrayFree(_tableInfo.table.types);
      shards::arrayFree(_tableInfo.table.keys);
      shards::arrayPush(_tableInfo.table.types, _weakType);
      if (!_key.isVariable()) {
        shards::arrayPush(_tableInfo.table.keys, *_key);
      }
      if (_global) {
        _exposedInfo =
            ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed table."), SHTypeInfo(_tableInfo), true));
      } else {
        _exposedInfo =
            ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed table."), SHTypeInfo(_tableInfo), true));
      }
    };

    // Ensure variable did not exist
    if (!_isTable) {
      for (uint32_t i = 0; i < data.shared.len; i++) {
        auto &reference = data.shared.elements[i];
        if (strcmp(reference.name, _name.c_str()) == 0) {
          throw ComposeError("Table - Variable " + _name + " already exists.");
        }
      }
    } else {
      for (uint32_t i = 0; data.shared.len > i; i++) {
        auto &reference = data.shared.elements[i];
        if (strcmp(reference.name, _name.c_str()) == 0 && data.shared.elements[i].exposedType.table.types.elements) {
          auto &tableKeys = data.shared.elements[i].exposedType.table.keys;
          for (uint32_t y = 0; y < tableKeys.len; y++) {
            // if here, key is not variable
            if (*_key == tableKeys.elements[y]) {
              auto msg = fmt::format("Table - Variable {} in table {} already exists.", *_key, _name);
              throw ComposeError(msg);
            }
          }
        }
      }
    }

    // Process type to expose
    if (_typeDesc.valueType == SHType::None) {
      _weakType = CoreInfo::AnyTableType;
    } else {
      assert(_typeDesc.valueType == SHType::Type);
      auto typeDesc = _typeDesc.payload.typeValue;
      _weakType = *typeDesc;
    }

    if (!_isTable) {
      if (_global) {
        _exposedInfo = ExposedInfo(ExposedInfo::GlobalVariable(_name.c_str(), SHCCSTR("The exposed table."), _weakType, true));
      } else {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The exposed table."), _weakType, true));
      }
    } else {
      updateTableInfo();
    }

    if (_weakType.basicType != SHType::Table) {
      throw ComposeError("Table - Type must be a table.");
    }

    // Ensure declared
    _exposedInfo._innerInfo.elements[0].declared = true;

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillTableCell();
      }
    }

    return input;
  }
};

struct SeqUser : VariableBase {
  bool _blittable = false;

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  void initSeq() {
    if (_isTable) {
      if (_target->valueType != SHType::Table) {
        // We need to init this in order to fetch cell addr
        // Not initialized yet
        _target->valueType = SHType::Table;
        _target->payload.tableValue.api = &GetGlobals().TableInterface;
        _target->payload.tableValue.opaque = new SHMap();
      }

      if (!_key.isVariable()) {
        _cell = _target->payload.tableValue.api->tableAt(_target->payload.tableValue, *_key);
      } else {
        return; // checked during activate
      }
    } else {
      _cell = _target;
    }

    assert(_cell);
  }

  void fillVariableCell() {
    auto &kv = _key.get();
    _cell = _target->payload.tableValue.api->tableAt(_target->payload.tableValue, kv);
  }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
    auto info = findExposedVariablePtr(inherited->inherited, _name);

    if (!info) {
      throw ComposeError(fmt::format("Variable {} not found.", _name));
    }

    if (info->tracked) {
      throw ComposeError(fmt::format("Variable {} is tracked and can only be updated using Update.", _name));
    }

    return data.inputType;
  }

  void warmup(SHContext *context) {
    if (_global)
      _target = referenceGlobalVariable(context, _name.c_str());
    else
      _target = referenceVariable(context, _name.c_str());
    _key.warmup(context);
    initSeq();
  }

  SHExposedTypesInfo requiredVariables() {
    if (_name.size() > 0) {
      if (_isTable) {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The required table."), CoreInfo::AnyTableType));
      } else {
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_name.c_str(), SHCCSTR("The required variable."), CoreInfo::AnyType));
      }
      return SHExposedTypesInfo(_exposedInfo);
    } else {
      return {};
    }
  }
};

struct Count : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard counts the sequence, string or table variable specified in the Name parameter. If the variable specified is "
        "a string, it will count the number of characters. If the variable specified is a sequence, it will count the number of "
        "elements. If the variable specified is a table, it will count the number of key-value pairs.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the count of characters, elements, or key-value pairs in the specified variable. "
                   "If the variable type does not match, it outputs 0.");
  }

  SHTypeInfo compose(const SHInstanceData &data) { return CoreInfo::IntType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    switch (_cell->valueType) {
    case SHType::Seq:
      return shards::Var(int64_t(_cell->payload.seqValue.len));
    case SHType::Table:
      return shards::Var(int64_t(_cell->payload.tableValue.api->tableSize(_cell->payload.tableValue)));
    case SHType::Bytes:
      return shards::Var(int64_t(_cell->payload.bytesSize));
    case SHType::String:
      return shards::Var(int64_t(_cell->payload.stringLen > 0 || _cell->payload.stringValue == nullptr
                                     ? _cell->payload.stringLen
                                     : strlen(_cell->payload.stringValue)));
    default:
      return shards::Var(0);
    }
  }
};

struct Clear : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR("Clears all elements from the sequence or table passed to it. Applicable only to sequences and tables. For "
                   "sequences, this operation is very fast as Shards recycles memory extensively. If the variable does not exist "
                   "or the type is not a sequence or table, it simply passes through without failing.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input is passed through as the output."); }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    SeqUser::composeV2(data);

    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
    auto info = findExposedVariablePtr(inherited->inherited, _name);

    // info is valid because we run base compose first

    if (!info->isMutable) {
      throw ComposeError(fmt::format("Variable {} is not mutable.", _name));
    }

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_cell->valueType == SHType::Seq) {
      // Resize the sequence to 0, clearing it
      shards::arrayResize(_cell->payload.seqValue, 0);

      // If the input is the same sequence, adjust its length
      if (input.payload.seqValue.elements == _cell->payload.seqValue.elements) {
        const_cast<SHVar &>(input).payload.seqValue.len = 0;
      }
    } else if (_cell->valueType == SHType::Table) {
      // Clear all key-value pairs in the table
      _cell->payload.tableValue.api->tableClear(_cell->payload.tableValue);
      _cell->version++; // and bump version
    }

    return input;
  }
};

struct Shuffle : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR("Shuffles the elements of the sequence variable. Works only on sequences. If the variable is not a sequence, "
                   "it simply passes through without failing.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input is passed through as the output."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_cell->valueType == SHType::Seq) {
      shards::arrayShuffle(_cell->payload.seqValue);
    }

    return input;
  }
};

struct Drop : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR("Drops the last element of the sequence variable. Works only on sequences. If the variable is not a sequence, "
                   "it simply passes through without failing.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input is passed through as the output."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_cell->valueType == SHType::Seq) {
      auto len = _cell->payload.seqValue.len;
      if (len > 0) {
        // Resize the sequence to drop the last element
        shards::arrayResize(_cell->payload.seqValue, len - 1);

        // If the input is the same sequence, adjust its length
        if (input.payload.seqValue.elements == _cell->payload.seqValue.elements) {
          const_cast<SHVar &>(input).payload.seqValue.len = len - 1;
        }
      }
    }

    return input;
  }
};

struct DropFront : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR("Drops the first element of the sequence variable. Works only on sequences. If the variable is not a "
                   "sequence, it simply passes through without failing.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The input is passed through as the output."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_cell->valueType == SHType::Seq && _cell->payload.seqValue.len > 0) {
      auto &arr = _cell->payload.seqValue;
      shards::arrayDel(arr, 0);

      // If the input is the same sequence, adjust its length
      if (input.payload.seqValue.elements == arr.elements) {
        const_cast<SHVar &>(input).payload.seqValue.len = arr.len;
      }
    }

    return input;
  }
};

struct Pop : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR("Pops (removes and outputs) the last element of the sequence variable. Works only on sequences. If the "
                   "variable is not a sequence or the sequence is empty, an error is thrown.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The element popped from the sequence."); }

  SHVar _output{};

  void destroy() { destroyVar(_output); }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    SeqUser::composeV2(data);

    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
    auto info = findExposedVariablePtr(inherited->inherited, _name);

    // info is valid because we run base compose first

    if (_isTable) {
      auto &tableKeys = info->exposedType.table.keys;
      auto &tableTypes = info->exposedType.table.types;
      for (uint32_t y = 0; y < tableKeys.len; y++) {
        // if here _key is not variable
        if (*_key == tableKeys.elements[y] && tableTypes.elements[y].basicType == SHType::Seq) {
          // if we have 1 type we can predict the output
          // with more just make us a any seq, will need ExpectX shards
          // likely
          if (tableTypes.elements[y].seqTypes.len == 1)
            return tableTypes.elements[y].seqTypes.elements[0];
          else
            return CoreInfo::AnySeqType;
        }
      }
      throw SHException("Pop: key not found or key value is not a sequence!.");
    } else {
      if (info->exposedType.basicType == SHType::Seq) {
        // if we have 1 type we can predict the output
        // with more just make us a any seq, will need ExpectX shards likely
        if (info->exposedType.seqTypes.len == 1)
          return info->exposedType.seqTypes.elements[0];
        else
          return CoreInfo::AnySeqType;
      }
    }
    throw SHException("Variable is not a sequence.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_cell->valueType != SHType::Seq) {
      throw ActivationError("Variable is not a sequence, failed to Pop.");
    }

    if (_cell->payload.seqValue.len == 0) {
      throw ActivationError("Pop: sequence was empty.");
    }

    // Clone
    auto pops = shards::arrayPop<SHSeq, SHVar>(_cell->payload.seqValue);
    cloneVar(_output, pops);
    return _output;
  }
};

struct PopFront : SeqUser {
  static SHOptionalString help() {
    return SHCCSTR("Pops (removes and outputs) the first element of the sequence variable. Works only on sequences. If the "
                   "variable is not a sequence or the sequence is empty, an error is thrown.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString inputHelp() { return SHCCSTR("Any input is ignored."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The element popped from the sequence."); }

  SHVar _output{};

  void destroy() { destroyVar(_output); }

  SHTypeInfo composeV2(const SHInstanceData &data) {
    SeqUser::composeV2(data);

    shassert(data.privateContext && "Private context should be valid");
    auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
    auto info = findExposedVariablePtr(inherited->inherited, _name);

    // info is valid because we run base compose first

    if (_isTable) {
      auto &tableKeys = info->exposedType.table.keys;
      auto &tableTypes = info->exposedType.table.types;
      for (uint32_t y = 0; y < tableKeys.len; y++) {
        // if here _key is not variable
        if (*_key == tableKeys.elements[y] && tableTypes.elements[y].basicType == SHType::Seq) {
          // if we have 1 type we can predict the output
          // with more just make us a any seq, will need ExpectX shards
          // likely
          if (tableTypes.elements[y].seqTypes.len == 1)
            return tableTypes.elements[y].seqTypes.elements[0];
          else
            return CoreInfo::AnySeqType;
        }
      }
      throw SHException("Pop: key not found or key value is not a sequence!.");
    } else {
      if (info->exposedType.basicType == SHType::Seq) {
        // if we have 1 type we can predict the output
        // with more just make us a any seq, will need ExpectX shards likely
        if (info->exposedType.seqTypes.len == 1)
          return info->exposedType.seqTypes.elements[0];
        else
          return CoreInfo::AnySeqType;
      }
    }
    throw SHException("Variable is not a sequence.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (unlikely(_isTable)) {
      checkIfTableChanged();
      if (unlikely(_cell == nullptr)) {
        fillVariableCell();
      }
    }

    if (_cell->valueType != SHType::Seq) {
      throw ActivationError("Variable is not a sequence, failed to Pop.");
    }

    if (_cell->payload.seqValue.len == 0) {
      throw ActivationError("Pop: sequence was empty.");
    }

    auto &arr = _cell->payload.seqValue;
    const auto len = arr.len - 1;
    // store to put back at end
    // we do this to allow further grows
    // to recycle this _cell (well if not blittable that is)
    auto first = arr.elements[0];
    static_assert(sizeof(*arr.elements) == sizeof(SHVar), "Wrong seq elements size!");
    // shift backward current elements
    memmove(&arr.elements[0], &arr.elements[1], sizeof(*arr.elements) * len);
    // put first at end
    arr.elements[len] = first;
    // resize, will cut first out too
    shards::arrayResize(arr, len);

    cloneVar(_output, first);
    return _output;
  }
};

struct Take {
  static inline ParamsInfo indicesParamsInfo = ParamsInfo(ParamsInfo::Param(
      "Indices/Keys", SHCCSTR("One or more indices or keys to extract from a sequence or table."), CoreInfo::TakeTypes));

  static SHOptionalString help() {
    return SHCCSTR("Extracts one or more elements from a sequence or values from a table using the provided indices or keys. "
                   "This operation is non-destructive and does not modify the target sequence or table. If the key cannot be "
                   "established to exist at compose time, the output will be of type Any.");
  }

  SHSeq _cachedSeq{};
  SHVar _output{};
  SHVar _indices{};
  SHVar *_indicesVar{nullptr};
  ExposedInfo _exposedInfo{};
  bool _seqOutput{false};
  bool _tableOutput{false};

  SHVar _vectorOutput{};
  const VectorTypeTraits *_vectorInputType{nullptr};
  const VectorTypeTraits *_vectorOutputType{nullptr};
  const NumberConversion *_vectorConversion{nullptr};

  Type _seqOutputType{};
  std::vector<SHTypeInfo> _seqOutputTypes;

  void destroy() {
    destroyVar(_indices);
    destroyVar(_output);
  }

  void cleanup(SHContext *context) {
    if (_indicesVar) {
      releaseVariable(_indicesVar);
      _indicesVar = nullptr;
    }
    if (_cachedSeq.elements) {
      shards::arrayFree(_cachedSeq);
    }
  }

  static SHTypesInfo inputTypes() { return CoreInfo::Indexables; }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The sequence or table from which elements or values will be extracted.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The extracted elements from a sequence or values from a table. If the key cannot be established to exist at "
                   "compose time, the output will be of type Any.");
  }

  static SHParametersInfo parameters() { return SHParametersInfo(indicesParamsInfo); }

  SHTypeInfo compose(const SHInstanceData &data) {
    bool valid = false;
    bool isTable = data.inputType.basicType == SHType::Table;
    // Figure if we output a sequence or not
    if (_indices.valueType == SHType::Seq) {
      if (isTable) {
        // allow any key type
        valid = true;
        // just differentiate if we have multiple, a seq, of keys vs one key
        _seqOutput = true;
      } else {
        // allow only integers
        bool onlyInts = true; // empty is fine to be true
        for (auto &index : _indices) {
          if (index.valueType != SHType::Int) {
            onlyInts = false;
            break;
          }
        }

        if (onlyInts) {
          valid = true;
          _seqOutput = true;
        } else {
          throw ComposeError(
              fmt::format("Take with input as sequence expected an integer or a sequence of integers, but got: {}", _indices));
        }
      }
    } else if (!isTable && _indices.valueType == SHType::Int) {
      _seqOutput = false;
      valid = true;
    } else if (_indices.valueType == SHType::ContextVar) { // SHType::ContextVar
      for (auto &info : data.shared) {
        if (info.name == SHSTRVIEW(_indices)) {
          if (isTable) {
            // allow any key type
            valid = true;
            // just differentiate if we have multiple, a seq, of keys vs one key
            _seqOutput = info.exposedType.basicType == SHType::Seq;
          } else {
            // allow only integers
            if (info.exposedType.basicType == SHType::Seq && info.exposedType.seqTypes.len == 1 &&
                info.exposedType.seqTypes.elements[0].basicType == SHType::Int) {
              _seqOutput = true;
              valid = true;
              break;
            } else if (info.exposedType.basicType == SHType::Int) {
              _seqOutput = false;
              valid = true;
              break;
            } else {
              throw ComposeError(fmt::format(
                  "Take with input as sequence expected an integer or a sequence of integers, but got: {}", info.exposedType));
            }
          }
        }
      }
    } else if (isTable) {
      valid = true;
      _seqOutput = false;
    } else {
      throw ComposeError(fmt::format("Take: Expected indices to be either Seq, Int or String, but got: {}, with input: {}",
                                     _indices, data.inputType));
    }

    _tableOutput = isTable;

    if (!valid)
      throw SHException(
          fmt::format("Take, invalid indices or malformed input. input: {}, indices: {}", data.inputType, _indices));

    if (data.inputType.basicType == SHType::Seq) {
      OVERRIDE_ACTIVATE(data, activateSeq);
      if (_seqOutput) {
        // multiple values, leave SHType::Seq
        return data.inputType;
      } else if (data.inputType.seqTypes.len == 1) {
        // single unique seq type
        return data.inputType.seqTypes.elements[0];
      } else {
        // value from seq but not unique
        return CoreInfo::AnyType;
      }
    } else {
      _vectorInputType = VectorTypeLookup::getInstance().get(data.inputType.basicType);
      if (_vectorInputType) {
        if (_seqOutput)
          OVERRIDE_ACTIVATE(data, activateVector);
        else
          OVERRIDE_ACTIVATE(data, activateNumber);

        uint8_t indexTableLength = _seqOutput ? (uint8_t)_indices.payload.seqValue.len : 1;
        _vectorOutputType = VectorTypeLookup::getInstance().findCompatibleType(_vectorInputType->isInteger, indexTableLength);
        if (!_vectorOutputType) {
          std::stringstream errorMsgStream;
          errorMsgStream << "Take: No " << (_vectorInputType->isInteger ? "integer" : "float") << " vector type exists that fits "
                         << indexTableLength << "elements";
          std::string errorMsg = errorMsgStream.str();
          SHLOG_ERROR(errorMsg.c_str());
          throw ComposeError(errorMsg);
        }

        _vectorConversion =
            NumberTypeLookup::getInstance().getConversion(_vectorInputType->numberType, _vectorOutputType->numberType);
        if (!_vectorConversion) {
          std::stringstream errorMsgStream;
          errorMsgStream << "Take: No conversion from " << _vectorInputType->name << " to " << _vectorOutputType->name
                         << " exists";
          std::string errorMsg = errorMsgStream.str();
          SHLOG_ERROR(errorMsg.c_str());
          throw ComposeError(errorMsg);
        }

        _vectorOutput.valueType = _vectorOutputType->shType;
        return _vectorOutputType->type;
      } else if (data.inputType.basicType == SHType::Bytes) {
        OVERRIDE_ACTIVATE(data, activateBytes);
        if (_seqOutput) {
          return CoreInfo::IntSeqType;
        } else {
          return CoreInfo::IntType;
        }
      } else if (data.inputType.basicType == SHType::String) {
        OVERRIDE_ACTIVATE(data, activateString);
        if (_seqOutput) {
          return CoreInfo::IntSeqType;
        } else {
          return CoreInfo::IntType;
        }
      } else if (data.inputType.basicType == SHType::Table) {
        OVERRIDE_ACTIVATE(data, activateTable);
        if (data.inputType.table.keys.len > 0 && _indices.valueType != SHType::ContextVar) {
          // we can fully reconstruct a type in this case
          if (data.inputType.table.keys.len != data.inputType.table.types.len) {
            SHLOG_ERROR("Table input type: {}", data.inputType);
            throw ComposeError("Take: Expected same number of types for numer of "
                               "keys in table input.");
          }

          if (_seqOutput) {
            _seqOutputTypes.clear();
            for (uint32_t j = 0; j < _indices.payload.seqValue.len; j++) {
              auto &record = _indices.payload.seqValue.elements[j];
              for (uint32_t i = 0; i < data.inputType.table.keys.len; i++) {
                if (record == data.inputType.table.keys.elements[i]) {
                  _seqOutputTypes.emplace_back(data.inputType.table.types.elements[i]);
                }
              }
            }
            // if types is 0 we did not match any
            if (_seqOutputTypes.size() == 0) {
              SHLOG_ERROR("Table input type: {} missing keys: {}", data.inputType, _indices);
              throw ComposeError("Take: Failed to find a matching keys in the "
                                 "input type table");
            }
            _seqOutputType = Type::SeqOf(SHTypesInfo{_seqOutputTypes.data(), uint32_t(_seqOutputTypes.size()), 0});
            return _seqOutputType;
          } else {
            for (uint32_t i = 0; i < data.inputType.table.keys.len; i++) {
              if (_indices == data.inputType.table.keys.elements[i]) {
                return data.inputType.table.types.elements[i];
              }
            }
            // we didn't match any key...
            // and so we just return any type to allow extra keys
            return CoreInfo::AnyType;
          }
        } else {
          if (_seqOutput) {
            // multiple values, leave SHType::Seq
            return CoreInfo::AnySeqType;
          } else {
            // value from table, can be None
            return CoreInfo::AnyType;
          }
        }
      }
    }

    throw SHException("Take, invalid input type or not implemented.");
  }

  SHExposedTypesInfo requiredVariables() {
    if (_indices.valueType == SHType::ContextVar) {
      // the following stringValue are cloned so null terminated safe
      if (_seqOutput)
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_indices.payload.stringValue, SHCCSTR("The required variables."),
                                                         _tableOutput ? CoreInfo::AnySeqType : CoreInfo::IntSeqType));
      else
        _exposedInfo = ExposedInfo(ExposedInfo::Variable(_indices.payload.stringValue, SHCCSTR("The required variable."),
                                                         _tableOutput ? CoreInfo::AnyType : CoreInfo::IntType));
      return SHExposedTypesInfo(_exposedInfo);
    } else {
      return {};
    }
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      cloneVar(_indices, value);
      cleanup(nullptr);
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _indices;
    default:
      break;
    }
    return shards::Var::Empty;
  }

  struct OutOfRangeEx : public ActivationError {
    OutOfRangeEx(int64_t len, int64_t index)
        : ActivationError(fmt::format("Out of range! len: {} wanted index: {}", len, index)) {}
  };

  void warmup(SHContext *context) {
    if (_indices.valueType == SHType::ContextVar && !_indicesVar) {
      _indicesVar = referenceVariable(context, SHSTRVIEW(_indices));
    }
  }

#define ACTIVATE_INDEXABLE(__name__, __len__, __val__)                                                                     \
  ALWAYS_INLINE SHVar __name__(SHContext *context, const SHVar &input) {                                                   \
    shassert_extended(context, input.valueType == SHType::Seq || input.valueType == SHType::String ||                      \
                                   input.valueType == SHType::Bytes && "Take: Expected seq, string or bytes input type."); \
    const auto inputLen = size_t(__len__);                                                                                 \
    const auto &indices = _indicesVar ? *_indicesVar : _indices;                                                           \
    if (likely(!_seqOutput)) {                                                                                             \
      const auto index = indices.payload.intValue;                                                                         \
      if (index < 0 || size_t(index) >= inputLen) {                                                                        \
        throw OutOfRangeEx(inputLen, index);                                                                               \
      }                                                                                                                    \
      return __val__;                                                                                                      \
    } else {                                                                                                               \
      const uint32_t nindices = indices.payload.seqValue.len;                                                              \
      shards::arrayResize(_cachedSeq, nindices);                                                                           \
      for (uint32_t i = 0; nindices > i; i++) {                                                                            \
        const auto index = indices.payload.seqValue.elements[i].payload.intValue;                                          \
        if (index < 0 || size_t(index) >= inputLen) {                                                                      \
          throw OutOfRangeEx(inputLen, index);                                                                             \
        }                                                                                                                  \
        _cachedSeq.elements[i] = __val__;                                                                                  \
      }                                                                                                                    \
      return shards::Var(_cachedSeq);                                                                                      \
    }                                                                                                                      \
  }

  ACTIVATE_INDEXABLE(activateSeq, input.payload.seqValue.len, input.payload.seqValue.elements[index])
  ACTIVATE_INDEXABLE(activateString, SHSTRLEN(input), shards::Var(input.payload.stringValue[index]))
  ACTIVATE_INDEXABLE(activateBytes, input.payload.bytesSize, shards::Var(input.payload.bytesValue[index]))

  SHVar activateTable(SHContext *context, const SHVar &input) {
    shassert_extended(context, input.valueType == SHType::Table && "Take: Expected table input type.");

    const auto &indices = _indicesVar ? *_indicesVar : _indices;
    if (!_seqOutput) {
      SHMap *table = static_cast<SHMap *>(input.payload.tableValue.opaque);
      const auto key = indices;
      auto fk = shards::OwnedVar::Foreign(key);
      const auto val = table->find(fk);
      if (val == table->end()) {
        return Var::Empty;
      } else {
        return val->second;
      }
    } else {
      const uint32_t nkeys = indices.payload.seqValue.len;
      shards::arrayResize(_cachedSeq, nkeys);
      for (uint32_t i = 0; nkeys > i; i++) {
        const auto key = indices.payload.seqValue.elements[i];
        const auto val = input.payload.tableValue.api->tableGet(input.payload.tableValue, key);
        if (!val) {
          _cachedSeq.elements[i] = Var::Empty;
        } else {
          _cachedSeq.elements[i] = *val;
        }
      }
      return Var(_cachedSeq);
    }
  }

  SHVar activateVector(SHContext *context, const SHVar &input) {
    const auto &indices = _indices;

    try {
      _vectorConversion->convertMultipleSeq(&input.payload, &_vectorOutput.payload, _vectorInputType->dimension,
                                            indices.payload.seqValue);
    } catch (const NumberConversionOutOfRangeEx &ex) {
      throw OutOfRangeEx(_vectorInputType->dimension, ex.index);
    }

    return _vectorOutput;
  }

  SHVar activateNumber(SHContext *context, const SHVar &input) {
    SHInt index = _indices.payload.intValue;
    if (index < 0 || index >= (SHInt)_vectorInputType->dimension) {
      throw OutOfRangeEx(_vectorInputType->dimension, index);
    }

    const uint8_t *inputPtr = (uint8_t *)&input.payload + _vectorConversion->inStride * index;
    _vectorConversion->convertOne(inputPtr, &_vectorOutput.payload);

    return _vectorOutput;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    // Take branches during validation into different inlined shards
    // If we hit this, maybe that type of input is not yet implemented
    throw ActivationError("Take path not implemented for this type.");
  }
};

struct RTake : public Take {
  // Works only for sequences
  static inline ParamsInfo indicesParamsInfo = ParamsInfo(ParamsInfo::Param(
      "Indices", SHCCSTR("One or more indices (counted backwards from the last element) to extract from a sequence."),
      CoreInfo::RTakeTypes));

  static SHOptionalString help() {
    return SHCCSTR("Works exactly like `Take` except that the selection indices are counted backwards from the last element in "
                   "the target sequence. Also, `RTake` works only on sequences, not on tables.");
  }

  static SHParametersInfo parameters() { return SHParametersInfo(indicesParamsInfo); }

  static SHTypesInfo inputTypes() { return CoreInfo::RIndexables; }

  static SHOptionalString inputHelp() { return SHCCSTR("The sequence from which elements will be extracted."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHOptionalString outputHelp() { return SHCCSTR("The extracted elements."); }

  SHTypeInfo compose(const SHInstanceData &data) {
    SHTypeInfo result = Take::compose(data);
    if (data.inputType.basicType == SHType::Seq) {
      OVERRIDE_ACTIVATE(data, activate);
    } else {
      throw SHException("RTake is only supported on sequence types");
    }
    return result;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    const auto inputLen = input.payload.seqValue.len;
    const auto &indices = _indicesVar ? *_indicesVar : _indices;

    if (!_seqOutput) {
      const auto index = indices.payload.intValue;
      if (index >= inputLen || index < 0) {
        throw OutOfRangeEx(inputLen, index);
      }
      return input.payload.seqValue.elements[inputLen - 1 - index];
    } else {
      const uint32_t nindices = indices.payload.seqValue.len;
      shards::arrayResize(_cachedSeq, nindices);
      for (uint32_t i = 0; nindices > i; i++) {
        const auto index = indices.payload.seqValue.elements[i].payload.intValue;
        if (index >= inputLen || index < 0) {
          throw OutOfRangeEx(inputLen, index);
        }
        _cachedSeq.elements[i] = input.payload.seqValue.elements[inputLen - 1 - index];
      }
      return shards::Var(_cachedSeq);
    }
  }
};

struct Slice {
  static inline ParamsInfo indicesParamsInfo = ParamsInfo(
      ParamsInfo::Param("From",
                        SHCCSTR("The position/index of the first character or element that is to be extracted (including). "
                                "Negative position/indices simply loop over the target string/sequence counting backwards."),
                        CoreInfo::IntsVar),
      ParamsInfo::Param("To",
                        SHCCSTR("The position/index of the last character or element that is to be extracted (excluding). "
                                "Negative position/indices simply loop over the target string/sequence counting backwards."),
                        CoreInfo::IntsVarOrNone),
      ParamsInfo::Param("Step",
                        SHCCSTR("The increment between each position/index. Chooses every nth sample to extract, where n is the "
                                "increment. Value has to be greater than zero."),
                        CoreInfo::IntType));
  static SHOptionalString help() {
    return SHCCSTR("Extracts characters from a string or elements from a sequence based on the start and end positions/indices "
                   "and an increment parameter. Operation is non-destructive; the target string/sequence is not modified.");
  }

  SHSeq _cachedSeq{};
  std::vector<uint8_t> _cachedBytes{};
  SHVar _from{shards::Var(0)};
  SHVar *_fromVar = nullptr;
  SHVar _to{};
  SHVar *_toVar = nullptr;
  ExposedInfo _exposedInfo{};
  int64_t _step = 1;

  void destroy() {
    destroyVar(_from);
    destroyVar(_to);
  }

  void cleanup(SHContext *context) {
    if (_fromVar) {
      releaseVariable(_fromVar);
      _fromVar = nullptr;
    }
    if (_toVar) {
      releaseVariable(_toVar);
      _toVar = nullptr;
    }
    if (_cachedSeq.elements) {
      if (_step > 1) {
        // we cloned in this case
        for (auto i = _cachedSeq.len; i > 0; i--) {
          destroyVar(_cachedSeq.elements[i - 1]);
        }
      }
      shards::arrayFree(_cachedSeq);
    }
  }

  static inline Types InputTypes{{CoreInfo::AnySeqType, CoreInfo::BytesType, CoreInfo::StringType}};

  static SHTypesInfo inputTypes() { return InputTypes; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The string or sequence from which characters/elements have to be extracted.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The extracted characters/elements."); }

  static SHParametersInfo parameters() { return SHParametersInfo(indicesParamsInfo); }

  SHTypeInfo compose(const SHInstanceData &data) {
    bool valid = false;

    if (_from.valueType == SHType::Int) {
      valid = true;
    } else { // SHType::ContextVar
      for (auto &info : data.shared) {
        if (info.name == SHSTRVIEW(_from)) {
          valid = true;
          break;
        }
      }
    }

    if (!valid)
      throw SHException("Slice, invalid From variable.");

    if (_to.valueType == SHType::Int || _to.valueType == SHType::None) {
      valid = true;
    } else { // SHType::ContextVar
      for (auto &info : data.shared) {
        if (info.name == SHSTRVIEW(_to)) {
          valid = true;
          break;
        }
      }
    }

    if (!valid)
      throw SHException("Slice, invalid To variable.");

    if (data.inputType.basicType == SHType::Seq) {
      OVERRIDE_ACTIVATE(data, activateSeq);
    } else if (data.inputType.basicType == SHType::Bytes) {
      OVERRIDE_ACTIVATE(data, activateBytes);
    } else if (data.inputType.basicType == SHType::String) {
      OVERRIDE_ACTIVATE(data, activateString);
    }

    return data.inputType;
  }

  SHExposedTypesInfo requiredVariables() {
    // stringValue should be null terminated cos from and to are cloned!
    if (_from.valueType == SHType::ContextVar && _to.valueType == SHType::ContextVar) {
      _exposedInfo =
          ExposedInfo(ExposedInfo::Variable(_from.payload.stringValue, SHCCSTR("The required variable."), CoreInfo::IntType),
                      ExposedInfo::Variable(_to.payload.stringValue, SHCCSTR("The required variable."), CoreInfo::IntType));
      return SHExposedTypesInfo(_exposedInfo);
    } else if (_from.valueType == SHType::ContextVar) {
      _exposedInfo =
          ExposedInfo(ExposedInfo::Variable(_from.payload.stringValue, SHCCSTR("The required variable."), CoreInfo::IntType));
      return SHExposedTypesInfo(_exposedInfo);
    } else if (_to.valueType == SHType::ContextVar) {
      _exposedInfo =
          ExposedInfo(ExposedInfo::Variable(_to.payload.stringValue, SHCCSTR("The required variable."), CoreInfo::IntType));
      return SHExposedTypesInfo(_exposedInfo);
    } else {
      return {};
    }
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      cloneVar(_from, value);
      cleanup(nullptr);
      break;
    case 1:
      cloneVar(_to, value);
      cleanup(nullptr);
      break;
    case 2:
      _step = value.payload.intValue;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _from;
    case 1:
      return _to;
    case 2:
      return shards::Var(_step);
    default:
      break;
    }
    return shards::Var::Empty;
  }

  struct OutOfRangeEx : public ActivationError {
    OutOfRangeEx(int64_t len, int64_t from, int64_t to) : ActivationError("Slice out of range!") {
      SHLOG_ERROR("Out of range! from: {} to: {} len: {}", from, to, len);
    }
  };

  SHVar activateBytes(SHContext *context, const SHVar &input) {
    if (_from.valueType == SHType::ContextVar && !_fromVar) {
      _fromVar = referenceVariable(context, SHSTRVIEW(_from));
    }
    if (_to.valueType == SHType::ContextVar && !_toVar) {
      _toVar = referenceVariable(context, SHSTRVIEW(_to));
    }

    const auto inputLen = input.payload.bytesSize;
    const auto &vfrom = _fromVar ? *_fromVar : _from;
    const auto &vto = _toVar ? *_toVar : _to;
    SHInt from = vfrom.payload.intValue;
    SHInt to = vto.valueType == SHType::None ? SHInt(inputLen) : vto.payload.intValue;

    // Convert negative indices to positive
    from = from < 0 ? SHInt(inputLen) + from : from;
    to = to < 0 ? SHInt(inputLen) + to : to;

    // Bounds checking
    if (from < 0 || to < 0 || uint32_t(from) > inputLen || uint32_t(to) > inputLen) {
      throw OutOfRangeEx(inputLen, from, to);
    }

    // Ensure from is less than to
    if (from > to) {
      throw ActivationError("From index must be less than To index.");
    }

    uint32_t len = uint32_t(to - from);
    if (_step <= 0) {
      throw ActivationError("Slice's Step must be greater than 0");
    }

    uint32_t actualLen = len / _step + (len % _step != 0 ? 1 : 0);
    _cachedBytes.resize(actualLen); // Allocate sufficient space

    uint32_t idx = 0;
    for (int i = from; i < to && idx < actualLen; i += _step) {
      if (uint32_t(i) >= inputLen) {
        throw OutOfRangeEx(inputLen, i, i);
      }
      _cachedBytes[idx++] = input.payload.bytesValue[i];
    }
    return shards::Var(&_cachedBytes.front(), uint32_t(actualLen));
  }

  SHVar activateString(SHContext *context, const SHVar &input) {
    if (_from.valueType == SHType::ContextVar && !_fromVar) {
      _fromVar = referenceVariable(context, SHSTRVIEW(_from));
    }
    if (_to.valueType == SHType::ContextVar && !_toVar) {
      _toVar = referenceVariable(context, SHSTRVIEW(_to));
    }

    if (input.payload.stringValue == nullptr) {
      throw ActivationError("Input string is null.");
    }

    uint32_t inputLen = input.payload.stringLen > 0 ? input.payload.stringLen : uint32_t(strlen(input.payload.stringValue));

    const auto &vfrom = _fromVar ? *_fromVar : _from;
    const auto &vto = _toVar ? *_toVar : _to;
    SHInt from = vfrom.payload.intValue;
    SHInt to = vto.valueType == SHType::None ? int(inputLen) : vto.payload.intValue;

    // Convert negative indices to positive
    from = from < 0 ? SHInt(inputLen) + from : from;
    to = to < 0 ? SHInt(inputLen) + to : to;

    // Bounds checking
    if (from < 0 || to < 0 || uint32_t(from) > inputLen || uint32_t(to) > inputLen) {
      throw OutOfRangeEx(inputLen, from, to);
    }

    // Ensure from is less than to
    if (from > to) {
      throw ActivationError("From index must be less than To index.");
    }

    uint32_t len = uint32_t(to - from);
    if (_step <= 0) {
      throw ActivationError("Slice's Step must be greater than 0");
    }

    uint32_t actualLen = len / _step + (len % _step != 0 ? 1 : 0);
    _cachedBytes.resize(actualLen + 1); // +1 for the null terminator

    uint32_t idx = 0;
    for (int i = from; i < to && idx < actualLen; i += _step) {
      if (uint32_t(i) >= inputLen) {
        throw OutOfRangeEx(inputLen, i, i);
      }
      _cachedBytes[idx++] = input.payload.stringValue[i];
    }
    _cachedBytes[idx] = '\0'; // Ensure null termination
    return shards::Var((const char *)_cachedBytes.data(), actualLen);
  }

  SHVar activateSeq(SHContext *context, const SHVar &input) {
    if (_from.valueType == SHType::ContextVar && !_fromVar) {
      _fromVar = referenceVariable(context, SHSTRVIEW(_from));
    }
    if (_to.valueType == SHType::ContextVar && !_toVar) {
      _toVar = referenceVariable(context, SHSTRVIEW(_to));
    }

    const auto inputLen = input.payload.seqValue.len;
    const auto &vfrom = _fromVar ? *_fromVar : _from;
    const auto &vto = _toVar ? *_toVar : _to;
    SHInt from = vfrom.payload.intValue;
    SHInt to = vto.valueType == SHType::None ? SHInt(inputLen) : vto.payload.intValue;

    // Convert negative indices to positive
    from = from < 0 ? SHInt(inputLen) + from : from;
    to = to < 0 ? SHInt(inputLen) + to : to;

    // Bounds checking
    if (from < 0 || to < 0 || uint32_t(from) > inputLen || uint32_t(to) > inputLen) {
      throw OutOfRangeEx(inputLen, from, to);
    }

    // Ensure from is less than to
    if (from > to) {
      throw ActivationError("From index must be less than To index.");
    }

    const auto len = to - from;
    if (_step <= 0) {
      throw ActivationError("Slice's Step must be greater than 0");
    }

    if (_step == 1) {
      // Optimization for step of 1
      SHVar output{};
      output.valueType = SHType::Seq;
      output.payload.seqValue.elements = &input.payload.seqValue.elements[from];
      output.payload.seqValue.len = uint32_t(len);
      return output;
    } else {
      // General case for step greater than 1
      const auto actualLen = len / _step + (len % _step != 0 ? 1 : 0);
      shards::arrayResize(_cachedSeq, uint32_t(actualLen));
      auto idx = 0;
      for (int i = from; i < to && idx < actualLen; i += _step) {
        if (uint32_t(i) >= inputLen) {
          throw OutOfRangeEx(inputLen, i, i);
        }
        cloneVar(_cachedSeq.elements[idx], input.payload.seqValue.elements[i]);
        idx++;
      }
      return shards::Var(_cachedSeq);
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) { throw ActivationError("Slice: unreachable code path"); }
};

struct Limit {
  static inline shards::ParamsInfo indicesParamsInfo = shards::ParamsInfo(
      shards::ParamsInfo::Param("Max", SHCCSTR("How many maximum elements to take from the input sequence."), CoreInfo::IntType));

  SHSeq _cachedResult{};
  int64_t _max = 0;

  void destroy() {
    if (_cachedResult.elements) {
      shards::arrayFree(_cachedResult);
    }
  }

  static SHOptionalString help() {
    return SHCCSTR("This shard truncates the input sequence to the specified number of elements(specified by the Max parameter) "
                   "and outputs the truncated sequence.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnySeqType; }

  static SHOptionalString inputHelp() { return SHCCSTR("The input sequence to truncate."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHOptionalString outputHelp() { return SHCCSTR("The truncated sequence (or a single element if Max is 1)."); }

  static SHParametersInfo parameters() { return SHParametersInfo(indicesParamsInfo); }

  SHTypeInfo compose(const SHInstanceData &data) {
    // Figure if we output a sequence or not
    if (_max > 1) {
      if (data.inputType.basicType == SHType::Seq) {
        return data.inputType; // multiple values
      }
    } else {
      if (data.inputType.basicType == SHType::Seq && data.inputType.seqTypes.len == 1) {
        // single unique type
        return data.inputType.seqTypes.elements[0];
      } else {
        // multiple types
        return CoreInfo::AnySeqType;
      }
    }
    throw SHException("Limit expected a sequence as input.");
  }

  void setParam(int index, const SHVar &value) { _max = value.payload.intValue; }

  SHVar getParam(int index) { return shards::Var(_max); }

  SHVar activate(SHContext *context, const SHVar &input) {
    int64_t inputLen = input.payload.seqValue.len;

    if (_max == 1) {
      auto index = 0;
      if (index >= inputLen) {
        SHLOG_ERROR("Limit out of range! len: {} wanted index: {}", inputLen, index);
        throw ActivationError("Limit out of range!");
      }
      return input.payload.seqValue.elements[index];
    } else {
      // Else it's a seq
      auto nindices = (uint64_t)std::max((int64_t)0, std::min(inputLen, _max));
      shards::arrayResize(_cachedResult, uint32_t(nindices));
      for (uint64_t i = 0; i < nindices; i++) {
        _cachedResult.elements[i] = input.payload.seqValue.elements[i];
      }
      return shards::Var(_cachedResult);
    }
  }
};

struct RLimit {
  static inline shards::ParamsInfo paramsInfo = ParamsInfo(ParamsInfo::Param(
      "Max", SHCCSTR("The maximum number of elements to take from the end of the input sequence."), CoreInfo::IntType));

  SHSeq _cachedResult{};
  int64_t _max = 0;

  void destroy() {
    if (_cachedResult.elements) {
      shards::arrayFree(_cachedResult);
    }
  }

  static SHOptionalString help() {
    return SHCCSTR("This shard truncates the input sequence to the last specified number of elements (Max) and outputs the "
                   "truncated sequence. "
                   "If Max is set to 1, it outputs a single element.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnySeqType; }

  static SHOptionalString inputHelp() { return SHCCSTR("The input sequence to truncate."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The truncated sequence containing the last 'Max' elements, or a single element if Max is 1.");
  }

  static SHParametersInfo parameters() { return SHParametersInfo(paramsInfo); }

  SHTypeInfo compose(const SHInstanceData &data) {
    // Ensure the input is a sequence
    if (data.inputType.basicType != SHType::Seq) {
      throw ComposeError("RLimit shard expects a sequence as input.");
    }

    // Allow any sequence type for output
    return data.inputType;
  }

  void setParam(int index, const SHVar &value) {
    if (index == 0) {
      _max = value.payload.intValue;
    } else {
      throw SHException("RLimit: Parameter index out of range.");
    }
  }

  SHVar getParam(int index) {
    if (index == 0) {
      return shards::Var(_max);
    }
    throw SHException("RLimit: Parameter index out of range.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    int64_t inputLen = input.payload.seqValue.len;

    if (_max <= 0) {
      // If Max is zero or negative, return an empty sequence
      SHVar emptySeq;
      emptySeq.valueType = SHType::Seq;
      emptySeq.payload.seqValue.elements = nullptr;
      emptySeq.payload.seqValue.len = 0;
      return emptySeq;
    }

    if (_max == 1) {
      // Optimization for Max = 1: Return the last element
      if (inputLen == 0) {
        throw ActivationError("RLimit: Input sequence is empty.");
      }
      return input.payload.seqValue.elements[inputLen - 1];
    } else {
      // General case: Take the last 'Max' elements
      int64_t start = inputLen - _max;
      if (start < 0) {
        start = 0;
      }
      int64_t len = inputLen - start;

      // Resize the cached result to hold 'len' elements
      shards::arrayResize(_cachedResult, len);

      // Clone the last 'len' elements into the cached result
      for (int64_t i = 0; i < len; ++i) {
        cloneVar(_cachedResult.elements[i], input.payload.seqValue.elements[start + i]);
      }

      return shards::Var(_cachedResult);
    }
  }
};

struct ForRangeShard {
  static SHOptionalString help() {
    return SHCCSTR("Executes a series of shards while an iteration value is within a specified range.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHOptionalString inputHelp() { return SHCCSTR("The input value is not used and will pass through unchanged."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHOptionalString outputHelp() { return SHCCSTR("The output of this shard will be its input."); }

  static inline Parameters _params = {
      {"From", SHCCSTR("The initial iteration value (inclusive)."), {CoreInfo::IntType, CoreInfo::IntVarType}},
      {"To", SHCCSTR("The final iteration value (inclusive)."), {CoreInfo::IntType, CoreInfo::IntVarType}},
      {"Action",
       SHCCSTR("The action to perform at each iteration. The current iteration value will be passed as input."),
       {CoreInfo::ShardsOrNone}}};

  static SHParametersInfo parameters() { return _params; }

  ParamVar _from{Var(0)};
  ParamVar _to{Var(1)};
  ShardsVar _shards{};

  std::array<SHExposedTypeInfo, 2> _requiring;

  SHExposedTypesInfo requiredVariables() {
    uint32_t vars = 0;

    if (_from.isVariable()) {
      _requiring[vars].name = _from.variableName();
      _requiring[vars].help = SHCCSTR("The required variable index.");
      _requiring[vars].exposedType = CoreInfo::IntType;
      vars++;
    }

    if (_to.isVariable()) {
      _requiring[vars].name = _to.variableName();
      _requiring[vars].help = SHCCSTR("The required variable index.");
      _requiring[vars].exposedType = CoreInfo::IntType;
      vars++;
    }

    if (vars > 0) {
      return {_requiring.data(), vars, 0};
    } else {
      return {};
    }
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _from = value;
      break;
    case 1:
      _to = value;
      break;
    case 2:
      _shards = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _from;
    case 1:
      return _to;
    case 2:
      return _shards;
    default:
      throw InvalidParameterIndex();
    }
  }

  void cleanup(SHContext *context) {
    _from.cleanup(context);
    _to.cleanup(context);
    _shards.cleanup(context);
  }

  void warmup(SHContext *context) {
    _from.warmup(context);
    _to.warmup(context);
    _shards.warmup(context);
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    auto dataCopy = data;
    dataCopy.inputType = CoreInfo::IntType;

    _shards.compose(dataCopy);

    return data.inputType;
  }

  void activate(SHContext *context, const SHVar &input) {
    auto from = _from.get().payload.intValue;
    auto to = _to.get().payload.intValue;

    SHVar output{};
    SHVar item{};
    if (from <= to) {
      for (auto i = from; i <= to; i++) {
        item = Var(i);
        auto state = _shards.activate<true>(context, item, output);
        if (state != SHWireState::Continue)
          break;
      }
    } else if (from > to) {
      for (auto i = from; i >= to; i--) {
        item = Var(i);
        auto state = _shards.activate<true>(context, item, output);
        if (state != SHWireState::Continue)
          break;
      }
    }
  }
};

struct Repeat {
  ShardsVar _blks{};
  ShardsVar _pred{};
  ParamVar _times;
  bool _forever = false;
  ExposedInfo _requiredInfo{};

  void cleanup(SHContext *context) {
    _blks.cleanup(context);
    _pred.cleanup(context);
    _times.cleanup(context);
  }

  void warmup(SHContext *ctx) {
    _blks.warmup(ctx);
    _pred.warmup(ctx);
    _times.warmup(ctx);
  }

  static inline Parameters _params{
      {"Action", SHCCSTR("The shards to repeat."), CoreInfo::Shards},
      {"Times", SHCCSTR("How many times we should repeat the action."), {CoreInfo::IntOrIntVar, {CoreInfo::NoneType}}},
      {"Forever", SHCCSTR("If we should repeat the action forever."), {CoreInfo::BoolType}},
      {"Until",
       SHCCSTR("Optional shards to use as predicate to continue repeating "
               "until it's true"),
       CoreInfo::ShardsOrNone}};

  static SHOptionalString help() {
    return SHCCSTR("Repeat an action a given number of times or until the 'Until' parameter returns true.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The input will be passed to both the action and the `Until` condition if used.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output of this shard will be its input."); }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _blks = value;
      break;
    case 1:
      _times = value;
      break;
    case 2:
      _forever = value.payload.boolValue;
      break;
    case 3:
      _pred = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _blks;
    case 1:
      return _times;
    case 2:
      return shards::Var(_forever);
    case 3:
      return _pred;
    default:
      break;
    }
    throw SHException("Parameter out of range.");
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _blks.compose(data);
    const auto predres = _pred.compose(data);
    if (_pred && predres.outputType.basicType != SHType::Bool)
      throw ComposeError("Repeat shard Until predicate should output a boolean!");

    if (_pred) {
      OVERRIDE_ACTIVATE(data, activateUntilPred);
      data.shard->inlineShardId = InlineShard::NotInline;
    } else {
      OVERRIDE_ACTIVATE(data, activate);
      data.shard->inlineShardId = InlineShard::CoreRepeat;
    }

    _requiredInfo.clear();
    mergeIntoExposedInfo(_requiredInfo, predres.requiredInfo);
    mergeIntoExposedInfo(_requiredInfo, _times, CoreInfo::IntType, false);
    return data.inputType;
  }

  SHExposedTypesInfo requiredVariables() { return SHExposedTypesInfo(_requiredInfo); }

  FLATTEN ALWAYS_INLINE const SHVar &activate(SHContext *context, const SHVar &input) {
    bool forever = _forever || _times.isNone();
    int repeats;
    if (!_times.isNone()) {
      repeats = (int)(Var &)_times.get();
    } else {
      repeats = 0;
    }

    SHVar repeatOutput;

    if (forever) {
      while (true) {
        auto state = _blks.activate<true>(context, input, repeatOutput);
        if (state != SHWireState::Continue)
          break;
      }
    } else {
      for (int i = 0; i < repeats; ++i) {
        auto state = _blks.activate<true>(context, input, repeatOutput);
        if (state != SHWireState::Continue)
          break;
      }
    }

    return input;
  }

  FLATTEN ALWAYS_INLINE const SHVar &activateUntilPred(SHContext *context, const SHVar &input) {
    SHVar pres{};
    SHVar repeatOutput{};

    if (_times.isNone()) {
      // Infinite loop
      while (true) {
        auto state = _pred.activate<true>(context, input, pres);
        if (unlikely(state > SHWireState::Return || pres.payload.boolValue))
          break;

        state = _blks.activate<true>(context, input, repeatOutput);
        if (state != SHWireState::Continue)
          break;
      }
    } else {
      // Finite loop
      int repeats = _times.get().payload.intValue;
      while (repeats-- > 0) {
        auto state = _pred.activate<true>(context, input, pres);
        if (unlikely(state > SHWireState::Return || pres.payload.boolValue))
          break;

        state = _blks.activate<true>(context, input, repeatOutput);
        if (state != SHWireState::Continue)
          break;
      }
    }
    return input;
  }
};

RUNTIME_CORE_SHARD_TYPE(Const);
RUNTIME_CORE_SHARD_TYPE(And);
RUNTIME_CORE_SHARD_TYPE(Or);
RUNTIME_CORE_SHARD_TYPE(Not);
RUNTIME_CORE_SHARD_TYPE(Fail);
RUNTIME_CORE_SHARD_TYPE(Restart);
RUNTIME_CORE_SHARD_TYPE(Return);
RUNTIME_CORE_SHARD_TYPE(IsValidNumber);
RUNTIME_CORE_SHARD_TYPE(IsAlmost);
RUNTIME_CORE_SHARD_TYPE(Set);
RUNTIME_CORE_SHARD_TYPE(Ref);
RUNTIME_CORE_SHARD_TYPE(Update);
RUNTIME_CORE_SHARD_TYPE(Get);
RUNTIME_CORE_SHARD_TYPE(Swap);
RUNTIME_CORE_SHARD_TYPE(Take);
RUNTIME_CORE_SHARD_TYPE(RTake);
RUNTIME_CORE_SHARD_TYPE(Slice);
RUNTIME_CORE_SHARD_TYPE(Limit);
RUNTIME_CORE_SHARD_TYPE(RLimit);
RUNTIME_CORE_SHARD_TYPE(Push);
RUNTIME_CORE_SHARD_TYPE(Sequence);
RUNTIME_CORE_SHARD_TYPE(Pop);
RUNTIME_CORE_SHARD_TYPE(PopFront);
RUNTIME_CORE_SHARD_TYPE(Clear);
RUNTIME_CORE_SHARD_TYPE(Drop);
RUNTIME_CORE_SHARD_TYPE(DropFront);
RUNTIME_CORE_SHARD_TYPE(Count);
RUNTIME_CORE_SHARD_TYPE(Repeat);
}; // namespace shards

#endif // SH_CORE_SHARDS_CORE
