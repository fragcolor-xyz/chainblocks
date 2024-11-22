/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>

#ifdef NDEBUG
#define DEBUG_BREAK() abort()
#else
#if defined(_MSC_VER) // Microsoft Visual Studio
#include <intrin.h>
#define DEBUG_BREAK() __debugbreak()
#elif defined(__INTEL_COMPILER) // Intel C++ Compiler
#include <signal.h>
#define DEBUG_BREAK() raise(SIGTRAP)
#elif defined(__IBMCPP__) // IBM XL C/C++ Compiler
#include <builtins.h>
#define DEBUG_BREAK() __trap(0)
#elif defined(__clang__) // Clang
#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#define DEBUG_BREAK() __debugbreak()
#else
#include <signal.h>
#define DEBUG_BREAK() raise(SIGTRAP)
#endif
#elif defined(__GNUC__) // GCC
#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#define DEBUG_BREAK() __debugbreak()
#else
#include <signal.h>
#define DEBUG_BREAK() raise(SIGTRAP)
#endif
#elif defined(__HP_aCC) // HP C/aC++ Compiler
#include <signal.h>
#define DEBUG_BREAK() raise(SIGTRAP)
#elif defined(__SUNPRO_CC) // Oracle Developer Studio C++
#include <signal.h>
#define DEBUG_BREAK() raise(SIGTRAP)
#else
#error "Compiler not supported."
#endif
#endif

#ifdef BOOST_USE_VALGRIND
#include <valgrind/valgrind.h>
#include <valgrind/callgrind.h>
#endif

namespace shards {
namespace Assert {
struct Base {
  PARAM_PARAMVAR(_value, "Value", "The value to test against for equality.", {CoreInfo::AnyType});
  PARAM_VAR(_aborting, "Break", "If we should trigger a debug breakpoint on failure.", {CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_value), PARAM_IMPL_FOR(_aborting));

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The input can be of any type."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output will be the input (passthrough)."); }

  Base() { _aborting = Var(false); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }
};

struct Is : public Base {
  SHOptionalString help() {
    return SHCCSTR("This assertion is used to check whether the input is equal "
                   "to a given value.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input != _value.get()) {
      SHLOG_ERROR("Failed assertion Is, input: {} expected: {}", input, _value.get());
      if (isDebuggerPresent() && *_aborting)
        DEBUG_BREAK();
      else
        throw ActivationError("Assert failed - Is");
    }

    return input;
  }
};

struct IsStatic {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("Asserts that a value is static at run-time"); }

  PARAM_PARAMVAR(_value, "Value", "Any value to check for being static (non-var)", {shards::CoreInfo::AnyType});
  PARAM_IMPL(PARAM_IMPL_FOR(_value));

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    if (_value.isVariable()) {
      throw ActivationError("Assert failed - IsStatic");
    }
    return _value;
  }
};

struct IsVariable {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("Asserts that a value is a variable at run-time"); }

  PARAM_PARAMVAR(_value, "Value", "Any value to check for being a variable", {shards::CoreInfo::AnyType});
  PARAM_IMPL(PARAM_IMPL_FOR(_value));

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    if (!_value.isVariable()) {
      throw ActivationError("Assert failed - IsVariable");
    }
    return _value.get();
  }
};

struct IsNot : public Base {
  SHOptionalString help() {
    return SHCCSTR("This assertion is used to check whether the input is "
                   "different from a given value.");
  }
  SHVar activate(SHContext *context, const SHVar &input) {
    if (input == _value.get()) {
      SHLOG_ERROR("Failed assertion IsNot, input: {} not expected: {}", input, _value.get());
      if (isDebuggerPresent() && *_aborting)
        DEBUG_BREAK();
      else
        throw ActivationError("Assert failed - IsNot");
    }

    return input;
  }
};

struct IsAlmost {
  SHOptionalString help() {
    return SHCCSTR("This assertion is used to check whether the input is almost equal to a given value.");
  }

  static SHTypesInfo inputTypes() { return MathTypes; }
  static SHOptionalString inputHelp() { return SHCCSTR("The input can be of any number type or a sequence of such types."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output will be the input (passthrough)."); }

  SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &inValue) {
    switch (index) {
    case 0:
      _value = inValue;
      break;
    case 1:
      _aborting = inValue.payload.boolValue;
      break;
    case 2:
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
      return Var(_aborting);
    case 2:
      return Var(_threshold);
    default:
      return Var::Empty;
    }
  }

  void warmup(SHContext *context) { _value.warmup(context); }
  void cleanup(SHContext *context) { _value.cleanup(); }
  SHExposedTypesInfo requiredVariables() { return (SHExposedTypesInfo)_requiredVariables; }

  SHTypeInfo compose(const SHInstanceData &data) {
    _requiredVariables.clear();
    collectRequiredVariables(data, _requiredVariables, _value);

    if (_value.isVariable()) {
      auto exposed = findExposedVariable(data.shared, _value);
      if (!exposed) {
        throw SHException(fmt::format("Could not find exposed variable: {}", _value.variableName()));
      }
      if (exposed->exposedType.basicType != data.inputType.basicType) {
        throw SHException("Input and value types must match.");
      }
    } else {
      if (_value->valueType != data.inputType.basicType) {
        throw SHException("Input and value types must match.");
      }
    }

    if (_threshold <= 0.0)
      throw SHException("Threshold must be greater than 0.");

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!_almostEqual(input, _value.get(), _threshold)) {
      SHLOG_ERROR("Failed assertion IsAlmost, input: {} expected: {}", input, _value.get());
      if (isDebuggerPresent() && _aborting)
        DEBUG_BREAK();
      else
        throw ActivationError("Assert failed - IsAlmost");
    }

    return input;
  }

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
  static inline Type MathVarType = Type::VariableOf(MathTypes);
  static inline Parameters _params = {
      {"Value", SHCCSTR("The value to test against for almost equality."), {MathTypes, {MathVarType}}},
      {"Abort", SHCCSTR("If we should abort the process on failure."), {CoreInfo::BoolType}},
      {"Threshold",
       SHCCSTR("The smallest difference to be considered equal. Should be greater than zero."),
       {CoreInfo::FloatType, CoreInfo::IntType}}};

  bool _aborting;
  SHFloat _threshold{FLT_EPSILON};
  ParamVar _value{};
  shards::ExposedInfo _requiredVariables;
};

struct Break {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The input can be of any type."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output will be the input (passthrough)."); }

  bool _breaking = true; // set me to quit breaking while debugging

  SHVar activate(SHContext *context, const SHVar &input) {
    if (isDebuggerPresent() && _breaking)
      DEBUG_BREAK();
    return input;
  }
};

struct CallgrindStart {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The input can be of any type."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output will be the input (passthrough)."); }

  SHVar activate(SHContext *context, const SHVar &input) {
#ifdef BOOST_USE_VALGRIND
    CALLGRIND_START_INSTRUMENTATION;
#endif
    return input;
  }
};

struct CallgrindStop {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The input can be of any type."); }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The output will be the input (passthrough)."); }

  SHVar activate(SHContext *context, const SHVar &input) {
#ifdef BOOST_USE_VALGRIND
    CALLGRIND_STOP_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
#endif
    return input;
  }
};
} // namespace Assert
} // namespace shards

SHARDS_REGISTER_FN(assert) {
  using namespace shards::Assert;
  REGISTER_SHARD("Assert.Is", Is);
  REGISTER_SHARD("Assert.IsStatic", IsStatic);
  REGISTER_SHARD("Assert.IsVariable", IsVariable);
  REGISTER_SHARD("Assert.IsNot", IsNot);
  REGISTER_SHARD("Assert.IsAlmost", IsAlmost);
  REGISTER_SHARD("Debug.Break", Break);
  REGISTER_SHARD("_Callgrind.Start", CallgrindStart);
  REGISTER_SHARD("_Callgrind.Stop", CallgrindStop);
}
