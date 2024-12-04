/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

// TODO, remove most of C macros, use more templates

#ifndef SH_CORE_SHARDS_MATH
#define SH_CORE_SHARDS_MATH

#include <shards/core/foundation.hpp>
#include <shards/core/shards_macros.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <shards/common_types.hpp>
#include <shards/number_types.hpp>
#include <sstream>
#include <stdexcept>
#include <variant>
#include <shards/math_ops.hpp>
#include <shards/core/params.hpp>

#define _PC
#include "ShaderFastMathLib.h"
#undef _PC

namespace shards {
namespace Math {
struct Base {
  static inline Types MathTypesNoSeq{{CoreInfo::IntType, CoreInfo::Int2Type, CoreInfo::Int3Type, CoreInfo::Int4Type,
                                      CoreInfo::Int8Type, CoreInfo::Int16Type, CoreInfo::FloatType, CoreInfo::Float2Type,
                                      CoreInfo::Float3Type, CoreInfo::Float4Type, CoreInfo::ColorType}};
  static inline Types MathTypes{MathTypesNoSeq, {CoreInfo::AnySeqType}};

  SHVar _result{};

  void destroy() { destroyVar(_result); }

  SHTypeInfo compose(const SHInstanceData &data) { return data.inputType; }

  static SHTypesInfo inputTypes() { return MathTypes; }

  static SHOptionalString inputHelp() {
    return SHCCSTR("Any valid integer(s), floating point number(s), or a sequence of these types supported by this operation.");
  }

  static SHTypesInfo outputTypes() { return MathTypes; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The result of the operation, usually in the same type as the input value. If the input is a sequence, the "
                   "output will be a sequence of results, with possible broadcasting according to the input and operand.");
  }
};

enum OpType {
  Invalid,
  // Operation on scalar & vector type
  Broadcast,
  // Same types
  Direct,
  // Operation on sequence and vector/scalar type matching the sequence element type
  Seq1,
  // Operation on two sequences with the same vector/scalar element type
  SeqSeq
};

struct UnaryBase : public Base {
  OpType _opType = Invalid;

  void validateTypes(const SHTypeInfo &ti) {
    _opType = OpType::Invalid;
    if (ti.basicType == SHType::Seq) {
      _opType = OpType::Seq1;
      if (ti.seqTypes.len != 1)
        throw ComposeError("UnaryVarOperation expected a Seq with just one type as input");
    } else {
      _opType = OpType::Direct;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    validateTypes(data.inputType);
    return data.inputType;
  }
};

struct BinaryBase : public Base {
  static inline Types MathTypesOrVar{
      {CoreInfo::IntType,       CoreInfo::IntVarType,   CoreInfo::Int2Type,      CoreInfo::Int2VarType,  CoreInfo::Int3Type,
       CoreInfo::Int3VarType,   CoreInfo::Int4Type,     CoreInfo::Int4VarType,   CoreInfo::Int8Type,     CoreInfo::Int8VarType,
       CoreInfo::Int16Type,     CoreInfo::Int16VarType, CoreInfo::FloatType,     CoreInfo::FloatVarType, CoreInfo::Float2Type,
       CoreInfo::Float2VarType, CoreInfo::Float3Type,   CoreInfo::Float3VarType, CoreInfo::Float4Type,   CoreInfo::Float4VarType,
       CoreInfo::ColorType,     CoreInfo::ColorVarType, CoreInfo::AnySeqType,    CoreInfo::AnyVarSeqType}};

  static inline Types OnlyNumbers{{CoreInfo::IntType,    CoreInfo::IntVarType,    CoreInfo::Int2Type,   CoreInfo::Int2VarType,
                                   CoreInfo::Int3Type,   CoreInfo::Int3VarType,   CoreInfo::Int4Type,   CoreInfo::Int4VarType,
                                   CoreInfo::Int8Type,   CoreInfo::Int8VarType,   CoreInfo::Int16Type,  CoreInfo::Int16VarType,
                                   CoreInfo::FloatType,  CoreInfo::FloatVarType,  CoreInfo::Float2Type, CoreInfo::Float2VarType,
                                   CoreInfo::Float3Type, CoreInfo::Float3VarType, CoreInfo::Float4Type, CoreInfo::Float4VarType,
                                   CoreInfo::ColorType,  CoreInfo::ColorVarType}};

  static inline ParamsInfo mathParamsInfo =
      ParamsInfo(ParamsInfo::Param("Operand", SHCCSTR("The operand for this operation."), MathTypesOrVar));

  ParamVar _operand{shards::Var(0)};
  ExposedInfo _requiredInfo{};
  OpType _opType = Invalid;

  void cleanup(SHContext *context) { _operand.cleanup(); }

  void warmup(SHContext *context) { _operand.warmup(context); }

  static SHParametersInfo parameters() { return SHParametersInfo(mathParamsInfo); }

  ComposeError formatTypeError(const SHType &inputType, const SHType &paramType) {
    std::stringstream errStream;
    errStream << "Operation not supported between different types ";
    errStream << "(input=" << type2Name(inputType);
    errStream << ", param=" << type2Name(paramType) << ")";
    return ComposeError(errStream.str());
  }

  OpType validateTypes(const SHTypeInfo &lhs, const SHType &rhs, SHTypeInfo &resultType) {
    OpType opType = OpType::Invalid;
    if (rhs != SHType::Seq && lhs.basicType == SHType::Seq) {
      if (lhs.seqTypes.len != 1)
        throw ComposeError(fmt::format("Operation not supported with input sequence with multiple types: {}", lhs));
      if (rhs != lhs.seqTypes.elements[0].basicType)
        throw formatTypeError(lhs.seqTypes.elements[0].basicType, rhs);

      opType = Seq1;
    } else if (rhs == SHType::Seq && lhs.basicType == SHType::Seq) {
      // TODO need to have deeper types compatibility at least
      opType = SeqSeq;
    }
    return opType;
  }

  template <typename TValidator> SHTypeInfo genericCompose(TValidator &validator, const SHInstanceData &data) {
    SHTypeInfo resultType = data.inputType;
    SHVar operandSpec = _operand;
    if (operandSpec.valueType == SHType::ContextVar) {
      bool variableFound = false;
      for (uint32_t i = 0; i < data.shared.len; i++) {
        // normal variable
        if (data.shared.elements[i].name == SHSTRVIEW(operandSpec)) {
          _opType = validator.validateTypes(data.inputType, data.shared.elements[i].exposedType.basicType, resultType);
          variableFound = true;
          break;
        }
      }
      if (!variableFound)
        throw ComposeError(fmt::format("Operand variable \"{}\" not found", SHSTRVIEW(operandSpec)));
    } else {
      _opType = validator.validateTypes(data.inputType, operandSpec.valueType, resultType);
    }

    if (_opType == Invalid) {
      throw ComposeError("Incompatible types for binary operation");
    }

    return resultType;
  }

  SHTypeInfo compose(const SHInstanceData &data) { return genericCompose(*this, data); }

  SHExposedTypesInfo requiredVariables() {
    // operandSpec should be null terminated cos cloned over
    SHVar operandSpec = _operand;
    if (operandSpec.valueType == SHType::ContextVar) {
      _requiredInfo = ExposedInfo(
          ExposedInfo::Variable(operandSpec.payload.stringValue, SHCCSTR("The required operand."), CoreInfo::AnyType));
      return SHExposedTypesInfo(_requiredInfo);
    }
    return {};
  }

  void setParam(int index, const SHVar &value) { _operand = value; }

  SHVar getParam(int index) { return _operand; }
};

template <typename TOp> struct ApplyBinary {
  template <SHType ValueType> void apply(SHVarPayload &out, const SHVarPayload &a, const SHVarPayload &b) {
    typename PayloadTraits<ValueType>::ApplyBinary binary{};
    binary.template apply<TOp>(getPayloadContents<ValueType>(out), getPayloadContents<ValueType>(a),
                               getPayloadContents<ValueType>(b));
  }
};

struct ApplyBroadcast {
  template <SHType ValueType> void apply(SHVarPayload &out, const SHVarPayload &a) {
    using PayloadTraits = PayloadTraits<ValueType>;
    typename PayloadTraits::Broadcast broadcast{};
    typename PayloadTraits::UnitType utt{};
    broadcast.template apply<>(getPayloadContents<ValueType>(out), utt.getContents(const_cast<SHVarPayload &>(a)));
  }
};

template <typename TOp, DispatchType DispatchType = DispatchType::NumberTypes> struct BasicBinaryOperation {
  static constexpr shards::Math::DispatchType DispatchType_ = DispatchType;

  ApplyBinary<TOp> apply;
  ApplyBroadcast applyBroadcast;
  const VectorTypeTraits *_lhsVecType{};
  const VectorTypeTraits *_rhsVecType{};

  OpType validateTypes(const SHTypeInfo &lhs, const SHType &rhs, SHTypeInfo &resultType) {
    if (rhs != SHType::Seq && lhs.basicType != SHType::Seq) {
      _lhsVecType = VectorTypeLookup::getInstance().get(lhs.basicType);
      _rhsVecType = VectorTypeLookup::getInstance().get(rhs);
      if (_lhsVecType || _rhsVecType) {
        if (!_lhsVecType || !_rhsVecType)
          throw ComposeError(
              fmt::format("Unsupported types to binary operation ({} and {})", type2Name(lhs.basicType), type2Name(rhs)));

        bool sameDimension = _lhsVecType->dimension == _rhsVecType->dimension;
        if (!sameDimension && (_lhsVecType->dimension == 1 || _rhsVecType->dimension == 1)) {
          // Result is the vector type
          if (_rhsVecType->dimension == 1) {
            resultType = _lhsVecType->type;
          } else {
            resultType = _rhsVecType->type;
          }
          return Broadcast;
        } else {
          if (!sameDimension || _lhsVecType->numberType != _rhsVecType->numberType) {
            throw ComposeError(
                fmt::format("Can not multiply vector of size {} and {}", _lhsVecType->dimension, _rhsVecType->dimension));
          }
          return Direct;
        }
      }
    }

    return OpType::Invalid;
  }

  void operateDirect(SHVar &output, const SHVar &a, const SHVar &b) {
    output.valueType = a.valueType;
    dispatchType<DispatchType>(a.valueType, apply, output.payload, a.payload, b.payload);
  }

  void operateBroadcast(SHVar &output, const SHVar &a, const SHVar &b) {
    // This implements broadcast operators on float types
    const VectorTypeTraits *scalarType = _lhsVecType;
    const VectorTypeTraits *vecType = _rhsVecType;
    SHVarPayload aPayload = a.payload;
    SHVarPayload bPayload = b.payload;

    // Oparands might be swapped (e.g. v*s, s*v)
    if (vecType->dimension == 1) {
      std::swap(vecType, scalarType);
    }

    // Expand scalars to vectors
    if (_lhsVecType->dimension == 1) {
      SHVarPayload temp = aPayload; // Need temp var if input/output are the same
      dispatchType<DispatchType>(vecType->shType, applyBroadcast, aPayload, temp);
    }
    if (_rhsVecType->dimension == 1) {
      SHVarPayload temp = bPayload;
      dispatchType<DispatchType>(vecType->shType, applyBroadcast, bPayload, temp);
    }

    output.valueType = vecType->shType;
    dispatchType<DispatchType>(vecType->shType, apply, output.payload, aPayload, bPayload);
  }
};

///  The Op class has the following interface:
///
///  struct Op {
///    // Return the OpType based on the input types
///    // when OpType::Invalid is results, will fall back to BinaryOperation behaviour
///    // resultType needs to be set when OpType is not OpType::Invalid
///    OpType validateTypes(const SHTypeInfo &a, const SHType &b, SHTypeInfo &resultType);
///    // Apply OpType::Direct
///    void operateDirect(SHVar &output, const SHVar &a, const SHVar &b);
///    // Apply OpType::Broadcast
///    void operateBroadcast(SHVar &output, const SHVar &a, const SHVar &b);
///  };
template <class TOp> struct BinaryOperation : public BinaryBase {
  TOp op;

  static SHOptionalString help() {
    return SHCCSTR("Applies the binary operation on the input value and the operand and outputs the result (or a sequence of "
                   "results if the input and the operand are sequences).");
  }

  OpType validateTypes(const SHTypeInfo &lhs, const SHType &rhs, SHTypeInfo &resultType) {
    OpType opType = op.validateTypes(lhs, rhs, resultType);
    if (opType == OpType::Invalid)
      opType = BinaryBase::validateTypes(lhs, rhs, resultType);
    return opType;
  }

  SHTypeInfo compose(const SHInstanceData &data) { return this->genericCompose(*this, data); }

  void operate(OpType opType, SHVar &output, const SHVar &a, const SHVar &b) {
    if (opType == Broadcast) {
      op.operateBroadcast(output, a, b);
    } else if (opType == SeqSeq) {
      if (output.valueType != SHType::Seq) {
        destroyVar(output);
        output.valueType = SHType::Seq;
      }
      // TODO auto-parallelize with taskflow (should be optional)
      auto olen = b.payload.seqValue.len;
      shards::arrayResize(output.payload.seqValue, 0);
      for (uint32_t i = 0; i < a.payload.seqValue.len && olen > 0; i++) {
        const auto &sa = a.payload.seqValue.elements[i];
        const auto &sb = b.payload.seqValue.elements[i % olen];
        auto type = Direct;
        if (likely(sa.valueType == SHType::Seq && sb.valueType == SHType::Seq)) {
          type = SeqSeq;
        } else if (sa.valueType == SHType::Seq && sb.valueType != SHType::Seq) {
          type = Seq1;
        }
        const auto len = output.payload.seqValue.len;
        shards::arrayResize(output.payload.seqValue, len + 1);
        operate(type, output.payload.seqValue.elements[len], sa, sb);
      }
    } else {
      if (opType == Direct && output.valueType == SHType::Seq) {
        // something changed, avoid leaking
        // this should happen only here, because compose of SeqSeq is loose
        // we are going from an seq to a regular value, this could be expensive!
        SHLOG_DEBUG("Changing type of output during Math operation, this is ok "
                    "but potentially slow");
        destroyVar(output);
      }
      operateFast(opType, output, a, b);
    }
  }

  ALWAYS_INLINE void operateFast(OpType opType, SHVar &output, const SHVar &a, const SHVar &b) {
    if (likely(opType == Direct)) {
      op.operateDirect(output, a, b);
    } else if (opType == Seq1) {
      if (output.valueType != SHType::Seq) {
        destroyVar(output);
        output.valueType = SHType::Seq;
      }

      shards::arrayResize(output.payload.seqValue, 0);
      for (uint32_t i = 0; i < a.payload.seqValue.len; i++) {
        const auto len = output.payload.seqValue.len;
        shards::arrayResize(output.payload.seqValue, len + 1);
        op.operateDirect(output.payload.seqValue.elements[len], a.payload.seqValue.elements[i], b);
      }
    } else {
      operate(_opType, output, a, b);
    }
  }

  ALWAYS_INLINE const SHVar &activate(SHContext *context, const SHVar &input) {
    const auto operand = _operand.get();
    operateFast(_opType, _result, input, operand);
    return _result;
  }
};

template <class TOp> struct BinaryIntOperation : public BinaryOperation<TOp> {
  static inline Types IntOrSeqTypes{{CoreInfo::IntType, CoreInfo::Int2Type, CoreInfo::Int3Type, CoreInfo::Int4Type,
                                     CoreInfo::Int8Type, CoreInfo::Int16Type, CoreInfo::ColorType, CoreInfo::AnySeqType}};

  static inline Types IntOrSeqTypesOrBool{{CoreInfo::IntType, CoreInfo::Int2Type, CoreInfo::Int3Type, CoreInfo::Int4Type,
                                           CoreInfo::Int8Type, CoreInfo::Int16Type, CoreInfo::ColorType, CoreInfo::AnySeqType,
                                           CoreInfo::BoolType, CoreInfo::BoolVarType}};

  static SHParametersInfo parameters() {
    static Types ParamTypes = []() {
      static Types types = BinaryBase::MathTypesOrVar;
      if constexpr (hasDispatchType(TOp::DispatchType_, DispatchType::BoolTypes)) {
        types._types.push_back(CoreInfo::BoolType);
        types._types.push_back(CoreInfo::BoolVarType);
      }
      return types;
    }();
    static ParamsInfo Info(ParamsInfo::Param("Operand", SHCCSTR("The operand for this operation."), ParamTypes));
    return SHParametersInfo(Info);
  }

  static SHTypesInfo inputTypes() {
    static Types types = []() {
      Types types = IntOrSeqTypes;
      if constexpr (hasDispatchType(TOp::DispatchType_, DispatchType::BoolTypes)) {
        types._types.push_back(CoreInfo::BoolType);
      }
      return types;
    }();
    return types;
  }
  static SHTypesInfo outputTypes() { return inputTypes(); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("Any valid integer(s) or a sequence of integers supported by this operation.");
  }
};

template <typename TOp> struct ApplyUnary {
  template <SHType ValueType> void apply(SHVarPayload &out, const SHVarPayload &a) {
    typename PayloadTraits<ValueType>::ApplyUnary unary{};
    unary.template apply<TOp>(getPayloadContents<ValueType>(out), getPayloadContents<ValueType>(a));
  }
};

template <typename TOp, DispatchType DispatchType = DispatchType::NumberTypes> struct BasicUnaryOperation {
  ApplyUnary<TOp> apply;

  OpType validateTypes(const SHTypeInfo &a, SHTypeInfo &resultType) {
    OpType opType = OpType::Invalid;
    return opType;
  }

  void operateDirect(SHVar &output, const SHVar &a) {
    output.valueType = a.valueType;
    dispatchType<DispatchType>(a.valueType, apply, output.payload, a.payload);
  }
};

template <class TOp> struct UnaryOperation : public UnaryBase {
  TOp op;

  void destroy() { destroyVar(_result); }

  void validateTypes(const SHTypeInfo &ti, SHTypeInfo &resultType) {
    _opType = op.validateTypes(ti, resultType);
    if (_opType == OpType::Invalid)
      UnaryBase::validateTypes(ti);
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    SHTypeInfo resultType = data.inputType;
    validateTypes(data.inputType, resultType);
    return resultType;
  }

  static SHOptionalString help() {
    return SHCCSTR("Applies the unary operation on the input value and outputs the result. If the input is a sequence, the "
                   "operation is applied to each element of the sequence.");
  }

  ALWAYS_INLINE void operate(SHVar &output, const SHVar &a) {
    if (likely(_opType == OpType::Direct)) {
      op.operateDirect(output, a);
    } else if (_opType == OpType::Seq1) {
      if (output.valueType != SHType::Seq) {
        destroyVar(output);
        output.valueType = SHType::Seq;
      }

      shards::arrayResize(output.payload.seqValue, a.payload.seqValue.len);
      for (uint32_t i = 0; i < a.payload.seqValue.len; i++) {
        op.operateDirect(output.payload.seqValue.elements[i], a.payload.seqValue.elements[i]);
      }
    } else {
      throw std::logic_error("Invalid operation type for unary operation");
    }
  }

  ALWAYS_INLINE SHVar activate(SHContext *context, const SHVar &input) {
    operate(_result, input);
    return _result;
  }
};

template <class TOp> struct UnaryFloatOperation : public UnaryOperation<TOp> {
  static inline Types FloatOrSeqTypes{{CoreInfo::FloatType, CoreInfo::Float2Type, CoreInfo::Float3Type, CoreInfo::Float4Type,
                                       CoreInfo::ColorType, CoreInfo::AnySeqType}};

  static SHTypesInfo inputTypes() { return FloatOrSeqTypes; }
  static SHOptionalString inputHelp() {
    return SHCCSTR("A floating point number, a vector of floats (Float2, Float3, Float4), a color, or a sequence of these types "
                   "supported by this operation.");
  }
  static SHTypesInfo outputTypes() { return FloatOrSeqTypes; }
};

template <class TOp> struct UnaryVarOperation : public UnaryOperation<TOp> {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  ExposedInfo _requiredInfo{};
  ParamVar _value{};

  static inline Parameters params{
      {"Value",
       SHCCSTR("The value to apply the operation to."),
       {CoreInfo::IntVarType, CoreInfo::Int2VarType, CoreInfo::Int3VarType, CoreInfo::Int4VarType, CoreInfo::Int8VarType,
        CoreInfo::Int16VarType, CoreInfo::FloatVarType, CoreInfo::Float2VarType, CoreInfo::Float3VarType, CoreInfo::Float4VarType,
        CoreInfo::ColorVarType, CoreInfo::AnyVarSeqType}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) { _value = value; }

  SHVar getParam(int index) { return _value; }

  SHTypeInfo compose(const SHInstanceData &data) {
    SHTypeInfo resultType = data.inputType;

    if (!_value.isVariable()) {
      throw ComposeError("UnaryVarOperation Expected a variable");
    }

    for (const auto &share : data.shared) {
      if (share.name == SHSTRVIEW((*_value))) {
        if (share.isProtected)
          throw ComposeError("UnaryVarOperation cannot write protected variables");
        if (!share.isMutable)
          throw ComposeError("UnaryVarOperation attempt to write immutable variable");

        this->validateTypes(share.exposedType, resultType);
        assert(resultType == data.inputType);

        return resultType;
      }
    }

    throw ComposeError(fmt::format("Math.Inc/Dec variable {} not found", SHSTRVIEW((*_value))));
  }

  SHExposedTypesInfo requiredVariables() {
    if (_value.isVariable()) {
      _requiredInfo =
          ExposedInfo(ExposedInfo::Variable(_value.variableName(), SHCCSTR("The required operand."), CoreInfo::AnyType));
      return SHExposedTypesInfo(_requiredInfo);
    }
    return {};
  }

  void warmup(SHContext *context) { _value.warmup(context); }

  void cleanup(SHContext *context) { _value.cleanup(); }

  ALWAYS_INLINE void activate(SHContext *context, const SHVar &input) {
    this->operate(_value.get(), _value.get());
  }
};

#define MATH_BINARY_OPERATION(NAME, OPERATOR, DIV_BY_ZERO)      \
  using NAME = BinaryOperation<BasicBinaryOperation<NAME##Op>>; \
  RUNTIME_SHARD_TYPE(Math, NAME);

#define MATH_BINARY_INT_OPERATION(NAME, OPERATOR)                                          \
  using NAME = BinaryIntOperation<BasicBinaryOperation<NAME##Op, DispatchType::IntTypes>>; \
  RUNTIME_SHARD_TYPE(Math, NAME);

#define MATH_BINARY_INT_BOOL_OPERATION(NAME, OPERATOR)                                           \
  using NAME = BinaryIntOperation<BasicBinaryOperation<NAME##Op, DispatchType::IntOrBoolTypes>>; \
  RUNTIME_SHARD_TYPE(Math, NAME);

struct Add : public BinaryOperation<BasicBinaryOperation<AddOp>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard adds the input value to the value provided in the Operand parameter.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to add the value specified in the Operand parameter to.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the result of the addition."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to add to the input."), MathTypesOrVar));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Add);

struct Subtract : public BinaryOperation<BasicBinaryOperation<SubtractOp>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard subtracts the value provided in the Operand parameter from the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to subtract the value specified in the Operand parameter from.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the result of the subtraction."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to subtract from the input."), MathTypesOrVar));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Subtract);

struct Multiply : public BinaryOperation<BasicBinaryOperation<MultiplyOp>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard multiplies the input value by the value provided in the Operand parameter.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to multiply the value specified in the Operand parameter with.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the result of the multiplication."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to multiply the input by."), MathTypesOrVar));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Multiply);

struct Divide : public BinaryOperation<BasicBinaryOperation<DivideOp>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard divides the input value by the value provided in the Operand parameter.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to divide the value specified in the Operand parameter with.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the result of the division."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to divide the input by."), MathTypesOrVar));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Divide);

struct Mod : public BinaryOperation<BasicBinaryOperation<ModOp>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard calculates the remainder of the division of the input value by the value provided in the Operand parameter.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to divide the value specified in the Operand parameter with.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the result of the modulus operation."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(ParamsInfo::Param(
        "Operand", SHCCSTR("The value or sequence of values to divide the input by and get the remainder of."), MathTypesOrVar));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Mod);

struct Xor : public BinaryIntOperation<BasicBinaryOperation<XorOp, DispatchType::IntOrBoolTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard performs a bitwise XOR operation on the input with the value specified in the "
                   "Operand parameter and outputs the result. A bitwise XOR operation is a binary operation that compares each "
                   "bit of the binary representations of two numbers and outputs 1 if the bits are different and 0 if they are "
                   "the same. The shard then outputs a value, whose binary representation is the concatenation of the resulting "
                   "1s and 0s from the XOR comparison.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to compare the value specified in the Operand parameter with.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the value resulting from the XOR operation."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to compare the input with."), IntOrSeqTypesOrBool));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Xor);

struct And : public BinaryIntOperation<BasicBinaryOperation<AndOp, DispatchType::IntOrBoolTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard performs a bitwise AND operation on the input value with the value specified in the "
                   "Operand parameter and outputs the result. A bitwise AND operation is a binary operation that compares each "
                   "bit of the binary representations of two numbers and outputs 1 if the bits are 1 and 0 otherwise. The "
                   "shard then outputs a value, whose binary representation is the concatenation of the resulting 1s and 0s from "
                   "the AND comparison.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to compare the value specified in the Operand parameter with.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the value resulting from the AND operation."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to compare the input with."), IntOrSeqTypesOrBool));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, And);

struct Or : public BinaryIntOperation<BasicBinaryOperation<OrOp, DispatchType::IntOrBoolTypes>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard performs a bitwise OR operation on the input value with the value specified in the "
        "Operand parameter and outputs the result. A bitwise OR operation is a binary operation that compares each "
        "bit of the binary representations of two numbers and outputs 1 if either bit is 1 and 0 if both bits are 0. The "
        "shard then outputs a value, whose binary representation is the concatenation of the resulting 1s and 0s from the Or "
        "comparison.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The value or the sequence of values to compare the value specified in the Operand parameter with.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("This shard outputs the value resulting from the OR operation."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(
        ParamsInfo::Param("Operand", SHCCSTR("The value or sequence of values to compare the input with."), IntOrSeqTypesOrBool));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, Or);

struct LShift : public BinaryIntOperation<BasicBinaryOperation<LShiftOp, DispatchType::IntTypes>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard shifts the bits of the input value to the left by the number of positions specified in the Operand "
        "parameter. The shard then outputs a value, whose binary representation is the resulting shifted binary.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The integer or the sequence of integers to shift the bits of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the value resulting from the left shift operation."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(ParamsInfo::Param(
        "Operand", SHCCSTR("The number of positions to shift the bits of the input value to the left by."), IntOrSeqTypes));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, LShift);

struct RShift : public BinaryIntOperation<BasicBinaryOperation<RShiftOp, DispatchType::IntTypes>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard shifts the bits of the input value to the right by the number of positions specified in the Operand "
        "parameter. The shard then outputs a value, whose binary representation is the resulting shifted binary.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The integer or the sequence of integers to shift the bits of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the value resulting from the right shift operation."); }

  static SHParametersInfo parameters() {
    static ParamsInfo customParams(ParamsInfo::Param(
        "Operand", SHCCSTR("The number of positions to shift the bits of the input value to the right by."), IntOrSeqTypes));
    return SHParametersInfo(customParams);
  }
};
RUNTIME_SHARD_TYPE(Math, RShift);

#define MATH_UNARY_OPERATION(NAME, FUNCI, FUNCF)    \
  struct NAME##Op final {                           \
    template <typename T> T apply(const T &lhs) {   \
      if constexpr (std::is_unsigned_v<T>) {        \
        return lhs;                                 \
      } else if constexpr (std::is_integral_v<T>) { \
        return FUNCI(lhs);                          \
      } else {                                      \
        return FUNCF(lhs);                          \
      }                                             \
    }                                               \
  };

#define MATH_UNARY_FLOAT_OPERATION(NAME, FUNC, FUNCF)                 \
  struct NAME##Op final {                                             \
    template <typename T> T apply(const T &lhs) { return FUNC(lhs); } \
  };

MATH_UNARY_OPERATION(Abs, __builtin_llabs, __builtin_fabs);
MATH_UNARY_FLOAT_OPERATION(Exp, __builtin_exp, __builtin_expf);
MATH_UNARY_FLOAT_OPERATION(Exp2, __builtin_exp2, __builtin_exp2f);
MATH_UNARY_FLOAT_OPERATION(Expm1, __builtin_expm1, __builtin_expm1f);
MATH_UNARY_FLOAT_OPERATION(Log, __builtin_log, __builtin_logf);
MATH_UNARY_FLOAT_OPERATION(Log10, __builtin_log10, __builtin_log10f);
MATH_UNARY_FLOAT_OPERATION(Log2, __builtin_log2, __builtin_log2f);
MATH_UNARY_FLOAT_OPERATION(Log1p, __builtin_log1p, __builtin_log1pf);
MATH_UNARY_FLOAT_OPERATION(Sqrt, __builtin_sqrt, __builtin_sqrtf);
MATH_UNARY_FLOAT_OPERATION(FastSqrt, fastSqrtNR2, fastSqrtNR2);
MATH_UNARY_FLOAT_OPERATION(FastInvSqrt, fastRcpSqrtNR2, fastRcpSqrtNR2);
MATH_UNARY_FLOAT_OPERATION(Cbrt, __builtin_cbrt, __builtin_cbrt);
MATH_UNARY_FLOAT_OPERATION(Sin, __builtin_sin, __builtin_sinf);
MATH_UNARY_FLOAT_OPERATION(Cos, __builtin_cos, __builtin_cosf);
MATH_UNARY_FLOAT_OPERATION(Tan, __builtin_tan, __builtin_tanf);
MATH_UNARY_FLOAT_OPERATION(Asin, __builtin_asin, __builtin_asinf);
MATH_UNARY_FLOAT_OPERATION(Acos, __builtin_acos, __builtin_acosf);
MATH_UNARY_FLOAT_OPERATION(Atan, __builtin_atan, __builtin_atanf);
MATH_UNARY_FLOAT_OPERATION(Sinh, __builtin_sinh, __builtin_sinhf);
MATH_UNARY_FLOAT_OPERATION(Cosh, __builtin_cosh, __builtin_coshf);
MATH_UNARY_FLOAT_OPERATION(Tanh, __builtin_tanh, __builtin_tanhf);
MATH_UNARY_FLOAT_OPERATION(Asinh, __builtin_asinh, __builtin_asinhf);
MATH_UNARY_FLOAT_OPERATION(Acosh, __builtin_acosh, __builtin_acoshf);
MATH_UNARY_FLOAT_OPERATION(Atanh, __builtin_atanh, __builtin_atanhf);
MATH_UNARY_FLOAT_OPERATION(Erf, __builtin_erf, __builtin_erff);
MATH_UNARY_FLOAT_OPERATION(Erfc, __builtin_erfc, __builtin_erfcf);
MATH_UNARY_FLOAT_OPERATION(TGamma, __builtin_tgamma, __builtin_tgammaf);
MATH_UNARY_FLOAT_OPERATION(LGamma, __builtin_lgamma, __builtin_lgammaf);
MATH_UNARY_FLOAT_OPERATION(Ceil, __builtin_ceil, __builtin_ceilf);
MATH_UNARY_FLOAT_OPERATION(Floor, __builtin_floor, __builtin_floorf);
MATH_UNARY_FLOAT_OPERATION(Trunc, __builtin_trunc, __builtin_truncf);
MATH_UNARY_FLOAT_OPERATION(Round, __builtin_round, __builtin_roundf);

struct Abs : public UnaryOperation<BasicUnaryOperation<AbsOp, DispatchType::NumberTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard outputs the absolute value of the input."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The numeric value or a sequence of numeric values to get the absolute value of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the absolute value of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Abs);

struct Exp : public UnaryFloatOperation<BasicUnaryOperation<ExpOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the exponential function with base e (Euler's number) for the given input. The "
                   "exponential function is equivalent to raising Euler's number to the power of the input.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to use as the exponent for the base e exponential function.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the result of the exponential operation."); }
};
RUNTIME_SHARD_TYPE(Math, Exp);

struct Exp2 : public UnaryFloatOperation<BasicUnaryOperation<Exp2Op, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the exponential function with base 2 for the given input. The exponential function "
                   "with base 2 is equivalent to raising 2 to the power of the input.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats used as the exponent for the base 2 exponential function.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the result of the exponential operation."); }
};
RUNTIME_SHARD_TYPE(Math, Exp2);

struct Expm1 : public UnaryFloatOperation<BasicUnaryOperation<Expm1Op, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the exponential function with base e (Euler's number) for the given input and "
                   "subtracts 1 from the result.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats used as the exponent for the base e exponential function.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the result of the exponential operation."); }
};
RUNTIME_SHARD_TYPE(Math, Expm1);

struct Log : public UnaryFloatOperation<BasicUnaryOperation<LogOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the natural logarithm for the given input. The output is the exponent to which e must "
                   "be raised to obtain the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the natural logarithm of. This value must be a positive "
                   "number or sequence of positive numbers.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the natural logarithm of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Log);

struct Log10 : public UnaryFloatOperation<BasicUnaryOperation<Log10Op, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the base 10 logarithm for the given input. The output is the exponent to which 10 must "
                   "be raised to obtain the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the base 10 logarithm of. This value must be a positive "
                   "number or sequence of positive numbers.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the base 10 logarithm of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Log10);

struct Log2 : public UnaryFloatOperation<BasicUnaryOperation<Log2Op, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the base 2 logarithm for the given input. The output is the exponent to which 2 must "
                   "be raised to obtain the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the base 2 logarithm of. This value must be a positive "
                   "number or sequence of positive numbers.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the base 2 logarithm."); }
};
RUNTIME_SHARD_TYPE(Math, Log2);

struct Log1p : public UnaryFloatOperation<BasicUnaryOperation<Log1pOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard adds 1 to the input and then calculates the natural logarithm of the result.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to add 1 to and then calculate the natural logarithm of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the natural logarithm of the input plus 1."); }
};
RUNTIME_SHARD_TYPE(Math, Log1p);

struct Sqrt : public UnaryFloatOperation<BasicUnaryOperation<SqrtOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard calculates the square root of the given input."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the square root of. This value must be a positive number "
                   "or sequence of positive numbers.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the square root of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Sqrt);

struct FastSqrt : public UnaryFloatOperation<BasicUnaryOperation<FastSqrtOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard calculates the square root of the given input."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the square root of. This value must be a positive number "
                   "or sequence of positive numbers.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the square root of the input."); }
};
RUNTIME_SHARD_TYPE(Math, FastSqrt);

struct FastInvSqrt : public UnaryFloatOperation<BasicUnaryOperation<FastInvSqrtOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard calculates the inverse square root of the given input."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse square root of. This value must be a positive "
                   "number or sequence of positive numbers.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the inverse square root of the input."); }
};
RUNTIME_SHARD_TYPE(Math, FastInvSqrt);

struct Cbrt : public UnaryFloatOperation<BasicUnaryOperation<CbrtOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard calculates the cube root of the given input."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to calculate the cube root of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the cube root of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Cbrt);

struct Sin : public UnaryFloatOperation<BasicUnaryOperation<SinOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the sine of the given input, where the input is the angle in radians.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to calculate the sine of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the sine of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Sin);

struct Cos : public UnaryFloatOperation<BasicUnaryOperation<CosOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the cosine of the given input, where the input is the angle in radians.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to calculate the cosine of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the cosine of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Cos);

struct Tan : public UnaryFloatOperation<BasicUnaryOperation<TanOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the tangent of the given input, where the input is the angle in radians.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to calculate the tangent of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the tangent of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Tan);

struct Asin : public UnaryFloatOperation<BasicUnaryOperation<AsinOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the inverse sine of the given input, where the input is the sine value. The output is "
                   "the angle in radians whose sine is the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse sine of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the angle in radians whose sine is the input value."); }
};
RUNTIME_SHARD_TYPE(Math, Asin);

struct Acos : public UnaryFloatOperation<BasicUnaryOperation<AcosOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the inverse cosine of the given input, where the input is the cosine value. The output "
                   "is the angle in radians whose cosine is the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse cosine of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the angle in radians whose cosine is the input value."); }
};
RUNTIME_SHARD_TYPE(Math, Acos);

struct Atan : public UnaryFloatOperation<BasicUnaryOperation<AtanOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the inverse tangent of the given input, where the input is the tangent value. The "
                   "output is the angle in radians whose tangent is the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse tangent of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the angle in radians whose tangent is the input value."); }
};
RUNTIME_SHARD_TYPE(Math, Atan);

struct Sinh : public UnaryFloatOperation<BasicUnaryOperation<SinhOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the hyperbolic sine of the given input, where the input is the real number. The "
                   "hyperbolic sine is a hyperbolic function that is analogous to the circular sine function, but it uses "
                   "exponential functions instead of angles.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the hyperbolic sine of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the hyperbolic sine of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Sinh);

struct Cosh : public UnaryFloatOperation<BasicUnaryOperation<CoshOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the hyperbolic cosine of the given input, where the input is the real number. The "
                   "hyperbolic cosine is a hyperbolic function that is analogous to the circular cosine function, but it uses "
                   "exponential functions instead of angles.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the hyperbolic cosine of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the hyperbolic cosine of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Cosh);

struct Tanh : public UnaryFloatOperation<BasicUnaryOperation<TanhOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the hyperbolic tangent of the given input, where the input is the real number. The "
                   "hyperbolic tangent is a hyperbolic function that is analogous to the circular tangent function, but it uses "
                   "exponential functions instead of angles.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the hyperbolic tangent of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the hyperbolic tangent of the input."); }
};
RUNTIME_SHARD_TYPE(Math, Tanh);

struct Asinh : public UnaryFloatOperation<BasicUnaryOperation<AsinhOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the inverse hyperbolic sine of the given input, where the input is the hyperbolic sine "
                   "value. The output is the real number whose hyperbolic sine is the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse hyperbolic sine of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the real number whose hyperbolic sine is the input value."); }
};
RUNTIME_SHARD_TYPE(Math, Asinh);

struct Acosh : public UnaryFloatOperation<BasicUnaryOperation<AcoshOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the inverse hyperbolic cosine of the given input, where the input is the hyperbolic "
                   "cosine value. The output is the real number whose hyperbolic cosine is the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse hyperbolic cosine of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the real number whose hyperbolic cosine is the input value."); }
};
RUNTIME_SHARD_TYPE(Math, Acosh);

struct Atanh : public UnaryFloatOperation<BasicUnaryOperation<AtanhOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the inverse hyperbolic tangent of the given input (atanh(x)), where x, outputs y such "
                   "that tanh(y) = x.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the inverse hyperbolic tangent of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the real number whose hyperbolic tangent is the input value."); }
};
RUNTIME_SHARD_TYPE(Math, Atanh);

struct Erf : public UnaryFloatOperation<BasicUnaryOperation<ErfOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard calculates the error function of the given input. The error function is related to the probability "
        "that a random variable with normal distribution of mean 0 and variance 1/2 falls in the range specified by the input "
        "value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the error function of. This can be any real number.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs probability result of the error function of the input. The output is always between -1 and 1.");
  }
};
RUNTIME_SHARD_TYPE(Math, Erf);

struct Erfc : public UnaryFloatOperation<BasicUnaryOperation<ErfcOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the complementary error function of the given input. The complementary error function "
                   "is related to the probability "
                   "that the absolute value of a random variable with normal distribution of mean 0 and variance "
                   "1/2 is greater than the input value.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR(
        "The input float or sequence of floats to calculate the complementary error function of. This can be any real number.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR(
        "Outputs the probability result of the complementary error function of the input. The output is always between 0 and 2.");
  }
};
RUNTIME_SHARD_TYPE(Math, Erfc);

struct TGamma : public UnaryFloatOperation<BasicUnaryOperation<TGammaOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the gamma function of the given input. The gamma function is a mathematical function "
                   "that extends the concept of factorial to non-integer and complex numbers.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("This shard calculates the gamma function of the given input."); }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the gamma function of the input. The output is always positive for positive inputs.");
  }
};
RUNTIME_SHARD_TYPE(Math, TGamma);

struct LGamma : public UnaryFloatOperation<BasicUnaryOperation<LGammaOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the log gamma function of the given input. The log gamma function is the natural "
                   "logarithm of the absolute value of the gamma function.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input float or sequence of floats to calculate the log gamma function of.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the log gamma function of the input. The output is always positive for positive inputs.");
  }
};
RUNTIME_SHARD_TYPE(Math, LGamma);

struct Ceil : public UnaryFloatOperation<BasicUnaryOperation<CeilOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard rounds up the input to the nearest integer."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to round up."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input rounded up to the nearest integer (as a float)."); }
};
RUNTIME_SHARD_TYPE(Math, Ceil);

struct Floor : public UnaryFloatOperation<BasicUnaryOperation<FloorOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard rounds down the input to the nearest integer."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to round down."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input rounded down to the nearest integer (as a float)."); }
};
RUNTIME_SHARD_TYPE(Math, Floor);

struct Trunc : public UnaryFloatOperation<BasicUnaryOperation<TruncOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard truncates the input floating-point number towards zero, removing any fractional part without rounding.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to truncate."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input truncated to the nearest integer (as a float)."); }
};
RUNTIME_SHARD_TYPE(Math, Trunc);

struct Round : public UnaryFloatOperation<BasicUnaryOperation<RoundOp, DispatchType::FloatTypes>> {
  static SHOptionalString help() { return SHCCSTR("This shard rounds the input floating-point number to the nearest integer."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The input float or sequence of floats to round."); }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input rounded to the nearest integer (as a float)."); }
};
RUNTIME_SHARD_TYPE(Math, Round);

struct Mean {
  struct ArithMean {
    double operator()(const SHSeq &seq) {
      const uint32_t inputLen = seq.len;
      double mean = 0.0;
      for (uint32_t i = 0; i < inputLen; i++) {
        const auto &v = seq.elements[i];
        mean += v.payload.floatValue;
      }
      mean /= double(inputLen);
      return mean;
    }
  };

  struct GeoMean {
    double operator()(const SHSeq &seq) {
      const uint32_t inputLen = seq.len;
      double mean = 1.0;
      for (uint32_t i = 0; i < inputLen; i++) {
        const auto &v = seq.elements[i];
        mean *= v.payload.floatValue;
      }
      return std::pow(mean, 1.0 / double(inputLen));
    }
  };

  struct HarmoMean {
    double operator()(const SHSeq &seq) {
      const uint32_t inputLen = seq.len;
      double mean = 0.0;
      for (uint32_t i = 0; i < inputLen; i++) {
        const auto &v = seq.elements[i];
        mean += 1.0 / v.payload.floatValue;
      }
      return double(inputLen) / mean;
    }
  };

  enum class MeanKind { Arithmetic, Geometric, Harmonic };
  DECL_ENUM_INFO(
      MeanKind, Mean,
      "Type of mean calculation to be performed. Specifies whether to use arithmetic, geometric, or harmonic averaging.", 'mean');

  static SHOptionalString help() { return SHCCSTR("Calculates the average value of a sequence of floating point numbers."); }

  static SHTypesInfo inputTypes() { return CoreInfo::FloatSeqType; }
  static SHOptionalString inputHelp() { return SHCCSTR("The sequence of floating point numbers to calculate the average of."); }

  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }
  static SHOptionalString outputHelp() { return SHCCSTR("The calculated average as a float."); }

  static SHParametersInfo parameters() {
    static Parameters params{{"Kind", SHCCSTR("The type of average to calculate."), {MeanEnumInfo::Type}}};
    return params;
  }

  void setParam(int index, const SHVar &value) { mean = MeanKind(value.payload.enumValue); }

  SHVar getParam(int index) { return shards::Var::Enum(mean, CoreCC, MeanEnumInfo::TypeId); }

  SHVar activate(SHContext *context, const SHVar &input) {
    switch (mean) {
    case MeanKind::Arithmetic: {
      ArithMean m;
      return shards::Var(m(input.payload.seqValue));
    }
    case MeanKind::Geometric: {
      GeoMean m;
      return shards::Var(m(input.payload.seqValue));
    }
    case MeanKind::Harmonic: {
      HarmoMean m;
      return shards::Var(m(input.payload.seqValue));
    }
    default:
      throw ActivationError("Invalid mean case.");
    }
  }

  MeanKind mean{MeanKind::Arithmetic};
};

struct IncOp {
  template <typename T> T apply(const T &a) { return a + 1; }
};

struct Inc : public UnaryVarOperation<BasicUnaryOperation<IncOp>> {
  static SHOptionalString help() { return SHCCSTR("Increases the input by 1."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The float or integer (or sequence of floats or integers) to increase by 1.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("The input increased by 1."); }
};
RUNTIME_SHARD_TYPE(Math, Inc);

struct DecOp {
  template <typename T> T apply(const T &a) { return a - 1; }
};

struct Dec : public UnaryVarOperation<BasicUnaryOperation<DecOp>> {
  static SHOptionalString help() { return SHCCSTR("Decreases the input by 1."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The float or integer (or sequence of floats or integers) to decrease by 1.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("The input decreased by 1."); }
};
RUNTIME_SHARD_TYPE(Math, Dec);

struct NegateOp {
  template <typename T> T apply(const T &a) { return -a; }
};

struct Negate : public UnaryOperation<BasicUnaryOperation<NegateOp>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard reverses the sign of the input. (A positive number becomes negative, and vice versa).");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The float or integer (or sequence of floats or integers) to reverse the sign of.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("The input with its sign reversed."); }
};
RUNTIME_SHARD_TYPE(Math, Negate);

struct MaxOp {
  template <typename T> T apply(const T &lhs, const T &rhs) { return std::max(lhs, rhs); }
};
struct Max : public BinaryOperation<BasicBinaryOperation<MaxOp>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard compares the input with the value specified in the `Operand` parameter and outputs the larger value.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The first value to compare with."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The larger value between the input and the operand."); }
};
RUNTIME_SHARD_TYPE(Math, Max);

struct MinOp final {
  template <typename T> T apply(const T &lhs, const T &rhs) { return std::min(lhs, rhs); }
};
struct Min : public BinaryOperation<BasicBinaryOperation<MinOp>> {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard compares the input with the value specified in the `Operand` parameter and outputs the smaller value.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The first value to compare with."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The smaller value between the input and the operand."); }
};
RUNTIME_SHARD_TYPE(Math, Min);

struct PowOp final {
  template <typename T> T apply(const T &lhs, const T &rhs) { return std::pow(lhs, rhs); }
};
struct Pow : public BinaryOperation<BasicBinaryOperation<PowOp>> {
  static SHOptionalString help() {
    return SHCCSTR("This shard raises the input to the power of the exponent specified in the Operand parameter.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The base value to raise the power of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The result of raising the input to the power of the operand."); }
};
RUNTIME_SHARD_TYPE(Math, Pow);

struct LerpOp final {
  template <typename T> T apply(const T &lhs, const T &rhs, double t) { return T((double)lhs + (double(rhs) - double(lhs)) * t); }
};

struct ApplyLerp final {
  template <SHType ValueType> void apply(SHVarPayload &out, const SHVarPayload &a, const SHVarPayload &b, const SHFloat t) {
    typename PayloadTraits<ValueType>::ApplyBinary binary{};
    binary.template apply<LerpOp>(getPayloadContents<ValueType>(out), getPayloadContents<ValueType>(a),
                                  getPayloadContents<ValueType>(b), t);
  }
};

struct Lerp final {
  static SHTypesInfo inputTypes() { return CoreInfo::FloatType; }
  static SHTypesInfo outputTypes() { return Base::MathTypes; }

  static SHOptionalString inputHelp() { return SHCCSTR("The factor to interpolate between the start and end values."); }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The interpolated value between the start and end values based on the factor provided as input.");
  }

  static SHOptionalString help() {
    return SHCCSTR("Linearly interpolate between the start value specified in the `First` parameter and the end value specified "
                   "in the `Second` parameter based on the factor provided as input.");
  }

  static SHParametersInfo parameters() {
    static Parameters params{
        {"First", SHCCSTR("The start value"), BinaryBase::OnlyNumbers},
        {"Second", SHCCSTR("The end value"), BinaryBase::OnlyNumbers},
    };
    return params;
  }

  ParamVar _first;
  ParamVar _second;
  Var _result;

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _first = value;
      break;
    case 1:
      _second = value;
      break;
    default:
      throw std::out_of_range("index");
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _first;
    case 1:
      return _second;
    default:
      throw std::out_of_range("index");
    }
  }

  void warmup(SHContext *context) {
    _first.warmup(context);
    _second.warmup(context);
  }

  void cleanup(SHContext *context) {
    _first.cleanup();
    _second.cleanup();
  }

  SHTypeInfo compose(SHInstanceData &data) {
    SHType firstType{};
    SHType secondType{};
    firstType = _first.isVariable() ? findParamVarExposedTypeChecked(data, _first).exposedType.basicType : _first->valueType;
    secondType = _second.isVariable() ? findParamVarExposedTypeChecked(data, _second).exposedType.basicType : _second->valueType;

    if (firstType != secondType)
      throw ComposeError("Types should match");

    return SHTypeInfo{.basicType = firstType};
  }

  ALWAYS_INLINE SHVar activate(SHContext *context, const SHVar &input) {
    SHVar a = _first.get();
    SHVar b = _second.get();
    SHVar result{.valueType = a.valueType};
    dispatchType<DispatchType::NumberTypes>(a.valueType, ApplyLerp{}, result.payload, a.payload, b.payload,
                                            input.payload.floatValue);
    return result;
  }
};

struct ApplyClamp final {
  template <SHType ValueType>
  void apply(SHVarPayload &out, const SHVarPayload &input, const SHVarPayload &min, const SHVarPayload &max) {
    typename PayloadTraits<ValueType>::ApplyBinary binary{};
    binary.template apply<MaxOp>(getPayloadContents<ValueType>(out), getPayloadContents<ValueType>(input),
                                 getPayloadContents<ValueType>(min));
    binary.template apply<MinOp>(getPayloadContents<ValueType>(out), getPayloadContents<ValueType>(out),
                                 getPayloadContents<ValueType>(max));
  }
};

struct Clamp final {
  static SHTypesInfo inputTypes() { return Base::MathTypesNoSeq; }
  static SHTypesInfo outputTypes() { return Base::MathTypesNoSeq; }

  static SHOptionalString help() {
    return SHCCSTR("This shard ensures the input value falls within the specified range. If the value falls below the minimum, "
                   "the Min value is returned. If the value exceeds the maximum, the Max value is returned. Otherwise, the value "
                   "is returned unchanged.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The value to clamp."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The clamped value."); }

  static SHParametersInfo parameters() {
    static Parameters params{
        {"Min", SHCCSTR("The lower bound of the range"), BinaryBase::MathTypesOrVar},
        {"Max", SHCCSTR("The upper bound of the range"), BinaryBase::MathTypesOrVar},
    };
    return params;
  }

  ParamVar _first;
  ParamVar _second;
  Var _result;

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _first = value;
      break;
    case 1:
      _second = value;
      break;
    default:
      throw std::out_of_range("index");
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _first;
    case 1:
      return _second;
    default:
      throw std::out_of_range("index");
    }
  }

  void warmup(SHContext *context) {
    _first.warmup(context);
    _second.warmup(context);
  }

  void cleanup(SHContext *context) {
    _first.cleanup();
    _second.cleanup();
  }

  SHTypeInfo compose(SHInstanceData &data) {
    SHType firstType{};
    SHType secondType{};
    firstType = _first.isVariable() ? findParamVarExposedTypeChecked(data, _first).exposedType.basicType : _first->valueType;
    secondType = _second.isVariable() ? findParamVarExposedTypeChecked(data, _second).exposedType.basicType : _second->valueType;

    if (firstType != secondType || firstType != data.inputType.basicType)
      throw ComposeError("Types should match");

    return SHTypeInfo{.basicType = firstType};
  }

  ALWAYS_INLINE SHVar activate(SHContext *context, const SHVar &input) {
    SHVar a = _first.get();
    SHVar b = _second.get();
    SHVar result{.valueType = a.valueType};
    dispatchType<DispatchType::NumberTypes>(a.valueType, ApplyClamp{}, result.payload, input.payload, a.payload, b.payload);
    return result;
  }
};

struct Percentile {
  Percentile() { _percentile = Var{50.0}; }

  static SHTypesInfo inputTypes() { return CoreInfo::FloatSeqType; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }

  static SHOptionalString help() {
    return SHCCSTR("This shard calculates the percentile of the input value within the specified sequence.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The sequence of floats to calculate the percentile of."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The percentile of the input value within the specified sequence."); }

  PARAM_PARAMVAR(_percentile, "Percentile", "The percentile to calculate.", {CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_percentile));

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  std::vector<double> _sortedList;

  ALWAYS_INLINE SHVar activate(SHContext *context, const SHVar &input) {
    auto percentile = _percentile.get().payload.floatValue / 100.0;

    /*
    Sort the List
    Calculate the Index
    Interpolate if Necessary:
    */

    _sortedList.resize(input.payload.seqValue.len);
    for (size_t i = 0; i < input.payload.seqValue.len; ++i) {
      _sortedList[i] = input.payload.seqValue.elements[i].payload.floatValue;
    }

    std::sort(_sortedList.begin(), _sortedList.end());

    auto index = percentile * (_sortedList.size() - 1);

    // index is not an integer so we interpolate between the lower and upper bounds
    auto lower = _sortedList[static_cast<int>(std::floor(index))];
    auto upper = _sortedList[static_cast<int>(std::ceil(index))];

    auto threshold = lower + (upper - lower) * (index - std::floor(index));

    return Var{threshold};
  }
};

} // namespace Math
} // namespace shards

#endif // SH_CORE_SHARDS_MATH
