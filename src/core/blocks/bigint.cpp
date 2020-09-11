/* SPDX-License-Identifier: BSD 3-Clause "New" or "Revised" License */
/* Copyright © 2019-2020 Giovanni Petrantoni */

#ifndef CB_NO_BIGINT_BLOCKS

#include "chainblocks.hpp"
#include "shared.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;

namespace chainblocks {
namespace BigInt {
struct ToBigInt {
  std::vector<uint8_t> _buffer;

  static inline Types InputTypes{CoreInfo::IntType, CoreInfo::FloatType,
                                 CoreInfo::StringType};
  static CBTypesInfo inputTypes() { return InputTypes; }
  static CBTypesInfo outputTypes() { return CoreInfo::BytesType; }

  CBVar activate(CBContext *context, const CBVar &input) {
    _buffer.clear();
    cpp_int bi;
    switch (input.valueType) {
    case Int: {
      bi = input.payload.intValue;
    } break;
    case Float: {
      bi = cpp_int(input.payload.floatValue);
    } break;
    case String: {
      bi = cpp_int(input.payload.stringValue);
    } break;
    default: {
      throw ActivationError("Invalid input type");
    }
    }
    export_bits(bi, std::back_inserter(_buffer), 8);
    return Var(&_buffer.front(), _buffer.size());
  }
};

struct BigOperandBase {
  std::vector<uint8_t> _buffer;

  static CBTypesInfo inputTypes() { return CoreInfo::BytesType; }
  static CBTypesInfo outputTypes() { return CoreInfo::BytesType; }

  CBParametersInfo parameters() {
    static Parameters params{{"Operand",
                              "The bytes variable representing the operand",
                              {CoreInfo::BytesVarType}}};
    return params;
  }

  ParamVar _op{};

  void setParam(int index, CBVar value) { _op = value; }

  CBVar getParam(int index) { return _op; }

  void cleanup() { _op.cleanup(); }

  void warmup(CBContext *context) { _op.warmup(context); }

  const CBVar &getOperand() {
    CBVar &op = _op.get();
    if (op.valueType == None) {
      throw ActivationError("Operand is None, should be valid bigint bytes");
    }
    return op;
  }
};

#define BIGINT_MATH_OP(__NAME__, __OP__)                                       \
  struct __NAME__ : public BigOperandBase {                                    \
    CBVar activate(CBContext *context, const CBVar &input) {                   \
      _buffer.clear();                                                         \
      cpp_int bia;                                                             \
      import_bits(bia, input.payload.bytesValue,                               \
                  input.payload.bytesValue + input.payload.bytesSize);         \
      auto op = getOperand();                                                  \
      cpp_int bib;                                                             \
      import_bits(bib, op.payload.bytesValue,                                  \
                  op.payload.bytesValue + op.payload.bytesSize);               \
      cpp_int bres = bia __OP__ bib;                                           \
      export_bits(bres, std::back_inserter(_buffer), 8);                       \
      return Var(&_buffer.front(), _buffer.size());                            \
    }                                                                          \
  }

BIGINT_MATH_OP(Add, +);
BIGINT_MATH_OP(Subtract, -);
BIGINT_MATH_OP(Multiply, *);
BIGINT_MATH_OP(Divide, /);
BIGINT_MATH_OP(Xor, ^);
BIGINT_MATH_OP(And, &);
BIGINT_MATH_OP(Or, |);
BIGINT_MATH_OP(Mod, %);

#define BIGINT_LOGIC_OP(__NAME__, __OP__)                                      \
  struct __NAME__ : public BigOperandBase {                                    \
    static CBTypesInfo outputTypes() { return CoreInfo::BoolType; }            \
    CBVar activate(CBContext *context, const CBVar &input) {                   \
      _buffer.clear();                                                         \
      cpp_int bia;                                                             \
      import_bits(bia, input.payload.bytesValue,                               \
                  input.payload.bytesValue + input.payload.bytesSize);         \
      auto op = getOperand();                                                  \
      cpp_int bib;                                                             \
      import_bits(bib, op.payload.bytesValue,                                  \
                  op.payload.bytesValue + op.payload.bytesSize);               \
      bool bres = bia __OP__ bib;                                              \
      return Var(bres);                                                        \
    }                                                                          \
  }

BIGINT_LOGIC_OP(Is, ==);
BIGINT_LOGIC_OP(IsNot, !=);
BIGINT_LOGIC_OP(IsMore, >);
BIGINT_LOGIC_OP(IsLess, <);
BIGINT_LOGIC_OP(IsMoreEqual, >=);
BIGINT_LOGIC_OP(IsLessEqual, <=);

#define BIGINT_BINARY_OP(__NAME__, __OP__)                                     \
  struct __NAME__ : public BigOperandBase {                                    \
    CBVar activate(CBContext *context, const CBVar &input) {                   \
      _buffer.clear();                                                         \
      cpp_int bia;                                                             \
      import_bits(bia, input.payload.bytesValue,                               \
                  input.payload.bytesValue + input.payload.bytesSize);         \
      auto op = getOperand();                                                  \
      cpp_int bib;                                                             \
      import_bits(bib, op.payload.bytesValue,                                  \
                  op.payload.bytesValue + op.payload.bytesSize);               \
      cpp_int bres = __OP__(bia, bib);                                         \
      export_bits(bres, std::back_inserter(_buffer), 8);                       \
      return Var(&_buffer.front(), _buffer.size());                            \
    }                                                                          \
  }

BIGINT_BINARY_OP(Min, std::min);
BIGINT_BINARY_OP(Max, std::max);

struct ShiftBase {
  ParamVar _shift{Var(0)};

  void setParam(int index, CBVar value) { _shift = value; }

  CBVar getParam(int index) { return _shift; }

  void cleanup() { _shift.cleanup(); }

  void warmup(CBContext *context) { _shift.warmup(context); }
};

struct Shift : public ShiftBase {
  std::vector<uint8_t> _buffer;

  static CBTypesInfo inputTypes() { return CoreInfo::BytesType; }
  static CBTypesInfo outputTypes() { return CoreInfo::BytesType; }

  CBParametersInfo parameters() {
    static Parameters params{
        {"By",
         "The shift is of the decimal point, i.e. of powers of ten, and is to "
         "the left if n is negative or to the right if n is positive.",
         {CoreInfo::IntType, CoreInfo::IntVarType}}};
    return params;
  }

  CBVar activate(CBContext *context, const CBVar &input) {
    _buffer.clear();
    cpp_int bi;
    import_bits(bi, input.payload.bytesValue,
                input.payload.bytesValue + input.payload.bytesSize);
    cpp_dec_float_100 bf(bi);

    cpp_dec_float_100 bshift(_shift.get().payload.intValue);
    bshift = pow(cpp_dec_float_100(10), bshift);

    auto bres = cpp_int(bf * bshift);

    export_bits(bres, std::back_inserter(_buffer), 8);
    return Var(&_buffer.front(), _buffer.size());
  }
};

struct ToFloat : public ShiftBase {
  static CBTypesInfo inputTypes() { return CoreInfo::BytesType; }
  static CBTypesInfo outputTypes() { return CoreInfo::FloatType; }

  CBParametersInfo parameters() {
    static Parameters params{
        {"ShiftedBy",
         "The shift is of the decimal point, i.e. of powers of ten, and is to "
         "the left if n is negative or to the right if n is positive.",
         {CoreInfo::IntType}}};
    return params;
  }

  CBVar activate(CBContext *context, const CBVar &input) {
    cpp_int bi;
    import_bits(bi, input.payload.bytesValue,
                input.payload.bytesValue + input.payload.bytesSize);
    cpp_dec_float_100 bf(bi);

    cpp_dec_float_100 bshift(_shift.get().payload.intValue);
    bshift = pow(cpp_dec_float_100(10), bshift);

    auto bres = bf * bshift;

    return Var(bres.convert_to<double>());
  }
};

struct ToString {
  std::string _buffer;

  static CBTypesInfo inputTypes() { return CoreInfo::BytesType; }
  static CBTypesInfo outputTypes() { return CoreInfo::StringType; }

  CBVar activate(CBContext *context, const CBVar &input) {
    cpp_int bi;
    import_bits(bi, input.payload.bytesValue,
                input.payload.bytesValue + input.payload.bytesSize);
    _buffer = bi.str();
    return Var(_buffer);
  }
};

void registerBlocks() {
  REGISTER_CBLOCK("BigInt", ToBigInt);
  REGISTER_CBLOCK("BigInt.Add", Add);
  REGISTER_CBLOCK("BigInt.Subtract", Subtract);
  REGISTER_CBLOCK("BigInt.Multiply", Multiply);
  REGISTER_CBLOCK("BigInt.Divide", Divide);
  REGISTER_CBLOCK("BigInt.Xor", Xor);
  REGISTER_CBLOCK("BigInt.And", And);
  REGISTER_CBLOCK("BigInt.Or", Or);
  REGISTER_CBLOCK("BigInt.Mod", Mod);
  REGISTER_CBLOCK("BigInt.Shift", Shift);
  REGISTER_CBLOCK("BigInt.ToFloat", ToFloat);
  REGISTER_CBLOCK("BigInt.ToString", ToString);
  REGISTER_CBLOCK("BigInt.Is", Is);
  REGISTER_CBLOCK("BigInt.IsNot", IsNot);
  REGISTER_CBLOCK("BigInt.IsMore", IsMore);
  REGISTER_CBLOCK("BigInt.IsLess", IsLess);
  REGISTER_CBLOCK("BigInt.IsMoreEqual", IsMoreEqual);
  REGISTER_CBLOCK("BigInt.IsLessEqual", IsLessEqual);
  REGISTER_CBLOCK("BigInt.Min", Min);
  REGISTER_CBLOCK("BigInt.Max", Max);
}
} // namespace BigInt
} // namespace chainblocks

#else
namespace chainblocks {
namespace BigInt {
void registerBlocks() {}
} // namespace BigInt
} // namespace chainblocks
#endif