/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#ifndef SH_SHARDS_HPP
#define SH_SHARDS_HPP

#include "shards.h"
#include <cstring> // memcpy
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace shards {
constexpr uint32_t CoreCC = 'frag'; // FourCC = 1718772071 = 0x66726167

class SHException : public std::exception {
public:
  explicit SHException(std::string_view msg) : errorMessage(msg) {}

  [[nodiscard]] const char *what() const noexcept override { return errorMessage.data(); }

private:
  std::string errorMessage;
};

class InvalidParameterIndex : public SHException {
public:
  explicit InvalidParameterIndex() : SHException("Invalid parameter index") {}
};

class ActivationError : public SHException {
public:
  explicit ActivationError(std::string_view msg) : SHException(msg) {}
};

class WarmupError : public SHException {
public:
  explicit WarmupError(std::string_view msg) : SHException(msg) {}
};

class ComposeError : public SHException {
public:
  explicit ComposeError(std::string_view msg, bool fatal = true) : SHException(msg), fatal(fatal) {}

  bool triggerFailure() const { return fatal; }

private:
  bool fatal;
};

class InvalidVarTypeError : public SHException {
public:
  explicit InvalidVarTypeError(std::string_view msg) : SHException(msg) {}
};

Shard *createShard(std::string_view name);

struct Type {
  Type() : _type({SHType::None}) {}

  Type(SHTypeInfo type) : _type(type) {}

  Type(const Type &obj) { _type = obj._type; }

  Type &operator=(Type other) {
    _type = other._type;
    return *this;
  }

  Type &operator=(SHTypeInfo other) {
    _type = other;
    return *this;
  }

  operator SHTypesInfo() const {
    SHTypesInfo res{const_cast<SHTypeInfo *>(&_type), 1, 0};
    return res;
  }

  operator SHTypeInfo() const { return _type; }

  static Type SeqOf(SHTypesInfo types) {
    Type res;
    res._type = {SHType::Seq, {.seqTypes = types}};
    return res;
  }

  static Type SeqOf(SHTypesInfo types, uint32_t fixedLen) {
    Type res;
    res._type = {SHType::Seq, {.seqTypes = types}, fixedLen};
    return res;
  }

  static Type VariableOf(SHTypesInfo types) {
    Type res;
    res._type = {SHType::ContextVar, {.contextVarTypes = types}};
    return res;
  }

  static Type Enum(int32_t vendorId, int32_t typeId) {
    Type res;
    res._type = {SHType::Enum, {.enumeration = {.vendorId = vendorId, .typeId = typeId}}};
    return res;
  }

  static Type Object(int32_t vendorId, int32_t typeId) {
    Type res;
    res._type = {SHType::Object, {.object = {.vendorId = vendorId, .typeId = typeId}}};
    return res;
  }

  static Type TableOf(SHTypesInfo types) {
    Type res;
    res._type = {SHType::Table, {.table = {.types = types}}};
    return res;
  }

  template <size_t N> static Type TableOf(SHTypesInfo types, const std::array<SHString, N> &keys) {
    Type res;
    auto &k = const_cast<std::array<SHString, N> &>(keys);
    res._type = {SHType::Table, {.table = {.keys = {&k[0], uint32_t(k.size()), 0}, .types = types}}};
    return res;
  }

private:
  SHTypeInfo _type;
};

struct Types {
  std::vector<SHTypeInfo> _types;

  Types() {}

  Types(std::initializer_list<SHTypeInfo> types, bool addSelf = false) : _types(types) {
    if (addSelf) {
      auto idx = _types.size();
      // make sure the memory will stay valid
      _types.resize(idx + 1);
      // add self
      _types[idx] = Type::SeqOf(SHTypesInfo(*this));
      _types[idx].recursiveSelf = true;
    }
  }

  Types(const Types &others, std::initializer_list<SHTypeInfo> types, bool addSelf = false) {
    for (auto &type : others._types) {
      _types.push_back(type);
    }
    for (auto &type : types) {
      _types.push_back(type);
    }
    if (addSelf) {
      auto idx = _types.size();
      // make sure the memory will stay valid
      _types.resize(idx + 1);
      // add self
      _types[idx] = Type::SeqOf(SHTypesInfo(*this));
      _types[idx].recursiveSelf = true;
    }
  }

  Types(const std::vector<SHTypeInfo> &types) { _types = types; }

  Types &operator=(const std::vector<SHTypeInfo> &types) {
    _types = types;
    return *this;
  }

  operator SHTypesInfo() {
    SHTypesInfo res{&_types[0], (uint32_t)_types.size(), 0};
    return res;
  }
};

struct ParameterInfo {
  SHString _name;
  SHOptionalString _help;
  Types _types;

  ParameterInfo(SHString name, SHOptionalString help, Types types) : _name(name), _help(help), _types(types) {}
  ParameterInfo(SHString name, Types types) : _name(name), _help({}), _types(types) {}

  operator SHParameterInfo() {
    SHParameterInfo res{_name, _help, _types};
    return res;
  }
};

struct Parameters {
  std::vector<ParameterInfo> _infos{};
  std::vector<SHParameterInfo> _pinfos{};

  Parameters() = default;

  Parameters(const Parameters &others, const std::vector<ParameterInfo> &infos) {
    for (auto &info : others._infos) {
      _infos.push_back(info);
    }
    for (auto &info : infos) {
      _infos.push_back(info);
    }
    // update the C type cache
    for (auto &info : _infos) {
      _pinfos.push_back(info);
    }
  }

  // THE CONSTRUCTORS UNDER ARE UNSAFE
  // static inline is nice but it's likely unordered!
  // won't likely work with some compilers
  // and if linking dlls!

  Parameters(const Parameters &others, std::initializer_list<ParameterInfo> infos) {
    for (auto &info : others._infos) {
      _infos.push_back(info);
    }
    for (auto &info : infos) {
      _infos.push_back(info);
    }
    // update the C type cache
    for (auto &info : _infos) {
      _pinfos.push_back(info);
    }
  }

  Parameters(std::initializer_list<ParameterInfo> infos, const Parameters &others) {
    for (auto &info : infos) {
      _infos.push_back(info);
    }
    for (auto &info : others._infos) {
      _infos.push_back(info);
    }
    // update the C type cache
    for (auto &info : _infos) {
      _pinfos.push_back(info);
    }
  }

  Parameters(std::initializer_list<ParameterInfo> infos) : _infos(infos) {
    for (auto &info : _infos) {
      _pinfos.push_back(info);
    }
  }

  operator SHParametersInfo() {
    if (_pinfos.empty())
      return SHParametersInfo{nullptr, 0, 0};
    else
      return SHParametersInfo{&_pinfos.front(), (uint32_t)_pinfos.size(), 0};
  }
};

// used to explicitly specialize, hinting compiler
// mostly used internally for math shards
#define SH_PAYLOAD_CTORS0(SHPAYLOAD_TYPE, __inner_type__, __item__)                                            \
  constexpr static size_t Width{1};                                                                            \
  SHPAYLOAD_TYPE() : SHVarPayload() {}                                                                         \
  template <typename NUMBER> SHPAYLOAD_TYPE(NUMBER n) : SHVarPayload() { this->__item__ = __inner_type__(n); } \
  template <typename NUMBER> SHPAYLOAD_TYPE(std::initializer_list<NUMBER> l) : SHVarPayload() {                \
    const NUMBER *p = l.begin();                                                                               \
    this->__item__ = __inner_type__(p[0]);                                                                     \
  }

#define SH_PAYLOAD_CTORS1(SHPAYLOAD_TYPE, __inner_type__, __item__, _width_)                    \
  constexpr static size_t Width{_width_};                                                       \
  SHPAYLOAD_TYPE() : SHVarPayload() {}                                                          \
  template <typename NUMBER> SHPAYLOAD_TYPE(NUMBER n) : SHVarPayload() {                        \
    for (size_t i = 0; i < _width_; ++i)                                                        \
      this->__item__[i] = __inner_type__(n);                                                    \
  }                                                                                             \
  template <typename NUMBER> SHPAYLOAD_TYPE(std::initializer_list<NUMBER> l) : SHVarPayload() { \
    const NUMBER *p = l.begin();                                                                \
    for (size_t i = 0; i < l.size(); ++i) {                                                     \
      this->__item__[i] = __inner_type__(p[i]);                                                 \
    }                                                                                           \
  }

#define SH_PAYLOAD_MATH_OPS(SHPAYLOAD_TYPE, __item__)                            \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator+(const SHPAYLOAD_TYPE &b) const { \
    SHPAYLOAD_TYPE res;                                                          \
    res.__item__ = __item__ + b.__item__;                                        \
    return res;                                                                  \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator-(const SHPAYLOAD_TYPE &b) const { \
    SHPAYLOAD_TYPE res;                                                          \
    res.__item__ = __item__ - b.__item__;                                        \
    return res;                                                                  \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator*(const SHPAYLOAD_TYPE &b) const { \
    SHPAYLOAD_TYPE res;                                                          \
    res.__item__ = __item__ * b.__item__;                                        \
    return res;                                                                  \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator/(const SHPAYLOAD_TYPE &b) const { \
    SHPAYLOAD_TYPE res;                                                          \
    res.__item__ = __item__ / b.__item__;                                        \
    return res;                                                                  \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE &operator+=(const SHPAYLOAD_TYPE &b) {     \
    __item__ += b.__item__;                                                      \
    return *this;                                                                \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE &operator-=(const SHPAYLOAD_TYPE &b) {     \
    __item__ -= b.__item__;                                                      \
    return *this;                                                                \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE &operator*=(const SHPAYLOAD_TYPE &b) {     \
    __item__ *= b.__item__;                                                      \
    return *this;                                                                \
  }                                                                              \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE &operator/=(const SHPAYLOAD_TYPE &b) {     \
    __item__ /= b.__item__;                                                      \
    return *this;                                                                \
  }

#define SH_PAYLOAD_MATH_OPS_INT(SHPAYLOAD_TYPE, __item__)                         \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator^(const SHPAYLOAD_TYPE &b) const {  \
    SHPAYLOAD_TYPE res;                                                           \
    res.__item__ = __item__ ^ b.__item__;                                         \
    return res;                                                                   \
  }                                                                               \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator&(const SHPAYLOAD_TYPE &b) const {  \
    SHPAYLOAD_TYPE res;                                                           \
    res.__item__ = __item__ & b.__item__;                                         \
    return res;                                                                   \
  }                                                                               \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator|(const SHPAYLOAD_TYPE &b) const {  \
    SHPAYLOAD_TYPE res;                                                           \
    res.__item__ = __item__ | b.__item__;                                         \
    return res;                                                                   \
  }                                                                               \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator%(const SHPAYLOAD_TYPE &b) const {  \
    SHPAYLOAD_TYPE res;                                                           \
    res.__item__ = __item__ % b.__item__;                                         \
    return res;                                                                   \
  }                                                                               \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator<<(const SHPAYLOAD_TYPE &b) const { \
    SHPAYLOAD_TYPE res;                                                           \
    res.__item__ = __item__ << b.__item__;                                        \
    return res;                                                                   \
  }                                                                               \
  ALWAYS_INLINE inline SHPAYLOAD_TYPE operator>>(const SHPAYLOAD_TYPE &b) const { \
    SHPAYLOAD_TYPE res;                                                           \
    res.__item__ = __item__ >> b.__item__;                                        \
    return res;                                                                   \
  }

#define SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, __op__)             \
  ALWAYS_INLINE static inline SHPAYLOAD_TYPE __op__(const SHPAYLOAD_TYPE &x) { \
    SHPAYLOAD_TYPE res;                                                        \
    for (size_t i = 0; i < SHPAYLOAD_TYPE::Width; i++) {                       \
      res.__item__[i] = __builtin_##__op__(x.__item__[i]);                     \
    }                                                                          \
    return res;                                                                \
  }

#define SH_PAYLOAD_MATH_OPS_FLOAT(SHPAYLOAD_TYPE, __item__) \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, sqrt); \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, log);  \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, sin);  \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, cos);  \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, exp);  \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, tanh); \
  SH_PAYLOAD_MATH_OP_FLOAT(SHPAYLOAD_TYPE, __item__, fabs);

#define SH_PAYLOAD_MATH_POW_FLOAT(SHPAYLOAD_TYPE, __item__)                                            \
  template <typename P> ALWAYS_INLINE static inline SHPAYLOAD_TYPE pow(const SHPAYLOAD_TYPE &x, P p) { \
    SHPAYLOAD_TYPE res;                                                                                \
    for (size_t i = 0; i < SHPAYLOAD_TYPE::Width; i++) {                                               \
      res.__item__[i] = __builtin_pow(x.__item__[i], p);                                               \
    }                                                                                                  \
    return res;                                                                                        \
  }

#define SH_PAYLOAD_MATH_OPS_SIMPLE(SHPAYLOAD_TYPE, __item__)                                             \
  ALWAYS_INLINE inline bool operator<=(const SHPAYLOAD_TYPE &b) const { return __item__ <= b.__item__; } \
  ALWAYS_INLINE inline bool operator>=(const SHPAYLOAD_TYPE &b) const { return __item__ >= b.__item__; } \
  ALWAYS_INLINE inline bool operator==(const SHPAYLOAD_TYPE &b) const { return __item__ == b.__item__; } \
  ALWAYS_INLINE inline bool operator!=(const SHPAYLOAD_TYPE &b) const { return __item__ != b.__item__; } \
  ALWAYS_INLINE inline bool operator>(const SHPAYLOAD_TYPE &b) const { return __item__ > b.__item__; }   \
  ALWAYS_INLINE inline bool operator<(const SHPAYLOAD_TYPE &b) const { return __item__ < b.__item__; }

struct IntVarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS0(IntVarPayload, int64_t, intValue);
  SH_PAYLOAD_MATH_OPS(IntVarPayload, intValue);
  SH_PAYLOAD_MATH_OPS_SIMPLE(IntVarPayload, intValue);
  SH_PAYLOAD_MATH_OPS_INT(IntVarPayload, intValue);
};
struct Int2VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Int2VarPayload, int64_t, int2Value, 2);
  SH_PAYLOAD_MATH_OPS(Int2VarPayload, int2Value);
  SH_PAYLOAD_MATH_OPS_INT(Int2VarPayload, int2Value);
};
struct Int3VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Int3VarPayload, int32_t, int3Value, 3);
  SH_PAYLOAD_MATH_OPS(Int3VarPayload, int3Value);
  SH_PAYLOAD_MATH_OPS_INT(Int3VarPayload, int3Value);
};
struct Int4VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Int4VarPayload, int32_t, int4Value, 4);
  SH_PAYLOAD_MATH_OPS(Int4VarPayload, int4Value);
  SH_PAYLOAD_MATH_OPS_INT(Int4VarPayload, int4Value);
};
struct Int8VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Int8VarPayload, int16_t, int8Value, 8);
  SH_PAYLOAD_MATH_OPS(Int8VarPayload, int8Value);
  SH_PAYLOAD_MATH_OPS_INT(Int8VarPayload, int8Value);
};
struct Int16VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Int16VarPayload, int8_t, int16Value, 16);
  SH_PAYLOAD_MATH_OPS(Int16VarPayload, int16Value);
  SH_PAYLOAD_MATH_OPS_INT(Int16VarPayload, int16Value);
};
struct FloatVarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS0(FloatVarPayload, double, floatValue);
  SH_PAYLOAD_MATH_OPS(FloatVarPayload, floatValue);
  SH_PAYLOAD_MATH_OPS_SIMPLE(FloatVarPayload, floatValue);
};
struct Float2VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Float2VarPayload, double, float2Value, 2);
  SH_PAYLOAD_MATH_OPS(Float2VarPayload, float2Value);
  SH_PAYLOAD_MATH_OPS_FLOAT(Float2VarPayload, float2Value);
  SH_PAYLOAD_MATH_POW_FLOAT(Float2VarPayload, float2Value);
};
struct Float3VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Float3VarPayload, float, float3Value, 3);
  SH_PAYLOAD_MATH_OPS(Float3VarPayload, float3Value);
  SH_PAYLOAD_MATH_OPS_FLOAT(Float3VarPayload, float3Value);
  SH_PAYLOAD_MATH_POW_FLOAT(Float3VarPayload, float3Value);
};
struct Float4VarPayload : public SHVarPayload {
  SH_PAYLOAD_CTORS1(Float4VarPayload, float, float4Value, 4);
  SH_PAYLOAD_MATH_OPS(Float4VarPayload, float4Value);
  SH_PAYLOAD_MATH_OPS_FLOAT(Float4VarPayload, float4Value);
  SH_PAYLOAD_MATH_POW_FLOAT(Float4VarPayload, float4Value);
};

// forward declare this as we use it in Var
class Wire;

struct Var : public SHVar {
  Var(const IntVarPayload &p) : SHVar() {
    this->valueType = SHType::Int;
    this->payload.intValue = p.intValue;
  }

  Var &operator=(const IntVarPayload &p) {
    this->valueType = SHType::Int;
    this->payload.intValue = p.intValue;
    return *this;
  }

  Var(const Int2VarPayload &p) : SHVar() {
    this->valueType = SHType::Int2;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Int2VarPayload &p) {
    this->valueType = SHType::Int2;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const Int3VarPayload &p) : SHVar() {
    this->valueType = SHType::Int3;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Int3VarPayload &p) {
    this->valueType = SHType::Int3;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const Int4VarPayload &p) : SHVar() {
    this->valueType = SHType::Int4;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Int4VarPayload &p) {
    this->valueType = SHType::Int4;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const Int8VarPayload &p) : SHVar() {
    this->valueType = SHType::Int8;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Int8VarPayload &p) {
    this->valueType = SHType::Int8;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const Int16VarPayload &p) : SHVar() {
    this->valueType = SHType::Int16;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Int16VarPayload &p) {
    this->valueType = SHType::Int16;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const FloatVarPayload &p) : SHVar() {
    this->valueType = SHType::Float;
    this->payload.floatValue = p.floatValue;
  }

  Var &operator=(const FloatVarPayload &p) {
    this->valueType = SHType::Float;
    this->payload.floatValue = p.floatValue;
    return *this;
  }

  Var(const Float2VarPayload &p) : SHVar() {
    this->valueType = SHType::Float2;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Float2VarPayload &p) {
    this->valueType = SHType::Float2;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const Float3VarPayload &p) : SHVar() {
    this->valueType = SHType::Float3;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Float3VarPayload &p) {
    this->valueType = SHType::Float3;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  Var(const Float4VarPayload &p) : SHVar() {
    this->valueType = SHType::Float4;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
  }

  Var &operator=(const Float4VarPayload &p) {
    this->valueType = SHType::Float4;
    memcpy(&this->payload, &p, sizeof(SHVarPayload));
    return *this;
  }

  constexpr Var() : SHVar() {}

  explicit Var(const SHVar &other) { memcpy((void *)this, (void *)&other, sizeof(SHVar)); }

  Var &operator=(const SHVar &other) {
    memcpy((void *)this, (void *)&other, sizeof(SHVar));
    return *this;
  }

  bool isNone() const { return valueType == SHType::None; }

  explicit operator bool() const {
    if (valueType != Bool) {
      throw InvalidVarTypeError("Invalid variable casting! expected Bool");
    }
    return payload.boolValue;
  }

  explicit operator int() const {
    if (valueType != Int) {
      throw InvalidVarTypeError("Invalid variable casting! expected Int");
    }
    return static_cast<int>(payload.intValue);
  }

  explicit operator uintptr_t() const {
    if (valueType != Int) {
      throw InvalidVarTypeError("Invalid variable casting! expected Int");
    }
    return static_cast<uintptr_t>(payload.intValue);
  }

  explicit operator int16_t() const {
    if (valueType != Int) {
      throw InvalidVarTypeError("Invalid variable casting! expected Int");
    }
    return static_cast<int16_t>(payload.intValue);
  }

  explicit operator uint8_t() const {
    if (valueType != Int) {
      throw InvalidVarTypeError("Invalid variable casting! expected Int");
    }
    return static_cast<uint8_t>(payload.intValue);
  }

  explicit operator int64_t() const {
    if (valueType != Int) {
      throw InvalidVarTypeError("Invalid variable casting! expected Int");
    }
    return payload.intValue;
  }

  explicit operator float() const {
    if (valueType == Float) {
      return static_cast<float>(payload.floatValue);
    } else if (valueType == Int) {
      return static_cast<float>(payload.intValue);
    } else {
      throw InvalidVarTypeError("Invalid variable casting! expected Float");
    }
  }

  explicit operator double() const {
    if (valueType == Float) {
      return payload.floatValue;
    } else if (valueType == Int) {
      return static_cast<double>(payload.intValue);
    } else {
      throw InvalidVarTypeError("Invalid variable casting! expected Float");
    }
  }

  explicit operator const char *() const {
    if (valueType == SHType::String || valueType == SHType::Path || valueType == SHType::ContextVar) {
      return payload.stringValue;
    } else {
      throw InvalidVarTypeError("Invalid variable casting! expected String/Path/ContextVar");
    }
  }

  template <typename T> void intoVector(std::vector<T> &outVec) const {
    if (valueType != Seq) {
      throw InvalidVarTypeError("Invalid variable casting! expected Seq");
    }
    outVec.resize(payload.seqValue.len);
    for (uint32_t i = 0; i < payload.seqValue.len; i++) {
      outVec[i] = T(*reinterpret_cast<Var *>(&payload.seqValue.elements[i]));
    }
  }

  template <typename T> explicit operator std::vector<T>() const {
    std::vector<T> result;
    intoVector(result);
    return result;
  }

  explicit operator std::vector<Var>() const {
    std::vector<Var> result;
    intoVector(result);
    return result;
  }

  constexpr static SHVar Empty{};
  constexpr static SHVar True{{true}, SHType::Bool};
  constexpr static SHVar False{{false}, SHType::Bool};
  // this is a special one used to flag default value in Wire/Shards C++ DSL
  constexpr static SHVar Any{{false}, SHType::Any};

  template <typename T> static Var Object(T valuePtr, uint32_t objectVendorId, uint32_t objectTypeId) {
    Var res;
    res.valueType = SHType::Object;
    res.payload.objectValue = valuePtr;
    res.payload.objectVendorId = objectVendorId;
    res.payload.objectTypeId = objectTypeId;
    return res;
  }

  template <typename T> static Var Enum(T value, uint32_t enumVendorId, uint32_t enumTypeId) {
    Var res;
    res.valueType = SHType::Enum;
    res.payload.enumValue = SHEnum(value);
    res.payload.enumVendorId = enumVendorId;
    res.payload.enumTypeId = enumTypeId;
    return res;
  }

  Var(uint8_t *ptr, uint32_t size) : SHVar() {
    valueType = Bytes;
    payload.bytesSize = size;
    payload.bytesValue = ptr;
  }

  Var(const std::vector<uint8_t> &bytes) : SHVar() {
    valueType = Bytes;
    const auto size = bytes.size();
    if (size > UINT32_MAX)
      throw SHException("std::vector<uint8_t> to Var size exceeded uint32 maximum");
    payload.bytesSize = uint32_t(size);
    payload.bytesValue = const_cast<uint8_t *>(bytes.data());
  }

  Var(uint8_t *data, uint16_t width, uint16_t height, uint8_t channels, uint8_t flags = 0) : SHVar() {
    valueType = SHType::Image;
    payload.imageValue.width = width;
    payload.imageValue.height = height;
    payload.imageValue.channels = channels;
    payload.imageValue.flags = flags;
    payload.imageValue.data = data;
  }

  explicit Var(int src) : SHVar() {
    valueType = Int;
    payload.intValue = src;
  }

  explicit Var(uint8_t src) : SHVar() {
    valueType = Int;
    payload.intValue = int64_t(src);
  }

  explicit Var(char src) : SHVar() {
    valueType = Int;
    payload.intValue = int64_t(src);
  }

  explicit Var(unsigned int src) : SHVar() {
    valueType = Int;
    payload.intValue = int64_t(src);
  }

  explicit Var(int a, int b) : SHVar() {
    valueType = Int2;
    payload.int2Value[0] = a;
    payload.int2Value[1] = b;
  }

  explicit Var(int a, int b, int c) : SHVar() {
    valueType = Int3;
    payload.int3Value[0] = a;
    payload.int3Value[1] = b;
    payload.int3Value[2] = c;
  }

  explicit Var(int a, int b, int c, int d) : SHVar() {
    valueType = Int4;
    payload.int4Value[0] = a;
    payload.int4Value[1] = b;
    payload.int4Value[2] = c;
    payload.int4Value[3] = d;
  }

  explicit Var(int16_t a, int16_t b, int16_t c, int16_t d, int16_t e, int16_t f, int16_t g, int16_t h) : SHVar() {
    valueType = Int8;
    payload.int8Value[0] = a;
    payload.int8Value[1] = b;
    payload.int8Value[2] = c;
    payload.int8Value[3] = d;
    payload.int8Value[4] = e;
    payload.int8Value[5] = f;
    payload.int8Value[6] = g;
    payload.int8Value[7] = h;
  }

  explicit Var(int8_t a, int8_t b, int8_t c, int8_t d, int8_t e, int8_t f, int8_t g, int8_t h, int8_t i, int8_t j, int8_t k,
               int8_t l, int8_t m, int8_t n, int8_t o, int8_t p)
      : SHVar() {
    valueType = Int16;
    payload.int16Value[0] = a;
    payload.int16Value[1] = b;
    payload.int16Value[2] = c;
    payload.int16Value[3] = d;
    payload.int16Value[4] = e;
    payload.int16Value[5] = f;
    payload.int16Value[6] = g;
    payload.int16Value[7] = h;
    payload.int16Value[8] = i;
    payload.int16Value[9] = j;
    payload.int16Value[10] = k;
    payload.int16Value[11] = l;
    payload.int16Value[12] = m;
    payload.int16Value[13] = n;
    payload.int16Value[14] = o;
    payload.int16Value[15] = p;
  }

  explicit Var(int64_t a, int64_t b) : SHVar() {
    valueType = Int2;
    payload.int2Value[0] = a;
    payload.int2Value[1] = b;
  }

  explicit Var(double a, double b) : SHVar() {
    valueType = Float2;
    payload.float2Value[0] = a;
    payload.float2Value[1] = b;
  }

  explicit Var(double a, double b, double c) : SHVar() {
    valueType = Float3;
    payload.float3Value[0] = a;
    payload.float3Value[1] = b;
    payload.float3Value[2] = c;
  }

  explicit Var(double a, double b, double c, double d) : SHVar() {
    valueType = Float4;
    payload.float4Value[0] = a;
    payload.float4Value[1] = b;
    payload.float4Value[2] = c;
    payload.float4Value[3] = d;
  }

  explicit Var(float a, float b) : SHVar() {
    valueType = Float2;
    payload.float2Value[0] = a;
    payload.float2Value[1] = b;
  }

  explicit Var(double src) : SHVar() {
    valueType = Float;
    payload.floatValue = src;
  }

  explicit Var(bool src) : SHVar() {
    valueType = Bool;
    payload.boolValue = src;
  }

  explicit Var(SHSeq seq) : SHVar() {
    valueType = Seq;
    payload.seqValue = seq;
  }

  explicit Var(SHAudio audio) : SHVar() {
    valueType = SHType::Audio;
    payload.audioValue = audio;
  }

  explicit Var(SHWireRef wire) : SHVar() {
    valueType = SHType::Wire;
    payload.wireValue = wire;
  }

  explicit Var(const std::shared_ptr<SHWire> &wire) : SHVar() {
    // INTERNAL USE ONLY, ABI LIKELY NOT COMPATIBLE!
    valueType = SHType::Wire;
    payload.wireValue = reinterpret_cast<SHWireRef>(&const_cast<std::shared_ptr<SHWire> &>(wire));
  }

  explicit Var(Shard *shard) : SHVar() {
    // mostly internal use only, anyway use with care!
    valueType = SHType::ShardRef;
    payload.shardValue = shard;
  }

  explicit Var(SHImage img) : SHVar() {
    valueType = Image;
    payload.imageValue = img;
  }

  explicit Var(uint64_t src) : SHVar() {
    valueType = Int;
    payload.intValue = src;
  }

  explicit Var(int64_t src) : SHVar() {
    valueType = Int;
    payload.intValue = src;
  }

  explicit Var(const char *src, size_t len = 0) : SHVar() {
    valueType = SHType::String;
    payload.stringValue = src;
    payload.stringLen = uint32_t(len == 0 && src != nullptr ? strlen(src) : len);
  }

  explicit Var(const std::string &src) : SHVar() {
    valueType = SHType::String;
    payload.stringValue = src.c_str();
    payload.stringLen = uint32_t(src.length());
  }

  explicit Var(const std::string_view &src) : SHVar() {
    valueType = SHType::String;
    payload.stringValue = src.data();
    payload.stringLen = uint32_t(src.size());
  }

  static Var ContextVar(const std::string &src) {
    Var res{};
    res.valueType = SHType::ContextVar;
    res.payload.stringValue = src.c_str();
    res.payload.stringLen = uint32_t(src.length());
    return res;
  }

  explicit Var(SHTable &src) : SHVar() {
    valueType = Table;
    payload.tableValue = src;
  }

  explicit Var(SHColor color) : SHVar() {
    valueType = Color;
    payload.colorValue = color;
  }

  static Var ColorFromInt(uint32_t rgba) {
    Var res{};
    res.valueType = SHType::Color;
    res.payload.colorValue.r = uint8_t((rgba & 0xFF000000) >> 24);
    res.payload.colorValue.g = uint8_t((rgba & 0x00FF0000) >> 16);
    res.payload.colorValue.b = uint8_t((rgba & 0x0000FF00) >> 8);
    res.payload.colorValue.a = uint8_t((rgba & 0x000000FF) >> 0);
    return res;
  }

  uint32_t colorToInt() {
    if (valueType != Color) {
      throw InvalidVarTypeError("Invalid variable casting! expected Color");
    }
    uint32_t res = 0;
    res |= (uint32_t(payload.colorValue.r) << 24);
    res |= (uint32_t(payload.colorValue.g) << 16);
    res |= (uint32_t(payload.colorValue.b) << 8);
    res |= (uint32_t(payload.colorValue.a) << 0);
    return res;
  }

  explicit Var(const SHVar *data, size_t size) : SHVar() {
    valueType = Seq;
    payload.seqValue.len = uint32_t(size);
    payload.seqValue.elements = payload.seqValue.len > 0 ? const_cast<SHVar *>(data) : nullptr;
  }

  template <size_t N> explicit Var(const std::array<uint8_t, N> &arrRef) : SHVar() {
    valueType = SHType::Bytes;
    payload.bytesValue = N > 0 ? const_cast<uint8_t *>(arrRef.data()) : nullptr;
    payload.bytesSize = N;
  }

  template <typename TVAR> explicit Var(const std::vector<TVAR> &vectorRef) : SHVar() {
    valueType = Seq;
    payload.seqValue.len = uint32_t(vectorRef.size());
    payload.seqValue.elements = payload.seqValue.len > 0 ? const_cast<TVAR *>(vectorRef.data()) : nullptr;
  }

  template <typename TVAR, size_t N> explicit Var(const std::array<TVAR, N> &arrRef) : SHVar() {
    valueType = Seq;
    payload.seqValue.elements = N > 0 ? const_cast<TVAR *>(arrRef.data()) : nullptr;
    payload.seqValue.len = N;
  }

  Var(const Wire &wire);
};

using VarPayload = std::variant<IntVarPayload, Int2VarPayload, Int3VarPayload, Int4VarPayload, Int8VarPayload, Int16VarPayload,
                                FloatVarPayload, Float2VarPayload, Float3VarPayload, Float4VarPayload>;

template <class Function> inline void ForEach(const SHTable &table, Function &&f) {
  SHTableIterator tit;
  table.api->tableGetIterator(table, &tit);
  SHString k;
  SHVar v;
  while (table.api->tableNext(table, &tit, &k, &v)) {
    f(k, v);
  }
}

template <class Function> inline void ForEach(const SHSet &set, Function &&f) {
  SHSetIterator sit;
  set.api->setGetIterator(set, &sit);
  SHVar v;
  while (set.api->setNext(set, &sit, &v)) {
    f(v);
  }
}

template <class Function> inline void ForEach(const SHSeq &seq, Function &&f) {
  for (size_t i = 0; i < seq.len; i++)
    f(seq.elements[i]);
}

class WireProvider {
  // used specially for live editing wires, from host languages
public:
  static inline Type NoneType{{SHType::None}};
  static inline Type ProviderType{{SHType::Object, {.object = {.vendorId = CoreCC, .typeId = 'chnp'}}}};
  static inline Types ProviderOrNone{{ProviderType, NoneType}};

  WireProvider() {
    _provider.userData = this;
    _provider.reset = [](SHWireProvider *provider) {
      auto p = reinterpret_cast<WireProvider *>(provider->userData);
      p->reset();
    };
    _provider.ready = [](SHWireProvider *provider) {
      auto p = reinterpret_cast<WireProvider *>(provider->userData);
      return p->ready();
    };
    _provider.setup = [](SHWireProvider *provider, const char *path, SHInstanceData data) {
      auto p = reinterpret_cast<WireProvider *>(provider->userData);
      p->setup(path, data);
    };
    _provider.updated = [](SHWireProvider *provider) {
      auto p = reinterpret_cast<WireProvider *>(provider->userData);
      return p->updated();
    };
    _provider.acquire = [](SHWireProvider *provider) {
      auto p = reinterpret_cast<WireProvider *>(provider->userData);
      return p->acquire();
    };
    _provider.release = [](SHWireProvider *provider, SHWire *wire) {
      auto p = reinterpret_cast<WireProvider *>(provider->userData);
      return p->release(wire);
    };
  }

  virtual ~WireProvider() {}

  virtual void reset() = 0;

  virtual bool ready() = 0;
  virtual void setup(const char *path, const SHInstanceData &data) = 0;

  virtual bool updated() = 0;
  virtual SHWireProviderUpdate acquire() = 0;

  virtual void release(SHWire *wire) = 0;

  operator SHVar() {
    SHVar res{};
    res.valueType = Object;
    res.payload.objectVendorId = CoreCC;
    res.payload.objectTypeId = 'chnp';
    res.payload.objectValue = &_provider;
    return res;
  }

private:
  SHWireProvider _provider;
};

class Weave {
public:
  Weave() {}

  Weave &shard(std::string_view name, std::vector<Var> params);

  template <typename... Vars> Weave &shard(std::string_view name, Vars... params) {
    std::vector<Var> vars = {Var(params)...};
    return shard(name, vars);
  }

  Weave &let(Var value);

  template <typename V> Weave &let(V value) {
    auto val = Var(value);
    return let(val);
  }

  template <typename... Vs> Weave &let(Vs... values) {
    auto val = Var(values...);
    return let(val);
  }

  operator Var() { return Var(_shards); }

private:
  std::vector<Var> _shards;
};

class Wire {
public:
  Wire() {}
  Wire(std::string_view name);

  Wire &shard(std::string_view name, std::vector<Var> params);

  template <typename... Vars> Wire &shard(std::string_view name, Vars... params) {
    std::vector<Var> vars = {Var(params)...};
    return shard(name, vars);
  }

  Wire &let(Var value);

  template <typename V> Wire &let(V value) {
    auto val = Var(value);
    return let(val);
  }

  template <typename... Vs> Wire &let(Vs... values) {
    auto val = Var(values...);
    return let(val);
  }

  Wire &looped(bool looped);
  Wire &unsafe(bool unsafe);
  Wire &stackSize(size_t size);
  Wire &name(std::string_view name);

  SHWire *operator->() { return _wire.get(); }
  SHWire *get() { return _wire.get(); }

  operator std::shared_ptr<SHWire>() { return _wire; }
  SHWireRef weakRef() const;

private:
  std::shared_ptr<SHWire> _wire;
};

void abortWire(struct SHContext *context, std::string_view errorText);
} // namespace shards

inline SHVar *begin(SHVar &a) {
  if (a.valueType != SHType::Seq) {
    throw shards::SHException("begin expected a Seq");
  }
  return &a.payload.seqValue.elements[0];
}

inline const SHVar *begin(const SHVar &a) {
  if (a.valueType != SHType::Seq) {
    throw shards::SHException("begin expected a Seq");
  }
  return &a.payload.seqValue.elements[0];
}

inline SHVar *end(SHVar &a) {
  if (a.valueType != SHType::Seq) {
    throw shards::SHException("begin expected a Seq");
  }
  return begin(a) + a.payload.seqValue.len;
}

inline const SHVar *end(const SHVar &a) {
  if (a.valueType != SHType::Seq) {
    throw shards::SHException("begin expected a Seq");
  }
  return begin(a) + a.payload.seqValue.len;
}

inline SHExposedTypeInfo *begin(SHExposedTypesInfo &a) { return &a.elements[0]; }

inline const SHExposedTypeInfo *begin(const SHExposedTypesInfo &a) { return &a.elements[0]; }

inline SHExposedTypeInfo *end(SHExposedTypesInfo &a) { return begin(a) + a.len; }

inline const SHExposedTypeInfo *end(const SHExposedTypesInfo &a) { return begin(a) + a.len; }

inline SHTypeInfo *begin(SHTypesInfo &a) { return &a.elements[0]; }

inline const SHTypeInfo *begin(const SHTypesInfo &a) { return &a.elements[0]; }

inline SHTypeInfo *end(SHTypesInfo &a) { return begin(a) + a.len; }

inline const SHTypeInfo *end(const SHTypesInfo &a) { return begin(a) + a.len; }

#endif
