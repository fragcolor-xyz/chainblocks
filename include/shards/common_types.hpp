/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

/*
Utility to auto load auto discover shards from DLLs
All that is needed is to declare a shards::registerShards
At runtime just dlopen the dll, that's it!
*/

#ifndef SH_COMMON_TYPES_HPP
#define SH_COMMON_TYPES_HPP

#include "shards.hpp"

namespace shards {

enum class BasicTypes {
  None,
  Any,
  Bool,
  Int,
  Int2,
  Int3,
  Int4,
  Int8,
  Int16,
  Float,
  Float2,
  Float3,
  Float4,
  Color,
  Wire,
  Shard,
  Bytes,
  String,
  Image,
  Audio,
};

struct CoreInfo {
  static inline Type NoneType{{SHType::None}};

#define SH_CORE_TYPE_DEF(_shtype_)                                                                       \
  static inline Type _shtype_##Type{{SHType::_shtype_}};                                                 \
  static inline Type _shtype_##SeqType{{SHType::Seq, {.seqTypes = _shtype_##Type}}};                     \
  static inline Type _shtype_##TableType{{SHType::Table, {.table = {.types = _shtype_##Type}}}};         \
  static inline Type _shtype_##VarType{{SHType::ContextVar, {.contextVarTypes = _shtype_##Type}}};       \
  static inline Type _shtype_##VarSeqType{{SHType::ContextVar, {.contextVarTypes = _shtype_##SeqType}}}; \
  static inline Type _shtype_##VarTableType {                                                            \
    {                                                                                                    \
      SHType::ContextVar, { .contextVarTypes = _shtype_##TableType }                                     \
    }                                                                                                    \
  }

  SH_CORE_TYPE_DEF(Any);
  SH_CORE_TYPE_DEF(Bool);
  SH_CORE_TYPE_DEF(Int);
  SH_CORE_TYPE_DEF(Int2);
  SH_CORE_TYPE_DEF(Int3);
  SH_CORE_TYPE_DEF(Int4);
  SH_CORE_TYPE_DEF(Int8);
  SH_CORE_TYPE_DEF(Int16);
  SH_CORE_TYPE_DEF(Float);
  SH_CORE_TYPE_DEF(Float2);
  SH_CORE_TYPE_DEF(Float3);
  SH_CORE_TYPE_DEF(Float4);
  SH_CORE_TYPE_DEF(Color);
  SH_CORE_TYPE_DEF(Wire);
  SH_CORE_TYPE_DEF(ShardRef);
  SH_CORE_TYPE_DEF(Bytes);
  SH_CORE_TYPE_DEF(String);
  SH_CORE_TYPE_DEF(Path);
  SH_CORE_TYPE_DEF(Image);
  SH_CORE_TYPE_DEF(Audio);
  SH_CORE_TYPE_DEF(Type);
  SH_CORE_TYPE_DEF(Trait);

  static inline Type AnyEnumType = Type::Enum(0, 0);

  static inline Type SeqOfAnySeqType = Type::SeqOf(AnySeqType);
  static inline Type SeqOfAnyTableType = Type::SeqOf(AnyTableType);
  static inline Types SeqOfSeqsTypes{{AnySeqType, AnyVarSeqType}};
  static inline Type SeqOfSeqsType = Type::SeqOf(SeqOfSeqsTypes);

  static inline Type TableOfAnySeqType = Type::TableOf(AnySeqType);

  static inline Type Float4x4Type{{SHType::Seq, {.seqTypes = Float4Type}, 4}};
  static inline Type Float4x4SeqType = Type::SeqOf(Float4x4Type);
  static inline Types Float4x4Types{{Float4x4Type, Float4x4SeqType}};
  static inline Type Float3x3Type{{SHType::Seq, {.seqTypes = Float3Type}, 3}};
  static inline Type Float3x3SeqType = Type::SeqOf(Float3x3Type);
  static inline Types Float3x3Types{{Float3x3Type, Float3x3SeqType}};
  static inline Type Float2x2Type{{SHType::Seq, {.seqTypes = Float2Type}, 2}};
  static inline Type Float2x2SeqType = Type::SeqOf(Float2x2Type);
  static inline Types Float2x2Types{{Float2x2Type, Float2x2SeqType}};

  static inline Types DifferentMatrixes{{Float4x4Type, Float3x3Type, Float2x2Type}};

  static inline Types DifferentMatrixesAndMatrixSeq{
      {Float4x4Type, Float3x3Type, Float2x2Type, Float4x4SeqType, Float3x3SeqType, Float2x2SeqType}};

  static inline Types IntOrFloat{{IntType, FloatType}};

  static inline Types FloatOrFloatSeq{{FloatType, FloatSeqType}};
  static inline Types Float3OrFloat3Seq{{Float3Type, Float3SeqType}};

  static inline Types NoneIntOrFloat{{NoneType, IntType, FloatType}};

  static inline Types Indexables{{Int2Type, Int3Type, Int4Type, Int8Type, Int16Type, Float2Type, Float3Type, Float4Type,
                                  BytesType, ColorType, StringType, AnySeqType, AnyTableType}};

  static inline Types RIndexables{{BytesType, StringType, AnySeqType}};

  static inline Types FloatVectors{{Float2Type, Float2SeqType, Float3Type, Float3SeqType, Float4Type, Float4SeqType}};
  static inline Types FloatSeqs{{FloatSeqType, Float2SeqType, Float3SeqType, Float4SeqType}};
  static inline Types FloatSeqsOrAudio{FloatSeqs, {AudioType}};
  static inline Types FloatVectorsOrFloatSeq{
      {FloatSeqType, Float2Type, Float2SeqType, Float3Type, Float3SeqType, Float4Type, Float4SeqType}};
  static inline Types FloatVectorsOrVar{
      FloatVectors, {Float2VarType, Float2VarSeqType, Float3VarType, Float3VarSeqType, Float4VarType, Float4VarSeqType}};

  static inline Types MatrixOrVector{{Float4x4Type, Float4Type, Float3x3Type, Float3Type, Float2x2Type, Float2Type}};

  static inline Types IntOrNone{{IntType, NoneType}};

  static inline Types IntsVar{{IntType, IntSeqType, IntVarType, IntVarSeqType}};

  static inline Types TakeTypes{AnyType, AnyVarType};

  static inline Types RTakeTypes{{IntType, IntSeqType, IntVarType, IntVarSeqType}};

  static inline Types IntsVarOrNone{IntsVar, {NoneType}};

  static inline Types IntIntSeqOrNone{{IntType, IntSeqType, NoneType}};

  static inline Types ShardSeqOrNone{{ShardRefSeqType, NoneType}};

  static inline Types Shards{{ShardRefType, ShardRefSeqType}};

  static inline Types ShardsOrNone{Shards, {NoneType}};

  static inline Types BytesOrAny{{BytesType, AnyType}};

  static inline Types WireOrNone{{WireType, NoneType}};

  static inline Type ShardsOrNoneSeq{{SHType::Seq, {.seqTypes = ShardsOrNone}}};

  static inline Types StringOrBytes{{StringType, BytesType}};

  static inline Types StringOrNone{{StringType, NoneType}};

  static inline Types StringOrStringVar{{StringType, StringVarType}};

  static inline Types StringOrAnyVar{{StringType, AnyVarType}};

  static inline Types ColorOrNone{{ColorType, NoneType}};

  static inline Types AnyNumbers{{IntType,    IntSeqType,    Int2Type,   Int2SeqType,   Int3Type,   Int3SeqType,
                                  Int4Type,   Int4SeqType,   Int8Type,   Int8SeqType,   Int16Type,  Int16SeqType,
                                  FloatType,  FloatSeqType,  Float2Type, Float2SeqType, Float3Type, Float3SeqType,
                                  Float4Type, Float4SeqType, ColorType,  ColorSeqType}};

  static inline Types StringOrBytesVarOrNone{{StringVarType, BytesVarType, NoneType}};

  static inline Types StringStringVarOrNone{{StringType, StringVarType, NoneType}};

  static inline Types IntOrIntVar{{IntType, IntVarType}};

  static inline Types SeqOrTable{{AnySeqType, AnyTableType}};
};
} // namespace shards

#endif