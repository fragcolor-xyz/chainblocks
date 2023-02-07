#ifndef C7AA640C_1748_42BC_8772_0B8F4E136CEA
#define C7AA640C_1748_42BC_8772_0B8F4E136CEA

#include "gfx/error_utils.hpp"
#include "gfx/shader/types.hpp"
#include "shards.h"
#include "shards_types.hpp"
#include "shards_utils.hpp"
#include <ops_internal.hpp>
#include <foundation.hpp>
#include <gfx/material.hpp>
#include <gfx/params.hpp>
#include <gfx/texture.hpp>
#include <magic_enum.hpp>
#include <optional>
#include <shards.hpp>
#include <spdlog/spdlog.h>
#include <variant>

namespace gfx {
inline TexturePtr varToTexture(const SHVar &var) {
  if (var.payload.objectTypeId == Types::TextureCubeTypeId) {
    return *varAsObjectChecked<TexturePtr>(var, Types::TextureCube);
  } else if (var.payload.objectTypeId == Types::TextureTypeId) {
    return *varAsObjectChecked<TexturePtr>(var, Types::Texture);
  } else {
    SHInstanceData data{};
    auto varType = shards::deriveTypeInfo(var, data);
    DEFER({ shards::freeDerivedInfo(varType); });
    throw formatException("Invalid texture variable type: {}", varType);
  }
}

inline ParamVariant varToParam(const SHVar &var) {
  ParamVariant result;
  switch (var.valueType) {
  case SHType::Float: {
    float vec = float(var.payload.floatValue);
    result = vec;
  } break;
  case SHType::Float2: {
    float2 vec;
    vec.x = float(var.payload.float2Value[0]);
    vec.y = float(var.payload.float2Value[1]);
    result = vec;
  } break;
  case SHType::Float3: {
    float3 vec;
    memcpy(&vec.x, &var.payload.float3Value, sizeof(float) * 3);
    result = vec;
  } break;
  case SHType::Float4: {
    float4 vec;
    memcpy(&vec.x, &var.payload.float4Value, sizeof(float) * 4);
    result = vec;
  } break;
  case SHType::Seq:
    if (var.innerType == SHType::Float4) {
      float4x4 matrix;
      const SHSeq &seq = var.payload.seqValue;
      for (size_t i = 0; i < std::min<size_t>(4, seq.len); i++) {
        float4 row;
        memcpy(&row.x, &seq.elements[i].payload.float4Value, sizeof(float) * 4);
        matrix[i] = row;
      }
      result = matrix;
    } else {
      throw formatException("Seq inner type {} can not be converted to ParamVariant", magic_enum::enum_name(var.valueType));
    }
    break;
  default:
    throw formatException("Value type {} can not be converted to ParamVariant", magic_enum::enum_name(var.valueType));
  }
  return result;
}

using ShaderFieldTypeVariant = std::variant<std::monostate, shader::NumFieldType, gfx::TextureDimension>;
inline ShaderFieldTypeVariant toShaderParamType(const SHTypeInfo &typeInfo) {
  switch (typeInfo.basicType) {
    return shader::FieldTypes::Float;
  case SHType::Float2:
    return shader::FieldTypes::Float2;
  case SHType::Float3:
    return shader::FieldTypes::Float3;
  case SHType::Float4:
    return shader::FieldTypes::Float4;
  case SHType::Seq:
    if (typeInfo.seqTypes.len == 1 && typeInfo.seqTypes.elements[0].basicType == SHType::Float4) {
      return shader::FieldTypes::Float4x4;
    }
    break;
  case SHType::Object: {
    if (typeInfo == Types::Texture)
      return TextureDimension::D2;
    else if (typeInfo == Types::TextureCube)
      return TextureDimension::Cube;
  }
  default:
    break;
  }
  return std::monostate();
}

inline std::optional<ParamVariant> tryVarToParam(const SHVar &var) {
  try {
    return varToParam(var);
  } catch (std::exception &e) {
    SHLOG_ERROR("{}", e.what());
    return std::nullopt;
  }
}

inline void initConstantShaderParams(const SHTable &paramsTable, MaterialParameters &out) {
  SHTableIterator it{};
  SHString key{};
  SHVar value{};
  paramsTable.api->tableGetIterator(paramsTable, &it);
  while (paramsTable.api->tableNext(paramsTable, &it, &key, &value)) {
    auto param = tryVarToParam(value);
    if (param) {
      out.set(key, std::move(param.value()));
    }
  }
}

inline void initConstantTextureParams(const SHTable &texturesTable, MaterialParameters &out) {
  SHTableIterator it{};
  SHString key{};
  SHVar value{};
  texturesTable.api->tableGetIterator(texturesTable, &it);
  while (texturesTable.api->tableNext(texturesTable, &it, &key, &value)) {
    out.set(key, varToTexture(value));
  }
}

inline void initReferencedShaderParams(SHContext *shContext, const SHTable &inTable,
                                       std::vector<SHBasicShaderParameter> &outParams) {
  SHTableIterator it{};
  SHString key{};
  SHVar value{};
  inTable.api->tableGetIterator(inTable, &it);
  while (inTable.api->tableNext(inTable, &it, &key, &value)) {
    shards::ParamVar paramVar(value);
    paramVar.warmup(shContext);
    outParams.emplace_back(key, std::move(paramVar));
  }
}

inline void initShaderParams(SHContext *shContext, const SHTable &inputTable, const shards::ParamVar &inParams,
                             const shards::ParamVar &inTextures, MaterialParameters &outParams, SHShaderParameters &outSHParams) {
  SHVar paramsVar{};
  if (getFromTable(shContext, inputTable, "Params", paramsVar)) {
    initConstantShaderParams(paramsVar.payload.tableValue, outParams);
  }
  if (inParams->valueType != SHType::None) {
    initReferencedShaderParams(shContext, inParams.get().payload.tableValue, outSHParams.basic);
  }

  SHVar texturesVar{};
  if (getFromTable(shContext, inputTable, "Textures", texturesVar)) {
    initConstantTextureParams(texturesVar.payload.tableValue, outParams);
  }
  if (inTextures->valueType != SHType::None) {
    initReferencedShaderParams(shContext, inTextures.get().payload.tableValue, outSHParams.textures);
  }
}

inline void validateShaderParamsType(const SHTypeInfo &type) {
  using shards::ComposeError;

  if (type.basicType != SHType::Table) {
    throw formatException("Wrong type for Params: {}, should be a table", magic_enum::enum_name(type.basicType));
  }

  size_t tableLen = type.table.types.len;
  for (size_t i = 0; i < tableLen; i++) {
    const char *key = type.table.keys.elements[i];
    auto valueType = type.table.types.elements[i];
    bool matched = false;
    for (auto &supportedType : Types::ShaderParamTypes._types) {
      if (valueType == supportedType) {
        matched = true;
        break;
      }
    }

    if (!matched) {
      throw formatException("Unsupported parameter type for param {}: {}", key, magic_enum::enum_name(valueType.basicType));
    }
  }
}

inline void validateTexturesInputType(const SHTypeInfo &type) {
  if (type.basicType != SHType::Table)
    throw formatException("Textures should be a table");

  auto &tableTypes = type.table.types;
  for (auto &type : tableTypes) {
    if (type != Types::Texture)
      throw formatException("Unexpected type in Textures table");
  }
}

struct TableValidationResult {
  bool isValid{};
  std::string unexpectedKey;
  operator bool() const { return isValid; }
};

inline void validateDrawableInputTableEntry(const char *key, const SHTypeInfo &type) {
  if (strcmp(key, "Params") == 0) {
    validateShaderParamsType(type);
  } else if (strcmp(key, "Textures") == 0) {
    validateTexturesInputType(type);
  } else {
    auto expectedTypeIt = Types::DrawableInputTableTypes.find(key);
    if (expectedTypeIt == Types::DrawableInputTableTypes.end()) {
      throw formatException("Unexpected input table key: {}", key);
    }

    if (expectedTypeIt->second != type) {
      throw formatException("Unexpected input type for key: {}. expected {}, got {}", key, (SHTypeInfo &)expectedTypeIt->second,
                            type);
    }
  }
}

inline void validateDrawableInputTableType(const SHTypeInfo &type) {
  auto &inputTable = type.table;
  size_t inputTableLen = inputTable.keys.len;
  for (size_t i = 0; i < inputTableLen; i++) {
    const char *key = inputTable.keys.elements[i];
    SHTypeInfo &type = inputTable.types.elements[i];

    validateDrawableInputTableEntry(key, type);
  }
}

} // namespace gfx

#endif /* C7AA640C_1748_42BC_8772_0B8F4E136CEA */
