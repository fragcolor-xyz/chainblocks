#ifndef GFX_SHADER_WGSL_MAPPING
#define GFX_SHADER_WGSL_MAPPING

#include "../enums.hpp"
#include "../params.hpp"
#include "types.hpp"
#include <cassert>
#include <nameof.hpp>
#include <spdlog/fmt/fmt.h>

namespace gfx {
namespace shader {

inline ShaderFieldBaseType getCompatibleShaderFieldBaseType(StorageType type) {
  switch (type) {
  case StorageType::UInt8:
    return ShaderFieldBaseType::UInt8;
  case StorageType::Int8:
    return ShaderFieldBaseType::Int8;
  case StorageType::UInt16:
    return ShaderFieldBaseType::UInt16;
  case StorageType::Int16:
    return ShaderFieldBaseType::Int16;
  case StorageType::UNorm8:
  case StorageType::SNorm8:
  case StorageType::UNorm16:
  case StorageType::SNorm16:
    return ShaderFieldBaseType::Float32;
  case StorageType::UInt32:
    return ShaderFieldBaseType::UInt32;
  case StorageType::Int32:
    return ShaderFieldBaseType::Int32;
  case StorageType::Float16:
    return ShaderFieldBaseType::Float16;
  case StorageType::Float32:
    return ShaderFieldBaseType::Float32;
  default:
    assert(false);
    return ShaderFieldBaseType(~0);
  }
}

inline String getFieldWGSLTypeName(const NumFieldType &type) {
  const char *baseType = nullptr;
  switch (type.baseType) {
  case ShaderFieldBaseType::UInt8:
    baseType = "u32";
    break;
  case ShaderFieldBaseType::Int8:
    baseType = "i32";
    break;
  case ShaderFieldBaseType::UInt16:
    baseType = "u32";
    break;
  case ShaderFieldBaseType::Int16:
    baseType = "i32";
    break;
  case ShaderFieldBaseType::UInt32:
    baseType = "u32";
    break;
  case ShaderFieldBaseType::Int32:
    baseType = "i32";
    break;
  case ShaderFieldBaseType::Float16:
    baseType = "f32";
    break;
  case ShaderFieldBaseType::Float32:
    baseType = "f32";
    break;
  case gfx::ShaderFieldBaseType::Bool:
    baseType = "bool";
    break;
  default:
    throw std::out_of_range(NAMEOF(NumFieldType::baseType).str());
  }

  if (type.matrixDimension > 1) {
    // Check, can not create mat1x1 types
    if (type.numComponents <= 1)
      throw std::out_of_range(NAMEOF(NumFieldType::numComponents).str());
    return fmt::format("mat{}x{}<{}>", type.numComponents, type.matrixDimension, baseType);
  } else {
    const char *vecType = nullptr;
    switch (type.numComponents) {
    case 1:
      break;
    case 2:
      vecType = "vec2";
      break;
    case 3:
      vecType = "vec3";
      break;
    case 4:
      vecType = "vec4";
      break;
    default:
      throw std::out_of_range(NAMEOF(NumFieldType::numComponents).str());
    }
    return vecType ? fmt::format("{}<{}>", vecType, baseType) : baseType;
  }
}

inline String getFieldWGSLTypeName(const TextureFieldType &type) {
  const char *textureFormat{};
  switch (type.format) {
  case TextureSampleType::Int:
    textureFormat = "i32";
    break;
  case TextureSampleType::UInt:
    textureFormat = "u32";
    break;
  case TextureSampleType::Float:
  case TextureSampleType::UnfilterableFloat:
    textureFormat = "f32";
    break;
  }

  const char *textureType{};
  switch (type.dimension) {
  case TextureDimension::D1:
    textureType = "texture_1d";
    break;
  case TextureDimension::D2:
    textureType = "texture_2d";
    break;
  case TextureDimension::Cube:
    textureType = "texture_cube";
    break;
  }

  return fmt::format("{}<{}>", textureType, textureFormat);
}

inline const char *getFieldWGSLTypeName(const SamplerFieldType &type) { return "sampler"; }

inline String getFieldWGSLTypeName(const FieldType &type) {
  return std::visit([&](auto &arg) -> String { return getFieldWGSLTypeName(arg); }, type);
}

} // namespace shader
} // namespace gfx

#endif // GFX_SHADER_WGSL_MAPPING
