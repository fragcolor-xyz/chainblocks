#ifndef CB5ADAE0_5644_4E52_9F32_42D73C93947A
#define CB5ADAE0_5644_4E52_9F32_42D73C93947A

#include "shards_types.hpp"

namespace gfx::detail {
using namespace shards;
static inline Types ValidShaderTypes{{
    ShardsTypes::Texture,
    ShardsTypes::TextureCube,
    ShardsTypes::Buffer,
    CoreInfo::Float4x4Type,
    CoreInfo::Float4Type,
    CoreInfo::Float3Type,
    CoreInfo::Float2Type,
    CoreInfo::FloatType,
    CoreInfo::IntType,
    CoreInfo::Int2Type,
    CoreInfo::Int3Type,
    CoreInfo::Int4Type,
    CoreInfo::AnyTableType
}};

static inline Type ValidShaderVarTypes = Type::VariableOf(ValidShaderTypes);

static inline Types ShaderParamTypes = []() {
  Types res{};
  for (auto &type : ValidShaderTypes._types) {
    res._types.push_back(type);
  }
  res._types.push_back(ValidShaderVarTypes);
  return res;
}();

// Valid types for shader :Params
static inline Type ShaderParamTable = Type::TableOf(gfx::detail::ShaderParamTypes);

static inline ParameterInfo ParamsParameterInfo{"Params",
                                                SHCCSTR("Shader parameters used for Drawable rendering"),
                                                {CoreInfo::NoneType, ShaderParamTable, Type::VariableOf(ShaderParamTable)}};

static inline ParameterInfo FeatureParamsParameterInfo{
    "Params",
    SHCCSTR("The parameters to add to the object buffer and expose to shaders, these default values can later be modified "
            "by the Params parameter in GFX.Material or GFX.Drawable"),
    {CoreInfo::NoneType, ShaderParamTable, Type::VariableOf(ShaderParamTable)}};
} // namespace gfx::detail

#endif /* CB5ADAE0_5644_4E52_9F32_42D73C93947A */
