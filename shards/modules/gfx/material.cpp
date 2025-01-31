#include "gfx.hpp"
#include "shards_types.hpp"
#include "shards_utils.hpp"
#include "param_types.hpp"
#include <shards/linalg_shim.hpp>
#include "drawable_utils.hpp"
#include <gfx/material.hpp>
#include <shards/core/params.hpp>

using namespace shards;
namespace gfx {
struct MaterialShard {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::Material; }

  static SHOptionalString help() { return SHCCSTR("This shard creates a material from the given parameters and features."); }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the material created as a material object."); }

  PARAM_EXT(ParamVar, _params, gfx::detail::ParamsParameterInfo);
  PARAM_EXT(ParamVar, _features, ShardsTypes::FeaturesParameterInfo);

  PARAM_IMPL(PARAM_IMPL_FOR(_params), PARAM_IMPL_FOR(_features));

  SHMaterial *_material{};

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    _material = ShardsTypes::MaterialObjectVar.New();
    _material->material = std::make_shared<Material>();
  }

  void cleanup(SHContext* context) {
    PARAM_CLEANUP(context);
    if (_material) {
      ShardsTypes::MaterialObjectVar.Release(_material);
      _material = nullptr;
    }
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &material = _material->material;

    if (!_params.isNone()) {
      initShaderParams(shContext, _params.get().payload.tableValue, material->parameters);
    }

    if (!_features.isNone()) {
      material->features.clear();
      applyFeatures(shContext, material->features, _features.get());
    }

    return ShardsTypes::MaterialObjectVar.Get(_material);
  }
};

void registerMaterialShards() { REGISTER_SHARD("GFX.Material", MaterialShard); }

} // namespace gfx
