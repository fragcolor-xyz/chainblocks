#include "gfx.hpp"
#include "shards_types.hpp"
#include "shards_utils.hpp"
#include <shards/linalg_shim.hpp>
#include "drawable_utils.hpp"
#include <gfx/material.hpp>
#include <shards/core/params.hpp>

using namespace shards;
namespace gfx {
struct MaterialShard {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return Types::Material; }

  PARAM_EXT(ParamVar, _params, Types::ParamsParameterInfo);
  PARAM_EXT(ParamVar, _features, Types::FeaturesParameterInfo);

  PARAM_IMPL(MaterialShard, PARAM_IMPL_FOR(_params), PARAM_IMPL_FOR(_features));

  SHMaterial *_material{};

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    _material = Types::MaterialObjectVar.New();
    _material->material = std::make_shared<Material>();
  }

  void cleanup() {
    PARAM_CLEANUP();
    if (_material) {
      Types::MaterialObjectVar.Release(_material);
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

    return Types::MaterialObjectVar.Get(_material);
  }
};

void registerMaterialShards() { REGISTER_SHARD("GFX.Material", MaterialShard); }

} // namespace gfx
