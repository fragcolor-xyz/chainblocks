#ifndef EF379F4B_AEF2_44CC_BF87_5A2D2EA9BBB5
#define EF379F4B_AEF2_44CC_BF87_5A2D2EA9BBB5

#include <gfx/data_cache/types.hpp>
#include <shards/core/serialization/linalg.hpp>
#include "mesh_drawable.hpp"
#include "../material.hpp"
#include "../trs_serde.hpp"

namespace shards {
template <> struct Serialize<fast_string::FastString> {
  template <SerializerStream S> static void serialize(S &stream, fast_string::FastString &str) {
    serdeAs<std::string>(stream, str);
  }
};

template <> struct Serialize<gfx::MaterialParameters> {
  template <SerializerStream S> static void serialize(S &stream, gfx::MaterialParameters &params) {
    // stream(params.albedo);
    // stream(params.roughness);
    // stream(params.metallic);
    // stream(params.normal);
    // stream(params.emissive);
    // stream(params.occlusion);
  }
};
} // namespace shards

namespace gfx {

struct SerializedMaterialParameter {
  std::string key;
  NumParameter value;
};

struct SerializedMaterialParameters {
  std::vector<
  SerializedMaterialParameters() = default;
  SerializedMaterialParameters(const MaterialParameters &params) {
    params.basic
    // for (auto &[key, value] : params) {
    //   params[key] = value;
    // }
  }
};

struct SerializedMeshDrawable {
  data::AssetKey mesh;
  data::AssetKey skin;
  TRS transform;
  SerializedMaterialParameters params;

  SerializedMeshDrawable() = default;
  SerializedMeshDrawable(const MeshDrawable &drawable) {
    transform = TRS(drawable.transform);
    // TODO: mesh source
    // drawable.mesh->getNumVertices()
  }
  SerializedMeshDrawable(const MeshDrawable::Ptr &drawable) : MeshDrawableState(*drawable) {}
};
} // namespace gfx

#endif /* EF379F4B_AEF2_44CC_BF87_5A2D2EA9BBB5 */
