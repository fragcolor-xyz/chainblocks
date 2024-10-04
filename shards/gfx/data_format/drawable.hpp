#ifndef D33DD9E7_C63E_42FF_BBBC_0F09E31E23D4
#define D33DD9E7_C63E_42FF_BBBC_0F09E31E23D4

#include "base.hpp"
#include "mesh.hpp"
#include "../trs_serde.hpp"
#include <shards/core/serialization/linalg.hpp>
#include <gfx/data_cache/types.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/material.hpp>
#include <gfx/texture.hpp>
#include <gfx/trs_serde.hpp>

namespace gfx {
struct SerializedMaterialParameter {
  FastString key;
  NumParameter value;
};

struct SerializedTextureParameter {
  FastString key;
  data::AssetKey value;
  TextureFormat format;
  uint8_t defaultTexcoordBinding;
  SerializedTextureParameter() = default;
  SerializedTextureParameter(FastString key, const data::AssetKey &value, TextureFormat format, uint8_t defaultTexcoordBinding = 0)
      : key(key), value(value), defaultTexcoordBinding(defaultTexcoordBinding) {}
};

struct SerializedMaterialParameters {
  std::vector<SerializedMaterialParameter> basic;
  std::vector<SerializedTextureParameter> textures;
  SerializedMaterialParameters() = default;
  SerializedMaterialParameters(const MaterialParameters &params) {
    for (auto &[key, value] : params.basic) {
      basic.emplace_back(SerializedMaterialParameter{key, value});
    }
    for (auto &[key, value] : params.textures) {
      if (auto assetDesc = std::get_if<TextureDescAsset>(&value.texture->getDesc())) {
        textures.emplace_back(SerializedTextureParameter{key, assetDesc->key, value.texture->getFormat(), value.defaultTexcoordBinding});
      }
    }
  }
};

struct SerializedMeshDrawable {
  data::AssetKey mesh;
  MeshFormat meshFormat;
  data::AssetKey skin;
  TRS transform;
  SerializedMaterialParameters params;

  SerializedMeshDrawable() = default;
  SerializedMeshDrawable(const MeshDrawable &drawable) {
    transform = TRS(drawable.transform);
    // TODO: mesh source
    // drawable.mesh->getNumVertices()
  }
  SerializedMeshDrawable(const MeshDrawable::Ptr &drawable) : SerializedMeshDrawable(*drawable) {}
};

} // namespace gfx

namespace shards {

template <> struct Serialize<gfx::SerializedMaterialParameter> {
  template<SerializerStream S>
  void operator()(S &stream, gfx::SerializedMaterialParameter &value) {
    serde(stream, value.key);
    serde(stream, value.value);
  }
};

template <> struct Serialize<gfx::SerializedTextureParameter> {
  template<SerializerStream S>
  void operator()(S &stream, gfx::SerializedTextureParameter &value) {
    serde(stream, value.key);
    serde(stream, value.value);
    serde(stream, value.format);
    serde(stream, value.defaultTexcoordBinding);
  }
};

template <> struct Serialize<gfx::SerializedMaterialParameters> {
  template<SerializerStream S>
  void operator()(S &stream, gfx::SerializedMaterialParameters &value) {
    serde(stream, value.basic);
    serde(stream, value.textures);
  }
};

template <> struct Serialize<gfx::SerializedMeshDrawable> {
  template<SerializerStream S>
  void operator()(S &stream, gfx::SerializedMeshDrawable &value) {
    serde(stream, value.mesh);
    serde(stream, value.meshFormat);
    serde(stream, value.skin);
    // serde(stream, value.transform);
    // serde(stream, value.params);
  }
};
} // namespace shards

#endif /* D33DD9E7_C63E_42FF_BBBC_0F09E31E23D4 */
