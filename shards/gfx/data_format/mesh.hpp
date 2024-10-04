#ifndef B0FC745D_756F_4FC5_9C3A_E401187EAE3D
#define B0FC745D_756F_4FC5_9C3A_E401187EAE3D

#include "base.hpp"
#include <gfx/mesh.hpp>

namespace gfx {
struct SerializedMesh {
  MeshDescCPUCopy meshDesc;
};
} // namespace gfx

namespace shards {
template <> struct Serialize<gfx::MeshVertexAttribute> {
  template <SerializerStream S> void operator()(S &stream, gfx::MeshVertexAttribute &v) {
    serde(stream, v.name);
    serde(stream, v.numComponents);
    serdeAs<uint8_t>(stream, v.type);
  }
};
template <> struct Serialize<gfx::MeshFormat> {
  template <SerializerStream S> void operator()(S &stream, gfx::MeshFormat &v) {
    serdeAs<uint8_t>(stream, v.primitiveType);
    serdeAs<uint8_t>(stream, v.windingOrder);
    serdeAs<uint8_t>(stream, v.indexFormat);
    serde(stream, v.vertexAttributes);
  }
};
template <> struct Serialize<gfx::MeshDescCPUCopy> {
  template <SerializerStream S> void operator()(S &stream, gfx::MeshDescCPUCopy &v) {
    serde(stream, v.format);
    serde(stream, v.vertexData);
    serde(stream, v.indexData);
  }
};
template <> struct Serialize<gfx::SerializedMesh> {
  template <SerializerStream S> void operator()(S &stream, gfx::SerializedMesh &v) { serde(stream, v.meshDesc); }
};
} // namespace shards

#endif /* B0FC745D_756F_4FC5_9C3A_E401187EAE3D */
