#ifndef A1665234_253E_49DD_B428_FA0D8802DD91
#define A1665234_253E_49DD_B428_FA0D8802DD91

#include "linalg.hpp"
#include "mesh.hpp"
#include <cfloat>

namespace gfx {
struct AABounds {
  float3 min;
  float3 max;

  constexpr AABounds() = default;
  constexpr AABounds(float3 min, float3 max) : min(min), max(max) {}

  static AABounds empty() { return AABounds(float3(FLT_MAX), float3(-FLT_MAX)); }
  void expand(float3 pt) {
    min = linalg::min(min, pt);
    max = linalg::max(max, pt);
  }

  float3 center() const { return (min + max) * 0.5f; }
  float3 size() const { return max - min; }
  float3 halfSize() const { return size() * 0.5f; }

  AABounds expand(float3 offset) const { return AABounds(min - offset, max + offset); }

  static AABounds both(const AABounds &a, const AABounds &b) {
    return AABounds(linalg::min(a.min, b.min), linalg::max(a.max, b.max));
  }

  template <typename F> std::enable_if_t<std::is_invocable_r_v<bool, F, float3>> forEachPoint(F func) const {
    // The first 4 points
    if (!func(float3(min.x, min.y, max.z)))
      return;
    if (!func(float3(max.x, min.y, min.z)))
      return;
    if (!func(float3(min.x, max.y, min.z)))
      return;
    if (!func(float3(min.x, min.y, max.z)))
      return;
    // The other 4 points
    if (!func(float3(max.x, max.y, max.z)))
      return;
    if (!func(float3(max.x, max.y, min.z)))
      return;
    if (!func(float3(max.x, min.y, max.z)))
      return;
    if (!func(float3(min.x, max.y, max.z)))
      return;
  }
};

struct OrientedBounds {
  AABounds base;

private:
  float4x4 _transform;
  mutable std::optional<float4x4> _inverseTransform;

public:
  OrientedBounds() = default;
  OrientedBounds(AABounds base, float4x4 transform) : base(base), _transform(transform) {}

  static OrientedBounds empty() { return OrientedBounds(AABounds::empty(), linalg::identity); }

  OrientedBounds expand(float3 offset) const { return OrientedBounds(base.expand(offset), _transform); }

  float4x4 transform() const { return _transform; }
  float4x4 inverseTransform() const {
    if (!_inverseTransform.has_value()) {
      _inverseTransform = linalg::inverse(_transform);
    }
    return *_inverseTransform;
  }

  AABounds toAligned() const {
    AABounds bounds = AABounds::empty();
    base.forEachPoint([&](float3 pt) -> bool {
      bounds.expand(linalg::mul(transform(), float4(pt, 1)).xyz());
      return true;
    });
    return bounds;
  }
};

inline AABounds getMeshBounds(MeshPtr mesh) {
  auto &fmt = mesh->getFormat();
  size_t offset{};
  AABounds bounds = AABounds::empty();

  std::optional<size_t> positionIndex;
  std::optional<size_t> positionOffset;
  size_t stride = fmt.getVertexSize();
  for (size_t i = 0; i < fmt.vertexAttributes.size(); i++) {
    auto &attrib = fmt.vertexAttributes[i];
    if (attrib.name == "position") {
      positionIndex = i;
      positionOffset = offset;
    }
    offset += attrib.numComponents * getStorageTypeSize(attrib.type);
  }

  if (!positionIndex.has_value()) {
    throw std::runtime_error("Mesh has no position attribute");
  }

  if (mesh->getNumVertices() == 0) {
    return bounds;
  }

  const uint8_t *basePtr = mesh->getVertexData().data() + *positionOffset;
  for (size_t i = 0; i < mesh->getNumVertices(); i++) {
    auto &pt = *(float3 *)(basePtr + stride * i);
    bounds.expand(pt);
  }

  return bounds;
}

// Represents both a sphere and box around a mesh
struct CompleteBounds {
  float3 center;
  float sphereRadius;
  float3 boxHalfExtent;
};
} // namespace gfx

#endif /* A1665234_253E_49DD_B428_FA0D8802DD91 */
