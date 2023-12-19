#include "features/wireframe.hpp"
#include "drawables/mesh_drawable.hpp"
#include "linalg.hpp"
#include "mesh.hpp"
#include "transform_updater.hpp"
#include "wireframe.hpp"
#include "mesh_utils.hpp"
#include <stdexcept>

namespace gfx {
MeshPtr WireframeMeshGenerator::generate() {
  if (!mesh)
    throw std::logic_error("Input mesh required");

  auto &srcFormat = mesh->getFormat();
  auto indicesPtr = mesh->getIndexData().data();
  auto verticesPtr = mesh->getVertexData().data();
  if (mesh->getNumIndices() == 0)
    return mesh;

  boost::container::small_vector<size_t, 16> attributesToCopy;
  boost::container::small_vector<AttribBuffer, 16> attributes;

  size_t srcStride = srcFormat.getVertexSize();

  std::optional<size_t> positionIndex;
  size_t offset{};
  for (size_t i = 0; i < srcFormat.vertexAttributes.size(); i++) {
    bool copyAttrib = true;
    auto &attrib = srcFormat.vertexAttributes[i];
    if (attrib.name == "position") {
      positionIndex = i;
    } else if (attrib.name == "normal" || attrib.name == "tangent" || attrib.name.str().starts_with("texCoord")) {
      copyAttrib = false;
    }

    if (copyAttrib) {
      attributesToCopy.push_back(i);
      auto &buffer = attributes.emplace_back();
      buffer.initFromIndexedTriangleList(attrib.type, attrib.numComponents, srcFormat.indexFormat, indicesPtr,
                                         mesh->getNumIndices(), srcStride, verticesPtr + offset);
    }
    offset += attrib.numComponents * getStorageTypeSize(attrib.type);
  }

  if (!positionIndex.has_value()) {
    throw std::runtime_error("Mesh does not have a position attribute");
  }

  boost::container::small_vector<std::tuple<AttribBuffer *, FastString>, 16> outAttributes;
  for (size_t i = 0; i < attributesToCopy.size(); i++) {
    size_t srcAttribIdx  = attributesToCopy[i];
    outAttributes.push_back({&attributes[i], srcFormat.vertexAttributes[srcAttribIdx].name});
  }
  return generateMesh(std::nullopt, boost::span(outAttributes));
}

WireframeRenderer::WireframeRenderer(bool showBackfaces) { wireframeFeature = features::Wireframe::create(showBackfaces); }

void WireframeRenderer::overlayWireframe(DrawQueue &queue, IDrawable &drawable) {
  if (MeshDrawable *meshDrawable = dynamic_cast<MeshDrawable *>(&drawable)) {
    Mesh *meshPtr = meshDrawable->mesh.get();
    auto it = meshCache.find(meshPtr);
    if (it == meshCache.end()) {
      WireframeMeshGenerator meshGenerator;
      meshGenerator.mesh = meshDrawable->mesh;
      MeshCacheEntry entry{
          .wireMesh = meshGenerator.generate(),
          .weakMesh = meshDrawable->mesh,
      };

      it = meshCache.insert_or_assign(meshPtr, entry).first;
    }

    auto clone = std::static_pointer_cast<MeshDrawable>(meshDrawable->clone());
    clone->skin = meshDrawable->skin; // Make sure to copy source skin so the wireframe can be animated
    clone->mesh = it->second.wireMesh;
    clone->parameters.clear();
    clone->material.reset();
    clone->parameters.set("baseColor", float4(1.0f, 0.0f, 0.0f, 1.0f));
    clone->features.clear();
    clone->features.push_back(wireframeFeature);
    queue.add(clone);
  } else if (MeshTreeDrawable *treeDrawable = dynamic_cast<MeshTreeDrawable *>(&drawable)) {
    TransformUpdaterCollector collector;
    collector.collector = [&](DrawablePtr drawable) { overlayWireframe(queue, *drawable.get()); };
    collector.update(*treeDrawable);
  }
}
} // namespace gfx
