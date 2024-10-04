#include "features/wireframe.hpp"
#include "drawables/mesh_drawable.hpp"
#include "linalg.hpp"
#include "mesh.hpp"
#include "transform_updater.hpp"
#include "../renderer_cache.hpp"
#include "wireframe.hpp"
#include "mesh_utils.hpp"
#include "worker_memory.hpp"
#include <stdexcept>

namespace gfx {
MeshPtr WireframeMeshGenerator::generate() {
  if (!mesh)
    throw std::logic_error("Input mesh required");

  auto meshDescPtr = std::get_if<MeshDescCPUCopy>(&mesh->getDesc());
  if (!meshDescPtr)
    throw std::runtime_error("MeshDesc is not a MeshDescCPUCopy");
  const MeshDescCPUCopy &meshDesc = *meshDescPtr;
  auto &srcFormat = meshDesc.format;
  auto indicesPtr = meshDesc.indexData.data();
  auto verticesPtr = meshDesc.vertexData.data();
  if (meshDesc.getNumIndices() == 0)
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
                                         meshDesc.getNumIndices(), srcStride, verticesPtr + offset);
    }
    offset += attrib.numComponents * getStorageTypeSize(attrib.type);
  }

  if (!positionIndex.has_value()) {
    throw std::runtime_error("Mesh does not have a position attribute");
  }

  boost::container::small_vector<std::tuple<AttribBuffer *, FastString>, 16> outAttributes;
  for (size_t i = 0; i < attributesToCopy.size(); i++) {
    size_t srcAttribIdx = attributesToCopy[i];
    outAttributes.push_back({&attributes[i], srcFormat.vertexAttributes[srcAttribIdx].name});
  }
  return generateMesh(std::nullopt, boost::span(outAttributes));
}

WireframeRenderer::WireframeRenderer(bool showBackfaces) {
  wireframeFeature = features::Wireframe::create(showBackfaces);
  allocator = std::make_shared<detail::WorkerMemory>();
}

MeshDrawable::Ptr WireFrameDrawablePoolTraits::newItem() { return std::make_shared<MeshDrawable>(); }

void WireframeRenderer::reset(size_t frameCounter) {
  detail::clearOldCacheItemsIn(meshCache, frameCounter, 32);
  for (auto &it : meshCache) {
    auto &entry = it.second;
    entry.drawablePool.recycle();
  }
  currentFrameCounter = frameCounter;
}
void WireframeRenderer::overlayWireframe(DrawQueue &queue, IDrawable &drawable, float4 color) {
  if (MeshDrawable *meshDrawable = dynamic_cast<MeshDrawable *>(&drawable)) {
    auto drawable = getWireframeDrawable(meshDrawable->mesh, color);
    drawable->skin = meshDrawable->skin; // Make sure to copy source skin so the wireframe can be animated
    drawable->transform = meshDrawable->transform;
    queue.add(drawable);
  } else if (MeshTreeDrawable *treeDrawable = dynamic_cast<MeshTreeDrawable *>(&drawable)) {
    allocator->reset();
    TransformUpdaterCollector collector(*allocator.get());
    collector.collector = [&](DrawablePtr drawable, const float4x4 &) { overlayWireframe(queue, *drawable.get(), color); };
    collector.update(*treeDrawable);
  }
}

std::shared_ptr<MeshDrawable> WireframeRenderer::getWireframeDrawable(const MeshPtr &mesh, float4 color) {
  Mesh *meshPtr = mesh.get();
  auto it = meshCache.find(meshPtr);
  if (it == meshCache.end()) {
    WireframeMeshGenerator meshGenerator;
    meshGenerator.mesh = mesh;
    MeshCacheEntry entry{
        .wireMesh = meshGenerator.generate(),
        .weakMesh = mesh,
    };

    it = meshCache.insert_or_assign(meshPtr, entry).first;
  }
  detail::touchCacheItem(it->second, currentFrameCounter);

  auto drawable = it->second.drawablePool.newValue([&](auto &drawable) {
    drawable->mesh = it->second.wireMesh;
    drawable->material.reset();
    drawable->features.clear();
    drawable->parameters.clear();
    drawable->features.push_back(wireframeFeature);
  });
  static FastString fs_baseColor = "baseColor";
  drawable->parameters.set(fs_baseColor, color);
  return drawable;
}

} // namespace gfx
