#include "mesh.hpp"
#include "context.hpp"
#include "math.hpp"
#include "renderer_cache.hpp"
#include "../core/assert.hpp"
#include <magic_enum.hpp>
#include <spdlog/spdlog.h>

namespace gfx {

UniqueId Mesh::getNextId() {
  static UniqueIdGenerator gen(UniqueIdTag::Mesh);
  return gen.getNext();
}

size_t MeshFormat::getVertexSize() const {
  size_t vertexSize = 0;
  for (const auto &attr : vertexAttributes) {
    size_t typeSize = getStorageTypeSize(attr.type);
    vertexSize += attr.numComponents * typeSize;
  }
  return vertexSize;
}

size_t MeshDescCPUCopy::getNumIndices() const {
  return indexData.getLength() / getIndexFormatSize(format.indexFormat);
}

size_t MeshDescCPUCopy::getNumVertices() const {
  size_t vertexSize = format.getVertexSize();
  shassert(vertexSize > 0 && "vertexSize was 0");
  return vertexData.getLength() / vertexSize;
}

void Mesh::update(const MeshFormat &format, const void *inVertexData, size_t vertexDataLength, const void *inIndexData,
                  size_t indexDataLength) {
  MeshDescCPUCopy desc;
  desc.format = format;

  // FIXME: wgpu-rs requires buffer writes to be aligned to 4 currently
  std::vector<uint8_t> vertexData;
  vertexData.resize(alignTo<4>(vertexDataLength));
  std::vector<uint8_t> indexData;
  indexData.resize(alignTo<4>(indexDataLength));

  if (vertexDataLength > 0) {
    memcpy(vertexData.data(), inVertexData, vertexDataLength);
    desc.vertexData = std::move(vertexData);
  }

  if (indexDataLength > 0) {
    memcpy(indexData.data(), inIndexData, indexDataLength);
    desc.indexData = std::move(indexData);
  }

  update(desc);
}

void Mesh::update(const MeshFormat &format, std::vector<uint8_t> &&vertexData, std::vector<uint8_t> &&indexData) {
  MeshDescCPUCopy desc;
  desc.format = format;

  // FIXME: wgpu-rs requires buffer writes to be aligned to 4 currently
  vertexData.resize(alignTo<4>(vertexData.size()));
  indexData.resize(alignTo<4>(indexData.size()));

  desc.vertexData = std::move(vertexData);
  desc.indexData = std::move(indexData);

  update();
}

void Mesh::update(MeshDesc desc) {
  this->desc = desc;
  update();
}
void Mesh::update() { version++; }

MeshPtr Mesh::clone() const {
  auto result = cloneSelfWithId(this, getNextId());
  return result;
}

void Mesh::initContextData(Context &context, MeshContextData &contextData) {
  contextData.init(getLabel());

  contextData.vertexBufferLength = 0;
  contextData.indexBufferLength = 0;
}

void Mesh::updateContextData(Context &context, MeshContextData &contextData) {
  // TODO
  // WGPUDevice device = context.wgpuDevice;
  // shassert(device);

#if SH_GFX_CONTEXT_DATA_LABELS
  contextData.label = getLabel();
#endif

  // contextData.format = format;
  // contextData.numIndices = numIndices;
  // contextData.numVertices = numVertices;

  // if (vertexData.size() > 0) {
  //   // Grow/create vertex buffer
  //   if (contextData.vertexBufferLength < vertexData.size()) {
  //     WGPUBufferDescriptor desc = {};
  //     desc.size = vertexData.size();
  //     desc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
  //     contextData.vertexBuffer.reset(wgpuDeviceCreateBuffer(device, &desc));
  //     contextData.vertexBufferLength = desc.size;
  //   }

  //   shassert(contextData.vertexBuffer);
  //   wgpuQueueWriteBuffer(context.wgpuQueue, contextData.vertexBuffer, 0, vertexData.data(), vertexData.size());
  // }

  // if (indexData.size() > 0) {
  //   // Grow/create index buffer
  //   if (indexData.size() > contextData.indexBufferLength) {
  //     WGPUBufferDescriptor desc = {};
  //     desc.size = indexData.size();
  //     desc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
  //     contextData.indexBuffer.reset(wgpuDeviceCreateBuffer(device, &desc));
  //     contextData.indexBufferLength = desc.size;
  //   }

  //   shassert(contextData.indexBuffer);
  //   wgpuQueueWriteBuffer(context.wgpuQueue, contextData.indexBuffer, 0, indexData.data(), indexData.size());
  // }
}

} // namespace gfx
