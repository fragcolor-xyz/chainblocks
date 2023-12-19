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

void Mesh::update(const MeshFormat &format, const void *inVertexData, size_t vertexDataLength, const void *inIndexData,
                  size_t indexDataLength) {
  this->format = format;

  size_t vertexSize = format.getVertexSize();
  size_t indexSize = getIndexFormatSize(format.indexFormat);

  calculateElementCounts(vertexDataLength, indexDataLength, vertexSize, indexSize);

  // FIXME: wgpu-rs requires buffer writes to be aligned to 4 currently
  vertexData.resize(alignTo<4>(vertexDataLength));
  indexData.resize(alignTo<4>(indexDataLength));

  if (vertexDataLength > 0)
    memcpy(vertexData.data(), inVertexData, vertexDataLength);

  if (indexDataLength > 0)
    memcpy(indexData.data(), inIndexData, indexDataLength);

  update();
}

void Mesh::update(const MeshFormat &format, std::vector<uint8_t> &&vertexData, std::vector<uint8_t> &&indexData) {
  this->format = format;

  size_t vertexSize = format.getVertexSize();
  size_t indexSize = getIndexFormatSize(format.indexFormat);

  this->vertexData = std::move(vertexData);
  this->indexData = std::move(indexData);
  calculateElementCounts(this->vertexData.size(), this->indexData.size(), vertexSize, indexSize);

  // FIXME: wgpu-rs requires buffer writes to be aligned to 4 currently
  this->vertexData.resize(alignTo<4>(this->vertexData.size()));
  this->indexData.resize(alignTo<4>(this->indexData.size()));

  update();
}

void Mesh::calculateElementCounts(size_t vertexDataLength, size_t indexDataLength, size_t vertexSize, size_t indexSize) {
  shassert(vertexSize > 0 && "vertexSize was 0");

  numVertices = vertexDataLength / vertexSize;
  shassert(numVertices * vertexSize == vertexDataLength);

  numIndices = indexDataLength / indexSize;
  shassert(numIndices * indexSize == indexDataLength);
}

void Mesh::update() {
  // This causes the GPU data to be recreated the next time it is requested
  updateData = true;
  version++;
}

MeshPtr Mesh::clone() const {
  auto result = cloneSelfWithId(this, getNextId());
  result->contextData.reset();
  return result;
}

void Mesh::initContextData(Context &context, MeshContextData &contextData) {
  contextData.vertexBufferLength = 0;
  contextData.indexBufferLength = 0;
}

void Mesh::updateContextData(Context &context, MeshContextData &contextData) {
  if (contextData.vertexBufferLength != 0 && !updateData)
    return;
  updateData = false;

  WGPUDevice device = context.wgpuDevice;
  shassert(device);

  contextData.format = format;
  contextData.numIndices = numIndices;
  contextData.numVertices = numVertices;

  // Grow/create vertex buffer
  if (vertexData.size() > 0) {
    if (contextData.vertexBufferLength < vertexData.size()) {
      WGPUBufferDescriptor desc = {};
      desc.size = vertexData.size();
      desc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
      contextData.vertexBuffer = wgpuDeviceCreateBuffer(device, &desc);
      contextData.vertexBufferLength = desc.size;
    }

    shassert(contextData.vertexBuffer);
    wgpuQueueWriteBuffer(context.wgpuQueue, contextData.vertexBuffer, 0, vertexData.data(), vertexData.size());
  }

  if (indexData.size() > 0) {
    // Grow/create index buffer
    if (indexData.size() > contextData.indexBufferLength) {
      WGPUBufferDescriptor desc = {};
      desc.size = indexData.size();
      desc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
      contextData.indexBuffer = wgpuDeviceCreateBuffer(device, &desc);
      contextData.indexBufferLength = desc.size;
    }

    shassert(contextData.indexBuffer);
    wgpuQueueWriteBuffer(context.wgpuQueue, contextData.indexBuffer, 0, indexData.data(), indexData.size());
  }
}

} // namespace gfx
