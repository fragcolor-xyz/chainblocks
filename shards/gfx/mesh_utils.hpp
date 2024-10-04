#ifndef DC6B12F5_63B6_4FBD_BC92_DFF99E0E3323
#define DC6B12F5_63B6_4FBD_BC92_DFF99E0E3323

#include "fwd.hpp"
#include "mesh.hpp"
#include "xxh3.h"
#include <spdlog/fmt/fmt.h>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/core/span.hpp>

namespace gfx {
inline StorageType indexFormatToStorageType(IndexFormat fmt) {
  switch (fmt) {
  case IndexFormat::UInt16:
    return StorageType::UInt16;
  case IndexFormat::UInt32:
    return StorageType::UInt32;
  default:
    throw std::runtime_error("Invalid index format");
  }
}

inline IndexFormat storageTypeToIndexFormat(StorageType st) {
  switch (st) {
  case StorageType::UInt16:
    return IndexFormat::UInt16;
  case StorageType::UInt32:
    return IndexFormat::UInt32;
  default:
    throw std::runtime_error("Invalid storage type for index format");
  }
}

template <typename T> inline auto visitNumComponents(size_t size, T cb) {
  switch (size) {
  case 0:
    return cb(0);
  case 1:
    return cb(1);
  case 2:
    return cb(2);
  case 3:
    return cb(3);
  default:
    throw std::runtime_error(fmt::format("Invalid number of components: {}", size));
  }
}

template <typename T, bool OnlyRepresentable = true> inline auto visitStorageType(StorageType st, T cb) {
  if (OnlyRepresentable) {
    switch (st) {
    case StorageType::UInt8:
      return cb(uint8_t{});
    case StorageType::Int8:
      return cb(int8_t{});
    case StorageType::UInt16:
      return cb(uint16_t{});
    case StorageType::Int16:
      return cb(int16_t{});
    case StorageType::UInt32:
      return cb(uint32_t{});
    case StorageType::Int32:
      return cb(int32_t{});
    case StorageType::Float32:
      return cb(float{});
    default:
      throw std::runtime_error("Invalid or non-representable storage type");
    }
  } else {
    switch (st) {
    case StorageType::UInt8:
      return cb(uint8_t{});
    case StorageType::Int8:
      return cb(int8_t{});
    case StorageType::UNorm8:
      return cb(uint8_t{});
    case StorageType::SNorm8:
      return cb(int8_t{});
    case StorageType::UInt16:
      return cb(uint16_t{});
    case StorageType::Int16:
      return cb(int16_t{});
    case StorageType::UNorm16:
      return cb(uint16_t{});
    case StorageType::SNorm16:
      return cb(int16_t{});
    case StorageType::UInt32:
      return cb(uint32_t{});
    case StorageType::Int32:
      return cb(int32_t{});
    case StorageType::Float16:
      return cb(uint16_t{});
    case StorageType::Float32:
      return cb(float{});
    default:
      throw std::runtime_error("Invalid storage type");
    }
  }
}

struct AttribBuffer {
  uint8_t numComponents{1};
  StorageType type;
  std::vector<uint8_t> buf;
  size_t length{};

  size_t getSize() const { return numComponents * getStorageTypeSize(type); }

  void hash(XXH3_state_t &state, size_t index) const { XXH3_128bits_update(&state, buf.data() + getSize() * index, getSize()); }

  template <typename T, int N> void checkArgs() {
    if (numComponents != N)
      throw std::runtime_error("Invalid number of components");

    if constexpr (std::is_same_v<T, float>) {
      if (type != StorageType::Float32)
        throw std::runtime_error("Invalid storage type");
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      if (type != StorageType::UInt32)
        throw std::runtime_error("Invalid storage type");
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      if (type != StorageType::UInt16)
        throw std::runtime_error("Invalid storage type");
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      if (type != StorageType::UInt8)
        throw std::runtime_error("Invalid storage type");
    } else if constexpr (std::is_same_v<T, int8_t>) {
      if (type != StorageType::Int8)
        throw std::runtime_error("Invalid storage type");
    } else {
      static_assert(!std::is_same_v<T, T>, "Invalid type");
    }
  }

  template <typename T, int N, bool Checked = true> linalg::vec<T, N> read(size_t index) const {
    if constexpr (Checked) {
      checkArgs<T, N>();
    }

    return linalg::vec<T, N>((T *)(buf.data() + index * getSize()));
  }

  template <typename T, int N, bool Checked = true> void write(size_t index, const linalg::vec<T, N> &value) {
    if constexpr (Checked) {
      checkArgs<T, N>();
    }

    memcpy(buf.data() + index * getSize(), &value.x, getSize());
  }

  void initZero(StorageType type, uint8_t numComponents, size_t length) {
    this->type = type;
    this->numComponents = numComponents;
    this->length = length;
    this->buf.resize(length * getSize());
  }

  void initFrom(StorageType type, uint8_t numComponents, size_t length, size_t srcStride, const uint8_t *srcData) {
    initZero(type, numComponents, length);
    size_t attribSize = getSize();
    for (size_t i = 0; i < length; i++) {
      memcpy(this->buf.data() + attribSize * i, srcData + srcStride * i, attribSize);
    }
  }

  void initFromIndexedTriangleList(StorageType type, uint8_t numComponents, IndexFormat indexFormat, const uint8_t *indexData,
                                   size_t numIndices, size_t srcStride, const uint8_t *srcData) {
    initZero(type, numComponents, numIndices);
    size_t attribSize = getSize();
    if (indexFormat == IndexFormat::UInt16) {
      for (size_t i = 0; i < numIndices; i++) {
        uint32_t vertIndex = ((uint16_t *)indexData)[i];
        memcpy(this->buf.data() + attribSize * i, srcData + srcStride * vertIndex, attribSize);
      }
    } else {
      for (size_t i = 0; i < numIndices; i++) {
        uint32_t vertIndex = ((uint32_t *)indexData)[i];
        memcpy(this->buf.data() + attribSize * i, srcData + srcStride * vertIndex, attribSize);
      }
    }
  }

  void copyInto(uint8_t *dstData, size_t dstStride) const {
    size_t attribSize = getSize();
    for (size_t i = 0; i < length; i++) {
      memcpy(dstData + dstStride * i, buf.data() + attribSize * i, attribSize);
    }
  }

  static AttribBuffer generateIndexBuffer(size_t numTriangles) {
    AttribBuffer result;
    if (numTriangles < size_t(std::numeric_limits<uint16_t>::max())) {
      result.initZero(StorageType::UInt16, 1, numTriangles * 3);
      for (size_t t = 0; t < numTriangles * 3; t++) {
        ((uint16_t *)result.buf.data())[t] = t;
      }
    } else {
      result.initZero(StorageType::UInt32, 1, numTriangles * 3);
      for (size_t t = 0; t < numTriangles * 3; t++) {
        ((uint32_t *)result.buf.data())[t] = t;
      }
    }

    return result;
  }

  const uint8_t *data() const { return buf.data(); }
  uint8_t *data() { return buf.data(); }
};

// Reassembles a mesh out of a set of attributes and an optional index buffer
inline MeshPtr generateMesh(std::optional<AttribBuffer> indexBuffer,
                            boost::span<std::tuple<AttribBuffer *, FastString>> attributes) {
  MeshPtr newMesh = std::make_shared<Mesh>();
  MeshFormat format;
  format.primitiveType = PrimitiveType::TriangleList;
  format.windingOrder = WindingOrder::CCW;

  boost::container::small_vector<size_t, 16> dstOffsets;
  size_t stride{};
  for (size_t i = 0; i < attributes.size(); i++) {
    auto &attrib = std::get<0>(attributes[i]);
    auto &name = std::get<1>(attributes[i]);
    format.vertexAttributes.push_back(MeshVertexAttribute{name, attrib->numComponents, attrib->type});

    dstOffsets.push_back(stride);
    stride += attrib->numComponents * getStorageTypeSize(attrib->type);
  }

  std::vector<uint8_t> vertexBuffer;
  vertexBuffer.resize(stride * std::get<0>(attributes[0])->length);

  for (size_t i = 0; i < attributes.size(); i++) {
    auto &attrib = std::get<0>(attributes[i]);
    attrib->copyInto(vertexBuffer.data() + dstOffsets[i], stride);
  }

  if (indexBuffer.has_value()) {
    auto &ibAttribBuffer = indexBuffer.value();

    std::vector<uint8_t> indexBuffer;
    format.indexFormat = storageTypeToIndexFormat(ibAttribBuffer.type);
    indexBuffer.resize(getStorageTypeSize(ibAttribBuffer.type) * ibAttribBuffer.length);
    ibAttribBuffer.copyInto(indexBuffer.data(), getStorageTypeSize(ibAttribBuffer.type));

    newMesh->update(format, std::move(vertexBuffer), std::move(indexBuffer));
  } else {
    newMesh->update(format, std::move(vertexBuffer));
  }
#if SH_GFX_CONTEXT_DATA_LABELS
  newMesh->setLabel(fmt::format("wireframe"));
#endif

  return newMesh;
}

template <typename TVert, typename TIndex>
MeshPtr createMesh(const std::vector<TVert> &verts, const std::vector<TIndex> &indices) {
  static_assert(sizeof(TIndex) == 2 || sizeof(TIndex) == 4, "Invalid index format");

  MeshPtr mesh = std::make_shared<Mesh>();
  MeshFormat format{
      .indexFormat = (sizeof(TIndex) == 2) ? IndexFormat::UInt16 : IndexFormat::UInt32,
      .vertexAttributes = TVert::getAttributes(),
  };

  mesh->update(format, verts.data(), verts.size() * sizeof(TVert), indices.data(), indices.size() * sizeof(TIndex));
  return mesh;
}

MeshPtr generateLocalBasisAttribute(const MeshDescCPUCopy& mesh);
} // namespace gfx

#endif /* DC6B12F5_63B6_4FBD_BC92_DFF99E0E3323 */
