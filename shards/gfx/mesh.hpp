#ifndef A9FF2947_97BD_4662_A743_E2DB36A32FA2
#define A9FF2947_97BD_4662_A743_E2DB36A32FA2

#include "context_data.hpp"
#include "enums.hpp"
#include "gfx_wgpu.hpp"
#include "wgpu_handle.hpp"
#include "linalg/linalg.h"
#include "fwd.hpp"
#include "unique_id.hpp"
#include "log.hpp"
#include "data_cache/types.hpp"
#include "isb.hpp"
#include <string>
#include <vector>
#include <optional>
#include <variant>

namespace gfx {

namespace detail {
struct PipelineHashCollector;
}

/// <div rustbindgen opaque></div>
struct MeshVertexAttribute {
  FastString name;
  uint8_t numComponents;
  StorageType type;

  MeshVertexAttribute() = default;
  MeshVertexAttribute(FastString name, uint8_t numComponents, StorageType type = StorageType::Float32)
      : name(name), numComponents(numComponents), type(type) {}

  // Compares everthing except for the name
  bool isSameDataFormat(const MeshVertexAttribute &other) const {
    return numComponents == other.numComponents && type == other.type;
  }

  std::strong_ordering operator<=>(const MeshVertexAttribute &other) const = default;

  template <typename T> void getPipelineHash(T &hasher) const {
    hasher(name);
    hasher(numComponents);
    hasher(type);
  }
};

/// <div rustbindgen opaque></div>
struct MeshFormat {
  PrimitiveType primitiveType = PrimitiveType::TriangleList;
  WindingOrder windingOrder = WindingOrder::CCW;
  IndexFormat indexFormat = IndexFormat::UInt16;
  std::vector<MeshVertexAttribute> vertexAttributes;

  size_t getVertexSize() const;

  template <typename T> void getPipelineHash(T &hasher) const {
    hasher(primitiveType);
    hasher(windingOrder);
    hasher(indexFormat);
    hasher(vertexAttributes);
  }

  std::strong_ordering operator<=>(const MeshFormat &other) const = default;
};

/// <div rustbindgen opaque></div>
struct MeshContextData : public ContextData {
  MeshFormat format;
  size_t numVertices = 0;
  size_t numIndices = 0;
  WgpuHandle<WGPUBuffer> vertexBuffer;
  size_t vertexBufferLength = 0;
  WgpuHandle<WGPUBuffer> indexBuffer;
  size_t indexBufferLength = 0;

#if SH_GFX_CONTEXT_DATA_LABELS
  std::string label;
  std::string_view getLabel() const { return label; }
#endif

  MeshContextData() = default;
  MeshContextData(MeshContextData &&) = default;

  void init(std::string_view label) {
#if SH_GFX_CONTEXT_DATA_LABELS
    this->label = label;
#endif
#if SH_GFX_CONTEXT_DATA_LOG_LIFETIME
    SPDLOG_LOGGER_DEBUG(getContextDataLogger(), "Mesh {} data created", getLabel());
#endif
  }

#if SH_GFX_CONTEXT_DATA_LOG_LIFETIME
  ~MeshContextData() {
    if (vertexBuffer)
      SPDLOG_LOGGER_DEBUG(getContextDataLogger(), "Mesh {} data destroyed", getLabel());
  }
#endif
};

// Add these new structures
struct MeshDescCPUCopy {
  MeshFormat format;
  ImmutableSharedBuffer vertexData;
  ImmutableSharedBuffer indexData;

  size_t getNumIndices() const;
  size_t getNumVertices() const;

  std::strong_ordering operator<=>(const MeshDescCPUCopy &other) const = default;
};

struct MeshDescAsset {
  MeshFormat format;
  data::AssetKey key;
  std::strong_ordering operator<=>(const MeshDescAsset &other) const = default;
};

struct MeshDesc : public std::variant<MeshDescCPUCopy, MeshDescAsset> {
  using variant::variant;
};

// Modify the Mesh struct
struct Mesh final {
  using ContextDataType = MeshContextData;

private:
  UniqueId id = getNextId();
  MeshDesc desc;
  size_t version{};
#if SH_GFX_CONTEXT_DATA_LABELS
  std::string label;
#endif

  friend struct gfx::UpdateUniqueId<Mesh>;

public:
#if SH_GFX_CONTEXT_DATA_LABELS
  void setLabel(std::string_view label) {
    this->label = label;
  }
  
  std::string_view getLabel() const {
    return label;
  }
#else
  void setLabel(std::string_view) {}
  
  std::string_view getLabel() const {
    return "<unknown>";
  }
#endif
  const MeshDesc &getDesc() const { return desc; }
  const MeshFormat &getFormat() const {
    return std::visit([](const auto &d) -> const MeshFormat & { return d.format; }, desc);
  }

  // This increments every time the mesh is updated
  size_t getVersion() const { return version; }
  // Updates mesh data with length in bytes
  void update(const MeshFormat &format, const void *inVertexData, size_t vertexDataLength, const void *inIndexData,
              size_t indexDataLength);
  void update(const MeshFormat &format, std::vector<uint8_t> &&vertexData,
              std::vector<uint8_t> &&indexData = std::vector<uint8_t>());
  void update(MeshDesc desc);

  UniqueId getId() const { return id; }
  MeshPtr clone() const;

  void pipelineHashCollect(detail::PipelineHashCollector &) const;

  void initContextData(Context &context, MeshContextData &contextData);
  void updateContextData(Context &context, MeshContextData &contextData);

protected:
  void calculateElementCounts(size_t vertexDataLength, size_t indexDataLength, size_t vertexSize, size_t indexSize);
  void update();

  static UniqueId getNextId();
};

} // namespace gfx

#endif /* A9FF2947_97BD_4662_A743_E2DB36A32FA2 */
