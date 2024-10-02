#ifndef EBC1B4F9_8C5A_4721_8F9B_B9072C5BC5DD
#define EBC1B4F9_8C5A_4721_8F9B_B9072C5BC5DD

#include "context_data.hpp"
#include "wgpu_handle.hpp"
#include "gfx_wgpu.hpp"
#include "isb.hpp"
#include "linalg.hpp"
#include "unique_id.hpp"
#include "fwd.hpp"
#include "enums.hpp"
#include "log.hpp"
#include "data_cache/types.hpp"
#include <compare>
#include <variant>
#include <optional>
#include <string>
#include <boost/container_hash/hash_fwd.hpp>
#include <boost/tti/has_member_data.hpp>

namespace gfx {
struct SamplerState {
  WGPUAddressMode addressModeU : 8 = WGPUAddressMode::WGPUAddressMode_Repeat;
  WGPUAddressMode addressModeV : 8 = WGPUAddressMode::WGPUAddressMode_Repeat;
  WGPUAddressMode addressModeW : 8 = WGPUAddressMode::WGPUAddressMode_Repeat;
  WGPUFilterMode filterMode : 8 = WGPUFilterMode::WGPUFilterMode_Linear;

  template <typename T> void getPipelineHash(T &hasher) const {
    hasher(addressModeU);
    hasher(addressModeV);
    hasher(addressModeW);
    hasher(filterMode);
  }

  std::strong_ordering operator<=>(const SamplerState &) const = default;

  friend size_t hash_value(SamplerState const &v) {
    static_assert(sizeof(SamplerState) == sizeof(uint32_t), "SamplerState is not the same size as uint32_t");
    return reinterpret_cast<const uint32_t &>(v);
  }
};

struct TextureFormat {
  int2 resolution;
  TextureDimension dimension = TextureDimension::D2;
  TextureFormatFlags flags = TextureFormatFlags::AutoGenerateMips;
  WGPUTextureFormat pixelFormat = WGPUTextureFormat::WGPUTextureFormat_Undefined;
  uint8_t mipLevels = 1;

  bool hasMips() { return mipLevels > 1; }
  std::strong_ordering operator<=>(const TextureFormat &other) const = default;

  static TextureFormat deriveFrom(size_t numComponents, StorageType storageType, bool preferSrgb);
};

struct InputTextureFormat {
  uint8_t pixelSize;
  size_t numComponents;
};

/// <div rustbindgen opaque></div>
struct TextureContextData : public ContextData {
  std::weak_ptr<Texture> keepAliveRef;
  TextureFormat format{};
  // Only set for managed textures
  WgpuHandle<WGPUTexture> texture{};
  // Only set for externally managed textures, not freed
  WGPUTexture externalTexture{};
  WGPUExtent3D size{};
  UniqueId id{};

#if SH_GFX_CONTEXT_DATA_LABELS
  std::string label;
  std::string_view getLabel() const { return label; }
#endif

  TextureContextData() = default;
  TextureContextData(TextureContextData &&) = default;

  void init(std::string_view label) {
#if SH_GFX_CONTEXT_DATA_LABELS
    this->label = label;
#endif
#if SH_GFX_CONTEXT_DATA_LOG_LIFETIME
    SPDLOG_LOGGER_DEBUG(getContextDataLogger(), "Texture {} data created", getLabel());
#endif
  }

#if SH_GFX_CONTEXT_DATA_LOG_LIFETIME
  ~TextureContextData() {
    if (texture)
      SPDLOG_LOGGER_DEBUG(getContextDataLogger(), "Texture {} data destroyed", getLabel());
  }
#endif
};

// Describes an asset loaded from the asset cache
struct TextureDescAsset {
  TextureFormat format;
  data::AssetKey key;
  std::strong_ordering operator<=>(const TextureDescAsset &other) const = default;
};

// Describes a texture that is initialized from a CPU copy of the texture data
struct TextureDescCPUCopy {
  TextureFormat format;
  uint8_t sourceChannels = 4;
  uint32_t sourceRowStride{};
  ImmutableSharedBuffer sourceData;

  std::strong_ordering operator<=>(const TextureDescCPUCopy &other) const = default;
};

// Describes texture data that is kept on the GPU side only
struct TextureDescGPUOnly {
  TextureFormat format;

  std::strong_ordering operator<=>(const TextureDescGPUOnly &other) const = default;
  static TextureDescGPUOnly getDefault();
};

// Describes an external WGPUTexture object
struct TextureDescExternal {
  TextureFormat format;
  std::optional<WGPUTexture> externalTexture;
  std::strong_ordering operator<=>(const TextureDescExternal &other) const = default;
};

/// <div rustbindgen hide></div>
struct TextureDesc : public std::variant<TextureDescGPUOnly, TextureDescCPUCopy, TextureDescAsset, TextureDescExternal> {
  using variant::variant;

  bool isExternal() const { return std::holds_alternative<TextureDescExternal>(*this); }
  bool isAsset() const { return std::holds_alternative<TextureDescAsset>(*this); }
  bool isGPUOnly() const { return std::holds_alternative<TextureDescGPUOnly>(*this); }
  bool isCPUCopy() const { return std::holds_alternative<TextureDescCPUCopy>(*this); }

  const TextureFormat &getFormat() const {
    return std::visit([](auto &arg) -> const TextureFormat & { return arg.format; }, *this);
  }

  TextureFormat &getFormat() {
    return std::visit([](auto &arg) -> TextureFormat & { return arg.format; }, *this);
  }

  int2 getResolution() const { return getFormat().resolution; }
  WGPUTextureFormat getPixelFormat() const { return getFormat().pixelFormat; }
};

std::vector<uint8_t> convertToRGBA(const TextureDescCPUCopy &desc);

/// <div rustbindgen opaque></div>
struct Texture final {
  using ContextDataType = TextureContextData;

private:
  UniqueId id = getNextId();
  TextureDesc desc = TextureDescGPUOnly::getDefault();
  SamplerState samplerState{};
  std::string label;
  size_t version{};

  friend struct gfx::UpdateUniqueId<Texture>;

public:
  Texture() = default;
  Texture(std::string &&label) : label(label) {}

#if SH_GFX_CONTEXT_DATA_LOG_LIFETIME
  ~Texture() { SPDLOG_LOGGER_DEBUG(getContextDataLogger(), "Texture {} ({}) destroyed", label, id); }
#endif

  // Creates a texture
  Texture &init(const TextureDesc &desc);
  Texture &initWithSamplerState(const SamplerState &samplerState);
  /// <div rustbindgen hide></div>
  Texture &initWithLabel(std::string &&label);

  SamplerState &getSamplerState() { return samplerState; }
  const std::string &getLabel() const { return label; }
  const TextureDesc &getDesc() const { return desc; }

  /// <div rustbindgen hide></div>
  const TextureFormat &getFormat() const { return desc.getFormat(); }
  int2 getResolution() const { return getFormat().resolution; }
  WGPUTextureFormat getPixelFormat() const { return getFormat().pixelFormat; }

  // Causes an update of renderer side data
  void update();

  TexturePtr clone() const;

  UniqueId getId() const { return id; }
  size_t getVersion() const { return version; }

  bool keepAlive() const { return true; }

  /// <div rustbindgen hide></div>
  static TexturePtr makeRenderAttachment(WGPUTextureFormat format, std::string &&label);

  void initContextData(Context &context, TextureContextData &contextData);
  void updateContextData(Context &context, TextureContextData &contextData);

protected:
  static UniqueId getNextId();
};

static_assert(TWithContextDataKeepAlive<Texture>, "");

} // namespace gfx

#endif /* EBC1B4F9_8C5A_4721_8F9B_B9072C5BC5DD */
