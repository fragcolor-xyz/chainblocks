#ifndef EF379F4B_AEF2_44CC_BF87_5A2D2EA9BBB5
#define EF379F4B_AEF2_44CC_BF87_5A2D2EA9BBB5

#include <shards/core/serialization/linalg.hpp>
#include <gfx/texture.hpp>
#include "base.hpp"

namespace gfx {

enum SerializedTextureDataFormat {
  // Raw pixel format
  RawPixels,
  // Any stb_image compatible file image format
  STBImageCompatible,
};

struct SerializedTextureHeader {
  std::string name;
  ::gfx::TextureFormat format;
  SerializedTextureDataFormat serializedFormat;
  uint8_t sourceChannels = 4;
  uint32_t sourceRowStride{};
};

struct SerializedTexture {
  SerializedTextureHeader header;
  ImmutableSharedBuffer diskImageData;
  ImmutableSharedBuffer rawImageData;

  SerializedTexture() = default;
  SerializedTexture(const TexturePtr &other) : SerializedTexture(*other) {}
  SerializedTexture(const gfx::Texture &texture) {
    if (auto desc = std::get_if<TextureDescCPUCopy>(&texture.getDesc())) {
      header.format = desc->format;
      header.sourceChannels = desc->sourceChannels;
      header.sourceRowStride = desc->sourceRowStride;
      header.serializedFormat = SerializedTextureDataFormat::RawPixels;
      rawImageData = ImmutableSharedBuffer(desc->sourceData.data(), desc->sourceData.size());
    } else {
      throw std::logic_error("Unsupported texture for serialization");
    }
  }
};
} // namespace gfx

namespace shards {
template <> struct Serialize<::gfx::TextureFormat> {
  template <SerializerStream S> void operator()(S &stream, ::gfx::TextureFormat &v) {
    serde(stream, v.resolution);
    serdeAs<uint8_t>(stream, v.dimension);
    serdeAs<uint8_t>(stream, v.flags);
    serdeAs<uint32_t>(stream, v.pixelFormat);
    serdeAs<uint8_t>(stream, v.mipLevels);
  }
};

template <> struct Serialize<gfx::SerializedTextureHeader> {
  template <SerializerStream S> void operator()(S &stream, gfx::SerializedTextureHeader &v) {
    serde(stream, v.name);
    serde(stream, v.format);
    serdeAs<uint8_t>(stream, v.serializedFormat);
    serdeAs<uint8_t>(stream, v.sourceChannels);
    serdeAs<uint32_t>(stream, v.sourceRowStride);
  }
};

template <> struct Serialize<gfx::SerializedTexture> {
  template <SerializerStream S> void operator()(S &stream, gfx::SerializedTexture &v) {
    serde(stream, v.header);
    switch (v.header.serializedFormat) {
    case gfx::SerializedTextureDataFormat::RawPixels:
      serde(stream, v.rawImageData);
      break;
    case gfx::SerializedTextureDataFormat::STBImageCompatible:
      serde(stream, v.diskImageData);
      break;
    }
  }
};
} // namespace shards

#endif /* EF379F4B_AEF2_44CC_BF87_5A2D2EA9BBB5 */
