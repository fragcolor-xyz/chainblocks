#ifndef GFX_TEXTURE_PLACEHOLDER
#define GFX_TEXTURE_PLACEHOLDER

#include "context.hpp"
#include "context_data.hpp"
#include "enums.hpp"
#include "gfx_wgpu.hpp"
#include "texture.hpp"
#include <vector>

namespace gfx {

struct PlaceholderTexture {
  static TexturePtr create(TextureDimension dimension, int2 resolution, float4 fillColor) {
    size_t numLayerPixels = resolution.x * resolution.y;
    size_t layerSizeBytes = sizeof(uint32_t) * numLayerPixels;
    std::vector<uint8_t> data;
    auto fillLayer = [&](uint32_t *offset) {
      for (size_t p = 0; p < numLayerPixels; p++) {
        Color c = colorFromFloat(fillColor);
        uint32_t *pixel = offset + p;
        memcpy(pixel, &c.x, sizeof(uint32_t));
      }
    };

    TextureDescCPUCopy desc{};
    desc.format.resolution = resolution;
    desc.format.pixelFormat = WGPUTextureFormat_RGBA8UnormSrgb;
    desc.format.dimension = dimension;
    switch (dimension) {
    case TextureDimension::D1:
    case TextureDimension::D2: {
      data.resize(layerSizeBytes);
      fillLayer((uint32_t *)data.data());
    } break;
    case TextureDimension::Cube: {
      data.resize(layerSizeBytes * 6);
      for (size_t i = 0; i < 6; i++)
        fillLayer((uint32_t *)data.data() + numLayerPixels * i);
    } break;
    }

    desc.sourceData = std::move(data);

    TexturePtr texture = std::make_shared<Texture>();
    texture->init(desc);
    return texture;
  }
};
} // namespace gfx

#endif // GFX_TEXTURE_PLACEHOLDER
