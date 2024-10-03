#include "texture_file.hpp"
#include "spdlog/spdlog.h"
#include "../../core/assert.hpp"
#include <gfx/texture.hpp>
#include <memory>
#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#pragma GCC diagnostic pop

namespace gfx {
TexturePtr textureFromFile(const char *path) {
  TexturePtr texture;

  int2 size;
  int numComponentsInFile;
  if (!stbi_info(path, &size.x, &size.y, &numComponentsInFile)) {
    spdlog::error("Could not open image at path \"{}\"", path);
    return texture;
  }
  int numComponents = numComponentsInFile;

  // Extend to RGBA
  if (numComponents == 3)
    numComponents = 4;

  stbi_set_flip_vertically_on_load_thread(1);
  uint8_t *data = stbi_load(path, &size.x, &size.y, &numComponentsInFile, numComponents);
  if (data) {
    texture = std::make_shared<Texture>();
    TextureFormat format{};
    size_t pixelSize{};
    switch (numComponents) {
    case 1:
      format.pixelFormat = WGPUTextureFormat_R8Unorm;
      pixelSize = 1;
      break;
    case 2:
      format.pixelFormat = WGPUTextureFormat_RG8Unorm;
      pixelSize = 2;
      break;
    case 4:
      format.pixelFormat = WGPUTextureFormat_RGBA8Unorm;
      pixelSize = 4;
      break;
    default:
      shassert(false);
    }

    size_t dataSize = size.x * size.y * pixelSize;
    format.resolution = size;
    texture->init(TextureDescCPUCopy{
        .format = format,
        .sourceData = ImmutableSharedBuffer(data, dataSize, [](void *data, void *_) { stbi_image_free(data); }),
    });
  }

  return texture;
}
} // namespace gfx
