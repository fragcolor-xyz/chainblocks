#include "texture.hpp"

// For STB
#include <stb_image.h>
#include <stb_image_write.h>

namespace gfx {

void SerializedTexture::decodeImageData() {
  SerializedTexture &serialized = *this;
  if (serialized.rawImageData) {
    return;
  }
  if (!serialized.diskImageData) {
    throw std::logic_error("Image data is empty");
  }

  int2 loadedImageSize;
  int loadedNumChannels{};
  stbi_set_flip_vertically_on_load_thread(1);
  uint8_t *data = stbi_load_from_memory(serialized.diskImageData.data(), serialized.diskImageData.size(), &loadedImageSize.x,
                                        &loadedImageSize.y, &loadedNumChannels, 4);
  if (!data)
    throw std::logic_error(fmt::format("Failed to decode image data: {}", stbi_failure_reason()));
  serialized.rawImageData = ImmutableSharedBuffer(
      data, loadedImageSize.x * loadedImageSize.y * 4, [](void *data, void *user) { stbi_image_free(data); }, nullptr);
  serialized.header.sourceChannels = 4;
  serialized.header.sourceRowStride = loadedImageSize.x * 4;
}

void SerializedTexture::encodeImageData() {
  SerializedTexture &serialized = *this;
  if (serialized.diskImageData) {
    return;
  }
  shassert(serialized.rawImageData && "Image data is empty");

  using Vec = std::vector<uint8_t>;
  Vec pngData;
  pngData.reserve(1024 * 1024 * 4);
  auto writePng = [](void *context, void *data, int size) {
    auto &pngData = *reinterpret_cast<Vec *>(context);
    size_t ofs = pngData.size();
    pngData.resize(pngData.size() + size);
    memcpy(pngData.data() + ofs, data, size);
  };
  stbi_flip_vertically_on_write(1);
  auto &fmt = serialized.header.format;
  stbi_write_png_to_func(writePng, &pngData, fmt.resolution.x, fmt.resolution.y, 4, serialized.rawImageData.data(),
                         serialized.header.sourceRowStride);
  serialized.diskImageData = ImmutableSharedBuffer(pngData.data(), pngData.size());
  serialized.header.serializedFormat = SerializedTextureDataFormat::STBImageCompatible;
}
} // namespace gfx