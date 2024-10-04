#include "asset_loader.hpp"
#include "loaded_asset_tracker.hpp"
#include "data_format/drawable.hpp"
#include "data_format/texture.hpp"
#include "data_format/mesh.hpp"

// For STB
#include <stb_image.h>
#include <stb_image_write.h>

namespace gfx::data {

template <AssetCategory Cat> void insertLoadedAssetData(LoadedAssetData &outData, boost::span<uint8_t> source) {
  throw std::logic_error("Not implemented");
}
static void loadDrawable(LoadedAssetData &outData, boost::span<uint8_t> source) {
  SerializedMeshDrawable serialized = shards::fromByteArray<SerializedMeshDrawable>(source);
  auto drawable = std::make_shared<MeshDrawable>();

  // drawable->mesh =
  auto loadedAssetTracker = getStaticLoadedAssetTracker();
  auto mesh = loadedAssetTracker->getOrInsert<Mesh>(serialized.mesh, [&]() {
    auto mesh = std::make_shared<Mesh>();
    mesh->update(MeshDescAsset{.format = serialized.meshFormat, .key = serialized.mesh});
    return mesh;
  });

  drawable->mesh = mesh;
  drawable->transform = serialized.transform.getMatrix();
  for (auto &param : serialized.params.basic) {
    drawable->parameters.basic[param.key] = param.value;
  }
  for (auto &param : serialized.params.textures) {
    auto texture = loadedAssetTracker->getOrInsert<Texture>(param.value, [&]() {
      auto texture = std::make_shared<Texture>();
      texture->init(TextureDescAsset{.format = param.format, .key = param.value});
      return texture;
    });
    drawable->parameters.textures.emplace(param.key, TextureParameter(texture, param.defaultTexcoordBinding));
  }
  outData = std::move(drawable);
}

static void loadImage(LoadedAssetData &outData, boost::span<uint8_t> source) {
  SerializedTexture serialized = shards::fromByteArray<SerializedTexture>(source);
  auto &hdr = serialized.header;
  TextureDescCPUCopy desc;
  if (hdr.serializedFormat == SerializedTextureDataFormat::RawPixels) {
    desc.format = hdr.format;
    desc.sourceData = std::move(serialized.diskImageData);
    desc.sourceChannels = hdr.sourceChannels;
    desc.sourceRowStride = hdr.sourceRowStride;
  } else {
    desc.format = hdr.format;
  }

  outData = std::move(desc);
}

static void loadMesh(LoadedAssetData &outData, boost::span<uint8_t> source) {
  SerializedMesh serialized = shards::fromByteArray<SerializedMesh>(source);
  outData = std::move(serialized.meshDesc);
}

static void storeMesh(AssetStoreRequest &request, shards::pmr::vector<uint8_t> &outData) {
  if (MeshDescCPUCopy *meshDesc = std::get_if<MeshDescCPUCopy>(request.data.get())) {
    shards::BufferRefWriterA writer(outData);
    shards::serdeConst(writer, SerializedMesh{*meshDesc});
  } else {
    throw std::logic_error("Invalid data type for mesh");
  }
}

static void decodeImageData(SerializedTexture &serialized) {
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

static void encodeImageData(SerializedTexture &serialized) {
  if (serialized.diskImageData) {
    return;
  }
  shassert(!serialized.rawImageData && "Image data is empty");

  std::vector<uint8_t> pngData;
  pngData.reserve(1024 * 1024 * 4);
  auto writePng = [](void *context, void *data, int size) {
    auto &pngData = reinterpret_cast<std::vector<uint8_t> &>(context);
    size_t ofs = pngData.size();
    pngData.resize(pngData.size() + size);
    memcpy(pngData.data() + ofs, data, size);
  };
  auto &fmt = serialized.header.format;
  stbi_write_png_to_func(writePng, &pngData, fmt.resolution.x, fmt.resolution.y, 4, serialized.rawImageData.data(),
                         serialized.header.sourceRowStride);
  serialized.diskImageData = ImmutableSharedBuffer(pngData.data(), pngData.size());
  serialized.header.serializedFormat = SerializedTextureDataFormat::STBImageCompatible;
}

static LoadedAssetDataPtr loadedImageAsset(SerializedTexture &ser) {
  auto resultData = TextureDescCPUCopy();
  resultData.format = ser.header.format;
  resultData.sourceChannels = ser.header.sourceChannels;
  resultData.sourceRowStride = ser.header.sourceRowStride;
  if (!ser.rawImageData) {
    decodeImageData(ser);
    shassert(ser.rawImageData);
  }
  resultData.sourceData = std::move(ser.rawImageData);
  return LoadedAssetData::makePtr(resultData);
}

void finializeAssetLoadRequest(AssetLoadRequest &request, boost::span<uint8_t> source) {
  request.data = std::make_shared<LoadedAssetData>();
  auto &outData = *request.data;
  switch (request.key.category) {
  case gfx::data::AssetCategory::Drawable:
    loadDrawable(outData, source);
    break;
  case gfx::data::AssetCategory::Image:
    loadImage(outData, source);
    break;
  case gfx::data::AssetCategory::Mesh:
    loadMesh(outData, source);
    break;
  default:
    throw std::logic_error("Not implemented");
    break;
  }
}

void processAssetStoreRequest(AssetStoreRequest &request, shards::pmr::vector<uint8_t> &outData) {
  if (auto *rawData = std::get_if<std::vector<uint8_t>>(request.data.get())) {
    outData.resize(rawData->size());
    std::memcpy(outData.data(), rawData->data(), rawData->size());
    return;
  }

  switch (request.key.category) {
  case AssetCategory::Image:
    if (auto *ser = std::get_if<SerializedTexture>(request.data.get())) {
      // encodeImageData(std::get<SerializedTexture>(*request.data.get()));
      // storeImage(request, outData);
    }
    break;
  case AssetCategory::Mesh:
    storeMesh(request, outData);
    break;
  default:
    throw std::logic_error("Not implemented");
    break;
  }
}

void processAssetLoadFromStoreRequest(AssetLoadRequest &request, const AssetStoreRequest &inRequest) {
  switch (request.key.category) {
  case AssetCategory::Image:
    if (std::holds_alternative<TextureDescCPUCopy>(*inRequest.data.get())) {
      request.data = inRequest.data;
    } else if (auto ser = std::get_if<SerializedTexture>(inRequest.data.get())) {
      request.data = loadedImageAsset(*ser);
    } else if (auto rawData = std::get_if<std::vector<uint8_t>>(inRequest.data.get())) {
      SerializedTexture st = shards::fromByteArray<SerializedTexture>(*rawData);
      request.data = loadedImageAsset(st);
    } else {
      throw std::logic_error("Invalid data type for image");
    }
    break;
  default:
    throw std::logic_error("Not implemented");
  }
}

} // namespace gfx::data