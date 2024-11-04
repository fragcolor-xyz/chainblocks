#include "asset_loader.hpp"
#include "loaded_asset_tracker.hpp"
#include "data_format/drawable.hpp"
#include "data_format/texture.hpp"
#include "data_format/mesh.hpp"

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
    serialized.decodeImageData();
  }

  outData = std::move(serialized);
}

static void loadMesh(LoadedAssetData &outData, boost::span<uint8_t> source) {
  outData = shards::fromByteArray<SerializedMesh>(source);
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
  case AssetCategory::Image: {
    SerializedTexture &tex = std::get<SerializedTexture>(*request.data.get());
    tex.encodeImageData();
    shards::BufferRefWriterA writer(outData);
    shards::serde(writer, tex);
  } break;
  case AssetCategory::Mesh: {
    auto &mesh = std::get<SerializedMesh>(*request.data.get());
    shards::BufferRefWriterA writer(outData);
    shards::serdeConst(writer, mesh);
  } break;
  default:
    throw std::logic_error("Not implemented");
    break;
  }
}

void processAssetLoadFromStoreRequest(AssetLoadRequest &request, const AssetStoreRequest &inRequest) {
  switch (request.key.category) {
  case AssetCategory::Image: {
    auto &texture = std::get<SerializedTexture>(*inRequest.data.get());
    request.data = LoadedAssetData::makePtr(texture);
  } break;
  default:
    throw std::logic_error("Not implemented");
  }
}

} // namespace gfx::data