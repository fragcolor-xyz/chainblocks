#include "./context.hpp"
#include "./data.hpp"
#include "./renderer.hpp"
#include "renderer_utils.hpp"
#include <catch2/catch_all.hpp>
#include <thread>

#include <gfx/filesystem.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/geom.hpp>
#include <gfx/data_cache/data_cache.hpp>
#include <gfx/data_cache/data_cache_impl.hpp>
#include <gfx/data_cache/derived.hpp>
#include <gfx/data_format/mesh.hpp>
#include <gfx/texture_file/texture_file.hpp>
#include <gfx/paths.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#include <boost/uuid/uuid_generators.hpp>

static auto setupCache() {

  gfx::fs::Path cachePath = ".shards";
  if (gfx::fs::exists(cachePath)) {
    gfx::fs::remove_all(cachePath);
  }

  auto cacheIO = gfx::data::createShardsDataCacheIO(cachePath.string());
  std::shared_ptr<gfx::data::DataCache> cache = std::make_shared<gfx::data::DataCache>(cacheIO);

  return cache;
}

static gfx::data::AssetInfo blankAssetKey(gfx::data::AssetCategory category) {
  gfx::data::AssetInfo info;
  info.category = category;
  return info;
}

TEST_CASE("Mesh format", "[DataFormats]") {
  std::shared_ptr<gfx::data::DataCache> cache = setupCache();

  gfx::geom::SphereGenerator sphereGen;
  sphereGen.generate();

  // Create and write the mesh
  auto mesh = gfx::createMesh(sphereGen.vertices, sphereGen.indices);

  gfx::SerializedMesh smesh{.meshDesc = std::get<gfx::MeshDescCPUCopy>(mesh->getDesc())};
  auto op = cache->store(blankAssetKey(gfx::data::AssetCategory::Mesh), gfx::data::LoadedAssetData::makePtr(smesh));
  auto key = op->key;

  // Make sure that the asset is available as soon as it is pending store
  // This will allow other usages to load the asset already
  CHECK(cache->hasAsset(key));

  while (!op->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(op->isSuccess());
  CHECK(cache->hasAsset(key));

  auto loadOp = cache->load(key);
  CHECK(!loadOp->isFinished());
  while (!loadOp->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  CHECK(loadOp->isSuccess());
  CHECK(loadOp->data);
  CHECK(std::holds_alternative<gfx::SerializedMesh>(*loadOp->data));

  // Validate that the hash is still the same
  auto serializedMesh = std::get<gfx::SerializedMesh>(*loadOp->data);
  auto reserializedData = shards::toByteArray(serializedMesh);
  gfx::data::AssetInfo newCheckKey = cache->generateSourceKey(boost::span(reserializedData), gfx::data::AssetCategory::Mesh);

  CHECK(newCheckKey.key == key.key);
}

TEST_CASE("Texture Format", "[DataFormats]") {
  std::shared_ptr<gfx::data::DataCache> cache = setupCache();

  auto logoPath = gfx::resolveDataPath("shards/gfx/tests/assets/logo.png").string();
  auto logoTexture = gfx::textureFromFile(logoPath.c_str());

  gfx::SerializedTexture stex(logoTexture);
  stex.header.name = "logo";
  std::vector<uint8_t> data = shards::toByteArray(stex);
  auto op = cache->store(blankAssetKey(gfx::data::AssetCategory::Image), gfx::data::LoadedAssetData::makePtr(std::move(data)));
  auto key = op->key;

  // Make sure that the asset is available as soon as it is pending store
  // This will allow other usages to load the asset already
  CHECK(cache->hasAsset(key));

  // Load asset from pending write operation
  auto loadedFromStore = cache->load(key);
  CHECK(loadedFromStore->state == (uint8_t)gfx::data::AssetLoadRequestState::Pending);

  while (!op->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(op->isSuccess());
  CHECK(cache->hasAsset(key));

  // Check that the pending write request also succeeded
  CHECK(loadedFromStore->isSuccess());

  auto loadOp = cache->load(key);
  CHECK(!loadOp->isFinished());
  while (!loadOp->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  CHECK(loadOp->isSuccess());
  CHECK(loadOp->data);
  CHECK(std::holds_alternative<gfx::SerializedTexture>(*loadOp->data));

  // Validate that the hash is still the same
  auto loadedTextureDesc = std::get<gfx::SerializedTexture>(*loadOp->data);
  auto reserializedData = shards::toByteArray(loadedTextureDesc);
  gfx::data::AssetInfo newCheckKey = cache->generateSourceKey(boost::span(reserializedData), gfx::data::AssetCategory::Image);

  CHECK(newCheckKey.key == key.key);
}

TEST_CASE("Texture Format (Compressed)", "[DataFormats]") {
  std::shared_ptr<gfx::data::DataCache> cache = setupCache();

  auto logoPath = gfx::resolveDataPath("shards/gfx/tests/assets/logo.png").string();
  auto logoTexture = gfx::textureFromFile(logoPath.c_str());

  // gfx::SerializedTexture stex(logoTexture);
  // stex.header.serializedFormat = gfx::SerializedTextureDataFormat::STBImageCompatible;
  // stex.header.name = "logo";
  // std::vector<uint8_t> data = shards::toByteArray(stex);
  auto &textureDesc = std::get<gfx::TextureDescCPUCopy>(logoTexture->getDesc());
  boost::uuids::random_generator uuidgen;

  // -- TEST TextureDescCPUCopy
  gfx::data::AssetKey key0(gfx::data::AssetCategory::Image, uuidgen());
  auto op0 = cache->store(key0, gfx::data::LoadedAssetData::makePtr(textureDesc));

  // Make sure that the asset is available as soon as it is pending store
  // This will allow other usages to load the asset already
  CHECK(cache->hasAsset(key0));

  // Load asset from pending write operation
  auto loadedFromStore0 = cache->load(key0);
  CHECK(loadedFromStore0->state == (uint8_t)gfx::data::AssetLoadRequestState::Pending);

  // -- TEST SerializedTexture
  gfx::data::AssetKey key1(gfx::data::AssetCategory::Image, uuidgen());
  auto op1 = cache->store(key1, gfx::data::LoadedAssetData::makePtr(gfx::SerializedTexture(logoTexture)));
  CHECK(cache->hasAsset(key1));

  // Load asset from pending write operation
  auto loadedFromStore1 = cache->load(key1);
  CHECK(loadedFromStore1->state == (uint8_t)gfx::data::AssetLoadRequestState::Pending);

  while (!op0->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(op0->isSuccess());
  CHECK(cache->hasAsset(key0));

  while (!op1->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(op1->isSuccess());
  CHECK(cache->hasAsset(key1));

  // Check that the pending write request also succeeded
  while (!loadedFromStore0->isFinished() || !loadedFromStore1->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(loadedFromStore0->isSuccess());
  CHECK(loadedFromStore1->isSuccess());

  auto loadOp = cache->load(key0);
  CHECK(!loadOp->isFinished());
  while (!loadOp->isFinished())
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  CHECK(loadOp->isSuccess());
  CHECK(loadOp->data);
  CHECK(std::holds_alternative<gfx::SerializedTexture>(*loadOp->data));

  // Validate that the hash is still the same
  auto serializedTexture = std::get<gfx::SerializedTexture>(*loadOp->data);
  CHECK(serializedTexture.rawImageData.size() == textureDesc.sourceData.size());
  auto dim = serializedTexture.header.format.resolution;
  for (int y = 0; y < dim.y; y++) {
    uint32_t *rowOld = (uint32_t *)(textureDesc.sourceData.data() + y * textureDesc.sourceRowStride);
    uint32_t *rowNew = (uint32_t *)(serializedTexture.rawImageData.data() + y * serializedTexture.header.sourceRowStride);
    for (int x = 0; x < dim.x; x++) {
      CHECK(rowNew[x] == rowOld[x]);
      if (rowNew[x] != rowOld[x]) {
        SPDLOG_ERROR("Mismatch at ({}, {})", x, y);
        y = dim.y;
        break;
      }
    }
  }
}

#include <cmp_core.h>

struct TestGenerator : public gfx::data::IDerivedDataGenerator {
  static constexpr uint64_t StaticID = GFX_DATA_ID64('Test', 'Gen0');
  static std::shared_ptr<TestGenerator> &getInstance() {
    static std::shared_ptr<TestGenerator> instance = std::make_shared<TestGenerator>();
    return instance;
  }
  uint64_t getID() override { return StaticID; }
  gfx::data::LoadedAssetDataPtr generate(const gfx::data::LoadedAssetDataPtr &input) override {
    // Only process texture data
    if (!std::holds_alternative<gfx::SerializedTexture>(*input)) {
      return input;
    }

    auto &texture = std::get<gfx::SerializedTexture>(*input);
    auto dim = texture.header.format.resolution;

    // Only compress if dimensions are multiples of 4
    if ((dim.x % 4) != 0 || (dim.y % 4) != 0) {
      return input;
    }

    // Allocate compressed buffer
    size_t numBlocks = (dim.x * dim.y) / 16; // 4x4 blocks
    std::vector<uint8_t> compressedData(numBlocks * 16);

    // Process 4x4 blocks
    for (int y = 0; y < dim.y; y += 4) {
      for (int x = 0; x < dim.x; x += 4) {
        uint8_t blockData[64]; // 4x4 block * 4 bytes per pixel = 64 bytes

        // Extract 4x4 block data
        for (int by = 0; by < 4; by++) {
          uint32_t *row = (uint32_t *)(texture.rawImageData.data() + (y + by) * texture.header.sourceRowStride);
          for (int bx = 0; bx < 4; bx++) {
            uint32_t pixel = row[x + bx];
            int idx = (by * 4 + bx) * 4;
            blockData[idx + 0] = (pixel >> 0) & 0xFF;  // R
            blockData[idx + 1] = (pixel >> 8) & 0xFF;  // G
            blockData[idx + 2] = (pixel >> 16) & 0xFF; // B
            blockData[idx + 3] = (pixel >> 24) & 0xFF; // A
          }
        }

        // Compress block
        size_t blockIdx = (y / 4) * (dim.x / 4) + (x / 4);
        CompressBlockBC7(blockData, 16, &compressedData[blockIdx * 16], nullptr);
      }
    }

    // Create compressed texture
    gfx::SerializedTexture compressedTex;
    compressedTex.header = texture.header;
    compressedTex.header.format.pixelFormat = WGPUTextureFormat_BC7RGBAUnorm;
    compressedTex.rawImageData = gfx::ImmutableSharedBuffer(compressedData.data(), compressedData.size());

    return gfx::data::LoadedAssetData::makePtr(compressedTex);

    return input;
  }
};

TEST_CASE("Derived Texture", "[DataFormats]") {
  std::shared_ptr<gfx::data::DataCache> cache = setupCache();
  cache->getDerivedDataGeneratorRegistry()->register_(TestGenerator::getInstance());

  auto logoPath = gfx::resolveDataPath("shards/assets/ShardsLogo_1024.png").string();
  auto logoTexture = gfx::textureFromFile(logoPath.c_str());

  // Load asset from pending write operation
  gfx::SerializedTexture stex(logoTexture);
  auto op0 = cache->store(blankAssetKey(gfx::data::AssetCategory::Image), gfx::data::LoadedAssetData::makePtr(stex));
  CHECK(cache->hasAsset(op0->key));

  cache->flushRequests();
  cache->flushAssets();

  auto derivedKey = op0->key;
  derivedKey.rootAsset = op0->key.key;
  derivedKey.flags = gfx::data::AssetFlags::AllowGC;
  auto op1 = cache->load(derivedKey);
  CHECK(op1->state == (uint8_t)gfx::data::AssetLoadRequestState::Pending);
}
