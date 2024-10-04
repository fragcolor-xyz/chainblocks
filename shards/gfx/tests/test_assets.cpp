#include "./context.hpp"
#include "./data.hpp"
#include "./renderer.hpp"
#include "renderer_utils.hpp"
#include <catch2/catch_all.hpp>

#include <gfx/filesystem.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/geom.hpp>
#include <gfx/data_cache/data_cache.hpp>
#include <gfx/data_cache/data_cache_impl.hpp>
#include <gfx/data_format/mesh.hpp>
#include <gfx/texture_file/texture_file.hpp>
#include <gfx/paths.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>

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
  std::vector<uint8_t> data = shards::toByteArray(smesh);
  auto op = cache->store(blankAssetKey(gfx::data::AssetCategory::Mesh), gfx::data::LoadedAssetData::makePtr(std::move(data)));
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
  CHECK(std::holds_alternative<gfx::MeshDescCPUCopy>(*loadOp->data));

  // Validate that the hash is still the same
  auto loadedMeshDesc = std::get<gfx::MeshDescCPUCopy>(*loadOp->data);
  auto reserializedData = shards::toByteArray(gfx::SerializedMesh{.meshDesc = loadedMeshDesc});
  gfx::data::AssetInfo newCheckKey = cache->generateSourceKey(boost::span(reserializedData), gfx::data::AssetCategory::Mesh);

  CHECK(newCheckKey.key == key.key);
}

TEST_CASE("Texture Format", "[DataFormats]") {

  std::shared_ptr<gfx::data::DataCache> cache = setupCache();

  auto logoPath = gfx::resolveDataPath("shards/gfx/tests/assets/logo.png").string();
  auto logoTexture = gfx::textureFromFile(logoPath.c_str());

  gfx::SerializedTexture stex(logoTexture);
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
  CHECK(std::holds_alternative<gfx::TextureDescCPUCopy>(*loadOp->data));

  // Validate that the hash is still the same
  auto loadedMeshDesc = std::get<gfx::MeshDescCPUCopy>(*loadOp->data);
  auto reserializedData = shards::toByteArray(gfx::SerializedMesh{.meshDesc = loadedMeshDesc});
  gfx::data::AssetInfo newCheckKey = cache->generateSourceKey(boost::span(reserializedData), gfx::data::AssetCategory::Mesh);

  CHECK(newCheckKey.key == key.key);
}
