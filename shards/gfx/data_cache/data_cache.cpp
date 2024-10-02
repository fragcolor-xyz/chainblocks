#include "data_cache.hpp"
#include "../hasherxxh3.hpp"
#include <tbb/concurrent_map.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <shards/core/pmr/shared_temp_allocator.hpp>

namespace fs {
using namespace boost::filesystem;
using Path = boost::filesystem::path;
} // namespace fs
namespace gfx::data {

struct TrackedFile {
  uint64_t lwt;
  Hash128 derivedHash;
};

struct TrackedFileCache {
  tbb::concurrent_map<std::string, TrackedFile> files;
};

inline boost::uuids::uuid hash128ToUuid(Hash128 hash) {
  boost::uuids::uuid uuid;
  std::memcpy(uuid.data, &hash, sizeof(Hash128));
  return uuid;
}

DataCache::DataCache(std::shared_ptr<IDataCacheIO> io) : io(io) { trackedFiles = std::make_shared<TrackedFileCache>(); }
AssetInfo DataCache::generateSourceKey(std::string_view path, AssetCategory category) {
  auto filepath = fs::Path(path.begin(), path.end()).lexically_normal();
  auto filepathStr = filepath.string();
  auto &collection = trackedFiles->files;
  auto it = collection.find(filepathStr);

  std::optional<Hash128> filehash;
  auto lastWriteTime = uint64_t(fs::last_write_time(filepath));
  if (it != collection.end()) {
    if (it->second.lwt == lastWriteTime) {
      filehash = it->second.derivedHash;
    }
  }

  if (!filehash) {
    uint64_t lastWriteTimeCheck;
    shards::pmr::SharedTempAllocator allocator;
    do {
      lastWriteTime = uint64_t(fs::last_write_time(filepath));

      std::fstream inFile(filepathStr, std::ios::in | std::ios::binary);
      shards::pmr::vector<uint8_t> data(allocator.getAllocator());
      inFile.seekg(0, std::ios::end);
      data.resize(inFile.tellg());
      inFile.seekg(0, std::ios::beg);
      inFile.read(reinterpret_cast<char *>(data.data()), data.size());
      filehash = XXH128(data.data(), data.size(), 0);
      inFile.close();

      // Check if the file has been modified since fetching the hash, to assure atomicity
      lastWriteTimeCheck = uint64_t(fs::last_write_time(filepath));
    } while (lastWriteTimeCheck != lastWriteTime);
  }

  collection.emplace(filepathStr, TrackedFile{.lwt = uint64_t(lastWriteTime), .derivedHash = *filehash});

  AssetInfo info;
  info.key = hash128ToUuid(*filehash);
  info.category = category;
  return info;
}

AssetInfo DataCache::generateSourceKey(boost::span<uint8_t> data, AssetCategory category) {
  auto hash = XXH128(data.data(), data.size(), 0);
  AssetInfo info;
  info.key = hash128ToUuid(hash);
  info.category = category;
  return info;
}

bool DataCache::hasAsset(const AssetInfo &info) { return io->hasAsset(info); }

void DataCache::store(const AssetInfo &info, ImmutableSharedBuffer data) { io->store(info, data); }

std::shared_ptr<AssetRequest> DataCache::fetch(AssetInfo key) {
  auto req = std::make_shared<AssetRequest>();
  req->key = key;
  io->enqueueRequest(req);
  return req;
}
void DataCache::fetchImmediate(AssetInfo key, shards::pmr::vector<uint8_t> &data) {
  io->fetchImmediate(key, data);
}

// TODO: Mayhaps replace this with a per-context cache, athough fully shared might be better
static std::shared_ptr<DataCache> instance;
std::shared_ptr<DataCache> getInstance() {
  return instance; }
void setInstance(std::shared_ptr<DataCache> cache) {
  instance = cache; }

} // namespace gfx::data
