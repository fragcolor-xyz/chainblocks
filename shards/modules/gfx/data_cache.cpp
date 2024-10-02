#include "data_cache.hpp"
#include <gfx/data_cache/fmt.hpp>
#include <shards/core/serialization.hpp>
#include <boost/filesystem.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fstream>
#include <queue>

namespace fs {
using namespace boost::filesystem;
}

namespace gfx {
using Path = fs::path;

struct SubTypeIO {
  Path path;
  std::string filenameSuffix;

  Path getPath(const boost::uuids::uuid &key) const { return path / (boost::uuids::to_string(key) + filenameSuffix); }
};

struct ShardsDataCacheIO : public data::IDataCacheIO {
  Path rootPath;
  std::array<SubTypeIO, magic_enum::enum_count<gfx::data::AssetCategory>() * 2> subTypeIO;

  std::mutex workerMutex;
  std::condition_variable workerCv;
  bool runWorker = false;
  std::queue<std::shared_ptr<data::AssetRequest>> requestQueue;
  std::thread workerThread;

  SubTypeIO &getSubTypeIO(gfx::data::AssetCategory category, gfx::data::AssetCategoryFlags flags) {
    bool isMetaData = gfx::data::assetCategoryFlagsContains(flags, gfx::data::AssetCategoryFlags::MetaData);
    return subTypeIO[2 * size_t(category) + (isMetaData ? 1 : 0)];
  }

  void retrieveInternal(const data::AssetInfo &key, shards::pmr::vector<uint8_t> &outData) {
    auto &inner = getSubTypeIO(key.category, key.categoryFlags);
    auto path = inner.getPath(key.key);

    std::ifstream inFile(path.string(), std::ios::in | std::ios::binary);
    inFile.seekg(0, std::ios::end);
    outData.resize(inFile.tellg());
    inFile.seekg(0, std::ios::beg);
    inFile.read(reinterpret_cast<char *>(outData.data()), outData.size());
    inFile.close();
  }

  void worker() {
    while (true) {
      std::unique_lock lock(workerMutex);
      workerCv.wait(lock, [&]() { return !runWorker; });

      if (!runWorker)
        break;

      if (requestQueue.empty())
        continue;

      auto request = requestQueue.front();
      requestQueue.pop();
      lock.unlock();

      try {
        retrieveInternal(request->key, request->data);
        request->state = 1;
      } catch (std::exception &e) {
        SPDLOG_ERROR("Failed to retrieve asset: {}, {}", request->key, e.what());
        request->state = 2;
      }
    }
  }

  ShardsDataCacheIO(std::string_view rootPath) { init(rootPath); }
  ~ShardsDataCacheIO() {
    runWorker = false;
    workerCv.notify_one();
    workerThread.join();
  }

  void init(std::string_view rootPath_) {
    rootPath = Path(rootPath_.begin(), rootPath_.end());
    fs::create_directories(rootPath);

    magic_enum::enum_for_each<gfx::data::AssetCategory>([&](gfx::data::AssetCategory assetType) {
      auto &inner = subTypeIO[2 * size_t(assetType)];
      inner.path = rootPath / std::string(magic_enum::enum_name(assetType));
      auto &innerMeta = subTypeIO[2 * size_t(assetType) + 1];
      innerMeta.path = inner.path;
      innerMeta.filenameSuffix = ".meta";
      fs::create_directories(inner.path);
    });

    workerThread = std::thread([&]() { worker(); });
  }

  void enqueueRequest(std::shared_ptr<data::AssetRequest> request) {
    std::unique_lock lock(workerMutex);
    requestQueue.push(request);
    workerCv.notify_one();
  }

  virtual void fetchImmediate(data::AssetInfo key, shards::pmr::vector<uint8_t> &data) { retrieveInternal(key, data); }

  void store(const data::AssetInfo &key, gfx::ImmutableSharedBuffer data) {
    auto &inner = getSubTypeIO(key.category, key.categoryFlags);
    auto path = inner.getPath(key.key);

    try {
      if (fs::file_size(path) == data.getLength()) {
        return;
      }
    } catch (fs::filesystem_error &e) {
      // ignore
    }

    std::ofstream outFile(path.string(), std::ios::out | std::ios::binary);
    outFile.write(reinterpret_cast<const char *>(data.getData()), data.getLength());
    outFile.close();
  }

  bool hasAsset(const data::AssetInfo &key) {
    auto &inner = getSubTypeIO(key.category, key.categoryFlags);
    auto path = inner.getPath(key.key);
    return fs::exists(path);
  }
};

std::shared_ptr<data::IDataCacheIO> createShardsDataCacheIO(std::string_view rootPath) {
  return std::make_shared<ShardsDataCacheIO>(rootPath);
}
std::shared_ptr<data::IDataCacheIO> getDefaultDataCacheIO() {
  static auto instance = createShardsDataCacheIO(".shards");
  return instance;
}

} // namespace gfx