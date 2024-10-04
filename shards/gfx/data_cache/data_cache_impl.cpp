#include "data_cache_impl.hpp"
#include "data_cache.hpp"
#include "types.hpp"
#include "fmt.hpp"
#include "../filesystem.hpp"
#include "log.hpp"
#include <shards/core/serialization/generic.hpp>
#include <shards/core/taskflow.hpp>
#include <shards/core/utils.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <chrono>
#include <fstream>
#include <queue>

namespace gfx::data {
using Path = fs::path;

static inline shards::logging::Logger logger = gfx::data::getLogger();

struct SubTypeIO {
  Path path;
  std::string filenameSuffix;

  Path getPath(const boost::uuids::uuid &key) const { return path / (boost::uuids::to_string(key) + filenameSuffix); }
};

struct ShardsDataCacheIO : public std::enable_shared_from_this<ShardsDataCacheIO>, public data::IDataCacheIO {
  Path rootPath;
  std::array<SubTypeIO, magic_enum::enum_count<gfx::data::AssetCategory>() * 2> subTypeIO;

  std::mutex workerMutex;
  std::condition_variable workerCv;
  std::atomic_bool runWorker = true;
  std::queue<std::shared_ptr<data::AssetLoadRequest>> loadRequestQueue;
  std::queue<std::shared_ptr<data::AssetStoreRequest>> storeRequestQueue;
  std::unordered_map<AssetKey, std::shared_ptr<data::AssetStoreRequest>> storeRequestMap;
  std::thread workerThread;

  SubTypeIO &getSubTypeIO(gfx::data::AssetCategory category, gfx::data::AssetCategoryFlags flags) {
    bool isMetaData = gfx::data::assetCategoryFlagsContains(flags, gfx::data::AssetCategoryFlags::MetaData);
    return subTypeIO[2 * size_t(category) + (isMetaData ? 1 : 0)];
  }

  template <typename TAlloc> void retrieveInternal(const data::AssetInfo &key, std::vector<uint8_t, TAlloc> &outData) {
    auto &inner = getSubTypeIO(key.category, key.categoryFlags);
    auto path = inner.getPath(key.key);

    std::ifstream inFile(path.string(), std::ios::in | std::ios::binary);
    inFile.seekg(0, std::ios::end);
    outData.resize(inFile.tellg());
    inFile.seekg(0, std::ios::beg);
    inFile.read(reinterpret_cast<char *>(outData.data()), outData.size());
    inFile.close();
  }

  // Do decoding/deserializing/etc.
  static void loadAssetWork(std::shared_ptr<data::AssetLoadRequest> request, std::vector<uint8_t> data) {
    try {
      finializeAssetLoadRequest(*request.get(), boost::span(data));
      request->state = (uint8_t)AssetLoadRequestState::Loaded;
    } catch (std::exception &e) {
      SPDLOG_LOGGER_ERROR(logger, "Failed to decode asset: {}, {}", request->key, e.what());
      request->state = (uint8_t)AssetLoadRequestState::Failure;
      return;
    }
  }

  static void loadAssetFromStoreRequest(std::shared_ptr<data::AssetLoadRequest> request,
                                        std::shared_ptr<data::AssetStoreRequest> storeRequest) {
    try {
      processAssetLoadFromStoreRequest(*request.get(), *storeRequest.get());
      request->state = (uint8_t)AssetLoadRequestState::Loaded;
    } catch (std::exception &e) {
      SPDLOG_LOGGER_ERROR(logger, "Failed to load asset from store request: {}, {}", request->key, e.what());
      request->state = (uint8_t)AssetLoadRequestState::Failure;
    }
  }

  static void storeAssetWork(std::shared_ptr<ShardsDataCacheIO> self, std::shared_ptr<data::AssetStoreRequest> request) {
    auto &inner = self->getSubTypeIO(request->key.category, request->key.categoryFlags);
    auto path = inner.getPath(request->key.key);

    try {
      shards::pmr::vector<uint8_t> data;
      data.reserve(1024 * 1024 * 16);
      processAssetStoreRequest(*request.get(), data);

      std::ofstream outFile(path.string(), std::ios::out | std::ios::binary);
      outFile.write(reinterpret_cast<const char *>(data.data()), data.size());
      outFile.close();
      request->state = (uint8_t)AssetLoadRequestState::Loaded;
    } catch (std::exception &e) {
      SPDLOG_LOGGER_ERROR(logger, "Failed to store asset: {}, {}", request->key, e.what());
      request->state = (uint8_t)AssetLoadRequestState::Failure;
    }

    std::unique_lock lock(self->workerMutex);
    self->storeRequestMap.erase(request->key);
  }

  // The IO loop, dispatches decoding task after
  void worker() {
    shards::pushThreadName("DataCacheIO");
    SPDLOG_LOGGER_INFO(logger, "DataCacheIO worker started");

    auto &taskflow = shards::TaskFlowInstance::instance();
    while (runWorker) {
      using namespace std::chrono_literals;

      std::unique_lock lock(workerMutex);
      workerCv.wait_for(lock, 500ms);

      while (runWorker && !loadRequestQueue.empty()) {
        auto request = loadRequestQueue.front();
        loadRequestQueue.pop();

        SPDLOG_LOGGER_TRACE(logger, "Processing load request: {}", request->key);

        auto awaitingStore = storeRequestMap.find(request->key);
        if (awaitingStore != storeRequestMap.end()) {
          // Early out, data is still being stored reuse in memory data
          taskflow.async(loadAssetFromStoreRequest, request, awaitingStore->second);
          continue;
        }

        lock.unlock(); // UNLOCK FOR IO

        std::vector<uint8_t> data;
        try {
          retrieveInternal(request->key, data);
        } catch (std::exception &e) {
          SPDLOG_LOGGER_ERROR(logger, "Failed to retrieve asset: {}, {}", request->key, e.what());
          request->state = (uint8_t)AssetLoadRequestState::Failure;
          continue;
        }

        taskflow.async(loadAssetWork, request, std::move(data));

        lock.lock(); // RELOCK AFTER IO AND DISPATCH CONTINUATION
      }

      while (runWorker && !storeRequestQueue.empty()) {
        auto request = storeRequestQueue.front();
        storeRequestQueue.pop();

        SPDLOG_LOGGER_TRACE(logger, "Processing store request: {}", request->key);

        auto &subIo = getSubTypeIO(request->key.category, request->key.categoryFlags);
        auto path = subIo.getPath(request->key.key);
        taskflow.async(storeAssetWork, shared_from_this(), request);
      }
    }

    SPDLOG_LOGGER_INFO(logger, "DataCacheIO worker exiting");
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

  void enqueueLoadRequest(std::shared_ptr<data::AssetLoadRequest> request) {
    std::unique_lock lock(workerMutex);
    loadRequestQueue.push(request);
    lock.unlock();
    workerCv.notify_one();
  }

  virtual void loadImmediate(data::AssetInfo key, shards::pmr::vector<uint8_t> &data) { retrieveInternal(key, data); }

  void enqueueStoreRequest(std::shared_ptr<data::AssetStoreRequest> request) {
    std::unique_lock lock(workerMutex);
    storeRequestQueue.push(request);
    storeRequestMap.emplace(request->key, request);
    lock.unlock();
    workerCv.notify_one();
  }

  // void enqueueStoreRequest(const data::AssetInfo &key, gfx::ImmutableSharedBuffer data) {
  //   auto &inner = getSubTypeIO(key.category, key.categoryFlags);
  //   auto path = inner.getPath(key.key);

  //   try {
  //     if (fs::file_size(path) == data.getLength()) {
  //       return;
  //     }
  //   } catch (fs::filesystem_error &e) {
  //     // ignore
  //   }

  //   std::ofstream outFile(path.string(), std::ios::out | std::ios::binary);
  //   outFile.write(reinterpret_cast<const char *>(data.getData()), data.getLength());
  //   outFile.close();
  // }

  bool hasAsset(const data::AssetInfo &key) {
    auto &inner = getSubTypeIO(key.category, key.categoryFlags);
    auto path = inner.getPath(key.key);
    bool fileExists = fs::exists(path);
    if (fileExists) {
      return true;
    }

    auto lock = std::scoped_lock(workerMutex);
    auto storeIt = storeRequestMap.find(key);
    if (storeIt != storeRequestMap.end()) {
      return true;
    }
    return false;
  }
};

std::shared_ptr<data::IDataCacheIO> createShardsDataCacheIO(std::string_view rootPath) {
  return std::make_shared<ShardsDataCacheIO>(rootPath);
}
std::shared_ptr<data::IDataCacheIO> getDefaultDataCacheIO() {
  static auto instance = createShardsDataCacheIO(".shards");
  return instance;
}

} // namespace gfx::data