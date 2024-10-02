#ifndef D6B9DBDC_4CD6_4B09_8860_BC3E73379844
#define D6B9DBDC_4CD6_4B09_8860_BC3E73379844

#include "types.hpp"
#include "../isb.hpp"
#include <string_view>
#include <shards/core/pmr/vector.hpp>
#include <boost/core/span.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <optional>
#include <memory>
#include <spdlog/fmt/fmt.h>

namespace gfx::data {
enum class AssetRequestState : uint32_t {
  Pending = 0,
  Loaded = 1,
  Failure = 2,
};

struct AssetRequest {
  AssetInfo key;
  std::atomic_uint32_t state;
  shards::pmr::vector<uint8_t> data;
};

struct IDataCacheIO {
  virtual ~IDataCacheIO() = default;
  // Enqueue an asset fetch request to be processed in the background
  virtual void enqueueRequest(std::shared_ptr<AssetRequest> request) = 0;
  // Fetch an asset immediately, this should only be used for small metadata
  virtual void fetchImmediate(AssetInfo key, shards::pmr::vector<uint8_t> &data) = 0;
  virtual void store(const AssetInfo &key, ImmutableSharedBuffer data) = 0;
  virtual bool hasAsset(const AssetInfo &key) = 0;
};

struct TrackedFileCache;
struct DataCache {
private:
  std::shared_ptr<IDataCacheIO> io;
  std::shared_ptr<TrackedFileCache> trackedFiles;

public:
  DataCache(std::shared_ptr<IDataCacheIO> io);

  // Generate a new key for a source asset given a file location on disk
  // Key is based on the contents of the file
  AssetInfo generateSourceKey(std::string_view path, AssetCategory category);

  // Generate a new key for a binary blob, key is based on the contents of the blob
  AssetInfo generateSourceKey(boost::span<uint8_t> data, AssetCategory category);

  bool hasAsset(const AssetInfo &info);
  void store(const AssetInfo &info, ImmutableSharedBuffer data);

  // Asynchronously fetch an asset, returns a request object that can be used to check the status
  std::shared_ptr<AssetRequest> fetch(AssetInfo key);

  // Fetch an asset immediately, this should only be used for small metadata
  void fetchImmediate(AssetInfo key, shards::pmr::vector<uint8_t> &data);
};

std::shared_ptr<DataCache> getInstance();
void setInstance(std::shared_ptr<DataCache> cache);

// Reference to an asset in the cache
struct AssetCacheRef {
  boost::uuids::uuid key;
  AssetCategory type;
};

} // namespace gfx::data

#endif /* D6B9DBDC_4CD6_4B09_8860_BC3E73379844 */
