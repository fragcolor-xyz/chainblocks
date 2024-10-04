#ifndef D6B9DBDC_4CD6_4B09_8860_BC3E73379844
#define D6B9DBDC_4CD6_4B09_8860_BC3E73379844

#include "types.hpp"
#include "../isb.hpp"
#include "../texture.hpp"
#include "../mesh.hpp"
#include "../data_format/texture.hpp"
#include <string_view>
#include <shards/core/pmr/vector.hpp>
#include <boost/core/span.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <optional>
#include <memory>
#include <spdlog/fmt/fmt.h>

namespace gfx::data {
enum class AssetLoadRequestState : uint32_t {
  Pending = 0,
  Loaded = 1,
  Failure = 2,
};

using LoadedAssetDataVariant = std::variant<std::monostate, std::vector<uint8_t>, SerializedTexture, gfx::TextureDescCPUCopy,
                                            gfx::MeshDescCPUCopy, gfx::DrawablePtr>;
struct LoadedAssetData : public LoadedAssetDataVariant {
  using variant::variant;

  LoadedAssetData() = default;
  LoadedAssetData(LoadedAssetDataVariant variant) : LoadedAssetDataVariant(variant) {}
  operator bool() const { return !std::holds_alternative<std::monostate>(*this); }

  template <typename T> static std::shared_ptr<LoadedAssetData> makePtr(T &&data) {
    return std::make_shared<LoadedAssetData>(std::forward<T>(data));
  }
};

using LoadedAssetDataPtr = std::shared_ptr<LoadedAssetData>;

struct AssetLoadRequest {
  AssetInfo key;
  std::atomic_uint32_t state = 0;
  LoadedAssetDataPtr data;

  bool isFinished() const { return state.load() != (uint32_t)AssetLoadRequestState::Pending; }
  bool isSuccess() const { return state.load() == (uint32_t)AssetLoadRequestState::Loaded; }
};

struct AssetStoreRequest {
  AssetInfo key;
  std::atomic_uint32_t state = 0;
  LoadedAssetDataPtr data;

  bool isFinished() const { return state.load() != (uint32_t)AssetLoadRequestState::Pending; }
  bool isSuccess() const { return state.load() == (uint32_t)AssetLoadRequestState::Loaded; }
};

// After binary data is loaded, constructs the typed LoadedAssetData for the asset being loaded
void finializeAssetLoadRequest(AssetLoadRequest &request, boost::span<uint8_t> sourceData);
void processAssetStoreRequest(AssetStoreRequest &request, shards::pmr::vector<uint8_t> &outData);
void processAssetLoadFromStoreRequest(AssetLoadRequest &request, const AssetStoreRequest &inRequest);

struct IDataCacheIO {
  virtual ~IDataCacheIO() = default;
  // Enqueue an asset load request to be processed in the background
  virtual void enqueueLoadRequest(std::shared_ptr<AssetLoadRequest> request) = 0;
  // Load an asset immediately, this should only be used for small metadata
  virtual void loadImmediate(AssetInfo key, shards::pmr::vector<uint8_t> &data) = 0;
  virtual void enqueueStoreRequest(std::shared_ptr<AssetStoreRequest> request) = 0;
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

  // Store an asset, returns a request object that can be used to check the status
  std::shared_ptr<AssetStoreRequest> store(const AssetInfo &info, const LoadedAssetDataPtr &data);

  // Asynchronously fetch an asset, returns a request object that can be used to check the status
  std::shared_ptr<AssetLoadRequest> load(AssetInfo key);

  // Fetch an asset immediately, this should only be used for small metadata
  void loadImmediate(AssetInfo key, shards::pmr::vector<uint8_t> &data);
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
