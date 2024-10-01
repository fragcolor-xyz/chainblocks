#ifndef D6B9DBDC_4CD6_4B09_8860_BC3E73379844
#define D6B9DBDC_4CD6_4B09_8860_BC3E73379844

#include "../hasherxxh3.hpp"
#include <string_view>
#include <shards/core/pmr/vector.hpp>
#include <boost/core/span.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <optional>
#include <memory>
#include <spdlog/fmt/fmt.h>

namespace gfx::data {

enum class AssetCategory : uint8_t {
  // Source types
  GLTF,
  Image,
  // Derived types
  TextureLod,
  MeshLod,
  Pipeline,
};

enum class AssetCategoryFlags : uint8_t {
  None = 0x0,
  MetaData = 0x1,
};

inline AssetCategoryFlags operator|(AssetCategoryFlags a, AssetCategoryFlags b) {
  return static_cast<AssetCategoryFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
inline bool assetCategoryFlagsContains(AssetCategoryFlags a, AssetCategoryFlags b) {
  return (static_cast<uint8_t>(a) & static_cast<uint8_t>(b)) == static_cast<uint8_t>(b);
}

enum class AssetFlags : uint16_t {
  None = 0x0,
  AllowGC = 0x1,
};
inline AssetFlags operator|(AssetFlags a, AssetFlags b) {
  return static_cast<AssetFlags>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}
inline AssetFlags &operator|=(AssetFlags &a, AssetFlags b) { return a = a | b; }
inline bool assetFlagsContains(AssetFlags flags, AssetFlags b) {
  return static_cast<bool>(static_cast<uint16_t>(flags) & static_cast<uint16_t>(b));
}

struct AssetInfo {
  AssetCategory category;
  AssetCategoryFlags categoryFlags;
  AssetFlags flags : 16 = AssetFlags::None;
  boost::uuids::uuid key;
  // In case this asset is derived from a source asset, this will be the source asset
  std::optional<boost::uuids::uuid> rootAsset;
};

struct AssetKey {
  AssetCategory category;
  AssetCategoryFlags categoryFlags;
  boost::uuids::uuid key;

  AssetKey(AssetCategory category, boost::uuids::uuid key, AssetCategoryFlags categoryFlags = AssetCategoryFlags::None)
      : category(category), categoryFlags(categoryFlags), key(key) {}
  AssetKey(const AssetInfo &info) : category(info.category), categoryFlags(info.categoryFlags), key(info.key) {}

  std::strong_ordering operator<=>(const AssetKey &other) const = default;
};
} // namespace gfx::data

template <> struct fmt::formatter<gfx::data::AssetInfo> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  template <typename FormatContext> auto format(const gfx::data::AssetInfo &info, FormatContext &ctx) {
    return format_to(ctx.out(), "AssetInfo(cat: {}/{}, key: {}, flags: {})", info.category, info.key, info.flags);
  }
};

template <> struct fmt::formatter<gfx::data::AssetKey> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  template <typename FormatContext> auto format(const gfx::data::AssetKey &key, FormatContext &ctx) {
    if (assetCategoryFlagsContains(key.categoryFlags, gfx::data::AssetCategoryFlags::MetaData))
      return format_to(ctx.out(), "AssetKey(cat: {}, key: {}, metadata)", key.category, key.key);
    else
      return format_to(ctx.out(), "AssetKey(cat: {}, key: {})", key.category, key.key);
  }
};

template <> struct std::hash<gfx::data::AssetKey> {
  std::size_t operator()(const gfx::data::AssetKey &key) const {
    size_t hash = std::hash<gfx::data::AssetCategory>()(key.category);
    hash = hash * 31 + std::hash<gfx::data::AssetCategoryFlags>()(key.categoryFlags);
    hash = hash * 31 + boost::uuids::hash_value(key.key);
    return hash;
  }
};

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
  virtual void store(const AssetInfo &key, boost::span<uint8_t> data) = 0;
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
  void store(const AssetInfo &info, boost::span<uint8_t> data);

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
