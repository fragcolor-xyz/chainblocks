#ifndef E52D1428_FF30_41BD_BF58_74530373D525
#define E52D1428_FF30_41BD_BF58_74530373D525

#include <string_view>
#include <boost/uuid/uuid.hpp>
#include <optional>

namespace gfx::data {

enum class AssetCategory : uint8_t {
  // Source types
  Drawable,
  Mesh,
  Image,
  // Derived types
  // TextureLod,
  // MeshLod,
  // Pipeline,
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

struct AssetKey;
struct AssetInfo {
  AssetCategory category;
  AssetCategoryFlags categoryFlags;
  AssetFlags flags : 16 = AssetFlags::None;
  boost::uuids::uuid key;
  // In case this asset is derived from a source asset, this will be the source asset
  std::optional<boost::uuids::uuid> rootAsset;

  AssetInfo() = default;
  AssetInfo(const AssetKey &key);
};

struct AssetKey {
  AssetCategory category;
  AssetCategoryFlags categoryFlags;
  boost::uuids::uuid key;

  AssetKey(AssetCategory category, boost::uuids::uuid key, AssetCategoryFlags categoryFlags = AssetCategoryFlags::None)
      : category(category), categoryFlags(categoryFlags), key(key) {}
  AssetKey(const AssetInfo &info) : category(info.category), categoryFlags(info.categoryFlags), key(info.key) {}

  std::strong_ordering operator<=>(const AssetKey &other) const = default;

  AssetKey metaKey() const { return AssetKey(category, key, categoryFlags | AssetCategoryFlags::MetaData); }
};

inline AssetInfo::AssetInfo(const AssetKey &key) : category(key.category), categoryFlags(key.categoryFlags), key(key.key) {}
} // namespace gfx::data


#endif /* E52D1428_FF30_41BD_BF58_74530373D525 */
