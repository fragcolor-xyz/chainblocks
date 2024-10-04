#ifndef E52D1428_FF30_41BD_BF58_74530373D525
#define E52D1428_FF30_41BD_BF58_74530373D525

#include <string_view>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/nil_generator.hpp>
#include <shards/core/serialization/generic.hpp>
#include <optional>
#include <magic_enum.hpp>

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
  AssetCategoryFlags categoryFlags{};
  AssetFlags flags : 16 = AssetFlags::None;
  boost::uuids::uuid key{};
  // In case this asset is derived from a source asset, this will be the source asset
  std::optional<boost::uuids::uuid> rootAsset;

  AssetInfo() = default;
  AssetInfo(const AssetKey &key);
};

struct AssetKey {
  AssetCategory category;
  AssetCategoryFlags categoryFlags;
  boost::uuids::uuid key;

  AssetKey() : key(boost::uuids::nil_uuid()) {}
  AssetKey(AssetCategory category, boost::uuids::uuid key, AssetCategoryFlags categoryFlags = AssetCategoryFlags::None)
      : category(category), categoryFlags(categoryFlags), key(key) {}
  AssetKey(const AssetInfo &info) : category(info.category), categoryFlags(info.categoryFlags), key(info.key) {}

  std::strong_ordering operator<=>(const AssetKey &other) const = default;

  AssetKey metaKey() const { return AssetKey(category, key, categoryFlags | AssetCategoryFlags::MetaData); }

  operator bool() const { return key.is_nil(); }
};

inline AssetInfo::AssetInfo(const AssetKey &key) : category(key.category), categoryFlags(key.categoryFlags), key(key.key) {}
} // namespace gfx::data

namespace std {
template <> struct hash<gfx::data::AssetKey> {
  size_t operator()(const gfx::data::AssetKey &key) const {
    size_t base = boost::uuids::hash_value(key.key);
    base = base * 3 + std::hash<gfx::data::AssetCategoryFlags>()(key.categoryFlags);
    base = base * 3 + std::hash<gfx::data::AssetCategory>()(key.category);
    return base;
  }
};
} // namespace std

namespace shards {
template <> struct Serialize<boost::uuids::uuid> {
  template <SerializerStream S> void operator()(S &stream, boost::uuids::uuid &uuid) {
    stream((uint8_t *)uuid.data, sizeof(uuid.data));
  }
};

template <> struct Serialize<gfx::data::AssetKey> {
  template <SerializerStream S> void operator()(S &stream, gfx::data::AssetKey &key) {
    serde(stream, key.key);
    serdeAs<uint8_t>(stream, key.categoryFlags);
    serdeAs<uint8_t>(stream, key.category);
  }
};
} // namespace shards

namespace magic_enum::customize {
template <> struct enum_range<gfx::data::AssetFlags> {
  static constexpr bool is_flags = true;
};
template <> struct enum_range<gfx::data::AssetCategoryFlags> {
  static constexpr bool is_flags = true;
};
} // namespace magic_enum::customize


#endif /* E52D1428_FF30_41BD_BF58_74530373D525 */
