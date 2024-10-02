#ifndef E99703D5_783E_44A1_BF22_B198F8E23C9D
#define E99703D5_783E_44A1_BF22_B198F8E23C9D

#include "types.hpp"
#include <spdlog/fmt/fmt.h>

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

#endif /* E99703D5_783E_44A1_BF22_B198F8E23C9D */
