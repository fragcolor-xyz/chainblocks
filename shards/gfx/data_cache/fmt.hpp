#ifndef E99703D5_783E_44A1_BF22_B198F8E23C9D
#define E99703D5_783E_44A1_BF22_B198F8E23C9D

#include "types.hpp"
#include <magic_enum.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <boost/uuid/uuid_io.hpp>

template <> struct fmt::formatter<gfx::data::AssetInfo> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  template <typename FormatContext> auto format(const gfx::data::AssetInfo &info, FormatContext &ctx) {
    auto it = ctx.out();
    it = format_to(it, "AssetInfo({}", magic_enum::enum_name(info.category));
    if (info.categoryFlags != gfx::data::AssetCategoryFlags::None) {
      it = format_to(it, "/{}", magic_enum::enum_name(info.categoryFlags));
    }
    it = format_to(it, ", key: {}", info.key);
    if (info.flags != gfx::data::AssetFlags::None) {
      it = format_to(it, ", flags: {}", magic_enum::enum_name(info.flags));
    }
    it = format_to(it, ")");
    return it;
  }
};

template <> struct fmt::formatter<gfx::data::AssetKey> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  template <typename FormatContext> auto format(const gfx::data::AssetKey &key, FormatContext &ctx) {
    if (assetCategoryFlagsContains(key.categoryFlags, gfx::data::AssetCategoryFlags::MetaData))
      return format_to(ctx.out(), "AssetKey(cat: {}, key: {}, metadata)", magic_enum::enum_name(key.category), key.key);
    else
      return format_to(ctx.out(), "AssetKey(cat: {}, key: {})", magic_enum::enum_name(key.category), key.key);
  }
};

#endif /* E99703D5_783E_44A1_BF22_B198F8E23C9D */
