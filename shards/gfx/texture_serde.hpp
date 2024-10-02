#ifndef E332B211_27DC_4F7F_BFCD_CF63EE8257FB
#define E332B211_27DC_4F7F_BFCD_CF63EE8257FB

#include "texture.hpp"
#include <shards/core/serialization/generic.hpp>
#include <shards/core/serialization/linalg.hpp>

namespace shards {
template <SerializerStream T> void shards::serde(T &stream, ::gfx::TextureFormat &v) {
  serde(stream, v.resolution);
  serdeAs<uint8_t>(stream, v.dimension);
  serdeAs<uint8_t>(stream, v.flags);
  serdeAs<uint32_t>(stream, v.pixelFormat);
  serdeAs<uint8_t>(stream, v.mipLevels);
}
} // namespace shards

#endif /* E332B211_27DC_4F7F_BFCD_CF63EE8257FB */
