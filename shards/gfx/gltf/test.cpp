// #include <shards/core/serialization/generic.hpp>
// #include <shards/core/serialization/linalg.hpp>
#include <gfx/texture_serde.hpp>

int something() {
  gfx::TextureFormat format{};
  auto bytes = shards::toByteArray(format);
  auto format2 = shards::fromByteArray<gfx::TextureFormat>(bytes);
  (void)format2;
  return 0;
}