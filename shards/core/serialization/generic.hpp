#ifndef EB7C6582_4FDE_4A76_A10F_E7AAF9D2B894
#define EB7C6582_4FDE_4A76_A10F_E7AAF9D2B894

#include <linalg.h>
#include <stdexcept>
#include <cassert>
#include <string_view>
#include <concepts>
#include <unordered_map>
#include <tracy/Wrapper.hpp>
#include "reader_writer.hpp"

// Enable for more detailed breakdown of what types are being (de)serialized in tracy
#define SH_SERIALIZATION_TRACY_DETAILED 0

namespace shards {
template <typename T>
concept SerializerStream = requires(T &stream, uint8_t *buf, size_t size) {
  { stream(buf, size) };
  { T::Mode } -> std::convertible_to<IOMode>;
};

struct Serializer {
  enum Mode { Mode_Read, Mode_Write };
  Mode mode = Mode_Read;

  Serializer(Mode mode) : mode(mode) {}
};

template <SerializerStream T, typename V>
std::enable_if_t<std::is_integral_v<V> || std::is_floating_point_v<V> || std::is_same_v<V, bool>, void> //
serde(T &stream, V &v) {
  stream((uint8_t *)&v, sizeof(V));
}
template <SerializerStream T> void serde(T &stream, std::monostate &v) {}
template <SerializerStream T, typename V> void serdeConst(T &stream, const V &v) { serde(stream, const_cast<V &>(v)); }
template <typename As, SerializerStream T, typename V> void serdeAs(T &stream, V &v) {
  if constexpr (T::Mode == IOMode::Read) {
    As a{};
    serde(stream, a);
    v = V(a);
  } else {
    As a = As(v);
    serde(stream, a);
  }
}

template <SerializerStream T, typename V> void serde(T &stream, boost::span<V> &v) {
  for (size_t i = 0; i < v.size(); i++) {
    serde(stream, v[i]);
  }
}

template <typename SizeType, SerializerStream T, typename V> void serdeWithSize(T &stream, std::vector<V> &v) {
  if constexpr (T::Mode == IOMode::Read) {
    SizeType size{};
    serde(stream, size);
    v.resize(size);
    serde(stream, boost::span<V>(v));
  } else {
    if (v.size() > std::numeric_limits<SizeType>::max()) {
      throw std::runtime_error("Vector too large to serialize");
    }
    SizeType size = v.size();
    serde(stream, size);
    serde(stream, boost::span<V>(v));
  }
}

} // namespace shards

#endif /* EB7C6582_4FDE_4A76_A10F_E7AAF9D2B894 */
