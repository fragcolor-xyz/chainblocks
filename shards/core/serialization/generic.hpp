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
concept SerializerStream = requires(T &stream__, uint8_t *buf, size_t size) {
  { stream__(buf, size) };
  { T::Mode } -> std::convertible_to<IOMode>;
};

// Generic template, specialize this for custom types
template <typename V> struct Serialize {
  template <SerializerStream S> void operator()(S &stream, V &v) { static_assert(false, "Type is not serializable"); }
};

// Helper serialization functions
template <SerializerStream S, typename V> void serde(S &stream, V &v) { Serialize<V>{}(stream, v); }
template <SerializerStream S, typename V> void serdeConst(S &stream, const V &v) { serde(stream, const_cast<V &>(v)); }
template <typename As, SerializerStream S, typename V> void serdeAs(S &stream, V &v) {
  if constexpr (S::Mode == IOMode::Read) {
    As a{};
    serde(stream, a);
    v = V(a);
  } else {
    As a = As(v);
    serde(stream, a);
  }
}

template <typename V>
concept TriviallySerializable = std::is_integral_v<V> || std::is_floating_point_v<V> || std::is_same_v<V, bool>;
template <TriviallySerializable V> struct Serialize<V> {
  template <SerializerStream S> void operator()(S &stream, V &v) { stream((uint8_t *)&v, sizeof(V)); }
};

template <> struct Serialize<std::monostate> {
  template <SerializerStream S> void operator()(S &stream, std::monostate &v) {}
};

template <typename V> struct Serialize<boost::span<V>> {
  template <SerializerStream S> void operator()(S &stream, boost::span<V> &v) {
    for (size_t i = 0; i < v.size(); i++) {
      Serialize<V>{}(stream, v[i]);
    }
  }
};

template <typename V>
concept Container = requires(V &v) {
  typename V::value_type;
  { v.size() } -> std::convertible_to<size_t>;
  { v.resize(0) } -> std::same_as<void>;
  { boost::span(v.data(), v.size()) };
};

template <typename SizeType, SerializerStream T, Container V> void serdeWithSize(T &stream, V &v) {
  if constexpr (T::Mode == IOMode::Read) {
    SizeType size{};
    serde(stream, size);
    v.resize(size);
    boost::span<typename V::value_type> span(v.data(), v.size());
    serde(stream, span);
  } else {
    if (v.size() > std::numeric_limits<SizeType>::max()) {
      throw std::runtime_error("Vector too large to serialize");
    }
    SizeType size = v.size();
    serde(stream, size);
    boost::span<typename V::value_type> span(v.data(), v.size());
    serde(stream, span);
  }
}

template <Container C> struct Serialize<C> {
  template <SerializerStream S> void operator()(S &stream, C &v) { serdeWithSize<uint32_t>(stream, v); }
};

template <typename... TArgs> struct Serialize<std::variant<TArgs...>> {
  template <SerializerStream S> void operator()(S &stream, std::variant<TArgs...> &v) {
    uint32_t index = v.index();
    serde(stream, index);
    if constexpr (S::Mode == IOMode::Read) {
      // Emplace by index
      size_t counter = 0;
      ((
          [&]<typename T>(T *) {
            if (counter == index) {
              T value;
              Serialize<T>{}(stream, value);
              v = std::move(value);
            }
            counter++;
          }(static_cast<TArgs *>(nullptr)),
          ...));
    } else {
      std::visit([&](auto &arg) { Serialize<std::decay_t<decltype(arg)>>{}(stream, arg); }, v);
    }
  }
};

template <typename V, typename Allocator = std::allocator<uint8_t>>
inline std::vector<uint8_t, Allocator> toByteArray(V &v, Allocator alloc = Allocator()) {
  BufferWriter<Allocator> writer(alloc);
  ::shards::template serde<>((BufferWriter<Allocator> &)writer, v);
  return writer.takeBuffer();
}

template <typename V> inline V fromByteArray(boost::span<uint8_t> bytes) {
  V v{};
  BytesReader reader(bytes);
  ::shards::serde(reader, (V &)v);
  return v;
}

} // namespace shards

#endif /* EB7C6582_4FDE_4A76_A10F_E7AAF9D2B894 */
