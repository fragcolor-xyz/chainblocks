#ifndef EB7C6582_4FDE_4A76_A10F_E7AAF9D2B894
#define EB7C6582_4FDE_4A76_A10F_E7AAF9D2B894

#include <linalg.h>
#include <stdexcept>
#include <cassert>
#include <string_view>
#include <concepts>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <tracy/Wrapper.hpp>
#include "reader_writer.hpp"

// Enable for more detailed breakdown of what types are being (de)serialized in tracy
#define SH_SERIALIZATION_TRACY_DETAILED 0

namespace shards {
template <typename T>
concept SerializerStream = requires(T &stream__, uint8_t *buf, size_t size) {
  {stream__(buf, size)};
  { T::Mode } -> std::convertible_to<IOMode>;
};

template <typename T> struct ErrorHelper { static_assert(std::is_same<T, void>::value, "Type is not serializable"); };

// Generic template, specialize this for custom types
template <typename V> struct Serialize {
  template <SerializerStream S> void operator()(S &stream, V &v) { ErrorHelper<V> e{}; }
};

// Helper serialization functions
template <SerializerStream S, typename V> void serde(S &stream, V &v) { Serialize<V>{}(stream, v); }
template <SerializerStream S, typename V> void serdeConst(S &stream, const V &v) {
  static_assert(S::Mode == IOMode::Write, "Cannot deserialize const value");
  serde(stream, const_cast<V &>(v));
}
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
  {boost::span(v.data(), v.size())};
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

template <typename K, typename V> struct Serialize<std::pair<K, V>> {
  template <SerializerStream S> void operator()(S &stream, std::pair<K, V> &v) {
    serde(stream, v.first);
    serde(stream, v.second);
  }
};

template <typename K, typename V> struct Serialize<std::pair<const K, V>> {
  template <SerializerStream S> void operator()(S &stream, std::pair<const K, V> &v) {
    static_assert(S::Mode == IOMode::Write, "Cannot deserialize const key-value pair");
    serde(stream, const_cast<K &>(v.first));
    serde(stream, v.second);
  }
};

// The default size type to use for collections/variants
using SerializeSizeType = uint32_t;
using SerializeVariantType = uint16_t;

template <Container C> struct Serialize<C> {
  template <SerializerStream S> void operator()(S &stream, C &v) { serdeWithSize<SerializeSizeType>(stream, v); }
};

template <typename V>
concept KeyValueContainer = requires(V &v, typename V::value_type pair) {
  typename V::key_type;
  typename V::mapped_type;
  typename V::value_type;
  { v.size() } -> std::convertible_to<size_t>;
  {v.emplace(pair)};
  {v.insert(pair)};
};

template <typename SizeType, SerializerStream T, KeyValueContainer V> void serdeWithSize(T &stream, V &v) {
  if constexpr (T::Mode == IOMode::Read) {
    SizeType size{};
    serde(stream, size);
    for (size_t i = 0; i < size; i++) {
      std::pair<typename V::key_type, typename V::mapped_type> pair;
      serde(stream, pair);
      v.insert(pair);
    }
  } else {
    if (v.size() > std::numeric_limits<SizeType>::max()) {
      throw std::runtime_error("Table too large to serialize");
    }
    SizeType size = v.size();
    serde(stream, size);
    for (auto &pair : v) {
      serde(stream, pair);
    }
  }
}

template <KeyValueContainer C> struct Serialize<C> {
  template <SerializerStream S> void operator()(S &stream, C &v) { serdeWithSize<SerializeSizeType>(stream, v); }
};

template <typename... TArgs> struct Serialize<std::variant<TArgs...>> {
  template <SerializerStream S> void operator()(S &stream, std::variant<TArgs...> &v) {
    SerializeVariantType index = v.index();
    serde(stream, index);
    if constexpr (S::Mode == IOMode::Read) {
      // Emplace by index
      size_t counter = 0;
      ((
          [&]<typename T>(T *) {
            if (counter == index) {
              T value;
              ::shards::serde((S &)stream, (decltype(value) &)value);
              v = std::move(value);
            }
            counter++;
          }(static_cast<TArgs *>(nullptr)),
          ...));
    } else {
      std::visit(
          [&](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            ::shards::serde((S &)stream, (T &)arg);
          },
          v);
    }
  }
};

template <typename V, typename Allocator = std::allocator<uint8_t>>
inline std::vector<uint8_t, Allocator> toByteArray(const V &v, Allocator alloc = Allocator()) {
  BufferWriterA<Allocator> writer(alloc);
  ::shards::template serdeConst<>((BufferWriterA<Allocator> &)writer, v);
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
