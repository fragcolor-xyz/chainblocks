#ifndef BD053FCC_AC62_460C_9551_2B54DF500814
#define BD053FCC_AC62_460C_9551_2B54DF500814

#include <shards/core/serialization/generic.hpp>
#include <shards/fast_string/fast_string.hpp>
#include "../isb.hpp"

namespace shards {

template <> struct Serialize<fast_string::FastString> {
  template <SerializerStream S> inline void operator()(S &stream, fast_string::FastString &str) {
    serdeAs<std::string>(stream, str);
  }
};

template <> struct Serialize<gfx::ImmutableSharedBuffer> {
  template <SerializerStream S> inline void operator()(S &stream, gfx::ImmutableSharedBuffer &buffer) {
    if constexpr (S::Mode == IOMode::Read) {
      std::vector<uint8_t> data;
      serde(stream, data);
      buffer = gfx::ImmutableSharedBuffer(std::move(data));
    } else {
      serdeConst<S, SerializeSizeType>(stream, buffer.size());
      serdeConst(stream, buffer.asSpan());
    }
  }
};
} // namespace shards

#endif /* BD053FCC_AC62_460C_9551_2B54DF500814 */
