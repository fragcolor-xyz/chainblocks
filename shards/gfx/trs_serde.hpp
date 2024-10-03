#ifndef B4A8EC33_CCC3_49C5_BFA1_BBAD18D2BE24
#define B4A8EC33_CCC3_49C5_BFA1_BBAD18D2BE24

#include "trs.hpp"
#include <shards/core/serialization/linalg.hpp>

namespace shards {
template <> struct Serialize<gfx::TRS> {
  template<SerializerStream S>
  static void serialize(S& stream, gfx::TRS& trs) {
    stream(trs.translation);
    stream(trs.rotation);
    stream(trs.scale);
  }
};
  
} // namespace shards

#endif /* B4A8EC33_CCC3_49C5_BFA1_BBAD18D2BE24 */
