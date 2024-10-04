#ifndef A99B1602_B944_46C1_976F_F4AB9EFBA982
#define A99B1602_B944_46C1_976F_F4AB9EFBA982

#include "generic.hpp"
#include <linalg.h>

namespace shards {
template <typename V, int M> struct Serialize<linalg::vec<V, M>> {
  template <SerializerStream S> void operator()(S &stream, linalg::vec<V, M> &v) {
    for (int i = 0; i < M; i++)
      serde(stream, v[i]);
  }
};
template <typename V, int N, int M> struct Serialize<linalg::mat<V, M, N>> {
  template <SerializerStream S> void operator()(S &stream, linalg::mat<V, M, N> &v) {
    for (int m = 0; m < N; m++) {
      serde(stream, v[m]);
    }
  }
};
} // namespace shards

#endif /* A99B1602_B944_46C1_976F_F4AB9EFBA982 */
