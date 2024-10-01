#ifndef A99B1602_B944_46C1_976F_F4AB9EFBA982
#define A99B1602_B944_46C1_976F_F4AB9EFBA982

#include "generic.hpp"
#include <linalg.h>

namespace shards {
template <SerializerStream T, typename V, int M> void serde(T &stream, linalg::vec<V, M> &v) {
  for (int i = 0; i < M; i++)
    serde(stream, v[i]);
}
template <SerializerStream T, typename V, int N, int M> void serde(T &stream, linalg::mat<V, M, N> &v) {
  for (int m = 0; m < N; m++) {
    serde(stream, v[m]);
  }
}
} // namespace shards

#endif /* A99B1602_B944_46C1_976F_F4AB9EFBA982 */
