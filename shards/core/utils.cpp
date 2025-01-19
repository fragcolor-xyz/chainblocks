#include "utils.hpp"
namespace shards {
thread_local std::list<NativeStrViewType> *_debugThreadStack;
std::list<NativeStrViewType> &getThreadNameStack() {
  thread_local std::list<NativeStrViewType> stack;
  return stack;
}
} // namespace shards