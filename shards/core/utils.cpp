#include "utils.hpp"
namespace shards {
thread_local std::list<const char*> *_debugThreadStack;
std::list<const char*> &getThreadNameStack() {
  thread_local std::list<const char*> stack;
  return stack;
}
} // namespace shards