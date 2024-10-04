#include "utils.hpp"
namespace shards {
thread_local std::list<std::string> *_debugThreadStack;
std::list<std::string> &getThreadNameStack() {
  thread_local std::list<std::string> stack;
  return stack;
}
} // namespace shards