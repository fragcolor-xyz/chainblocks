#ifndef F676CA55_B5BE_4F37_8AF8_84A2AD0989F5
#define F676CA55_B5BE_4F37_8AF8_84A2AD0989F5

#include "platform.hpp"
#include "assert.hpp"

#if SH_WINDOWS
#include <windows.h>
#include <processthreadsapi.h>
#elif SH_LINUX || SH_APPLE
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#endif

#include <string>
#include <string_view>
#include <list>
#include <cassert>
#include <spdlog/fmt/fmt.h>

#ifndef SH_DEBUG_THREAD_NAMES
#define SH_DEBUG_THREAD_NAMES 0
#endif

namespace shards {

#if SH_WINDOWS
inline std::wstring toWindowsWString(std::string_view utf8) {
  std::wstring result;
  result.resize(MultiByteToWideChar(CP_UTF8, 0, utf8.data(), utf8.size(), nullptr, 0));
  result.resize(MultiByteToWideChar(CP_UTF8, 0, utf8.data(), utf8.size(), result.data(), result.size()));
  return result;
}
#endif

inline void setThreadName(std::string_view name_sv) {
#if SH_WINDOWS
  std::wstring name = toWindowsWString(name_sv);
  SetThreadDescription(GetCurrentThread(), name.c_str());
#elif SH_LINUX
  std::string name{name_sv};
  pthread_setname_np(pthread_self(), name.c_str());
#elif SH_APPLE
  std::string name{name_sv};
  pthread_setname_np(name.c_str());
#endif
}

std::list<std::string> &getThreadNameStack();

#if SH_DEBUG_THREAD_NAMES
inline void pushThreadName(std::string_view name) {
  auto &stack = getThreadNameStack();
  stack.emplace_back(name);
  setThreadName(name);
}
#else
template <typename T> inline void pushThreadName(const T &v) {}
#endif

#if SH_DEBUG_THREAD_NAMES
// You can add this to the debugger watch window (shards::_debugThreadStack)
//  to see the current thread stack
extern thread_local std::list<std::string>* _debugThreadStack;
#endif

inline void popThreadName() {
#if SH_DEBUG_THREAD_NAMES
  auto &stack = getThreadNameStack();
  _debugThreadStack = &stack;
  shassert(stack.size() > 0);
  stack.pop_back();
  setThreadName(stack.size() > 0 ? stack.back() : "Unnamed thread");
#endif
}
} // namespace shards

#endif /* F676CA55_B5BE_4F37_8AF8_84A2AD0989F5 */
