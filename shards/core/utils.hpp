#ifndef F676CA55_B5BE_4F37_8AF8_84A2AD0989F5
#define F676CA55_B5BE_4F37_8AF8_84A2AD0989F5

#include "platform.hpp"

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

template <auto N> struct conststr {
  char value[N];
  constexpr conststr(const char (&str)[N]) { std::copy_n(str, N, value); }
};

#if SH_WINDOWS
inline std::wstring toWindowsWString(std::string_view utf8) {
  std::wstring result;
  result.resize(MultiByteToWideChar(CP_UTF8, 0, utf8.data(), utf8.size(), nullptr, 0));
  result.resize(MultiByteToWideChar(CP_UTF8, 0, utf8.data(), utf8.size(), result.data(), result.size()));
  return result;
}
#endif

struct NativeString {
#if SH_WINDOWS
  NativeString() = default;
  NativeString(std::string_view name) : name(toWindowsWString(name)) {}
  NativeString &operator=(const std::string_view &name) {
    this->name = toWindowsWString(name);
    return *this;
  }
  ~NativeString() {}
  std::wstring name;
  using value_type = std::wstring;
#else
  NativeString() = default;
  NativeString(std::string_view name) : name(name) {}
  NativeString &operator=(const std::string_view &name) {
    this->name = name;
    return *this;
  }
  ~NativeString() {}
  std::string name;
  using value_type = std::string;
#endif
};
#if SH_WINDOWS
using NativeStrType = std::wstring;
using NativeStrViewType = std::wstring_view;
#else
using NativeStrType = std::string;
using NativeStrViewType = std::string_view;
#endif
using NativeCStrType = const NativeStrType::value_type *;

namespace detail {
template <conststr Name> struct ConstNativeStringHolder {
  static inline NativeString v;
  static const auto &get() {
    if (v.name.empty()) {
      v = Name.value;
    }
    return v.name;
  }
};
struct ConstNativeStringValue {
  const NativeStrType &name;
};
} // namespace detail

namespace literals {
template <conststr cts> constexpr auto operator""_ns() {
  return detail::ConstNativeStringValue{detail::ConstNativeStringHolder<cts>::get()};
}
} // namespace literals

// NOTE: Should be null terminated
inline void setThreadName(NativeCStrType v) {
#if SH_WINDOWS
  SetThreadDescription(GetCurrentThread(), v);
#elif SH_LINUX
  pthread_setname_np(pthread_self(), v);
#elif SH_APPLE
  pthread_setname_np(v);
#endif
}

std::list<NativeStrViewType> &getThreadNameStack();

#if SH_DEBUG_THREAD_NAMES
// Use _ns suffix after constant
inline void pushThreadName(const detail::ConstNativeStringValue &v) {
  auto &stack = getThreadNameStack();
  stack.emplace_back(v.name);
  setThreadName(v.name.c_str());
}
// NOTE: by reference, since you should keep the string alive for the duration of the thread
inline void pushThreadName(NativeString &str) {
  auto &stack = getThreadNameStack();
  stack.emplace_back(str.name);
  setThreadName(str.name.c_str());
}
#else
template <typename T> inline void pushThreadName(const T &v) {}
#endif

#if SH_DEBUG_THREAD_NAMES
// You can add this to the debugger watch window (shards::_debugThreadStack)
//  to see the current thread stack
extern thread_local std::list<NativeStrViewType> *_debugThreadStack;
#endif

inline void popThreadName() {
#if SH_DEBUG_THREAD_NAMES
  auto &stack = getThreadNameStack();
  _debugThreadStack = &stack;
  shassert(stack.size() > 0);
  stack.pop_back();
  static NativeString unnamed{"Unnamed thread"};
  setThreadName(stack.size() > 0 ? stack.back().data() : unnamed.name.data());
#endif
}
} // namespace shards

#endif /* F676CA55_B5BE_4F37_8AF8_84A2AD0989F5 */
