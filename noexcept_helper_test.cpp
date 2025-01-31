

void abort_on_exception(const char *what);

struct ThrowHelper {
  ThrowHelper() { abort_on_exception("Unknown exception"); }
  // ThrowHelper(const std::exception& e) { abort_on_exception(e.what()); }

  template <typename T> ThrowHelper(T &&e) { abort_on_exception("unknown"); }
  template <typename T> ThrowHelper operator+(T &&) { return *this; }
  ThrowHelper operator++() { return *this; }
  ThrowHelper operator++(int) { return *this; }
};

#define try
#define catch(__x) if (false)
// #define throw ++(ThrowHelper)
#define throw(...) noexcept
#define throw 

#define _ALLOW_KEYWORD_MACROS 1

#include <exception>
#include <utility>
#include <stdio.h>
#include <stdexcept>

int main() {
  try {
    printf("hi");
    throw std::runtime_error("test");
  } catch (auto &e) {
    throw;
  }
}

void abort_on_exception(const char *what) {
  printf("abort_on_exception: %s\n", what);
  abort();
}