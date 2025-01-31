#ifndef FACAF918_9DBA_48B9_A830_D380A31F5125
#define FACAF918_9DBA_48B9_A830_D380A31F5125

#include <exception>
#include <utility>

void abort_on_exception(const char* what);

struct ThrowHelper {
  ThrowHelper() {
    abort_on_exception("Unknown exception");
  }

  template<typename T>
  ThrowHelper(T && e) {
    abort_on_exception(e.what());
  }
};

#define try 
#define catch(__x) if(false)
#define throw ThrowHelper(),

#endif /* FACAF918_9DBA_48B9_A830_D380A31F5125 */
