#ifndef GFX_PLATFORM
#define GFX_PLATFORM

#ifndef GFX_DEBUG
#ifdef NDEBUG
#define GFX_DEBUG 0
#else
#define GFX_DEBUG 1
#endif
#endif

#if defined(_WIN32)
#define GFX_WINDOWS 1
#elif defined(__ANDROID__)
#define GFX_ANDROID 1
#elif defined(__EMSCRIPTEN__)
#define GFX_EMSCRIPTEN 1
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#define GFX_APPLE 1

#if TARGET_IPHONE_SIMULATOR || TARGET_OS_IPHONE
#define GFX_IOS 1
#else
#define GFX_OSX 1
#endif

#elif defined(__linux__) && !defined(__EMSCRIPTEN__)
#define GFX_LINUX 1
#else
#define GFX_UNKNOWN 1
#endif

#ifndef GFX_ANDROID
#define GFX_ANDROID 0
#endif

#ifndef GFX_APPLE
#define GFX_APPLE 0
#endif

#ifndef GFX_OSX
#define GFX_OSX 0
#endif

#ifndef GFX_IOS
#define GFX_IOS 0
#endif

#ifndef GFX_WINDOWS
#define GFX_WINDOWS 0
#endif

#ifndef GFX_LINUX
#define GFX_LINUX 0
#endif

#ifndef GFX_EMSCRIPTEN
#define GFX_EMSCRIPTEN 0
#endif

#endif // GFX_PLATFORM
