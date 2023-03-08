#ifndef GFX_SDL_NATIVE_WINDOW
#define GFX_SDL_NATIVE_WINDOW

#include <SDL3/SDL.h>
#include <SDL3/SDL_syswm.h>

#define NEED_SYSWM (GFX_WINDOWS || GFX_ANDROID || GFX_LINUX)

#include <SDL3/SDL_syswm.h>

#include "error_utils.hpp"
#include "platform.hpp"
#include <stdexcept>

inline void *SDL_GetNativeWindowPtr(SDL_Window *window) {
#if NEED_SYSWM
  SDL_SysWMinfo winInfo{};
  if (SDL_GetWindowWMInfo(window, &winInfo, SDL_SYSWM_CURRENT_VERSION) != 0) {
    throw gfx::formatException("Failed to call SDL_GetWindowWMInfo: {}", SDL_GetError());
  }
#endif

#if GFX_WINDOWS
  return winInfo.info.win.window;
#elif GFX_ANDROID
  return winInfo.info.android.window;
#elif GFX_LINUX
  return (void *)winInfo.info.x11.window;
#else
  return nullptr;
#endif
}

inline void *SDL_GetNativeDisplayPtr(SDL_Window *window) {
#if NEED_SYSWM
  SDL_SysWMinfo winInfo{};
  if (SDL_GetWindowWMInfo(window, &winInfo, SDL_SYSWM_CURRENT_VERSION) != 0) {
    throw gfx::formatException("Failed to call SDL_GetWindowWMInfo: {}", SDL_GetError());
  }
#endif

#if GFX_LINUX
  return (void *)winInfo.info.x11.display;
#else
  return nullptr;
#endif
}

#endif // GFX_SDL_NATIVE_WINDOW
