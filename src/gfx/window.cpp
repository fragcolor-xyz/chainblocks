#include "window.hpp"
#include "error_utils.hpp"
#include "platform.hpp"
#include "sdl_native_window.hpp"
#include <SDL.h>
#include <SDL_video.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include "fmt.hpp"

#if GFX_WINDOWS
#include <Windows.h>
#elif GFX_APPLE
#include <SDL_metal.h>
#elif GFX_EMSCRIPTEN
#include <emscripten/html5.h>
#elif GFX_ANDROID
#include <android/native_window.h>
#endif

namespace gfx {
void Window::init(const WindowCreationOptions &options) {
  if (window)
    throw std::logic_error("Already initialized");

#ifdef _WIN32
  SetProcessDPIAware();
#endif

  auto initErr = SDL_Init(SDL_INIT_EVENTS | SDL_INIT_VIDEO);
  if (initErr != 0) {
    throw formatException("SDL_Init failed: {}", SDL_GetError());
  }

  uint32_t flags = SDL_WINDOW_SHOWN;

  int width{options.width}, height{options.height};

// Base OS flags
#if GFX_APPLE || GFX_ANDROID
  flags |= SDL_WINDOW_FULLSCREEN;
#else
  flags |= SDL_WINDOW_RESIZABLE | (options.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
#endif

  if ((flags & SDL_WINDOW_FULLSCREEN) != 0) {
    width = 0;
    height = 0;
  }

#if !GFX_EMSCRIPTEN
  flags |= SDL_WINDOW_ALLOW_HIGHDPI;
#endif
#if GFX_APPLE
  flags |= SDL_WINDOW_METAL;
#endif

  SDL_SetHint(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, "1");
  SDL_SetHint(SDL_HINT_VIDEO_EXTERNAL_CONTEXT, "1");
  window = SDL_CreateWindow(options.title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, flags);

  if (!window) {
    throw formatException("SDL_CreateWindow failed: {}", SDL_GetError());
  }
}

void Window::cleanup() {
  if (window) {
    SDL_DestroyWindow(window);
    SDL_Quit();
    window = nullptr;
  }
}

void Window::pollEvents(std::vector<SDL_Event> &events) {
  events.clear();
  SDL_Event event;
  while (pollEvent(event)) {
    events.push_back(event);
  }
}

bool Window::pollEvent(SDL_Event &outEvent) { return SDL_PollEvent(&outEvent); }

void *Window::getNativeWindowHandle() {
#if GFX_APPLE
  return nullptr;
#elif GFX_EMSCRIPTEN
  return (void *)("#canvas");
#else
  return (void *)SDL_GetNativeWindowPtr(window);
#endif
}

int2 Window::getDrawableSize() const {
  int2 r;
#if GFX_APPLE
  SDL_Metal_GetDrawableSize(window, &r.x, &r.y);
#elif GFX_ANDROID
  ANativeWindow *nativeWindow = (ANativeWindow *)SDL_GetNativeWindowPtr(window);
  r.x = ANativeWindow_getWidth(nativeWindow);
  r.y = ANativeWindow_getHeight(nativeWindow);
#else
  r = getSize();
#endif
  return r;
}

float2 Window::getInputScale() const { return float2(getDrawableSize()) / float2(getSize()); }

int2 Window::getSize() const {
  int2 r;
  SDL_GetWindowSize(window, &r.x, &r.y);
  return r;
}

static float2 getUIScaleFromDisplayDPI(int displayIndex) {
  float2 dpi;
  float diagonalDpi;
  if (SDL_GetDisplayDPI(displayIndex, &diagonalDpi, &dpi.x, &dpi.y) != 0) {
    return float2(1.0f);
  }

#if GFX_WINDOWS
  // DPI for 100% on windows
  return dpi / 96;
#else
  const float referenceDpi = 440;
  const float referenceScale = 4.0;
  return dpi / referenceDpi * referenceScale;
#endif
}

float2 Window::getUIScale() const {
#if GFX_APPLE
  // On apple, derive display scale from drawable size / window size
  return float2(getDrawableSize()) / float2(getSize());
#else
  return getUIScaleFromDisplayDPI(SDL_GetWindowDisplayIndex(window));
#endif
}

Window::~Window() { cleanup(); }
} // namespace gfx
