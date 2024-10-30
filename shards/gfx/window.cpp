#include "window.hpp"
#include "SDL3/SDL_properties.h"
#include "error_utils.hpp"
#include "../core/platform.hpp"
#include "sdl_native_window.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include "fmt.hpp"
#include "log.hpp"

#if SH_WINDOWS
#include <Windows.h>
#elif SH_APPLE
#include <SDL3/SDL_metal.h>
#elif SH_ANDROID
#include <android/native_window.h>
#endif

#if SH_IOS
struct UIEdgeInsets {
  double top, left, bottom, right;
};
extern "C" {
void shards_get_uiview_safe_area(UIEdgeInsets *outInsets, void *metalView);
}
#elif SH_ANDROID
#include <jni.h>
extern "C" {
JNIEXPORT void JNICALL Java_com_fragcolor_formabble_FblView_nativeSetViewInsets(JNIEnv *env, jclass cls, jint left, jint top,
                                                                                jint right, jint bottom) {
  SPDLOG_INFO("Received android_set_view_insets: ({}, {}, {}, {})", left, top, right, bottom);
  gfx::Window::viewInset = gfx::float4(left, top, right, bottom);
}
}
#endif

namespace gfx {

float4 Window::viewInset{};

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

  uint32_t flags{};

  flags |= SDL_WINDOW_RESIZABLE;

  int width{options.width}, height{options.height};

// Base OS flags
#if SH_IOS || SH_ANDROID
  flags |= SDL_WINDOW_FULLSCREEN;
#endif

  if ((flags & SDL_WINDOW_FULLSCREEN) != 0) {
    width = 0;
    height = 0;
  }

  flags |= SDL_WINDOW_HIGH_PIXEL_DENSITY;

#if SH_APPLE
  flags |= SDL_WINDOW_METAL;
#endif

  SDL_SetEventEnabled(SDL_EVENT_DROP_FILE, true);

  SDL_SetHint(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, "1");

#if SH_VISION
  // for whatever reason, vision has 1 keyboard always and this will prevent the keyboard from showing up
  SDL_SetHint(SDL_HINT_ENABLE_SCREEN_KEYBOARD, "1");
#endif

  SDL_PropertiesID props = SDL_CreateProperties();
  SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_EXTERNAL_GRAPHICS_CONTEXT_BOOLEAN, true);
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width);
  SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height);
  SDL_SetStringProperty(props, "title", options.title.c_str());
  SDL_SetNumberProperty(props, "flags", flags);
  window = SDL_CreateWindowWithProperties(props);
  SDL_DestroyProperties(props);

  if (!window) {
    throw formatException("SDL_CreateWindow failed: {}", SDL_GetError());
  }

  if (options.fullscreen || (flags & SDL_WINDOW_FULLSCREEN) == SDL_WINDOW_FULLSCREEN) {
    SDL_SetWindowFullscreen(window, true);
  }

#if SH_APPLE
  metalView.emplace(window);
#endif
}

void Window::cleanup() {
  if (window) {
#if SH_APPLE
    metalView.reset();
#endif

    SDL_DestroyWindow(window);
    SDL_Quit();
    window = nullptr;
  }
}

void Window::update() {
  // Update edge insets
#if SH_IOS
  SDL_MetalView uiView = metalView->view;
  UIEdgeInsets insets{};
  shards_get_uiview_safe_area(&insets, uiView);
  viewInset = float4(insets.left, insets.top, insets.right, insets.bottom);
#endif
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
#if SH_APPLE
  return nullptr;
#else
  return (void *)SDL_GetNativeWindowPtr(window);
#endif
}

int2 Window::getDrawableSize() const {
  int2 r;
  SDL_GetWindowSizeInPixels(Window::window, &r.x, &r.y);
  return r;
}

float2 Window::getInputScale() const { return float2(getDrawableSize()) / float2(getSize()); }

int2 Window::getSize() const {
  int2 r;
  SDL_GetWindowSize(window, &r.x, &r.y);
  return r;
}

void Window::resize(int2 targetSize) { SDL_SetWindowSize(window, targetSize.x, targetSize.y); }

// window position
int2 Window::getPosition() const {
  int2 r;
  SDL_GetWindowPosition(window, &r.x, &r.y);
  return r;
}

void Window::move(int2 targetPosition) { SDL_SetWindowPosition(window, targetPosition.x, targetPosition.y); }

static float2 getUIScaleFromDisplayDPI(int displayIndex) { return float2(SDL_GetDisplayContentScale(displayIndex)); }

float Window::getUIScale() const {
  float2 scale{};
#if SH_APPLE
  // On apple, derive display scale from drawable size / window size
  scale = float2(getDrawableSize()) / float2(getSize());
#else
  scale = getUIScaleFromDisplayDPI(SDL_GetDisplayForWindow(window));
#endif
  return std::max<float>(scale.x, scale.y);
}

bool Window::isWindowSizeInPixels() {
#if SH_APPLE
  return false;
#else
  return true;
#endif
}

Window::~Window() { cleanup(); }

bool Window::isFocused() const {
  Uint32 flags = SDL_GetWindowFlags(window);
  return (flags & SDL_WINDOW_INPUT_FOCUS) || (flags & SDL_WINDOW_MOUSE_FOCUS);
}
} // namespace gfx
