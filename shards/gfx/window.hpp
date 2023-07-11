#ifndef AEC3EB9B_7819_42B0_BB31_40818880ECE2
#define AEC3EB9B_7819_42B0_BB31_40818880ECE2

#include "linalg.hpp"
#include "platform.hpp"
#include <SDL_events.h>
#include <string>
#include <vector>

#if GFX_APPLE
#include "platform_surface.hpp"
#endif

struct SDL_Window;
namespace gfx {
/// <div rustbindgen opaque></div>
struct WindowCreationOptions {
  int width = 1280;
  int height = 720;
  bool fullscreen = false;
  std::string title;
};

struct Window {
  SDL_Window *window = nullptr;

#if GFX_APPLE
  std::optional<MetalViewContainer> metalView;
#endif

  void init(const WindowCreationOptions &options = WindowCreationOptions{});
  void cleanup();

  bool isInitialized() const { return window != nullptr; }

  template <typename T> void pollEventsForEach(T &&callback) {
    SDL_Event event;
    while (pollEvent(event)) {
      callback(event);
    }
  }

  void pollEvents(std::vector<SDL_Event> &events);
  bool pollEvent(SDL_Event &outEvent);

  void *getNativeWindowHandle();

  // draw surface size
  int2 getDrawableSize() const;

  // The scale that converts from input coordinates to drawable surface coordinates
  float2 getInputScale() const;

  // window size
  int2 getSize() const;

  void resize(int2 targetSize);

  // window position
  int2 getPosition() const;

  void move(int2 targetPosition);

  // OS UI scaling factor for the display the window is on
  float getUIScale() const;

  ~Window();
};
}; // namespace gfx

#endif /* AEC3EB9B_7819_42B0_BB31_40818880ECE2 */
