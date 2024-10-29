#ifndef C66F3EA3_DB04_4F00_82DD_C7DF2263361D
#define C66F3EA3_DB04_4F00_82DD_C7DF2263361D

#include <emscripten.h>
#include "linalg.hpp"
#include "../core/platform.hpp"

namespace gfx {
struct Window {
  static inline float4 viewInset{};

  void init(const WindowCreationOptions &options = WindowCreationOptions{});
  void cleanup();

  bool isInitialized() const { return true; }

  // Only for platforms that automatically size the output window
  void update();

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

  // Returns true when the window sizes are specified in pixels (on windows for example)
  static bool isWindowSizeInPixels();

  bool isFocused() const { return true; }

  ~Window();
};
} // namespace gfx

#endif /* C66F3EA3_DB04_4F00_82DD_C7DF2263361D */
