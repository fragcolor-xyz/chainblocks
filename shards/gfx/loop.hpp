#ifndef GFX_LOOP
#define GFX_LOOP

#include <chrono>

namespace gfx {

struct Loop {
  using Duration = std::chrono::high_resolution_clock::duration;
  using TimePoint = std::chrono::high_resolution_clock::time_point;
  using Clock = std::chrono::high_resolution_clock;

  TimePoint startTime;
  TimePoint lastFrameTime;
  bool haveFirstFrame = false;

  Loop();
  bool beginFrame(double targetDeltaTime, double &deltaTime);
  double getAbsoluteTime() const;
};
} // namespace gfx

#endif // GFX_LOOP
