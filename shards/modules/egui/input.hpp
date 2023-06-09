#ifndef CA4EC896_C746_4AC6_A229_560CF01EEFEA
#define CA4EC896_C746_4AC6_A229_560CF01EEFEA

#include "egui_types.hpp"
#include <gfx/linalg.hpp>
#include <deque>
#include <vector>

namespace gfx {
struct Window;
struct EguiInputTranslatorPrivate;

struct EguiInputTranslatorArgs {
  const std::vector<SDL_Event> &events;
  Window &window;
  double time;
  float deltaTime;
  // Size of the UI region (in pixels)
  int2 viewportSize;
  // Mapped window region (in pixels)
  int4 mappedWindowRegion;
  // Scale to render the UI at, on top of OS scaling factor
  float scalingFactor = 1.0f;
};

/// <div rustbindgen opaque></div>
struct EguiInputTranslator {
private:
  egui::Input input;
  std::deque<std::string> strings;
  std::vector<egui::InputEvent> events;
  egui::Pos2 lastCursorPosition;
  bool imeComposing{};
  bool textInputActive{};
  float2 windowToEguiScale;
  int4 mappedWindowRegion;
  Window *window{};

public:
  EguiInputTranslator() = default;
  EguiInputTranslator(const EguiInputTranslator &) = delete;
  const EguiInputTranslator &operator=(const EguiInputTranslator &) = delete;

  // Setup input mapping from a window to a specific subregion
  // all coordinates are pixel coordinates
  void setupWindowInput(Window &window, int4 mappedWindowRegion, int2 viewportSize, float scalingFactor = 1.0f);

  // Resets the conversion output
  void begin(double time, float deltaTime);
  // Takes the SDL event and return true when it was converted into an egui event
  bool translateEvent(const SDL_Event &event);
  // Finalizes the egui::Input result
  void end();

  // ALternative to calling the above 4 and returning the output
  const egui::Input *translateFromInputEvents(const EguiInputTranslatorArgs &args);

  const std::vector<egui::InputEvent> &getTranslatedEvents() const { return events; }

  // Manualy push an event into the result
  void pushEvent(const egui::InputEvent &event) { events.push_back(event); }

  const egui::Input *getOutput();

  // Translate from window coordinates to local egui coordinates
  egui::Pos2 translatePointerPos(const egui::Pos2 &pos);

  // Applies egui output to update cursor pos, clipboard, etc.
  void applyOutput(const egui::FullOutput &output);

  // Set or clear the position for the text cursor
  void updateTextCursorPosition(Window &window, const egui::Pos2 *pos);
  void copyText(const char *text);
  void updateCursorIcon(egui::CursorIcon icon);

  // Automatically called before translation to clear internal caches
  void reset();

  static EguiInputTranslator *create();
  static void destroy(EguiInputTranslator *renderer);
};
} // namespace gfx

namespace egui {
inline Pos2 toPos2(const gfx::float2 &v) { return Pos2{.x = v.x, .y = v.y}; }
inline Rect toRect(const gfx::float4 &v) {
  return Rect{
      .min = Pos2{.x = v.x, .y = v.y},
      .max = Pos2{.x = v.z, .y = v.w},
  };
}
} // namespace egui

#endif /* CA4EC896_C746_4AC6_A229_560CF01EEFEA */
