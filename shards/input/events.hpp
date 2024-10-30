#ifndef A58BF89A_157D_440C_9398_773EB3D0A410
#define A58BF89A_157D_440C_9398_773EB3D0A410
#include "input.hpp"
#include "sdl.hpp"
#include <memory>
#include <compare>
#include <linalg.h>
#include <optional>
#include <variant>
#include "../core/platform.hpp"
#include <boost/container/string.hpp>

// Defines the primary command key
//   on apple this is the cmd key
//   otherwise the ctrl key
#if SH_APPLE
#define KMOD_PRIMARY SDL_KMOD_GUI
#define KMOD_SECONDARY SDL_KMOD_CTRL
#else
#define KMOD_PRIMARY SDL_KMOD_CTRL
#define KMOD_SECONDARY SDL_KMOD_GUI
#endif

namespace shards::input {
using namespace linalg::aliases;

struct PointerTouchMoveEvent {
  float2 pos;
  float2 delta;
  SDL_FingerID index;
  float pressure;

  std::partial_ordering operator<=>(const PointerTouchMoveEvent &other) const = default;
};

struct PointerTouchEvent {
  float2 pos;
  float2 delta;
  SDL_FingerID index;
  float pressure;
  bool pressed;

  std::partial_ordering operator<=>(const PointerTouchEvent &other) const = default;
};

struct PointerMoveEvent {
  float2 pos;
  float2 delta;

  std::partial_ordering operator<=>(const PointerMoveEvent &other) const = default;
};

struct PointerButtonEvent {
  float2 pos;
  int index;
  bool pressed;
  SDL_Keymod modifiers;

  std::partial_ordering operator<=>(const PointerButtonEvent &other) const = default;
};

struct ScrollEvent {
  float delta;

  std::partial_ordering operator<=>(const ScrollEvent &other) const = default;
};

struct KeyEvent {
  SDL_Keycode key;
  bool pressed;
  SDL_Keymod modifiers;
  uint32_t repeat;

  std::partial_ordering operator<=>(const KeyEvent &other) const = default;
};

struct SuspendEvent {
  std::partial_ordering operator<=>(const KeyEvent &other) const { return std::partial_ordering::equivalent; }
};

struct ResumeEvent {
  std::partial_ordering operator<=>(const KeyEvent &other) const { return std::partial_ordering::equivalent; }
};

struct InputRegionEvent {
  InputRegion region;

  std::partial_ordering operator<=>(const KeyEvent &other) const { return std::partial_ordering::equivalent; }
};

struct TextEvent {
  boost::container::string text;

  std::partial_ordering operator<=>(const TextEvent &other) const = default;
};

struct TextCompositionEvent {
  boost::container::string text;

  std::partial_ordering operator<=>(const TextCompositionEvent &other) const = default;
};

struct TextCompositionEndEvent {
  boost::container::string text;

  std::partial_ordering operator<=>(const TextCompositionEndEvent &other) const = default;
};

// Posted when the user or OS attempts to close the window
struct RequestCloseEvent {
  std::partial_ordering operator<=>(const RequestCloseEvent &other) const = default;
};

// Posted when a file is dropped on the window
struct DropFileEvent {
  boost::container::string path;

  std::partial_ordering operator<=>(const DropFileEvent &other) const = default;
};

using Event = std::variant<
    // Pointer events
    PointerTouchMoveEvent, PointerTouchEvent, PointerMoveEvent, PointerButtonEvent, ScrollEvent,
    // Text events
    KeyEvent, TextEvent, TextCompositionEvent, TextCompositionEndEvent,
    // Other events
    SuspendEvent, ResumeEvent, InputRegionEvent, RequestCloseEvent, DropFileEvent>;

inline std::partial_ordering operator<=>(const Event &a, const Event &b) {
  auto ci = a.index() <=> b.index();
  if (ci != 0)
    return ci;

  return std::visit(
      [&](auto &arg) {
        using T = std::decay_t<decltype(arg)>;
        return arg <=> std::get<T>(b);
      },
      a);
}

#if !SH_ANDROID
// Some platforms need these, on android 33 they don't compile
inline bool operator==(const Event &a, const Event &b) { return a <=> b == std::partial_ordering::equivalent; }
#endif

template <typename T, typename... Ts> constexpr size_t get_variant_index(std::variant<Ts...> const &) {
  size_t r = 0;
  auto test = [&](bool b) {
    if (!b)
      ++r;
    return b;
  };
  (test(std::is_same_v<T, Ts>) || ...);
  return r;
}

inline bool isPointerEvent(const Event &event) {
  return event.index() >= get_variant_index<PointerTouchMoveEvent>(event) &&
         event.index() <= get_variant_index<ScrollEvent>(event);
  // return std::get_if<PointerMoveEvent>(&event) || std::get_if<PointerButtonEvent>(&event) || std::get_if<ScrollEvent>(&event)
  // || std::get_if<PointerTouchEvent>(&event) || std::get_if<PointerTouchMoveEvent>(&event);
}

inline bool isKeyEvent(const Event &event) {

  return event.index() >= get_variant_index<KeyEvent>(event) &&
         event.index() <= get_variant_index<TextCompositionEndEvent>(event);
  // return std::get_if<KeyEvent>(&event) || std::get_if<TextEvent>(&event) || std::get_if<TextCompositionEvent>(&event) ||
  // std::get_if<TextCompositionEndEvent>(&event);
}

struct IInputHandler;
struct ConsumedTag {
  std::weak_ptr<IInputHandler> handler;
  ConsumedTag(std::weak_ptr<IInputHandler> handler) : handler(handler) {}
};

struct ConsumableEvent {
  Event event;
  std::optional<ConsumedTag> consumed;

  ConsumableEvent(const Event &event) : event(event) {}
  bool isConsumed() const { return consumed.has_value(); }
  void consume(std::weak_ptr<IInputHandler> by) {
    if (isConsumed())
      return;
    consumed.emplace(by);
  }
};

} // namespace shards::input

#endif /* A58BF89A_157D_440C_9398_773EB3D0A410 */
