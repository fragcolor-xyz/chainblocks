#ifndef A1E28346_A7FE_49E4_BB48_A95C21114BAD
#define A1E28346_A7FE_49E4_BB48_A95C21114BAD

#include "events.hpp"
#include "messages.hpp"
#include <gfx/fmt.hpp>
#include <spdlog/fmt/fmt.h>
#include <magic_enum.hpp>

namespace magic_enum::customize {
template <> struct enum_range<SDL_Keymod> {
  static constexpr bool is_flags = true;
};
} // namespace magic_enum::customize
namespace shards::input {

inline std::string debugFormat(const Event &event) {
  return std::visit(
      [&](auto &&arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, PointerTouchMoveEvent>) {
          return fmt::format("PointerTouchMoveEvent {{ pos: {}, delta: {}, index: {}, pressure: {} }}", arg.pos, arg.delta,
                             arg.index, arg.pressure);
        } else if constexpr (std::is_same_v<T, PointerTouchEvent>) {
          return fmt::format("PointerTouchEvent {{ pos: {}, delta: {}, index: {}, pressure: {}, pressed: {} }}", arg.pos,
                             arg.delta, arg.index, arg.pressure, arg.pressed);
        } else if constexpr (std::is_same_v<T, PointerMoveEvent>) {
          return fmt::format("PointerMoveEvent {{ pos: {}, delta: {} }}", arg.pos, arg.delta);
        } else if constexpr (std::is_same_v<T, PointerButtonEvent>) {
          return fmt::format("PointerButtonEvent {{ index: {}, pressed: {}, modifiers: {}, pos: {} }}", arg.index, arg.pressed,
                             arg.modifiers, arg.pos);
        } else if constexpr (std::is_same_v<T, ScrollEvent>) {
          return fmt::format("ScrollEvent {{ delta: {} }}", arg.delta);
        } else if constexpr (std::is_same_v<T, KeyEvent>) {
          return fmt::format("KeyEvent {{ keycode: {}, pressed: {}, modifiers: {}}}", arg.key, arg.pressed, arg.modifiers);
        } else if constexpr (std::is_same_v<T, SuspendEvent>) {
          return fmt::format("SuspendEvent {{}}");
        } else if constexpr (std::is_same_v<T, ResumeEvent>) {
          return fmt::format("ResumeEvent {{}}");
        } else if constexpr (std::is_same_v<T, InputRegionEvent>) {
          return fmt::format("InputRegionEvent {{ size: {}, pixelSize: {}, uiScalingFactor: {} }}", arg.region.size,
                             arg.region.pixelSize, arg.region.uiScalingFactor);
        } else if constexpr (std::is_same_v<T, TextEvent>) {
          return fmt::format("TextEvent {{ text: {} }}", arg.text);
        } else if constexpr (std::is_same_v<T, TextCompositionEvent>) {
          return fmt::format("TextCompositionEvent {{ text: {} }}", arg.text);
        } else if constexpr (std::is_same_v<T, TextCompositionEndEvent>) {
          return fmt::format("TextCompositionEndEvent {{ text: {} }}", arg.text);
        } else if constexpr (std::is_same_v<T, DropFileEvent>) {
          return fmt::format("DropFileEvent {{ {} }}", arg.path);
        } else {
          return fmt::format("UnknownEvent {{}}");
        }
      },
      event);
}

inline std::string debugFormat(const Message &message) {
  return std::visit(
      [&](auto &&arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, BeginTextInputMessage>) {
          return fmt::format("BeginTextInputMessage {{ inputRect: {{x: {}, y: {}, w: {}, h: {}}} }}", arg.inputRect.x,
                             arg.inputRect.y, arg.inputRect.w, arg.inputRect.h);
        } else if constexpr (std::is_same_v<T, EndTextInputMessage>) {
          return fmt::format("EndTextInputMessage {{}}");
        } else if constexpr (std::is_same_v<T, SetCursorMessage>) {
          return fmt::format("SetCursorMessage {{ visible: {}, cursor: {} }}", arg.visible, magic_enum::enum_name(arg.cursor));
        } else if constexpr (std::is_same_v<T, TerminateMessage>) {
          return fmt::format("TerminateMessage {{}}");
        } else {
          return fmt::format("UnknownMessage {{}}");
        }
      },
      message);
}
} // namespace shards::input

#endif /* A1E28346_A7FE_49E4_BB48_A95C21114BAD */
