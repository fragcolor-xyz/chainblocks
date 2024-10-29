#ifndef BB4427B0_5641_4FB1_86E3_6E9D1AD7C986
#define BB4427B0_5641_4FB1_86E3_6E9D1AD7C986
#include "input.hpp"
#include "state.hpp"
#include "events.hpp"
#include "log.hpp"
#include "sdl.hpp"
#include "event_decoder.hpp"
#include <boost/container/string.hpp>
#include <boost/container/flat_map.hpp>
#include <type_traits>


namespace shards::input {
// Keeps track of all input state separately and synthesizes it's own events by
// diffing the previous state with the new state
struct DetachedInput {
  // Events
  std::vector<Event> virtualInputEvents;

  InputState state;

  NativeEventDecoderBuffer buffers[2];
  NativeEventDecoderState decoderState;

  size_t currentBufferIndex{};

public:
  float getScrollDelta() const { return buffers[currentBufferIndex].scrollDelta; }

  // Given a callback function that receives an apply function, update the state and synthesize events
  // the apply function receives a consumable event that is applied to the input state
  //
  // Example:
  //  input.update([&](auto& apply) {
  //   for(auto& event : events) {
  //     apply(event, true /* hasFocus */);
  //   }
  //  });
  using UpdateApplyFn = decltype([](const ConsumableEvent &consumableEvent, bool hasFocus) {});
  template <typename T> std::enable_if_t<std::is_invocable_v<T, UpdateApplyFn>> update(T callback) {
    swapBuffers();

    virtualInputEvents.clear();

    InputState newState = state;
    auto applyFn = [&](const ConsumableEvent &consumableEvent, bool hasFocus) {
      apply(consumableEvent, hasFocus, newState); //
    };
    callback(applyFn);

    // NOTE: These events are already translated from the original
    // synthesizeKeyUpEvents(newState);
    // synthesizeKeyDownEvents(newState);
    // synthesizeMouseButtonEvents(newState);
    synthesizeMouseEvents(newState);

    state = newState;
  }

// Apply Native events to the new state
#if SHARDS_GFX_SDL
  using NativeEventType = SDL_Event;
#else
  using NativeEventType = gfx::em::EventVar;
#endif

  using UpdateApplyFn2 = decltype([](const NativeEventType &evt) {});
  template <typename T> std::enable_if_t<std::is_invocable_v<T, UpdateApplyFn2>> updateNative(T callback) {
    swapBuffers();

    virtualInputEvents.clear();

    InputState newState = state;
    newState.update();
    NativeEventDecoder decoder{decoderState, virtualInputEvents, state, newState, buffers[getBufferIndex(1)]};
    auto applyFn = [&](const NativeEventType &event) {
      decoder.apply(event);
    };
    callback(applyFn);

    endUpdate(newState);
  }

  void update(InputState &state) {
    beginUpdate();
    endUpdate(state);
  }

private:
  // called before endUpdate, apply() can be called with SDL_Events after this
  void beginUpdate() { virtualInputEvents.clear(); }

  // Updates the input state and synthesizes events based on
  //  the difference between the previous state and the new state
  void endUpdate(const InputState &inputState) {
    swapBuffers();

#if SHARDS_GFX_SDL
    synthesizeKeyUpEvents(inputState);

    synthesizeTouchUpEvents(inputState);

    if constexpr (Pointer::HasPersistentPointer) {
      synthesizeMouseButtonEvents(inputState);
      synthesizeMouseEvents(inputState);
      synthesizeScrollEvents(inputState);
    } else {
      synthesizeMouseFromTouchEvents(inputState);
    }

    state = inputState;

    if constexpr (!Pointer::HasPersistentPointer) {
      updateCursorPositionFromPointerState(state);
    }
#else
    state = inputState;
    synthesizeScrollEvents(inputState);
#endif
  }

  // Apply a consumable event to this detached input state
  // any non-consumed event will be handled normally
  // consumed events will only contribute to loss-of-focus & button/key release events
  void apply(const ConsumableEvent &consumableEvent, bool hasFocus, InputState &newState) {
    bool consumed = consumableEvent.isConsumed();
    auto &buffer = buffers[getBufferIndex(1)];
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, PointerMoveEvent>) {
            if (!consumed)
              newState.cursorPosition = arg.pos;
          } else if constexpr (std::is_same_v<T, PointerButtonEvent>) {
            if (!consumed && hasFocus && arg.pressed && !newState.isMouseButtonHeld(arg.index)) {
              newState.mouseButtonState |= SDL_BUTTON(arg.index);
              virtualInputEvents.push_back(PointerButtonEvent{
                  .pos = newState.cursorPosition,
                  .index = arg.index,
                  .pressed = true,
                  .modifiers = arg.modifiers,
              });
            }
            if (!arg.pressed && newState.isMouseButtonHeld(arg.index)) {
              newState.mouseButtonState &= ~uint32_t(SDL_BUTTON(arg.index));
              virtualInputEvents.push_back(PointerButtonEvent{
                  .pos = newState.cursorPosition,
                  .index = arg.index,
                  .pressed = false,
                  .modifiers = arg.modifiers,
              });
            }
          } else if constexpr (std::is_same_v<T, PointerTouchMoveEvent>) {
            if (!consumed) {
              if (auto ptr = newState.pointers.get(arg.index)) {
                ptr->get().position = arg.pos;
              }
            }
          } else if constexpr (std::is_same_v<T, PointerTouchEvent>) {
            if (!consumed && hasFocus && arg.pressed && !newState.pointers.get(arg.index)) {
              auto &pointer = newState.pointers.getOrInsert(arg.index);
              pointer.position = arg.pos;
              pointer.pressure = arg.pressure;
              virtualInputEvents.push_back(PointerTouchEvent{
                  .pos = pointer.position,
                  .index = arg.index,
                  .pressed = true,
              });
            }
            if (!arg.pressed && newState.pointers.get(arg.index)) {
              newState.pointers.erase(arg.index);
              virtualInputEvents.push_back(PointerTouchEvent{
                  .pos = arg.pos,
                  .index = arg.index,
                  .pressed = false,
              });
            }
          } else if constexpr (std::is_same_v<T, KeyEvent>) {
            if (!consumed && hasFocus && arg.pressed && !newState.isKeyHeld(arg.key)) {
              newState.heldKeys.insert(arg.key);
              virtualInputEvents.push_back(KeyEvent{
                  .key = arg.key,
                  .pressed = true,
                  .modifiers = arg.modifiers,
              });
            }
            if (newState.isKeyHeld(arg.key)) {
              if (!arg.pressed) {
                newState.heldKeys.erase(arg.key);
                virtualInputEvents.push_back(KeyEvent{
                    .key = arg.key,
                    .pressed = false,
                    .modifiers = arg.modifiers,
                });
              } else if (arg.repeat > 0) {
                virtualInputEvents.push_back(KeyEvent{
                    .key = arg.key,
                    .pressed = true,
                    .modifiers = arg.modifiers,
                    .repeat = arg.repeat,
                });
              }
            }
          } else if constexpr (std::is_same_v<T, ScrollEvent>) {
            if (!consumed && hasFocus)
              buffer.scrollDelta += arg.delta;
          } else if constexpr (std::is_same_v<T, SuspendEvent>) {
            virtualInputEvents.push_back(SuspendEvent{});
          } else if constexpr (std::is_same_v<T, ResumeEvent>) {
            virtualInputEvents.push_back(ResumeEvent{});
          } else if constexpr (std::is_same_v<T, InputRegionEvent>) {
            newState.region = arg.region;
          } else if constexpr (std::is_same_v<T, TextEvent>) {
            virtualInputEvents.push_back(TextEvent{.text = arg.text});
          } else if constexpr (std::is_same_v<T, TextCompositionEvent>) {
            virtualInputEvents.push_back(TextCompositionEvent{.text = arg.text});
          } else if constexpr (std::is_same_v<T, TextCompositionEndEvent>) {
            virtualInputEvents.push_back(TextCompositionEndEvent{.text = arg.text});
          }
        },
        consumableEvent.event);
  }

  size_t getBufferIndex(int offset) const { return size_t((int(currentBufferIndex) + offset) % std::size(buffers)); }
  void swapBuffers() {
    buffers[currentBufferIndex].reset();
    currentBufferIndex = getBufferIndex(1);
  }

  void synthesizeKeyUpEvents(const InputState &inputState) {
    for (auto key : state.heldKeys) {
      if (!inputState.isKeyHeld(key)) {
        virtualInputEvents.push_back(KeyEvent{
            .key = key,
            .pressed = false,
            .modifiers = inputState.modifiers,
        });
      }
    }
  }

  void synthesizeKeyDownEvents(const InputState &inputState) {
    for (auto key : inputState.heldKeys) {
      if (!state.heldKeys.contains(key)) {
        virtualInputEvents.push_back(KeyEvent{
            .key = key,
            .pressed = true,
            .modifiers = inputState.modifiers,
        });
      }
    }
  }

  void synthesizeMouseButtonEvents(const InputState &inputState) {
    uint32_t oldState = state.mouseButtonState;
    uint32_t newState = inputState.mouseButtonState;
    if (oldState != newState) {
      for (int i = 1; i <= NumMouseButtons; i++) {
        if ((newState & SDL_BUTTON(i)) != 0 && (oldState & SDL_BUTTON(i)) == 0) {
          virtualInputEvents.push_back(PointerButtonEvent{
              .pos = inputState.cursorPosition,
              .index = i,
              .pressed = true,
              .modifiers = inputState.modifiers,
          });
        } else if ((newState & SDL_BUTTON(i)) == 0 && (oldState & SDL_BUTTON(i)) != 0) {
          virtualInputEvents.push_back(PointerButtonEvent{
              .pos = inputState.cursorPosition,
              .index = i,
              .pressed = false,
              .modifiers = inputState.modifiers,
          });
        }
      }
    }
  }

  void synthesizeMouseEvents(const InputState &inputState) {
    if (inputState.cursorPosition != state.cursorPosition) {
      float2 delta = inputState.cursorPosition - state.cursorPosition;
      virtualInputEvents.push_back(PointerMoveEvent{.pos = inputState.cursorPosition, .delta = delta});
    }
  }

  void synthesizeScrollEvents(const InputState &inputState) {
    if (getScrollDelta() != 0.0f) {
      virtualInputEvents.push_back(ScrollEvent{.delta = getScrollDelta()});
    }
  }

  void synthesizeTouchUpEvents(const InputState &inputState) {
    for (auto &[k, v] : state.pointers.pointers) {
      if (!inputState.pointers.get(k)) {
        virtualInputEvents.push_back(PointerTouchEvent{
            .pos = v.position,
            .index = k,
            .pressed = false,
        });
      }
    }
  }

  void synthesizeMouseFromTouchEvents(const InputState &inputState) {
    auto &oldState = state.pointers;
    auto &newState = inputState.pointers;
    if (oldState.any() != newState.any()) {
      bool pressed = newState.any();
      virtualInputEvents.push_back(PointerButtonEvent{
          .pos = pressed ? newState.centroid() : oldState.centroid(),
          .index = SDL_BUTTON_LEFT,
          .pressed = pressed,
          .modifiers = inputState.modifiers,
      });
    }

    if (oldState.any() && newState.any()) {
      float2 oldPos = oldState.centroid();
      float2 newPos = newState.centroid();
      if (oldPos != newPos) {
        float2 delta = newPos - oldPos;
        virtualInputEvents.push_back(PointerMoveEvent{.pos = newPos, .delta = delta});
      }
    }
  }

  void updateCursorPositionFromPointerState(InputState &inputState) {
    if (inputState.pointers.any())
      inputState.cursorPosition = inputState.pointers.centroid();
  }
};
} // namespace shards::input

#endif /* BB4427B0_5641_4FB1_86E3_6E9D1AD7C986 */
