#include "event_decoder.hpp"
#include "log.hpp"
#include "../shards/core/platform.hpp"

#if SH_EMSCRIPTEN
#include <emscripten/html5.h>
#endif

namespace shards::input {
#if SHARDS_GFX_SDL
void NativeEventDecoder::apply(const NativeEventType &event) {
  if (event.type == SDL_EVENT_MOUSE_WHEEL) {
    buffer.scrollDelta += event.wheel.y;
  } else if (event.type == SDL_EVENT_TEXT_EDITING) {
    auto &ievent = event.edit;

    if (strlen(ievent.text) > 0) {
      if (!decoderState.imeComposing) {
        decoderState.imeComposing = true;
      }

      virtualInputEvents.push_back(TextCompositionEvent{.text = ievent.text});
    }
  } else if (event.type == SDL_EVENT_KEY_DOWN) {
    virtualInputEvents.push_back(
        KeyEvent{.key = event.key.key, .pressed = true, .modifiers = state.modifiers, .repeat = event.key.repeat});
  } else if (event.type == SDL_EVENT_TEXT_INPUT) {
    auto &ievent = event.text;
    if (decoderState.imeComposing) {
      virtualInputEvents.push_back(TextCompositionEndEvent{.text = ievent.text});
      decoderState.imeComposing = false;
    } else {
      virtualInputEvents.push_back(TextEvent{.text = ievent.text});
    }
  } else if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
    virtualInputEvents.push_back(RequestCloseEvent{});
  } else if (event.type == SDL_EVENT_WINDOW_RESIZED) {
  } else if (event.type == SDL_EVENT_DID_ENTER_BACKGROUND) {
    virtualInputEvents.push_back(SuspendEvent{});
  } else if (event.type == SDL_EVENT_DID_ENTER_FOREGROUND) {
    virtualInputEvents.push_back(ResumeEvent{});
  } else if (event.type == SDL_EVENT_FINGER_MOTION) {
    auto &ievent = event.tfinger;
    virtualInputEvents.push_back(PointerTouchMoveEvent{
        .pos = float2(ievent.x, ievent.y) * state.region.size,
        .delta = float2(event.tfinger.dx, event.tfinger.dy) * state.region.size,
        .index = ievent.fingerID,
        .pressure = ievent.pressure,
    });
  } else if (event.type == SDL_EVENT_FINGER_DOWN) {
    auto &ievent = event.tfinger;

    auto &pointer = newState.pointers.getOrInsert(ievent.fingerID);
    pointer.position = float2(ievent.x, ievent.y) * state.region.size;
    pointer.pressure = ievent.pressure;
    pointer.touchId = ievent.touchID;

    virtualInputEvents.push_back(PointerTouchEvent{
        .pos = float2(ievent.x, ievent.y) * state.region.size,
        .delta = float2(event.tfinger.dx, event.tfinger.dy) * state.region.size,
        .index = ievent.fingerID,
        .pressure = ievent.pressure,
        .pressed = true,
    });
  } else if (event.type == SDL_EVENT_DROP_FILE) {
    virtualInputEvents.push_back(DropFileEvent{.path = event.drop.data});
    SPDLOG_LOGGER_DEBUG(getLogger(), "Window dropped file: {}", event.drop.data);
  }
}
#else
static SDL_Keymod extractEventKeyModidiers(const gfx::em::KeyEvent &e) {
  int r{};
  if (e.altKey)
    r |= SDL_KMOD_ALT;
  if (e.ctrlKey)
    r |= SDL_KMOD_CTRL;
  if (e.shiftKey)
    r |= SDL_KMOD_SHIFT;
  return SDL_Keymod(r);
}
/*
.keyCode to SDL keycode
https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent
https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/keyCode
*/
static const SDL_Keycode emscripten_keycode_table[] = {
    /*  0 */ SDLK_UNKNOWN,
    /*  1 */ SDLK_UNKNOWN,
    /*  2 */ SDLK_UNKNOWN,
    /*  3 */ SDLK_CANCEL,
    /*  4 */ SDLK_UNKNOWN,
    /*  5 */ SDLK_UNKNOWN,
    /*  6 */ SDLK_HELP,
    /*  7 */ SDLK_UNKNOWN,
    /*  8 */ SDLK_BACKSPACE,
    /*  9 */ SDLK_TAB,
    /*  10 */ SDLK_UNKNOWN,
    /*  11 */ SDLK_UNKNOWN,
    /*  12 */ SDLK_KP_5,
    /*  13 */ SDLK_RETURN,
    /*  14 */ SDLK_UNKNOWN,
    /*  15 */ SDLK_UNKNOWN,
    /*  16 */ SDLK_LSHIFT,
    /*  17 */ SDLK_LCTRL,
    /*  18 */ SDLK_LALT,
    /*  19 */ SDLK_PAUSE,
    /*  20 */ SDLK_CAPSLOCK,
    /*  21 */ SDLK_UNKNOWN,
    /*  22 */ SDLK_UNKNOWN,
    /*  23 */ SDLK_UNKNOWN,
    /*  24 */ SDLK_UNKNOWN,
    /*  25 */ SDLK_UNKNOWN,
    /*  26 */ SDLK_UNKNOWN,
    /*  27 */ SDLK_ESCAPE,
    /*  28 */ SDLK_UNKNOWN,
    /*  29 */ SDLK_UNKNOWN,
    /*  30 */ SDLK_UNKNOWN,
    /*  31 */ SDLK_UNKNOWN,
    /*  32 */ SDLK_SPACE,
    /*  33 */ SDLK_PAGEUP,
    /*  34 */ SDLK_PAGEDOWN,
    /*  35 */ SDLK_END,
    /*  36 */ SDLK_HOME,
    /*  37 */ SDLK_LEFT,
    /*  38 */ SDLK_UP,
    /*  39 */ SDLK_RIGHT,
    /*  40 */ SDLK_DOWN,
    /*  41 */ SDLK_UNKNOWN,
    /*  42 */ SDLK_UNKNOWN,
    /*  43 */ SDLK_UNKNOWN,
    /*  44 */ SDLK_UNKNOWN,
    /*  45 */ SDLK_INSERT,
    /*  46 */ SDLK_DELETE,
    /*  47 */ SDLK_UNKNOWN,
    /*  48 */ SDLK_0,
    /*  49 */ SDLK_1,
    /*  50 */ SDLK_2,
    /*  51 */ SDLK_3,
    /*  52 */ SDLK_4,
    /*  53 */ SDLK_5,
    /*  54 */ SDLK_6,
    /*  55 */ SDLK_7,
    /*  56 */ SDLK_8,
    /*  57 */ SDLK_9,
    /*  58 */ SDLK_UNKNOWN,
    /*  59 */ SDLK_SEMICOLON,
    /*  60 */ SDLK_BACKSLASH /*SDL_SCANCODE_NONUSBACKSLASH*/,
    /*  61 */ SDLK_EQUALS,
    /*  62 */ SDLK_UNKNOWN,
    /*  63 */ SDLK_MINUS,
    /*  64 */ SDLK_UNKNOWN,
    /*  65 */ SDLK_a,
    /*  66 */ SDLK_b,
    /*  67 */ SDLK_c,
    /*  68 */ SDLK_d,
    /*  69 */ SDLK_e,
    /*  70 */ SDLK_f,
    /*  71 */ SDLK_g,
    /*  72 */ SDLK_h,
    /*  73 */ SDLK_i,
    /*  74 */ SDLK_j,
    /*  75 */ SDLK_k,
    /*  76 */ SDLK_l,
    /*  77 */ SDLK_m,
    /*  78 */ SDLK_n,
    /*  79 */ SDLK_o,
    /*  80 */ SDLK_p,
    /*  81 */ SDLK_q,
    /*  82 */ SDLK_r,
    /*  83 */ SDLK_s,
    /*  84 */ SDLK_t,
    /*  85 */ SDLK_u,
    /*  86 */ SDLK_v,
    /*  87 */ SDLK_w,
    /*  88 */ SDLK_x,
    /*  89 */ SDLK_y,
    /*  90 */ SDLK_z,
    /*  91 */ SDLK_LGUI,
    /*  92 */ SDLK_UNKNOWN,
    /*  93 */ SDLK_APPLICATION,
    /*  94 */ SDLK_UNKNOWN,
    /*  95 */ SDLK_UNKNOWN,
    /*  96 */ SDLK_KP_0,
    /*  97 */ SDLK_KP_1,
    /*  98 */ SDLK_KP_2,
    /*  99 */ SDLK_KP_3,
    /* 100 */ SDLK_KP_4,
    /* 101 */ SDLK_KP_5,
    /* 102 */ SDLK_KP_6,
    /* 103 */ SDLK_KP_7,
    /* 104 */ SDLK_KP_8,
    /* 105 */ SDLK_KP_9,
    /* 106 */ SDLK_KP_MULTIPLY,
    /* 107 */ SDLK_KP_PLUS,
    /* 108 */ SDLK_UNKNOWN,
    /* 109 */ SDLK_KP_MINUS,
    /* 110 */ SDLK_KP_PERIOD,
    /* 111 */ SDLK_KP_DIVIDE,
    /* 112 */ SDLK_F1,
    /* 113 */ SDLK_F2,
    /* 114 */ SDLK_F3,
    /* 115 */ SDLK_F4,
    /* 116 */ SDLK_F5,
    /* 117 */ SDLK_F6,
    /* 118 */ SDLK_F7,
    /* 119 */ SDLK_F8,
    /* 120 */ SDLK_F9,
    /* 121 */ SDLK_F10,
    /* 122 */ SDLK_F11,
    /* 123 */ SDLK_F12,
    /* 124 */ SDLK_F13,
    /* 125 */ SDLK_F14,
    /* 126 */ SDLK_F15,
    /* 127 */ SDLK_F16,
    /* 128 */ SDLK_F17,
    /* 129 */ SDLK_F18,
    /* 130 */ SDLK_F19,
    /* 131 */ SDLK_F20,
    /* 132 */ SDLK_F21,
    /* 133 */ SDLK_F22,
    /* 134 */ SDLK_F23,
    /* 135 */ SDLK_F24,
    /* 136 */ SDLK_UNKNOWN,
    /* 137 */ SDLK_UNKNOWN,
    /* 138 */ SDLK_UNKNOWN,
    /* 139 */ SDLK_UNKNOWN,
    /* 140 */ SDLK_UNKNOWN,
    /* 141 */ SDLK_UNKNOWN,
    /* 142 */ SDLK_UNKNOWN,
    /* 143 */ SDLK_UNKNOWN,
    /* 144 */ SDLK_NUMLOCKCLEAR,
    /* 145 */ SDLK_SCROLLLOCK,
    /* 146 */ SDLK_UNKNOWN,
    /* 147 */ SDLK_UNKNOWN,
    /* 148 */ SDLK_UNKNOWN,
    /* 149 */ SDLK_UNKNOWN,
    /* 150 */ SDLK_UNKNOWN,
    /* 151 */ SDLK_UNKNOWN,
    /* 152 */ SDLK_UNKNOWN,
    /* 153 */ SDLK_UNKNOWN,
    /* 154 */ SDLK_UNKNOWN,
    /* 155 */ SDLK_UNKNOWN,
    /* 156 */ SDLK_UNKNOWN,
    /* 157 */ SDLK_UNKNOWN,
    /* 158 */ SDLK_UNKNOWN,
    /* 159 */ SDLK_UNKNOWN,
    /* 160 */ SDLK_GRAVE,
    /* 161 */ SDLK_UNKNOWN,
    /* 162 */ SDLK_UNKNOWN,
    /* 163 */ SDLK_KP_HASH, /*KaiOS phone keypad*/
    /* 164 */ SDLK_UNKNOWN,
    /* 165 */ SDLK_UNKNOWN,
    /* 166 */ SDLK_UNKNOWN,
    /* 167 */ SDLK_UNKNOWN,
    /* 168 */ SDLK_UNKNOWN,
    /* 169 */ SDLK_UNKNOWN,
    /* 170 */ SDLK_KP_MULTIPLY, /*KaiOS phone keypad*/
    /* 171 */ SDLK_RIGHTBRACKET,
    /* 172 */ SDLK_UNKNOWN,
    /* 173 */ SDLK_MINUS,                /*FX*/
    /* 174 */ SDLK_VOLUMEDOWN,           /*IE, Chrome*/
    /* 175 */ SDLK_VOLUMEUP,             /*IE, Chrome*/
    /* 176 */ SDLK_MEDIA_NEXT_TRACK,     /*IE, Chrome*/
    /* 177 */ SDLK_MEDIA_PREVIOUS_TRACK, /*IE, Chrome*/
    /* 178 */ SDLK_UNKNOWN,
    /* 179 */ SDLK_MEDIA_PLAY, /*IE, Chrome*/
    /* 180 */ SDLK_UNKNOWN,
    /* 181 */ SDLK_MUTE,       /*FX*/
    /* 182 */ SDLK_VOLUMEDOWN, /*FX*/
    /* 183 */ SDLK_VOLUMEUP,   /*FX*/
    /* 184 */ SDLK_UNKNOWN,
    /* 185 */ SDLK_UNKNOWN,
    /* 186 */ SDLK_SEMICOLON, /*IE, Chrome, D3E legacy*/
    /* 187 */ SDLK_EQUALS,    /*IE, Chrome, D3E legacy*/
    /* 188 */ SDLK_COMMA,
    /* 189 */ SDLK_MINUS, /*IE, Chrome, D3E legacy*/
    /* 190 */ SDLK_PERIOD,
    /* 191 */ SDLK_SLASH,
    /* 192 */ SDLK_GRAVE, /*FX, D3E legacy (SDLK_APOSTROPHE in IE/Chrome)*/
    /* 193 */ SDLK_UNKNOWN,
    /* 194 */ SDLK_UNKNOWN,
    /* 195 */ SDLK_UNKNOWN,
    /* 196 */ SDLK_UNKNOWN,
    /* 197 */ SDLK_UNKNOWN,
    /* 198 */ SDLK_UNKNOWN,
    /* 199 */ SDLK_UNKNOWN,
    /* 200 */ SDLK_UNKNOWN,
    /* 201 */ SDLK_UNKNOWN,
    /* 202 */ SDLK_UNKNOWN,
    /* 203 */ SDLK_UNKNOWN,
    /* 204 */ SDLK_UNKNOWN,
    /* 205 */ SDLK_UNKNOWN,
    /* 206 */ SDLK_UNKNOWN,
    /* 207 */ SDLK_UNKNOWN,
    /* 208 */ SDLK_UNKNOWN,
    /* 209 */ SDLK_UNKNOWN,
    /* 210 */ SDLK_UNKNOWN,
    /* 211 */ SDLK_UNKNOWN,
    /* 212 */ SDLK_UNKNOWN,
    /* 213 */ SDLK_UNKNOWN,
    /* 214 */ SDLK_UNKNOWN,
    /* 215 */ SDLK_UNKNOWN,
    /* 216 */ SDLK_UNKNOWN,
    /* 217 */ SDLK_UNKNOWN,
    /* 218 */ SDLK_UNKNOWN,
    /* 219 */ SDLK_LEFTBRACKET,
    /* 220 */ SDLK_BACKSLASH,
    /* 221 */ SDLK_RIGHTBRACKET,
    /* 222 */ SDLK_APOSTROPHE, /*FX, D3E legacy*/
};

static SDL_Keycode mapKeyCode(const gfx::em::KeyEvent &keyEvent) {
  SDL_Keycode keycode = SDLK_UNKNOWN;
  if (keyEvent.domKey_ < SDL_arraysize(emscripten_keycode_table)) {
    keycode = emscripten_keycode_table[keyEvent.domKey_];
    if (keycode != SDLK_UNKNOWN) {
      if (keyEvent.location == DOM_KEY_LOCATION_RIGHT) {
        switch (keycode) {
        case SDLK_LSHIFT:
          keycode = SDLK_RSHIFT;
          break;
        case SDLK_LCTRL:
          keycode = SDLK_RCTRL;
          break;
        case SDLK_LALT:
          keycode = SDLK_RALT;
          break;
        case SDLK_LGUI:
          keycode = SDLK_RGUI;
          break;
        }
      } else if (keyEvent.location == DOM_KEY_LOCATION_NUMPAD) {
        switch (keycode) {
        case SDLK_0:
        case SDLK_INSERT:
          keycode = SDLK_KP_0;
          break;
        case SDLK_1:
        case SDLK_END:
          keycode = SDLK_KP_1;
          break;
        case SDLK_2:
        case SDLK_DOWN:
          keycode = SDLK_KP_2;
          break;
        case SDLK_3:
        case SDLK_PAGEDOWN:
          keycode = SDLK_KP_3;
          break;
        case SDLK_4:
        case SDLK_LEFT:
          keycode = SDLK_KP_4;
          break;
        case SDLK_5:
          keycode = SDLK_KP_5;
          break;
        case SDLK_6:
        case SDLK_RIGHT:
          keycode = SDLK_KP_6;
          break;
        case SDLK_7:
        case SDLK_HOME:
          keycode = SDLK_KP_7;
          break;
        case SDLK_8:
        case SDLK_UP:
          keycode = SDLK_KP_8;
          break;
        case SDLK_9:
        case SDLK_PAGEUP:
          keycode = SDLK_KP_9;
          break;
        case SDLK_RETURN:
          keycode = SDLK_KP_ENTER;
          break;
        case SDLK_DELETE:
          keycode = SDLK_KP_PERIOD;
          break;
        }
      }
    }
  }

  return keycode;
}
void NativeEventDecoder::apply(const NativeEventType &event) {
  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, gfx::em::KeyEvent>) {
          bool pressed = arg.type_ == 0;
          SDL_Keycode keyCode = mapKeyCode(arg);
          SDL_Keymod keyMod = extractEventKeyModidiers(arg);
          virtualInputEvents.push_back(KeyEvent{
              .key = keyCode,
              .pressed = pressed,
              .modifiers = keyMod,
              .repeat = arg.repeat ? 1u : 0u,
          });

          // Emulate text input as well
          if (pressed && arg.key_ != 0 && !arg.ctrlKey && !arg.altKey) {
            auto &te = std::get<TextEvent>(virtualInputEvents.emplace_back(TextEvent{}));
            auto &str = te.text;
            str.resize(8);
            auto after = (char *)utf8catcodepoint(str.data(), arg.key_, str.size());
            if (after == nullptr) {
              virtualInputEvents.pop_back();
            } else {
              str.resize(after - str.data());
            }
          }

          if (pressed) {
            newState.heldKeys.insert(keyCode);
          } else {
            newState.heldKeys.erase(keyCode);
          }
        } else if constexpr (std::is_same_v<T, gfx::em::MouseEvent>) {
          uint8_t buttonIndex = (arg.button + 1);
          float2 newCursorPos(arg.x, arg.y);
          if (arg.type_ == 0) {
            float2 delta =  newCursorPos - state.cursorPosition;
            virtualInputEvents.push_back(PointerMoveEvent{
              .pos = float2(arg.x, arg.y),
              .delta = delta,
            });
          } else {
            bool pressed = arg.type_ == 1;
            virtualInputEvents.push_back(PointerButtonEvent{
                .pos = float2(arg.x, arg.y), .index = buttonIndex, .pressed = pressed, .modifiers = newState.modifiers});
            if (pressed) {
              newState.mouseButtonState |= SDL_BUTTON(buttonIndex);
            } else {
              newState.mouseButtonState &= ~SDL_BUTTON(buttonIndex);
            }
          }
          newState.cursorPosition =  newCursorPos;
        } else if constexpr (std::is_same_v<T, gfx::em::MouseWheelEvent>) {
          buffer.scrollDelta += arg.deltaY;
        }
      },
      event);
}
#endif
} // namespace shards::input
