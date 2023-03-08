#include "input.hpp"
#include "../linalg.hpp"
#include "../platform.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <gfx/window.hpp>
#include "renderer.hpp"
#include <map>

namespace gfx {

// Defines the primary command key
//   on apple this is the cmd key
//   otherwise the ctrl key
#if GFX_APPLE
#define KMOD_PRIMARY SDL_KMOD_GUI
#else
#define KMOD_PRIMARY SDL_KMOD_CTRL
#endif

struct SDLCursor {
  SDL_Cursor *cursor{};
  SDLCursor(SDL_SystemCursor id) { cursor = SDL_CreateSystemCursor(id); }
  SDLCursor(SDLCursor &&rhs) {
    cursor = rhs.cursor;
    rhs.cursor = nullptr;
  }
  SDLCursor(const SDLCursor &) = delete;
  SDLCursor &operator=(const SDLCursor &) = delete;
  SDLCursor &operator=(SDLCursor &&rhs) {
    cursor = rhs.cursor;
    rhs.cursor = nullptr;
    return *this;
  }
  ~SDLCursor() {
    if (cursor)
      SDL_DestroyCursor(cursor);
  }
  operator SDL_Cursor *() const { return cursor; }
};

struct CursorMap {
  std::map<egui::CursorIcon, SDLCursor> cursorMap{};

  CursorMap() {
    cursorMap.insert_or_assign(egui::CursorIcon::Text, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_IBEAM));
    cursorMap.insert_or_assign(egui::CursorIcon::PointingHand, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_HAND));
    cursorMap.insert_or_assign(egui::CursorIcon::Crosshair, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_CROSSHAIR));
    cursorMap.insert_or_assign(egui::CursorIcon::ResizeNeSw, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_SIZENESW));
    cursorMap.insert_or_assign(egui::CursorIcon::ResizeNwSe, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_SIZENWSE));
    cursorMap.insert_or_assign(egui::CursorIcon::Default, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_ARROW));
    cursorMap.insert_or_assign(egui::CursorIcon::ResizeVertical, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_SIZENS));
    cursorMap.insert_or_assign(egui::CursorIcon::ResizeHorizontal, SDLCursor(SDL_SystemCursor::SDL_SYSTEM_CURSOR_SIZEWE));
  }

  SDL_Cursor *getCursor(egui::CursorIcon cursor) {
    auto it = cursorMap.find(cursor);
    if (it == cursorMap.end())
      return nullptr;
    return it->second;
  }

  static CursorMap &getInstance() {
    static CursorMap map;
    return map;
  }
};

static egui::ModifierKeys translateModifierKeys(SDL_Keymod flags) {
  return egui::ModifierKeys{
      .alt = (flags & SDL_KMOD_ALT) != 0,
      .ctrl = (flags & SDL_KMOD_CTRL) != 0,
      .shift = (flags & SDL_KMOD_SHIFT) != 0,
      .macCmd = (flags & SDL_KMOD_GUI) != 0,
      .command = (flags & SDL_KMOD_GUI) != 0,
  };
}

void EguiInputTranslator::setupWindowInput(Window &window, int4 mappedWindowRegion, int2 viewportSize, float scalingFactor) {
  this->window = &window;

  // UI Points per pixel
  float eguiDrawScale = EguiRenderer::getDrawScale(window) * scalingFactor;

  // Drawable/Window scale
  float2 inputScale = window.getInputScale();
  windowToEguiScale = inputScale / eguiDrawScale;

  // Convert from pixel to window coordinates
  this->mappedWindowRegion = int4(float4(mappedWindowRegion) / float4(inputScale.x, inputScale.y, inputScale.x, inputScale.y));

  // Take viewport size and scale it by the draw scale
  float2 viewportSizeFloat = float2(float(viewportSize.x), float(viewportSize.y));
  float2 eguiScreenSize = viewportSizeFloat / eguiDrawScale;

  input.screenRect = egui::Rect{
      .min = egui::Pos2{0, 0},
      .max = egui::Pos2{eguiScreenSize.x, eguiScreenSize.y},
  };
  input.pixelsPerPoint = eguiDrawScale;
}

void EguiInputTranslator::begin(double time, float deltaTime) {
  reset();

  egui::ModifierKeys modifierKeys = translateModifierKeys(SDL_GetModState());

  input.time = time;
  input.predictedDeltaTime = deltaTime;
  input.modifierKeys = modifierKeys;
}

bool EguiInputTranslator::translateEvent(const SDL_Event &sdlEvent) {
  using egui::InputEvent;
  using egui::InputEventType;

  bool handled = false;
  auto newEvent = [&](egui::InputEventType type) -> InputEvent & {
    InputEvent &event = events.emplace_back();
    event.common.type = type;
    handled = true;
    return event;
  };

  auto updateCursorPosition = [&](int32_t x, int32_t y) -> const egui::Pos2 & {
    float2 cursorPosition = float2(x, y);
    return lastCursorPosition = egui::Pos2{.x = cursorPosition.x, .y = cursorPosition.y};
  };

  switch (sdlEvent.type) {
  case SDL_EVENT_MOUSE_BUTTON_DOWN:
  case SDL_EVENT_MOUSE_BUTTON_UP: {
    auto &ievent = sdlEvent.button;
    auto &oevent = newEvent(InputEventType::PointerButton).pointerButton;
    switch (ievent.button) {
    case SDL_BUTTON_LEFT:
      oevent.button = egui::PointerButton::Primary;
      break;
    case SDL_BUTTON_MIDDLE:
      oevent.button = egui::PointerButton::Middle;
      break;
    case SDL_BUTTON_RIGHT:
      oevent.button = egui::PointerButton::Secondary;
      break;
    default:
      // ignore this button
      events.pop_back();
      break;
    }
    oevent.pressed = ievent.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
    oevent.modifiers = input.modifierKeys;
    oevent.pos = translatePointerPos(updateCursorPosition(ievent.x, ievent.y));

    // Synthesize PointerGone event to indicate there's no more fingers
    if (!oevent.pressed) {
      (void)newEvent(InputEventType::PointerGone).pointerGone;
    }
    break;
  }
  case SDL_EVENT_MOUSE_MOTION: {
    auto &ievent = sdlEvent.motion;
    auto &oevent = newEvent(InputEventType::PointerMoved).pointerMoved;
    oevent.pos = translatePointerPos(updateCursorPosition(ievent.x, ievent.y));
    break;
  }
  case SDL_EVENT_MOUSE_WHEEL: {
    auto &ievent = sdlEvent.wheel;
    auto &oevent = newEvent(InputEventType::Scroll).scroll;
    oevent.delta = egui::Pos2{
        .x = float(ievent.x),
        .y = float(ievent.y),
    };
    break;
  }
  case SDL_EVENT_TEXT_EDITING: {
    auto &ievent = sdlEvent.edit;

    std::string editingText = ievent.text;
    if (!imeComposing) {
      imeComposing = true;
      newEvent(InputEventType::CompositionStart);
    }

    auto &evt = newEvent(InputEventType::CompositionUpdate);
    evt.compositionUpdate.text = strings.emplace_back(ievent.text).c_str();
    break;
  }
  case SDL_EVENT_TEXT_INPUT: {
    auto &ievent = sdlEvent.text;

    if (imeComposing) {
      auto &evt = newEvent(InputEventType::CompositionEnd);
      evt.compositionEnd.text = strings.emplace_back(ievent.text).c_str();
      imeComposing = false;
    } else {
      auto &evt = newEvent(InputEventType::Text);
      evt.text.text = strings.emplace_back(ievent.text).c_str();
    }
    break;
  }
  case SDL_EVENT_KEY_DOWN:
  case SDL_EVENT_KEY_UP: {
    auto &ievent = sdlEvent.key;
    auto &oevent = newEvent(InputEventType::Key).key;
    oevent.key = SDL_KeyCode(ievent.keysym.sym);
    oevent.pressed = ievent.type == SDL_EVENT_KEY_DOWN;
    oevent.modifiers = translateModifierKeys(SDL_Keymod(ievent.keysym.mod));

    // Translate cut/copy/paste using the standard keys combos
    if (ievent.type == SDL_EVENT_KEY_DOWN) {
      if ((ievent.keysym.mod & KMOD_PRIMARY) && ievent.keysym.sym == SDLK_c) {
        newEvent(InputEventType::Copy);
      } else if ((ievent.keysym.mod & KMOD_PRIMARY) && ievent.keysym.sym == SDLK_v) {
        auto &evt = newEvent(InputEventType::Paste);

        evt.paste.str = strings.emplace_back(SDL_GetClipboardText()).c_str();
      } else if ((ievent.keysym.mod & KMOD_PRIMARY) && ievent.keysym.sym == SDLK_x) {
        newEvent(InputEventType::Cut);
      }
    }
    break;
  }
  }
  return handled;
}

void EguiInputTranslator::end() {
  input.inputEvents = events.data();
  input.numInputEvents = events.size();

  input.numDroppedFiles = 0;
  input.numHoveredFiles = 0;
}

const egui::Input *EguiInputTranslator::translateFromInputEvents(const EguiInputTranslatorArgs &args) {
  begin(args.time, args.deltaTime);
  setupWindowInput(args.window, args.mappedWindowRegion, args.viewportSize, args.scalingFactor);
  for (const auto &event : args.events)
    translateEvent(event);
  end();
  return getOutput();
}

const egui::Input *EguiInputTranslator::getOutput() { return &input; }

egui::Pos2 EguiInputTranslator::translatePointerPos(const egui::Pos2 &pos) {
  return egui::Pos2{
      (pos.x - mappedWindowRegion.x) * windowToEguiScale.x,
      (pos.y - mappedWindowRegion.y) * windowToEguiScale.y,
  };
}

void EguiInputTranslator::applyOutput(const egui::FullOutput &output) {
  if (window)
    updateTextCursorPosition(*window, output.textCursorPosition);

  if (output.copiedText)
    copyText(output.copiedText);

  if (input.numInputEvents > 0)
    updateCursorIcon(output.cursorIcon);
}

void EguiInputTranslator::updateTextCursorPosition(Window &window, const egui::Pos2 *pos) {
  if (pos) {
    SDL_Rect rect;
    rect.x = int(pos->x);
    rect.y = int(pos->y);
    rect.w = 20;
    rect.h = 20;
    SDL_SetTextInputRect(&rect);

    if (!textInputActive) {
      SDL_StartTextInput();
      textInputActive = true;
    }
  } else {
    if (textInputActive) {
      SDL_StopTextInput();
      textInputActive = false;
      imeComposing = false;
    }
  }
}

void EguiInputTranslator::copyText(const char *text) { SDL_SetClipboardText(text); }

void EguiInputTranslator::updateCursorIcon(egui::CursorIcon icon) {
  if (icon == egui::CursorIcon::None) {
    SDL_HideCursor();
  } else {
    SDL_ShowCursor();
    SDL_Cursor *cursor = CursorMap::getInstance().getCursor(icon);
    SDL_SetCursor(cursor);
  }
}

void EguiInputTranslator::reset() {
  strings.clear();
  events.clear();
  this->window = nullptr;
}

EguiInputTranslator *EguiInputTranslator::create() { return new EguiInputTranslator(); }

void EguiInputTranslator::destroy(EguiInputTranslator *obj) { delete obj; }

} // namespace gfx
