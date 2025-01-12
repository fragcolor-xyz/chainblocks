/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include <winsock2.h>

#include <Windows.h>
#include <psapi.h>

#include "shards/core/module.hpp"
#include "desktop.capture.win.hpp"
#include "desktop.hpp"

using namespace shards;

#ifndef NDEBUG
#define DBG_CHECK_OFF 1
#else
#define DBG_CHECK_OFF 0
#endif

#define DBG_CHECK(_expr_) \
  if (DBG_CHECK_OFF) {    \
    assert((_expr_));     \
  } else {                \
    (_expr_);             \
  }

namespace Desktop {
template <> HWND WindowBase<HWND>::WindowDefault() { return NULL; }

class WindowWindows : public WindowBase<HWND> {
protected:
  std::vector<wchar_t> _wTitle;
  std::vector<wchar_t> _wClass;

public:
  void setParam(int index, const SHVar &value) override {
    WindowBase<HWND>::setParam(index, value);

    switch (index) {
    case 0:
      _wTitle.resize(MultiByteToWideChar(CP_UTF8, 0, _winName.c_str(), -1, 0, 0));
      MultiByteToWideChar(CP_UTF8, 0, _winName.c_str(), -1, &_wTitle[0], _wTitle.size());
      break;
    case 1:
      _wClass.resize(MultiByteToWideChar(CP_UTF8, 0, _winClass.c_str(), -1, 0, 0));
      MultiByteToWideChar(CP_UTF8, 0, _winClass.c_str(), -1, &_wClass[0], _wClass.size());
      break;
    default:
      break;
    }
  }
};

class HasWindow : public WindowWindows {
public:
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!_window || !IsWindow(_window)) {
      if (_winClass.size() > 0) {
        _window = FindWindowW(&_wClass[0], &_wTitle[0]);
      } else {
        _window = FindWindowW(NULL, &_wTitle[0]);
      }
    }

    return _window ? Var::True : Var::False;
  }
};

class WaitWindow : public WindowWindows {
public:
  static SHTypesInfo outputTypes() { return Globals::windowType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    while (!_window || !IsWindow(_window)) {
      if (_winClass.size() > 0) {
        _window = FindWindowW(&_wClass[0], &_wTitle[0]);
      } else {
        _window = FindWindowW(NULL, &_wTitle[0]);
      }

      if (!_window) {
        SH_SUSPEND(context, 0.1);
      }
    }

    return Var::Object(_window, CoreCC, windowCC);
  }
};

static HWND AsHWND(const SHVar &var) {
  if (var.valueType == SHType::Object && var.payload.objectVendorId == CoreCC && var.payload.objectTypeId == windowCC) {
    return reinterpret_cast<HWND>(var.payload.objectValue);
  }
  return NULL;
}

struct PID : public PIDBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      DWORD pid;
      GetWindowThreadProcessId(hwnd, &pid);
      return Var(int64_t(pid));
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
  };
};

struct IsForeground : public ActiveBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      auto currentForeground = GetForegroundWindow();
      return hwnd == currentForeground ? Var::True : Var::False;
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
  };
};

struct SetForeground : public WinOpBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      auto foreground = GetForegroundWindow();
      if (foreground != hwnd) {
        auto target = GetWindowThreadProcessId(foreground, NULL);
        auto current = GetWindowThreadProcessId(hwnd, NULL);
        AttachThreadInput(current, target, TRUE);

        BOOL res;
        res = SetForegroundWindow(hwnd);
        assert(res);
        BringWindowToTop(hwnd);
        SetFocus(hwnd);

        AttachThreadInput(current, target, FALSE);
      }
      return input;
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
  };
};

struct NotForeground : public ActiveBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      auto currentForeground = GetForegroundWindow();
      return hwnd != currentForeground ? Var::True : Var::False;
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
  };
};

struct Resize : public ResizeWindowBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      RECT rect;
      DBG_CHECK(GetWindowRect(hwnd, &rect));
      DBG_CHECK(::MoveWindow(hwnd, rect.left, rect.top, _width, _height, TRUE));
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }

    return input;
  };
};

struct Move : public MoveWindowBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      RECT rect;
      DBG_CHECK(GetWindowRect(hwnd, &rect));
      auto width = rect.right - rect.left;
      auto height = rect.bottom - rect.top;
      DBG_CHECK(::MoveWindow(hwnd, _x, _y, width, height, TRUE));
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }

    return input;
  };
};

struct Bounds : public SizeBase {
  // gets client size, work area
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      RECT r;
      DBG_CHECK(GetClientRect(hwnd, &r));
      int64_t width = r.right - r.left;
      int64_t height = r.bottom - r.top;
      return Var(width, height);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
  };
};

struct Size : public SizeBase {
  // gets window size, including OS bar
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      RECT r;
      DBG_CHECK(GetWindowRect(hwnd, &r));
      int64_t width = r.right - r.left;
      int64_t height = r.bottom - r.top;
      return Var(width, height);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
  };
};

struct SetBorderless : public WinOpBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      LONG lStyle = GetWindowLong(hwnd, GWL_STYLE);
      lStyle &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE | WS_SYSMENU);
      SetWindowLong(hwnd, GWL_STYLE, lStyle);
      LONG lExStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
      lExStyle &= ~(WS_EX_DLGMODALFRAME | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE);
      SetWindowLong(hwnd, GWL_EXSTYLE, lExStyle);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
    return input;
  };
};
// TODO add Unset

struct SetClickthrough : public WinOpBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      LONG lStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
      // Make it layered if it's not
      if (!(lStyle & WS_EX_LAYERED)) {
        lStyle |= WS_EX_LAYERED;
      }
      // Make transparent
      lStyle |= WS_EX_TRANSPARENT;
      SetWindowLong(hwnd, GWL_EXSTYLE, lStyle);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
    return input;
  };
};

struct UnsetClickthrough : public WinOpBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      LONG lStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
      if (lStyle & WS_EX_TRANSPARENT) {
        lStyle &= ~WS_EX_TRANSPARENT;
      }
      SetWindowLong(hwnd, GWL_EXSTYLE, lStyle);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
    return input;
  };
};

struct SetTopmost : public WinOpBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
    return input;
  };
};

struct UnsetTopmost : public WinOpBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }
    return input;
  };
};

struct SetTitle : public SetTitleBase {
  std::vector<wchar_t> _wTitle;

  void setParam(int index, const SHVar &value) override {
    SetTitleBase::setParam(index, value);

    _wTitle.resize(MultiByteToWideChar(CP_UTF8, 0, _title.c_str(), -1, 0, 0));
    MultiByteToWideChar(CP_UTF8, 0, _title.c_str(), -1, &_wTitle[0], _wTitle.size());
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto hwnd = AsHWND(input);
    if (hwnd) {
      SetWindowTextW(hwnd, &_wTitle[0]);
    } else {
      throw ActivationError("Input object was not a Desktop's window!");
    }

    return input;
  };
};

struct PixelBase {
  std::string variableName;

  static inline ParamsInfo params = ParamsInfo(
      ParamsInfo::Param("Window", SHCCSTR("The window variable name to use as coordinate origin."), Globals::windowVarOrNone));

  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  ExposedInfo exposedInfo;
  SHVar *windowVar = nullptr;

  void cleanup(SHContext *context) {
    if (windowVar) {
      releaseVariable(windowVar);
      windowVar = nullptr;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (variableName.size() == 0) {
      return {};
    } else {
      return SHExposedTypesInfo(exposedInfo);
    }
  }

  virtual void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      if (value.valueType == SHType::ContextVar) {
        variableName = SHSTRVIEW(value);
        exposedInfo = ExposedInfo(
            ExposedInfo::Variable(variableName.c_str(), SHCCSTR("The window to use as origin."), Globals::windowType));
      } else {
        variableName.clear();
      }
      windowVar = nullptr;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    auto res = SHVar();
    switch (index) {
    case 0:
      if (variableName.size() == 0) {
        return Var::Empty;
      } else {
        SHVar v{};
        v.valueType = SHType::ContextVar;
        v.payload.stringValue = variableName.c_str();
        v.payload.stringLen = variableName.size();
        return v;
      }
    default:
      break;
    }
    return res;
  }

  static inline std::vector<std::unique_ptr<DXGIDesktopCapture>> Grabbers;

  static DXGIDesktopCapture *findOrCreate(int x, int y) {
    for (auto &grabber : Grabbers) {
      if (x >= grabber->left() && x < grabber->right() && y >= grabber->top() && y < grabber->bottom()) {
        return grabber.get();
      }
    }

    auto info = DXGIDesktopCapture::FindScreen(x, y, 1, 1);
    if (info.adapter != nullptr) {
      SHLOG_INFO("Found screen for pixel grabber");
      auto grabber = new DXGIDesktopCapture(info);
      Grabbers.emplace_back(grabber);
      info.adapter->Release();
      info.output->Release();
      return grabber;
    }

    SHLOG_ERROR("Failed to find screen for pixel grabber.");
    return nullptr;
  }

  DXGIDesktopCapture *preActivate(SHContext *context, int &x, int &y) {
    if (windowVar == nullptr && variableName.size() != 0) {
      windowVar = referenceVariable(context, variableName.c_str());
    }

    if (windowVar) {
      auto wnd = AsHWND(*windowVar);
      if (!wnd) {
        throw ActivationError("Input object was not a Desktop's window!");
      }
      POINT po;
      po.x = x;
      po.y = y;
      ClientToScreen(wnd, &po);
      LogicalToPhysicalPoint(wnd, &po);
      x = po.x;
      y = po.y;
    }

    return findOrCreate(x, y);
  }
};

struct CaptureFrame {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    int len = PixelBase::Grabbers.size();
    for (int i = len - 1; i >= 0; i--) {
      auto &grabber = PixelBase::Grabbers[i];
      auto state = grabber->capture();
      if (state != DXGIDesktopCapture::Timeout && state != DXGIDesktopCapture::Normal) {
        PixelBase::Grabbers.erase(PixelBase::Grabbers.begin() + i);
      } else {
        grabber->update();
      }
    }
    return input;
  }
};

struct Pixel : public PixelBase {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::ColorType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    int x = input.payload.int2Value[0];
    int y = input.payload.int2Value[1];
    auto grabber = preActivate(context, x, y);
    if (grabber) {
      auto img = grabber->image();
      // properly clamp the single pixel
      x = std::max(0, x);
      y = std::max(0, y);
      x = std::min(grabber->width() - 1, x);
      y = std::min(grabber->height() - 1, y);
      // grab from the bgra image
      auto pindex = ((grabber->width() * y) + x) * 4;
      SHColor pixel = {img[pindex + 2], img[pindex + 1], img[pindex], 255};
      return Var(pixel);
    } else {
      SHColor pixel = {0, 0, 0, 255};
      return Var(pixel);
    }
  }
};

struct Pixels : public PixelBase {
  static SHTypesInfo inputTypes() { return CoreInfo::Int4Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  OwnedVar _output;

  SHVar activate(SHContext *context, const SHVar &input) {
    int left = input.payload.int4Value[0];
    int top = input.payload.int4Value[1];
    const int right = input.payload.int4Value[2];
    const int bottom = input.payload.int4Value[3];
    int w = right - left;
    int h = bottom - top;

    auto grabber = preActivate(context, left, top);
    if (grabber) {
      auto img = grabber->image();
      // adjust, due to virtual coordinates of multiple screens
      auto x = left - grabber->left();
      auto y = top - grabber->top();

      // properly clamp the capture area
      x = std::clamp(x, 0, grabber->width() - w);
      y = std::clamp(y, 0, grabber->height() - h);
      w = std::clamp(w, 2, grabber->width());
      h = std::clamp(h, 2, grabber->height());

      auto nbytes = 4 * w * h;
      _output = makeImage(w * h * nbytes);
      _output.payload.imageValue->width = w;
      _output.payload.imageValue->height = h;
      _output.payload.imageValue->channels = 4;
      _output.payload.imageValue->flags = SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA;
      SHImage &outImage = *_output.payload.imageValue;

      // grab from the bgra image
      auto yindex = y;
      for (auto i = 0; i < h; i++) {
        auto xindex = x;
        for (auto j = 0; j < w; j++) {
          auto sindex = ((grabber->width() * yindex) + xindex) * 4;
          auto dindex = ((w * i) + j) * 4;
          outImage.data[dindex + 2] = img[sindex + 0];
          outImage.data[dindex + 1] = img[sindex + 1];
          outImage.data[dindex + 0] = img[sindex + 2];
          outImage.data[dindex + 3] = img[sindex + 3];
          xindex++;
        }
        yindex++;
      }

      return _output;
    } else {
      throw ActivationError("Failed to grab screen.");
    }
  }
};

struct WaitKeyEvent : public WaitKeyEventBase {
  struct KeyboardHookState {
    int refCount = 0;
    HHOOK hook{};
    std::vector<WaitKeyEvent *> receivers;
  };

  static inline thread_local KeyboardHookState *hookState = nullptr;

  bool attached = false;
  std::deque<SHVar> events;

  void keyboardEvent(int state, int vkCode) { events.push_back(Var(state, vkCode)); }

  void cleanup(SHContext *context) {
    if (attached && hookState) {
      auto selfIt = std::find(hookState->receivers.begin(), hookState->receivers.end(), this);
      hookState->receivers.erase(selfIt);
      hookState->refCount--;
      if (hookState->refCount == 0) {
        auto res = UnhookWindowsHookEx(hookState->hook);
        assert(res);
        delete hookState;
        hookState = nullptr;
      }
      attached = false;
    }
  }

  static LRESULT __attribute__((stdcall)) Hookproc(int nCode, WPARAM wParam, LPARAM lParam) {
    auto wp = int(wParam);
    auto lp = reinterpret_cast<KBDLLHOOKSTRUCT *>(lParam);
    for (auto &rcvr : hookState->receivers) {
      if (wp == WM_KEYDOWN || wp == WM_SYSKEYDOWN) {
        rcvr->keyboardEvent(0, lp->vkCode);
      } else if (wp == WM_KEYUP || wp == WM_SYSKEYUP) {
        rcvr->keyboardEvent(1, lp->vkCode);
      }
    }
    return CallNextHookEx(hookState->hook, nCode, wParam, lParam);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!attached) {
      if (!hookState) {
        hookState = new KeyboardHookState();
        hookState->refCount++;
        hookState->receivers.push_back(this);
        hookState->hook = SetWindowsHookEx(WH_KEYBOARD_LL, Hookproc, NULL, 0);
        assert(hookState->hook);
      } else {
        hookState->refCount++;
        hookState->receivers.push_back(this);
      }
      attached = true;
    }

    while (events.empty()) {
      // We need to do this to allow the hook to work, when there is no message loop/window
      MSG msg;
      PeekMessage(&msg, 0, 0, 0, 0);
      // Wait for events
      SH_SUSPEND(context, 0);
    }

    auto event = events.front();
    events.pop_front();
    return event;
  }
};

struct SendKeyEvent : public SendKeyEventBase {
  SHVar *_window = nullptr;

  void cleanup(SHContext *context) {
    if (_window) {
      releaseVariable(_window);
      _window = nullptr;
    }
  }

  static bool IsExtendedKey(UINT vkCode) {
    // List of extended key codes (this is just an example, modify as needed)
    static const std::set<UINT> extendedKeys = {
        VK_RCONTROL, VK_RMENU,                                       // Right Control and Alt keys
        VK_INSERT,   VK_DELETE, VK_HOME, VK_END,  VK_PRIOR, VK_NEXT, // Insert, Delete, Home, End, Page Up, Page Down
        VK_LEFT,     VK_RIGHT,  VK_UP,   VK_DOWN,                    // Arrow keys
        VK_NUMLOCK,  VK_CANCEL,                                      // Num Lock, Break
        VK_SNAPSHOT, VK_DIVIDE                                       // Print Screen, Divide (on numpad)
                                                                     // Add other extended keys as needed
    };

    return extendedKeys.find(vkCode) != extendedKeys.end();
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto state = input.payload.int2Value[0];
    UINT vkCode = input.payload.int2Value[1];

    // Determine if the key is an extended key
    bool isExtendedKey = IsExtendedKey(vkCode);

    // Retrieve window only if needed
    if (_windowVarName.size() > 0) {
      if (!_window) {
        _window = referenceVariable(context, _windowVarName.c_str());
      }
      auto window = AsHWND(*_window);
      if (!window) {
        throw ActivationError("Input object was not a Desktop's window!");
      }

      // Calculate lparam for WM_KEYDOWN or WM_KEYUP
      auto scanCode = MapVirtualKey(vkCode, MAPVK_VK_TO_VSC);
      auto lparam = scanCode << 16;
      if (state == 0) {       // Key Press
        lparam |= 0x00000001; // Repeat count
        if (isExtendedKey) {
          lparam |= 0x01000000; // Set the extended key flag
        }
      } else {                // Key Release
        lparam |= 0xC0000001; // Release flag and repeat count
        if (!isExtendedKey) {
          lparam &= ~0x01000000; // Clear the extended key flag if not an extended key
        }
      }

      // Simulate keyboard event for a specific window
      PostMessage(window, state == 0 ? WM_KEYDOWN : WM_KEYUP, vkCode, lparam);
    } else {
      // Simulate global keyboard event
      INPUT keyboardEvent{};
      keyboardEvent.type = INPUT_KEYBOARD;
      keyboardEvent.ki.wVk = vkCode;
      keyboardEvent.ki.wScan = MapVirtualKey(vkCode, MAPVK_VK_TO_VSC);
      keyboardEvent.ki.dwFlags = (state == 1) ? KEYEVENTF_KEYUP : 0;
      if (isExtendedKey) {
        keyboardEvent.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;
      }
      keyboardEvent.ki.dwExtraInfo = GetMessageExtraInfo();
      SendInput(1, &keyboardEvent, sizeof(keyboardEvent));
    }

    return input;
  }
};

struct GetMousePos : public MousePosBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    POINT p;

    if (!GetPhysicalCursorPos(&p)) {
      throw ActivationError("GetPhysicalCursorPos failed.");
    }

    auto wnd = AsHWND(_window.get());
    if (wnd) {
      // both return bool but checking might be not necessary
      // if we fail hopefully the struct is unchanged
      PhysicalToLogicalPoint(wnd, &p);
      ScreenToClient(wnd, &p);
    }

    return Var((int64_t)p.x, (int64_t)p.y);
  }
};

struct SetMousePos : public MousePosBase {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }

  SHVar activate(SHContext *context, const SHVar &input) {
    POINT p;
    p.x = input.payload.int2Value[0];
    p.y = input.payload.int2Value[1];

    auto wnd = AsHWND(_window.get());
    if (wnd) {
      // both return bool but checking might be not necessary
      // if we fail hopefully the struct is unchanged
      ClientToScreen(wnd, &p);
      LogicalToPhysicalPoint(wnd, &p);
    }

    if (!SetPhysicalCursorPos(p.x, p.y)) {
      throw ActivationError("SetPhysicalCursorPos failed.");
    }

    return input;
  }
};

struct SetMouseRelativePos : public MousePosBase {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }

  SHVar activate(SHContext *context, const SHVar &input) {
    // Delta movement
    int dx = input.payload.int2Value[0];
    int dy = input.payload.int2Value[1];

    // Using SendInput for relative movement
    INPUT inputMove = {0};
    inputMove.type = INPUT_MOUSE;
    inputMove.mi.dx = dx;
    inputMove.mi.dy = dy;
    inputMove.mi.mouseData = 0;
    inputMove.mi.dwFlags = MOUSEEVENTF_MOVE;
    inputMove.mi.time = 0;
    inputMove.mi.dwExtraInfo = 0;

    if (!SendInput(1, &inputMove, sizeof(INPUT))) {
      throw ActivationError("SendInput failed for relative mouse movement.");
    }

    return input;
  }
};

struct Tap : public MousePosBase {
  typedef BOOL (*InitializeTouchInjectionProc)(UINT32 maxCount, DWORD dwMode);
  typedef BOOL (*InjectTouchInputProc)(UINT32 count, const POINTER_TOUCH_INFO *contacts);
  struct GlobalInjector {
    static inline InjectTouchInputProc InjectTouch;
    GlobalInjector() {
      // win7 compatibility
      auto user32 = LoadLibraryA("User32.dll");
      assert(user32);
      auto touchInit = (InitializeTouchInjectionProc)GetProcAddress(user32, "InitializeTouchInjection");
      if (touchInit) {
        DBG_CHECK(touchInit(10, TOUCH_FEEDBACK_NONE));
        InjectTouch = (InjectTouchInputProc)GetProcAddress(user32, "InjectTouchInput");
      } else {
        InjectTouch = nullptr;
      }
    }
  };

  static inline GlobalInjector *sInjectorState = nullptr;

  void setup() {
    // we cannot use static inline with value type..
    // runtime error with 32 bit builds...
    if (!sInjectorState)
      sInjectorState = new GlobalInjector();
  }

  bool _delays = true;
  bool _longTap = false;

  static inline ParamsInfo params = ParamsInfo(
      MousePosBase::params,
      ParamsInfo::Param("Long",
                        SHCCSTR("A big delay will be injected after tap down "
                                "to simulate a long tap."),
                        CoreInfo::BoolType),
      ParamsInfo::Param("Natural", SHCCSTR("Small pauses will be injected after tap events down & up."), CoreInfo::BoolType));

  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }

  void setParam(int index, const SHVar &value) {
    if (index == 0)
      MousePosBase::setParam(index, value);
    else {
      if (index == 1)
        _longTap = value.payload.boolValue;
      else // 2
        _delays = value.payload.boolValue;
    }
  }

  SHVar getParam(int index) {
    if (index == 0)
      return MousePosBase::getParam(index);
    else {
      if (index == 1)
        return Var(_longTap);
      else
        return Var(_delays);
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!GlobalInjector::InjectTouch)
      return input;

    POINTER_TOUCH_INFO pinfo;
    memset(&pinfo, 0x0, sizeof(POINTER_TOUCH_INFO));
    pinfo.pointerInfo.pointerType = PT_TOUCH;
    pinfo.pointerInfo.pointerId = 0;
    pinfo.touchFlags = TOUCH_FLAG_NONE;
    pinfo.touchMask = TOUCH_MASK_CONTACTAREA | TOUCH_MASK_ORIENTATION | TOUCH_MASK_PRESSURE;
    pinfo.orientation = 90; // 0~359 afaik
    pinfo.pressure = 1024;  // 0~1024 afaik

    auto wnd = AsHWND(_window.get());
    if (wnd) {
      POINT p;
      p.x = input.payload.int2Value[0];
      p.y = input.payload.int2Value[1];
      ClientToScreen(wnd, &p);
      LogicalToPhysicalPoint(wnd, &p);
      pinfo.pointerInfo.ptPixelLocation.x = p.x;
      pinfo.pointerInfo.ptPixelLocation.y = p.y;
    } else {
      pinfo.pointerInfo.ptPixelLocation.x = input.payload.int2Value[0];
      pinfo.pointerInfo.ptPixelLocation.y = input.payload.int2Value[1];
    }

    pinfo.rcContact.left = pinfo.pointerInfo.ptPixelLocation.x - 2;
    pinfo.rcContact.right = pinfo.pointerInfo.ptPixelLocation.x + 2;
    pinfo.rcContact.top = pinfo.pointerInfo.ptPixelLocation.y - 2;
    pinfo.rcContact.bottom = pinfo.pointerInfo.ptPixelLocation.y + 2;

    // down
    pinfo.pointerInfo.pointerFlags = POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT;
    if (!GlobalInjector::InjectTouch(1, &pinfo)) {
      SHLOG_ERROR("InjectTouchInput (down) error: {0:x}", GetLastError());
      throw ActivationError("InjectTouchInput failed.");
    }

    if (_delays) {
      SH_SUSPEND(context, 0.05);
    }

    if (_longTap) {
      // we need to send updates or it will fail, a simple sleep is not enough
      for (auto i = 0; i < 10; i++) {
        pinfo.pointerInfo.pointerFlags = POINTER_FLAG_UPDATE | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT;
        if (!GlobalInjector::InjectTouch(1, &pinfo)) {
          SHLOG_ERROR("InjectTouchInput (down) error: {0:x}", GetLastError());
          throw ActivationError("InjectTouchInput failed.");
        }
        SH_SUSPEND(context, 0.1);
      }
    }

    // up
    pinfo.pointerInfo.pointerFlags = POINTER_FLAG_UP;
    if (!GlobalInjector::InjectTouch(1, &pinfo)) {
      SHLOG_ERROR("InjectTouchInput (up) error: {0:x}", GetLastError());
      throw ActivationError("InjectTouchInput failed.");
    }

    if (_delays) {
      SH_SUSPEND(context, 0.05);
    }

    return input;
  }
};

template <DWORD MBD, DWORD MBU> struct Click : public MousePosBase {
  bool _delays = true;

  static inline ParamsInfo params = ParamsInfo(
      MousePosBase::params,
      ParamsInfo::Param("Natural", SHCCSTR("Small pauses will be injected after click events down & up."), CoreInfo::BoolType));

  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }

  void setParam(int index, const SHVar &value) {
    if (index == 0)
      MousePosBase::setParam(index, value);
    else {
      _delays = value.payload.boolValue;
    }
  }

  SHVar getParam(int index) {
    if (index == 0)
      return MousePosBase::getParam(index);
    else {
      return Var(_delays);
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    INPUT event;
    event.type = INPUT_MOUSE;
    event.mi.mouseData = 0;
    event.mi.dwFlags = 0;
    event.ki.dwExtraInfo = GetMessageExtraInfo();
    event.mi.dwFlags = MOUSEEVENTF_ABSOLUTE;

    auto wnd = AsHWND(_window.get());
    if (wnd) {
      POINT p;
      p.x = input.payload.int2Value[0];
      p.y = input.payload.int2Value[1];
      ClientToScreen(wnd, &p);
      LogicalToPhysicalPoint(wnd, &p);
      event.mi.dx = p.x;
      event.mi.dy = p.y;
    } else {
      event.mi.dx = input.payload.int2Value[0];
      event.mi.dy = input.payload.int2Value[1];
    }

    event.mi.dx = (event.mi.dx * 65536) / GetSystemMetrics(SM_CXSCREEN);
    event.mi.dy = (event.mi.dy * 65536) / GetSystemMetrics(SM_CYSCREEN);

    // down
    event.mi.dwFlags = event.mi.dwFlags | MBD;
    if (!SendInput(1, &event, sizeof(INPUT))) {
      SHLOG_ERROR("SendInput (down) error: {0:x}", GetLastError());
      throw ActivationError("LeftClick failed.");
    }

    if (_delays) {
      SH_SUSPEND(context, 0.05);
    }

    // up
    event.mi.dwFlags = event.mi.dwFlags | MBU;
    if (!SendInput(1, &event, sizeof(INPUT))) {
      SHLOG_ERROR("SendInput (up) error: {0:x}", GetLastError());
      throw ActivationError("LeftClick failed.");
    }

    if (_delays) {
      SH_SUSPEND(context, 0.05);
    }

    return input;
  }
};

typedef Click<MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP> LeftClick;
typedef Click<MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP> RightClick;
typedef Click<MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP> MiddleClick;

template <bool HORIZONTAL> struct Scroll : public MousePosBase {
  static SHTypesInfo inputTypes() { return CoreInfo::FloatType; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    INPUT event;
    event.type = INPUT_MOUSE;
    event.mi.mouseData = 0;
    event.mi.dwFlags = 0;
    event.ki.dwExtraInfo = GetMessageExtraInfo();
    event.mi.dwFlags = MOUSEEVENTF_ABSOLUTE;

    // just get current mouse pos and pass it thru
    event.mi.dwFlags = MOUSEEVENTF_ABSOLUTE;
    POINT p;
    if (!GetPhysicalCursorPos(&p)) {
      throw ActivationError("GetPhysicalCursorPos failed.");
    }
    event.mi.dx = p.x;
    event.mi.dy = p.y;
    event.mi.dx = (event.mi.dx * 65536) / GetSystemMetrics(SM_CXSCREEN);
    event.mi.dy = (event.mi.dy * 65536) / GetSystemMetrics(SM_CYSCREEN);

    if constexpr (HORIZONTAL) {
      event.mi.dwFlags = MOUSEEVENTF_HWHEEL;
      event.mi.mouseData = (DWORD)floor(input.payload.floatValue * WHEEL_DELTA);
    } else {
      event.mi.dwFlags = MOUSEEVENTF_WHEEL;
      event.mi.mouseData = (DWORD)floor(input.payload.floatValue * WHEEL_DELTA);
    }

    if (!SendInput(1, &event, sizeof(INPUT))) {
      SHLOG_ERROR("SendInput (scroll) error: {0:x}", GetLastError());
      throw ActivationError("Scroll failed.");
    }

    return input;
  }
};

struct ScrollHorizontal : public Scroll<true> {};
struct ScrollVertical : public Scroll<false> {};

struct CursorBitmap {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::ImageType; }

  HDC _hdcMem{};
  HBITMAP _bitmap{};
  // SHImage _img;
  OwnedVar _output;

  ~CursorBitmap() {
    DeleteObject(_bitmap);
    DeleteDC(_hdcMem);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (_bitmap)
      DeleteObject(_bitmap);

    auto iconw = GetSystemMetrics(SM_CXICON);
    auto iconh = GetSystemMetrics(SM_CYICON);
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = iconw;
    info.bmiHeader.biHeight = iconh;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 24;
    info.bmiHeader.biCompression = BI_RGB;

    _hdcMem = CreateCompatibleDC(0);

    _output = makeImage(iconw * iconh * 3);
    _output.payload.imageValue->width = iconw;
    _output.payload.imageValue->height = iconh;
    _output.payload.imageValue->channels = 3;
    SHImage &outImage = *_output.payload.imageValue;

    _bitmap = CreateDIBSection(_hdcMem, &info, DIB_RGB_COLORS, (void **)&outImage.data, 0, 0);

    auto oldObj = SelectObject(_hdcMem, _bitmap);
    CURSORINFO ci;
    ci.cbSize = sizeof(ci);

    // reset to 0s the bitmap, cos the next call won't fill all!
    memset(outImage.data, 0x0, outImage.width * outImage.height * outImage.channels);
    if (GetCursorInfo(&ci)) {
      DrawIconEx(_hdcMem, 0, 0, ci.hCursor, 0, 0, 0, 0, DI_NORMAL);
    }
    SelectObject(_hdcMem, oldObj);
    return _output;
  }
};

extern "C" NTSYSAPI NTSTATUS NTAPI NtSetTimerResolution(ULONG DesiredResolution, BOOLEAN SetResolution, PULONG CurrentResolution);

struct SetTimerResolution {
  static SHTypesInfo inputTypes() { return CoreInfo::IntType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }

  void cleanup(SHContext *context) {
    ULONG tmp;
    NtSetTimerResolution(0, FALSE, &tmp);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    // in 100 ns, 5000 = 0.5 milliseconds
    // max 156250 , default OS seems
    ULONG res = (ULONG)input.payload.intValue;
    ULONG tmp;

    if (NtSetTimerResolution(res, TRUE, &tmp)) {
      throw ActivationError("Failed to call NtSetTimerResolution - TRUE");
    }

    return Var((int64_t)tmp);
  }
};

struct LastInput : public LastInputBase {
  SHVar activate(SHContext *context, const SHVar &input) {
    LASTINPUTINFO info;
    info.cbSize = sizeof(info);
    info.dwTime = 0;
    if (GetLastInputInfo(&info)) {
      auto tickCount = double(GetTickCount64());
      auto ticks = double(info.dwTime);
      auto diff = tickCount - ticks;
      return Var(diff / 1000.0);
    } else {
      throw ActivationError("GetLastInputInfo failed");
    }
  }
};
// TODO REFACTOR

RUNTIME_SHARD(Desktop, HasWindow);
RUNTIME_SHARD_cleanup(HasWindow);
RUNTIME_SHARD_inputTypes(HasWindow);
RUNTIME_SHARD_outputTypes(HasWindow);
RUNTIME_SHARD_parameters(HasWindow);
RUNTIME_SHARD_setParam(HasWindow);
RUNTIME_SHARD_getParam(HasWindow);
RUNTIME_SHARD_activate(HasWindow);
RUNTIME_SHARD_END(HasWindow);

RUNTIME_SHARD(Desktop, WaitWindow);
RUNTIME_SHARD_cleanup(WaitWindow);
RUNTIME_SHARD_inputTypes(WaitWindow);
RUNTIME_SHARD_outputTypes(WaitWindow);
RUNTIME_SHARD_parameters(WaitWindow);
RUNTIME_SHARD_setParam(WaitWindow);
RUNTIME_SHARD_getParam(WaitWindow);
RUNTIME_SHARD_activate(WaitWindow);
RUNTIME_SHARD_END(WaitWindow);

RUNTIME_SHARD(Desktop, PID);
RUNTIME_SHARD_inputTypes(PID);
RUNTIME_SHARD_outputTypes(PID);
RUNTIME_SHARD_activate(PID);
RUNTIME_SHARD_END(PID);

RUNTIME_SHARD(Desktop, IsForeground);
RUNTIME_SHARD_inputTypes(IsForeground);
RUNTIME_SHARD_outputTypes(IsForeground);
RUNTIME_SHARD_activate(IsForeground);
RUNTIME_SHARD_END(IsForeground);

RUNTIME_SHARD(Desktop, SetForeground);
RUNTIME_SHARD_inputTypes(SetForeground);
RUNTIME_SHARD_outputTypes(SetForeground);
RUNTIME_SHARD_activate(SetForeground);
RUNTIME_SHARD_END(SetForeground);

RUNTIME_SHARD(Desktop, NotForeground);
RUNTIME_SHARD_inputTypes(NotForeground);
RUNTIME_SHARD_outputTypes(NotForeground);
RUNTIME_SHARD_activate(NotForeground);
RUNTIME_SHARD_END(NotForeground);

RUNTIME_SHARD(Desktop, Resize);
RUNTIME_SHARD_inputTypes(Resize);
RUNTIME_SHARD_outputTypes(Resize);
RUNTIME_SHARD_parameters(Resize);
RUNTIME_SHARD_setParam(Resize);
RUNTIME_SHARD_getParam(Resize);
RUNTIME_SHARD_activate(Resize);
RUNTIME_SHARD_END(Resize);

RUNTIME_SHARD(Desktop, Move);
RUNTIME_SHARD_inputTypes(Move);
RUNTIME_SHARD_outputTypes(Move);
RUNTIME_SHARD_parameters(Move);
RUNTIME_SHARD_setParam(Move);
RUNTIME_SHARD_getParam(Move);
RUNTIME_SHARD_activate(Move);
RUNTIME_SHARD_END(Move);

RUNTIME_SHARD(Desktop, Size);
RUNTIME_SHARD_inputTypes(Size);
RUNTIME_SHARD_outputTypes(Size);
RUNTIME_SHARD_activate(Size);
RUNTIME_SHARD_END(Size);

RUNTIME_SHARD(Desktop, Bounds);
RUNTIME_SHARD_inputTypes(Bounds);
RUNTIME_SHARD_outputTypes(Bounds);
RUNTIME_SHARD_activate(Bounds);
RUNTIME_SHARD_END(Bounds);

RUNTIME_SHARD(Desktop, SetBorderless);
RUNTIME_SHARD_inputTypes(SetBorderless);
RUNTIME_SHARD_outputTypes(SetBorderless);
RUNTIME_SHARD_activate(SetBorderless);
RUNTIME_SHARD_END(SetBorderless);

RUNTIME_SHARD(Desktop, SetClickthrough);
RUNTIME_SHARD_inputTypes(SetClickthrough);
RUNTIME_SHARD_outputTypes(SetClickthrough);
RUNTIME_SHARD_activate(SetClickthrough);
RUNTIME_SHARD_END(SetClickthrough);

RUNTIME_SHARD(Desktop, UnsetClickthrough);
RUNTIME_SHARD_inputTypes(UnsetClickthrough);
RUNTIME_SHARD_outputTypes(UnsetClickthrough);
RUNTIME_SHARD_activate(UnsetClickthrough);
RUNTIME_SHARD_END(UnsetClickthrough);

RUNTIME_SHARD(Desktop, SetTopmost);
RUNTIME_SHARD_inputTypes(SetTopmost);
RUNTIME_SHARD_outputTypes(SetTopmost);
RUNTIME_SHARD_activate(SetTopmost);
RUNTIME_SHARD_END(SetTopmost);

RUNTIME_SHARD(Desktop, UnsetTopmost);
RUNTIME_SHARD_inputTypes(UnsetTopmost);
RUNTIME_SHARD_outputTypes(UnsetTopmost);
RUNTIME_SHARD_activate(UnsetTopmost);
RUNTIME_SHARD_END(UnsetTopmost);

RUNTIME_SHARD(Desktop, SetTitle);
RUNTIME_SHARD_inputTypes(SetTitle);
RUNTIME_SHARD_outputTypes(SetTitle);
RUNTIME_SHARD_parameters(SetTitle);
RUNTIME_SHARD_setParam(SetTitle);
RUNTIME_SHARD_getParam(SetTitle);
RUNTIME_SHARD_activate(SetTitle);
RUNTIME_SHARD_END(SetTitle);

RUNTIME_SHARD(Desktop, Pixel);
RUNTIME_SHARD_inputTypes(Pixel);
RUNTIME_SHARD_outputTypes(Pixel);
RUNTIME_SHARD_activate(Pixel);
RUNTIME_SHARD_cleanup(Pixel);
RUNTIME_SHARD_parameters(Pixel);
RUNTIME_SHARD_setParam(Pixel);
RUNTIME_SHARD_getParam(Pixel);
RUNTIME_SHARD_requiredVariables(Pixel);
RUNTIME_SHARD_END(Pixel);

RUNTIME_SHARD(Desktop, Pixels);
RUNTIME_SHARD_inputTypes(Pixels);
RUNTIME_SHARD_outputTypes(Pixels);
RUNTIME_SHARD_activate(Pixels);
RUNTIME_SHARD_cleanup(Pixels);
RUNTIME_SHARD_parameters(Pixels);
RUNTIME_SHARD_setParam(Pixels);
RUNTIME_SHARD_getParam(Pixels);
RUNTIME_SHARD_requiredVariables(Pixels);
RUNTIME_SHARD_END(Pixels);

RUNTIME_SHARD(Desktop, WaitKeyEvent);
RUNTIME_SHARD_inputTypes(WaitKeyEvent);
RUNTIME_SHARD_outputTypes(WaitKeyEvent);
RUNTIME_SHARD_activate(WaitKeyEvent);
RUNTIME_SHARD_cleanup(WaitKeyEvent);
RUNTIME_SHARD_help(WaitKeyEvent);
RUNTIME_SHARD_END(WaitKeyEvent);

RUNTIME_SHARD(Desktop, SendKeyEvent);
RUNTIME_SHARD_inputTypes(SendKeyEvent);
RUNTIME_SHARD_outputTypes(SendKeyEvent);
RUNTIME_SHARD_parameters(SendKeyEvent);
RUNTIME_SHARD_setParam(SendKeyEvent);
RUNTIME_SHARD_getParam(SendKeyEvent);
RUNTIME_SHARD_activate(SendKeyEvent);
RUNTIME_SHARD_cleanup(SendKeyEvent);
RUNTIME_SHARD_help(SendKeyEvent);
RUNTIME_SHARD_requiredVariables(SendKeyEvent);
RUNTIME_SHARD_END(SendKeyEvent);

RUNTIME_SHARD(Desktop, GetMousePos);
RUNTIME_SHARD_cleanup(GetMousePos);
RUNTIME_SHARD_warmup(GetMousePos);
RUNTIME_SHARD_inputTypes(GetMousePos);
RUNTIME_SHARD_outputTypes(GetMousePos);
RUNTIME_SHARD_parameters(GetMousePos);
RUNTIME_SHARD_setParam(GetMousePos);
RUNTIME_SHARD_getParam(GetMousePos);
RUNTIME_SHARD_activate(GetMousePos);
RUNTIME_SHARD_requiredVariables(GetMousePos);
RUNTIME_SHARD_END(GetMousePos);

RUNTIME_SHARD(Desktop, SetMousePos);
RUNTIME_SHARD_cleanup(SetMousePos);
RUNTIME_SHARD_warmup(SetMousePos);
RUNTIME_SHARD_inputTypes(SetMousePos);
RUNTIME_SHARD_outputTypes(SetMousePos);
RUNTIME_SHARD_parameters(SetMousePos);
RUNTIME_SHARD_setParam(SetMousePos);
RUNTIME_SHARD_getParam(SetMousePos);
RUNTIME_SHARD_activate(SetMousePos);
RUNTIME_SHARD_requiredVariables(SetMousePos);
RUNTIME_SHARD_END(SetMousePos);

RUNTIME_SHARD(Desktop, Tap);
RUNTIME_SHARD_setup(Tap);
RUNTIME_SHARD_cleanup(Tap);
RUNTIME_SHARD_warmup(Tap);
RUNTIME_SHARD_inputTypes(Tap);
RUNTIME_SHARD_outputTypes(Tap);
RUNTIME_SHARD_parameters(Tap);
RUNTIME_SHARD_setParam(Tap);
RUNTIME_SHARD_getParam(Tap);
RUNTIME_SHARD_activate(Tap);
RUNTIME_SHARD_requiredVariables(Tap);
RUNTIME_SHARD_END(Tap);

RUNTIME_SHARD(Desktop, LeftClick);
RUNTIME_SHARD_cleanup(LeftClick);
RUNTIME_SHARD_warmup(LeftClick);
RUNTIME_SHARD_inputTypes(LeftClick);
RUNTIME_SHARD_outputTypes(LeftClick);
RUNTIME_SHARD_parameters(LeftClick);
RUNTIME_SHARD_setParam(LeftClick);
RUNTIME_SHARD_getParam(LeftClick);
RUNTIME_SHARD_activate(LeftClick);
RUNTIME_SHARD_requiredVariables(LeftClick);
RUNTIME_SHARD_END(LeftClick);

RUNTIME_SHARD(Desktop, RightClick);
RUNTIME_SHARD_cleanup(RightClick);
RUNTIME_SHARD_warmup(RightClick);
RUNTIME_SHARD_inputTypes(RightClick);
RUNTIME_SHARD_outputTypes(RightClick);
RUNTIME_SHARD_parameters(RightClick);
RUNTIME_SHARD_setParam(RightClick);
RUNTIME_SHARD_getParam(RightClick);
RUNTIME_SHARD_activate(RightClick);
RUNTIME_SHARD_requiredVariables(RightClick);
RUNTIME_SHARD_END(RightClick);

RUNTIME_SHARD(Desktop, MiddleClick);
RUNTIME_SHARD_cleanup(MiddleClick);
RUNTIME_SHARD_warmup(MiddleClick);
RUNTIME_SHARD_inputTypes(MiddleClick);
RUNTIME_SHARD_outputTypes(MiddleClick);
RUNTIME_SHARD_parameters(MiddleClick);
RUNTIME_SHARD_setParam(MiddleClick);
RUNTIME_SHARD_getParam(MiddleClick);
RUNTIME_SHARD_activate(MiddleClick);
RUNTIME_SHARD_requiredVariables(MiddleClick);
RUNTIME_SHARD_END(MiddleClick);

RUNTIME_SHARD(Desktop, CursorBitmap);
RUNTIME_SHARD_inputTypes(CursorBitmap);
RUNTIME_SHARD_outputTypes(CursorBitmap);
RUNTIME_SHARD_activate(CursorBitmap);
RUNTIME_SHARD_END(CursorBitmap);

RUNTIME_SHARD(Desktop, SetTimerResolution);
RUNTIME_SHARD_inputTypes(SetTimerResolution);
RUNTIME_SHARD_outputTypes(SetTimerResolution);
RUNTIME_SHARD_activate(SetTimerResolution);
RUNTIME_SHARD_cleanup(SetTimerResolution);
RUNTIME_SHARD_END(SetTimerResolution);
}; // namespace Desktop

SHARDS_REGISTER_FN(desktop) {
  using namespace Desktop;
  REGISTER_SHARD2(Desktop, HasWindow);
  REGISTER_SHARD2(Desktop, WaitWindow);
  REGISTER_SHARD2(Desktop, PID);
  REGISTER_SHARD2(Desktop, IsForeground);
  REGISTER_SHARD2(Desktop, SetForeground);
  REGISTER_SHARD2(Desktop, NotForeground);
  REGISTER_SHARD2(Desktop, Resize);
  REGISTER_SHARD2(Desktop, Move);
  REGISTER_SHARD2(Desktop, Size);
  REGISTER_SHARD2(Desktop, Bounds);
  REGISTER_SHARD2(Desktop, SetBorderless);
  REGISTER_SHARD2(Desktop, SetClickthrough);
  REGISTER_SHARD2(Desktop, UnsetClickthrough);
  REGISTER_SHARD2(Desktop, SetTopmost);
  REGISTER_SHARD2(Desktop, UnsetTopmost);
  REGISTER_SHARD2(Desktop, SetTitle);
  REGISTER_SHARD2(Desktop, Pixel);
  REGISTER_SHARD2(Desktop, Pixels);
  REGISTER_SHARD2(Desktop, WaitKeyEvent);
  REGISTER_SHARD2(Desktop, SendKeyEvent);
  REGISTER_SHARD2(Desktop, GetMousePos);
  REGISTER_SHARD2(Desktop, SetMousePos);
  REGISTER_SHARD2(Desktop, Tap);
  REGISTER_SHARD2(Desktop, LeftClick);
  REGISTER_SHARD2(Desktop, RightClick);
  REGISTER_SHARD2(Desktop, MiddleClick);
  REGISTER_SHARD2(Desktop, CursorBitmap);
  REGISTER_SHARD2(Desktop, SetTimerResolution);
  REGISTER_SHARD("Desktop.LastInput", LastInput);
  REGISTER_SHARD("Desktop.ScrollHorizontal", ScrollHorizontal);
  REGISTER_SHARD("Desktop.ScrollVertical", ScrollVertical);
  REGISTER_SHARD("Desktop.MoveMouse", SetMouseRelativePos);
  REGISTER_SHARD("Desktop.CaptureFrame", CaptureFrame);
}
