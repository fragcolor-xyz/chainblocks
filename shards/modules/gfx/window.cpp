#include "gfx.hpp"
#include "gfx/platform_surface.hpp"
#include "input/input_stack.hpp"
#include "modules/gfx/gfx.hpp"
#include "window.hpp"
#include "renderer.hpp"
#include <modules/inputs/inputs.hpp>
#include <shards/input/master.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/params.hpp>
#include <SDL_events.h>
#include <SDL_keyboard.h>
#include <SDL_keycode.h>
#include <SDL_scancode.h>
#include <gfx/context.hpp>
#include <gfx/window.hpp>

using namespace shards;
using namespace shards::input;

namespace shards {
void WindowContext::nextFrame() {}
gfx::Window &WindowContext::getWindow() { return *window.get(); }
SDL_Window *WindowContext::getSdlWindow() { return getWindow().window; }
} // namespace shards

namespace gfx {

struct MainWindowQuitException : public std::runtime_error {
  MainWindowQuitException() : std::runtime_error("MainWindow Quit Requested") {}
};

struct InlineInputContext : public IInputContext {
  InputMaster *master{};
  ConsumeFlags dummyConsumeFlags{};

  float time{};
  float deltaTime{};

  InputStack inputStack;

  virtual InputStack &getInputStack() override { return inputStack; }

  virtual shards::input::InputMaster *getMaster() const override { return master; }

  virtual void postMessage(const Message &message) override { master->postMessage(message); }
  virtual const InputState &getState() const override { return master->getState(); }
  virtual const std::vector<Event> &getEvents() const override { return master->getEvents(); }

  // Writable, controls how events are consumed
  virtual ConsumeFlags &getConsumeFlags() override { return dummyConsumeFlags; }

  virtual float getTime() const override { return time; }
  virtual float getDeltaTime() const override { return deltaTime; }
};

struct MainWindow final {
  PARAM_VAR(_title, "Title", "The title of the window to create.", {CoreInfo::StringType});
  PARAM_VAR(_width, "Width", "The width of the window to create. In pixels and DPI aware.", {CoreInfo::IntType});
  PARAM_VAR(_height, "Height", "The height of the window to create. In pixels and DPI aware.", {CoreInfo::IntType});
  PARAM(ShardsVar, _contents, "Contents", "The main input loop of this window.", {CoreInfo::ShardsOrNone});
  PARAM_VAR(_detachRenderer, "DetachRenderer",
            "When enabled, no default graphics renderer will be available in the contents wire.", {CoreInfo::BoolType});
  PARAM_VAR(
      _handleCloseEvent, "HandleCloseEvent",
      "When set to false; the close event will not be handled and abort the wire the window is running in. True by default.",
      {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_title), PARAM_IMPL_FOR(_width), PARAM_IMPL_FOR(_height), PARAM_IMPL_FOR(_contents),
             PARAM_IMPL_FOR(_detachRenderer), PARAM_IMPL_FOR(_handleCloseEvent));

  static inline Type OutputType = Type(WindowContext::Type);

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return OutputType; }

  MainWindow() {
    _width = Var(1280);
    _height = Var(1280);
    _title = Var("Shards Window");
    _detachRenderer = Var(false);
    _handleCloseEvent = Var(true);
  }

  Window _window;

  std::optional<WindowContext> _windowContext;
  SHVar *_windowContextVar{};

  std::optional<InlineInputContext> _inlineInputContext;
  SHVar *_inputContextVar{};

  ExposedInfo _innerExposedVariables;
  ExposedInfo _exposedVariables;

  std::optional<ShardsRenderer> _renderer;

  SHExposedTypesInfo exposedVariables() { return SHExposedTypesInfo(_exposedVariables); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _exposedVariables.clear();

    // Make sure that windows are UNIQUE
    for (uint32_t i = 0; i < data.shared.len; i++) {
      if (strcmp(data.shared.elements[i].name, WindowContext::VariableName) == 0) {
        throw SHException("GFX.MainWindow must be unique, found another use!");
      }
    }

    _innerExposedVariables = ExposedInfo(data.shared);
    _innerExposedVariables.push_back(WindowContext::VariableInfo);

    _windowContext.emplace();

    if (_contents) {
      if (!(bool)*_detachRenderer) {
        _renderer.emplace();
        _renderer->compose(data);
        _renderer->getExposedContextVariables(_innerExposedVariables);
      } else {
        _renderer.reset();
      }

      _inlineInputContext.emplace();
      _inlineInputContext->master = &_windowContext->inputMaster;
      _innerExposedVariables.push_back(RequiredInputContext::getExposedTypeInfo());

      SHInstanceData innerData = data;
      innerData.shared = SHExposedTypesInfo(_innerExposedVariables);
      _contents.compose(innerData);
    }

    mergeIntoExposedInfo(_exposedVariables, _contents.composeResult().exposedInfo);

    // Merge required, but without the context variables
    for (auto &required : _contents.composeResult().requiredInfo) {
      std::string_view varName(required.name);
      if (varName == WindowContext::VariableName)
        continue;
      if (_inlineInputContext && varName == IInputContext::VariableName)
        continue;
      if (_renderer && (varName == GraphicsContext::VariableName || varName == GraphicsRendererContext::VariableName))
        continue;
      _requiredVariables.push_back(required);
    }

    return OutputType;
  }

  void initWindow(SHContext *shContext) {
    SHLOG_DEBUG("Creating window");

    WindowCreationOptions windowOptions = {};
    windowOptions.width = (int)*_width;
    windowOptions.height = (int)*_height;
    windowOptions.title = (const char *)*_title;
    _windowContext->window = std::make_shared<Window>();
    _windowContext->window->init(windowOptions);

    // Adjust window size so they're specified in virtual points
    float scaling = _windowContext->window->getUIScale();
    _windowContext->window->resize((int2)(float2(int2((int)*_width, (int)*_height)) * scaling));

#if GFX_APPLE
    gfx::MetalViewContainer &metalView = _windowContext->window->metalView.value();
    auto &dispatcher = shContext->main->dispatcher;
    dispatcher.trigger(std::ref(metalView));
#endif
  }

  void warmup(SHContext *context) {
    _windowContextVar = referenceVariable(context, WindowContext::VariableName);
    assignVariableValue(*_windowContextVar, Var::Object(&_windowContext.value(), WindowContext::Type));

    if (_inlineInputContext) {
      _inputContextVar = referenceVariable(context, IInputContext::VariableName);
      assignVariableValue(*_inputContextVar, Var::Object(&_inlineInputContext.value(), IInputContext::Type));
    }

    if (_renderer) {
      _renderer->warmup(context);
    }

    PARAM_WARMUP(context);
  }

  void cleanup() {
    PARAM_CLEANUP();

    if (_renderer) {
      _renderer->cleanup();
      _renderer.reset();
    }

    if (_windowContext) {
      if (_windowContext->window) {
        SHLOG_DEBUG("Destroying window");
        _windowContext->window->cleanup();
      }

      _windowContext.reset();
    }

    if (_windowContextVar) {
      if (_windowContextVar->refcount > 1) {
        SHLOG_ERROR("MainWindow: Found {} dangling reference(s) to {}", _windowContextVar->refcount - 1,
                    WindowContext::VariableName);
      }
      releaseVariable(_windowContextVar);
    }

    releaseVariable(_inputContextVar);
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    if (!_windowContext->window) {
      initWindow(shContext);
    }

    auto &window = _windowContext->window;

    bool shouldRun = true;
    if (_renderer) {
      if (!_renderer->begin(shContext, _windowContext.value()))
        shouldRun = false;
    }

    if (shouldRun) {
      // Poll & distribute input events
      _windowContext->inputMaster.update(*window.get());

      for (auto &event : _windowContext->inputMaster.getEvents()) {
        if (const RequestCloseEvent *evt = std::get_if<RequestCloseEvent>(&event)) {
          bool handleClose = _handleCloseEvent->isNone() || (bool)*_handleCloseEvent;
          if (handleClose) {
            throw MainWindowQuitException();
          }
        }
      }

      if (_contents) {
        _inlineInputContext->time = _windowContext->time;
        _inlineInputContext->deltaTime = _windowContext->deltaTime;

        // Push root input region
        auto &inputStack = _inlineInputContext->inputStack;
        inputStack.reset();
        inputStack.push(input::InputStack::Item{
            .windowMapping = input::WindowSubRegion::fromEntireWindow(*window.get()),
        });

        SHVar _shardsOutput{};
        _contents.activate(shContext, input, _shardsOutput);

        inputStack.pop();

        if (_renderer) {
          _renderer->end();
        }
      }
    }

    return *_windowContextVar;
  }
};

struct WindowSize {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  PARAM_PARAMVAR(_window, "Window", "The window to get the size of.",
                 {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
  PARAM_IMPL(PARAM_IMPL_FOR(_window));

  RequiredWindowContext _requiredWindowContext;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredWindowContext.compose(data, _requiredVariables, &_window);
    return outputTypes().elements[0];
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredWindowContext.warmup(context, &_window);
  }

  void cleanup() {
    PARAM_CLEANUP();
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return toVar(_requiredWindowContext->window->getSize()); }
};

struct ResizeWindow {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  PARAM_PARAMVAR(_window, "Window", "The window to get the size of.",
                 {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
  PARAM_IMPL(PARAM_IMPL_FOR(_window));

  RequiredWindowContext _requiredWindowContext;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredWindowContext.compose(data, _requiredVariables, &_window);
    return data.inputType;
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredWindowContext.warmup(context, &_window);
  }

  void cleanup() {
    PARAM_CLEANUP();
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    _requiredWindowContext->window->resize(toInt2(input));
    return input;
  }
};
struct WindowPosition {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  PARAM_PARAMVAR(_window, "Window", "The window to get the size of.",
                 {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
  PARAM_IMPL(PARAM_IMPL_FOR(_window));

  RequiredWindowContext _requiredWindowContext;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredWindowContext.compose(data, _requiredVariables, &_window);
    return outputTypes().elements[0];
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredWindowContext.warmup(context, &_window);
  }
  void cleanup() {
    PARAM_CLEANUP();
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return toVar(_requiredWindowContext->window->getPosition()); }
};

struct MoveWindow {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  PARAM_PARAMVAR(_window, "Window", "The window to get the size of.",
                 {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
  PARAM_IMPL(PARAM_IMPL_FOR(_window));

  RequiredWindowContext _requiredWindowContext;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredWindowContext.compose(data, _requiredVariables, &_window);
    return data.inputType;
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredWindowContext.warmup(context, &_window);
  }

  void cleanup() {
    PARAM_CLEANUP();
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    _requiredWindowContext->window->move(toInt2(input));
    return input;
  }
};

void registerMainWindowShards() {
  REGISTER_SHARD("GFX.MainWindow", MainWindow);
  REGISTER_SHARD("GFX.WindowSize", WindowSize);
  REGISTER_SHARD("GFX.ResizeWindow", ResizeWindow);
  REGISTER_SHARD("GFX.WindowPosition", WindowPosition);
  REGISTER_SHARD("GFX.MoveWindow", MoveWindow);
}

} // namespace gfx
