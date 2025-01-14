#include "shards/core/runtime.hpp"
#include "gfx.hpp"
#include "window.hpp"
#include "renderer.hpp"
#include <shards/input/input_stack.hpp>
#include <shards/input/master.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/params.hpp>
#include <shards/modules/core/time.hpp>
#include <shards/modules/inputs/inputs.hpp>
#include <shards/modules/gfx/gfx.hpp>
#include <shards/input/sdl.hpp>
#include <gfx/context.hpp>
#include <gfx/platform_surface.hpp>
#include <gfx/window.hpp>

using namespace shards;
using namespace shards::input;

namespace shards {
void WindowContext::nextFrame() {}
gfx::Window &WindowContext::getWindow() { return *window.get(); }
} // namespace shards

namespace gfx {

struct MainWindowQuitException : public std::runtime_error {
  MainWindowQuitException() : std::runtime_error("MainWindow Quit Requested") {}
};

struct InlineInputContext : public IInputContext {
  InputMaster *master{};

  float time{};
  float deltaTime{};

  InputStack inputStack;

  InputStack &getInputStack() override { return inputStack; }

  shards::input::InputMaster &getMaster() const override { return *master; }
  const std::weak_ptr<IInputHandler> &getHandler() const override {
    static std::weak_ptr<IInputHandler> none;
    return none;
  }

  void postMessage(const Message &message) override { master->postMessage(message); }
  const InputState &getState() const override { return master->getState(); }
  std::vector<ConsumableEvent> &getEvents() override { return master->getEvents(); }

  float getTime() const override { return time; }
  float getDeltaTime() const override { return deltaTime; }

  bool requestFocus() override { return true; }
  bool canReceiveInput() const override { return true; }
};

struct MainWindow final {
  PARAM_VAR(_title, "Title", "The title of the window to create.", {CoreInfo::NoneType, CoreInfo::StringType});
  PARAM_VAR(_width, "Width", "The width of the window to create. In pixels and DPI aware.", {CoreInfo::IntType});
  PARAM_VAR(_height, "Height", "The height of the window to create. In pixels and DPI aware.", {CoreInfo::IntType});
  PARAM(ShardsVar, _contents, "Contents", "The main input loop of this window.", {CoreInfo::ShardsOrNone});
  PARAM_VAR(_detachRenderer, "DetachRenderer",
            "When enabled, no default graphics renderer will be available in the contents wire.", {CoreInfo::BoolType});
  PARAM_VAR(
      _handleCloseEvent, "HandleCloseEvent",
      "When set to false; the close event will not be handled and abort the wire the window is running in. True by default.",
      {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_VAR(_useDisplayScaling, "UseDisplayScaling",
            "When enabled, the window size will be scaled using the OS's provided scaling value. True by default.",
            {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_title), PARAM_IMPL_FOR(_width), PARAM_IMPL_FOR(_height), PARAM_IMPL_FOR(_contents),
             PARAM_IMPL_FOR(_detachRenderer), PARAM_IMPL_FOR(_handleCloseEvent), PARAM_IMPL_FOR(_useDisplayScaling));

  static inline Type OutputType = Type(WindowContext::Type);

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return OutputType; }

  MainWindow() {
    _width = Var(1280);
    _height = Var(720);
    _title = Var("Shards Window");
    _detachRenderer = Var(false);
    _handleCloseEvent = Var(true);
    _useDisplayScaling = Var(true);
  }

  Window _window;

  std::optional<WindowContext> _windowContext;
  SHVar *_windowContextVar{};

  std::optional<InlineInputContext> _inlineInputContext;
  SHVar *_inputContextVar{};

  ExposedInfo _innerExposedVariables;
  ExposedInfo _exposedVariables;

  std::optional<ShardsRenderer> _renderer;

  shards::Time::DeltaTimer _deltaTimer;

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
    windowOptions.title = SHSTRVIEW(*_title);
    _windowContext->window = std::make_shared<Window>();
    _windowContext->window->init(windowOptions);
    _windowContext->windowMesh = shContext->main->mesh;

    // Adjust window size so they're specified in virtual points
    float uiScale = _windowContext->window->getUIScale();
    float scaling = 1.0f;
    if (*_useDisplayScaling) {
      if (Window::isWindowSizeInPixels()) {
        scaling = uiScale;
      }
    } else {
      if (!Window::isWindowSizeInPixels()) {
        scaling = 1.0f / uiScale;
      }
    }
    _windowContext->window->resize((int2)(float2(int2((int)*_width, (int)*_height)) * scaling));

#if SH_APPLE
    gfx::MetalViewContainer &metalView = _windowContext->window->metalView.value();
    entt::locator<std::reference_wrapper<gfx::MetalViewContainer>>::emplace(metalView);
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

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);

    SPDLOG_DEBUG("cleanup window 0");
    if (_renderer || _windowContext) {
      SPDLOG_DEBUG("cleanup window 1");
      callOnMeshThread(context, [&] {
        SPDLOG_DEBUG("cleanup window 2");
        if (_renderer) {
          SPDLOG_DEBUG("cleanup renderer");
          _renderer->cleanup(context);
          _renderer.reset();
        }

        if (_windowContext) {
          SPDLOG_DEBUG("cleanup windowContext");
          if (_windowContext->window) {
            SHLOG_DEBUG("Destroying window");
            _windowContext->window->cleanup();
          }

          _windowContext.reset();
        }
      });
    }

    if (_windowContextVar) {
      if (_windowContextVar->refcount > 1) {
        SHLOG_ERROR("MainWindow: Found {} dangling reference(s) to {}", _windowContextVar->refcount - 1,
                    WindowContext::VariableName);
      }
      releaseVariable(_windowContextVar);
      _windowContextVar = nullptr;
    }

    if (_inputContextVar) {
      releaseVariable(_inputContextVar);
      _inputContextVar = nullptr;
    }
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    if (!_windowContext->window) {
      callOnMeshThread(shContext, [&]() { initWindow(shContext); });
    }

    double deltaTime = _deltaTimer.update();
    _windowContext->time += deltaTime;
    _windowContext->deltaTime = (float)deltaTime;

    auto &window = _windowContext->window;

    bool shouldRun = true;
    if (_renderer) {
      if (!_renderer->begin(shContext, _windowContext.value()))
        shouldRun = false;
    }

    if (shouldRun) {
      // Poll & distribute input events
      callOnMeshThread(shContext, [&]() {
        window->update();
        _windowContext->inputMaster.update(*window.get());
      });

      for (auto &event : _windowContext->inputMaster.getEvents()) {
        if (const RequestCloseEvent *evt = std::get_if<RequestCloseEvent>(&event.event)) {
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

  static SHOptionalString help() {
    return SHCCSTR("This shard outputs the dimensions of the window specified in the Window parameter.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the dimensions of the window as an int2. The first element represents the width and the second "
                   "represents the height.");
  }

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

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return toVar(_requiredWindowContext->window->getSize()); }
};

struct ResizeWindow {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  static SHOptionalString help() {
    return SHCCSTR("This shard resizes the window specified in the Window parameter to the dimensions specified in the input.");
  }
  static SHOptionalString inputHelp() {
    return SHCCSTR(
        "The size to resize the window to. The first element represents the width and the second represents the height.");
  }
  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  PARAM_PARAMVAR(_window, "Window", "The window to resize", {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
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

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    // This will hang on windows with thread fiber, so send a message instead
    auto windowMesh = _requiredWindowContext->windowMesh.lock();
    if (!windowMesh)
      return input;
    if (windowMesh == shContext->main->mesh.lock()) {
      callOnMeshThread(shContext, [&]() { _requiredWindowContext->window->resize((int2)toInt2(input)); });
    } else {
      _requiredWindowContext->inputMaster.postMessage(ResizeWindowMessage{(int2)toInt2(input)});
    }
    return input;
  }
};

struct WindowPosition {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  static SHOptionalString help() {
    return SHCCSTR("This shard outputs the position of the window specified in the Window parameter.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the position of the window as an int2. The first element represents the x-coordinate and the second "
                   "represents the y-coordinate.");
  }

  PARAM_PARAMVAR(_window, "Window", "The window to get the position of.",
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
  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return toVar(_requiredWindowContext->window->getPosition()); }
};

struct MoveWindow {
  static SHTypesInfo inputTypes() { return CoreInfo::Int2Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::Int2Type; }

  static SHOptionalString help() {
    return SHCCSTR("This shard moves the window specified in the Window parameter to the position specified in the input.");
  }
  static SHOptionalString inputHelp() {
    return SHCCSTR("The position to move the window to. The first element represents the x-coordinate and the second represents "
                   "the y-coordinate.");
  }
  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  PARAM_PARAMVAR(_window, "Window", "The window to move.", {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
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

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    callOnMeshThread(shContext, [&]() { _requiredWindowContext->window->move((int2)toInt2(input)); });
    return input;
  }
};

struct OsUiScaleFactor {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }
  static SHOptionalString help() { return SHCCSTR("Outputs the scaling factor for UI as determined by the operating system."); }

  PARAM_PARAMVAR(_window, "Window", "The window to get the UI scaling factor of.",
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

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &window = _requiredWindowContext->window;
    return Var(window->getUIScale());
  }
};

struct WindowInsets {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::Float4Type; }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The window inset values as a float4."); }
  static SHOptionalString help() {
    return SHCCSTR("Retrieves the window inset values when rendering on mobile devices and screens with a keep-out area.");
  }

  RequiredWindowContext _requiredWindowContext;

  PARAM_PARAMVAR(_window, "Window", "The window to get the insets of.",
                 {CoreInfo::NoneType, Type::VariableOf(WindowContext::Type)});
  PARAM_IMPL(PARAM_IMPL_FOR(_window));

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredWindowContext.warmup(context, &_window);
  }
  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredWindowContext.compose(data, _requiredVariables, &_window);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    // auto &window = _requiredWindowContext->window;
    // Actually is static for now, but keep this here might it be of use in the future
    float scale = 1.0f;
#if SH_APPLE
    // call on mesh thread to avoid race condition
    callOnMeshThread(shContext, [&]() { scale *= _requiredWindowContext->window->getInputScale().x; });
#endif
    return toVar(Window::viewInset * scale);
  }
};

struct WindowFocused {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }

  static SHOptionalString help() {
    return SHCCSTR("This shard outputs whether the window specified in the Window parameter currently has focus.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs true if the window currently has focus, false otherwise."); }

  PARAM_PARAMVAR(_window, "Window", "The window to check focus state of.",
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

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredWindowContext.cleanup();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return Var(_requiredWindowContext->window->isFocused()); }
};

void registerMainWindowShards() {
  REGISTER_SHARD("GFX.MainWindow", MainWindow);
  REGISTER_SHARD("GFX.WindowSize", WindowSize);
  REGISTER_SHARD("GFX.ResizeWindow", ResizeWindow);
  REGISTER_SHARD("GFX.WindowPosition", WindowPosition);
  REGISTER_SHARD("GFX.UIScaleFactor", OsUiScaleFactor);
  REGISTER_SHARD("GFX.MoveWindow", MoveWindow);
  REGISTER_SHARD("GFX.WindowInsets", WindowInsets);
  REGISTER_SHARD("GFX.WindowFocused", WindowFocused);
}

} // namespace gfx
