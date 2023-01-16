#include "vui.hpp"
#include "../gfx.hpp"
#include "../inputs.hpp"
#include "gfx/egui/egui_types.hpp"
#include "gfx/linalg.hpp"
#include "linalg_shim.hpp"
#include "object_var_util.hpp"
#include "params.hpp"
#include "../egui/context.hpp"
#include <gfx/renderer.hpp>
#include <input/input.hpp>

namespace shards::VUI {

#define VUI_PANEL_SHARD_NAME "VUI.Panel"

struct VUIPanelShard;
struct VUIContextShard {
  VUIContext _vuiContext{};
  SHVar *_vuiContextVar{};

  Inputs::RequiredInputContext _inputContext;
  gfx::RequiredGraphicsContext _graphicsContext;

  ExposedInfo _exposedVariables;

  std::vector<VUIPanelShard *> _panels;

  input::InputBuffer _inputBuffer;

  PARAM_PARAMVAR(_queue, "Queue", "The draw queue to insert draw commands into.", {Type::VariableOf(gfx::Types::DrawQueue)});
  PARAM_PARAMVAR(_view, "View", "The view that is being used to render.", {Type::VariableOf(gfx::Types::View)});
  PARAM(ShardsVar, _contents, "Contents", "The list of UI panels to render.", {CoreInfo::ShardsOrNone});
  PARAM_VAR(_scale, "Scale", "The scale of world units to UI units.", {CoreInfo::FloatType});
  PARAM_IMPL(VUIContextShard, PARAM_IMPL_FOR(_queue), PARAM_IMPL_FOR(_view), PARAM_IMPL_FOR(_contents), PARAM_IMPL_FOR(_scale));

  VUIContextShard() { _scale = Var(1.0f / 1000.0f); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString help() {
    return SHCCSTR("Creates a context for virtual UI panels to make sure input is correctly handled between them");
  }

  SHExposedTypesInfo requiredVariables() {
    static auto e =
        exposedTypesOf(decltype(_inputContext)::getExposedTypeInfo(), decltype(_graphicsContext)::getExposedTypeInfo());
    return e;
  }

  void warmup(SHContext *context);

  void cleanup() {
    PARAM_CLEANUP()

    _contents.cleanup();
    _graphicsContext.cleanup();
    _inputContext.cleanup();

    if (_vuiContextVar) {
      if (_vuiContextVar->refcount > 1) {
        SHLOG_ERROR("VUI.Context: Found {} dangling reference(s) to {}", _vuiContextVar->refcount - 1, VUIContext::VariableName);
      }
      releaseVariable(_vuiContextVar);
    }
  }

  SHTypeInfo compose(SHInstanceData &data) {
    _exposedVariables = ExposedInfo(data.shared);
    _exposedVariables.push_back(VUIContext::VariableInfo);
    data.shared = SHExposedTypesInfo(_exposedVariables);

    _contents.compose(data);

    if (_queue->valueType == SHType::None)
      throw ComposeError("Queue is required");

    if (_view->valueType == SHType::None)
      throw ComposeError("View is required");

    return shards::CoreInfo::NoneType;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &viewStack = _graphicsContext->renderer->getViewStack();
    auto viewStackTop = viewStack.getOutput();

    auto &queue = *varAsObjectChecked<gfx::SHDrawQueue>(_queue.get(), gfx::Types::DrawQueue);
    auto &view = *varAsObjectChecked<gfx::SHView>(_view.get(), gfx::Types::View);

    // TODO: Move to input context
    _inputBuffer.clear();
    for (auto &event : _inputContext->events)
      _inputBuffer.push_back(event);

    _vuiContext.activationContext = shContext;
    _vuiContext.context.virtualPointScale = _scale.payload.floatValue;
    withObjectVariable(*_vuiContextVar, &_vuiContext, VUIContext::Type, [&]() {
      gfx::float2 inputToViewScale{1.0f};
      gfx::SizedView sizedView(view.view, gfx::float2(viewStackTop.viewport.getSize()));
      _vuiContext.context.prepareInputs(_inputBuffer, inputToViewScale, sizedView);
      _vuiContext.context.evaluate(queue.queue, _graphicsContext->time, _graphicsContext->deltaTime);
    });
    _vuiContext.activationContext = nullptr;

    SHVar output{};
    return output;
  }
};

struct VUIPanelShard {
  PARAM_PARAMVAR(_transform, "Transform", "The world transform of this panel.",
                 {CoreInfo::Float4x4Type, Type::VariableOf(CoreInfo::Float4x4Type)});
  PARAM_PARAMVAR(_size, "Size", "The size of the panel.", {CoreInfo::Float2Type, Type::VariableOf(CoreInfo::Float2Type)});
  PARAM(ShardsVar, _contents, "Contents", "The panel UI contents.", {CoreInfo::ShardsOrNone});
  PARAM_IMPL(VUIPanelShard, PARAM_IMPL_FOR(_transform), PARAM_IMPL_FOR(_size), PARAM_IMPL_FOR(_contents));

  RequiredVUIContext _context;
  Shard *_uiShard{};
  EguiContext _eguiContext;
  ExposedInfo _exposedTypes;
  std::shared_ptr<Panel> _panel;

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString help() { return SHCCSTR("Defines a virtual UI panel"); }

  SHExposedTypesInfo requiredVariables() {
    static auto e = exposedTypesOf(RequiredVUIContext::getExposedTypeInfo());
    return e;
  }

  void ensureEguiContext() {
    if (!_eguiContext)
      _eguiContext = egui_createContext();
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _context.warmup(context);

    ensureEguiContext();
    egui_warmup(_eguiContext, context);

    _panel = std::make_shared<Panel>(*this);
    _context->context.panels.emplace_back(_panel);
  }

  void cleanup() {
    PARAM_CLEANUP();
    _context.cleanup();

    if (_eguiContext) {
      egui_cleanup(_eguiContext);
      egui_destroyContext(_eguiContext);
      _eguiContext = nullptr;
    }

    _panel.reset();
  }

  SHTypeInfo compose(SHInstanceData &data) {
    ensureEguiContext();

    SHExposedTypesInfo eguiExposedTypes{};
    egui_getExposedTypeInfo(_eguiContext, eguiExposedTypes);

    _exposedTypes = ExposedInfo(data.shared);
    mergeIntoExposedInfo(_exposedTypes, eguiExposedTypes);

    // Compose contents
    data.shared = SHExposedTypesInfo(_exposedTypes);
    _contents.compose(data);

    return shards::CoreInfo::NoneType;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    throw ActivationError("Invalid activation, VUIPanel can not be used directly");
  }

  // This evaluates the egui contents for this panel
  virtual const egui::FullOutput &render(const egui::Input &inputs) {
    SHVar output{};
    const char *error = egui_activate(_eguiContext, inputs, _contents.shards(), _context->activationContext, SHVar{}, output);
    if (error)
      throw ActivationError(fmt::format("egui activation error: {}", error));

    const egui::FullOutput &eguiOutput = *egui_getOutput(_eguiContext);
    return eguiOutput;
  }

  vui::PanelGeometry getGeometry() const {
    gfx::float4x4 transform = toFloat4x4(_transform.get());
    gfx::float2 alignment{0.5f};

    vui::PanelGeometry result;
    result.anchor = gfx::extractTranslation(transform);
    result.up = transform.y.xyz();
    result.right = transform.x.xyz();
    result.size = toFloat2(_size.get());
    result.center =
        result.anchor + result.right * (0.5f - alignment.x) * result.size.x + result.up * (0.5f - alignment.y) * result.size.y;
    return result;
  }
};

const egui::FullOutput &Panel::render(const egui::Input &inputs) { return panelShard.render(inputs); }
vui::PanelGeometry Panel::getGeometry() const { return panelShard.getGeometry(); }

void VUIContextShard::warmup(SHContext *context) {
  _vuiContextVar = referenceVariable(context, VUIContext::VariableName);

  _inputContext.warmup(context);
  _graphicsContext.warmup(context);

  withObjectVariable(*_vuiContextVar, &_vuiContext, VUIContext::Type, [&]() { PARAM_WARMUP(context); });

  // Collect VUI.Panel shards similar to UI dock does
  // _panels.clear();
  // auto contentShards = _contents.shards();
  // for (size_t i = 0; i < contentShards.len; i++) {
  //   ShardPtr shard = contentShards.elements[i];

  //   if (std::string(VUI_PANEL_SHARD_NAME) == shard->name(shard)) {
  //     using ShardWrapper = shards::ShardWrapper<VUIPanelShard>;
  //     VUIPanelShard &vuiPanel = reinterpret_cast<ShardWrapper *>(shard)->shard;
  //     _panels.emplace_back(&vuiPanel);
  //   }
  // }
}

void registerShards() {
  REGISTER_SHARD("VUI.Context", VUIContextShard);
  REGISTER_SHARD(VUI_PANEL_SHARD_NAME, VUIPanelShard);
}
} // namespace shards::VUI
