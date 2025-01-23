#include "rust_interop.hpp"
#include <shards/modules/gfx/gfx.hpp>
#include <shards/modules/gfx/window.hpp>
#include <shards/modules/inputs/inputs.hpp>
#include <gfx/renderer.hpp>
#include <shards/core/foundation.hpp>
#include <spdlog/spdlog.h>
#include <shards/modules/gfx/shards_types.hpp>

using namespace shards::input;
using namespace gfx;
using namespace shards;

SHTypeInfo *gfx_getGraphicsContextType() {
  static SHTypeInfo type = gfx::GraphicsContext::Type;
  return &type;
}
const char *gfx_getGraphicsContextVarName() { return gfx::GraphicsContext::VariableName; }

SHTypeInfo *gfx_getWindowContextType() {
  static SHTypeInfo type = WindowContext::Type;
  return &type;
}
const char *gfx_getWindowContextVarName() { return WindowContext::VariableName; }

SHTypeInfo *gfx_getInputContextType() {
  static SHTypeInfo type = IInputContext::Type;
  return &type;
}
const char *gfx_getInputContextVarName() { return IInputContext::VariableName; }

SHTypeInfo *gfx_getQueueType() {
  static SHTypeInfo type = ShardsTypes::DrawQueue;
  return &type;
}

gfx::Context *gfx_GraphicsContext_getContext(const SHVar &graphicsContext) {
  GraphicsContext &globals = varAsObjectChecked<GraphicsContext>(graphicsContext, GraphicsContext::Type);
  return globals.context.get();
}
gfx::Renderer *gfx_GraphicsContext_getRenderer(const SHVar &graphicsContext) {
  GraphicsContext &globals = varAsObjectChecked<GraphicsContext>(graphicsContext, GraphicsContext::Type);
  return globals.renderer.get();
}

DrawQueuePtr *gfx_getDrawQueueFromVar(const SHVar &var) {
  SHDrawQueue &shDrawQueue = varAsObjectChecked<SHDrawQueue>(var, ShardsTypes::DrawQueue);
  return &shDrawQueue.queue;
}

const egui::Input *gfx_getEguiWindowInputs(gfx::EguiInputTranslator *translator, const SHVar &inputContextVar,
                                           float scalingFactor) {
  IInputContext &inputContext = varAsObjectChecked<IInputContext>(inputContextVar, IInputContext::Type);

  static std::vector<ConsumableEvent> noEvents{};
  const std::vector<ConsumableEvent> *eventsPtr = &noEvents;
  int4 mappedWindowRegion{};

  InputRegion region = inputContext.getState().region;
  region.uiScalingFactor *= scalingFactor;

  // Get events based on input stack
  auto &inputStack = inputContext.getInputStack();
  InputStack::Item inputStackOutput = inputStack.getTop();
  if (inputStackOutput.windowMapping) {
    auto &windowMapping = inputStackOutput.windowMapping.value();
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, WindowSubRegion>) {
            mappedWindowRegion = arg.region;
            eventsPtr = &inputContext.getEvents();
          }
        },
        windowMapping);
  }

  return translator->translateFromInputEvents(EguiInputTranslatorArgs{
      .events = *eventsPtr,
      .time = inputContext.getTime(),
      .deltaTime = inputContext.getDeltaTime(),
      .region = region,
      .canReceiveInput = inputContext.canReceiveInput(),
      .mappedWindowRegion = mappedWindowRegion,
  });
}

void gfx_applyEguiIOOutput(gfx::EguiInputTranslator *translator, const egui::IOOutput &output, const SHVar &inputContextVar) {
  IInputContext &inputContext = varAsObjectChecked<IInputContext>(inputContextVar, IInputContext::Type);

  translator->applyOutput(output);
  for (auto &msg : translator->getOutputMessages()) {
    inputContext.postMessage(msg);
  }

  auto consume = inputContext.getEventConsumer();
  for (auto &evt : inputContext.getEvents()) {
    if (output.wantsPointerInput) {
      if (isPointerEvent(evt.event)) {
        consume(evt);
      }
    }
    if (output.wantsKeyboardInput) {
      if (isKeyEvent(evt.event)) {
        consume(evt);
      }
    }
  }

  // Request focus during drag operations
  if (output.wantsPointerInput && inputContext.getState().mouseButtonState != 0) {
    inputContext.requestFocus();
  }
}
