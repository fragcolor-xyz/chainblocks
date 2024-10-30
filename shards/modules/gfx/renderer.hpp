#ifndef BED776D7_7318_4FA7_92C1_CC63BC286F1A
#define BED776D7_7318_4FA7_92C1_CC63BC286F1A

#include "shards/core/runtime.hpp"
#include "gfx.hpp"
#include "shards/shards.h"
#include "window.hpp"
#include <gfx/loop.hpp>
#include <gfx/renderer.hpp>

namespace gfx {

inline void endFrame(GraphicsContext &ctx) {
  if (!ctx.frameInProgress)
    return;

  ctx.frameInProgress = false;
  ctx.renderer->endFrame();
  ctx.context->endFrame();
}

// This wraps the graphics context variables and renderer instance
//  it's used either by the main window or by the renderer shard to provide
//  a graphics context for all the GFX.* shards
struct ShardsRenderer {
  GraphicsContext _graphicsContext;
  SHVar *_graphicsContextVar{};

  GraphicsRendererContext _graphicsRendererContext;
  SHVar *_graphicsRendererContextVar{};

  gfx::Loop _loop;

  bool _ignoreCompilationErrors{};

  void compose(SHInstanceData &data) {
#if SH_CORO_NEED_STACK_MEM
    // Require extra stack space for the wire containing the renderer
    data.wire->stackSize = std::max<size_t>(data.wire->stackSize, 4 * 1024 * 1024);
#endif
  }

  void getExposedContextVariables(shards::ExposedInfo &outExposedInfo) {
    outExposedInfo.push_back(GraphicsContext::VariableInfo);
    outExposedInfo.push_back(GraphicsRendererContext::VariableInfo);
  }

  void initRenderer(const std::shared_ptr<Window> &window) {
    _graphicsContext.window = window;

    ContextCreationOptions contextOptions = {};
    _graphicsContext.context = std::make_shared<Context>();
    _graphicsContext.context->init(*_graphicsContext.window.get(), contextOptions);

    _graphicsContext.renderer = std::make_shared<Renderer>(*_graphicsContext.context.get());
    _graphicsContext.renderer->setIgnoreCompilationErrors(_ignoreCompilationErrors);
    _graphicsRendererContext.renderer = _graphicsContext.renderer.get();
  }

  void warmup(SHContext *context) {
    _graphicsContextVar = shards::referenceVariable(context, GraphicsContext::VariableName);
    assignVariableValue(*_graphicsContextVar, shards::Var::Object(&_graphicsContext, GraphicsContext::Type));

    _graphicsRendererContextVar = shards::referenceVariable(context, GraphicsRendererContext::VariableName);
    assignVariableValue(*_graphicsRendererContextVar,
                        shards::Var::Object(&_graphicsRendererContext, GraphicsRendererContext::Type));
  }

  void cleanup(SHContext* context) {
    if (_graphicsContextVar) {
      if (_graphicsContextVar->refcount > 1) {
        SHLOG_ERROR("MainWindow: Found {} dangling reference(s) to {}", _graphicsContextVar->refcount - 1,
                    GraphicsContext::VariableName);
      }
      shards::releaseVariable(_graphicsContextVar);
      _graphicsContextVar = nullptr;
    }

    if (_graphicsRendererContextVar) {
      if (_graphicsRendererContextVar->refcount > 1) {
        SHLOG_ERROR("MainWindow: Found {} dangling reference(s) to {}", _graphicsRendererContextVar->refcount - 1,
                    GraphicsRendererContext::VariableName);
      }
      shards::releaseVariable(_graphicsRendererContextVar);
      _graphicsRendererContextVar = nullptr;
    }

    _graphicsContext = GraphicsContext{};
    _graphicsRendererContext = GraphicsRendererContext{};
  }

  bool begin(SHContext* shContext, shards::WindowContext &windowContext) {
    // Need to lazily init since we depend on renderer
    if (!_graphicsContext.context) {
      shards::callOnMeshThread(shContext, [&] { initRenderer(windowContext.window); });
    }

    auto &window = _graphicsContext.window;
    if (!window->isInitialized()) {
      SHLOG_WARNING("Failed to render to surface, window is closed. Frame skipped.");
      return false;
    }

    auto &renderer = _graphicsRendererContext.renderer;
    auto &context = _graphicsContext.context;

    gfx::int2 windowSize = window->getDrawableSize();
    try {
      shards::callOnMeshThread(shContext, [&] { context->resizeMainOutputConditional(windowSize); });
    } catch (std::exception &err) {
      SHLOG_WARNING("Swapchain creation failed: {}. Frame skipped.", err.what());
      return false;
    }

    double deltaTime = 0.0;
    _loop.beginFrame(0.0, deltaTime);
    if (context->beginFrame()) {
      _graphicsContext.time = _loop.getAbsoluteTime();
      _graphicsContext.deltaTime = deltaTime;
      renderer->beginFrame();

      _graphicsContext.frameInProgress = true;

      return true;
    }

    return false;
  }

  void end() { endFrame(_graphicsContext); }
};
} // namespace gfx

#endif /* BED776D7_7318_4FA7_92C1_CC63BC286F1A */
