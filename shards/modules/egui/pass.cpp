#include <shards/modules/gfx/shards_types.hpp>
#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include "egui_render_pass.hpp"

using namespace gfx;

namespace shards::egui {
using gfx::Types;
using shards::CoreInfo;

struct UIPassShard {

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return gfx::Types::PipelineStep; }

  PARAM_PARAMVAR(_queue, "Queue", "The queue to draw from (Optional). Uses the default queue if not specified",
                 {CoreInfo::NoneType, Type::VariableOf(Types::DrawQueue)});
  PARAM_IMPL(PARAM_IMPL_FOR(_queue));

  PipelineStepPtr *_stepPtr{};

  RenderDrawablesStep &getRenderDrawablesStep() {
    assert(_stepPtr);
    return std::get<RenderDrawablesStep>(*_stepPtr->get());
  }

  void cleanup() {
    if (_stepPtr) {
      Types::PipelineStepObjectVar.Release(_stepPtr);
      _stepPtr = nullptr;
    }
    PARAM_CLEANUP();
  }

  void warmup(SHContext *context) {
    _stepPtr = Types::PipelineStepObjectVar.New();
    PARAM_WARMUP(context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    Var queueVar(_queue.get());
    if (queueVar.isNone())
      throw ActivationError("Queue is required");

    SHDrawQueue *shDrawQueue = (reinterpret_cast<SHDrawQueue *>(queueVar.payload.objectValue));

    if (!(*_stepPtr)) {
      *_stepPtr = EguiRenderPass::createPipelineStep(shDrawQueue->queue);
    }
    getRenderDrawablesStep().drawQueue = shDrawQueue->queue;

    return Types::PipelineStepObjectVar.Get(_stepPtr);
  }
};

} // namespace shards::egui
SHARDS_REGISTER_FN(pass) { REGISTER_SHARD("GFX.UIPass", shards::egui::UIPassShard); }
