#include "gfx.hpp"
#include "param_types.hpp"
#include <shards/common_types.hpp>
#include "drawable_utils.hpp"
#include <shards/core/foundation.hpp>
#include <shards/shards.hpp>
#include "shards_utils.hpp"
#include <gfx/drawable.hpp>
#include <gfx/error_utils.hpp>
#include <gfx/material.hpp>
#include <gfx/mesh.hpp>
#include <magic_enum.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/core/params.hpp>
#include <shards/core/pool.hpp>
#include <type_traits>
#include <variant>

using namespace shards;
namespace gfx {
struct DrawableShard {
  static inline Type MeshVarType = Type::VariableOf(ShardsTypes::Mesh);
  static inline Type TransformVarType = Type::VariableOf(CoreInfo::Float4x4Type);

  PARAM_PARAMVAR(_mesh, "Mesh", "The mesh to use for this drawable.", {MeshVarType});
  PARAM_EXT(ParamVar, _material, ShardsTypes::MaterialParameterInfo);
  PARAM_EXT(ParamVar, _params, gfx::detail::ParamsParameterInfo);
  PARAM_EXT(ParamVar, _features, ShardsTypes::FeaturesParameterInfo);

  PARAM_IMPL(PARAM_IMPL_FOR(_mesh), PARAM_IMPL_FOR(_material), PARAM_IMPL_FOR(_params), PARAM_IMPL_FOR(_features));

  static SHOptionalString help() { return SHCCSTR("This shard creates a drawable object that can be added to a drawables queue for the render pipeline."); }
  static SHOptionalString inputHelp() { return SHCCSTR("The transformation matrix of the drawable object to adopt."); }
  static SHOptionalString outputHelp() { return SHCCSTR("The drawable object."); }
  static SHTypesInfo inputTypes() { return CoreInfo::Float4x4Type; }
  static SHTypesInfo outputTypes() { return ShardsTypes::Drawable; }

  SHDrawable *_drawable{};

  MeshDrawable::Ptr &getMeshDrawable() { return std::get<MeshDrawable::Ptr>(_drawable->drawable); }

  void releaseReturnVar() {}

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    _drawable = ShardsTypes::DrawableObjectVar.New();
    _drawable->drawable = std::make_shared<MeshDrawable>();
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);

    if (_drawable) {
      ShardsTypes::DrawableObjectVar.Release(_drawable);
      _drawable = {};
    }
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    if (_mesh.isNone())
      throw formatException("Mesh is required");

    return ShardsTypes::Drawable;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &meshDrawable = getMeshDrawable();

    bool changed{};

    float4x4 newTransform = toFloat4x4(input);
    if (newTransform != meshDrawable->transform) {
      meshDrawable->transform = newTransform;
      changed = true;
    }

    auto mesh = varAsObjectChecked<MeshPtr>(_mesh.get(), ShardsTypes::Mesh);
    if (mesh != meshDrawable->mesh) {
      meshDrawable->mesh = mesh;
      changed = true;
    }

    if (!_material.isNone()) {
      auto newMat = varAsObjectChecked<SHMaterial>(_material.get(), ShardsTypes::Material).material;
      if (newMat != meshDrawable->material) {
        meshDrawable->material = newMat;
        changed = true;
      }
    }

    if (!_params.isNone()) {
      if (initShaderParamsIfChanged(shContext, _params.get().payload.tableValue, meshDrawable->parameters))
        changed = true;
    }

    if (!_features.isNone()) {
      if (applyFeaturesIfChanged(shContext, meshDrawable->features, _features.get())) {
        changed = true;
      }
    }

    if (changed)
      meshDrawable->update();

    return ShardsTypes::DrawableObjectVar.Get(_drawable);
  }
};

struct DrawShard {
  static inline shards::Types SingleDrawableTypes = shards::Types{ShardsTypes::Drawable};
  static inline Type DrawableSeqType = Type::SeqOf(SingleDrawableTypes);
  static inline shards::Types DrawableTypes{ShardsTypes::Drawable, DrawableSeqType};

  PARAM_PARAMVAR(_queue, "Queue", "The queue object to add the drawable object to.",
                 {Type::VariableOf(ShardsTypes::DrawQueue)});
  PARAM_IMPL(PARAM_IMPL_FOR(_queue));

  static SHOptionalString help() { return SHCCSTR("This shard takes the input drawable object (or sequence of drawable objects) and adds them to the draw queue (created by GFX.DrawQueue) specified in the Queue parameter."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The drawable object (or sequence of drawable objects) to add to the draw queue."); }
  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  static SHTypesInfo inputTypes() { return DrawableTypes; }
  static SHTypesInfo outputTypes() { return DrawableTypes; }

  DrawShard() {}

  void warmup(SHContext *shContext) { PARAM_WARMUP(shContext); }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    if (!_queue.isVariable())
      throw ComposeError("Draw requires a queue");

    if (data.inputType.basicType == SHType::Seq) {
      OVERRIDE_ACTIVATE1(data, activateSeq);
    } else {
      OVERRIDE_ACTIVATE1(data, activateSingle);
    }

    return data.inputType;
  }

  DrawQueue &getDrawQueue() { return *varAsObjectChecked<SHDrawQueue>(_queue.get(), ShardsTypes::DrawQueue).queue.get(); }

  template <typename T> void addDrawableToQueue(T &drawable) { getDrawQueue().add(drawable); }

  void activateSingle(SHContext *shContext, const SHVar &input) {
    assert(input.valueType == SHType::Object);
    std::visit(
        [&](auto &drawable) {
          using T = std::decay_t<decltype(drawable)>;
          if constexpr (!std::is_same_v<T, std::monostate>) {
            addDrawableToQueue(drawable); //
          }
        },
        varAsObjectChecked<SHDrawable>(input, ShardsTypes::Drawable).drawable);
  }

  void activateSeq(SHContext *shContext, const SHVar &input) {
    auto &seq = input.payload.seqValue;
    for (size_t i = 0; i < seq.len; i++) {
      activateSingle(shContext, seq.elements[i]);
    }
  }

  void activate(SHContext *shContext, const SHVar &input) { throw ActivationError("Unsupported input type"); }
};

} // namespace gfx

namespace shards {
template <> struct PoolItemTraits<gfx::SHDrawQueue *> {
  gfx::SHDrawQueue *newItem() {
    auto queue = gfx::ShardsTypes::DrawQueueObjectVar.New();
    queue->queue = std::make_shared<gfx::DrawQueue>();
    return queue;
  }
  void release(gfx::SHDrawQueue *&v) { gfx::ShardsTypes::DrawQueueObjectVar.Release(v); }
  size_t getRefCount(gfx::SHDrawQueue *&v) { return gfx::ShardsTypes::DrawQueueObjectVar.GetRefCount(v); }
  bool canRecycle(gfx::SHDrawQueue *&v) { return getRefCount(v) == 1; }
  void recycled(gfx::SHDrawQueue *&v) { v->queue->clear(); }
};
} // namespace shards

namespace gfx {
struct DrawQueueShard {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::DrawQueue; }
  static SHOptionalString help() { return SHCCSTR("Creates a new drawable queue object to add drawables to (using GFX.Draw)."); }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The drawable queue object."); }

  PARAM_VAR(_autoClear, "AutoClear", "When enabled, automatically clears the queue after items have been rendered",
            {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_VAR(_threaded, "Threaded", "When enabled, output unique queue references to be able to use them with channels",
            {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_VAR(_trace, "Trace", "Enables debug tracing on this queue", {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_autoClear), PARAM_IMPL_FOR(_threaded), PARAM_IMPL_FOR(_trace));

  std::variant<SHDrawQueue *, Pool<SHDrawQueue *>> _output;

  DrawQueueShard() {
    _autoClear = Var(true);
    _threaded = Var(false);
    _trace = Var(false);
  }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    if ((bool)*_threaded) {
      _output.emplace<Pool<SHDrawQueue *>>();
    } else {
      _output.emplace<SHDrawQueue *>();
    }
    return ShardsTypes::DrawQueue;
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    if (SHDrawQueue **_queue = std::get_if<SHDrawQueue *>(&_output)) {
      auto &queue = *_queue;
      assert(!queue);
      queue = ShardsTypes::DrawQueueObjectVar.New();
      queue->queue = std::make_shared<DrawQueue>();
      initQueue(queue->queue);
    }
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);

    if (SHDrawQueue **_queue = std::get_if<SHDrawQueue *>(&_output)) {
      auto &queue = *_queue;
      if (queue) {
        ShardsTypes::DrawQueueObjectVar.Release(queue);
        queue = nullptr;
      }
    }
  }

  void initQueue(const DrawQueuePtr &queue) {
    queue->setAutoClear(_autoClear->isNone() ? true : (bool)*_autoClear);
    queue->trace = _trace->isNone() ? false : (bool)*_trace;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    if (SHDrawQueue **_queue = std::get_if<SHDrawQueue *>(&_output)) {
      (*_queue)->queue->clear();
      return ShardsTypes::DrawQueueObjectVar.Get((*_queue));
    } else {
      auto &pool = std::get<Pool<SHDrawQueue *>>(_output);
      pool.recycle();
      auto queue = pool.newValue([&](SHDrawQueue *&_queue) { initQueue(_queue->queue); });
      queue->queue->clear();
      return ShardsTypes::DrawQueueObjectVar.Get(queue);
    }
  }
};

struct ClearQueueShard {
  static SHTypesInfo inputTypes() { return ShardsTypes::DrawQueue; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString help() { return SHCCSTR("Removes all drawable objects from the input drawable queue."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The drawable queue object to clear."); }
  static SHOptionalString outputHelp() { return SHCCSTR("None."); }

  static SHParametersInfo parameters() {
    static Parameters parameters = {};
    return parameters;
  }

  void setParam(int index, const SHVar &value) {}
  SHVar getParam(int index) { return Var::Empty; }

  void warmup(SHContext *context) {}
  void cleanup(SHContext *context) {}

  SHVar activate(SHContext *shContext, const SHVar &input) {
    SHDrawQueue *shQueue = reinterpret_cast<SHDrawQueue *>(input.payload.objectValue);
    shQueue->queue->clear();
    return SHVar{};
  }
};

struct GetQueueDrawablesShard {
  static inline const Type OutputSeqType = Type::SeqOf(ShardsTypes::Drawable);

  static SHTypesInfo inputTypes() { return ShardsTypes::DrawQueue; }
  static SHTypesInfo outputTypes() { return OutputSeqType; }
  static SHOptionalString help() { return SHCCSTR("Retrieves the drawable objects from the input drawable queue and outputs it as a sequence of drawable objects."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The drawable queue to retrieve the drawable objects from."); }
  static SHOptionalString outputHelp() { return SHCCSTR("A sequence of drawable objects."); }

  SeqVar _outputSeq;
  std::vector<SHDrawable *> _objects;

  static SHParametersInfo parameters() {
    static Parameters parameters = {};
    return parameters;
  }
  void setParam(int index, const SHVar &value) {}
  SHVar getParam(int index) { return Var::Empty; }

  void warmup(SHContext *context) {}
  void cleanup(SHContext *context) {}

  SHVar activate(SHContext *shContext, const SHVar &input) {
    SHDrawQueue *shQueue = reinterpret_cast<SHDrawQueue *>(input.payload.objectValue);
    auto &drawables = shQueue->queue->getDrawables();
    _objects.clear();
    _outputSeq.clear();
    _objects.reserve(drawables.size());
    for (auto &drawable : drawables) {
      SHDrawable *ptr = _objects.emplace_back(ShardsTypes::DrawableObjectVar.New());
      ptr->assign(drawable->clone());
      _outputSeq.push_back(ShardsTypes::DrawableObjectVar.Get(ptr));
      ShardsTypes::DrawableObjectVar.Release(ptr);
    }

    return _outputSeq;
  }
};

void registerDrawableShards() {
  REGISTER_SHARD("GFX.Drawable", DrawableShard);
  REGISTER_SHARD("GFX.Draw", DrawShard);
  REGISTER_SHARD("GFX.DrawQueue", DrawQueueShard);
  REGISTER_SHARD("GFX.ClearQueue", ClearQueueShard);
  REGISTER_SHARD("GFX.QueueDrawables", GetQueueDrawablesShard);
}
} // namespace gfx
