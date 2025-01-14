#include <shards/linalg_shim.hpp>
#include "context.hpp"
#include "../shards_types.hpp"

namespace shards {
namespace Gizmos {

using namespace gfx;

struct HighlightShard : public Base {
  // TODO: Merge with DrawShard type
  static inline shards::Types SingleDrawableTypes = shards::Types{gfx::ShardsTypes::Drawable};
  static inline Type DrawableSeqType = Type::SeqOf(SingleDrawableTypes);
  static inline shards::Types DrawableTypes{gfx::ShardsTypes::Drawable, DrawableSeqType};

  static SHTypesInfo inputTypes() { return DrawableTypes; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString help() { return SHCCSTR("Queues a draw operation to highlight a drawable"); }

  PARAM_PARAMVAR(_color, "Color", "Color to render wireframe at", {CoreInfo::Float4Type});
  PARAM_IMPL(PARAM_IMPL_FOR(_color));

  const float4 &getColor() const { return (float4 &)_color.get().payload; }

  HighlightShard() { _color = toVar(float4(1.0f, 0.0f, 0.0f, 1.0f)); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    if (data.inputType.basicType == SHType::Seq) {
      OVERRIDE_ACTIVATE(data, activateSeq);
    } else {
      OVERRIDE_ACTIVATE(data, activateSingle);
    }
    return CoreInfo::NoneType;
  }

  SHVar activateSingle(SHContext *shContext, const SHVar &input) {
    SHDrawable *drawable = static_cast<SHDrawable *>(input.payload.objectValue);
    std::visit(
        [&](auto &drawable) {
          _gizmoContext->wireframeRenderer.overlayWireframe(*_gizmoContext->queue.get(), *drawable.get(), getColor());
        },
        drawable->drawable);

    return SHVar{};
  }

  SHVar activateSeq(SHContext *shContext, const SHVar &input) {
    auto &seq = input.payload.seqValue;
    for (size_t i = 0; i < seq.len; i++) {
      (void)activateSingle(shContext, seq.elements[i]);
    }

    return SHVar{};
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { throw ActivationError("Unsupported input type"); }
  void warmup(SHContext *shContext) {
    PARAM_WARMUP(shContext);
    baseWarmup(shContext);
  }
  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    baseCleanup(context);
  }
};

void registerHighlightShards() { REGISTER_SHARD("Gizmos.Highlight", HighlightShard); }
} // namespace Gizmos
} // namespace shards
