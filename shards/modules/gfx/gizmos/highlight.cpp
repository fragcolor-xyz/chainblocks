#include "context.hpp"

namespace shards {
namespace Gizmos {

using namespace gfx;

struct HighlightShard : public Base {
  // TODO: Merge with DrawShard type
  static inline shards::Types SingleDrawableTypes = shards::Types{gfx::Types::Drawable};
  static inline Type DrawableSeqType = Type::SeqOf(SingleDrawableTypes);
  static inline shards::Types DrawableTypes{gfx::Types::Drawable, DrawableSeqType};

  static SHTypesInfo inputTypes() { return DrawableTypes; }
  static SHTypesInfo outputTypes() { return CoreInfo::NoneType; }
  static SHOptionalString help() { return SHCCSTR("Queues a draw operation to highlight a drawable"); }

  SHTypeInfo compose(const SHInstanceData &data) {
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
        [&](auto &drawable) { _gizmoContext->wireframeRenderer.overlayWireframe(*_gizmoContext->queue.get(), *drawable.get()); },
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
  void warmup(SHContext *shContext) { baseWarmup(shContext); }
  void cleanup() { baseCleanup(); }
};

void registerHighlightShards() { REGISTER_SHARD("Gizmos.Highlight", HighlightShard); }
} // namespace Gizmos
} // namespace shards
