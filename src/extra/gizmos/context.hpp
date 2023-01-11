#ifndef C55FCF15_9B8F_44F0_AA7B_73BA2144579D
#define C55FCF15_9B8F_44F0_AA7B_73BA2144579D

#include "../gfx.hpp"
#include <common_types.hpp>
#include <foundation.hpp>
#include <gfx/fwd.hpp>
#include <gfx/gizmos/gizmos.hpp>
#include <gfx/gizmos/wireframe.hpp>

namespace shards {
// Shards for rendering visual helpers
// Gizmos, grids, lines, etc.
namespace Gizmos {

struct Context {
  static constexpr uint32_t TypeId = 'HLPc';
  static inline shards::Type Type = shards::Type::Object(gfx::VendorId, TypeId);

  static inline const char contextVarName[] = "Gizmos.Context";
  static inline SHExposedTypeInfo contextExposedType =
      shards::ExposedInfo::Variable(contextVarName, SHCCSTR("The helper context."), Context::Type);
  static inline shards::ExposedInfo contextExposedTypes = shards::ExposedInfo(contextExposedType);

  gfx::DrawQueuePtr queue;
  gfx::gizmos::Context gizmoContext;
  gfx::WireframeRenderer wireframeRenderer;
};

struct BaseConsumer {
  SHVar *_contextVar{nullptr};

  Context &getContext() {
    Context *contextPtr = reinterpret_cast<Context *>(_contextVar->payload.objectValue);
    if (!contextPtr) {
      throw shards::ActivationError("Context not set");
    }
    return *contextPtr;
  }

  SHExposedTypesInfo requiredVariables() { return SHExposedTypesInfo(Context::contextExposedTypes); }

  void baseConsumerWarmup(SHContext *context, bool contextRequired = true) {
    assert(!_contextVar);
    _contextVar = shards::referenceVariable(context, Context::contextVarName);
    if (contextRequired) {
      assert(_contextVar->valueType == SHType::Object);
    }
  }

  void baseConsumerCleanup() {
    if (_contextVar) {
      shards::releaseVariable(_contextVar);
      _contextVar = nullptr;
    }
  }

  void composeCheckGfxThread(const SHInstanceData &data) {
    if (data.onWorkerThread) {
      throw shards::ComposeError("GFX Shards cannot be used on a worker thread (e.g. "
                                 "within an Await shard)");
    }
  }

  void composeCheckContext(const SHInstanceData &data) {
    bool variableFound = false;
    for (uint32_t i = 0; i < data.shared.len; i++) {
      if (strcmp(data.shared.elements[i].name, Context::contextVarName) == 0) {
        variableFound = true;
      }
    }

    if (!variableFound)
      throw shards::ComposeError("Context required, but not found");
  }

  void warmup(SHContext *context) { baseConsumerWarmup(context); }

  void cleanup(SHContext *context) { baseConsumerCleanup(); }

  SHTypeInfo compose(const SHInstanceData &data) {
    composeCheckGfxThread(data);
    composeCheckContext(data);
    return shards::CoreInfo::NoneType;
  }
};

} // namespace Gizmos
} // namespace shards

#endif /* C55FCF15_9B8F_44F0_AA7B_73BA2144579D */
