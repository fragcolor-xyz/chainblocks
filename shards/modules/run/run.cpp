/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>

namespace shards {
namespace run_ {
struct Schedule {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }

  PARAM_PARAMVAR(_mesh, "Mesh", "The mesh to run", {SHMesh::MeshType});
  PARAM_PARAMVAR(_wire, "Wire", "The wire to run", {shards::CoreInfo::WireType, shards::CoreInfo::WireVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_mesh), PARAM_IMPL_FOR(_wire));

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &sharedMesh = *reinterpret_cast<std::shared_ptr<SHMesh> *>(_mesh->payload.objectValue);
    auto &wire = SHWire::sharedFromRef(_wire->payload.wireValue);
    sharedMesh->schedule(wire, input);
    return input;
  }
};

using RunClock = std::chrono::high_resolution_clock;
static std::optional<RunClock::time_point> resumeAt;

struct Run {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::BoolType; }

  PARAM_PARAMVAR(_mesh, "Mesh", "The mesh to run", {SHMesh::MeshType});
  PARAM_PARAMVAR(_fps, "FPS", "Frames per second",
                 {shards::CoreInfo::NoneType, shards::CoreInfo::IntType, shards::CoreInfo::IntVarType});
  PARAM_PARAMVAR(_iterations, "Iterations", "Number of iterations",
                 {shards::CoreInfo::NoneType, shards::CoreInfo::IntType, shards::CoreInfo::IntVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_mesh), PARAM_IMPL_FOR(_fps), PARAM_IMPL_FOR(_iterations));

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    std::function<void()> delay;
    auto &fpsVar = (Var &)_fps.get();
    auto &mesh = *reinterpret_cast<std::shared_ptr<SHMesh> *>(_mesh->payload.objectValue);
    auto &iterationsVar = (Var &)_iterations.get();

    size_t numIterations = ~0; // Indefinitely
    if (iterationsVar.valueType == SHType::Int) {
      numIterations = iterationsVar.payload.intValue;
    }
    size_t iteration = 0;
    bool noErrors = true;

    if (!fpsVar.isNone()) {
      double sleepDuration = 1.0f / float(fpsVar.payload.intValue);
      while (!mesh->empty()) {
        if(!mesh->tick()) {
          noErrors = false;
        }
        SH_SUSPEND(context, sleepDuration);
        if (numIterations != ~0 && ++iteration >= numIterations) {
          break;
        }
      }
    } else {
      while (!mesh->empty()) {
        if(!mesh->tick()) {
          noErrors = false;
        }
        SH_SUSPEND(context, 0);
        if (numIterations != ~0 && ++iteration >= numIterations) {
          break;
        }
      }
    }

    return Var(noErrors);
  }
};

} // namespace run_
} // namespace shards
SHARDS_REGISTER_FN(run) {
  using namespace shards::run_;
  REGISTER_SHARD("Schedule", Schedule);
  REGISTER_SHARD("Run", Run);
}