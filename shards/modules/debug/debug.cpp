#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>
#include <boost/core/span.hpp>

namespace shards::Debug {
struct DebugNoop {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("A shard that can be set to break in the debugger at a specific location"); }

  PARAM_PARAMVAR(_tag, "Tag", "Any tag to identify this debug shard", {shards::CoreInfo::NoneType, shards::CoreInfo::AnyType});
  PARAM_PARAMVAR(_inspect, "Inspect", "Anything to visualize", {shards::CoreInfo::NoneType, shards::CoreInfo::AnyType});
  PARAM_IMPL(PARAM_IMPL_FOR(_tag), PARAM_IMPL_FOR(_inspect));

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    volatile SHWire *__wire = shContext->currentWire();
    volatile SHMesh *__mesh = shContext->currentWire()->mesh.lock().get();
    if (!_tag.isNone()) {
      SHVar &tagValue = _tag.get();
      SHLOG_TRACE("Triggered debug noop ({})", tagValue);
    } else {
      SHLOG_TRACE("Triggered debug noop");
    }
    return input;
  }
};

struct DumpEnv {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("Dumps the variable environment during compose"); }

  PARAM_IMPL();

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    std::stringstream ss;
    ss << "Compose environment:\n";
    auto span = boost::span(data.shared.elements, data.shared.len);

    std::sort(span.begin(), span.end(),
              [](const SHExposedTypeInfo &a, const SHExposedTypeInfo &b) { return std::string_view(a.name) < b.name; });
    for (auto &v : data.shared) {
      ss << fmt::format(" - {} ({})\n", v.name, v.exposedType);
    }
    SHLOG_INFO("{}", ss.str());

    return data.inputType;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return input; }
};
} // namespace shards::Debug

namespace shards {
SHARDS_REGISTER_FN(debug) {
  using namespace Debug;
  REGISTER_SHARD("Debug.Noop", DebugNoop);
  REGISTER_SHARD("Debug.DumpEnv", DumpEnv);
}
} // namespace shards
