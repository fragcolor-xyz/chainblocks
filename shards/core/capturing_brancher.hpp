#ifndef C33BD856_265F_46BD_9B9C_EF38EBF8E36B
#define C33BD856_265F_46BD_9B9C_EF38EBF8E36B

#include "brancher.hpp"
#include <string_view>
#include <unordered_map>

namespace shards {

// Variant of brancher that captures variables
// example usage could be a wire running on a different thread
struct CapturingBrancher {
  using IgnoredVariables = Brancher::IgnoredVariables;
  using VariableRefs = std::unordered_map<std::string, SHVar *>;
  using CapturedVariables = std::unordered_map<std::string, OwnedVar>;

  Brancher brancher;
  std::unordered_map<std::string, SHVar> variableStorage;

private:
  bool _variablesApplied{};

public:
  auto &mesh() { return brancher.mesh; }
  auto &wires() { return brancher.wires; }

  void addRunnable(auto &v, const char *name = "inline-wire") { brancher.addRunnable(v, name); }

  auto requiredVariables() { return brancher.requiredVariables(); }

  void compose(const SHInstanceData &data, const ExposedInfo& shared = ExposedInfo{}, const IgnoredVariables &ignored = {}) {
    _variablesApplied = false;
    brancher.compose(data, shared, ignored, false);
  }

  // context: The context to capture from
  // ignored: A list of variable names to ignore / provided manually
  void intializeCaptures(VariableRefs &outRefs, SHContext *context, const IgnoredVariables &ignored = {}) {
    for (const auto &req : brancher.requiredVariables()) {
      if (!outRefs.contains(req.name)) {
        if (req.exposedType.basicType == SHType::Object) {
          auto *objType = findObjectInfo(req.exposedType.object.vendorId, req.exposedType.object.typeId);
          if (!objType || !objType->isThreadSafe)
            throw ComposeError(fmt::format("Unable to capture object variable {}", req.name));
        }

        SHVar *vp = referenceVariable(context, req.name);
        outRefs.insert_or_assign(req.name, vp);
      }
    }
  }

  // WARNING: Need to keep variables alive during the liftime of this brancher
  void applyCapturedVariables(CapturedVariables &variables) {
    if (!_variablesApplied) {
      // Initialize references here
      for (auto &vr : variables) {
        SHVar &var = variableStorage[vr.first];
        var.flags = SHVAR_FLAGS_REF_COUNTED;
        var.refcount = 1; 
        brancher.mesh->addRef(ToSWL(vr.first), &var);
      }
      _variablesApplied = true;
    }

    for (auto &vr : variables) {
      auto ref = brancher.mesh->getRefIfExists(ToSWL(vr.first));
      assert(ref != nullptr);
      assignVariableValue(*ref, vr.second);
    }
  }

  void warmup(const SHVar &input = Var::Empty) {
    if (!_variablesApplied)
      throw std::logic_error("Variables not applied before warmup, call applyCapturedVariables before warmup");
    brancher.schedule(input);
  }

  void cleanup() {
    brancher.mesh->terminate();
    variableStorage.clear();
  }

  void activate() { brancher.activate(); }
};
} // namespace shards

#endif /* C33BD856_265F_46BD_9B9C_EF38EBF8E36B */
