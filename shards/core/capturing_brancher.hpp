#ifndef C33BD856_265F_46BD_9B9C_EF38EBF8E36B
#define C33BD856_265F_46BD_9B9C_EF38EBF8E36B

#include "brancher.hpp"
#include "shards/core/foundation.hpp"
#include "shards/shards.h"
#include <shards/core/serialization.hpp>
#include <string_view>
#include <unordered_map>

namespace shards {

// Variant of brancher that captures variables by copying
// example usage could be a wire running on a different thread
struct CapturingBrancher {
  struct CloningContext {
    shards::Serialization s;
    std::vector<uint8_t> buffer;
  };

  struct VariableRef {
    SHVar *_var{};
    bool _copyBySerialize;

  public:
    ~VariableRef() {
      if (_var) {
        releaseVariable(_var);
        _var = nullptr;
      }
    }
    VariableRef(SHVar *_var, bool _copyBySerialize) : _var(_var), _copyBySerialize(_copyBySerialize) {}
    VariableRef(VariableRef &&other) {
      _var = other._var;
      _copyBySerialize = other._copyBySerialize;
      other._var = nullptr;
    }
    VariableRef &operator=(VariableRef &&other) {
      _var = other._var;
      _copyBySerialize = other._copyBySerialize;
      other._var = nullptr;
      return *this;
    }
    VariableRef(const VariableRef &) = delete;
    VariableRef &operator=(const VariableRef &) = delete;
    void cloneInto(OwnedVar &target, CloningContext &ctx) const {
      if (_copyBySerialize) {
        ZoneScopedN("Serialize variable");
        ctx.buffer.clear();
        BufferRefWriter w(ctx.buffer);
        ctx.s.serialize(*_var, w);
        BufferRefReader r(ctx.buffer);
        OwnedVar _new;
        ctx.s.deserialize(r, _new);
        target = std::move(_new);
      } else {
        ZoneScopedN("Copy variable");
        target = *_var;
      }
    }
  };

  using IgnoredVariables = Brancher::IgnoredVariables;
  using VariableRefs = std::unordered_map<std::string, VariableRef>;
  using CapturedVariables = std::unordered_map<std::string, OwnedVar>;

  Brancher brancher;
  std::unordered_map<std::string, SHVar> variableStorage;
  CapturedVariables variableState;

private:
  bool _variablesApplied{};

public:
  ~CapturingBrancher() { cleanup(); }

  auto &mesh() { return brancher.mesh; }
  auto &wires() { return brancher.wires; }

  void addRunnable(auto &v, const char *name = "inline-wire") { brancher.addRunnable(v, name); }

  auto requiredVariables() { return brancher.requiredVariables(); }

  void compose(const SHInstanceData &data, const ExposedInfo &shared = ExposedInfo{}, const IgnoredVariables &ignored = {},
               bool shareObjectVariables = false) {
    _variablesApplied = false;
    ExposedInfo sharedNonMutable{data.shared};
    for (auto &data : sharedNonMutable._innerInfo) {
      // Make all captured variables non-mutable as it doesn't make sense to mutate them
      // they would get overwritten every time the capturing brancher updates it's variables
      data.isMutable = false;
    }
    SHInstanceData dataInner = data;
    dataInner.shared = SHExposedTypesInfo(sharedNonMutable);
    brancher.compose(dataInner, shared, ignored, shareObjectVariables);
  }

  // context: The context to capture from
  // ignored: A list of variable names to ignore / provided manually
  void intializeCaptures(VariableRefs &outRefs, SHContext *context, const IgnoredVariables &ignored = {}) {
    for (const auto &req : brancher.requiredVariables()) {
      if (!outRefs.contains(req.name)) {
        bool copyBySerialize = brancher.getCopyBySerialize().contains(req.name);
        SHVar *vp = referenceVariable(context, req.name);
        outRefs.insert_or_assign(req.name, VariableRef(vp, copyBySerialize));
      }
    }
  }

  void applyCapturedVariablesSwap(CapturedVariables &_variables) {
    std::swap(variableState, _variables);

    if (!_variablesApplied) {
      // Initialize references here
      for (auto &vr : variableState) {
        SHVar &var = variableStorage[vr.first];
        var.flags = SHVAR_FLAGS_FOREIGN | SHVAR_FLAGS_REF_COUNTED;
        var.refcount = 1;
        brancher.mesh->addRef(toSWL(vr.first), &var);
      }
      _variablesApplied = true;
    }

    for (auto &vr : variableState) {
      auto ref = brancher.mesh->getRefIfExists(toSWL(vr.first));
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
    for (auto &storage : variableStorage) {
      if (storage.second.refcount > 1) {
        SHLOG_ERROR("Variable {} still has a refcount of {}", storage.first, storage.second.refcount);
      }
    }
    variableStorage.clear();
    variableState.clear();
  }

  void activate() { brancher.activate(); }
};
} // namespace shards

#endif /* C33BD856_265F_46BD_9B9C_EF38EBF8E36B */
