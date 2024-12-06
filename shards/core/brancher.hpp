#ifndef AC3ADB13_86B0_4D83_9DA4_AF8BC421F010
#define AC3ADB13_86B0_4D83_9DA4_AF8BC421F010

#include "foundation.hpp"
#include <shards/common_types.hpp>
#include <shards/core/exposed_type_utils.hpp>
#include "runtime.hpp"
#include "into_wire.hpp"
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <unordered_set>

namespace shards {

enum BranchFailureBehavior { Everything, Known, Ignore };
DECL_ENUM_INFO(BranchFailureBehavior, BranchFailure,
               "Defines how to handle failures in branching operations. Determines whether to continue execution, handle known "
               "errors, or ignore failures entirely.",
               'brcB');

// Runs wires in a separate mesh while capturing variables
struct Brancher {
public:
  using IgnoredVariables = std::unordered_set<std::string_view>;

  std::shared_ptr<SHMesh> mesh = SHMesh::make();
  std::vector<std::shared_ptr<SHWire>> wires;
  bool captureAll = false;
  BranchFailureBehavior failureBehavior = BranchFailureBehavior::Everything;

private:
  std::unordered_map<std::string_view, SHExposedTypeInfo> _collectedRequirements;
  std::unordered_set<std::string_view> _copyBySerialize;
  ExposedInfo _mergedRequirements;
  ExposedInfo _shared;
  mutable std::vector<SHTypeInfo> _cachedObjectTypes;

public:
  ~Brancher() { cleanup(nullptr); }

  // Adds a single wire or sequence of shards as a looped wire
  void addRunnable(const SHVar &var, const char *defaultWireName = "inline-wire") {
    wires.push_back(IntoWire{}.defaultWireName(defaultWireName).var(var));
  }

  // Sets the runnables (wires or shards)
  void setRunnables(const SHVar &var) {
    wires.clear();
    IntoWires{wires}.var(var);
  }

  SHExposedTypesInfo requiredVariables() { return (SHExposedTypesInfo)_mergedRequirements; }

  const shards::ExposedInfo &getMergedRequirements() const { return _mergedRequirements; }
  const std::unordered_set<std::string_view> &getCopyBySerialize() const { return _copyBySerialize; }

  struct ShareableObjectFlags {
    bool sharable{};
    bool copyBySerialize{};
  };

  // Provides information about this type:
  // - if it doesn't contain objects it can be shared between threads by cloning (sharable = true)
  // - if it contains objects but they can be serialized and deserialized,
  //   it can be shared by serializing, then deserializing (copyBySerialize = true, sharable = true)
  ShareableObjectFlags getSharableObjectFlags(const SHTypeInfo &type, bool shareObjectVariables) const {
    _cachedObjectTypes.clear();
    getObjectTypes(_cachedObjectTypes, type);

    bool copyBySerialize = false;
    for (auto &objType : _cachedObjectTypes) {
      auto *ti = findObjectInfo(objType.object.vendorId, objType.object.typeId);
      if (ti && ti->deserialize && ti->serialize) {
        copyBySerialize = true;
      } else if (!ti || !ti->isThreadSafe) {
        return ShareableObjectFlags{false, false};
      }
    }

    return ShareableObjectFlags{true, copyBySerialize};
  }

  void compose(const SHInstanceData &data, const ExposedInfo &shared_ = ExposedInfo{}, const IgnoredVariables &ignored = {},
               bool shareObjectVariables = true) {
    _collectedRequirements.clear();
    _copyBySerialize.clear();

    SHInstanceData tmpData = data;
    tmpData.privateContext = nullptr; // null this, in order to create a new context!
    ExposedInfo shared{shared_};
    for (auto &type : data.shared) {
      if (ignored.find(type.name) != ignored.end())
        continue;
      if (!shareObjectVariables) {
        ShareableObjectFlags flags = getSharableObjectFlags(type.exposedType, shareObjectVariables);
        if (!flags.sharable)
          continue;
        if (flags.copyBySerialize)
          _copyBySerialize.insert(type.name);
      }
      shared.push_back(type);
    }
    tmpData.shared = SHExposedTypesInfo(shared);

    for (auto &wire : wires) {
      composeSubWire(tmpData, wire);
    }

    if (captureAll) {
      // Merge deep requirements
      for (auto &avail : data.shared) {
        SHLOG_TRACE("Branch: adding variable to requirements: {}", avail.name);
        _mergedRequirements.push_back(avail);
      }
    } else {
      for (auto &avail : data.shared) {
        auto it = _collectedRequirements.find(avail.name);
        if (it != _collectedRequirements.end()) {
          SHLOG_TRACE("Branch: adding variable to requirements: {}", avail.name);
          _mergedRequirements.push_back(it->second);
        }
      }
    }

    // Clear exposed flags, since these are copies
    for (auto &req : _mergedRequirements._innerInfo) {
      req.tracked = false;
    }

    // Copy shared
    _shared = ExposedInfo(_mergedRequirements);
    mesh->instanceData.shared = (SHExposedTypesInfo)_mergedRequirements;
  }

  // Calls initVariableReferences, then schedule (in that order)
  void warmup(SHContext *context, const SHVar &input = Var::Empty) {
    auto parentMesh = context->main->mesh.lock();
    if (parentMesh) {
      mesh->parent = parentMesh.get();
      shassert(mesh->parent != mesh.get());
    }
    initVariableReferences(context);
    schedule(input);
  }

  // This grabs all the variables that are needed and references them
  void initVariableReferences(SHContext *context) {
    for (const auto &req : _mergedRequirements._innerInfo) {
      std::string_view name = req.name; // TODO calls strlen :(
      if (!mesh->hasRef(toSWL(name))) {
        SHLOG_TRACE("Branch: referencing required variable: {}", name);
        auto vp = referenceVariable(context, name);
        // TODO, this should just be mesh external variables, just like we do for wire
        mesh->addRef(toSWL(name), vp);
        refs.push_back(vp);
      }
    }
  }

  // Schedules the wires to run on the mesh (calling warmup in the process)
  void schedule(const SHVar &input = Var::Empty) {
    for (const auto &wire : wires) {
      mesh->schedule(wire, input, false);
    }
  }

  void cleanup(SHContext *context = nullptr) {
    mesh->terminate();
    for (auto ref : refs) {
      shards::releaseVariable(ref);
    }
    refs.clear();
    mesh->parent = nullptr;
  }

  // Update the that the branched wires read
  void updateInputs(const SHVar &input = Var::Empty) {
    emplaceInputs([&](OwnedVar &outVar) { outVar = input; });
  }

  // Update inputs directly without copying, should still transfer ownership
  template <typename T> void emplaceInputs(T &&cb) {
    for (auto &wire : wires) {
      cb(wire->currentInput);
    }
  }

  void activate() {
    if (!mesh->tick()) {
      switch (failureBehavior) {
      case BranchFailureBehavior::Ignore:
        break;
      case BranchFailureBehavior::Everything:
        throw ActivationError("Branched mesh had errors");
      case BranchFailureBehavior::Known:
        for (const auto &wire : wires) {
          auto failed = mesh->failedWires();
          if (std::count(failed.begin(), failed.end(), wire.get())) {
            throw ActivationError("Branched mesh had errors");
          }
        }
        break;
      }
    }
  }

private:
  void composeSubWire(const SHInstanceData &data, const std::shared_ptr<SHWire> &wire) {
    wire->mesh = mesh; // this is needed to be correct

    auto dataCopy = data;
    dataCopy.wire = wire.get();

    // Branch needs to capture all it needs, so we need deeper informations
    // this is triggered by populating requiredVariables variable
    dataCopy.requiredVariables = &_collectedRequirements;

    wire->composeResult = composeWire(wire.get(), dataCopy);
  }

  std::vector<SHVar *> refs;
};
} // namespace shards

#endif /* AC3ADB13_86B0_4D83_9DA4_AF8BC421F010 */
