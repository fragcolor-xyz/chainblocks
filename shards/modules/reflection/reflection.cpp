/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>

namespace shards {
namespace reflection {
struct Shards {
  SHOptionalString help() {
    return SHCCSTR("Given a wire as input it will recurse deep inside it and gather all "
                   "shards in the wire, its sub-wires and sub-flows.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::WireType; }
  static SHTypesInfo outputTypes() { return CoreInfo::ShardRefSeqType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    std::vector<ShardInfo> shards;
    _output.clear();
    auto wire = SHWire::sharedFromRef(input.payload.wireValue);
    gatherShards(wire.get(), shards);
    for (auto &item : shards) {
      _output.emplace_back(const_cast<Shard *>(item.shard));
    }
    return Var(_output);
  }

  std::vector<Var> _output;
};

struct Name {
  static SHTypesInfo inputTypes() { return CoreInfo::ShardRefType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    Shard *blk = input.payload.shardValue;
    _name.assign(blk->name(blk));
    return Var(_name);
  }
  std::string _name;
};

struct Get {
  SHOptionalString help() { return SHCCSTR("Gets a variable dynamically by name. Returns None if the variable doesn't exist."); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  PARAM_PARAMVAR(_variableName, "Name", "The name of the variable to get", {CoreInfo::StringType, CoreInfo::StringVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_variableName))

  std::string _currentVarName;
  SHVar *_ref{};

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) {
    resetRef();
    PARAM_CLEANUP(context);
  }

  void resetRef() {
    if (_ref) {
      releaseVariable(_ref);
      _ref = nullptr;
      _currentVarName.clear();
    }
  }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto varName = SHSTRVIEW(_variableName.get());
    if (varName != _currentVarName || _ref == nullptr) {
      resetRef();
      _ref = findVariable(context, varName);
      if (_ref) {
        _currentVarName = varName;
      }
    }
    return _ref ? *_ref : Var::Empty;
  }
};

} // namespace reflection
} // namespace shards
SHARDS_REGISTER_FN(reflection) {
  REGISTER_SHARD("Reflect.Shards", shards::reflection::Shards);
  REGISTER_SHARD("Reflect.Name", shards::reflection::Name);
  REGISTER_SHARD("Reflect.Get", shards::reflection::Get);
}