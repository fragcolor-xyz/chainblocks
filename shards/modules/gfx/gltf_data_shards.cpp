#include "data_cache.hpp"
#include <shards/core/module.hpp>
#include <shards/core/serialization.hpp>
#include <shards/core/params.hpp>
#include <shards/core/object_type.hpp>
#include <shards/common_types.hpp>
#include <gfx/gltf/impl.hpp>
#include <shards/gfx/gltf/gltf2.hpp>
#include <gfx/filesystem.hpp>

using shards::CoreInfo;
using shards::Var;
namespace gfx {

struct GLTF2Obj {
  glTF2 gltf;
};
constexpr char GLTF2_NAME[] = "GFX.glTF2";
using GLTF2ObjInfo = shards::TObjectTypeInfo<GLTF2Obj, 'glf2', shards::CoreCC, GLTF2_NAME>;

struct GLTF2Shard {
  PARAM_PARAMVAR(_rootPath, "RootPath", "Root path for relative file paths",
                 {CoreInfo::NoneType, CoreInfo::StringType, CoreInfo::StringVarType})
  PARAM_IMPL(PARAM_IMPL_FOR(_rootPath))

  static SHTypesInfo inputTypes() {
    static shards::Types inputTypes{CoreInfo::StringType, CoreInfo::BytesType};
    return inputTypes;
  }
  static SHTypesInfo outputTypes() { return GLTF2ObjInfo::Type; }

  shards::OwnedVar _outputCache;

  GLTF2Shard() { _rootPath = Var::Empty; }

  enum ReadMode {
    ReadFile,
    ReadMemory,
  } _readMode;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    if (data.inputType == CoreInfo::StringType) {
      _readMode = ReadFile;
    } else if (data.inputType == CoreInfo::BytesType) {
      _readMode = ReadMemory;
    } else {
      throw std::invalid_argument("Invalid input type");
    }
    return outputTypes().elements[0];
  }

  void warmup(SHContext *ctx) {
    PARAM_WARMUP(ctx);
    if (!gfx::data::getInstance()) {
      gfx::data::setInstance(std::make_shared<gfx::data::DataCache>(gfx::getDefaultDataCacheIO()));
    }
  }

  void cleanup(SHContext *ctx) {
    PARAM_CLEANUP(ctx);
    _outputCache.reset();
  }

  SHVar activate(SHContext *ctx, const SHVar &input) {
    auto [gltfObj, gltfObjVar] = GLTF2ObjInfo::ObjectVar.NewOwnedVar();

    std::optional<fs::Path> rootPath;

    Var &rootPathVar = (Var &)_rootPath.get();
    if (!rootPathVar.isNone()) {
      auto strView = SHSTRVIEW(rootPathVar);
      rootPath = fs::Path(strView.begin(), strView.end());
    }

    std::optional<tinygltf::Model> model;
    if (_readMode == ReadFile) {
      auto filePath = SHSTRVIEW(input);
      model = loadGltfModelFromFile2(filePath);
      if (!rootPath) {
        rootPath = fs::Path(filePath.begin(), filePath.end()).parent_path();
      }
    } else {
      auto bytes = SHBYTESVIEW(input);
      if (!rootPath) {
        rootPath = shards::GetGlobals().RootPath;
      }
      model = loadGltfModelFromMemory2(bytes, *rootPath);
    }

    gltfObj.gltf = processGltfModel2(*model, *rootPath);

    return (_outputCache = gltfObjVar);
  }
};

SHARDS_REGISTER_FN(gltf_data) { REGISTER_SHARD("GLTF.Read", GLTF2Shard); }
} // namespace gfx