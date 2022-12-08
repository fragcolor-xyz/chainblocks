#include "shards_types.hpp"
#include "shards_utils.hpp"
#include "drawable_utils.hpp"
#include <shards_macros.hpp>
#include <foundation.hpp>
#include <gfx/gltf/gltf.hpp>
#include <gfx/paths.hpp>
#include <linalg_shim.hpp>
#include <runtime.hpp>
#include <spdlog/spdlog.h>
#include "params.hpp"

using namespace shards;

namespace gfx {

struct GLTFShard {
  static inline Type PathInputType = CoreInfo::StringType;
  static inline Type ByteInputType = CoreInfo::BytesType;
  static inline Type TransformVarType = Type::VariableOf(CoreInfo::Float4x4Type);

  static SHTypesInfo inputTypes() { return CoreInfo::AnyTableType; }
  static SHTypesInfo outputTypes() { return Types::DrawableHierarchy; }

  PARAM_PARAMVAR(_transformVar, "Transform", "The transform variable to use", {CoreInfo::NoneType, TransformVarType});
  PARAM_VAR(_staticModelPath, "Path", "The static path to load a model from during warmup",
            {CoreInfo::NoneType, CoreInfo::StringType});
  PARAM_IMPL(GLTFShard, PARAM_IMPL_FOR(_transformVar), PARAM_IMPL_FOR(_staticModelPath));

  enum LoadMode {
    Invalid,
    LoadStaticFile,
    LoadFile,
    LoadMemory,
    LoadCopy,
  };

  LoadMode _loadMode{};
  DrawableHierarchyPtr _staticModel;
  bool _hasConstTransform{};
  SHDrawableHierarchy *_returnVar{};
  std::vector<FeaturePtr> _features;

  void releaseReturnVar() {
    if (_returnVar)
      Types::DrawableHierarchyObjectVar.Release(_returnVar);
    _returnVar = {};
  }

  void makeNewReturnVar() {
    releaseReturnVar();
    _returnVar = Types::DrawableHierarchyObjectVar.New();
  }

  SHTypeInfo compose(SHInstanceData &data) {
    auto &inputTableType = data.inputType.table;
    size_t inputTableLen = inputTableType.keys.len;

    auto findInputTableType = [&](const std::string &key) -> const SHTypeInfo * {
      for (size_t i = 0; i < inputTableLen; i++) {
        if (key == inputTableType.keys.elements[i]) {
          return &inputTableType.types.elements[i];
        }
      }
      return nullptr;
    };

    const SHTypeInfo *pathType = findInputTableType("Path");
    const SHTypeInfo *bytesType = findInputTableType("Bytes");
    const SHTypeInfo *copyType = findInputTableType("Copy");
    size_t numSources = (pathType ? 1 : 0) + (bytesType ? 1 : 0) + (copyType ? 1 : 0);
    if (numSources > 1) {
      throw ComposeError("glTF can only have one source");
    } else if (pathType) {
      if (*pathType != PathInputType)
        throw ComposeError("Path should be a string");
      _loadMode = LoadFile;
      OVERRIDE_ACTIVATE(data, activatePath);
    } else if (bytesType) {
      if (*bytesType != ByteInputType)
        throw ComposeError("Bytes should be a byte array");
      _loadMode = LoadMemory;
      OVERRIDE_ACTIVATE(data, activateBytes);
    } else if (copyType) {
      if (*copyType != Types::DrawableHierarchy)
        throw ComposeError("Copy should be an already loaded glTF model");
      _loadMode = LoadCopy;
      OVERRIDE_ACTIVATE(data, activateCopy);
    } else if (!_staticModelPath.isNone()) {
      _loadMode = LoadStaticFile;
      OVERRIDE_ACTIVATE(data, activateStatic);
    } else {
      throw ComposeError("glTF Binary or file path required when not loading a static model");
    }

    for (size_t i = 0; i < inputTableLen; i++) {
      std::string_view key = inputTableType.keys.elements[i];
      SHTypeInfo &type = inputTableType.types.elements[i];
      if (key != "Path" && key != "Bytes" && key != "Copy") {
        validateDrawableInputTableEntry(inputTableType.keys.elements[i], type);
      }
    }

    const SHTypeInfo *transformType = findInputTableType("Transform");
    _hasConstTransform = transformType != nullptr;
    if (_hasConstTransform) {
      if (*transformType != CoreInfo::Float4x4Type)
        throw ComposeError("Transform should be a Float4x4");
    }

    return Types::DrawableHierarchy;
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    if (!_staticModelPath.isNone()) {
      // Since loading from files is more of a debug funcitonality, try to load using the shards relative path
      std::string resolvedPath = gfx::resolveDataPath((const char *)_staticModelPath).string();

      SPDLOG_DEBUG("Loading static glTF model from {}", resolvedPath);
      _staticModel = loadGltfFromFile(resolvedPath.c_str());
    }
  }

  void cleanup() {
    PARAM_CLEANUP()

    _staticModel.reset();
    releaseReturnVar();
  }

  // Set & link transform
  void initModel(SHContext *context, SHDrawableHierarchy &shDrawable, const SHVar &input) {
    const SHTable &inputTable = input.payload.tableValue;

    // Set constant transform
    if (_hasConstTransform) {
      SHVar transformVar;
      getFromTable(context, inputTable, "Transform", transformVar);
      float4x4 transform = shards::Mat4(transformVar);

      shDrawable.drawableHierarchy->transform = transform;
    }

    // Link transform variable
    if (_transformVar.isVariable()) {
      shDrawable.transformVar = (SHVar &)_transformVar;
      shDrawable.transformVar.warmup(context);
    }

    _features.clear();
    SHVar featuresVar;
    if (getFromTable(context, inputTable, "Features", featuresVar)) {
      applyFeatures(context, _features, featuresVar);
      DrawableHierarchy::foreach (shDrawable.drawableHierarchy, [&](DrawableHierarchyPtr item) {
        for (auto &drawable : item->drawables) {
          drawable->features = _features;
        }
      });
    }
  }

  SHVar activateStatic(SHContext *context, const SHVar &input) {
    assert(_loadMode == LoadStaticFile);

    makeNewReturnVar();
    try {
      if (_staticModel) {
        _returnVar->drawableHierarchy = _staticModel->clone();
      } else {
        throw ActivationError("No glTF model was loaded");
      }

    } catch (...) {
      Types::DrawableHierarchyObjectVar.Release(_returnVar);
      throw;
    }

    initModel(context, *_returnVar, input);
    return Types::DrawableHierarchyObjectVar.Get(_returnVar);
  }

  SHVar activatePath(SHContext *context, const SHVar &input) {
    assert(_loadMode == LoadFile);

    SHVar pathVar{};
    getFromTable(context, input.payload.tableValue, "Path", pathVar);

    makeNewReturnVar();
    _returnVar->drawableHierarchy = loadGltfFromFile(pathVar.payload.stringValue);

    initModel(context, *_returnVar, input);
    return Types::DrawableHierarchyObjectVar.Get(_returnVar);
  }

  SHVar activateBytes(SHContext *context, const SHVar &input) {
    assert(_loadMode == LoadMemory);

    SHVar bytesVar{};
    getFromTable(context, input.payload.tableValue, "Bytes", bytesVar);

    makeNewReturnVar();
    _returnVar->drawableHierarchy = loadGltfFromMemory(bytesVar.payload.bytesValue, bytesVar.payload.bytesSize);

    initModel(context, *_returnVar, input);
    return Types::DrawableHierarchyObjectVar.Get(_returnVar);
  }

  SHVar activateCopy(SHContext *context, const SHVar &input) {
    assert(_loadMode == LoadCopy);

    SHVar sourceVar{};
    getFromTable(context, input.payload.tableValue, "Copy", sourceVar);

    SHDrawableHierarchy *source = reinterpret_cast<SHDrawableHierarchy *>(sourceVar.payload.objectValue);
    if (!source)
      throw ActivationError("Undefined glTF model passed to Copy");

    SHDrawableHierarchy *result = Types::DrawableHierarchyObjectVar.New();
    result->drawableHierarchy = source->drawableHierarchy->clone();

    initModel(context, *result, input);
    return Types::DrawableHierarchyObjectVar.Get(result);
  }

  SHVar activate(SHContext *context, const SHVar &input) { throw std::logic_error("invalid activation"); }
};

void registerGLTFShards() { REGISTER_SHARD("GFX.glTF", GLTFShard); }
} // namespace gfx