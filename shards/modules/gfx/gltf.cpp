#include "boost/filesystem/path.hpp"
#include <shards/common_types.hpp>
#include "gfx/gltf/animation.hpp"
#include <shards/core/params.hpp>
#include <shards/shards.hpp>
#include "shards_types.hpp"
#include "shards_utils.hpp"
#include "drawable_utils.hpp"
#include "anim/types.hpp"
#include "anim/path.hpp"
#include <memory>
#include <deque>
#include <optional>
#include <shards/core/shards_macros.hpp>
#include <shards/core/foundation.hpp>
#include <gfx/drawables/mesh_tree_drawable.hpp>
#include <gfx/gltf/gltf.hpp>
#include <gfx/paths.hpp>
#include <tracy/Wrapper.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/core/runtime.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string_view>

using namespace shards;

namespace gfx {

#define GLTF

static const char internalComponentIdentifier = '$';

// Checks if the given name points to an internal glTF component (translation/rotation/scale)
static bool isGltfBuiltinTarget(std::string_view path) { return path.starts_with(internalComponentIdentifier); }

static std::optional<animation::BuiltinTarget> getGltfBuiltinTarget(std::string_view path) {
  if (!isGltfBuiltinTarget(path))
    return std::nullopt;
  path = path.substr(1);
  if (path == "t") {
    return animation::BuiltinTarget::Translation;
  } else if (path == "r") {
    return animation::BuiltinTarget::Rotation;
  } else if (path == "s") {
    return animation::BuiltinTarget::Scale;
  } else {
    throw std::runtime_error(fmt::format("Invalid internal glTF target {}", path));
  }
}

static OwnedVar getGltfBuiltinTargetPath(animation::BuiltinTarget target) {
  OwnedVar result;
  char pathStr[] = {internalComponentIdentifier, 0, 0};
  switch (target) {
  case animation::BuiltinTarget::Rotation:
    pathStr[1] = 'r';
    break;
  case animation::BuiltinTarget::Scale:
    pathStr[1] = 's';
    break;
  case animation::BuiltinTarget::Translation:
    pathStr[1] = 't';
    break;
  }
  pathStr[2] = 0;
  cloneVar(result, Var(pathStr, 2));
  return result;
}

SeqVar getAnimationPath(MeshTreeDrawable::Ptr node) {
  SeqVar result;
  auto rootNode = MeshTreeDrawable::findRoot(node);
  MeshTreeDrawable::traverseDown(rootNode, node, [&](auto &node) {
    std::string_view sv(node->name.str());
    Var tmp(sv.data(), sv.size());
    SHVar cloned{};
    cloneVar(cloned, tmp);
    result.push_back(cloned);
  });

  return result;
};

SeqVar getAnimationPath(MeshTreeDrawable::Ptr node, animation::BuiltinTarget target) {
  SeqVar result = getAnimationPath(node);
  result.emplace_back_fast() = getGltfBuiltinTargetPath(target);
  return result;
};

struct GLTFShard {
  static inline Type PathInputType = CoreInfo::StringType;
  static inline Type ByteInputType = CoreInfo::BytesType;
  static inline Type TransformVarType = Type::VariableOf(CoreInfo::Float4x4Type);
  static inline Type AnimationTable = Type::TableOf(Animations::ShardsTypes::Animation);

  static inline shards::Types InputTableTypes{CoreInfo::Float4x4Type, CoreInfo::AnyType};
  static inline std::array<SHVar, 2> InputTableKeys{Var("transform"), Var()};
  static inline Type InputTable = Type::TableOf(InputTableTypes, InputTableKeys);

  static inline shards::Types InputTypes{CoreInfo::Float4x4Type, InputTable};

  static SHTypesInfo inputTypes() { return InputTypes; }
  static SHTypesInfo outputTypes() { return ShardsTypes::Drawable; }

  static SHOptionalString help() {
    return SHCCSTR("This shard takes the glTF model (specified either in the Path, Bytes or Copy parameter) and outputs a "
                   "drawable object which can be subsequently added to the drawables queue for the render pipline.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("This shard can take 2 different types of input. It can take a 4x4 transformation "
                   "for the glTF model to adopt. Or, it can accept a table which contain the "
                   "transformation matrix and additional information to modify the materials of the glTF model.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the drawable object."); }

  PARAM_PARAMVAR(_path, "Path", "The path to load the model from",
                 {CoreInfo::NoneType, CoreInfo::StringType, CoreInfo::StringVarType});
  PARAM_PARAMVAR(_bytes, "Bytes", "The bytes to load the model from",
                 {CoreInfo::NoneType, CoreInfo::BytesType, CoreInfo::BytesVarType});
  PARAM_PARAMVAR(_copy, "Copy", "Reference to another glTF model to copy",
                 {CoreInfo::NoneType, Type::VariableOf(ShardsTypes::Drawable)});
  PARAM_VAR(_wrapRootNode, "WrapRootNode", "Wrap the root node so animations applied to it still work",
            {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_EXT(ParamVar, _params, ShardsTypes::ParamsParameterInfo);
  PARAM_EXT(ParamVar, _features, ShardsTypes::FeaturesParameterInfo);
  PARAM(ShardsVar, _animController, "AnimationController", "The animation controller", {CoreInfo::ShardsOrNone});
  PARAM_IMPL(PARAM_IMPL_FOR(_path), PARAM_IMPL_FOR(_bytes), PARAM_IMPL_FOR(_copy), PARAM_IMPL_FOR(_params),
             PARAM_IMPL_FOR(_features), PARAM_IMPL_FOR(_animController), PARAM_IMPL_FOR(_wrapRootNode));

  enum LoadMode {
    Invalid,
    LoadFileStatic,
    LoadFileDynamic,
    LoadMemory,
    LoadCopy,
  };

  std::optional<glTF> _model;
  LoadMode _loadMode{};
  bool _hasConstTransform{};
  SHDrawable *_drawable{};
  bool _dynamicsApplied{};
  TableVar _animations;

  bool _hasInputTable{};

  MeshTreeDrawable::Ptr &getMeshTreeDrawable() { return std::get<MeshTreeDrawable::Ptr>(_drawable->drawable); }

  bool hasAnimationController() const { return _animController.shards().len > 0; }

  void wrapRootNode(glTF &gltf) {
    auto oldRoot = gltf.root;
    gltf.root = std::make_shared<MeshTreeDrawable>();
    gltf.root->name = "__shards_root";
    gltf.root->addChild(oldRoot);

    for (auto &[name, animation] : gltf.animations) {
      for (auto &track : animation.tracks) {
        if (track.targetNode.lock() == oldRoot) {
          track.targetNode = gltf.root;
        }
      }
    }
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    _hasInputTable = data.inputType.basicType == SHType::Table;
    bool havePath = !_path.isNone();
    bool haveBytes = !_bytes.isNone();
    bool haveCopy = !_copy.isNone();

    size_t numSources = (havePath ? 1 : 0) + (haveBytes ? 1 : 0) + (haveCopy ? 1 : 0);
    if (numSources > 1) {
      throw ComposeError("glTF can only have one source (Path, Bytes or Copy)");
    } else if (havePath) {
      if (_path.isNotNullConstant()) {
        _loadMode = LoadFileStatic;
      } else {
        _loadMode = LoadFileDynamic;
      }
    } else if (haveBytes) {
      _loadMode = LoadMemory;
    } else if (haveCopy) {
      _loadMode = LoadCopy;
    } else {
      throw ComposeError("glTF Binary, file path or copy source required");
    }

    if (hasAnimationController()) {
      SHInstanceData childData = data;
      childData.inputType = AnimationTable;
      _animController.compose(childData);
      SHLOG_TRACE("Checking animation frame data {}: {}", data.wire->name, _animController.composeResult().outputType);
      if (!shards::matchTypes(_animController.composeResult().outputType, Animations::ShardsTypes::AnimationValues, false, true,
                              false))
        throw std::runtime_error(fmt::format("Invalid animation frame data: {}, expected: {}",
                                             _animController.composeResult().outputType,
                                             Animations::ShardsTypes::AnimationValues));
    }

    return ShardsTypes::Drawable;
  }

  void shardifyAnimationData() {
    ZoneScoped;
    for (auto &[name, animation] : _model->animations) {
      SeqVar &tracks = _animations.get<SeqVar>(Var(name));
      for (auto &track : animation.tracks) {
        TableVar trackTable;
        trackTable.get<SeqVar>(Var("Path")) = getAnimationPath(track.targetNode.lock(), track.target);

        std::optional<Var> interpolationValue;
        if (track.interpolation != animation::Interpolation::Linear) {
          auto &enumType = Animations::ShardsTypes::InterpolationEnumInfo::Type;
          switch (track.interpolation) {
          case animation::Interpolation::Step:
            interpolationValue = Var::Enum(Animations::Interpolation::Step, enumType);
            break;
          case animation::Interpolation::Cubic:
            interpolationValue = Var::Enum(Animations::Interpolation::Cubic, enumType);
            break;
          default:
            throw std::out_of_range("Unsupported value");
            break;
          }
        }

        SeqVar frames;
        frames.resize(track.times.size());

        size_t frameIndex{};
        for (auto &time : track.times) {
          TableVar frameTableVar;

          frameTableVar.get<Var>(Var("Time")) = Var{time};

          if (interpolationValue) {
            frameTableVar.get<Var>(Var("Interpolation")) = *interpolationValue;
          }

          if (track.interpolation == animation::Interpolation::Cubic) {
            std::visit([&](auto &&v) -> void { frameTableVar.get<Var>(Var("In")) = toVar(v); },
                       track.getValue(frameIndex * 3 + 0));
            std::visit([&](auto &&v) -> void { frameTableVar.get<Var>(Var("Value")) = toVar(v); },
                       track.getValue(frameIndex * 3 + 1));
            std::visit([&](auto &&v) -> void { frameTableVar.get<Var>(Var("Out")) = toVar(v); },
                       track.getValue(frameIndex * 3 + 2));
          } else {
            Var &value = frameTableVar.get<Var>(Var("Value"));
            std::visit([&](auto &&v) -> void { value = toVar(v); }, track.getValue(frameIndex));
          }

          frames[frameIndex] = std::move(frameTableVar.asOwned());
          ++frameIndex;
        }

        trackTable.get<SeqVar>(Var("Frames")) = std::move(frames);
        tracks.emplace_back_fast() = std::move(trackTable.asOwned());
      }
    }
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    _drawable = ShardsTypes::DrawableObjectVar.New();
    _drawable->drawable = MeshTreeDrawable::Ptr();

    switch (_loadMode) {
    case LoadMode::LoadFileStatic:
      _model.emplace(loadGltfFromFile(SHSTRVIEW(_path.get())));
      break;
    case LoadMode::LoadFileDynamic:
    case LoadMode::LoadMemory:
    case LoadMode::LoadCopy:
      break;
    default:
      throw std::out_of_range("glTF load mode");
    }

    _dynamicsApplied = false;
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context)

    if (_drawable) {
      ShardsTypes::DrawableObjectVar.Release(_drawable);
      _drawable = {};
    }

    _model.reset();
  }

  void applyRecursiveParams(SHContext *context) {
    auto &meshTreeDrawable = getMeshTreeDrawable();

    MeshTreeDrawable::foreach (meshTreeDrawable, [&](MeshTreeDrawable::Ptr item) {
      for (auto &drawable : item->drawables) {
        bool changed{};
        if (!_params.isNone()) {
          if (initShaderParamsIfChanged(context, _params.get().payload.tableValue, drawable->parameters))
            changed = true;
        }

        drawable->features.clear();
        if (!_features.isNone()) {
          applyFeatures(context, drawable->features, _features.get());
          changed = true;
        }

        if (changed)
          drawable->update();
      }
    });
  }

  // Modifies path and returns the mesh tree node found
  // the path will contain the elements after the node that was found
  // there can only be builtin target identifiers left in the path
  MeshTreeDrawable::Ptr findNode(Animations::Path &p) {
    MeshTreeDrawable::Ptr node = _model->root;
    if (_drawable->rootNodeWrapped)
      node = node->getChildren()[0];

    while (true) {
      if (!node)
        return MeshTreeDrawable::Ptr();

      // Stop when end of path is reached
      // or we encounter a builtin target identifier
      if (!p || isGltfBuiltinTarget(p.getHead()))
        return node;

      for (auto &child : node->getChildren()) {
        if (child->name.str() == p.getHead()) {
          node = child;
          p = p.next();
          goto _next;
        }
      }
      return MeshTreeDrawable::Ptr();
    _next:;
    }
  }

  std::optional<animation::BuiltinTarget> findTarget(Animations::Path &p) {
    if (p.length == 1) {
      return getGltfBuiltinTarget(p.getHead());
    }
    return std::nullopt;
  }

  void applyFrame(const MeshTreeDrawable::Ptr &node, animation::BuiltinTarget target, const SHVar &value) {
    switch (target) {
    case animation::BuiltinTarget::Rotation:
      node->trs.rotation = toVec<float4>(value);
      break;
    case animation::BuiltinTarget::Scale:
      node->trs.scale = toVec<float3>(value);
      break;
    case animation::BuiltinTarget::Translation:
      node->trs.translation = toVec<float3>(value);
      break;
    }
    node->update();
  }

  void applyAnimationData(SeqVar &data) {
    ZoneScoped;
    for (auto &v : data) {
      TableVar &valueTable = (TableVar &)v;
      auto &pathSeq = valueTable.get<SeqVar>(Var("Path"));

      Animations::Path path(pathSeq);
      MeshTreeDrawable::Ptr node = findNode(path);
      if (!node) {
        SHLOG_WARNING("Animation node path not found: {}", pathSeq);
        continue;
      }
      auto target = findTarget(path);
      if (!target) {
        SHLOG_WARNING("Animation node path not found: {}", pathSeq);
        continue;
      }

      auto &value = valueTable.get<Var>(Var("Value"));
      applyFrame(node, target.value(), value);
    }
  }

  bool isFloat4x4(const SHVar &var) {
    return var.valueType == SHType::Seq && var.innerType == SHType::Float4 && var.payload.seqValue.len == 4;
  }
  bool isValidPath(const SHVar &var) {
    for (auto &elem : var.payload.seqValue) {
      if (elem.valueType != SHType::String)
        return false;
    }
    return true;
  }

  void replaceMaterial(std::string_view materialName, MaterialPtr dst) {
    auto matIt = _model->materials.find(std::string(materialName));
    if (matIt == _model->materials.end()) {
      throw std::runtime_error(fmt::format("Material not found {}", materialName));
    }

    if (matIt->second == dst)
      return; // Nothing to do

    MeshTreeDrawable::foreach (_model->root, [&](MeshTreeDrawable::Ptr &ptr) {
      for (auto &drawable : ptr->drawables) {
        if (drawable->material == matIt->second) {
          drawable->material = dst;
        }
      }
    });

    // Replace mapping
    matIt->second = dst;
  }

  void applyInputTable(SHContext *context, const TableVar &table) {
    const auto &rootNode = getMeshTreeDrawable();

    auto &transformVar = table["transform"];
    if (!transformVar->isNone()) {
      TRS newTRS = toFloat4x4(transformVar);
      if (rootNode->trs != newTRS) {
        rootNode->trs = newTRS;
        rootNodeUpdated = true;
      }
    }

    auto &materialsTable = (TableVar &)table["materials"];
    if (materialsTable.valueType != SHType::None) {
      if (materialsTable.valueType != SHType::Table)
        throw std::runtime_error("materials should be a table");
      for (auto &[k, v] : materialsTable.payload.tableValue) {
        if (k.valueType != SHType::String)
          throw std::runtime_error(fmt::format("material key should be a string ({})", k));
        if (v.valueType == SHType::Object && isObjectType(v, ShardsTypes::Material)) {
          auto &shmaterial = *reinterpret_cast<SHMaterial *>(v.payload.objectValue);
          replaceMaterial(SHSTRVIEW(k), shmaterial.material);
          continue;
        }

        if (v.valueType != SHType::Table)
          throw std::runtime_error(fmt::format("material value for {} should be a table or material (was {})", k, v));

        auto material = _model->materials[k.payload.stringValue];
        if (!material) {
          SHLOG_WARNING("Material not found {}", k);
          continue;
        }

        auto &table = (TableVar &)v;

        auto &drawableParams = table["params"];
        if (!drawableParams->isNone()) {
          if (drawableParams.valueType != SHType::Table)
            throw std::runtime_error(fmt::format("params for key {} should be a table (v)", k, v));
          if (initShaderParamsIfChanged(context, drawableParams.payload.tableValue, material->parameters)) {
            updateEverything = true;
          }
        }
      }
    }

    for (auto &[k, v] : table) {
      if (k.valueType == SHType::Seq) {
        if (!isValidPath(k)) {
          throw std::runtime_error(fmt::format("Invalid path {}", k));
        }

        bool nodeUpdated{};
        auto updateVal = [&](auto &dst, const auto &src) {
          if (src != dst)
            dst = src;
          nodeUpdated = true;
        };

        Animations::Path path(k);
        auto node = findNode(path);
        if (!node) {
          SHLOG_WARNING("glTF Node not found {}", k);
          continue;
        }

        if (v.valueType != SHType::Table)
          throw std::runtime_error(fmt::format("Node data for key {} should be a table, was: {}", k, v));
        auto &valueTable = (TableVar &)v;

        auto &translation = valueTable["translation"];
        if (!translation->isNone()) {
          if (translation.valueType != SHType::Float3)
            throw std::runtime_error(fmt::format("translation for key {} should be a float3", k));
          updateVal(node->trs.translation, toFloat3(translation));
        }

        auto &rotation = valueTable["rotation"];
        if (!rotation->isNone()) {
          if (rotation.valueType != SHType::Float4)
            throw std::runtime_error(fmt::format("rotation for key {} should be a float3", k));
          updateVal(node->trs.rotation, toFloat4(rotation));
        }

        auto &scale = valueTable["scale"];
        if (!scale->isNone()) {
          if (scale.valueType != SHType::Float3)
            throw std::runtime_error(fmt::format("scale for key {} should be a float3", k));
          updateVal(node->trs.scale, toFloat3(scale));
        }

        auto &transform = valueTable["transform"];
        if (!transform->isNone()) {
          if (transform.valueType != SHType::Seq)
            throw std::runtime_error(fmt::format("transform for key {} should be a float4x4", k));
          updateVal(node->trs, TRS(toFloat4x4(transform)));
        }

        auto &drawableParams = valueTable["params"];
        if (!drawableParams->isNone()) {
          if (drawableParams.valueType != SHType::Table)
            throw std::runtime_error(fmt::format("params for key {} should be a table", k));
          for (auto &d : node->drawables) {
            if (initShaderParamsIfChanged(context, drawableParams.payload.tableValue, d->parameters))
              nodeUpdated = true;
          }
        }

        if (nodeUpdated && !updateEverything /*Skip till later*/) {
          node->update();
        }
      }
    }
  }

  bool rootNodeUpdated{};
  bool updateEverything{};

  SHVar activate(SHContext *context, const SHVar &input) {
    rootNodeUpdated = false;
    updateEverything = false;

    auto &drawable = getMeshTreeDrawable();
    if (!drawable) {
      bool rootNodeWrapped = false;

      switch (_loadMode) {
      case LoadFileDynamic:
        _model.emplace(loadGltfFromFile(SHSTRVIEW(_path.get())));
        break;
      case LoadMemory:
        _model.emplace(loadGltfFromMemory(_bytes.get().payload.bytesValue, _bytes.get().payload.bytesSize));
        break;
      case LoadCopy: {
        auto &shOther = varAsObjectChecked<SHDrawable>(_copy.get(), ShardsTypes::Drawable);
        MeshTreeDrawable::Ptr &other = *std::get_if<MeshTreeDrawable::Ptr>(&shOther.drawable);
        _model.emplace();
        _model->root = std::static_pointer_cast<MeshTreeDrawable>(other->clone());
        _model->animations = shOther.animations;
        _model->materials = shOther.materials;
        _model->boundsMin = shOther.boundsMin;
        _model->boundsMax = shOther.boundsMax;
        rootNodeWrapped = shOther.rootNodeWrapped;
      } break;
      case LoadMode::LoadFileStatic:
        assert(_model); // Loaded in warmup
        break;
      default:
        throw std::out_of_range("glTF load mode");
        break;
      }

      if (!rootNodeWrapped && !_wrapRootNode->isNone() && (bool)*_wrapRootNode) {
        wrapRootNode(*_model);
        _drawable->rootNodeWrapped = true;
      }

      _drawable->drawable = _model->root;
      _drawable->animations = _model->animations;
      _drawable->materials = _model->materials;
      _drawable->boundsMin = _model->boundsMin;
      _drawable->boundsMax = _model->boundsMax;

      if (hasAnimationController()) {
        shardifyAnimationData();
      }
    }

    if (_animController.shards().len > 0) {
      SHVar animationData;
      {
        ZoneScopedN("AnimationController");
        SHWireState state = _animController.activate(context, _animations, animationData);
        if (state == SHWireState::Error) {
          throw std::runtime_error("Animation controller failed");
        }
      }

      applyAnimationData((SeqVar &)animationData);
    }

    if (_hasInputTable) {
      applyInputTable(context, (TableVar &)input);
    } else {
      TRS newTRS = toFloat4x4(input);
      if (newTRS != drawable->trs) {
        drawable->trs = newTRS;
        rootNodeUpdated = true;
      }
    }

    // Only apply recursive parameters once
    if (!_dynamicsApplied) {
      applyRecursiveParams(context);
    }

    if (updateEverything) {
      MeshTreeDrawable::foreach (getMeshTreeDrawable(), [&](MeshTreeDrawable::Ptr node) { node->update(); });
    } else {
      if (rootNodeUpdated) {
        getMeshTreeDrawable()->update();
      }
    }

    return ShardsTypes::DrawableObjectVar.Get(_drawable);
  }
};

struct GLBPacker {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BytesType; }

  static SHOptionalString help() {
    return SHCCSTR("This shard takes the glTF file (specified by the input file path) and converts it into GLB format.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("Path to the glTF file to convert."); }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the GLB bytes of the converted glTF file."); }

  std::vector<uint8_t> _output;

  SHVar activate(SHContext *ctx, const SHVar &input) {
    auto path = SHSTRING_PREFER_SHSTRVIEW(input);
    _output = convertToGlb(path);
    return Var(_output);
  }
};

void registerGLTFShards() {
  REGISTER_SHARD("GFX.glTF", GLTFShard);
  REGISTER_SHARD("GLTF.PackGLB", GLBPacker);
}
} // namespace gfx