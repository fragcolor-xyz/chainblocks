#include "gfx.hpp"
#include "drawable_utils.hpp"
#include "gfx.hpp"
#include "drawable_utils.hpp"
#include "shards_types.hpp"
#include <shards/common_types.hpp>
#include <shards/core/capturing_brancher.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/runtime.hpp>
#include <shards/core/params.hpp>
#include <shards/core/shared.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <shards/utility.hpp>
#include <shards/iterator.hpp>
#include "shader/composition.hpp"
#include "shards_utils.hpp"
#include "buffer.hpp"
#include <gfx/context.hpp>
#include <gfx/enums.hpp>
#include <gfx/error_utils.hpp>
#include <gfx/feature.hpp>
#include <gfx/params.hpp>
#include <gfx/shader/types.hpp>
#include <gfx/texture.hpp>
#include <gfx/features/base_color.hpp>
#include <gfx/features/debug_color.hpp>
#include <gfx/features/transform.hpp>
#include <gfx/features/wireframe.hpp>
#include <gfx/features/velocity.hpp>
#include <shards/linalg_shim.hpp>
#include <magic_enum.hpp>
#include <array>
#include <memory>
#include <stdexcept>
#include <string.h>
#include <unordered_map>
#include <variant>

using namespace shards;

namespace gfx {
enum class BuiltinFeatureId { Transform, BaseColor, VertexColorFromNormal, Wireframe, Velocity, AlphaBlend };
enum class RequiredAttributes_ { Tangent };
} // namespace gfx

ENUM_HELP(gfx::RequiredAttributes_, gfx::RequiredAttributes_::Tangent, SHCCSTR("Require mesh tangents to be available"));

ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::Transform, SHCCSTR("Add basic world/view/projection transform"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::BaseColor,
          SHCCSTR("Add basic color from vertex color and (optional) color texture"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::VertexColorFromNormal, SHCCSTR("Outputs color from vertex color"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::Wireframe, SHCCSTR("Modifies the main color to visualize vertex edges"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::Velocity,
          SHCCSTR("Outputs object velocity into the velocity global & output"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::AlphaBlend, SHCCSTR("Simple feature that enables alpha blending"));

namespace gfx {
DECL_ENUM_INFO(
    RequiredAttributes_, RequiredAttributes,
    "Attributes required for a graphics feature. Specifies the necessary properties or capabilities for a feature to function.",
    'fatt');

struct BuiltinFeatureShard {
  DECL_ENUM_INFO(BuiltinFeatureId, BuiltinFeatureId,
                 "Identifier for built-in graphics features. Used to reference pre-defined functionality in the graphics system.",
                 'feid');

  static SHOptionalString help() {
    return SHCCSTR("This shard creates the ready-made feature object of the feature specified in the ID parameter, for use in a "
                   "rendering pass.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The feature object for use in a rendering pass."); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::Feature; }

  static inline Parameters params{{"Id", SHCCSTR("Builtin feature id."), {BuiltinFeatureIdEnumInfo::Type}}};
  static SHParametersInfo parameters() { return params; }

  BuiltinFeatureId _id{};
  FeaturePtr *_feature{};

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _id = BuiltinFeatureId(value.payload.enumValue);
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var::Enum(_id, VendorId, BuiltinFeatureIdEnumInfo::TypeId);
    default:
      return Var::Empty;
    }
  }

  void cleanup(SHContext *context) {
    if (_feature) {
      ShardsTypes::FeatureObjectVar.Release(_feature);
      _feature = nullptr;
    }
  }

  void warmup(SHContext *context) {
    assert(!_feature);
    _feature = ShardsTypes::FeatureObjectVar.New();
    switch (_id) {
    case BuiltinFeatureId::Transform:
      *_feature = features::Transform::create();
      break;
    case BuiltinFeatureId::BaseColor:
      *_feature = features::BaseColor::create();
      break;
    case BuiltinFeatureId::VertexColorFromNormal:
      *_feature = features::DebugColor::create("normal", ProgrammableGraphicsStage::Vertex);
      break;
    case BuiltinFeatureId::Wireframe:
      *_feature = features::Wireframe::create();
      break;
    case BuiltinFeatureId::Velocity:
      *_feature = features::Velocity::create();
      break;
    case BuiltinFeatureId::AlphaBlend:
      *_feature = std::make_shared<Feature>();
      (*_feature)->state.set_blend(BlendState{.color = BlendComponent::Alpha, .alpha = BlendComponent::Opaque});
      break;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) { return ShardsTypes::FeatureObjectVar.Get(_feature); }
};

struct FeatureShard {
  static inline shards::Types GeneratedViewInputTableTypes{{ShardsTypes::DrawQueue, ShardsTypes::View, ShardsTypes::FeatureSeq}};
  static inline std::array<SHVar, 3> GeneratedViewInputTableKeys{Var("Queue"), Var("View"), Var("Features")};
  static inline Type GeneratedViewInputTableType = Type::TableOf(GeneratedViewInputTableTypes, GeneratedViewInputTableKeys);

  static inline shards::Type DrawableDataSeqType = Type::SeqOf(CoreInfo::IntType);
  static inline shards::Types GeneratedDrawInputTableTypes{
      {ShardsTypes::DrawQueue, ShardsTypes::View, ShardsTypes::FeatureSeq, DrawableDataSeqType}};
  static inline std::array<SHVar, 4> GeneratedDrawInputTableKeys{Var("Queue"), Var("View"), Var("Features"), Var("Drawables")};
  static inline Type GeneratedDrawInputTableType = Type::TableOf(GeneratedDrawInputTableTypes, GeneratedDrawInputTableKeys);

  static inline Type ShaderEntryPointType = Type::SeqOf(CoreInfo::AnyTableType);

  static inline shards::Types ParamSpecEntryTypes{ShardsTypes::ShaderParamOrVarTypes, {CoreInfo::AnyTableType}};
  static inline Type ParameterSpecType = Type::TableOf(ParamSpecEntryTypes);

  static inline Type BlockSpecType = CoreInfo::AnyTableType;

  static inline Type RequiredAttributesSeqType = Type::SeqOf(RequiredAttributesEnumInfo::Type);

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::Feature; }

  static SHOptionalString help() {
    return SHCCSTR("This shard creates a feature object based on what was provided in the different parameters.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The feature object for use in a rendering pass."); }

  // A shader entry point looks like the following
  //   {:Name <string> :Stage <ProgrammableGraphicsStage> :Before [...] :After [...] :EntryPoint (-> ...)}
  // The Before/After parameters are optional and can be used to specify dependencies
  // Each dependency will make sure that entry point is evaluated before or after the named dependency
  // You can give each entry point a name for this purpose using the Name parameter
  PARAM_PARAMVAR(_shaders, "Shaders", "A list of shader entry points", {CoreInfo::NoneType, ShaderEntryPointType});
  // Any constant undefined variables used in the shader can be injected using this parameter
  PARAM_PARAMVAR(_composeWith, "ComposeWith", "Any table of values that need to be injected into this feature's shaders",
                 {CoreInfo::NoneType, CoreInfo::AnyTableType, CoreInfo::AnyVarTableType});
  // A table, any key used in the applyState function can be used here (you can also check shards\gfx\pipeline_states.def for a
  // list of states) States are applied in the order that the features are references, so features that are added after this one
  // can override this feature's state Any states not in this table will inherit the parent's values
  PARAM_PARAMVAR(_state, "State", "The table of render state flags to override",
                 {CoreInfo::NoneType, CoreInfo::AnyTableType, CoreInfo::AnyVarTableType});
  // [(-> ...)]/(-> ...)/[<wire>]/<wire>
  // Any variables used in the callback will be copied during this shard's activation
  // The output should be a single parameter table
  PARAM_VAR(_viewGenerators, "ViewGenerators",
            "A collection of callbacks that will be run to generate per-view shader parameters during rendering. These "
            "parameters are added to the view buffer.",
            {IntoWires::RunnableTypes, {CoreInfo::NoneType}});
  // [(-> ...)]/(-> ...)/[<wire>]/<wire>
  // Any variables used in the callback will be copied during this shard's activation
  // The output should be a sequence of parameter tables (each with the same type)
  PARAM_VAR(_drawableGenerators, "DrawableGenerators",
            "A collection of callbacks that will be run to generate per-drawable shader parameters during rendering",
            {IntoWires::RunnableTypes, {CoreInfo::NoneType}});
  // Table of shader parameters, can be defined using any of the formats below:
  //   :<name> {:Type <ShaderFieldBaseType> :Dimension <number>}
  //   :<name> {:Default <default>}   (type will be derived from value)
  //   :<name> <default-value>        (type will be derived from value)
  PARAM_PARAMVAR(_params, "Params",
                 "The parameters to add to the object buffer and expose to shaders, these default values can later be modified "
                 "by the Params parameter in GFX.Material or GFX.Drawable.",
                 {CoreInfo::NoneType, ParameterSpecType, Type::VariableOf(ParameterSpecType)});

  // Table of block shader parameters, can be defined using any of the formats below:
  //   :<name> {:Type <ShaderFieldBaseType> :Dimension <number>}
  //   :<name> {:Default <default>}   (type will be derived from value)
  //   :<name> <default-value>        (type will be derived from value)
  PARAM_VAR(_blockParams, "BlockParams", "Custom bindings to expose to shaders", {CoreInfo::NoneType, BlockSpecType});

  PARAM_VAR(_uniqueVariables, "UniqueVariables", "List of variables that should be made unique",
            {CoreInfo::NoneType, CoreInfo::StringSeqType});

  PARAM_PARAMVAR(
      _requiredAttributes, "RequiredAttributes",
      "The parameters to expose to shaders, these default values can later be overriden by materials or drawable Params",
      {CoreInfo::NoneType, RequiredAttributesSeqType});

  PARAM_IMPL(PARAM_IMPL_FOR(_shaders), PARAM_IMPL_FOR(_composeWith), PARAM_IMPL_FOR(_state), PARAM_IMPL_FOR(_viewGenerators),
             PARAM_IMPL_FOR(_drawableGenerators), PARAM_IMPL_FOR(_params), PARAM_IMPL_FOR(_blockParams),
             PARAM_IMPL_FOR(_requiredAttributes), PARAM_IMPL_FOR(_uniqueVariables));

private:
  // Parameters derived from generators
  boost::container::flat_map<FastString, NumParamDecl> _derivedShaderParams;
  boost::container::flat_map<FastString, TextureParamDecl> _derivedTextureParams;

  FeaturePtr *_featurePtr{};

  gfx::shader::VariableMap _composedWith;
  SHVar _composedWithHash{};

  static inline CapturingBrancher::IgnoredVariables IgnoredVariables{
      GraphicsRendererContext::VariableName,
      GraphicsContext::VariableName,
  };

  struct SharedData {
    CapturingBrancher viewGeneratorBranch;
    CapturingBrancher drawableGeneratorBranch;
    SHVar externalContextVar{};
    bool warmedUp{};
    std::map<FastString, FastString> globalVariableRemapping;

    ~SharedData() {}

    FastString resolveGlobalVariableName(std::string_view inName) {
      auto it = globalVariableRemapping.find(inName);
      if (it == globalVariableRemapping.end()) {
        return inName;
      } else {
        return it->second;
      }
    }
    FastString resolveGlobalVariableName(const SHVar &inName) { return resolveGlobalVariableName(SHSTRVIEW(inName)); }

    bool haveDrawableGenerators() const { return !drawableGeneratorBranch.brancher.wires.empty(); }
    bool haveViewGenerators() const { return !viewGeneratorBranch.brancher.wires.empty(); }

    void cleanup() {
      viewGeneratorBranch.cleanup();
      drawableGeneratorBranch.cleanup();
    }
  };
  std::shared_ptr<SharedData> _sharedData;
  ExposedInfo _instanceDataCopy;
  CapturingBrancher::VariableRefs _viewGeneratorRefs;
  CapturingBrancher::VariableRefs _drawableGeneratorRefs;

  CapturingBrancher::CloningContext _brancherCloningContext;
  CapturingBrancher::CapturedVariables _drawableGeneratorVariables;
  CapturingBrancher::CapturedVariables _viewGeneratorVariables;

public:
  void cleanup(SHContext *context) {
    // This makes sure all branchers are cleaned up on the correct thread
    if (_sharedData)
      _sharedData->cleanup();
    _sharedData.reset();
    _drawableGeneratorRefs.clear();
    _viewGeneratorRefs.clear();

    PARAM_CLEANUP(context);

    if (_featurePtr) {
      ShardsTypes::FeatureObjectVar.Release(_featurePtr);
      _featurePtr = nullptr;
    }
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    assert(!_featurePtr);
    _featurePtr = ShardsTypes::FeatureObjectVar.New();
    *_featurePtr = std::make_shared<Feature>();

    if (!_sharedData) {
      _sharedData = std::make_shared<SharedData>();
      SHInstanceData data{
          .shared = SHExposedTypesInfo(_instanceDataCopy),
      };
      composeGeneratorWires(data);
    }
    _sharedData->drawableGeneratorBranch.intializeCaptures(_drawableGeneratorRefs, context, IgnoredVariables);
    _sharedData->viewGeneratorBranch.intializeCaptures(_viewGeneratorRefs, context, IgnoredVariables);
  }

  void collectComposeResult(const std::shared_ptr<SHWire> &wire,
                            boost::container::flat_map<FastString, NumParamDecl> &outBasicParams,
                            boost::container::flat_map<FastString, TextureParamDecl> &outTextureParams, BindGroupId bindingFreq,
                            bool expectSeqOutput) {
    auto parseParamTable = [&](const SHTypeInfo &type) {
      for (size_t i = 0; i < type.table.keys.len; i++) {
        auto k = type.table.keys.elements[i];
        auto v = type.table.types.elements[i];

        if (v == ShardsTypes::Buffer) {
          // Ignore buffer bindings at this stage
          continue;
        }

        auto type = deriveShaderFieldType(v);

        if (!type.has_value()) {
          throw formatException("Generator wire returns invalid type {} for key {}", v, k);
        }
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, shader::NumType>) {
                if (k.valueType != SHType::String) {
                  throw formatException("Generator wire returns invalid type {} for key {}", v, k);
                }
                auto &param = outBasicParams[resolveGlobalVariableName(k)] = NumParamDecl(arg);
                param.bindGroupId = bindingFreq;
              } else if constexpr (std::is_same_v<T, shader::TextureType>) {
                if (k.valueType != SHType::String) {
                  throw formatException("Generator wire returns invalid type {} for key {}", v, k);
                }
                auto &param = outTextureParams[resolveGlobalVariableName(k)] = TextureParamDecl(arg);
                param.bindGroupId = bindingFreq;
              } else {
                throw formatException("Generator wire returns invalid type {} for key {}", v, k);
              }
            },
            type.value());
      }
    };

    const auto &outputType = wire->composeResult->outputType;
    if (expectSeqOutput) {
      if (outputType.basicType != SHType::Seq) {
        throw formatException("Feature generator wire should return a sequence of parameter tables (one for each object)");
      }

      for (size_t i = 0; i < outputType.seqTypes.len; i++) {
        auto &tableType = outputType.seqTypes.elements[i];
        if (tableType.basicType != SHType::Table) {
          throw formatException("Feature generator wire should return a sequence of parameter tables (one for each object). "
                                "Element {} ({}) was not a table",
                                tableType, i);
        }
        parseParamTable(tableType);
      }
    } else {
      if (outputType.basicType != SHType::Table) {
        throw formatException("Feature generator wire should return a parameter table");
      }
      parseParamTable(outputType);
    }
  }

  static OwnedVar deepClone(const SHVar &in) {
    static thread_local Serialization ser;
    static thread_local std::vector<uint8_t> buffer;
    BufferRefWriter w{buffer};
    ser.serialize(in, w);
    shards::OwnedVar genCopy;
    BufferRefReader r{buffer};
    ser.deserialize(r, genCopy);
    return genCopy;
  }

  void composeGeneratorWires(const SHInstanceData &data) {
    SHInstanceData generatorInstanceData = data;
    generatorInstanceData.privateContext = nullptr; // force a new context

    bool hasGeneratorWires = !_drawableGenerators->isNone() || !_viewGenerators->isNone();
    if (hasGeneratorWires) {
      _instanceDataCopy = ExposedInfo(data.shared);

      ExposedInfo innerExposed;

      // Expose custom render context
      innerExposed.push_back(RequiredGraphicsRendererContext::getExposedTypeInfo());

      generatorInstanceData.inputType = GeneratedDrawInputTableType;
      _sharedData->drawableGeneratorBranch.brancher.setRunnables(deepClone(_drawableGenerators));
      if (!_sharedData->drawableGeneratorBranch.wires().empty())
        _sharedData->drawableGeneratorBranch.compose(generatorInstanceData, innerExposed, IgnoredVariables);

      generatorInstanceData.inputType = GeneratedViewInputTableType;
      _sharedData->viewGeneratorBranch.brancher.setRunnables(deepClone(_viewGenerators));
      if (!_sharedData->viewGeneratorBranch.wires().empty())
        _sharedData->viewGeneratorBranch.compose(generatorInstanceData, innerExposed, IgnoredVariables);
    }
  }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    static std::atomic_uint64_t uniqueIdCounter;

    _sharedData = std::make_shared<SharedData>();

    if (!_uniqueVariables->isNone()) {
      auto uniqueId = ++uniqueIdCounter;
      for (auto &var : _uniqueVariables) {
        _sharedData->globalVariableRemapping[SHSTRVIEW(var)] = fmt::format("{}_{:08x}", SHSTRVIEW(var), uniqueId);
      }
    }

    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    composeGeneratorWires(data);

    // Collect derived shader parameters from wire outputs
    _derivedShaderParams.clear();
    _derivedTextureParams.clear();
    for (auto &wire : _sharedData->drawableGeneratorBranch.wires()) {
      collectComposeResult(wire, _derivedShaderParams, _derivedTextureParams, BindGroupId::Draw, true);
    }
    for (auto &wire : _sharedData->viewGeneratorBranch.wires()) {
      collectComposeResult(wire, _derivedShaderParams, _derivedTextureParams, BindGroupId::View, false);
    }

    return ShardsTypes::Feature;
  }

  void applyBlendComponent(SHContext *context, BlendComponent &blendComponent, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":Blend Alpha/Color table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar operationVar;
    if (getFromTable(context, inputTable, Var("Operation"), operationVar)) {
      blendComponent.operation = WGPUBlendOperation(operationVar.payload.enumValue);
    } else {
      throw formatException(":Blend table require an :Operation");
    }

    auto applyFactor = [](SHContext *context, WGPUBlendFactor &factor, const SHVar &key, const SHTable &inputTable) {
      SHVar factorVar;
      if (getFromTable(context, inputTable, key, factorVar)) {
        factor = WGPUBlendFactor(factorVar.payload.enumValue);
      } else {
        throw formatException(":Blend table require a :{} factor", key);
      }
    };
    applyFactor(context, blendComponent.srcFactor, Var("Src"), inputTable);
    applyFactor(context, blendComponent.dstFactor, Var("Dst"), inputTable);
  }

  void applyBlendState(SHContext *context, BlendState &blendState, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":State :Blend table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar colorVar;
    if (getFromTable(context, inputTable, Var("Color"), colorVar)) {
      applyBlendComponent(context, blendState.color, colorVar);
    }
    SHVar alphaVar;
    if (getFromTable(context, inputTable, Var("Alpha"), alphaVar)) {
      applyBlendComponent(context, blendState.alpha, alphaVar);
    }
  }

  void applyState(SHContext *context, FeaturePipelineState &state, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":State table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar depthCompareVar;
    if (getFromTable(context, inputTable, Var("DepthCompare"), depthCompareVar)) {
      checkEnumType(depthCompareVar, ShardsTypes::CompareFunctionEnumInfo::Type, ":Shaders DepthCompare");
      state.set_depthCompare(WGPUCompareFunction(depthCompareVar.payload.enumValue));
    }

    SHVar depthWriteVar;
    if (getFromTable(context, inputTable, Var("DepthWrite"), depthWriteVar)) {
      state.set_depthWrite(depthWriteVar.payload.boolValue);
    }

    SHVar colorWriteVar;
    if (getFromTable(context, inputTable, Var("ColorWrite"), colorWriteVar)) {
      WGPUColorWriteMask mask{};
      auto apply = [&mask](SHVar &var) {
        checkEnumType(var, ShardsTypes::ColorMaskEnumInfo::Type, ":ColorWrite");
        (uint8_t &)mask |= WGPUColorWriteMask(var.payload.enumValue);
      };

      // Combine single enum or seq[enum] bitflags into mask
      if (colorWriteVar.valueType == SHType::Enum) {
        apply(colorWriteVar);
      } else if (colorWriteVar.valueType == SHType::Seq) {
        auto &seq = colorWriteVar.payload.seqValue;
        for (size_t i = 0; i < seq.len; i++) {
          apply(seq.elements[i]);
        }
      } else {
        throw formatException(":ColorWrite requires an enum (sequence)");
      }

      state.set_colorWrite(mask);
    }

    SHVar blendVar;
    if (getFromTable(context, inputTable, Var("Blend"), blendVar)) {
      BlendState blendState{
          .color = BlendComponent::Opaque,
          .alpha = BlendComponent::Opaque,
      };
      applyBlendState(context, blendState, blendVar);
      state.set_blend(blendState);
    }

    SHVar flipFrontFaceVar;
    if (getFromTable(context, inputTable, Var("FlipFrontFace"), flipFrontFaceVar)) {
      state.set_flipFrontFace(flipFrontFaceVar.payload.boolValue);
    }

    SHVar cullingVar;
    if (getFromTable(context, inputTable, Var("Culling"), cullingVar)) {
      state.set_culling(cullingVar.payload.boolValue);
    }
  }

  void applyShaderDependency(SHContext *context, shader::EntryPoint &entryPoint, shader::DependencyType type,
                             const SHVar &input) {
    checkType(input.valueType, SHType::String, "Shader dependency");
    shader::NamedDependency &dep = entryPoint.dependencies.emplace_back();
    dep.name = SHSTRVIEW(input);
    dep.type = type;
  }

  void applyShader(SHContext *context, Feature &feature, const SHVar &input) {
    shader::EntryPoint &entryPoint = feature.shaderEntryPoints.emplace_back();

    checkType(input.valueType, SHType::Table, ":Shaders table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar stageVar;
    if (getFromTable(context, inputTable, Var("Stage"), stageVar)) {
      checkEnumType(stageVar, ShardsTypes::ProgrammableGraphicsStageEnumInfo::Type, ":Shaders Stage");
      entryPoint.stage = ProgrammableGraphicsStage(stageVar.payload.enumValue);
    } else
      entryPoint.stage = ProgrammableGraphicsStage::Fragment;

    SHVar depsVar;
    if (getFromTable(context, inputTable, Var("Before"), depsVar)) {
      checkType(depsVar.valueType, SHType::Seq, ":Shaders Dependencies (Before)");
      const SHSeq &seq = depsVar.payload.seqValue;
      for (size_t i = 0; i < seq.len; i++) {
        applyShaderDependency(context, entryPoint, shader::DependencyType::Before, seq.elements[i]);
      }
    }
    if (getFromTable(context, inputTable, Var("After"), depsVar)) {
      checkType(depsVar.valueType, SHType::Seq, ":Shaders Dependencies (After)");
      const SHSeq &seq = depsVar.payload.seqValue;
      for (size_t i = 0; i < seq.len; i++) {
        applyShaderDependency(context, entryPoint, shader::DependencyType::After, seq.elements[i]);
      }
    }

    SHVar nameVar;
    if (getFromTable(context, inputTable, Var("Name"), nameVar)) {
      checkType(nameVar.valueType, SHType::String, ":Shaders Name");
      entryPoint.name = SHSTRVIEW(nameVar);
    }

    SHVar entryPointVar;
    if (getFromTable(context, inputTable, Var("EntryPoint"), entryPointVar)) {
      applyShaderEntryPoint(context, entryPoint, entryPointVar, _composedWith, _sharedData->globalVariableRemapping);
    } else {
      throw formatException(":Shader table requires an :EntryPoint");
    }
  }

  void applyShaders(SHContext *context, Feature &feature, const SHVar &input) {
    checkType(input.valueType, SHType::Seq, ":Shaders param");

    const SHSeq &seq = input.payload.seqValue;
    for (size_t i = 0; i < seq.len; i++) {
      applyShader(context, feature, seq.elements[i]);
    }
  }

  static ShaderParamVariant paramVarToShaderParameter(SHContext *context, SHVar v) {
    SHVar *ref{};
    if (v.valueType == SHType::ContextVar) {
      ref = referenceVariable(context, SHSTRVIEW(v));
      v = *ref;
    }

    auto shaderParam = varToShaderParameter(v);

    if (ref)
      releaseVariable(ref);

    return shaderParam;
  }

  // Returns the field type, or std::monostate if not specified
  std::optional<shader::Type> getShaderFieldType(SHContext *context, const SHTable &inputTable) {
    std::optional<shader::Type> fieldType;

    SHVar typeVar;
    if (getFromTable(context, inputTable, Var("Type"), typeVar)) {
      auto enumType = Type::Enum(typeVar.payload.enumVendorId, typeVar.payload.enumTypeId);
      if (enumType == ShardsTypes::ShaderFieldBaseTypeEnumInfo::Type) {
        checkEnumType(typeVar, ShardsTypes::ShaderFieldBaseTypeEnumInfo::Type, ":Type");
        fieldType = ShaderFieldBaseType(typeVar.payload.enumValue);
      } else if (enumType == ShardsTypes::TextureDimensionEnumInfo::Type) {
        return TextureDimension(typeVar.payload.enumValue);
      } else {
        throw formatException("Invalid Type for shader Param, should be either TextureDimension... or ShaderFieldBaseType...");
      }
    }

    SHVar dimVar;
    if (getFromTable(context, inputTable, Var("Dimension"), dimVar)) {
      checkType(dimVar.valueType, SHType::Int, ":Dimension");
      if (!fieldType)
        fieldType = shader::NumType();
      shader::NumType &NumType = std::get<shader::NumType>(fieldType.value());

      NumType.numComponents = size_t(typeVar.payload.intValue);
    }

    return fieldType;
  }

  void applyParam(SHContext *context, Feature &feature, FastString name, const SHVar &value) {
    SHVar tmp;
    std::optional<ShaderParamVariant> defaultValue;
    std::optional<shader::Type> explicitType;
    std::optional<gfx::BindGroupId> bindGroup;
    std::optional<gfx::TextureSampleType> sampleType;
    if (value.valueType == SHType::Table) {
      const SHTable &inputTable = value.payload.tableValue;

      if (getFromTable(context, inputTable, Var("Default"), tmp)) {
        defaultValue = paramVarToShaderParameter(context, tmp);
      }

      if (getFromTable(context, inputTable, Var("BindGroup"), tmp)) {
        checkEnumType(tmp, ShardsTypes::BindGroupIdEnumInfo::Type, "BindGroup");
        bindGroup = (BindGroupId)tmp.payload.enumValue;
      }

      if (getFromTable(context, inputTable, Var("SampleType"), tmp)) {
        checkEnumType(tmp, ShardsTypes::TextureSampleTypeEnumInfo::Type, "SampleType");
        sampleType = (TextureSampleType)tmp.payload.enumValue;
      }

      explicitType = getShaderFieldType(context, inputTable);
    } else {
      defaultValue = paramVarToShaderParameter(context, value);
    }

    if (!explicitType.has_value()) {
      if (defaultValue.has_value()) {
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, NumParameter>) {
                // Derive field type from given default value
                auto fieldType = getNumParameterType(arg);
                feature.shaderParams[name] = NumParamDecl(fieldType, arg);
              } else if constexpr (std::is_same_v<T, TextureParameter>) {
                TextureParamDecl &param = feature.textureParams[name];
                param.defaultValue = arg.texture;
                param.type.dimension = arg.texture->getFormat().dimension;
                if (bindGroup)
                  param.bindGroupId = bindGroup.value();
                if (sampleType)
                  param.type.format = sampleType.value();
              } else if constexpr (std::is_same_v<T, BufferPtr>) {
                throw formatException("Can not set default parameter for buffer");
              }
            },
            defaultValue.value());
      } else {
        throw formatException("Shader parameter \"{}\" should have a type or default value", name);
      }
    } else {
      std::visit(
          [&](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, shader::NumType>) {
              feature.shaderParams[name] = NumParamDecl(arg);
            } else if constexpr (std::is_same_v<T, shader::TextureType>) {
              auto &param = feature.textureParams[name];
              param = TextureParamDecl(arg);
              if (bindGroup)
                param.bindGroupId = bindGroup.value();
              if (sampleType)
                param.type.format = sampleType.value();
            }
          },
          explicitType.value());
    }
  }

  FastString resolveGlobalVariableName(std::string_view inName) { return _sharedData->resolveGlobalVariableName(inName); }
  FastString resolveGlobalVariableName(const SHVar &inName) { return _sharedData->resolveGlobalVariableName(inName); }

  void applyParams(SHContext *context, Feature &feature, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":Params");
    const SHTable &inputTable = input.payload.tableValue;

    ForEach(inputTable, [&](SHVar key, SHVar v) {
      if (key.valueType != SHType::String)
        throw formatException("Shader parameter key must be a string");
      applyParam(context, feature, resolveGlobalVariableName(key), v);
    });
  }

  void applyBlockParam(SHContext *context, Feature &feature, FastString name, const SHVar &value) {
    checkType(value.valueType, SHType::Table, ":BlockParam");
    TableVar &inputTable = (TableVar &)value;

    shader::AddressSpace addressSpace = shader::AddressSpace::Storage;
    if (auto as = inputTable.find("AddressSpace")) {
      checkEnumType(*as, ShardsTypes::BufferAddressSpaceEnumInfo::Type, fmt::format("Block {} address space", name).c_str());
      addressSpace = (shader::AddressSpace)as->payload.enumValue;
    }

    BindGroupId bindGroup = BindGroupId::View;
    if (auto as = inputTable.find("BindGroup")) {
      checkEnumType(*as, ShardsTypes::BindGroupIdEnumInfo::Type, fmt::format("Block {} bind group id", name).c_str());
      bindGroup = (BindGroupId)as->payload.enumValue;
    }

    auto typeVar = inputTable.find("Type");
    if (!typeVar)
      throw std::runtime_error(fmt::format("Block parameter {} needs a 'type' field", name));
    shader::Type parsedType;
    DerivedLayoutInfo dli = parseStructType(parsedType, *typeVar, addressSpace);

    auto &outParam = feature.blockParams[name];
    outParam.type = parsedType;
    outParam.addressSpace = addressSpace;
    outParam.bindGroupId = bindGroup;
  }

  void applyBlockParams(SHContext *context, Feature &feature, const SHVar &input) {
    checkType(input.valueType, SHType::Table, "BlockParams");
    TableVar &inputTable = (TableVar &)input;
    inputTable.find("space");

    for (auto &[key, v] : inputTable) {
      if (key.valueType != SHType::String)
        throw formatException("Shader parameter key must be a string");
      applyBlockParam(context, feature, resolveGlobalVariableName(key), v);
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (_sharedData->haveDrawableGenerators()) {
      for (auto &vr : _drawableGeneratorRefs) {
        vr.second.cloneInto(_drawableGeneratorVariables[vr.first], _brancherCloningContext);
      }
      _sharedData->drawableGeneratorBranch.applyCapturedVariablesSwap(_drawableGeneratorVariables);
      if (!_sharedData->warmedUp)
        _sharedData->drawableGeneratorBranch.warmup();
    }

    if (_sharedData->haveViewGenerators()) {
      for (auto &vr : _viewGeneratorRefs) {
        vr.second.cloneInto(_viewGeneratorVariables[vr.first], _brancherCloningContext);
      }
      _sharedData->viewGeneratorBranch.applyCapturedVariablesSwap(_viewGeneratorVariables);
      if (!_sharedData->warmedUp)
        _sharedData->viewGeneratorBranch.warmup();
    }
    _sharedData->warmedUp = true;

    if (!_composeWith.isNone()) {
      shader::applyComposeWithHashed(context, _composeWith.get(), _composedWithHash, _composedWith,
                                     _sharedData->globalVariableRemapping, [&]() {
                                       // Create a new object to force shader recompile
                                       *_featurePtr = std::make_shared<Feature>();
                                     });
    }

    Feature &feature = *_featurePtr->get();

    if (!_shaders.isNone()) {
      if (feature.shaderEntryPoints.empty()) {
        // These are static so can be initialized only once
        applyShaders(context, feature, _shaders);
      }
    }

    if (!_state.isNone()) {
      applyState(context, feature.state, _state);
    }

    // Reset to default
    feature.shaderParams = _derivedShaderParams;
    feature.textureParams = _derivedTextureParams;

    if (!_params.isNone()) {
      applyParams(context, feature, _params.get());
    }

    // Apply block parameters (statically, only once)
    if (!_blockParams->isNone() && feature.blockParams.empty()) {
      applyBlockParams(context, feature, *_blockParams);
    }

    feature.generators.clear();

    if (_sharedData->haveDrawableGenerators()) {
      feature.generators.emplace_back([sharedData = _sharedData](FeatureDrawableGeneratorContext &ctx) {
        auto applyResults = [&](FeatureDrawableGeneratorContext &ctx, SHContext *shContext, const SHVar &output) {
          size_t index{};
          ForEach(output.payload.seqValue, [&](SHVar &val) {
            if (index >= ctx.getSize())
              throw formatException("Value returned by drawable generator is out of range");

            auto &collector = ctx.getParameterCollector(index);
            collectParameters(collector, shContext, val.payload.tableValue, sharedData);
            ++index;
          });
        };
        runGenerators<true>(sharedData->drawableGeneratorBranch.brancher, sharedData->externalContextVar, ctx, applyResults);
      });
    }

    if (_sharedData->haveViewGenerators()) {
      feature.generators.emplace_back([sharedData = _sharedData](FeatureViewGeneratorContext &ctx) {
        auto applyResults = [&](FeatureViewGeneratorContext &ctx, SHContext *shContext, const SHVar &output) {
          auto &collector = ctx.getParameterCollector();
          collectParameters(collector, shContext, output.payload.tableValue, sharedData);
        };
        runGenerators<false>(sharedData->viewGeneratorBranch.brancher, sharedData->externalContextVar, ctx, applyResults);
      });
    }

    feature.requiredAttributes = RequiredAttributes();
    if (!_requiredAttributes.isNone()) {
      ForEach(_requiredAttributes.get().payload.seqValue, [&](SHVar &v) {
        RequiredAttributes_ e = (RequiredAttributes_)v.payload.enumValue;
        if (e == RequiredAttributes_::Tangent) {
          feature.requiredAttributes.requirePerVertexLocalBasis = true;
        }
      });
    }

    return ShardsTypes::FeatureObjectVar.Get(_featurePtr);
  }

  static void collectParameters(IParameterCollector &collector, SHContext *context, const SHTable &table,
                                const std::shared_ptr<SharedData> &sharedData) {
    ForEach(table, [&](const SHVar &k, const SHVar &v) {
      if (k.valueType != SHType::String)
        throw formatException("Shader parameter key must be a string");
      auto key = sharedData->resolveGlobalVariableName(k);
      if (v.valueType == SHType::Object) {
        auto objType = shards::Type::Object(v.payload.objectVendorId, v.payload.objectTypeId);
        if (objType == ShardsTypes::Buffer) {
          auto &buffer = *(SHBuffer *)v.payload.objectValue;
          collector.setBlock(key, buffer.buffer);
        } else {
          collector.setTexture(key, varToTexture(v));
        }
      } else {
        auto shaderParam = paramVarToShaderParameter(context, v);
        collector.setParam(key, std::get<NumParameter>(shaderParam));
      }
    });
  }

  template <bool PerDrawable, typename T, typename T1>
  static void runGenerators(Brancher &brancher, SHVar &externalContextVar, T &ctx, T1 applyResults) {
    auto &mesh = brancher.mesh;
    auto &wires = brancher.wires;

    // Avoid running after errors happened
    if (!mesh->failedWires().empty())
      return;

    GraphicsRendererContext graphicsRendererContext{
        .renderer = &ctx.renderer,
        .render = [&ctx = ctx](ViewPtr view, const PipelineSteps &pipelineSteps) { ctx.render(view, pipelineSteps); },
    };

    // Override render context
    static auto swl_context = toSWL(GraphicsRendererContext::VariableName);
    externalContextVar = Var::Object(&graphicsRendererContext, GraphicsRendererContext::Type);
    externalContextVar.flags = SHVAR_FLAGS_EXTERNAL;

    for (auto &wire : wires) {
      auto &externalVariables = wire->getExternalVariables();
      externalVariables[shards::OwnedVar::Foreign(swl_context)] = SHExternalVariable{
          .var = &externalContextVar,
          .type = &GraphicsRendererContext::Type,
      };
    }

    // Setup input table
    SHDrawQueue queue{ctx.queue};
    SHView view{.view = ctx.view};
    TableVar input;
    input.get<Var>(Var("Queue")) = Var::Object(&queue, ShardsTypes::DrawQueue);
    input.get<Var>(Var("View")) = Var::Object(&view, ShardsTypes::View);

    if constexpr (PerDrawable) {
      FeatureDrawableGeneratorContext &drawableCtx = ctx;
      auto &drawablesSeq = input.get<SeqVar>(Var("Drawables"));
      for (size_t i = 0; i < drawableCtx.getSize(); i++) {
        drawablesSeq.push_back(Var(reinterpret_cast<SHInt &>(drawableCtx.getDrawable(i).getId().value)));
      }
    }

    boost::container::small_vector<FeaturePtr, 32> featurePtrsTmp;
    SeqVar &features = input.get<SeqVar>(Var("Features"));
    for (auto &weakFeature : ctx.features) {
      FeaturePtr feature = weakFeature.lock();
      if (feature) {
        featurePtrsTmp.push_back(feature);
      }
    }

    for (auto &tmp : featurePtrsTmp)
      features.push_back(Var::Object(&tmp, ShardsTypes::Feature));

    brancher.updateInputs(input);
    try {
      brancher.activate();

      // Fetch results and insert into parameter collector
      for (auto &wire : wires) {
        if (wire->state == SHWire::State::IterationEnded) {
          if (wire->previousOutput.valueType != SHType::None) {
            applyResults(ctx, wire->context, wire->previousOutput);
          }
        }
      }
    } catch (std::exception &e) {
      SHLOG_ERROR("Error running GFX.Feature generator: {}", e.what());
    }
  }
};

void registerFeatureShards() {
  REGISTER_ENUM(BuiltinFeatureShard::BuiltinFeatureIdEnumInfo);
  REGISTER_ENUM(RequiredAttributesEnumInfo);

  REGISTER_SHARD("GFX.BuiltinFeature", BuiltinFeatureShard);
  REGISTER_SHARD("GFX.Feature", FeatureShard);
}
} // namespace gfx
