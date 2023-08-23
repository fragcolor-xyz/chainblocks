#include "gfx.hpp"
#include "buffer_vars.hpp"
#include "drawable_utils.hpp"
#include "gfx.hpp"
#include "drawable_utils.hpp"
#include "shards_types.hpp"
#include <shards/common_types.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/runtime.hpp>
#include <shards/core/params.hpp>
#include <shards/core/shared.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <shards/core/brancher.hpp>
#include <shards/iterator.hpp>
#include "shader/translator.hpp"
#include "shader/composition.hpp"
#include "shards_utils.hpp"
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
#include <algorithm>
#include <array>
#include <deque>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string.h>
#include <unordered_map>
#include <variant>

using namespace shards;

namespace gfx {
enum class BuiltinFeatureId { Transform, BaseColor, VertexColorFromNormal, Wireframe, Velocity };
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

namespace gfx {
DECL_ENUM_INFO(RequiredAttributes_, RequiredAttributes, 'fatt');

struct BuiltinFeatureShard {
  DECL_ENUM_INFO(BuiltinFeatureId, BuiltinFeatureId, 'feid');

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return Types::Feature; }

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

  void cleanup() {
    if (_feature) {
      Types::FeatureObjectVar.Release(_feature);
      _feature = nullptr;
    }
  }

  void warmup(SHContext *context) {
    _feature = Types::FeatureObjectVar.New();
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
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) { return Types::FeatureObjectVar.Get(_feature); }
};

struct FeatureShard {
  static inline shards::Types GeneratedViewInputTableTypes{{Types::DrawQueue, Types::View, Types::FeatureSeq}};
  static inline std::array<SHVar, 3> GeneratedViewInputTableKeys{Var("Queue"), Var("View"), Var("Features")};
  static inline Type GeneratedViewInputTableType = Type::TableOf(GeneratedViewInputTableTypes, GeneratedViewInputTableKeys);

  static inline shards::Type DrawableDataSeqType = Type::SeqOf(CoreInfo::IntType);
  static inline shards::Types GeneratedDrawInputTableTypes{
      {Types::DrawQueue, Types::View, Types::FeatureSeq, DrawableDataSeqType}};
  static inline std::array<SHVar, 4> GeneratedDrawInputTableKeys{Var("Queue"), Var("View"), Var("Features"), Var("Drawables")};
  static inline Type GeneratedDrawInputTableType = Type::TableOf(GeneratedDrawInputTableTypes, GeneratedDrawInputTableKeys);

  static inline Type ShaderEntryPointType = Type::SeqOf(CoreInfo::AnyTableType);

  static inline shards::Types ParamSpecEntryTypes{Types::ShaderParamOrVarTypes, {CoreInfo::AnyTableType}};
  static inline Type ParameterSpecType = Type::TableOf(ParamSpecEntryTypes);

  static inline Type RequiredAttributesSeqType = Type::SeqOf(RequiredAttributesEnumInfo::Type);

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return Types::Feature; }

  // A shader entry point looks like the following
  //   {:Name <string> :Stage <ProgrammableGraphicsStage> :Before [...] :After [...] :EntryPoint (-> ...)}
  // The Before/After parameters are optional and can be used to specify dependencies
  // Each dependency will make sure that entry point is evaluated before or after the named dependency
  // You can give each entry point a name for this purpose using the Name parameter
  PARAM_PARAMVAR(_shaders, "Shaders", "A list of shader entry points", {ShaderEntryPointType});
  // Any constant undefined variables used in the shader can be injected using this parameter
  PARAM_PARAMVAR(_composeWith, "ComposeWith", "Any table of values that need to be injected into this feature's shaders",
                 {CoreInfo::AnyTableType, CoreInfo::AnyVarTableType});
  // A table, any key used in the applyState function can be used here (you can also check shards\gfx\pipeline_states.def for a
  // list of states) States are applied in the order that the features are references, so features that are added after this one
  // can override this feature's state Any states not in this table will inherit the parent's values
  PARAM_PARAMVAR(_state, "State", "The table of render state flags to override",
                 {CoreInfo::AnyTableType, CoreInfo::AnyVarTableType});
  // [(-> ...)]/(-> ...)/[<wire>]/<wire>
  // Any variables used in the callback will be copied during this shard's activation
  // The output should be a single parameter table
  PARAM_VAR(_viewGenerators, "ViewGenerators",
            "A collection of callbacks that will be run to generate per-view shader parameters during rendering",
            {Brancher::RunnableTypes});
  // [(-> ...)]/(-> ...)/[<wire>]/<wire>
  // Any variables used in the callback will be copied during this shard's activation
  // The output should be a sequence of parameter tables (each with the same type)
  PARAM_VAR(_drawableGenerators, "DrawableGenerators",
            "A collection of callbacks that will be run to generate per-drawable shader parameters during rendering",
            {Brancher::RunnableTypes});
  // Table of shader parameters, can be defined using any of the formats below:
  //   :<name> {:Type <ShaderFieldBaseType> :Dimension <number>}
  //   :<name> {:Default <default>}   (type will be derived from value)
  //   :<name> <default-value>        (type will be derived from value)
  PARAM_PARAMVAR(
      _params, "Params",
      "The parameters to expose to shaders, these default values can later be overriden by materials or drawable Params",
      {CoreInfo::NoneType, ParameterSpecType, Type::VariableOf(ParameterSpecType)});

  PARAM_PARAMVAR(
      _requiredAttributes, "RequiredAttributes",
      "The parameters to expose to shaders, these default values can later be overriden by materials or drawable Params",
      {CoreInfo::NoneType, RequiredAttributesSeqType});

  PARAM_IMPL(PARAM_IMPL_FOR(_shaders), PARAM_IMPL_FOR(_composeWith), PARAM_IMPL_FOR(_state), PARAM_IMPL_FOR(_viewGenerators),
             PARAM_IMPL_FOR(_drawableGenerators), PARAM_IMPL_FOR(_params), PARAM_IMPL_FOR(_requiredAttributes));

private:
  Brancher _viewGeneratorBranch;

  Brancher _drawableGeneratorBranch;

  bool _branchesWarmedUp{};

  // Parameters derived from generators
  std::vector<NamedShaderParam> _derivedShaderParams;
  std::vector<NamedTextureParam> _derivedTextureParams;

  FeaturePtr *_featurePtr{};

  std::list<FeaturePtr> _featurePtrsTemp;

  gfx::shader::VariableMap _composedWith;

public:
  void cleanup() {
    PARAM_CLEANUP();

    _viewGeneratorBranch.cleanup();
    _drawableGeneratorBranch.cleanup();
    _branchesWarmedUp = false;

    if (_featurePtr) {
      Types::FeatureObjectVar.Release(_featurePtr);
      _featurePtr = nullptr;
    }
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);

    _featurePtr = Types::FeatureObjectVar.New();
    *_featurePtr = std::make_shared<Feature>();
  }

  void collectComposeResult(const std::shared_ptr<SHWire> &wire, std::vector<NamedShaderParam> &outBasicParams,
                            std::vector<NamedTextureParam> &outTextureParams, BindGroupId bindingFreq,
                            bool expectSeqOutput) {
    auto parseParamTable = [&](const SHTypeInfo &type) {
      for (size_t i = 0; i < type.table.keys.len; i++) {
        auto k = type.table.keys.elements[i];
        auto v = type.table.types.elements[i];

        auto type = deriveShaderFieldType(v);

        if (!type.has_value()) {
          throw formatException("Generator wire returns invalid type {} for key {}", v, k);
        }
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, shader::NumFieldType>) {
                if (k.valueType != SHType::String) {
                  throw formatException("Generator wire returns invalid type {} for key {}", v, k);
                }
                std::string ks(SHSTRVIEW(k));
                auto &param = outBasicParams.emplace_back(std::move(ks), arg);
                param.bindGroupId = bindingFreq;
              } else if constexpr (std::is_same_v<T, shader::TextureFieldType>) {
                if (k.valueType != SHType::String) {
                  throw formatException("Generator wire returns invalid type {} for key {}", v, k);
                }
                std::string ks(SHSTRVIEW(k));
                auto &param = outTextureParams.emplace_back(std::move(ks), arg);
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

  void composeGeneratorWires(const SHInstanceData &data) {
    SHInstanceData generatorInstanceData{};

    ExposedInfo exposed(data.shared);

    // Expose custom render context
    exposed.push_back(RequiredGraphicsRendererContext::getExposedTypeInfo());

    generatorInstanceData.shared = SHExposedTypesInfo(exposed);

    generatorInstanceData.inputType = GeneratedDrawInputTableType;
    _drawableGeneratorBranch.setRunnables(_drawableGenerators);
    _drawableGeneratorBranch.compose(generatorInstanceData);

    generatorInstanceData.inputType = GeneratedViewInputTableType;
    _viewGeneratorBranch.setRunnables(_viewGenerators);
    _viewGeneratorBranch.compose(generatorInstanceData);

    // Collect derived shader parameters from wire outputs
    _derivedShaderParams.clear();
    _derivedTextureParams.clear();
    for (auto &wire : _viewGeneratorBranch.wires) {
      collectComposeResult(wire, _derivedShaderParams, _derivedTextureParams, BindGroupId::View, false);
    }
    for (auto &wire : _drawableGeneratorBranch.wires) {
      collectComposeResult(wire, _derivedShaderParams, _derivedTextureParams, BindGroupId::Draw, true);
    }
  }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    composeGeneratorWires(data);
    return Types::Feature;
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
      checkEnumType(depthCompareVar, Types::CompareFunctionEnumInfo::Type, ":Shaders DepthCompare");
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
        checkEnumType(var, Types::ColorMaskEnumInfo::Type, ":ColorWrite");
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
      checkEnumType(stageVar, Types::ProgrammableGraphicsStageEnumInfo::Type, ":Shaders Stage");
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
      applyShaderEntryPoint(context, entryPoint, entryPointVar, _composedWith);
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

  static std::variant<ParamVariant, TextureParameter> paramVarToShaderParameter(SHContext *context, SHVar v) {
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
  std::optional<shader::FieldType> getShaderFieldType(SHContext *context, const SHTable &inputTable) {
    std::optional<shader::FieldType> fieldType;

    SHVar typeVar;
    if (getFromTable(context, inputTable, Var("Type"), typeVar)) {
      auto enumType = Type::Enum(typeVar.payload.enumVendorId, typeVar.payload.enumTypeId);
      if (enumType == Types::ShaderFieldBaseTypeEnumInfo::Type) {
        checkEnumType(typeVar, Types::ShaderFieldBaseTypeEnumInfo::Type, ":Type");
        fieldType = ShaderFieldBaseType(typeVar.payload.enumValue);
      } else if (enumType == Types::TextureDimensionEnumInfo::Type) {
        return TextureDimension(typeVar.payload.enumValue);
      } else {
        throw formatException("Invalid Type for shader Param, should be either TextureDimension... or ShaderFieldBaseType...");
      }
    }

    SHVar dimVar;
    if (getFromTable(context, inputTable, Var("Dimension"), dimVar)) {
      checkType(dimVar.valueType, SHType::Int, ":Dimension");
      if (!fieldType)
        fieldType = shader::NumFieldType();
      shader::NumFieldType &numFieldType = std::get<shader::NumFieldType>(fieldType.value());

      numFieldType.numComponents = size_t(typeVar.payload.intValue);
    }

    return fieldType;
  }

  void applyParam(SHContext *context, Feature &feature, const std::string &name, const SHVar &value) {
    SHVar defaultVar;
    std::optional<std::variant<ParamVariant, TextureParameter>> defaultValue;
    std::optional<shader::FieldType> explicitType;
    if (value.valueType == SHType::Table) {
      const SHTable &inputTable = value.payload.tableValue;

      if (getFromTable(context, inputTable, Var("Default"), defaultVar)) {
        defaultValue = paramVarToShaderParameter(context, defaultVar);
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
              if constexpr (std::is_same_v<T, ParamVariant>) {
                // Derive field type from given default value
                auto fieldType = getParamVariantType(arg);
                feature.shaderParams.emplace_back(name, fieldType, arg);
              } else if constexpr (std::is_same_v<T, TextureParameter>) {
                NamedTextureParam &param = feature.textureParams.emplace_back(name);
                param.defaultValue = arg.texture;
                param.type.dimension = arg.texture->getFormat().dimension;
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
            if constexpr (std::is_same_v<T, shader::NumFieldType>) {
              feature.shaderParams.emplace_back(name, arg);
            } else if constexpr (std::is_same_v<T, shader::TextureFieldType>) {
              feature.textureParams.emplace_back(name, arg);
            }
          },
          explicitType.value());
    }
  }

  void applyParams(SHContext *context, Feature &feature, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":Params");
    const SHTable &inputTable = input.payload.tableValue;

    ForEach(inputTable, [&](SHVar key, SHVar v) {
      if (key.valueType != SHType::String)
        throw formatException("Shader parameter key must be a string");
      std::string keyStr(SHSTRVIEW(key));
      applyParam(context, feature, keyStr, v);
    });
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    // Capture variable once during activate
    // activate happens during rendering
    if (!_branchesWarmedUp) {
      _drawableGeneratorBranch.warmup(context);
      _viewGeneratorBranch.warmup(context);
      _branchesWarmedUp = true;
    }

    // NOTE: First check these variables to see if we need to invalidate the feature Id (to break caching)
    if (!_composeWith.isNone()) {
      // Always create a new object to force shader recompile
      *_featurePtr = std::make_shared<Feature>();
      shader::applyComposeWith(context, _composedWith, _composeWith.get());
    }

    Feature &feature = *_featurePtr->get();
    feature.shaderEntryPoints.clear();

    if (!_shaders.isNone()) {
      applyShaders(context, feature, _shaders);
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

    feature.generators.clear();

    if (!_drawableGeneratorBranch.wires.empty()) {
      feature.generators.emplace_back([=](FeatureDrawableGeneratorContext &ctx) {
        auto applyResults = [](FeatureDrawableGeneratorContext &ctx, SHContext *shContext, const SHVar &output) {
          size_t index{};
          ForEach(output.payload.seqValue, [&](SHVar &val) {
            if (index >= ctx.getSize())
              throw formatException("Value returned by drawable generator is out of range");

            auto &collector = ctx.getParameterCollector(index);
            collectParameters(collector, shContext, val.payload.tableValue);
            ++index;
          });
        };
        runGenerators<true>(_drawableGeneratorBranch, ctx, applyResults);
      });
    }

    if (!_viewGeneratorBranch.wires.empty()) {
      feature.generators.emplace_back([=](FeatureViewGeneratorContext &ctx) {
        auto applyResults = [](FeatureViewGeneratorContext &ctx, SHContext *shContext, const SHVar &output) {
          auto &collector = ctx.getParameterCollector();
          collectParameters(collector, shContext, output.payload.tableValue);
        };
        runGenerators<false>(_viewGeneratorBranch, ctx, applyResults);
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

    return Types::FeatureObjectVar.Get(_featurePtr);
  }

  static void collectParameters(IParameterCollector &collector, SHContext *context, const SHTable &table) {
    ForEach(table, [&](const SHVar &k, const SHVar &v) {
      if (k.valueType != SHType::String)
        throw formatException("Shader parameter key must be a string");
      std::string keyStr(SHSTRVIEW(k));
      if (v.valueType == SHType::Object) {
        collector.setTexture(keyStr, varToTexture(v));
      } else {
        auto shaderParam = paramVarToShaderParameter(context, v);
        collector.setParam(keyStr, std::get<ParamVariant>(shaderParam));
      }
    });
  }

  template <bool PerDrawable, typename T, typename T1> void runGenerators(Brancher &brancher, T &ctx, T1 applyResults) {
    auto &mesh = brancher.mesh;
    auto &wires = brancher.wires;

    // Avoid running after errors happened
    if (!mesh->failedWires().empty())
      return;

    GraphicsRendererContext graphicsRendererContext{
        .renderer = &ctx.renderer,
        .render = [&ctx = ctx](ViewPtr view, const PipelineSteps &pipelineSteps) { ctx.render(view, pipelineSteps); },
    };
    mesh->variables.emplace(GraphicsRendererContext::VariableName,
                            Var::Object(&graphicsRendererContext, GraphicsRendererContext::Type));

    // Setup input table
    SHDrawQueue queue{ctx.queue};
    SHView view{.view = ctx.view};
    TableVar input;
    input.get<Var>(Var("Queue")) = Var::Object(&queue, Types::DrawQueue);
    input.get<Var>(Var("View")) = Var::Object(&view, Types::View);

    if constexpr (PerDrawable) {
      FeatureDrawableGeneratorContext &drawableCtx = ctx;
      auto &drawablesSeq = input.get<SeqVar>(Var("Drawables"));
      for (size_t i = 0; i < drawableCtx.getSize(); i++) {
        drawablesSeq.push_back(Var(reinterpret_cast<SHInt &>(drawableCtx.getDrawable(i).getId().value)));
      }
    }

    _featurePtrsTemp.clear();
    SeqVar &features = input.get<SeqVar>(Var("Features"));
    for (auto &weakFeature : ctx.features) {
      FeaturePtr feature = weakFeature.lock();
      if (feature) {
        _featurePtrsTemp.push_back(feature);
        features.push_back(Var::Object(&_featurePtrsTemp.back(), Types::Feature));
      }
    }

    brancher.updateInputs(input);
    brancher.activate();

    // Fetch results and insert into parameter collector
    for (auto &wire : wires) {
      if (wire->previousOutput.valueType != SHType::None) {
        applyResults(ctx, wire->context, wire->previousOutput);
      }
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
