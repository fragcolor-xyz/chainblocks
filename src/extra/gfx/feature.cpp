#include "../gfx.hpp"
#include "buffer_vars.hpp"
#include "common_types.hpp"
#include "drawable_utils.hpp"
#include "extra/gfx.hpp"
#include "extra/gfx/shards_types.hpp"
#include "foundation.hpp"
#include "gfx/error_utils.hpp"
#include "gfx/feature.hpp"
#include "gfx/params.hpp"
#include "gfx/shader/types.hpp"
#include "gfx/texture.hpp"
#include "runtime.hpp"
#include "shader/translator.hpp"
#include "shards.h"
#include "shards.hpp"
#include "shards_utils.hpp"
#include <array>
#include <deque>
#include <gfx/context.hpp>
#include <gfx/features/base_color.hpp>
#include <gfx/features/debug_color.hpp>
#include <gfx/features/transform.hpp>
#include <gfx/features/wireframe.hpp>
#include <gfx/features/velocity.hpp>
#include <linalg_shim.hpp>
#include <magic_enum.hpp>
#include <memory>
#include <queue>
#include <shards/shared.hpp>
#include <stdexcept>
#include <variant>
#include <webgpu-headers/webgpu.h>

using namespace shards;

namespace gfx {
enum class BuiltinFeatureId { Transform, BaseColor, VertexColorFromNormal, Wireframe, Velocity };
}

ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::Transform, SHCCSTR("Add basic world/view/projection transform"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::BaseColor,
          SHCCSTR("Add basic color from vertex color and (optional) color texture"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::VertexColorFromNormal, SHCCSTR("Outputs color from vertex color"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::Wireframe, SHCCSTR("Modifies the main color to visualize vertex edges"));
ENUM_HELP(gfx::BuiltinFeatureId, gfx::BuiltinFeatureId::Velocity,
          SHCCSTR("Outputs object velocity into the velocity global & output"));

namespace gfx {
using shards::Mat4;

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
  /* Input table
    {
      :Shaders [
        {:Name <string> :Stage <enum> :Before [...] :After [...] :EntryPoint (-> ...)}
      ]
      :State {
        :Blend {...}
      }
      :DrawData [(-> ...)]/(-> ...)
      :Params [
        {:Name <string> :Type <type> :Dimension <number>}
      ]
    }
  */

  static inline shards::Types GeneratedInputTableTypes{{Types::DrawQueue, Types::View, Types::FeatureSeq}};
  static inline std::array<SHString, 3> GeneratedInputTableKeys{"Queue", "View", "Features"};
  static inline Type GeneratedInputTableType = Type::TableOf(GeneratedInputTableTypes, GeneratedInputTableKeys);

  static SHTypesInfo inputTypes() { return CoreInfo::AnyTableType; }
  static SHTypesInfo outputTypes() { return Types::Feature; }

  static inline shards::Types GeneratorTypes{CoreInfo::NoneType, CoreInfo::WireType, Type::SeqOf(CoreInfo::WireType),
                                             CoreInfo::ShardRefSeqType, Type::SeqOf(CoreInfo::ShardRefSeqType)};
  static SHParametersInfo parameters() {
    static Parameters p{
        {"ViewGenerators",
         SHCCSTR(
             "The shards that are run to generate and return shader parameters per view, can use nested rendering commmands."),
         GeneratorTypes},
        {"DrawableGenerators",
         SHCCSTR("The shards that are run to generate and return shader parameters per drawable, can use nested rendering "
                 "commmands."),
         GeneratorTypes},
    };
    return p;
  }

  std::vector<std::shared_ptr<SHWire>> _viewGenerators;
  std::vector<std::shared_ptr<SHWire>> _drawableGenerators;
  OwnedVar _viewGeneratorsRaw;
  OwnedVar _drawableGeneratorsRaw;

  // Parameters derived from generators
  std::vector<NamedShaderParam> _derivedShaderParams;
  std::vector<NamedTextureParam> _derivedTextureParams;

  FeaturePtr *_featurePtr{};
  SHVar _variable{};

  std::shared_ptr<SHMesh> _viewGeneratorsMesh;
  std::shared_ptr<SHMesh> _drawableGeneratorsMesh;

  void setWireVector(const SHVar &var, std::vector<std::shared_ptr<SHWire>> &outVec) {

    auto createWireFromShards = [](const SHSeq &seq) {
      auto wire = SHWire::make();
      wire->looped = true;
      ForEach(seq, [&](SHVar &v) {
        assert(v.valueType == SHType::ShardRef);
        wire->addShard(v.payload.shardValue);
      });
      return wire;
    };

    outVec.clear();
    if (var.valueType != SHType::None) {
      if (var.valueType == SHType::Wire) {
        outVec.emplace_back(*(std::shared_ptr<SHWire> *)var.payload.wireValue);
      } else {
        assert(var.valueType == SHType::Seq);
        auto &seq = var.payload.seqValue;
        if (seq.len == 0)
          return;

        // Single collection of shards
        if (seq.elements[0].valueType == SHType::ShardRef) {
          outVec.push_back(createWireFromShards(seq));
        } else if (seq.elements[0].valueType == SHType::Wire) {
          // Multiple wires
          for (uint32_t i = 0; i < seq.len; i++) {
            outVec.emplace_back(*(std::shared_ptr<SHWire> *)seq.elements[i].payload.wireValue);
          }
        } else {

          for (uint32_t i = 0; i < seq.len; i++) {
            SHVar &v = seq.elements[i];
            assert(v.valueType == SHType::Seq);
            outVec.push_back(createWireFromShards(v.payload.seqValue));
          }
        }
      }
    }
  }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _viewGeneratorsRaw = value;
      setWireVector(_viewGeneratorsRaw, _viewGenerators);
      break;
    default:
      _drawableGeneratorsRaw = value;
      setWireVector(_drawableGeneratorsRaw, _drawableGenerators);
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _viewGeneratorsRaw;
    case 1:
      return _drawableGeneratorsRaw;
    default:
      return Var::Empty;
    }
  }

  void cleanup() {
    _viewGeneratorsMesh.reset();
    _drawableGeneratorsMesh.reset();

    if (_featurePtr) {
      Types::FeatureObjectVar.Release(_featurePtr);
      _featurePtr = nullptr;
    }
  }

  bool hasAnyGeneratorWires() const { return !_viewGenerators.empty() || !_drawableGenerators.empty(); }

  void warmup(SHContext *context) {
    _featurePtr = Types::FeatureObjectVar.New();
    *_featurePtr = std::make_shared<Feature>();

    if (!_viewGenerators.empty())
      _viewGeneratorsMesh = SHMesh::make();
    if (!_drawableGenerators.empty())
      _drawableGeneratorsMesh = SHMesh::make();
  }

  void composeGeneratorWire(std::shared_ptr<SHWire> &wire, std::vector<NamedShaderParam> &outBasicParams,
                            std::vector<NamedTextureParam> &outTextureParams, bool expectSeqOutput) {
    SHInstanceData generatorInstanceData{};
    generatorInstanceData.inputType = GeneratedInputTableType;

    ExposedInfo exposed;
    exposed.push_back(RequiredGraphicsRendererContext::getExposedTypeInfo());
    generatorInstanceData.shared = SHExposedTypesInfo(exposed);

    generatorInstanceData.wire = wire.get();
    wire->composeResult = composeWire(
        wire->shards,
        [](const struct Shard *errorShard, SHString errorTxt, SHBool nonfatalWarning, void *userData) {
          if (!nonfatalWarning) {
            auto msg = "Feature: failed inner wire validation, error: " + std::string(errorTxt);
            throw shards::SHException(msg);
          } else {
            SHLOG_INFO("Feature: warning during inner wire validation: {}", errorTxt);
          }
        },
        nullptr, generatorInstanceData);

    auto parseParamTable = [&](const SHTypeInfo &type) {
      for (size_t i = 0; i < type.table.keys.len; i++) {
        auto k = type.table.keys.elements[i];
        auto v = type.table.types.elements[i];

        auto type = toShaderParamType(v);
        std::visit(
            [&](auto &&arg) {
              using T = std::decay_t<decltype(arg)>;
              if constexpr (std::is_same_v<T, shader::FieldType>) {
                outBasicParams.emplace_back(k, arg);
              } else if constexpr (std::is_same_v<T, TextureType>) {
                outTextureParams.emplace_back(k, arg);
              } else {
                throw formatException("Generator wire returns invalid type {} for key {}", v, k);
              }
            },
            type);
      }
    };

    const auto &outputType = wire->composeResult->outputType;
    if (expectSeqOutput) {
      if (outputType.basicType != SHType::Seq) {
        throw formatException("Feature generator wire should return a sequence of parameter tables (one for each object)");
      }

      for (size_t i = 0; i < outputType.seqTypes.len; i++) {
        auto &tableType = outputType.seqTypes.elements[i];
        if (outputType.basicType != SHType::Table) {
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

  SHTypeInfo compose(const SHInstanceData &data) {
    _derivedShaderParams.clear();
    _derivedTextureParams.clear();

    // Compose generator wires
    for (auto &generatorWire : _viewGenerators) {
      composeGeneratorWire(generatorWire, _derivedShaderParams, _derivedTextureParams, false);
    }
    for (auto &generatorWire : _drawableGenerators) {
      composeGeneratorWire(generatorWire, _derivedShaderParams, _derivedTextureParams, true);
    }

    return Types::Feature;
  }

  void applyBlendComponent(SHContext *context, BlendComponent &blendComponent, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":Blend Alpha/Color table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar operationVar;
    if (getFromTable(context, inputTable, "Operation", operationVar)) {
      blendComponent.operation = WGPUBlendOperation(operationVar.payload.enumValue);
    } else {
      throw formatException(":Blend table require an :Operation");
    }

    auto applyFactor = [](SHContext *context, WGPUBlendFactor &factor, const char *key, const SHTable &inputTable) {
      SHVar factorVar;
      if (getFromTable(context, inputTable, key, factorVar)) {
        factor = WGPUBlendFactor(factorVar.payload.enumValue);
      } else {
        throw formatException(":Blend table require a :{} factor", key);
      }
    };
    applyFactor(context, blendComponent.srcFactor, "Src", inputTable);
    applyFactor(context, blendComponent.dstFactor, "Dst", inputTable);
  }

  void applyBlendState(SHContext *context, BlendState &blendState, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":State :Blend table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar colorVar;
    if (getFromTable(context, inputTable, "Color", colorVar)) {
      applyBlendComponent(context, blendState.color, colorVar);
    }
    SHVar alphaVar;
    if (getFromTable(context, inputTable, "Alpha", alphaVar)) {
      applyBlendComponent(context, blendState.alpha, alphaVar);
    }
  }

  void applyState(SHContext *context, FeaturePipelineState &state, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":State table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar depthCompareVar;
    if (getFromTable(context, inputTable, "DepthCompare", depthCompareVar)) {
      checkEnumType(depthCompareVar, Types::CompareFunctionEnumInfo::Type, ":Shaders DepthCompare");
      state.set_depthCompare(WGPUCompareFunction(depthCompareVar.payload.enumValue));
    }

    SHVar depthWriteVar;
    if (getFromTable(context, inputTable, "DepthWrite", depthWriteVar)) {
      state.set_depthWrite(depthWriteVar.payload.boolValue);
    }

    SHVar colorWriteVar;
    if (getFromTable(context, inputTable, "ColorWrite", colorWriteVar)) {
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
    if (getFromTable(context, inputTable, "Blend", blendVar)) {
      BlendState blendState{
          .color = BlendComponent::Opaque,
          .alpha = BlendComponent::Opaque,
      };
      applyBlendState(context, blendState, blendVar);
      state.set_blend(blendState);
    }

    SHVar flipFrontFaceVar;
    if (getFromTable(context, inputTable, "FlipFrontFace", flipFrontFaceVar)) {
      state.set_flipFrontFace(flipFrontFaceVar.payload.boolValue);
    }

    SHVar cullingVar;
    if (getFromTable(context, inputTable, "Culling", cullingVar)) {
      state.set_culling(cullingVar.payload.boolValue);
    }
  }

  void applyShaderDependency(SHContext *context, shader::EntryPoint &entryPoint, shader::DependencyType type,
                             const SHVar &input) {
    checkType(input.valueType, SHType::String, "Shader dependency");
    const SHString &inputString = input.payload.stringValue;

    shader::NamedDependency &dep = entryPoint.dependencies.emplace_back();
    dep.name = inputString;
    dep.type = type;
  }

  void applyShader(SHContext *context, Feature &feature, const SHVar &input) {
    shader::EntryPoint &entryPoint = feature.shaderEntryPoints.emplace_back();

    checkType(input.valueType, SHType::Table, ":Shaders table");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar stageVar;
    if (getFromTable(context, inputTable, "Stage", stageVar)) {
      checkEnumType(stageVar, Types::ProgrammableGraphicsStageEnumInfo::Type, ":Shaders Stage");
      entryPoint.stage = ProgrammableGraphicsStage(stageVar.payload.enumValue);
    } else
      entryPoint.stage = ProgrammableGraphicsStage::Fragment;

    SHVar depsVar;
    if (getFromTable(context, inputTable, "Before", depsVar)) {
      checkType(depsVar.valueType, SHType::Seq, ":Shaders Dependencies (Before)");
      const SHSeq &seq = depsVar.payload.seqValue;
      for (size_t i = 0; i < seq.len; i++) {
        applyShaderDependency(context, entryPoint, shader::DependencyType::Before, seq.elements[i]);
      }
    }
    if (getFromTable(context, inputTable, "After", depsVar)) {
      checkType(depsVar.valueType, SHType::Seq, ":Shaders Dependencies (After)");
      const SHSeq &seq = depsVar.payload.seqValue;
      for (size_t i = 0; i < seq.len; i++) {
        applyShaderDependency(context, entryPoint, shader::DependencyType::After, seq.elements[i]);
      }
    }

    SHVar nameVar;
    if (getFromTable(context, inputTable, "Name", nameVar)) {
      checkType(nameVar.valueType, SHType::String, ":Shaders Name");
      entryPoint.name = nameVar.payload.stringValue;
    }

    SHVar entryPointVar;
    if (getFromTable(context, inputTable, "EntryPoint", entryPointVar)) {
      applyShaderEntryPoint(context, entryPoint, entryPointVar);
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

  static ParamVariant paramToVariant(SHContext *context, SHVar v) {
    SHVar *ref{};
    if (v.valueType == SHType::ContextVar) {
      ref = referenceVariable(context, v.payload.stringValue);
      v = *ref;
    }

    ParamVariant variant;
    varToParam(v, variant);

    if (ref)
      releaseVariable(ref);

    return variant;
  }

  // Returns the field type, or std::monostate if not specified
  ShaderFieldTypeVariant getShaderFieldType(SHContext *context, const SHTable &inputTable) {
    shader::FieldType fieldType;
    bool isFieldTypeSet = false;

    SHVar typeVar;
    if (getFromTable(context, inputTable, "Type", typeVar)) {
      auto enumType = Type::Enum(typeVar.payload.enumVendorId, typeVar.payload.enumTypeId);
      if (enumType == Types::ShaderFieldBaseTypeEnumInfo::Type) {
        checkEnumType(typeVar, Types::ShaderFieldBaseTypeEnumInfo::Type, ":Type");
        fieldType.baseType = ShaderFieldBaseType(typeVar.payload.enumValue);
        isFieldTypeSet = true;
      } else if (enumType == Types::TextureDimensionEnumInfo::Type) {
        return TextureType(typeVar.payload.enumValue);
      } else {
        throw formatException("Invalid Type for shader Param, should be either TextureDimension... or ShaderFieldBaseType...");
      }
    } else {
      // Default type if not specified:
      fieldType.baseType = ShaderFieldBaseType::Float32;
    }

    SHVar dimVar;
    if (getFromTable(context, inputTable, "Dimension", dimVar)) {
      checkType(dimVar.valueType, SHType::Int, ":Dimension");
      fieldType.numComponents = size_t(typeVar.payload.intValue);
      isFieldTypeSet = true;
    } else {
      // Default size if not specified:
      fieldType.numComponents = 1;
    }

    if (isFieldTypeSet)
      return fieldType;

    return std::monostate();
  }

  void applyParam(SHContext *context, Feature &feature, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":Params Entry");
    const SHTable &inputTable = input.payload.tableValue;

    SHVar nameVar;
    SHString name{};
    if (getFromTable(context, inputTable, "Name", nameVar)) {
      checkType(nameVar.valueType, SHType::String, ":Params Name");
      name = nameVar.payload.stringValue;
    } else {
      throw formatException(":Params Entry requires a :Name");
    }

    SHVar defaultVar;
    ParamVariant defaultValue;
    if (getFromTable(context, inputTable, "Default", defaultVar)) {
      defaultValue = paramToVariant(context, defaultVar);
    }

    ShaderFieldTypeVariant typeVariant = getShaderFieldType(context, inputTable);
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, shader::FieldType>) {
            feature.shaderParams.emplace_back(name, arg);
          } else if constexpr (std::is_same_v<T, TextureType>) {
            feature.textureParams.emplace_back(name);
          } else {
            if (defaultValue.index() > 0) {
              // Derive field type from given default value
              auto fieldType = getParamVariantType(defaultValue);
              feature.shaderParams.emplace_back(name, fieldType, defaultValue);
            } else {
              throw formatException("Shader parameter \"{}\" should have a type or default value", name);
            }
          }
        },
        typeVariant);
  }

  void applyParams(SHContext *context, Feature &feature, const SHVar &input) {
    checkType(input.valueType, SHType::Seq, ":Params");
    const SHSeq &inputSeq = input.payload.seqValue;

    ForEach(inputSeq, [&](SHVar v) { applyParam(context, feature, v); });
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    Feature &feature = *_featurePtr->get();

    checkType(input.valueType, SHType::Table, "Input table");
    const SHTable &inputTable = input.payload.tableValue;

    feature.shaderEntryPoints.clear();

    SHVar shadersVar;
    if (getFromTable(context, inputTable, "Shaders", shadersVar))
      applyShaders(context, feature, shadersVar);

    SHVar stateVar;
    if (getFromTable(context, inputTable, "State", stateVar))
      applyState(context, feature.state, stateVar);

    // Reset to default
    feature.shaderParams = _derivedShaderParams;
    feature.textureParams = _derivedTextureParams;

    SHVar paramsVar;
    if (getFromTable(context, inputTable, "Params", paramsVar))
      applyParams(context, feature, paramsVar);

    feature.generators.clear();

    if (!_drawableGenerators.empty()) {
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
        runGenerators(_drawableGeneratorsMesh, _drawableGenerators, ctx, applyResults);
      });
    }

    if (!_viewGenerators.empty()) {
      feature.generators.emplace_back([=](FeatureViewGeneratorContext &ctx) {
        auto applyResults = [](FeatureViewGeneratorContext &ctx, SHContext *shContext, const SHVar &output) {
          auto &collector = ctx.getParameterCollector();
          collectParameters(collector, shContext, output.payload.tableValue);
        };
        runGenerators(_viewGeneratorsMesh, _viewGenerators, ctx, applyResults);
      });
    }

    return Types::FeatureObjectVar.Get(_featurePtr);
  }

  static void collectParameters(IParameterCollector &collector, SHContext *context, const SHTable &table) {
    ForEach(table, [&](const SHString &k, const SHVar &v) {
      if (v.valueType == SHType::Object) {
        collector.setTexture(k, varToTexture(v));
      } else {
        ParamVariant variant = paramToVariant(context, v);
        collector.setParam(k, variant);
      }
    });
  }

  std::list<FeaturePtr> featurePtrsTemp;

  template <typename T, typename T1>
  void runGenerators(std::shared_ptr<SHMesh> mesh, const std::vector<std::shared_ptr<SHWire>> &wires, T &ctx, T1 applyResults) {
    GraphicsRendererContext graphicsRendererContext{
        .renderer = &ctx.renderer,
        .render = [&ctx = ctx](std::vector<ViewPtr> views,
                               const PipelineSteps &pipelineSteps) { ctx.render(views, pipelineSteps); },
    };
    mesh->variables.emplace(GraphicsRendererContext::VariableName,
                            Var::Object(&graphicsRendererContext, GraphicsRendererContext::Type));

    // Setup input table
    SHDrawQueue queue{ctx.queue};
    SHView view{.view = ctx.view};
    TableVar input;
    input.get<Var>("Queue") = Var::Object(&queue, Types::DrawQueue);
    input.get<Var>("View") = Var::Object(&view, Types::View);

    featurePtrsTemp.clear();
    SeqVar &features = input.get<SeqVar>("Features");
    for (auto &weakFeature : ctx.features) {
      FeaturePtr feature = weakFeature.lock();
      if (feature) {
        featurePtrsTemp.push_back(feature);
        features.push_back(Var::Object(&featurePtrsTemp.back(), Types::Feature));
      }
    }

    // Schedule generator wires or update inputs
    for (auto &wire : wires) {
      if (!mesh->scheduled.contains(wire)) {
        mesh->schedule(wire, input, false);
      } else {
        // Update inputs
        (TableVar &)wire->currentInput = std::move(input);
      }
    }

    // Run one tick of the generator wires
    if (!mesh->tick())
      throw formatException("Generator tick failed");

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

  REGISTER_SHARD("GFX.BuiltinFeature", BuiltinFeatureShard);
  REGISTER_SHARD("GFX.Feature", FeatureShard);
}
} // namespace gfx
