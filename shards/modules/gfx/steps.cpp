#include "gfx.hpp"
#include "drawable_utils.hpp"
#include "param_types.hpp"
#include "shader/composition.hpp"
#include "shards_utils.hpp"
#include "shader/translator.hpp"
#include <shards/linalg_shim.hpp>
#include <shards/iterator.hpp>
#include <gfx/error_utils.hpp>
#include <gfx/fwd.hpp>
#include <gfx/hasherxxh3.hpp>
#include <gfx/unique_id.hpp>
#include <gfx/feature.hpp>
#include <gfx/pipeline_step.hpp>
#include <gfx/steps/defaults.hpp>
#include <gfx/steps/effect.hpp>
#include <gfx/features/transform.hpp>
#include <gfx/texture.hpp>
#include <gfx/gfx_wgpu.hpp>
#include <gfx/steps/effect.hpp>
#include <magic_enum.hpp>
#include <shards/core/params.hpp>
#include <optional>

using namespace shards;

namespace gfx {
namespace shared {

WGPUTextureFormat getAttachmentFormat(const std::string &name, const SHVar &formatVar) {
  if (formatVar.valueType == SHType::None) {
    if (name == "color") {
      return WGPUTextureFormat_BGRA8UnormSrgb;
    } else if (name == "depth") {
      return WGPUTextureFormat_Depth32Float;
    }
  } else {
    checkEnumType(formatVar, ShardsTypes::TextureFormatEnumInfo::Type, "Attachment format");
    return WGPUTextureFormat(formatVar.payload.enumValue);
  }

  throw std::runtime_error("Output :Format required");
}

static inline const SHVar &getDefaultOutputScale() {
  // The default output scale is to scale to the main output
  // set using {main: None}
  static TableVar table = []() {
    TableVar table;
    table["main"] = Var();
    return table;
  }();
  return table;
}

void applyInputs(SHContext *context, RenderStepInput &rsInput, const SHVar &input) {
  checkType(input.valueType, SHType::Seq, ":Inputs");

  rsInput.attachments.clear();
  for (size_t i = 0; i < input.payload.seqValue.len; i++) {
    auto &elem = input.payload.seqValue.elements[i];
    checkType(elem.valueType, SHType::String, "Input");
    rsInput.attachments.emplace_back(RenderStepInput::Named(std::string(SHSTRVIEW(elem))));
  }
}

template <typename T> void applyInputsToStep(SHContext *context, T &step, const SHVar &input) {
  RenderStepInput &rsInput = step.input;
  applyInputs(context, rsInput, input);
}

void applyOutputs(SHContext *context, RenderStepOutput &output, const SHVar &input) {
  checkType(input.valueType, SHType::Seq, "Outputs sequence");

  output.attachments.clear();
  for (size_t i = 0; i < input.payload.seqValue.len; i++) {
    auto &elem = input.payload.seqValue.elements[i];
    checkType(elem.valueType, SHType::Table, "Output");

    auto &table = elem.payload.tableValue;

    SHVar nameVar;
    if (!getFromTable(context, table, Var("Name"), nameVar)) {
      throw std::runtime_error("Output :Name is required");
    }

    SHVar formatVar{};
    bool hasFormat = getFromTable(context, table, Var("Format"), formatVar);

    SHVar textureVar;
    bool hasTexture = getFromTable(context, table, Var("Texture"), textureVar);

    if (hasTexture && hasFormat)
      throw std::runtime_error(":Format and :Texture are exclusive");

    std::string attachmentName(SHSTRVIEW(nameVar));
    WGPUTextureFormat textureFormat{};
    TexturePtr texture;
    if (hasTexture) {
      texture = varAsObjectChecked<TexturePtr>(textureVar, ShardsTypes::Texture);
      textureFormat = texture->getFormat().pixelFormat;
    } else {
      textureFormat = getAttachmentFormat(attachmentName, formatVar);
    }

    TextureFormatDesc textureFormatDesc = getTextureFormatDescription(textureFormat);

    std::optional<ClearValues> clearValues;

    SHVar clearValueVar;
    if (getFromTable(context, table, Var("Clear"), clearValueVar)) {
      if (clearValueVar.valueType == SHType::Bool) {
        // Clear with default values
        if (clearValueVar.payload.boolValue) {
          // Set clear value based on texture usage
          bool isDepthStencil =
              textureFormatUsageContains(textureFormatDesc.usage, TextureFormatUsage::Depth | TextureFormatUsage::Stencil);
          clearValues = isDepthStencil ? ClearValues::getDefaultDepthStencil() : ClearValues::getDefaultColor();
        }
      } else {
        checkType(clearValueVar.valueType, SHType::Float4, "Clear");
        clearValues = ClearValues::getColorValue(toVec<float4>(clearValueVar));
      }
    } else {
      float depth = 1.0f;
      uint32_t stencil = 0;
      bool anySet = false;
      if (getFromTable(context, table, Var("ClearDepth"), clearValueVar)) {
        depth = toVec<float>(clearValueVar);
        anySet = true;
      }
      if (getFromTable(context, table, Var("ClearStencil"), clearValueVar)) {
        checkType(clearValueVar.valueType, SHType::Int, "ClearStencil");
        stencil = clearValueVar.payload.intValue;
        anySet = true;
      }
      if (anySet) {
        clearValues = ClearValues::getDepthStencilValue(depth, stencil);
      }
    }

    if (hasTexture) {
      const TexturePtr &texture = varAsObjectChecked<TexturePtr>(textureVar, ShardsTypes::Texture);
      auto textureFormatFlags = texture->getFormat().flags;
      if (!textureFormatFlagsContains(textureFormatFlags, TextureFormatFlags::RenderAttachment)) {
        throw std::runtime_error("Invalid output texture, it wasn't created as a render target");
      }

      output.attachments.emplace_back(RenderStepOutput::Texture(std::string(SHSTRVIEW(nameVar)), texture, clearValues));
    } else {
      output.attachments.emplace_back(RenderStepOutput::Named(attachmentName, textureFormat, clearValues));
    }
  }
}

template <typename T> void applyOutputsToStep(SHContext *context, T &step, const SHVar &input) {
  RenderStepOutput &output = (step.output = RenderStepOutput{}).value();
  applyOutputs(context, output, input);
}

std::optional<float2> toOutputScale(const SHVar &input) {
  if (input.valueType == SHType::Float) {
    return float2(input.payload.floatValue);
  } else if (input.valueType == SHType::Float2) {
    return toVec<float2>(input);
  } else if (input.valueType == SHType::None) {
    return std::nullopt;
  } else {
    throw formatException("Output scale must be a float or float2, got {} ({})", input.valueType, input);
  }
}

void applyOutputScale(SHContext *context, RenderStepOutput::OutputSizing &sizing, const SHVar &input) {
  ParamVar v(input);
  v.warmup(context);

  if (input.valueType == SHType::Table) {
    SHTable &table = v.get().payload.tableValue;
    Var mainVar;
    if (getFromTable(context, table, Var("main"), mainVar)) {
      sizing = RenderStepOutput::RelativeToMainSize{.scale = toOutputScale(mainVar)};
    } else {
      std::optional<float2> scaleOpt;
      Var scaleVar;
      if (getFromTable(context, table, Var("scale"), scaleVar)) {
        scaleOpt = toOutputScale(scaleVar);
      }

      Var inputVar;
      std::optional<std::string> inputName;
      if (getFromTable(context, table, Var("name"), inputVar)) {
        checkType(inputVar.valueType, SHType::String, "OutputScale input name");
        inputName = SHSTRVIEW(inputVar);
      }

      sizing = RenderStepOutput::RelativeToInputSize{.name = std::move(inputName), .scale = scaleOpt};
    }
  } else {
    sizing = RenderStepOutput::RelativeToMainSize{.scale = toOutputScale(v.get())};
  }
}

template <typename T> void applyOutputScaleToStep(SHContext *context, T &step, const SHVar &input) {
  if (!step.output) {
    step.output = steps::getDefaultRenderStepOutput();
  }
  applyOutputScale(context, step.output->outputSizing, input);
}

template <typename T>
void applyAll(SHContext *context, T &step, const ParamVar &outputs, const ParamVar &outputScale, const ParamVar &features,
              const ParamVar &name) {
  if (!outputs.isNone()) {
    // Type checking is done in applyOutputs
    shared::applyOutputsToStep(context, step, outputs.get());
  } else {
    if (step.output) {
      step.output.reset();
    }
  }

  if (!outputScale.isNone()) {
    shared::applyOutputScaleToStep(context, step, outputScale.get());
  } else {
    if (step.output) {
      step.output->outputSizing = RenderStepOutput::ManualSize{};
    }
  }

  step.features.clear();
  if (!features.isNone()) {
    applyFeatures(context, step.features, features.get());
  }

  step.name.clear();
  if (!name.isNone()) {
    step.name = SHSTRVIEW(name.get());
  }
}

} // namespace shared

struct HashState {
  Hash128 hash;

  template <typename T> bool update(const T &step) {
    HasherXXH3<> hasher;
    for (const FeaturePtr &feature : step.features) {
      hasher(feature->getId());
    }

    auto newHash = hasher.getDigest();
    if (hash != newHash) {
      hash = newHash;
      return true;
    } else {
      return false;
    }
  }
};

struct DrawablePassShard {
  static inline Type DrawQueueVarType = Type::VariableOf(ShardsTypes::DrawQueue);

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::PipelineStep; }

  static SHOptionalString help() {
    return SHCCSTR("This shard creates a render pass object, meant for rendering drawable objects, using the drawables from the drawables "
                   "queue (specified in the Queue parameter) "
                   "and the sequence of features objects (specified in the Features parameter).");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The render pass object for use in a render pipeline."); }

  PARAM_EXT(ParamVar, _name, ShardsTypes::NameParameterInfo);
  PARAM_PARAMVAR(_queue, "Queue", "The drawables queue to get drawables from.", {DrawQueueVarType});
  PARAM_EXT(ParamVar, _features, ShardsTypes::FeaturesParameterInfo);
  PARAM_EXT(ParamVar, _outputs, ShardsTypes::OutputsParameterInfo);
  PARAM_EXT(ParamVar, _outputScale, ShardsTypes::OutputScaleParameterInfo);
  PARAM_PARAMVAR(_sort, "Sort",
                 "The sorting mode to use to sort the drawables. The default sorting behavior is to sort by optimal batching.",
                 {CoreInfo::NoneType, ShardsTypes::SortModeEnumInfo::Type,
                  Type::VariableOf(ShardsTypes::SortModeEnumInfo::Type)});
  PARAM_VAR(_ignoreDrawableFeatures, "IgnoreDrawableFeatures",
            "Ignore any features on drawables and only use the features specified in this pass.",
            {CoreInfo::NoneType, CoreInfo::BoolType});

  PARAM_IMPL(PARAM_IMPL_FOR(_name), PARAM_IMPL_FOR(_queue), PARAM_IMPL_FOR(_features), PARAM_IMPL_FOR(_outputs),
             PARAM_IMPL_FOR(_outputScale), PARAM_IMPL_FOR(_sort), PARAM_IMPL_FOR(_ignoreDrawableFeatures));

  PipelineStepPtr *_step{};
  HashState _hashState;

  DrawablePassShard() { _outputScale = shared::getDefaultOutputScale(); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  void cleanup(SHContext *context) {
    if (_step) {
      ShardsTypes::PipelineStepObjectVar.Release(_step);
      _step = nullptr;
    }

    PARAM_CLEANUP(context);
  }

  void warmup(SHContext *context) {
    _step = ShardsTypes::PipelineStepObjectVar.New();
    *_step = makePipelineStep(RenderDrawablesStep());

    PARAM_WARMUP(context);
  }

  void applyQueue(SHContext *context, RenderDrawablesStep &step) {
    step.drawQueue = varAsObjectChecked<DrawQueuePtr>(_queue.get(), ShardsTypes::DrawQueue);
  }

  void applySorting(SHContext *context, RenderDrawablesStep &step) {
    step.sortMode = (gfx::SortMode)_sort.get().payload.enumValue;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    RenderDrawablesStep &step = std::get<RenderDrawablesStep>(*_step->get());

    shared::applyAll(context, step, _outputs, _outputScale, _features, _name);

    if (!_sort.isNone()) {
      applySorting(context, step);
    } else {
      step.sortMode = SortMode::Batch;
    }

    if (!_queue.isNone()) {
      applyQueue(context, step);
    } else {
      throw formatException("DrawablePass requires a queue");
    }

    if (!_ignoreDrawableFeatures->isNone()) {
      step.ignoreDrawableFeatures = (bool)*_ignoreDrawableFeatures;
    } else {
      step.ignoreDrawableFeatures = false;
    }

    if (_hashState.update(step)) {
      step.id = renderStepIdGenerator.getNext();
    }

    return ShardsTypes::PipelineStepObjectVar.Get(_step);
  }
};

struct EffectPassShard {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::PipelineStep; }

  static SHOptionalString help() {
    return SHCCSTR("This shard creates a render pass object designed for applying post processing effects or full-screen "
                   "rendering techniques.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The render pass object for use in a render pipeline."); }

  PARAM_EXT(ParamVar, _name, ShardsTypes::NameParameterInfo);
  PARAM_EXT(ParamVar, _outputs, ShardsTypes::OutputsParameterInfo);
  PARAM_EXT(ParamVar, _outputScale, ShardsTypes::OutputScaleParameterInfo);
  PARAM_PARAMVAR(_inputs, "Inputs", "", {CoreInfo::NoneType, CoreInfo::StringSeqType, CoreInfo::StringVarSeqType});
  PARAM_PARAMVAR(_entryPoint, "EntryPoint", "", {CoreInfo::NoneType, CoreInfo::ShardRefSeqType, CoreInfo::ShardRefVarSeqType});
  PARAM_EXT(ParamVar, _params, gfx::detail::ParamsParameterInfo);
  PARAM_EXT(ParamVar, _features, ShardsTypes::FeaturesParameterInfo);
  PARAM_PARAMVAR(_composeWith, "ComposeWith", "Any table of values that need to be injected into this feature's shaders",
                 {CoreInfo::NoneType, CoreInfo::AnyTableType, CoreInfo::AnyVarTableType});

  PARAM_IMPL(PARAM_IMPL_FOR(_name), PARAM_IMPL_FOR(_outputs), PARAM_IMPL_FOR(_outputScale), PARAM_IMPL_FOR(_inputs),
             PARAM_IMPL_FOR(_entryPoint), PARAM_IMPL_FOR(_params), PARAM_IMPL_FOR(_features), PARAM_IMPL_FOR(_composeWith));

  FeaturePtr wrapperFeature;
  PipelineStepPtr *_step{};
  gfx::shader::VariableMap _composedWith;
  SHVar _composedWithHash{};
  std::vector<FeaturePtr> _generatedFeatures;

  EffectPassShard() {
    static SeqVar defaultOutputs = []() {
      SeqVar t;
      t.resize(1);
      auto &color = t.get<TableVar>(0);
      color["Name"] = Var("color");
      return t;
    }();
    _outputs = defaultOutputs;
    _outputScale = shared::getDefaultOutputScale();
  }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  void cleanup(SHContext *context) {
    if (_step) {
      ShardsTypes::PipelineStepObjectVar.Release(_step);
      _step = nullptr;
    }

    PARAM_CLEANUP(context);
  }

  void warmup(SHContext *context) {
    _step = ShardsTypes::PipelineStepObjectVar.New();
    *_step = makePipelineStep(RenderFullscreenStep());

    wrapperFeature = std::make_shared<Feature>();

    PARAM_WARMUP(context);
  }

  void applyParams(SHContext *context, RenderFullscreenStep &step, const SHVar &input) {
    checkType(input.valueType, SHType::Table, ":Params");
    const SHTable &inputTable = input.payload.tableValue;

    initShaderParams(context, inputTable, step.parameters);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    RenderFullscreenStep &step = std::get<RenderFullscreenStep>(*_step->get());

    if (!_composeWith.isNone()) {
      shader::applyComposeWithHashed(context, _composeWith.get(), _composedWithHash, _composedWith, shader::VariableRemapping(),
                                     [&]() {
                                       // Create a new object to force shader recompile
                                       wrapperFeature = std::make_shared<Feature>();
                                     });
    }

    shared::applyAll(context, step, _outputs, _outputScale, _features, _name);

    if (!_inputs.isNone()) {
      shared::applyInputsToStep(context, step, _inputs.get());
    } else {
      step.input.attachments.clear();
    }

    if (!_params.isNone()) {
      applyParams(context, step, _params.get());
    }

    // Only do this once
    if (wrapperFeature->shaderEntryPoints.empty()) {
      _generatedFeatures.clear();

      // Add default base transform
      _generatedFeatures.push_back(features::Transform::create(false, false));

      if (!_entryPoint.isNone()) {
        shader::EntryPoint entryPoint;
        entryPoint.stage = ProgrammableGraphicsStage::Fragment;
        shader::applyShaderEntryPoint(context, entryPoint, _entryPoint.get(), _composedWith);
        wrapperFeature->shaderEntryPoints.emplace_back(std::move(entryPoint));
        _generatedFeatures.push_back(wrapperFeature);
      }
    }

    for (auto &feature : _generatedFeatures) {
      step.features.push_back(feature);
    }

    return ShardsTypes::PipelineStepObjectVar.Get(_step);
  }
};

struct CopyPassShard {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::PipelineStep; }

  static SHOptionalString help() {
    return SHCCSTR("This shard creates a render pass object that is meant for transferring render data from one stage of the "
                   "render pipeline to the next. It is also able to make changes to the render data specified in the Inputs "
                   "parameter, like changing its texture format or down sampling the texture. It makes these changes through its "
                   "Outputs and OutputScale parameters.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("The render pass object for use in a render pipeline."); }

  PARAM_EXT(ParamVar, _name, ShardsTypes::NameParameterInfo);
  PARAM_EXT(ParamVar, _outputs, ShardsTypes::OutputsParameterInfo);
  PARAM_EXT(ParamVar, _outputScale, ShardsTypes::OutputScaleParameterInfo);
  PARAM_PARAMVAR(_inputs, "Inputs", "The names of the render pass objects to modify as a sequence of strings.", {CoreInfo::NoneType, CoreInfo::StringSeqType, CoreInfo::StringVarSeqType});

  PARAM_IMPL(PARAM_IMPL_FOR(_name), PARAM_IMPL_FOR(_outputs), PARAM_IMPL_FOR(_outputScale), PARAM_IMPL_FOR(_inputs));

  PipelineStepPtr *_step{};
  RenderStepInput rsInput;
  RenderStepOutput rsOutput;

  CopyPassShard() { _outputScale = shared::getDefaultOutputScale(); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  void cleanup(SHContext *context) {
    if (_step) {
      ShardsTypes::PipelineStepObjectVar.Release(_step);
      _step = nullptr;
    }

    PARAM_CLEANUP(context);
  }

  void warmup(SHContext *context) {
    _step = ShardsTypes::PipelineStepObjectVar.New();
    PARAM_WARMUP(context);
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!*_step) {
      if (!_inputs.isNone()) {
        shared::applyInputs(context, rsInput, _inputs.get());
      } else {
        throw formatException("CopyPass requires inputs");
      }

      if (!_outputs.isNone()) {
        // Type checking is done in applyOutputs
        shared::applyOutputs(context, rsOutput, _outputs.get());
      } else {
        throw formatException("CopyPass requires outputs");
      }

      if (!_outputScale.isNone()) {
        shared::applyOutputScale(context, rsOutput.outputSizing, _outputScale.get());
      } else {
        rsOutput.outputSizing = RenderStepOutput::ManualSize{};
      }

      *_step = steps::Copy::create(rsInput, rsOutput);

      if (!_name.isNone()) {
        std::get<RenderFullscreenStep>(*(*_step).get()).name = SHSTRVIEW(_name.get());
      }
    } else {
      auto &step = std::get<RenderFullscreenStep>(*(*_step).get());
      shared::applyInputsToStep(context, step, _inputs.get());
      shared::applyOutputsToStep(context, step, _outputs.get());
    }

    return ShardsTypes::PipelineStepObjectVar.Get(_step);
  }
};

void registerRenderStepShards() {
  REGISTER_SHARD("GFX.DrawablePass", DrawablePassShard);
  REGISTER_SHARD("GFX.EffectPass", EffectPassShard);
  REGISTER_SHARD("GFX.CopyPass", CopyPassShard);
}
} // namespace gfx
