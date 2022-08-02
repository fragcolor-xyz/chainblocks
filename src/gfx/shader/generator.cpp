#include "generator.hpp"
#include "fmt.hpp"
#include "wgsl_mapping.hpp"
#include <algorithm>
#include <boost/algorithm/string/join.hpp>
#include <gfx/error_utils.hpp>
#include <gfx/graph.hpp>
#include <magic_enum.hpp>
#include <optional>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace gfx {
namespace shader {

void GeneratorContext::write(const StringView &str) { result += str; }
void GeneratorContext::writeHeader(const StringView &str) { header += str; }

void GeneratorContext::readGlobal(const char *name) {
  auto it = globals.find(name);
  if (it == globals.end()) {
    pushError(formatError("Global {} does not exist", name));
  } else {
    result += fmt::format("{}.{}", globalsVariableName, name);
  }
}

bool GeneratorContext::hasInput(const char *name) { return inputs.find(name) != inputs.end(); }

void GeneratorContext::readInput(const char *name) {
  auto it = inputs.find(name);
  const FieldType *fieldType{};
  if (it != inputs.end()) {
    fieldType = &it->second;
  } else {
    fieldType = getOrCreateDynamicInput(name);
  }

  if (!fieldType) {
    pushError(formatError("Input {} does not exist", name));
    return;
  }

  result += fmt::format("{}.{}", inputVariableName, name);
}

const FieldType *GeneratorContext::getOrCreateDynamicInput(const char *name) {
  assert(inputs.find(name) == inputs.end());

  FieldType newField;
  for (auto &h : dynamicHandlers) {
    if (h->createDynamicInput(name, newField)) {
      return &inputs.insert_or_assign(name, newField).first->second;
    }
  }

  return nullptr;
}

bool GeneratorContext::hasOutput(const char *name) { return outputs.find(name) != outputs.end(); }

void GeneratorContext::writeOutput(const char *name, const FieldType &type) {
  auto it = outputs.find(name);
  const FieldType *outputFieldType{};
  if (it != outputs.end()) {
    outputFieldType = &it->second;
  } else {
    outputFieldType = getOrCreateDynamicOutput(name, type);
  }

  if (!outputFieldType) {
    pushError(formatError("Output {} does not exist", name));
    return;
  }

  if (*outputFieldType != type) {
    pushError(formatError("Output type doesn't match previously expected type"));
    return;
  }

  result += fmt::format("{}.{}", outputVariableName, name);
}

const FieldType *GeneratorContext::getOrCreateDynamicOutput(const char *name, FieldType requestedType) {
  assert(outputs.find(name) == outputs.end());

  for (auto &h : dynamicHandlers) {
    if (h->createDynamicOutput(name, requestedType)) {
      return &outputs.insert_or_assign(name, requestedType).first->second;
    }
  }

  return nullptr;
}

bool GeneratorContext::hasTexture(const char *name, bool defaultTexcoordRequired) {
  auto texture = getTexture(name);
  if (!texture)
    return false;
  if (defaultTexcoordRequired && !hasInput(texture->defaultTexcoordVariableName.c_str()))
    return false;
  return true;
}

const TextureDefinition *GeneratorContext::getTexture(const char *name) {
  auto it = textures.find(name);
  if (it == textures.end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}

void GeneratorContext::texture(const char *name) {
  if (const TextureDefinition *texture = getTexture(name)) {
    result += texture->variableName;
  } else {
    pushError(formatError("Texture {} does not exist", name));
  }
}

void GeneratorContext::textureDefaultTextureCoordinate(const char *name) {
  if (const TextureDefinition *texture = getTexture(name)) {
    if (hasInput(texture->defaultTexcoordVariableName.c_str())) {
      readInput(texture->defaultTexcoordVariableName.c_str());
    } else {
      result += "vec2<f32>(0.0, 0.0)";
    }
  }
}

void GeneratorContext::textureDefaultSampler(const char *name) {
  if (const TextureDefinition *texture = getTexture(name)) {
    result += texture->defaultSamplerVariableName;
  }
}

void GeneratorContext::readBuffer(const char *fieldName, const FieldType &expectedType, const char *bufferName) {
  auto bufferIt = buffers.find(bufferName);
  if (bufferIt == buffers.end()) {
    pushError(formatError("Buffer \"{}\" is not defined", bufferName));
    return;
  }

  const BufferDefinition &buffer = bufferIt->second;

  const UniformLayout *uniform = findUniform(fieldName, buffer);
  if (!uniform) {
    pushError(formatError("Field \"{}\" not found in buffer \"{}\"", fieldName, bufferName));
    return;
  }

  if (expectedType != uniform->type) {
    pushError(formatError("Field \"{}\", shader expected type {} but provided was {}", fieldName, expectedType, uniform->type));
    return;
  }

  if (buffer.indexedBy) {
    result += fmt::format("{}.elements[{}].{}", buffer.variableName, *buffer.indexedBy, fieldName);
  } else {
    result += fmt::format("{}.{}", buffer.variableName, fieldName);
  }
}

const UniformLayout *GeneratorContext::findUniform(const char *fieldName, const BufferDefinition &buffer) {
  for (size_t i = 0; i < buffer.layout.fieldNames.size(); i++) {
    if (buffer.layout.fieldNames[i] == fieldName) {
      return &buffer.layout.items[i];
    }
  }
  return nullptr;
}

void GeneratorContext::pushError(GeneratorError &&error) { errors.emplace_back(std::move(error)); }

enum class BufferType {
  Uniform,
  Storage,
};

template <typename T>
static void generateTextureVars(T &output, const TextureDefinition &def, size_t group, size_t binding, size_t samplerBinding) {
  const char *textureFormat = "f32";
  const char *textureType = "texture_2d";

  output += fmt::format("@group({}) @binding({})\n", group, binding);
  output += fmt::format("var {}: {}<{}>;\n", def.variableName, textureType, textureFormat);

  output += fmt::format("@group({}) @binding({})\n", group, samplerBinding);
  output += fmt::format("var {}: sampler;\n", def.defaultSamplerVariableName);
}

template <typename T>
static void generateBuffer(T &output, const String &name, BufferType type, size_t group, size_t binding,
                           const UniformBufferLayout &layout, bool isArray = false) {

  String structName = name + "_t";
  output += fmt::format("struct {} {{\n", structName);
  for (size_t i = 0; i < layout.fieldNames.size(); i++) {
    output += fmt::format("\t{}: {},\n", layout.fieldNames[i], getFieldWGSLTypeName(layout.items[i].type));
  }
  output += "};\n";

  // array struct wrapper
  const char *varType = structName.c_str();
  String containerTypeName = name + "_container";
  if (isArray) {
    output += fmt::format("struct {} {{ elements: array<{}> }};\n", containerTypeName, structName);
    varType = containerTypeName.c_str();
  }

  // global storage/uniform variable
  const char *varStorageType = nullptr;
  switch (type) {
  case BufferType::Uniform:
    varStorageType = "uniform";
    break;
  case BufferType::Storage:
    varStorageType = "storage";
    break;
  }
  output += fmt::format("@group({}) @binding({})\n", group, binding);
  output += fmt::format("var<{}> {}: {};\n", varStorageType, name, varType);
}

struct StructField {
  static constexpr size_t NO_LOCATION = ~0;
  NamedField base;
  size_t location = NO_LOCATION;
  String builtinTag;

  StructField() = default;
  StructField(const NamedField &base) : base(base) {}
  StructField(const NamedField &base, size_t location) : base(base), location(location) {}
  StructField(const NamedField &base, const String &builtinTag) : base(base), builtinTag(builtinTag) {}
  bool hasLocation() const { return location != NO_LOCATION; }
  bool hasBuiltinTag() const { return !builtinTag.empty(); }
};

template <typename T>
static void generateStruct(T &output, const String &typeName, const std::vector<StructField> &fields, bool interpolated = true) {
  output += fmt::format("struct {} {{\n", typeName);
  for (auto &field : fields) {
    std::string typeName = getFieldWGSLTypeName(field.base.type);

    String extraTags;
    if (interpolated && isIntegerType(field.base.type.baseType)) {
      // integer vertex outputs requires flat interpolation
      extraTags = "@interpolate(flat)";
    }

    if (field.hasBuiltinTag()) {
      output += fmt::format("\t@builtin({}) {}", field.builtinTag, extraTags);
    } else if (field.hasLocation()) {
      output += fmt::format("\t@location({}) {}", field.location, extraTags);
    }
    output += fmt::format("{}: {},\n", field.base.name, typeName);
  }
  output += "};\n";
}

static size_t getNextStructLocation(const std::vector<StructField> &_struct) {
  size_t loc = 0;
  for (auto &field : _struct) {
    if (field.hasLocation()) {
      loc = std::max(loc, field.location + 1);
    }
  }
  return loc;
}

struct StageOutput {
  String code;
  std::vector<GeneratorError> errors;
};

struct Stage {
  ProgrammableGraphicsStage stage;
  std::vector<const EntryPoint *> entryPoints;
  std::vector<String> extraEntryPointParameters;
  String mainFunctionHeader;
  String inputStructName;
  String inputVariableName;
  String outputStructName;
  String outputVariableName;
  String globalsStructName;
  String globalsVariableName;

  Stage(ProgrammableGraphicsStage stage, const String &inputStructName, const String &outputStructName)
      : stage(stage), inputStructName((inputStructName)), outputStructName((outputStructName)) {
    inputVariableName = fmt::format("p_{}_input", magic_enum::enum_name(stage));
    outputVariableName = fmt::format("p_{}_output", magic_enum::enum_name(stage));

    globalsStructName = fmt::format("{}_globals_t", magic_enum::enum_name(stage));
    globalsVariableName = fmt::format("p_{}_globals", magic_enum::enum_name(stage));
  }

  bool sort(bool ignoreMissingDependencies = true) {
    std::unordered_map<std::string, size_t> nodeNames;
    for (size_t i = 0; i < entryPoints.size(); i++) {
      const EntryPoint &entryPoint = *entryPoints[i];
      if (!entryPoint.name.empty())
        nodeNames.insert_or_assign(entryPoint.name, i);
    }

    auto resolveNodeIndex = [&](const std::string &name) -> const size_t * {
      auto it = nodeNames.find(name);
      if (it != nodeNames.end())
        return &it->second;
      return nullptr;
    };

    std::set<std::string> missingDependencies;
    graph::Graph graph;
    graph.nodes.resize(entryPoints.size());
    for (size_t i = 0; i < entryPoints.size(); i++) {
      const EntryPoint &entryPoint = *entryPoints[i];
      for (auto &dep : entryPoint.dependencies) {
        if (dep.type == DependencyType::Before) {
          if (const size_t *depIndex = resolveNodeIndex(dep.name)) {
            graph.nodes[*depIndex].dependencies.push_back(i);
          } else if (!ignoreMissingDependencies) {
            missingDependencies.insert(dep.name);
          }
        } else {
          if (const size_t *depIndex = resolveNodeIndex(dep.name)) {
            graph.nodes[i].dependencies.push_back(*depIndex);
          } else if (!ignoreMissingDependencies) {
            missingDependencies.insert(dep.name);
          }
        }
      }
    }

    if (!ignoreMissingDependencies && missingDependencies.size() > 0) {
      return false;
    }

    std::vector<size_t> sortedIndices;
    if (!graph::topologicalSort(graph, sortedIndices))
      return false;

    auto unsortedEntryPoints = std::move(entryPoints);
    for (size_t i = 0; i < sortedIndices.size(); i++) {
      entryPoints.push_back(unsortedEntryPoints[sortedIndices[i]]);
    }
    return true;
  }

  StageOutput process(const std::vector<NamedField> &inputs, const std::vector<NamedField> &outputs,
                      const std::map<String, BufferDefinition> &buffers,
                      const std::map<String, TextureDefinition> &textureDefinitions,
                      const std::vector<IGeneratorDynamicHandler *> &dynamicHandlers) {
    GeneratorContext context;
    context.buffers = buffers;
    context.inputVariableName = inputVariableName;
    context.outputVariableName = outputVariableName;
    context.globalsVariableName = globalsVariableName;
    context.textures = textureDefinitions;
    context.dynamicHandlers = dynamicHandlers;

    for (auto &input : inputs) {
      context.inputs.insert_or_assign(input.name, input.type);
    }

    for (auto &outputField : outputs) {
      context.outputs.insert_or_assign(outputField.name, outputField.type);
    }

    const char *wgslStageName{};
    switch (stage) {
    case ProgrammableGraphicsStage::Vertex:
      wgslStageName = "vertex";
      break;
    case ProgrammableGraphicsStage::Fragment:
      wgslStageName = "fragment";
      break;
    }

    context.write(fmt::format("var<private> {}: {};\n", inputVariableName, inputStructName));
    context.write(fmt::format("var<private> {}: {};\n", outputVariableName, outputStructName));

    std::vector<String> functionNamesToCall;
    size_t index = 0;
    for (const EntryPoint *entryPoint : entryPoints) {
      String entryPointFunctionName = fmt::format("entryPoint_{}_{}", wgslStageName, index++);
      functionNamesToCall.push_back(entryPointFunctionName);

      context.write(fmt::format("fn {}() {{\n", entryPointFunctionName));
      entryPoint->code->apply(context);
      context.write("}\n");
    }

    String entryPointParams = fmt::format("in: {}", inputStructName);
    if (!extraEntryPointParameters.empty()) {
      entryPointParams += ", " + boost::algorithm::join(extraEntryPointParameters, ", ");
    }

    context.write(
        fmt::format("@{}\nfn {}_main({}) -> {} {{\n", wgslStageName, wgslStageName, entryPointParams, outputStructName));
    context.write(fmt::format("\t{} = in;\n", inputVariableName));

    context.write("\t" + mainFunctionHeader + "\n");

    for (auto &functionName : functionNamesToCall) {
      context.write(fmt::format("\t{}();\n", functionName));
    }

    context.write(fmt::format("\treturn {};\n", outputVariableName));
    context.write("}\n");

    String globalsHeader;
    if (!context.globals.empty()) {
      std::vector<StructField> globalsStructFields;
      for (auto &field : context.globals) {
        globalsStructFields.emplace_back(NamedField(field.first, field.second));
      }
      generateStruct(globalsHeader, globalsStructName, globalsStructFields, false);
      globalsHeader += fmt::format("var<private> {}: {};\n", globalsVariableName, globalsStructName);
    }

    return StageOutput{
        globalsHeader + std::move(context.header) + std::move(context.result),
        std::move(context.errors),
    };
  }
};

GeneratorOutput Generator::build(const std::vector<EntryPoint> &_entryPoints) {
  std::vector<const EntryPoint *> entryPoints;
  std::transform(_entryPoints.begin(), _entryPoints.end(), std::back_inserter(entryPoints),
                 [](const EntryPoint &entryPoint) { return &entryPoint; });
  return build(entryPoints);
}

struct DynamicVertexInput : public IGeneratorDynamicHandler {
  std::vector<StructField> &inputStruct;

  DynamicVertexInput(std::vector<StructField> &inputStruct) : inputStruct(inputStruct) {}

  bool createDynamicInput(const char *name, FieldType &out) {
    if (strcmp(name, "vertex_index") == 0) {
      out = FieldType(ShaderFieldBaseType::UInt32);
      StructField newField = generateDynamicStructInput(name, out);
      inputStruct.push_back(newField);
      return true;
    }
    return false;
  }

  StructField generateDynamicStructInput(const String &name, const FieldType &type) {
    if (name == "vertex_index") {
      return StructField(NamedField(name, type), "vertex_index");
    } else {
      throw std::logic_error("Unknown dynamic vertex input");
    }
  }
};

struct DynamicVertexOutput : public IGeneratorDynamicHandler {
  std::vector<StructField> &outputStruct;

  DynamicVertexOutput(std::vector<StructField> &outputStruct) : outputStruct(outputStruct) {}

  bool createDynamicOutput(const char *name, FieldType requestedType) {
    StructField newField = generateDynamicStructOutput(name, requestedType);
    outputStruct.push_back(newField);
    return true;
  }

  StructField generateDynamicStructOutput(const String &name, const FieldType &type) {
    // Handle builtin outputs here
    if (name == "position") {
      return StructField(NamedField(name, type), "position");
    } else {
      size_t location = getNextStructLocation(outputStruct);
      return StructField(NamedField(name, type), location);
    }
  }
};

GeneratorOutput Generator::build(const std::vector<const EntryPoint *> &entryPoints) {
  String vertexInputStructName = "Input";
  String vertexOutputStructName = "VertOutput";
  String fragmentInputStructName = "FragInput";
  String fragmentOutputStructName = "Output";

  const size_t numStages = 2;
  Stage stages[2] = {
      Stage(ProgrammableGraphicsStage::Vertex, vertexInputStructName, vertexOutputStructName),
      Stage(ProgrammableGraphicsStage::Fragment, fragmentInputStructName, fragmentOutputStructName),
  };
  auto &vertexStage = stages[0];

  GeneratorOutput output;

  for (const auto &entryPoint : entryPoints) {
    stages[size_t(entryPoint->stage)].entryPoints.push_back(entryPoint);
  }

  std::string headerCode;
  headerCode.reserve(2 << 12);

  const char *instanceIndexer = "u_instanceIndex";
  headerCode += fmt::format("var<private> {}: u32;\n", instanceIndexer);

  std::vector<NamedField> vertexInputFields;
  for (auto &attr : meshFormat.vertexAttributes) {
    vertexInputFields.emplace_back(attr.name, FieldType(getCompatibleShaderFieldBaseType(attr.type), attr.numComponents));
  }

  std::vector<StructField> vertexInputStructFields;
  for (size_t i = 0; i < vertexInputFields.size(); i++) {
    vertexInputStructFields.emplace_back(vertexInputFields[i], i);
  }

  std::vector<StructField> fragmentOutputStructFields;
  for (size_t i = 0; i < outputFields.size(); i++) {
    fragmentOutputStructFields.emplace_back(outputFields[i], i);
  }

  stages[0].extraEntryPointParameters.push_back("@builtin(instance_index) _instanceIndex: u32");
  stages[0].mainFunctionHeader += fmt::format("{} = _instanceIndex;\n", instanceIndexer);
  stages[1].mainFunctionHeader += fmt::format("{} = {}.instanceIndex;\n", instanceIndexer, stages[1].inputVariableName);

  // Interpolate instance index
  stages[0].mainFunctionHeader += fmt::format("{}.instanceIndex = {};\n", stages[0].outputVariableName, instanceIndexer);

  if (!viewBufferLayout.fieldNames.empty())
    generateBuffer(headerCode, "u_view", BufferType::Uniform, 0, 0, viewBufferLayout);
  if (!objectBufferLayout.fieldNames.empty())
    generateBuffer(headerCode, "u_objects", BufferType::Storage, 0, 1, objectBufferLayout, true);

  std::map<String, BufferDefinition> buffers = {
      {"view", {"u_view", viewBufferLayout}},
      {"object", {"u_objects", objectBufferLayout, instanceIndexer}},
  };

  std::vector<StructField> vertexOutputStructFields;
  std::vector<NamedField> fragmentInputFields;

  vertexOutputStructFields.emplace_back(NamedField("instanceIndex", FieldType(ShaderFieldBaseType::UInt32, 1)), 0);

  std::map<String, TextureDefinition> textureDefinitions;
  size_t textureBindGroup = 1;
  size_t textureBindingCounter = 0;
  for (auto &texture : textureBindingLayout.bindings) {
    TextureDefinition def;
    def.variableName = "t_" + texture.name;
    def.defaultSamplerVariableName = "s_" + texture.name;
    def.defaultTexcoordVariableName = fmt::format("texCoord{}", texture.defaultTexcoordBinding);
    size_t textureBinding = textureBindingCounter++;
    size_t samplerBinding = textureBindingCounter++;
    generateTextureVars(headerCode, def, textureBindGroup, textureBinding, samplerBinding);
    textureDefinitions.insert_or_assign(texture.name, def);
  }

  DynamicVertexInput dynamicVertexInputHandler(vertexInputStructFields);
  DynamicVertexOutput dynamicVertexOutputHandler(vertexOutputStructFields);

  String stagesCode;
  stagesCode.reserve(2 << 12);
  for (size_t i = 0; i < numStages; i++) {
    auto &stage = stages[i];

    bool sorted = stage.sort();
    assert(sorted);

    static std::vector<NamedField> emptyStruct;
    std::vector<NamedField> *inputs{};
    std::vector<NamedField> *outputs{};
    std::vector<IGeneratorDynamicHandler *> dynamics;
    switch (stage.stage) {
    case gfx::ProgrammableGraphicsStage::Vertex:
      inputs = &vertexInputFields;
      outputs = &emptyStruct;
      dynamics.push_back(&dynamicVertexInputHandler);
      dynamics.push_back(&dynamicVertexOutputHandler);
      break;
    case gfx::ProgrammableGraphicsStage::Fragment:
      inputs = &fragmentInputFields;
      outputs = &outputFields;
      break;
    }

    StageOutput stageOutput = stage.process(*inputs, *outputs, buffers, textureDefinitions, dynamics);
    for (auto &error : stageOutput.errors)
      output.errors.emplace_back(error);

    stagesCode += stageOutput.code;

    // Copy interpolated fields from vertex to fragment
    if (&stage == &vertexStage) {
      for (auto &outputField : vertexOutputStructFields)
        fragmentInputFields.emplace_back(outputField.base);
    }
  }

  // Generate input/output structs here since they depend on shader code
  generateStruct(headerCode, vertexInputStructName, vertexInputStructFields, false);
  generateStruct(headerCode, vertexOutputStructName, vertexOutputStructFields);
  generateStruct(headerCode, fragmentInputStructName, vertexOutputStructFields);
  generateStruct(headerCode, fragmentOutputStructName, fragmentOutputStructFields, false);

  output.wgslSource = headerCode + stagesCode;

  return output;
}

void GeneratorOutput::dumpErrors(const GeneratorOutput &output) {
  if (!output.errors.empty()) {
    spdlog::error("Failed to generate shader code:");
    for (auto &error : output.errors) {
      spdlog::error(">  {}", error.error);
    }
  }
}
} // namespace shader
} // namespace gfx
