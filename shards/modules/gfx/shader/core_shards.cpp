#include "core_shards.hpp"
#include "translate_wrapper.hpp"
#include <gfx/shader/fmt.hpp>

using namespace shards;
namespace gfx::shader {

struct PushTranslator {
  static void translate(Push *shard, TranslationContext &context) {
    auto &scope = context.getTop();

    auto value = context.takeWGSLTop();
    if (!value)
      throw std::runtime_error("Missing value to push");

    NumType elementType = std::get<NumType>(value->getType());

    auto it = scope.virtualSequences.find(shard->_name);
    if (it == scope.virtualSequences.end()) {
      it = scope.virtualSequences.emplace(std::make_pair(shard->_name, VirtualSeq())).first;
      it->second.elementType = elementType;
    } else {
      if (it->second.elementType != elementType)
        throw std::runtime_error(
            fmt::format("Sequence value type mismatch {} (old) != {} (new)", it->second.elementType, elementType));
    }

    auto &virtualSeq = it->second;
    virtualSeq.elements.emplace_back(std::make_unique<WGSLBlock>(elementType, value->toBlock()));

    // Restore / passthrough
    context.wgslTop = std::move(value);
  }
};

struct PassTranslator {
  static void translate(std::monostate *shard, TranslationContext &context) {}
};

void registerCoreShards() {
  // Literal copy-paste into shader code
  REGISTER_ENUM(ShaderLiteralTypeEnumInfo);
  REGISTER_SHADER_SHARD("Shader.Literal", Literal);

  // Basic IO
  REGISTER_SHADER_SHARD("Shader.ReadInput", gfx::shader::ShaderReadInput);
  REGISTER_SHADER_SHARD("Shader.ReadBuffer", gfx::shader::ReadBuffer);
  REGISTER_SHADER_SHARD("Shader.ReadGlobal", gfx::shader::ShaderReadGlobal);
  REGISTER_SHADER_SHARD("Shader.WriteGlobal", gfx::shader::ShaderWriteGlobal);
  REGISTER_SHADER_SHARD("Shader.WriteOutput", gfx::shader::ShaderWriteOutput);
  REGISTER_SHADER_SHARD("Shader.SampleTexture", gfx::shader::SampleTexture);
  REGISTER_SHADER_SHARD("Shader.SampleTextureCoord", gfx::shader::SampleTextureCoord);
  REGISTER_SHADER_SHARD("Shader.RefTexture", gfx::shader::RefTexture);
  REGISTER_SHADER_SHARD("Shader.RefSampler", gfx::shader::RefSampler);
  REGISTER_SHADER_SHARD("Shader.RefBuffer", gfx::shader::RefBuffer);

  REGISTER_SHADER_SHARD("Shader.WithInput", gfx::shader::WithInput);
  REGISTER_SHADER_SHARD("Shader.WithTexture", gfx::shader::WithTexture);

  // Utilities
  REGISTER_SHADER_SHARD("Shader.LinearizeDepth", gfx::shader::LinearizeDepth);

  // Variable manipulation
  REGISTER_EXTERNAL_SHADER_SHARD(ConstTranslator, "Const", shards::Const);
  REGISTER_EXTERNAL_SHADER_SHARD_T1(SetTranslator, "Set", shards::Set);
  REGISTER_EXTERNAL_SHADER_SHARD_T1(SetTranslator, "Ref", shards::Ref);
  REGISTER_EXTERNAL_SHADER_SHARD(GetTranslator, "Get", shards::Get);
  REGISTER_EXTERNAL_SHADER_SHARD(UpdateTranslator, "Update", shards::Update);
  REGISTER_EXTERNAL_SHADER_SHARD(TakeTranslator, "Take", shards::Take);

  REGISTER_EXTERNAL_SHADER_SHARD(PushTranslator, "Push", shards::Push);

  REGISTER_EXTERNAL_SHADER_SHARD(PassTranslator, "Pass", std::monostate);
  REGISTER_EXTERNAL_SHADER_SHARD(PassTranslator, "Debug.Noop", std::monostate);
}
} // namespace gfx::shader