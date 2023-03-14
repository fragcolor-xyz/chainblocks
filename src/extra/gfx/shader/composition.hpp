#ifndef F7DDD110_DF89_4AA4_A005_18C9BBFDFC40
#define F7DDD110_DF89_4AA4_A005_18C9BBFDFC40

#include <optional>
#include <shards.h>
#include <gfx/shader/generator.hpp>

namespace gfx::shader {
using VariableMap = std::unordered_map<std::string, shards::OwnedVar>;
struct ShaderCompositionContext {
  IGeneratorContext &generatorContext;
  const VariableMap &composeWith;

  ShaderCompositionContext(IGeneratorContext &generatorContext, const VariableMap &composeWith)
      : generatorContext(generatorContext), composeWith(composeWith) {}

  std::optional<SHVar> getComposeTimeConstant(const std::string &key) {
    auto it = composeWith.find(key);
    if (it == composeWith.end())
      return std::nullopt;
    return it->second;
  }

  static ShaderCompositionContext &get();
  template <typename T> static auto withContext(ShaderCompositionContext &ctx, T &&cb) -> decltype(cb()) {
    using R = decltype(cb());

    setContext(&ctx);
    if constexpr (std::is_void_v<R>) {
      cb();
      setContext(nullptr);
    } else {
      auto result = cb();
      setContext(nullptr);
      return result;
    }
  }

private:
  static void setContext(ShaderCompositionContext *context);
};

void applyShaderEntryPoint(SHContext *context, shader::EntryPoint &entryPoint, const SHVar &input,
                           const VariableMap &composeWithVariables = VariableMap());
} // namespace gfx::shader

#endif /* F7DDD110_DF89_4AA4_A005_18C9BBFDFC40 */
