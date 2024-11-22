#ifndef F559C8F9_80A2_4AA0_A9A2_B854D54AF639
#define F559C8F9_80A2_4AA0_A9A2_B854D54AF639
#ifndef GFX_FEATURES_BASE_COLOR
#define GFX_FEATURES_BASE_COLOR

#include <gfx/shader/wgsl_mapping.hpp>
#include <gfx/enums.hpp>
#include <gfx/feature.hpp>
#include <gfx/params.hpp>
#include <gfx/shader/blocks.hpp>
#include <memory>

namespace gfx {
namespace features {

struct AlphaCutoff {
  static inline FeaturePtr create() {
    using namespace shader;
    using namespace shader::blocks;
    using shader::NumType;
    using shader::Types;

    FeaturePtr feature = std::make_shared<Feature>();
    feature->shaderParams["alphaCutoff"] = NumParamDecl(0.5f);

    auto alphaDiscard = makeBlock<Custom>([=](IGeneratorContext &context) {
      context.write("let cutoff = ");
      context.readBuffer("alphaCutoff", Types::Float, "object");
      context.write(";");
      context.write("let alpha = ");
      if(context.getDefinitions().globals.find("color") != context.getDefinitions().globals.end()) {
        context.readGlobal("color");
        context.write(".a;");
      } else {
        context.write("1.0;");
      }
      context.write(R"(
        if(alpha < cutoff) {
          discard;
        }
      )");
    });
    auto &alphaDiscardEntry =
        feature->shaderEntryPoints.emplace_back("alphaDiscard", ProgrammableGraphicsStage::Fragment, std::move(alphaDiscard));
    alphaDiscardEntry.dependencies.emplace_back("textureColor", DependencyType::After);
    alphaDiscardEntry.dependencies.emplace_back("applyLighting", DependencyType::Before);

    return feature;
  }
};

} // namespace features
} // namespace gfx

#endif // GFX_FEATURES_BASE_COLOR

#endif /* F559C8F9_80A2_4AA0_A9A2_B854D54AF639 */
