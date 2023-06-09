#include "./data.hpp"
#include "./renderer.hpp"
#include "./renderer_utils.hpp"
#include <catch2/catch_all.hpp>
#include <gfx/context.hpp>
#include <gfx/drawable.hpp>
#include <gfx/gltf/gltf.hpp>
#include <gfx/mesh.hpp>
#include <gfx/paths.hpp>
#include <spdlog/fmt/fmt.h>

using namespace gfx;

static constexpr float comparisonTolerance = 0.09f;

struct TestModelDesc {
  const char *name;
  float scale;
  float3 offset{};
};

// clang-format off
static TestModelDesc testModels[] = {
    {"Fox", 1.0f / 100.0f, float3(0, 0, 0)},
    {"Duck", 1.0f / 2.0f, float3(0, 0, 0)},
    {"BarramundiFish", 2.7f, float3(0, 0, 0)},
    {"Avocado", 12.2f, float3(0, 0, 0)},
    {"Lantern", 1.0f / 20.0f, float3(0, 0.f, 0)},
};
// clang-format on

TEST_CASE("glTF sample models", "[glTF]") {
  auto testRenderer = createTestRenderer(int2(512, 512));

  ViewPtr view = std::make_shared<View>();
  view->proj = ViewPerspectiveProjection{};
  view->view = linalg::lookat_matrix(float3(2.f, 2.0f, 2.0f), float3(0, 0, 0.0f), float3(0, 1, 0));

  DrawQueuePtr queue = std::make_shared<DrawQueue>();
  auto pipeline = PipelineSteps{
      makePipelineStep(RenderDrawablesStep{
          .drawQueue = queue,
          .features =
              {
                  features::Transform::create(),
                  features::BaseColor::create(),
              },
      }),
  };

  for (auto &testModel : testModels) {
    DYNAMIC_SECTION("Test " << testModel.name) {
      auto glbPath = gfx::resolveDataPath(fmt::format("external/glTF-Sample-Models/2.0/{0}/glTF-Binary/{0}.glb", testModel.name));
      auto gltfScene = loadGltfFromFile(glbPath.string().c_str()).root;
      assert(gltfScene);

      float4x4 origTransform = gltfScene->trs.getMatrix();
      gltfScene->trs = linalg::mul(origTransform, linalg::mul(linalg::translation_matrix(testModel.offset),
                                                              linalg::scaling_matrix(float3(testModel.scale))));

      queue->clear();
      queue->add(gltfScene);
      TEST_RENDER_LOOP(testRenderer) { testRenderer->renderer->render(view, pipeline); };

      std::string frameName = fmt::format("glTF.{}", testModel.name);
      CHECK(testRenderer->checkFrame(frameName.c_str(), comparisonTolerance));
    }
  }
}
