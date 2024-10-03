#include "./context.hpp"
#include "./data.hpp"
#include "./renderer.hpp"
#include "renderer_utils.hpp"
#include <catch2/catch_all.hpp>
#include <gfx/fwd.hpp>
#include <gfx/context.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/features/velocity.hpp>
#include <gfx/loop.hpp>
#include <gfx/paths.hpp>
#include <gfx/texture.hpp>
#include <gfx/render_target.hpp>
#include <gfx/steps/effect.hpp>
#include <gfx/render_graph.hpp>
#include <gfx/render_graph_builder.hpp>
#include <spdlog/spdlog.h>

using namespace gfx;
using namespace gfx::steps;

static constexpr float comparisonTolerance = 0.05f;

TEST_CASE("Viewport render target", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();
  const auto &viewStack = renderer.getViewStack();

  MeshPtr cubeMesh = createCubeMesh();

  ViewPtr view = std::make_shared<View>();
  view->proj = ViewPerspectiveProjection{};
  view->view = linalg::lookat_matrix(float3(1.0f, 1.0f, 1.0f) * 3.0f, float3(0, 0, 0), float3(0, 1, 0));

  ViewPtr subView = std::make_shared<View>();

  DrawQueuePtr queue0 = std::make_shared<DrawQueue>();
  float4x4 transform;
  MeshDrawable::Ptr drawable;

  std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>("testTarget");
  rt->configure("color", WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm);
  rt->configure("depth", WGPUTextureFormat::WGPUTextureFormat_Depth32Float);

  transform = linalg::identity;
  drawable = std::make_shared<MeshDrawable>(cubeMesh, transform);
  drawable->parameters.set("baseColor", float4(1, 0, 1, 1));
  queue0->add(drawable);

  FeaturePtr blendFeature = std::make_shared<Feature>();
  blendFeature->state.set_blend(BlendState{
      .color = BlendComponent::Alpha,
      .alpha = BlendComponent::Opaque,
  });

  PipelineSteps stepsRT{
      makePipelineStep(RenderDrawablesStep{
          .drawQueue = queue0,
          .features =
              {
                  features::Transform::create(),
                  features::BaseColor::create(),
                  blendFeature,
              },
          .output = steps::getDefaultRenderStepOutput(),
      }),
  };

  auto colorOut = steps::getDefaultColorOutput();
  colorOut.clearValues = ClearValues::getColorValue(float4(0.2, 0.2, 0.2, 1.0));

  PipelineSteps stepsMain{
      makePipelineStep(NoopStep{
          .output = RenderStepOutput::make(colorOut),
      }),
      makePipelineStep(RenderFullscreenStep{
          .features =
              {
                  features::Transform::create(false, false),
                  []() {
                    auto baseColor = features::BaseColor::create();
                    baseColor->state.clear_blend();
                    return baseColor;
                  }(),
              },
          .parameters{
              .textures = {{"baseColorTexture", TextureParameter(rt->getAttachment("color"))}},
          },
          .output = RenderStepOutput::make(steps::getDefaultColorOutput()),
          .overlay = true,
      }),
  };

  TEST_RENDER_LOOP(testRenderer) {
    auto outputSize = testRenderer->getOutputSize();
    rt->resizeConditional(outputSize);

    int2 mainSize = viewStack.getOutput().size;
    Rect subViewport = Rect(int2(mainSize.x / 2, 0), mainSize / 2);

    renderer.pushView(ViewStack::Item{.renderTarget = rt});
    renderer.render(view, stepsRT);
    renderer.popView();

    renderer.pushView(ViewStack::Item{.viewport = subViewport});
    renderer.render(subView, stepsMain);
    renderer.popView();
  };
  CHECK(testRenderer->checkFrame("rendergraph_viewport", comparisonTolerance));

  testRenderer.reset();
}

TEST_CASE("Velocity", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();
  const auto &viewStack = renderer.getViewStack();

  MeshPtr cubeMesh = createCubeMesh();

  ViewPtr view = std::make_shared<View>();
  view->proj = ViewPerspectiveProjection{};
  view->view = linalg::lookat_matrix(float3(1.0f, 1.0f, 1.0f) * 3.0f, float3(0, 0, 0), float3(0, 1, 0));

  DrawQueuePtr queue = std::make_shared<DrawQueue>();
  float4x4 transform;
  MeshDrawable::Ptr drawable;

  std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>("testTarget");
  rt->configure("velocity", WGPUTextureFormat::WGPUTextureFormat_RG16Float);

  transform = linalg::identity;
  drawable = std::make_shared<MeshDrawable>(cubeMesh, transform);

  auto getTransform = [](double time) {
    float yWave = std::cos(time / pi2) * 0.5f;
    float4x4 transMat = linalg::translation_matrix(float3(0, yWave, 0));

    float angle = degToRad(time * 15.0f + std::cos(time / pi2 * 3.0f) * 20.0f);
    float4 rotQY = linalg::rotation_quat(float3(0, 1, 0), angle);
    float4x4 rotMat = linalg::rotation_matrix(rotQY);
    return linalg::mul(transMat, rotMat);
  };

  queue->add(drawable);

  auto getWaveTransform = [](double time, size_t axis) {
    float3 t{};
    t[axis] = std::cos(time / pi2 * 20.0f) * 1.0f;
    return linalg::mul(linalg::translation_matrix(t), linalg::scaling_matrix(float3(0.25f)));
  };

  MeshDrawable::Ptr drawable1 = std::make_shared<MeshDrawable>(cubeMesh, getWaveTransform(0.0f, 0));
  queue->add(drawable1);

  MeshDrawable::Ptr drawable2 = std::make_shared<MeshDrawable>(cubeMesh, getWaveTransform(0.0f, 1));
  queue->add(drawable2);

  PipelineSteps stepsRT{
      makePipelineStep(RenderDrawablesStep{
          .drawQueue = queue,
          .features =
              {
                  features::Transform::create(),
                  features::Velocity::create(),
              },
          .output =
              []() {
                auto output = steps::getDefaultRenderStepOutput();
                output.attachments.emplace_back(RenderStepOutput::Named("velocity", WGPUTextureFormat_RG16Float));
                return output;
              }(),
      }),
  };

  auto postFeature = []() {
    using namespace shader;
    using namespace shader::blocks;

    auto feature = std::make_shared<Feature>();
    auto code = makeCompoundBlock();
    code->appendLine("let vel = ", SampleTexture("velocity"), ".xy");
    code->appendLine(WriteOutput("color", Types::Float4, "vec4<f32>(vel.x*0.5+0.5, vel.y*0.5+0.5, 0.0, 1.0)"));
    feature->shaderEntryPoints.emplace_back("", ProgrammableGraphicsStage::Fragment, std::move(code));
    feature->textureParams.emplace("velocity", TextureParamDecl());
    return feature;
  }();

  PipelineSteps stepsMain{
      makePipelineStep(RenderFullscreenStep{
          .features = steps::withDefaultFullscreenFeatures(postFeature),
          .parameters{
              .textures = {{"velocity", TextureParameter(rt->getAttachment("velocity"))}},
          },
          .output = RenderStepOutput::make(getDefaultColorOutput()),
      }),
  };

  double t = 3.0;
  double timeStep = 1.0 / 60.0;

  auto renderFrame = [&]() {
    auto outputSize = testRenderer->getOutputSize();
    rt->resizeConditional(outputSize);

    drawable->transform = getTransform(t);
    drawable1->transform = getWaveTransform(t, 0);
    drawable2->transform = getWaveTransform(t, 1);

    renderer.pushView(ViewStack::Item{.renderTarget = rt});
    renderer.render(view, stepsRT);
    renderer.popView();

    renderer.render(view, stepsMain);

    t += timeStep;
  };

  // dummy frame to intialize previous transforms
  testRenderer->begin();
  renderFrame();
  testRenderer->end();

  TEST_RENDER_LOOP(testRenderer) { renderFrame(); };
  CHECK(testRenderer->checkFrame("rendergraph_velocity", 1.5f));

  testRenderer.reset();
}

TEST_CASE("Format conversion", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();

  ViewPtr view = std::make_shared<View>();

  auto blk = makeCompoundBlock();
  PipelineSteps steps;

  // Pass that writes a fixed color (linear)
  {
    blk->appendLine(WriteOutput("color", Types::Float4, "vec4<f32>(1.0, 0.5, 0.25, 1.0)"));
    steps.emplace_back(Effect::create(RenderStepInput{},
                                      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA32Float)),
                                      std::move(blk)));
  }

  // Dummy pass that doesn't write colors
  // Convert to RGBA8UnormSrgb (srgb)
  {
    steps.emplace_back(Effect::create(RenderStepInput{},
                                      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8UnormSrgb)),
                                      std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*steps.back().get());
    step.overlay = true;
    auto &f = step.features.emplace_back(std::make_shared<Feature>());
    f->state.set_colorWrite(WGPUColorWriteMask::WGPUColorWriteMask_None);
  }

  // Dummy pass that doesn't write colors
  // convert to WGPUTextureFormat_RGBA16Float (linear)
  {
    steps.emplace_back(Effect::create(RenderStepInput{},
                                      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA16Float)),
                                      std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*steps.back().get());
    step.overlay = true;
    auto &f = step.features.emplace_back(std::make_shared<Feature>());
    f->state.set_colorWrite(WGPUColorWriteMask::WGPUColorWriteMask_None);
  }

  // Dummy pass that doesn't write colors
  // convert back to default (srgb)
  {
    steps.emplace_back(Effect::create(RenderStepInput{},
                                      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8UnormSrgb)),
                                      std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*steps.back().get());
    step.overlay = true;
    auto &f = step.features.emplace_back(std::make_shared<Feature>());
    f->state.set_colorWrite(WGPUColorWriteMask::WGPUColorWriteMask_None);
  }

  TEST_RENDER_LOOP(testRenderer) { renderer.render(view, steps); };
  CHECK(testRenderer->checkFrame("rendergraph_format_conversion", 1.5f));

  testRenderer.reset();
}

TEST_CASE("Graph output size mismatch", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();

  ViewPtr view = std::make_shared<View>();

  auto blk = makeCompoundBlock();
  PipelineSteps steps;

  // write "test" target
  {
    blk->appendLine(WriteOutput("test", Types::Float4, "vec4<f32>(1.0, 0.0, 0.0, 1.0)"));
    auto &stepPtr = steps.emplace_back(
        Effect::create(RenderStepInput{}, RenderStepOutput::make(RenderStepOutput::Named("test", WGPUTextureFormat_RGBA32Float)),
                       std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*stepPtr.get());
    step.output->outputSizing = RenderStepOutput::RelativeToMainSize{.scale = float2(0.5f)};
  }

  // write "color" & "test" target with mismatching sizes
  {
    blk = makeCompoundBlock();
    blk->appendLine(WriteOutput("color", Types::Float4, "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    blk->appendLine(WriteOutput("test", Types::Float4, "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    auto &stepPtr =
        steps.emplace_back(Effect::create(RenderStepInput{},
                                          RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA32Float),
                                                                 RenderStepOutput::Named("test", WGPUTextureFormat_RGBA32Float)),
                                          std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*stepPtr.get());
    step.overlay = true;
  }

  TEST_RENDER_LOOP(testRenderer) { renderer.render(view, steps); };

  testRenderer.reset();
}

TEST_CASE("Write to texture auto-size", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();

  ViewPtr view = std::make_shared<View>();

  auto blk = makeCompoundBlock();
  PipelineSteps steps;

  TexturePtr outputTexture = std::make_shared<Texture>("outputText");
  outputTexture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(100),
              .flags = TextureFormatFlags::RenderAttachment,
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA32Float,
          },
  });

  // write "color" & "test" target with mismatching sizes
  {
    blk = makeCompoundBlock();
    blk->appendLine(WriteOutput("color", Types::Float4, "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    blk->appendLine(WriteOutput("test", Types::Float4, "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    auto &stepPtr =
        steps.emplace_back(Effect::create(RenderStepInput{},
                                          RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA32Float),
                                                                 RenderStepOutput::Texture("test", outputTexture)),
                                          std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*stepPtr.get());
    step.overlay = true;
  }

  TEST_RENDER_LOOP(testRenderer) { renderer.render(view, steps); };

  testRenderer.reset();
}

TEST_CASE("Graph evaluate output size mismatch", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();

  ViewPtr view = std::make_shared<View>();

  auto blk = makeCompoundBlock();
  PipelineSteps steps;

  TexturePtr outputTexture = std::make_shared<Texture>("outputText");
  outputTexture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(100),
              .flags = TextureFormatFlags::RenderAttachment,
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA32Float,
          },
  });

  // write "color" & "test" target with mismatching sizes
  {
    blk = makeCompoundBlock();
    blk->appendLine(WriteOutput("color", Types::Float4, "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    blk->appendLine(WriteOutput("test", Types::Float4, "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    auto &stepPtr =
        steps.emplace_back(Effect::create(RenderStepInput{},
                                          RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA32Float),
                                                                 RenderStepOutput::Texture("test", outputTexture)),
                                          std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*stepPtr.get());
    step.output->outputSizing = RenderStepOutput::ManualSize{};
    step.overlay = true;
  }

  CHECK_THROWS(TEST_RENDER_LOOP(testRenderer) { renderer.render(view, steps); });

  testRenderer.reset();
}

TEST_CASE("Read and write same texture", "[RenderGraph]") {
  auto testRenderer = createTestRenderer();
  Renderer &renderer = *testRenderer->renderer.get();

  ViewPtr view = std::make_shared<View>();

  auto blk = makeCompoundBlock();
  PipelineSteps steps;

  TexturePtr outputTexture = std::make_shared<Texture>("outputText");
  outputTexture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(100),
              .flags = TextureFormatFlags::RenderAttachment,
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA32Float,
          },
  });

  // write "color" & "test" target with mismatching sizes
  {
    blk = makeCompoundBlock();
    blk->appendLine(WriteOutput("test", Types::Float4, SampleTexture("test"), "+", "vec4<f32>(0.0, 0.5, 0.0, 1.0)"));
    auto &stepPtr = steps.emplace_back(Effect::create(RenderStepInput::make(RenderStepInput::Texture("test", outputTexture)),
                                                      RenderStepOutput::make(RenderStepOutput::Texture("test", outputTexture)),
                                                      std::move(blk)));
    auto &step = std::get<RenderFullscreenStep>(*stepPtr.get());
    step.output->outputSizing = RenderStepOutput::ManualSize{};
    step.overlay = true;
  }

  TEST_RENDER_LOOP(testRenderer) { renderer.render(view, steps); };

  testRenderer.reset();
}
