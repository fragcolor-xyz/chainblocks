#include "./context.hpp"
#include "./data.hpp"
#include "./renderer.hpp"
#include "renderer_utils.hpp"
#include <catch2/catch_all.hpp>
#include <gfx/fwd.hpp>
#include <gfx/context.hpp>
#include <gfx/steps/effect.hpp>
#include <gfx/render_graph.hpp>
#include <gfx/render_graph_builder.hpp>
#include <spdlog/spdlog.h>

using namespace gfx;
using namespace gfx::steps;

TEST_CASE("RenderGraph read from output", "[RenderGraphHeadless]") {
  auto colorOutput = RenderStepOutput::Named("color");
  auto depthOutput = RenderStepOutput::Named("depth");

  RenderStepOutput output = RenderStepOutput::make(colorOutput, depthOutput);

  PipelineSteps steps;
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .output = output,
      .overlay = true,
  }));

  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = RenderStepOutput::make(colorOutput),
      .overlay = true,
  }));

  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("depth", "color"),
      .output = RenderStepOutput::make(colorOutput),
      .overlay = true,
  }));

  detail::RenderGraphBuilder builder;
  builder.outputs.emplace_back("color");
  for (auto &step : steps) {
    builder.addNode(step, 0);
  }
  std::optional<detail::RenderGraph> graph = builder.build();
  CHECK(graph);
}

TEST_CASE("RenderGraph with texture bindings", "[RenderGraphHeadless]") {
  auto colorOutput = RenderStepOutput::Named("color");
  auto depthOutput = RenderStepOutput::Named("depth", WGPUTextureFormat::WGPUTextureFormat_Depth32Float);

  RenderStepOutput output = RenderStepOutput::make(colorOutput, depthOutput);

  TexturePtr texture = std::make_shared<Texture>("testTexture");
  texture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(1024),
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm,
          },
  });

  PipelineSteps steps;
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make(RenderStepInput::Texture("color", TextureSubResource(texture))),
      .output = output,
      .overlay = true,
  }));

  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .output = RenderStepOutput::make(colorOutput),
      .overlay = true,
  }));

  detail::RenderGraphBuilder builder;
  builder.outputs.emplace_back("color");
  for (auto &step : steps) {
    builder.addNode(step, 0);
  }
  std::optional<detail::RenderGraph> graph = builder.build();
  CHECK(graph);
}

TEST_CASE("RenderGraph read/write chain", "[RenderGraphHeadless]") {
  auto colorOutput = RenderStepOutput::Named("color");

  RenderStepOutput output = RenderStepOutput::make(colorOutput);
  RenderStepInput input = RenderStepInput::make("color");

  TexturePtr texture = std::make_shared<Texture>("testTexture");
  texture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(1024),
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm,
          },
  });

  PipelineSteps steps;
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make(RenderStepInput::Texture("color", TextureSubResource(texture))),
      .output = output,
      .overlay = true,
  }));

  for (size_t i = 0; i < 5; i++) {
    steps.emplace_back(makePipelineStep(RenderFullscreenStep{
        .input = input,
        .output = output,
        .overlay = true,
    }));
  }

  RenderStepOutput outputWithClear = RenderStepOutput::make(
      RenderStepOutput::Named("color", WGPUTextureFormat::WGPUTextureFormat_RGBA8UnormSrgb, ClearValues::getDefaultColor()));
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = input,
      .output = outputWithClear,
      .overlay = true,
  }));

  detail::RenderGraphBuilder builder;
  builder.outputs.emplace_back("color");
  for (auto &step : steps) {
    builder.addNode(step, 0);
  }
  std::optional<detail::RenderGraph> graph = builder.build();
  CHECK(graph);
}

TEST_CASE("RenderGraph different formats", "[RenderGraphHeadless]") {
  auto colorOutput = RenderStepOutput::Named("color");

  RenderStepOutput output = RenderStepOutput::make(colorOutput);
  RenderStepInput input = RenderStepInput::make("color");

  TexturePtr texture = std::make_shared<Texture>("testTexture");
  texture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(1024),
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm,
          },
  });

  PipelineSteps steps;

  SECTION("Without transition") {
    steps.clear();
    steps.emplace_back(makePipelineStep(RenderFullscreenStep{
        .output = RenderStepOutput::make(
            RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor())),
    }));

    steps.emplace_back(makePipelineStep(RenderFullscreenStep{
        .output = RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8UnormSrgb)),
        .overlay = false,
    }));

    detail::RenderGraphBuilder builder;
    builder.outputs.emplace_back("color");
    for (auto &step : steps) {
      builder.addNode(step, 0);
    }
    std::optional<detail::RenderGraph> graph = builder.build();
    CHECK(graph);
  }

  SECTION("With Transition") {
    steps.clear();
    steps.emplace_back(makePipelineStep(RenderFullscreenStep{
        .output = RenderStepOutput::make(
            RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor())),
    }));

    steps.emplace_back(makePipelineStep(RenderFullscreenStep{
        .output = RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8UnormSrgb)),
        .overlay = true,
    }));

    detail::RenderGraphBuilder builder;
    for (auto &step : steps) {
      builder.addNode(step, 0);
    }
    std::optional<detail::RenderGraph> graph = builder.build();
    CHECK(graph);
  }
}

TEST_CASE("RenderGraph relative sizes", "[RenderGraphHeadless]") {
  auto colorOutput = RenderStepOutput::Named("color");

  RenderStepOutput output = RenderStepOutput::make(colorOutput);
  RenderStepInput input = RenderStepInput::make("color");

  TexturePtr texture = std::make_shared<Texture>("testTexture");
  texture->init(TextureDescGPUOnly{
      .format =
          TextureFormat{
              .resolution = int2(1024),
              .pixelFormat = WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm,
          },
  });

  PipelineSteps steps;

  auto o0 =
      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor()));
  o0.outputSizing = RenderStepOutput::RelativeToMainSize{.scale = float2(0.5f)};
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .output = o0,
  }));

  auto o1 =
      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor()));
  o1.outputSizing = RenderStepOutput::RelativeToMainSize{.scale = float2(1.0f)};
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = o1,
  }));

  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .output =
          RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor())),
  }));

  auto o2 =
      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor()));
  o2.outputSizing = RenderStepOutput::RelativeToInputSize{.name = "color", .scale = float2(0.33333f)};
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = o2,
  }));

  auto o3 =
      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor()));
  o3.outputSizing = RenderStepOutput::RelativeToInputSize{.name = "color"};
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = o3,
  }));

  auto o4 =
      RenderStepOutput::make(RenderStepOutput::Named("color", WGPUTextureFormat_RGBA8Unorm, ClearValues::getDefaultColor()));
  o4.outputSizing = RenderStepOutput::RelativeToInputSize{.name = "color"};
  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = o4,
  }));

  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = o4,
  }));

  steps.emplace_back(makePipelineStep(RenderFullscreenStep{
      .input = RenderStepInput::make("color"),
      .output = o4,
  }));

  detail::RenderGraphBuilder builder;
  for (auto &step : steps) {
    builder.addNode(step, 0);
  }
  std::optional<detail::RenderGraph> graph = builder.build();
  CHECK(graph);
}
