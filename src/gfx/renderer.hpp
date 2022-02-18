#pragma once
#include "fwd.hpp"
#include "gfx_wgpu.hpp"
#include "linalg.hpp"
#include "pipeline_step.hpp"
#include <functional>
#include <memory>

namespace gfx {

// Instance that caches render pipelines
struct RendererImpl;
struct Renderer {
	std::shared_ptr<RendererImpl> impl;

	struct MainOutput {
		WGPUTextureView view;
		WGPUTextureFormat format;
		int2 size;
	};

public:
	Renderer(Context &context);
	void render(const DrawQueue &drawQueue, std::vector<ViewPtr> views, const PipelineSteps &pipelineSteps);
	void render(const DrawQueue &drawQueue, ViewPtr view, const PipelineSteps &pipelineSteps);
	void setMainOutput(const MainOutput &output);

	void beginFrame();
	void endFrame();

	// Flushes rendering and cleans up all cached data kept by the renderer
	void cleanup();
};

} // namespace gfx
