#include "test_data.hpp"
#include <gfx/context.hpp>
#include <gfx/drawable.hpp>
#include <gfx/features/base_color.hpp>
#include <gfx/features/transform.hpp>
#include <gfx/geom.hpp>
#include <gfx/mesh.hpp>
#include <gfx/renderer.hpp>
#include <gfx/view.hpp>
#include <spdlog/fmt/fmt.h>

using namespace gfx;

struct HeadlessRenderer {
	Context &context;
	Renderer renderer;

	WGPUTexture rtTexture{};
	WGPUTextureFormat rtFormat{};
	WGPUTextureView rtView{};
	int2 rtSize{};

	HeadlessRenderer(Context &context) : context(context), renderer(context) {}
	~HeadlessRenderer() {
		cleanupRenderTarget();
		renderer.cleanup();
	}

	void createRenderTarget(int2 res = int2(1280, 720)) {
		WGPU_SAFE_RELEASE(wgpuTextureViewRelease, rtView);
		WGPU_SAFE_RELEASE(wgpuTextureRelease, rtTexture);
		rtSize = res;

		WGPUTextureDescriptor textureDesc{};
		textureDesc.size.width = res.x;
		textureDesc.size.height = res.y;
		textureDesc.size.depthOrArrayLayers = 1;
		textureDesc.format = rtFormat = WGPUTextureFormat_RGBA8Unorm;
		textureDesc.label = "headlessRenderTarget";
		textureDesc.sampleCount = 1;
		textureDesc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
		textureDesc.mipLevelCount = 1;
		textureDesc.dimension = WGPUTextureDimension_2D;
		rtTexture = wgpuDeviceCreateTexture(context.wgpuDevice, &textureDesc);
		assert(rtTexture);

		WGPUTextureViewDescriptor viewDesc{};
		viewDesc.arrayLayerCount = 1;
		viewDesc.aspect = WGPUTextureAspect_All;
		viewDesc.mipLevelCount = 1;
		viewDesc.dimension = WGPUTextureViewDimension_2D;
		viewDesc.label = textureDesc.label;
		viewDesc.format = textureDesc.format;
		rtView = wgpuTextureCreateView(rtTexture, &viewDesc);

		renderer.setMainOutput(Renderer::MainOutput{
			.view = rtView,
			.format = rtFormat,
			.size = rtSize,
		});
	}

	void cleanupRenderTarget() {
		WGPU_SAFE_RELEASE(wgpuTextureViewRelease, rtView);
		WGPU_SAFE_RELEASE(wgpuTextureRelease, rtTexture);
	}

	TestFrame getTestFrame() {

		WGPUCommandEncoderDescriptor ceDesc{};
		WGPUCommandEncoder commandEncoder = wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &ceDesc);

		WGPUBufferDescriptor desc{};
		size_t bufferPitch = sizeof(TestFrame::pixel_t) * rtSize.x;
		desc.size = bufferPitch * rtSize.y;
		desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
		WGPUBuffer tempBuffer = wgpuDeviceCreateBuffer(context.wgpuDevice, &desc);

		WGPUImageCopyTexture src{};
		src.texture = rtTexture;
		src.aspect = WGPUTextureAspect_All;
		src.mipLevel = 0;
		WGPUImageCopyBuffer dst{};
		dst.buffer = tempBuffer;
		dst.layout.bytesPerRow = bufferPitch;
		WGPUExtent3D extent{};
		extent.width = rtSize.x;
		extent.height = rtSize.y;
		extent.depthOrArrayLayers = 1;
		wgpuCommandEncoderCopyTextureToBuffer(commandEncoder, &src, &dst, &extent);

		WGPUCommandBufferDescriptor finishDesc{};
		WGPUCommandBuffer copyCommandBuffer = wgpuCommandEncoderFinish(commandEncoder, &finishDesc);
		wgpuQueueSubmit(context.wgpuQueue, 1, &copyCommandBuffer);

		auto mapBufferCallback = [](WGPUBufferMapAsyncStatus status, void *userdata) {};
		wgpuBufferMapAsync(tempBuffer, WGPUMapMode_Read, 0, desc.size, mapBufferCallback, nullptr);
		context.sync();

		uint8_t *bufferData = (uint8_t *)wgpuBufferGetMappedRange(tempBuffer, 0, desc.size);
		TestFrame testFrame(bufferData, rtSize, TestFrameFormat::RGBA8, bufferPitch, false);

		wgpuBufferUnmap(tempBuffer);
		wgpuBufferRelease(tempBuffer);

		return testFrame;
	}
};

struct VertexP {
	float position[3];

	VertexP() = default;
	VertexP(const geom::VertexPNT &other) { setPosition(*(float3 *)other.position); }

	void setPosition(const float3 &v) { memcpy(position, &v.x, sizeof(float) * 3); }

	static std::vector<MeshVertexAttribute> getAttributes() {
		std::vector<MeshVertexAttribute> attribs;
		attribs.emplace_back("position", 3, VertexAttributeType::Float32);
		return attribs;
	}
};

struct VertexPC {
	float position[3];
	float color[4];

	VertexPC() = default;
	VertexPC(const geom::VertexPNT &other) {
		setPosition(*(float3 *)other.position);
		setColor(float4(1, 1, 1, 1));
	}

	void setPosition(const float3 &v) { memcpy(position, &v.x, sizeof(float) * 3); }
	void setColor(const float4 &v) { memcpy(color, &v.x, sizeof(float) * 4); }

	static std::vector<MeshVertexAttribute> getAttributes() {
		std::vector<MeshVertexAttribute> attribs;
		attribs.emplace_back("position", 3, VertexAttributeType::Float32);
		attribs.emplace_back("color", 4, VertexAttributeType::Float32);
		return attribs;
	}
};

template <typename T> std::vector<T> convertVertices(const std::vector<geom::VertexPNT> &input) {
	std::vector<T> result;
	for (auto &vert : input)
		result.emplace_back(vert);
	return result;
}

template <typename T> MeshPtr createMesh(const std::vector<T> &verts, const std::vector<geom::GeneratorBase::index_t> &indices) {
	MeshPtr mesh = std::make_shared<Mesh>();
	MeshFormat format{
		.vertexAttributes = T::getAttributes(),
	};
	mesh->update(format, verts.data(), verts.size() * sizeof(T), indices.data(), indices.size() * sizeof(geom::GeneratorBase::index_t));
	return mesh;
}

MeshPtr createSphereMesh() {
	geom::SphereGenerator gen;
	gen.generate();
	return createMesh(gen.vertices, gen.indices);
}

ViewPtr createTestProjectionView() {
	ViewPtr view = std::make_shared<View>();
	view->view = linalg::lookat_matrix(float3(0, 10.0f, 10.0f), float3(0, 0, 0), float3(0, 1, 0));
	view->proj = ViewPerspectiveProjection{
		degToRad(45.0f),
		FovDirection::Horizontal,
	};
	return view;
}

PipelineSteps createTestPipelineSteps() {
	return PipelineSteps{
		makeDrawablePipelineStep(RenderDrawablesStep{
			.features =
				{
					features::Transform::create(),
					features::BaseColor::create(),
				},
		}),
	};
}

TEST_CASE("Renderer capture", "[Renderer]") {
	std::shared_ptr<Context> context = std::make_shared<Context>();
	context->init();

	auto headlessRenderer = std::make_shared<HeadlessRenderer>(*context.get());
	headlessRenderer->createRenderTarget(int2(1280, 720));
	Renderer &renderer = headlessRenderer->renderer;

	MeshPtr mesh = createSphereMesh();
	DrawablePtr drawable = std::make_shared<Drawable>(mesh);
	ViewPtr view = std::make_shared<View>();

	DrawQueue queue;
	queue.add(drawable);
	renderer.render(queue, view, createTestPipelineSteps());

	TestData testData(TestPlatformId::get(*context.get()));
	TestFrame testFrame = headlessRenderer->getTestFrame();
	CHECK(testData.checkFrame("capture", testFrame));

	headlessRenderer.reset();
	context.reset();
}

TEST_CASE("Multiple vertex formats", "[Renderer]") {
	std::shared_ptr<Context> context = std::make_shared<Context>();
	context->init();

	auto headlessRenderer = std::make_shared<HeadlessRenderer>(*context.get());
	headlessRenderer->createRenderTarget(int2(1280, 720));
	Renderer &renderer = headlessRenderer->renderer;

	std::vector<MeshPtr> meshes;

	geom::SphereGenerator sphere;
	sphere.generate();
	meshes.push_back(createMesh(sphere.vertices, sphere.indices));
	meshes.push_back(createMesh(convertVertices<VertexP>(sphere.vertices), sphere.indices));
	meshes.push_back(createMesh(convertVertices<VertexPC>(sphere.vertices), sphere.indices));

	ViewPtr view = createTestProjectionView();

	DrawQueue queue;

	float offset = -2.0f;
	for (auto &mesh : meshes) {
		float4x4 transform = linalg::translation_matrix(float3(offset, 0.0f, 0.0f));
		DrawablePtr drawable = std::make_shared<Drawable>(mesh, transform);
		queue.add(drawable);
		offset += 2.0f;
	}

	renderer.render(queue, view, createTestPipelineSteps());

	TestData testData(TestPlatformId::get(*context.get()));
	TestFrame testFrame = headlessRenderer->getTestFrame();
	CHECK(testData.checkFrame("vertexFormats", testFrame));

	headlessRenderer.reset();
	context.reset();
}

TEST_CASE("Reference tracking", "[Renderer]") {
	std::shared_ptr<Context> context = std::make_shared<Context>();
	context->init();

	auto headlessRenderer = std::make_shared<HeadlessRenderer>(*context.get());
	headlessRenderer->createRenderTarget(int2(1280, 720));
	Renderer &renderer = headlessRenderer->renderer;

	std::weak_ptr<Mesh> meshWeak;
	std::weak_ptr<Drawable> drawableWeak;
	std::weak_ptr<MeshContextData> meshContextData;
	{
		MeshPtr mesh = createSphereMesh();
		meshWeak = std::weak_ptr(mesh);

		mesh->createContextDataConditional(*context.get());
		meshContextData = mesh->contextData;

		DrawablePtr drawable = std::make_shared<Drawable>(mesh);
		drawableWeak = std::weak_ptr(drawable);

		ViewPtr view = std::make_shared<View>();

		DrawQueue queue;
		queue.add(drawable);
		renderer.render(queue, view, createTestPipelineSteps());
	}

	CHECK(meshWeak.expired());
	CHECK(drawableWeak.expired());
	CHECK(!meshContextData.expired());

	context->sync();
	for (size_t i = 0; i < 2; i++) {
		renderer.swapBuffers();
	}

	// Should be released now
	CHECK(meshContextData.expired());

	headlessRenderer.reset();
	context.reset();
}