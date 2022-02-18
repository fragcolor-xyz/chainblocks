#include "renderer.hpp"
#include "context.hpp"
#include "drawable.hpp"
#include "feature.hpp"
#include "hasherxxh128.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "params.hpp"
#include "renderer_types.hpp"
#include "shader/blocks.hpp"
#include "shader/entry_point.hpp"
#include "shader/generator.hpp"
#include "shader/textures.hpp"
#include "shader/uniforms.hpp"
#include "texture.hpp"
#include "texture_placeholder.hpp"
#include "view.hpp"
#include "view_texture.hpp"
#include <magic_enum.hpp>
#include <spdlog/spdlog.h>

#define GFX_RENDERER_MAX_BUFFERED_FRAMES (2)

namespace gfx {
using shader::TextureBindingLayout;
using shader::TextureBindingLayoutBuilder;
using shader::UniformBufferLayout;
using shader::UniformBufferLayoutBuilder;
using shader::UniformLayout;

PipelineStepPtr makeDrawablePipelineStep(RenderDrawablesStep &&step) { return std::make_shared<PipelineStep>(std::move(step)); }

typedef uint16_t TextureId;

struct TextureIds {
	std::vector<TextureId> textures;
	bool operator==(const TextureIds &other) const = default;
	auto operator<=>(const TextureIds &other) const = default;
	bool operator<(const TextureIds &other) const = default;
};

struct TextureIdMap {
private:
	std::map<Texture *, TextureId> mapping;
	std::vector<std::shared_ptr<TextureContextData>> textureData;

public:
	TextureId assign(Context &context, Texture *texture) {
		if (!texture)
			return ~0;

		auto it = mapping.find(texture);
		if (it != mapping.end()) {
			return it->second;
		} else {
			texture->createContextDataConditional(context);

			TextureId id = TextureId(textureData.size());
			textureData.emplace_back(texture->contextData);
			mapping.insert_or_assign(texture, id);
			return id;
		}
	}
	TextureContextData *resolve(TextureId id) const {
		if (id == TextureId(~0))
			return nullptr;
		else {
			return textureData[id].get();
		}
	}

	void reset() { mapping.clear(); }
};

struct SortableDrawable;
struct DrawGroupKey {
	MeshContextData *meshData{};
	TextureIds textures{};

	DrawGroupKey() = default;
	DrawGroupKey(const Drawable &drawable, const TextureIds &textureIds) : meshData(drawable.mesh->contextData.get()), textures(textureIds) {}
	bool operator!=(const DrawGroupKey &other) const = default;
	auto operator<=>(const DrawGroupKey &other) const = default;
	bool operator<(const DrawGroupKey &other) const = default;
};

struct SortableDrawable {
	Drawable *drawable{};
	TextureIds textureIds;
	DrawGroupKey key;
	float projectedDepth = 0.0f;
};

struct CachedPipeline {
	struct DrawGroup {
		DrawGroupKey key;
		size_t startIndex;
		size_t numInstances;
		DrawGroup(const DrawGroupKey &key, size_t startIndex, size_t numInstances) : key(key), startIndex(startIndex), numInstances(numInstances) {}
	};

	MeshFormat meshFormat;
	std::vector<const Feature *> features;
	UniformBufferLayout objectBufferLayout;
	TextureBindingLayout textureBindingLayout;
	DrawData baseDrawData;

	WGPURenderPipeline pipeline = {};
	WGPUShaderModule shaderModule = {};
	WGPUPipelineLayout pipelineLayout = {};
	std::vector<WGPUBindGroupLayout> bindGroupLayouts;

	std::vector<Drawable *> drawables;
	std::vector<SortableDrawable> drawablesSorted;
	std::vector<DrawGroup> drawGroups;
	TextureIdMap textureIdMap;

	DynamicWGPUBufferPool instanceBufferPool;

	void resetDrawables() {
		drawables.clear();
		drawablesSorted.clear();
		drawGroups.clear();
		textureIdMap.reset();
	}

	void resetPools() { instanceBufferPool.reset(); }

	void release() {
		wgpuPipelineLayoutRelease(pipelineLayout);
		wgpuShaderModuleRelease(shaderModule);
		wgpuRenderPipelineRelease(pipeline);
		for (WGPUBindGroupLayout &layout : bindGroupLayouts) {
			wgpuBindGroupLayoutRelease(layout);
		}
	}
};
typedef std::shared_ptr<CachedPipeline> CachedPipelinePtr;

static void packDrawData(uint8_t *outData, size_t outSize, const UniformBufferLayout &layout, const DrawData &drawData) {
	size_t layoutIndex = 0;
	for (auto &fieldName : layout.fieldNames) {
		auto drawDataIt = drawData.data.find(fieldName);
		if (drawDataIt != drawData.data.end()) {
			const UniformLayout &itemLayout = layout.items[layoutIndex];
			ShaderParamType drawDataFieldType = getParamVariantType(drawDataIt->second);
			if (itemLayout.type != drawDataFieldType) {
				spdlog::warn("WEBGPU: Field type mismatch layout:{} drawData:{}", magic_enum::enum_name(itemLayout.type),
							 magic_enum::enum_name(drawDataFieldType));
				continue;
			}

			packParamVariant(outData + itemLayout.offset, outSize - itemLayout.offset, drawDataIt->second);
		}
		layoutIndex++;
	}
}

struct CachedDrawableData {
	Drawable *drawable;
	CachedPipeline *pipeline;
	Hash128 pipelineHash;
};

struct CachedViewData {
	DynamicWGPUBufferPool viewBuffers;
	float4x4 projectionMatrix;
	~CachedViewData() {}

	void resetPools() { viewBuffers.reset(); }
};
typedef std::shared_ptr<CachedViewData> CachedViewDataPtr;

struct FrameReferences {
	std::vector<std::shared_ptr<ContextData>> contextDataReferences;
	void clear() { contextDataReferences.clear(); }
};

struct RendererImpl {
	Context &context;
	WGPUSupportedLimits deviceLimits = {};

	UniformBufferLayout viewBufferLayout;

	Renderer::MainOutput mainOutput;
	bool shouldUpdateMainOutputFromContext = false;

	Swappable<std::vector<std::function<void()>>, GFX_RENDERER_MAX_BUFFERED_FRAMES> postFrameQueue;
	Swappable<FrameReferences, GFX_RENDERER_MAX_BUFFERED_FRAMES> frameReferences;

	std::unordered_map<const View *, CachedViewDataPtr> viewCache;
	std::unordered_map<Hash128, CachedPipelinePtr> pipelineCache;
	std::unordered_map<const Drawable *, CachedDrawableData> drawableCache;

	size_t frameIndex = 0;
	const size_t maxBufferedFrames = GFX_RENDERER_MAX_BUFFERED_FRAMES;

	bool mainOutputWrittenTo = false;

	std::unique_ptr<ViewTexture> depthTexture = std::make_unique<ViewTexture>(WGPUTextureFormat_Depth24Plus);
	std::unique_ptr<PlaceholderTexture> placeholderTexture;

	RendererImpl(Context &context) : context(context) {
		UniformBufferLayoutBuilder viewBufferLayoutBuilder;
		viewBufferLayoutBuilder.push("view", ShaderParamType::Float4x4);
		viewBufferLayoutBuilder.push("proj", ShaderParamType::Float4x4);
		viewBufferLayoutBuilder.push("invView", ShaderParamType::Float4x4);
		viewBufferLayoutBuilder.push("invProj", ShaderParamType::Float4x4);
		viewBufferLayoutBuilder.push("viewport", ShaderParamType::Float4);
		viewBufferLayout = viewBufferLayoutBuilder.finalize();

		gfxWgpuDeviceGetLimits(context.wgpuDevice, &deviceLimits);

		placeholderTexture = std::make_unique<PlaceholderTexture>(context, int2(2, 2), float4(1, 1, 1, 1));
	}

	~RendererImpl() {
		context.sync();
		swapBuffers();
	}

	size_t alignToMinUniformOffset(size_t size) const { return alignTo(size, deviceLimits.limits.minUniformBufferOffsetAlignment); }
	size_t alignToArrayBounds(size_t size, size_t elementAlign) const { return alignTo(size, elementAlign); }

	size_t alignTo(size_t size, size_t alignTo) const {
		size_t alignment = alignTo;
		if (alignment == 0)
			return size;

		size_t remainder = size % alignment;
		if (remainder > 0) {
			return size + (alignment - remainder);
		}
		return size;
	}

	void updateMainOutputFromContext() {
		mainOutput.format = context.getMainOutputFormat();
		mainOutput.view = context.getMainOutputTextureView();
		mainOutput.size = context.getMainOutputSize();
	}

	void renderViews(const DrawQueue &drawQueue, const std::vector<ViewPtr> &views, const PipelineSteps &pipelineSteps) {
		for (auto &view : views) {
			renderView(drawQueue, view, pipelineSteps);
		}
	}

	void renderView(const DrawQueue &drawQueue, ViewPtr view, const PipelineSteps &pipelineSteps) {
		if (shouldUpdateMainOutputFromContext) {
			updateMainOutputFromContext();
		}

		View *viewPtr = view.get();

		Rect viewport;
		int2 viewSize;
		if (viewPtr->viewport) {
			viewSize = viewPtr->viewport->getSize();
			viewport = viewPtr->viewport.value();
		} else {
			viewSize = mainOutput.size;
			viewport = Rect(0, 0, viewSize.x, viewSize.y);
		}

		auto it = viewCache.find(viewPtr);
		if (it == viewCache.end()) {
			it = viewCache.insert(std::make_pair(viewPtr, std::make_shared<CachedViewData>())).first;
		}
		CachedViewData &viewData = *it->second.get();

		DrawData viewDrawData;
		viewDrawData.setParam("view", viewPtr->view);
		viewDrawData.setParam("invView", linalg::inverse(viewPtr->view));
		float4x4 projMatrix = viewData.projectionMatrix = viewPtr->getProjectionMatrix(viewSize);
		viewDrawData.setParam("proj", projMatrix);
		viewDrawData.setParam("invProj", linalg::inverse(projMatrix));

		DynamicWGPUBuffer &viewBuffer = viewData.viewBuffers.allocateBuffer(viewBufferLayout.size);
		viewBuffer.resize(context.wgpuDevice, viewBufferLayout.size, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "viewUniform");

		std::vector<uint8_t> stagingBuffer;
		stagingBuffer.resize(viewBufferLayout.size);
		packDrawData(stagingBuffer.data(), stagingBuffer.size(), viewBufferLayout, viewDrawData);
		wgpuQueueWriteBuffer(context.wgpuQueue, viewBuffer, 0, stagingBuffer.data(), stagingBuffer.size());

		for (auto &step : pipelineSteps) {
			std::visit(
				[&](auto &&arg) {
					using T = std::decay_t<decltype(arg)>;
					if constexpr (std::is_same_v<T, RenderDrawablesStep>) {
						renderDrawables(drawQueue, arg, view, viewport, viewBuffer);
					}
				},
				*step.get());
		}
	}

	WGPUBuffer createTransientBuffer(size_t size, WGPUBufferUsageFlags flags, const char *label = nullptr) {
		WGPUBufferDescriptor desc = {};
		desc.size = size;
		desc.label = label;
		desc.usage = flags;
		WGPUBuffer buffer = wgpuDeviceCreateBuffer(context.wgpuDevice, &desc);
		onFrameCleanup([buffer]() { wgpuBufferRelease(buffer); });
		return buffer;
	}

	void addFrameReference(std::shared_ptr<ContextData> &&contextData) {
		frameReferences(frameIndex).contextDataReferences.emplace_back(std::move(contextData));
	}

	void onFrameCleanup(std::function<void()> &&callback) { postFrameQueue(frameIndex).emplace_back(std::move(callback)); }

	void beginFrame() {
		mainOutputWrittenTo = false;
		swapBuffers();
	}

	void endFrame() {
		// FIXME
		if (!mainOutputWrittenTo && !context.isHeadless()) {
			submitDummyRenderPass();
		}
	}

	void submitDummyRenderPass() {
		WGPUCommandEncoderDescriptor desc = {};
		WGPUCommandEncoder commandEncoder = wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &desc);

		WGPURenderPassDescriptor passDesc = {};
		passDesc.colorAttachmentCount = 1;

		WGPURenderPassColorAttachment mainAttach = {};
		mainAttach.clearColor = WGPUColor{.r = 0.1f, .g = 0.1f, .b = 0.1f, .a = 1.0f};
		mainAttach.loadOp = WGPULoadOp_Clear;
		mainAttach.view = context.getMainOutputTextureView();
		mainAttach.storeOp = WGPUStoreOp_Store;

		passDesc.colorAttachments = &mainAttach;
		passDesc.colorAttachmentCount = 1;
		WGPURenderPassEncoder passEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder, &passDesc);
		wgpuRenderPassEncoderEndPass(passEncoder);

		WGPUCommandBufferDescriptor cmdBufDesc{};
		WGPUCommandBuffer cmdBuf = wgpuCommandEncoderFinish(commandEncoder, &cmdBufDesc);

		context.submit(cmdBuf);
	}

	void swapBuffers() {
		frameIndex = (frameIndex + 1) % maxBufferedFrames;

		frameReferences(frameIndex).clear();
		auto &postFrameQueue = this->postFrameQueue(frameIndex);
		for (auto &cb : postFrameQueue) {
			cb();
		}
		postFrameQueue.clear();

		for (auto &pair : viewCache) {
			pair.second->resetPools();
		}

		for (auto &pair : pipelineCache) {
			pair.second->resetPools();
		}
	}

	struct Bindable {
		WGPUBuffer buffer;
		UniformBufferLayout layout;
		size_t overrideSize = ~0;
		Bindable(WGPUBuffer buffer, UniformBufferLayout layout, size_t overrideSize = ~0) : buffer(buffer), layout(layout), overrideSize(overrideSize) {}
	};

	WGPUBindGroup createBindGroup(WGPUDevice device, WGPUBindGroupLayout layout, std::vector<Bindable> bindables) {
		WGPUBindGroupDescriptor desc = {};
		desc.label = "view";
		std::vector<WGPUBindGroupEntry> entries;

		size_t counter = 0;
		for (auto &bindable : bindables) {
			WGPUBindGroupEntry &entry = entries.emplace_back(WGPUBindGroupEntry{});
			entry.binding = counter++;
			entry.buffer = bindable.buffer;
			entry.size = (bindable.overrideSize != size_t(~0)) ? bindable.overrideSize : bindable.layout.size;
		}

		desc.entries = entries.data();
		desc.entryCount = entries.size();
		desc.layout = layout;
		return wgpuDeviceCreateBindGroup(device, &desc);
	}

	void fillInstanceBuffer(DynamicWGPUBuffer &instanceBuffer, CachedPipeline &cachedPipeline, View *view) {
		size_t alignedObjectBufferSize = alignToArrayBounds(cachedPipeline.objectBufferLayout.size, cachedPipeline.objectBufferLayout.maxAlignment);
		size_t numObjects = cachedPipeline.drawables.size();
		size_t instanceBufferLength = numObjects * alignedObjectBufferSize;
		instanceBuffer.resize(context.wgpuDevice, instanceBufferLength, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "objects");

		std::vector<uint8_t> drawDataTempBuffer;
		drawDataTempBuffer.resize(instanceBufferLength);
		for (size_t i = 0; i < numObjects; i++) {
			Drawable *drawable = cachedPipeline.drawablesSorted[i].drawable;

			DrawData objectDrawData = cachedPipeline.baseDrawData;
			objectDrawData.setParam("world", drawable->transform);

			// Grab draw data from material
			if (Material *material = drawable->material.get()) {
				for (auto &pair : material->parameters.basic) {
					objectDrawData.setParam(pair.first, pair.second);
				}
			}

			// Grab draw data from drawable
			for (auto &pair : drawable->parameters.basic) {
				objectDrawData.setParam(pair.first, pair.second);
			}

			// Grab draw data from features
			FeatureCallbackContext callbackContext{context, view, drawable};
			for (const Feature *feature : cachedPipeline.features) {
				for (const FeatureDrawDataFunction &drawDataGenerator : feature->drawData) {
					// TODO: Catch mismatch errors here
					drawDataGenerator(callbackContext, objectDrawData);
				}
			}

			size_t bufferOffset = alignedObjectBufferSize * i;
			size_t remainingBufferLength = instanceBufferLength - bufferOffset;

			packDrawData(drawDataTempBuffer.data() + bufferOffset, remainingBufferLength, cachedPipeline.objectBufferLayout, objectDrawData);
		}

		wgpuQueueWriteBuffer(context.wgpuQueue, instanceBuffer, 0, drawDataTempBuffer.data(), drawDataTempBuffer.size());
	}

	void buildBaseObjectParameters(CachedPipeline &cachedPipeline) {
		for (const Feature *feature : cachedPipeline.features) {
			for (auto &param : feature->shaderParams) {
				cachedPipeline.baseDrawData.setParam(param.name, param.defaultValue);
			}
		}
	}

	void depthSortBackToFront(CachedPipeline &cachedPipeline, const View &view) {
		const CachedViewData &viewData = *viewCache[&view].get();

		float4x4 viewProjMatrix = linalg::mul(viewData.projectionMatrix, view.view);

		size_t numDrawables = cachedPipeline.drawables.size();
		for (size_t i = 0; i < numDrawables; i++) {
			SortableDrawable &sortable = cachedPipeline.drawablesSorted[i];
			float4 projected = mul(viewProjMatrix, mul(sortable.drawable->transform, float4(0, 0, 0, 1)));
			sortable.projectedDepth = projected.z;
		}

		auto compareBackToFront = [](const SortableDrawable &left, const SortableDrawable &right) { return left.projectedDepth > right.projectedDepth; };
		std::stable_sort(cachedPipeline.drawablesSorted.begin(), cachedPipeline.drawablesSorted.end(), compareBackToFront);
	}

	void generateTextureIds(CachedPipeline &cachedPipeline, SortableDrawable &drawable) {
		std::vector<TextureId> &textureIds = drawable.textureIds.textures;
		textureIds.reserve(cachedPipeline.textureBindingLayout.bindings.size());

		for (auto &binding : cachedPipeline.textureBindingLayout.bindings) {
			auto &drawableParams = drawable.drawable->parameters.texture;

			auto it = drawableParams.find(binding.name);
			if (it != drawableParams.end()) {
				textureIds.push_back(cachedPipeline.textureIdMap.assign(context, it->second.texture.get()));
				continue;
			}

			if (drawable.drawable->material) {
				auto &materialParams = drawable.drawable->material->parameters.texture;
				it = materialParams.find(binding.name);
				if (it != materialParams.end()) {
					textureIds.push_back(cachedPipeline.textureIdMap.assign(context, it->second.texture.get()));
					continue;
				}
			}

			textureIds.push_back(TextureId(~0));
		}
	}

	SortableDrawable createSortableDrawable(CachedPipeline &cachedPipeline, Drawable &drawable) {
		SortableDrawable sortableDrawable{};
		sortableDrawable.drawable = &drawable;
		generateTextureIds(cachedPipeline, sortableDrawable);
		sortableDrawable.key = DrawGroupKey(drawable, sortableDrawable.textureIds);

		return sortableDrawable;
	}

	void groupDrawables(CachedPipeline &cachedPipeline, const RenderDrawablesStep &step, const View &view) {
		std::vector<SortableDrawable> &drawablesSorted = cachedPipeline.drawablesSorted;

		// Sort drawables based on mesh/texture bindings
		for (auto &drawable : cachedPipeline.drawables) {
			drawable->mesh->createContextDataConditional(context);
			addFrameReference(drawable->mesh->contextData);

			SortableDrawable sortableDrawable = createSortableDrawable(cachedPipeline, *drawable);

			auto comparison = [](const SortableDrawable &left, const SortableDrawable &right) { return left.key < right.key; };
			auto it = std::upper_bound(drawablesSorted.begin(), drawablesSorted.end(), sortableDrawable, comparison);
			drawablesSorted.insert(it, sortableDrawable);
		}

		if (step.sortMode == SortMode::BackToFront) {
			depthSortBackToFront(cachedPipeline, view);
		}

		// Creates draw call ranges based on DrawGroupKey
		if (drawablesSorted.size() > 0) {
			size_t groupStart = 0;
			DrawGroupKey currentDrawGroupKey = drawablesSorted[0].key;

			auto finishGroup = [&](size_t currentIndex) {
				size_t groupLength = currentIndex - groupStart;
				if (groupLength > 0) {
					cachedPipeline.drawGroups.emplace_back(currentDrawGroupKey, groupStart, groupLength);
				}
			};
			for (size_t i = 1; i < drawablesSorted.size(); i++) {
				DrawGroupKey drawGroupKey = drawablesSorted[i].key;
				if (drawGroupKey != currentDrawGroupKey) {
					finishGroup(i);
					groupStart = i;
					currentDrawGroupKey = drawGroupKey;
				}
			}
			finishGroup(drawablesSorted.size());
		}
	}

	void resetPipelineCacheDrawables() {
		for (auto &pair : pipelineCache) {
			pair.second->resetDrawables();
		}
	}

	WGPUBindGroup createTextureBindGroup(CachedPipeline &cachedPipeline, const TextureIds &textureIds) {
		std::vector<WGPUBindGroupEntry> entries;
		size_t bindingIndex = 0;
		for (auto &id : textureIds.textures) {
			WGPUBindGroupEntry &textureEntry = entries.emplace_back();
			textureEntry.binding = bindingIndex++;

			TextureContextData *textureData = cachedPipeline.textureIdMap.resolve(id);
			if (textureData) {
				textureEntry.textureView = textureData->defaultView;
			} else {
				textureEntry.textureView = placeholderTexture->textureView;
			}

			WGPUBindGroupEntry &samplerEntry = entries.emplace_back();
			samplerEntry.binding = bindingIndex++;
			samplerEntry.sampler = placeholderTexture->sampler;
		}

		WGPUBindGroupDescriptor textureBindGroupDesc{};
		textureBindGroupDesc.layout = cachedPipeline.bindGroupLayouts[1];
		textureBindGroupDesc.entries = entries.data();
		textureBindGroupDesc.entryCount = entries.size();
		return wgpuDeviceCreateBindGroup(context.wgpuDevice, &textureBindGroupDesc);
	}

	void renderDrawables(const DrawQueue &drawQueue, RenderDrawablesStep &step, ViewPtr view, Rect viewport, WGPUBuffer viewBuffer) {
		WGPUDevice device = context.wgpuDevice;
		WGPUCommandEncoderDescriptor desc = {};
		WGPUCommandEncoder commandEncoder = wgpuDeviceCreateCommandEncoder(device, &desc);

		WGPURenderPassDescriptor passDesc = {};
		passDesc.colorAttachmentCount = 1;

		WGPURenderPassColorAttachment mainAttach = {};
		mainAttach.clearColor = WGPUColor{.r = 0.1f, .g = 0.1f, .b = 0.1f, .a = 1.0f};

		// FIXME
		if (!mainOutputWrittenTo) {
			mainAttach.loadOp = WGPULoadOp_Clear;
			mainOutputWrittenTo = true;
		} else {
			mainAttach.loadOp = WGPULoadOp_Load;
		}
		mainAttach.view = mainOutput.view;
		mainAttach.storeOp = WGPUStoreOp_Store;

		WGPURenderPassDepthStencilAttachment depthAttach = {};
		depthAttach.clearDepth = 1.0f;
		depthAttach.depthLoadOp = WGPULoadOp_Clear;
		depthAttach.depthStoreOp = WGPUStoreOp_Store;
		depthAttach.view = depthTexture->update(device, viewport.getSize());

		passDesc.colorAttachments = &mainAttach;
		passDesc.colorAttachmentCount = 1;
		passDesc.depthStencilAttachment = &depthAttach;

		resetPipelineCacheDrawables();
		groupByPipeline(step, drawQueue.getDrawables());

		for (auto &pair : pipelineCache) {
			CachedPipeline &cachedPipeline = *pair.second.get();
			if (!cachedPipeline.pipeline) {
				buildPipeline(cachedPipeline);
			}

			groupDrawables(cachedPipeline, step, *view.get());
		}

		WGPURenderPassEncoder passEncoder = wgpuCommandEncoderBeginRenderPass(commandEncoder, &passDesc);
		wgpuRenderPassEncoderSetViewport(passEncoder, (float)viewport.x, (float)viewport.y, (float)viewport.width, (float)viewport.height, 0.0f, 1.0f);

		for (auto &pair : pipelineCache) {
			CachedPipeline &cachedPipeline = *pair.second.get();
			size_t drawBufferLength = cachedPipeline.objectBufferLayout.size * cachedPipeline.drawables.size();

			DynamicWGPUBuffer &instanceBuffer = cachedPipeline.instanceBufferPool.allocateBuffer(drawBufferLength);
			fillInstanceBuffer(instanceBuffer, cachedPipeline, view.get());

			auto drawGroups = cachedPipeline.drawGroups;
			std::vector<Bindable> bindables = {
				Bindable(viewBuffer, viewBufferLayout),
				Bindable(instanceBuffer, cachedPipeline.objectBufferLayout, drawBufferLength),
			};
			WGPUBindGroup bindGroup = createBindGroup(device, cachedPipeline.bindGroupLayouts[0], bindables);
			onFrameCleanup([bindGroup]() { wgpuBindGroupRelease(bindGroup); });

			wgpuRenderPassEncoderSetPipeline(passEncoder, cachedPipeline.pipeline);
			wgpuRenderPassEncoderSetBindGroup(passEncoder, 0, bindGroup, 0, nullptr);

			for (auto &drawGroup : drawGroups) {
				WGPUBindGroup textureBindGroup = createTextureBindGroup(cachedPipeline, drawGroup.key.textures);
				wgpuRenderPassEncoderSetBindGroup(passEncoder, 1, textureBindGroup, 0, nullptr);
				onFrameCleanup([textureBindGroup]() { wgpuBindGroupRelease(textureBindGroup); });

				MeshContextData *meshContextData = drawGroup.key.meshData;
				wgpuRenderPassEncoderSetVertexBuffer(passEncoder, 0, meshContextData->vertexBuffer, 0, meshContextData->vertexBufferLength);

				if (meshContextData->indexBuffer) {
					WGPUIndexFormat indexFormat = getWGPUIndexFormat(meshContextData->format.indexFormat);
					wgpuRenderPassEncoderSetIndexBuffer(passEncoder, meshContextData->indexBuffer, indexFormat, 0, meshContextData->indexBufferLength);

					wgpuRenderPassEncoderDrawIndexed(passEncoder, (uint32_t)meshContextData->numIndices, drawGroup.numInstances, 0, 0, drawGroup.startIndex);
				} else {
					wgpuRenderPassEncoderDraw(passEncoder, (uint32_t)meshContextData->numVertices, drawGroup.numInstances, 0, drawGroup.startIndex);
				}
			}
		}

		wgpuRenderPassEncoderEndPass(passEncoder);

		WGPUCommandBufferDescriptor cmdBufDesc = {};
		WGPUCommandBuffer cmdBuf = wgpuCommandEncoderFinish(commandEncoder, &cmdBufDesc);

		context.submit(cmdBuf);
	}

	void groupByPipeline(RenderDrawablesStep &step, const std::vector<DrawablePtr> &drawables) {
		// TODO: Paralellize
		std::vector<const Feature *> features;
		const std::vector<FeaturePtr> *featureSources[2] = {&step.features, nullptr};
		for (auto &drawable : drawables) {
			Drawable *drawablePtr = drawable.get();
			assert(drawablePtr->mesh);
			const Mesh &mesh = *drawablePtr->mesh.get();

			features.clear();
			for (auto &featureSource : featureSources) {
				if (!featureSource)
					continue;

				for (auto &feature : *featureSource) {
					features.push_back(feature.get());
				}
			}

			HasherXXH128 featureHasher;
			for (auto &feature : features) {
				// NOTE: Hashed by pointer since features are considered immutable & shared/ref-counted
				featureHasher(feature);
			}
			Hash128 featureHash = featureHasher.getDigest();

			auto drawableIt = drawableCache.find(drawablePtr);
			if (drawableIt == drawableCache.end()) {
				drawableIt = drawableCache.insert(std::make_pair(drawablePtr, CachedDrawableData{})).first;
				auto &drawableCache = drawableIt->second;
				drawableCache.drawable = drawablePtr;

				HasherXXH128<HashStaticVistor> hasher;
				hasher(mesh.getFormat());
				hasher(featureHash);
				if (const Material *material = drawablePtr->material.get()) {
					hasher(*material);
				}
				drawableCache.pipelineHash = hasher.getDigest();
			}

			Hash128 pipelineHash = drawableIt->second.pipelineHash;
			auto it1 = pipelineCache.find(pipelineHash);
			if (it1 == pipelineCache.end()) {
				it1 = pipelineCache.insert(std::make_pair(pipelineHash, std::make_shared<CachedPipeline>())).first;
				CachedPipeline &cachedPipeline = *it1->second.get();
				cachedPipeline.meshFormat = mesh.getFormat();
				cachedPipeline.features = features;

				buildTextureBindingLayout(cachedPipeline, drawablePtr->material.get());
			}

			CachedPipelinePtr &pipelineGroup = it1->second;
			pipelineGroup->drawables.push_back(drawablePtr);
		}
	}

	void buildTextureBindingLayout(CachedPipeline &cachedPipeline, const Material *material) {
		TextureBindingLayoutBuilder textureBindingLayoutBuilder;
		for (auto &feature : cachedPipeline.features) {
			for (auto &textureParam : feature->textureParams) {
				textureBindingLayoutBuilder.addOrUpdateSlot(textureParam.name, 0);
			}
		}

		if (material) {
			for (auto &pair : material->parameters.texture) {
				textureBindingLayoutBuilder.addOrUpdateSlot(pair.first, pair.second.defaultTexcoordBinding);
			}
		}

		cachedPipeline.textureBindingLayout = textureBindingLayoutBuilder.finalize();
	}

	shader::GeneratorOutput generateShader(const CachedPipeline &cachedPipeline) {
		using namespace shader;
		using namespace shader::blocks;

		shader::Generator generator;
		generator.meshFormat = cachedPipeline.meshFormat;

		FieldType colorFieldType(ShaderFieldBaseType::Float32, 4);
		generator.outputFields.emplace_back("color", colorFieldType);

		generator.viewBufferLayout = viewBufferLayout;
		generator.objectBufferLayout = cachedPipeline.objectBufferLayout;
		generator.textureBindingLayout = cachedPipeline.textureBindingLayout;

		std::vector<const EntryPoint *> entryPoints;
		for (auto &feature : cachedPipeline.features) {
			for (auto &entryPoint : feature->shaderEntryPoints) {
				entryPoints.push_back(&entryPoint);
			}
		}

		static const std::vector<EntryPoint> &builtinEntryPoints = []() -> const std::vector<EntryPoint> & {
			static std::vector<EntryPoint> builtin;
			builtin.emplace_back("interpolate", ProgrammableGraphicsStage::Vertex, blocks::DefaultInterpolation());
			return builtin;
		}();
		for (auto &builtinEntryPoint : builtinEntryPoints) {
			entryPoints.push_back(&builtinEntryPoint);
		}

		return generator.build(entryPoints);
	}

	void buildObjectBufferLayout(CachedPipeline &cachedPipeline) {
		UniformBufferLayoutBuilder objectBufferLayoutBuilder;
		objectBufferLayoutBuilder.push("world", ShaderParamType::Float4x4);
		for (const Feature *feature : cachedPipeline.features) {
			for (auto &param : feature->shaderParams) {
				objectBufferLayoutBuilder.push(param.name, param.type);
			}
		}

		cachedPipeline.objectBufferLayout = objectBufferLayoutBuilder.finalize();
	}

	FeaturePipelineState computePipelineState(const std::vector<const Feature *> &features) {
		FeaturePipelineState state{};
		for (const Feature *feature : features) {
			state = state.combine(feature->state);
		}
		return state;
	}

	// Bindgroup 0, the per-batch bound resources
	WGPUBindGroupLayout createBatchBindGroupLayout() {
		std::vector<WGPUBindGroupLayoutEntry> bindGroupLayoutEntries;

		WGPUBindGroupLayoutEntry &viewEntry = bindGroupLayoutEntries.emplace_back();
		viewEntry.binding = 0;
		viewEntry.visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Vertex;
		viewEntry.buffer.type = WGPUBufferBindingType_Uniform;
		viewEntry.buffer.hasDynamicOffset = false;

		WGPUBindGroupLayoutEntry &objectEntry = bindGroupLayoutEntries.emplace_back();
		objectEntry.binding = 1;
		objectEntry.visibility = WGPUShaderStage_Fragment | WGPUShaderStage_Vertex;
		objectEntry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
		objectEntry.buffer.hasDynamicOffset = false;

		WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {};
		bindGroupLayoutDesc.entries = bindGroupLayoutEntries.data();
		bindGroupLayoutDesc.entryCount = bindGroupLayoutEntries.size();
		return wgpuDeviceCreateBindGroupLayout(context.wgpuDevice, &bindGroupLayoutDesc);
	}

	// Bindgroup 1, bound textures
	WGPUBindGroupLayout createTextureBindGroupLayout(CachedPipeline &cachedPipeline) {
		std::vector<WGPUBindGroupLayoutEntry> bindGroupLayoutEntries;

		size_t bindingIndex = 0;
		for (auto &desc : cachedPipeline.textureBindingLayout.bindings) {
			(void)desc;

			WGPUBindGroupLayoutEntry &textureBinding = bindGroupLayoutEntries.emplace_back();
			textureBinding.binding = bindingIndex++;
			textureBinding.visibility = WGPUShaderStage_Fragment;
			textureBinding.texture.multisampled = false;
			textureBinding.texture.sampleType = WGPUTextureSampleType_Float;
			textureBinding.texture.viewDimension = WGPUTextureViewDimension_2D;

			WGPUBindGroupLayoutEntry &samplerBinding = bindGroupLayoutEntries.emplace_back();
			samplerBinding.binding = bindingIndex++;
			samplerBinding.visibility = WGPUShaderStage_Fragment;
			samplerBinding.sampler.type = WGPUSamplerBindingType_Filtering;
		}

		WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {};
		bindGroupLayoutDesc.entries = bindGroupLayoutEntries.data();
		bindGroupLayoutDesc.entryCount = bindGroupLayoutEntries.size();
		return wgpuDeviceCreateBindGroupLayout(context.wgpuDevice, &bindGroupLayoutDesc);
	}

	WGPUPipelineLayout createPipelineLayout(CachedPipeline &cachedPipeline) {
		assert(!cachedPipeline.pipelineLayout);
		assert(cachedPipeline.bindGroupLayouts.empty());

		cachedPipeline.bindGroupLayouts.push_back(createBatchBindGroupLayout());
		cachedPipeline.bindGroupLayouts.push_back(createTextureBindGroupLayout(cachedPipeline));

		WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {};
		pipelineLayoutDesc.bindGroupLayouts = cachedPipeline.bindGroupLayouts.data();
		pipelineLayoutDesc.bindGroupLayoutCount = cachedPipeline.bindGroupLayouts.size();
		cachedPipeline.pipelineLayout = wgpuDeviceCreatePipelineLayout(context.wgpuDevice, &pipelineLayoutDesc);
		return cachedPipeline.pipelineLayout;
	}

	struct VertexStateBuilder {
		std::vector<WGPUVertexAttribute> attributes;
		WGPUVertexBufferLayout vertexLayout = {};

		void build(WGPUVertexState &vertex, CachedPipeline &cachedPipeline) {
			size_t vertexStride = 0;
			size_t shaderLocationCounter = 0;
			for (auto &attr : cachedPipeline.meshFormat.vertexAttributes) {
				WGPUVertexAttribute &wgpuAttribute = attributes.emplace_back();
				wgpuAttribute.offset = uint64_t(vertexStride);
				wgpuAttribute.format = getWGPUVertexFormat(attr.type, attr.numComponents);
				wgpuAttribute.shaderLocation = shaderLocationCounter++;
				size_t typeSize = getVertexAttributeTypeSize(attr.type);
				vertexStride += attr.numComponents * typeSize;
			}
			vertexLayout.arrayStride = vertexStride;
			vertexLayout.attributeCount = attributes.size();
			vertexLayout.attributes = attributes.data();
			vertexLayout.stepMode = WGPUVertexStepMode::WGPUVertexStepMode_Vertex;

			vertex.bufferCount = 1;
			vertex.buffers = &vertexLayout;
			vertex.constantCount = 0;
			vertex.entryPoint = "vertex_main";
			vertex.module = cachedPipeline.shaderModule;
		}
	};

	void buildPipeline(CachedPipeline &cachedPipeline) {
		buildObjectBufferLayout(cachedPipeline);
		buildBaseObjectParameters(cachedPipeline);

		FeaturePipelineState pipelineState = computePipelineState(cachedPipeline.features);
		WGPUDevice device = context.wgpuDevice;

		shader::GeneratorOutput generatorOutput = generateShader(cachedPipeline);
		if (generatorOutput.errors.size() > 0) {
			shader::GeneratorOutput::dumpErrors(generatorOutput);
			assert(false);
		}

		WGPUShaderModuleDescriptor moduleDesc = {};
		WGPUShaderModuleWGSLDescriptor wgslModuleDesc = {};
		moduleDesc.label = "pipeline";
		moduleDesc.nextInChain = &wgslModuleDesc.chain;

		wgslModuleDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
		wgpuShaderModuleWGSLDescriptorSetCode(wgslModuleDesc, generatorOutput.wgslSource.c_str());
		spdlog::info("Generated WGSL:\n {}", generatorOutput.wgslSource);

		cachedPipeline.shaderModule = wgpuDeviceCreateShaderModule(context.wgpuDevice, &moduleDesc);
		assert(cachedPipeline.shaderModule);

		WGPURenderPipelineDescriptor desc = {};
		desc.layout = createPipelineLayout(cachedPipeline);

		VertexStateBuilder vertStateBuilder;
		vertStateBuilder.build(desc.vertex, cachedPipeline);

		WGPUFragmentState fragmentState = {};
		fragmentState.entryPoint = "fragment_main";
		fragmentState.module = cachedPipeline.shaderModule;

		WGPUColorTargetState mainTarget = {};
		mainTarget.format = mainOutput.format;
		mainTarget.writeMask = pipelineState.colorWrite.value_or(WGPUColorWriteMask_All);

		WGPUBlendState blendState{};
		if (pipelineState.blend.has_value()) {
			blendState = pipelineState.blend.value();
			mainTarget.blend = &blendState;
		}

		WGPUDepthStencilState depthStencilState{};
		depthStencilState.format = depthTexture->getFormat();
		depthStencilState.depthWriteEnabled = pipelineState.depthWrite.value_or(true);
		depthStencilState.depthCompare = pipelineState.depthCompare.value_or(WGPUCompareFunction_Less);
		depthStencilState.stencilBack.compare = WGPUCompareFunction_Always;
		depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
		desc.depthStencil = &depthStencilState;

		fragmentState.targets = &mainTarget;
		fragmentState.targetCount = 1;
		desc.fragment = &fragmentState;

		desc.multisample.count = 1;
		desc.multisample.mask = ~0;

		desc.primitive.frontFace = cachedPipeline.meshFormat.windingOrder == WindingOrder::CCW ? WGPUFrontFace_CCW : WGPUFrontFace_CW;
		if (pipelineState.culling.value_or(true)) {
			desc.primitive.cullMode = pipelineState.flipFrontFace.value_or(false) ? WGPUCullMode_Front : WGPUCullMode_Back;
		} else {
			desc.primitive.cullMode = WGPUCullMode_None;
		}

		switch (cachedPipeline.meshFormat.primitiveType) {
		case PrimitiveType::TriangleList:
			desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
			break;
		case PrimitiveType::TriangleStrip:
			desc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
			desc.primitive.stripIndexFormat = getWGPUIndexFormat(cachedPipeline.meshFormat.indexFormat);
			break;
		}

		cachedPipeline.pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
		assert(cachedPipeline.pipeline);
	}
};

Renderer::Renderer(Context &context) {
	impl = std::make_shared<RendererImpl>(context);
	if (!context.isHeadless()) {
		impl->shouldUpdateMainOutputFromContext = true;
	}
}

void Renderer::render(const DrawQueue &drawQueue, std::vector<ViewPtr> views, const PipelineSteps &pipelineSteps) {
	impl->renderViews(drawQueue, views, pipelineSteps);
}
void Renderer::render(const DrawQueue &drawQueue, ViewPtr view, const PipelineSteps &pipelineSteps) { impl->renderView(drawQueue, view, pipelineSteps); }
void Renderer::setMainOutput(const MainOutput &output) {
	impl->mainOutput = output;
	impl->shouldUpdateMainOutputFromContext = false;
}

void Renderer::beginFrame() { impl->beginFrame(); }
void Renderer::endFrame() { impl->endFrame(); }

void Renderer::cleanup() {
	Context &context = impl->context;
	impl.reset();
	impl = std::make_shared<RendererImpl>(context);
}

} // namespace gfx
