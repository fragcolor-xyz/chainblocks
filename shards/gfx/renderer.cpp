#include "renderer.hpp"
#include "wgpu_handle.hpp"

#ifdef __clang__
#pragma clang attribute push(__attribute__((no_sanitize("undefined"))), apply_to = function)
#endif

#include "boost/container/stable_vector.hpp"

#ifdef __clang__
#pragma clang attribute pop
#endif

#include "context.hpp"
#include "drawable.hpp"
#include "feature.hpp"
#include "fwd.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "mesh_utils.hpp"
#include "params.hpp"
#include "pipeline_builder.hpp"
#include "renderer_storage.hpp"
#include "renderer_types.hpp"
#include "renderer_cache.hpp"
#include "texture_view_cache.hpp"
#include "render_graph.hpp"
#include "drawables/mesh_drawable.hpp"
#include "shader/blocks.hpp"
#include "shader/entry_point.hpp"
#include "shader/fmt.hpp"
#include "shader/generator.hpp"
#include "shader/textures.hpp"
#include "shader/struct_layout.hpp"
#include "drawable_processor.hpp"
#include "texture.hpp"
#include "buffer.hpp"
#include "texture_placeholder.hpp"
#include "view.hpp"
#include "render_target.hpp"
#include "log.hpp"
#include "geom.hpp"
#include "view_stack.hpp"
#include "worker_memory.hpp"
#include <shards/core/pmr/vector.hpp>
#include <shards/core/pmr/unordered_map.hpp>
#include "frame_queue.hpp"
#include "gfx_wgpu.hpp"
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <tracy/Wrapper.hpp>
#include <thread>
#include <algorithm>
#include <magic_enum.hpp>
#include <spdlog/spdlog.h>
#include <span>
#include <type_traits>

#define GFX_RENDERER_MAX_BUFFERED_FRAMES (2)

using namespace gfx::detail;

namespace gfx {

static auto logger = gfx::getLogger();

// The render graph and outputs
struct PreparedRenderView {
  const RenderGraph &renderGraph;
  ViewData viewData;
  std::span<TextureSubResource> renderGraphOutputs;
};

struct RendererImpl final : public ContextData {
  Renderer &outer;
  Context &context;
  bool hasDevice{};

  RendererStorage storage;

  ViewStack viewStack;

  Renderer::MainOutput mainOutput;
  RenderTargetPtr mainOutputRenderTarget;
  bool shouldUpdateMainOutputFromContext = false;

  bool ignoreCompilationErrors = false;

  const size_t maxBufferedFrames = GFX_RENDERER_MAX_BUFFERED_FRAMES;

  bool mainOutputWrittenTo{};

  boost::container::stable_vector<std::optional<FrameQueue>> frameQueues;

  std::shared_ptr<ContextFlushTextureReferencesHandler> flushTextureReferencesHandler;
  std::shared_ptr<ContextDeviceStatusHandler> deviceStatusHandler;

  RendererImpl(Context &context, Renderer &outer) : outer(outer), context(context), storage(context) {
    struct FlushSurfaceTextureReferences : public ContextFlushTextureReferencesHandler {
      TextureViewCache &tvc;
      FlushSurfaceTextureReferences(TextureViewCache &tvc) : tvc(tvc) {}
      void flushTextureReferences() override { tvc.reset(); }
    };
    flushTextureReferencesHandler =
        context.onFlushTextureReferences->emplace<FlushSurfaceTextureReferences>(storage.textureViewCache);

    struct DeviceStatusHandler : public ContextDeviceStatusHandler {
      RendererImpl &renderer;
      DeviceStatusHandler(RendererImpl &renderer) : renderer(renderer) {}
      void deviceLost() override { renderer.onDeviceLost(); }
      void deviceAcquired() override { renderer.onDeviceAcquired(); }
    };
    deviceStatusHandler = context.onDeviceStatus->emplace<DeviceStatusHandler>(*this);

    if (!context.isHeadless()) {
      shouldUpdateMainOutputFromContext = true;
    }

    if (context.isReady())
      onDeviceAcquired();
  }

  ~RendererImpl() { onDeviceLost(); }

  WorkerMemory &getWorkerMemoryForCurrentFrame() { return storage.workerMemory; }

  void onDeviceAcquired() {
    if (hasDevice)
      return;

    shassert(context.isReady());
    gfxWgpuDeviceGetLimits(context.wgpuDevice, &storage.deviceLimits);
    hasDevice = true;
  }

  void onDeviceLost() {
    if (!hasDevice)
      return;

    context.poll();

    // Flush in-flight frame resources
    for (size_t i = 0; i < maxBufferedFrames; i++) {
      swapBuffers();
    }

    storage.clear();

    hasDevice = false;
  }

  void processTransientPtrCleanupQueue() {
    auto it = storage.transientPtrCleanupQueue.begin();
    while (it != storage.transientPtrCleanupQueue.end()) {
      it->destroy();
      ++it;
    }
    storage.transientPtrCleanupQueue.clear();
  }

  void updateMainOutputFromContext() {
    ZoneScoped;

    mainOutput.texture = context.getMainOutputTexture();
  }

  CachedView &getCachedView(const ViewPtr &view) {
    return *getSharedCacheEntry<CachedView>(storage.viewCache, view->getId()).get();
  }

  void pushView(ViewStack::Item &&item) {
    viewStack.push(std::move(item));
    const auto &viewStackOutput = viewStack.getOutput();

    bool needsNewFrameQueue = false;
    if (viewStack.size() == 1)
      needsNewFrameQueue = true;
    else {
      auto prevItem = viewStack.getOutput(1);
      if (prevItem.renderTarget != viewStackOutput.renderTarget) {
        needsNewFrameQueue = true;
      }
    }

    if (needsNewFrameQueue) {
      viewStackOutput.renderTarget->resizeConditional(viewStackOutput.referenceSize);
      auto &frameQueue = frameQueues.emplace_back();
      frameQueue.emplace(viewStackOutput.renderTarget, storage, getWorkerMemoryForCurrentFrame());
      frameQueue->fallbackClearColor.emplace(float4(1, 1, 1, 1));
    } else {
      // Can render to the exisiting queue
      frameQueues.push_back(std::nullopt);
    }
  }

  void popView() {
    auto &frameQueue = frameQueues.back();

    // Evaluate the queue for the popped view
    if (frameQueue.has_value()) {
      auto fallbackSize = viewStack.getOutput().size;
      frameQueue->evaluate(outer, fallbackSize);
    }

    frameQueues.pop_back();
    viewStack.pop();
  }

  FrameQueue &getFrameQueue() {
    for (auto it = frameQueues.rbegin(); it != frameQueues.rend(); ++it) {
      if (it->has_value())
        return it->value();
    }

    // Somehow there's no FrameQueue to enque render commands into
    throw std::logic_error("No frame queue in view stack");
  }

  void render(ViewPtr view, const PipelineSteps &pipelineSteps) {
    ZoneScoped;

    ViewData viewData{
        .view = view,
        .cachedView = getCachedView(view),
    };

    ViewStack::Output viewStackOutput = viewStack.getOutput();
    viewData.viewport = viewStackOutput.viewport;
    viewData.renderTarget = viewStackOutput.renderTarget;
    viewData.referenceOutputSize = viewStackOutput.referenceSize;

    int2 outputSize = viewStackOutput.size;
    if (outputSize.x == 0 || outputSize.y == 0)
      return;

    viewData.cachedView.touchWithNewTransform(view->view, view->getProjectionMatrix(float2(outputSize)), storage.frameCounter);

    getFrameQueue().enqueue(viewData, pipelineSteps);
  }

  struct DeferredTextureReadCommand {
    TextureSubResource texture;
    GpuTextureReadBufferPtr destination;

    std::shared_ptr<PooledWGPUBuffer> stagingBuffer;
    std::optional<void *> mappedBuffer;
    size_t rowSizeAligned{};
    size_t bufferSize{};
    int2 size{};

    DeferredTextureReadCommand(TextureSubResource texture, GpuTextureReadBufferPtr destination)
        : texture(texture), destination(destination) {}
    DeferredTextureReadCommand(DeferredTextureReadCommand &&) = default;
    DeferredTextureReadCommand &operator=(DeferredTextureReadCommand &&) = default;

    // Is queued for read?
    bool isQueued() const { return bufferSize > 0; }
  };

  std::list<DeferredTextureReadCommand> deferredTextureReadCommands;

  struct DeferredBufferReadCommand {
    BufferPtr buffer;
    GpuReadBufferPtr destination;

    std::shared_ptr<PooledWGPUBuffer> stagingBuffer;
    std::optional<void *> mappedBuffer;
    size_t rowSizeAligned{};
    size_t bufferSize{};
    int2 size{};

    DeferredBufferReadCommand(BufferPtr buffer, GpuReadBufferPtr destination) : buffer(buffer), destination(destination) {}
    DeferredBufferReadCommand(DeferredBufferReadCommand &&) = default;
    DeferredBufferReadCommand &operator=(DeferredBufferReadCommand &&) = default;

    // Is queued for read?
    bool isQueued() const { return bufferSize > 0; }
  };

  std::list<DeferredBufferReadCommand> deferredBufferReadCommands;

  bool queueTextureReadCommand(WGPUCommandEncoder encoder, DeferredTextureReadCommand &cmd) {
    auto &textureData = storage.contextDataStorage.getCreateOrUpdate(context, storage.frameCounter, cmd.texture.texture);
    if (!textureData.texture) {
      SPDLOG_LOGGER_WARN(logger, "Invalid texture queued for reading, ignoring");
      return false;
    }

    cmd.size = int2(textureData.size.width, textureData.size.height);
    auto &pixelFormatDesc = getTextureFormatDescription(textureData.format.pixelFormat);

    size_t rowSize = pixelFormatDesc.pixelSize * pixelFormatDesc.numComponents * cmd.size.x;
    cmd.rowSizeAligned = alignTo(rowSize, WGPU_COPY_BYTES_PER_ROW_ALIGNMENT);
    cmd.bufferSize = cmd.rowSizeAligned * cmd.size.y;
    cmd.stagingBuffer = storage.mapReadCopyDstBufferPool.allocateBuffer(cmd.bufferSize);

    WGPUImageCopyTexture srcDesc{
        .texture = textureData.texture,
        .mipLevel = cmd.texture.mipIndex,
        .origin = {.x = 0, .y = 0, .z = cmd.texture.faceIndex},
    };
    WGPUImageCopyBuffer dstDesc{
        .layout = {.offset = 0, .bytesPerRow = uint32_t(cmd.rowSizeAligned), .rowsPerImage = uint32_t(cmd.size.y)},
        .buffer = cmd.stagingBuffer->buffer,
    };
    WGPUExtent3D sizeDesc{
        .width = uint32_t(cmd.size.x),
        .height = uint32_t(cmd.size.y),
        .depthOrArrayLayers = 1,
    };
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &srcDesc, &dstDesc, &sizeDesc);
    return true;
  }

  bool queueBufferReadCommand(WGPUCommandEncoder encoder, DeferredBufferReadCommand &cmd) {
    auto &bufferData = storage.contextDataStorage.getCreateOrUpdate(context, storage.frameCounter, cmd.buffer);
    if (!bufferData.buffer) {
      SPDLOG_LOGGER_WARN(logger, "Invalid buffer queued for reading, ignoring");
      return false;
    }

    cmd.bufferSize = bufferData.bufferLength;
    cmd.stagingBuffer = storage.mapReadCopyDstBufferPool.allocateBuffer(cmd.bufferSize);

    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufferData.buffer, 0, cmd.stagingBuffer->buffer, 0, cmd.bufferSize);
    return true;
  }

  void queueTextureReadBufferMap(DeferredTextureReadCommand &cmd) {
    auto bufferMapped = [](WGPUBufferMapAsyncStatus status, void *ud) {
      DeferredTextureReadCommand &cmd = *(DeferredTextureReadCommand *)ud;
      if (status != WGPUBufferMapAsyncStatus_Success)
        throw formatException("Failed to map buffer: {}", magic_enum::enum_name(status));
#if WEBGPU_NATIVE
      cmd.mappedBuffer = wgpuBufferGetMappedRange(cmd.stagingBuffer->buffer, 0, cmd.bufferSize);
#else
      cmd.mappedBuffer = nullptr;
#endif
    };
#if WEBGPU_NATIVE
    wgpuBufferMapAsync(cmd.stagingBuffer->buffer, WGPUMapMode_Read, 0, cmd.bufferSize, bufferMapped, &cmd);
#else
    gfxWgpuBufferMapAsync(cmd.stagingBuffer->buffer, WGPUMapMode_Read, 0, cmd.bufferSize, bufferMapped, &cmd);
#endif
  }

  // Poll for mapped bugfer and copy data to target, returns true when completed
  bool pollQueuedTextureReadCommand(DeferredTextureReadCommand &cmd) {
    if (cmd.mappedBuffer) {
      cmd.destination->data.resize(cmd.bufferSize);

#if WEBGPU_NATIVE
      memcpy(cmd.destination->data.data(), cmd.mappedBuffer.value(), cmd.bufferSize);
#else
      gfxWgpuBufferReadInto(cmd.stagingBuffer->buffer, cmd.destination->data.data(), 0, cmd.bufferSize);
#endif

      cmd.destination->stride = cmd.rowSizeAligned;
      cmd.destination->size = cmd.size;
      cmd.destination->pixelFormat = cmd.texture.texture->getFormat().pixelFormat;

      wgpuBufferUnmap(cmd.stagingBuffer->buffer);
      cmd.mappedBuffer.reset();
      return true;
    } else {
      return false;
    }
  }

  void queueBufferReadBufferMap(DeferredBufferReadCommand &cmd) {
    auto bufferMapped = [](WGPUBufferMapAsyncStatus status, void *ud) {
      DeferredBufferReadCommand &cmd = *(DeferredBufferReadCommand *)ud;
      if (status != WGPUBufferMapAsyncStatus_Success)
        throw formatException("Failed to map buffer: {}", magic_enum::enum_name(status));
#if WEBGPU_NATIVE
      cmd.mappedBuffer = wgpuBufferGetMappedRange(cmd.stagingBuffer->buffer, 0, cmd.bufferSize);
#else
      cmd.mappedBuffer = nullptr;
#endif
    };
#if WEBGPU_NATIVE
    wgpuBufferMapAsync(cmd.stagingBuffer->buffer, WGPUMapMode_Read, 0, cmd.bufferSize, bufferMapped, &cmd);
#else
    gfxWgpuBufferMapAsync(cmd.stagingBuffer->buffer, WGPUMapMode_Read, 0, cmd.bufferSize, bufferMapped, &cmd);
#endif
  }

  // Poll for mapped bugfer and copy data to target, returns true when completed
  bool pollQueuedBufferReadCommand(DeferredBufferReadCommand &cmd) {
    if (cmd.mappedBuffer) {
      cmd.destination->data.resize(cmd.bufferSize);

#if WEBGPU_NATIVE
      memcpy(cmd.destination->data.data(), cmd.mappedBuffer.value(), cmd.bufferSize);
#else
      gfxWgpuBufferReadInto(cmd.stagingBuffer->buffer, cmd.destination->data.data(), 0, cmd.bufferSize);
#endif

      wgpuBufferUnmap(cmd.stagingBuffer->buffer);
      cmd.mappedBuffer.reset();
      return true;
    } else {
      return false;
    }
  }

  void copyTexture(TextureSubResource texture, GpuTextureReadBufferPtr destination, bool wait) {
    if (wait) {
      DeferredTextureReadCommand tmpCommand(texture, destination);
      WGPUCommandEncoderDescriptor encDesc{};
      WgpuHandle<WGPUCommandEncoder> encoder(wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &encDesc));

      if (!queueTextureReadCommand(encoder, tmpCommand))
        return;

      WGPUCommandBufferDescriptor desc{.label = "Renderer::copyTexture (blocking)"};
      WgpuHandle<WGPUCommandBuffer> cmdBuffer(wgpuCommandEncoderFinish(encoder, &desc));
      wgpuQueueSubmit(context.wgpuQueue, 1, &cmdBuffer.handle);

      queueTextureReadBufferMap(tmpCommand);

      context.poll(true);
      pollQueuedTextureReadCommand(tmpCommand);
    } else {
      deferredTextureReadCommands.emplace_back(texture, destination);
    }
  }

  void copyBuffer(BufferPtr buffer, GpuReadBufferPtr destination, bool wait) {
    if (wait) {
      DeferredBufferReadCommand tmpCommand(buffer, destination);
      WGPUCommandEncoderDescriptor encDesc{};
      WgpuHandle<WGPUCommandEncoder> encoder(wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &encDesc));

      if (!queueBufferReadCommand(encoder, tmpCommand))
        return;

      WGPUCommandBufferDescriptor desc{.label = "Renderer::copyTexture (blocking)"};
      WgpuHandle<WGPUCommandBuffer> cmdBuffer(wgpuCommandEncoderFinish(encoder, &desc));
      wgpuQueueSubmit(context.wgpuQueue, 1, &cmdBuffer.handle);

      queueBufferReadBufferMap(tmpCommand);

      context.poll(true);
      pollQueuedBufferReadCommand(tmpCommand);
    } else {
      deferredBufferReadCommands.emplace_back(buffer, destination);
    }
  }

  void queueTextureCopies() {
    boost::container::small_vector<DeferredTextureReadCommand *, 16> buffersToMap;
    WGPUCommandEncoderDescriptor encDesc{};
    WgpuHandle<WGPUCommandEncoder> encoder;
    for (auto it = deferredTextureReadCommands.begin(); it != deferredTextureReadCommands.end();) {
      if (!it->isQueued()) {
        if (!encoder) {
          // Lazy create encoder
          encoder.reset(wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &encDesc));
        }
        if (queueTextureReadCommand(encoder, *it)) {
          buffersToMap.push_back(&*it);
        } else {
          // Invalid texture, remove from queue
          it = deferredTextureReadCommands.erase(it);
        }
      }
      ++it;
    }

    if (encoder) {
      // Queue read commands for next frame
      WGPUCommandBufferDescriptor desc{.label = "Renderer::copyTexture"};
      WgpuHandle<WGPUCommandBuffer> cmdBuffer(wgpuCommandEncoderFinish(encoder, &desc));
      wgpuQueueSubmit(context.wgpuQueue, 1, &cmdBuffer.handle);
    }

    // Need to map buffers after submitting the texture copies
    for (auto &cmd : buffersToMap) {
      queueTextureReadBufferMap(*cmd);
    }
  }

  void queueBufferCopies() {
    boost::container::small_vector<DeferredBufferReadCommand *, 16> buffersToMap;
    WGPUCommandEncoderDescriptor encDesc{};
    WgpuHandle<WGPUCommandEncoder> encoder;
    for (auto it = deferredBufferReadCommands.begin(); it != deferredBufferReadCommands.end();) {
      if (!it->isQueued()) {
        if (!encoder) {
          // Lazy create encoder
          encoder.reset(wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &encDesc));
        }

        if (queueBufferReadCommand(encoder, *it)) {
          buffersToMap.push_back(&*it);
        } else {
          // Invalid texture, remove from queue
          it = deferredBufferReadCommands.erase(it);
        }
      }
      ++it;
    }

    if (encoder) {
      // Queue read commands for next frame
      WGPUCommandBufferDescriptor desc{.label = "Renderer::copyBuffer"};
      WgpuHandle<WGPUCommandBuffer> cmdBuffer(wgpuCommandEncoderFinish(encoder, &desc));
      wgpuQueueSubmit(context.wgpuQueue, 1, &cmdBuffer.handle);
    }

    // Need to map buffers after submitting the texture copies
    for (auto &cmd : buffersToMap) {
      queueBufferReadBufferMap(*cmd);
    }
  }

  void pollBufferCopies() {
    context.poll(false);

    for (auto it = deferredTextureReadCommands.begin(); it != deferredTextureReadCommands.end();) {
      if (it->isQueued()) {
        if (pollQueuedTextureReadCommand(*it)) {
          // Finished, remove from queue
          it = deferredTextureReadCommands.erase(it);
          continue;
        }
      }
      ++it;
    }

    for (auto it = deferredBufferReadCommands.begin(); it != deferredBufferReadCommands.end();) {
      if (it->isQueued()) {
        if (pollQueuedBufferReadCommand(*it)) {
          // Finished, remove from queue
          it = deferredBufferReadCommands.erase(it);
          continue;
        }
      }
      ++it;
    }
  }

  void beginFrame() {
    // the device might have it's context data cleared by manually calling renderer->cleanup()
    // calling beginFrame here assumes you already did context.beginFrame which implies a device
    // so recreate context data
    if (!hasDevice)
      onDeviceAcquired();

    swapBuffers();

    if (shouldUpdateMainOutputFromContext) {
      updateMainOutputFromContext();
    }

    // Update main render target
    if (!mainOutputRenderTarget)
      mainOutputRenderTarget = std::make_shared<RenderTarget>("mainOutput");
    mainOutputRenderTarget->attachments["color"].texture = mainOutput.texture;
    mainOutputRenderTarget->resizeFixed(mainOutput.texture->getResolution());

    storage.frameStats.reset();

    storage.drawableProcessorCache.beginFrame(storage.frameCounter);

    resetWorkerMemory();

    auto mainOutputResolution = mainOutput.texture->getResolution();
    pushView(ViewStack::Item{
        .referenceSize = mainOutputResolution,
        .renderTarget = mainOutputRenderTarget,
    });
  }

  void resetWorkerMemory() { storage.workerMemory.reset(); }

  void endFrame() {
    // Reset view stack and pop main output
    // This also evaluates the main frame queue
    popView();
    viewStack.reset();

    pollBufferCopies();
    queueTextureCopies();
    queueBufferCopies();

    processTransientPtrCleanupQueue();

#ifdef TRACY_ENABLE
    TracyPlot("Drawables Processed", int64_t(storage.frameStats.numDrawables));

    TracyPlot("BindGroups Created (Total)", int64_t(storage.frameStats.totalBindGroupsCreated()));
    TracyPlot("BindGroups Created (View)", int64_t(storage.frameStats.viewBindGroupsCreated));
    TracyPlot("BindGroups Created (Draw)", int64_t(storage.frameStats.drawBindGroupsCreated));
    TracyPlot("Buffers Created", int64_t(storage.frameStats.numCreatedBuffers));

    TracyPlotConfig("GFX WorkerMemory", tracy::PlotFormatType::Memory, true, true, 0);
    TracyPlot("GFX WorkerMemory", int64_t(storage.workerMemory.getMemoryResource().totalRequestedBytes));
    
    WGPUGlobalReport report{};
    wgpuGenerateReport(context.wgpuInstance, &report);

    WGPUHubReport *hubReport{};
    switch (context.getBackendType()) {
    case WGPUBackendType_Vulkan:
      hubReport = &report.vulkan;
      break;
    case WGPUBackendType_D3D12:
      hubReport = &report.dx12;
      break;
    default:
      break;
    }

    if (hubReport) {
      TracyPlot("WGPU Buffers", int64_t(hubReport->buffers.numAllocated));
      TracyPlot("WGPU BindGroups", int64_t(hubReport->bindGroups.numAllocated));
      TracyPlot("WGPU BindGroupLayouts", int64_t(hubReport->bindGroupLayouts.numAllocated));
      TracyPlot("WGPU CommandBuffers", int64_t(hubReport->commandBuffers.numAllocated));
      TracyPlot("WGPU Queues", int64_t(hubReport->queues.numAllocated));
      TracyPlot("WGPU Textures", int64_t(hubReport->textures.numAllocated));
    }
#endif
  }

  void clearOldCacheItems() {
    clearOldCacheItemsIn(storage.pipelineCache.map, storage.frameCounter, 120 * 60 * 5);
    clearOldCacheItemsIn(storage.viewCache, storage.frameCounter, 120 * 60 * 5);

    storage.contextDataStorage.buffers.clearOldCacheItems(storage.frameCounter, 32);
    storage.contextDataStorage.meshes.clearOldCacheItems(storage.frameCounter, 32);
    storage.contextDataStorage.textures.clearOldCacheItems(storage.frameCounter, 16);

    // NOTE: Because textures take up a lot of memory, try to clean this as frequent as possible
    storage.renderTextureCache.resetAndClearOldCacheItems(storage.frameCounter, 8);
    storage.textureViewCache.clearOldCacheItems(storage.frameCounter, 8);

    storage.mapReadCopyDstBufferPool.recycle();
  }

  void swapBuffers() {
    ++storage.frameCounter;
    storage.frameIndex = (storage.frameIndex + 1) % maxBufferedFrames;

    auto &debugVisualizers = storage.debugVisualizers.get(storage.frameCounter % 2);
    debugVisualizers.clear();

    clearOldCacheItems();
  }
};

Renderer::Renderer(Context &context) { impl = std::make_shared<RendererImpl>(context, *this); }

Context &Renderer::getContext() { return impl->context; }

void Renderer::render(ViewPtr view, const PipelineSteps &pipelineSteps) { impl->render(view, pipelineSteps); }

void Renderer::copyTexture(TextureSubResource texture, GpuTextureReadBufferPtr destination, bool wait) {
  impl->copyTexture(texture, destination, wait);
}

void Renderer::copyBuffer(BufferPtr buffer, GpuReadBufferPtr destination, bool wait) {
  impl->copyBuffer(buffer, destination, wait);
}

void Renderer::pollBufferCopies() { impl->pollBufferCopies(); }

void Renderer::setMainOutput(const MainOutput &output) {
  impl->mainOutput = output;
  impl->shouldUpdateMainOutputFromContext = false;
}

const ViewStack &Renderer::getViewStack() const { return impl->viewStack; }

/// <div rustbindgen hide></div>
void Renderer::pushView(ViewStack::Item &&item) { impl->pushView(std::move(item)); }

/// <div rustbindgen hide></div>
void Renderer::popView() { impl->popView(); }

void Renderer::beginFrame() { impl->beginFrame(); }
void Renderer::endFrame() { impl->endFrame(); }

void Renderer::cleanup() { impl->onDeviceLost(); }

void Renderer::setDebug(bool debug) { impl->storage.debug = debug; }
void Renderer::processDebugVisuals(ShapeRenderer &sr) {
  auto &storage = impl->storage;
  auto &vec = storage.debugVisualizers.get((storage.frameCounter - 1) % 2);
  for (auto &v : vec) {
    v(sr);
  }
  vec.clear();
}

void Renderer::setIgnoreCompilationErrors(bool ignore) { impl->storage.ignoreCompilationErrors = ignore; }

void Renderer::dumpStats() {
  static constexpr size_t Megabyte = 1 << 20;

  SPDLOG_LOGGER_INFO(logger, "== Renderer stats ==");

  size_t workerIndex{};
  SPDLOG_LOGGER_INFO(logger, " Graphics worker allocators:");
  auto &mem = impl->storage.workerMemory;
  SPDLOG_LOGGER_INFO(logger, "  Worker[{}]: {} MB (peak: {} MB, avg: {} MB)", workerIndex++,
                     mem.getMemoryResource().preallocatedBlock.size() / Megabyte,
                     mem.getMemoryResource().maxUsage.getMax() / Megabyte,
                     mem.getMemoryResource().maxUsage.getAverage() / Megabyte);

  SPDLOG_LOGGER_INFO(logger, " Caches:");
  SPDLOG_LOGGER_INFO(logger, "  Render Graphs: {}", impl->storage.renderGraphCache.map.size());
  SPDLOG_LOGGER_INFO(logger, "  Pipelines: {}", impl->storage.pipelineCache.map.size());
  SPDLOG_LOGGER_INFO(logger, "  Views: {}", impl->storage.viewCache.size());

  SPDLOG_LOGGER_INFO(logger, " Frame Index: {}", impl->storage.frameIndex);
  SPDLOG_LOGGER_INFO(logger, " Frame Counter: {}", impl->storage.frameCounter);
}

} // namespace gfx
