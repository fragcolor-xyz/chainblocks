#ifndef A136DB6F_2BE4_4AC2_B49D_57A7943F990B
#define A136DB6F_2BE4_4AC2_B49D_57A7943F990B

#include "fwd.hpp"
#include "worker_memory.hpp"
#include "render_graph.hpp"
#include "texture_view_cache.hpp"
#include "drawable_processor.hpp"
#include "context_data_storage.hpp"
#include "gizmos/shapes.hpp"

namespace gfx::detail {

struct FrameStats {
  size_t numDrawables{};
  size_t viewBindGroupsCreated{};
  size_t drawBindGroupsCreated{};
  size_t numCreatedBuffers{};

  void reset() {
    numDrawables = 0;
    viewBindGroupsCreated = 0;
    drawBindGroupsCreated = 0;
    numCreatedBuffers = 0;
  }

  size_t totalBindGroupsCreated() const { return viewBindGroupsCreated + drawBindGroupsCreated; }
};

using DebugVisualizer = shards::FunctionBase<512, void(gfx::ShapeRenderer &sr)>;

// Storage container for all renderer data
struct RendererStorage {
  WorkerMemory workerMemory;
  WGPUSupportedLimits deviceLimits = {};
  DrawableProcessorCache drawableProcessorCache;
  std::vector<TransientPtr> transientPtrCleanupQueue;
  RenderGraphCache renderGraphCache;
  std::unordered_map<UniqueId, CachedViewDataPtr> viewCache;
  std::unordered_map<UniqueId, QueueDataPtr> queueCache;
  PipelineCache pipelineCache;
  RenderTextureCache renderTextureCache;
  TextureViewCache textureViewCache;
  ContextDataStorage contextDataStorage;

  WGPUSharedBufferPool mapReadCopyDstBufferPool;

  // Enable debug visualization
  bool debug{};
  Swappable<std::vector<DebugVisualizer>, 2> debugVisualizers;

  FrameStats frameStats;

  bool ignoreCompilationErrors{};

  // Within the range [0, maxBufferedFrames)
  size_t frameIndex = 0;

  // Increments forever
  size_t frameCounter = 0;

  RendererStorage(Context &context) : drawableProcessorCache(context) {
    mapReadCopyDstBufferPool.initFn = [&context](size_t cap) -> WGPUBuffer {
      WGPUBufferDescriptor bufDesc{
          .label = "Temp copy buffer",
          .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
          .size = cap,
      };
      return wgpuDeviceCreateBuffer(context.wgpuDevice, &bufDesc);
    };
  }

  template <typename F> void debugVisualize(F &&f) {
    if (!debug)
      return;
    debugVisualizers.get(frameCounter % 2).emplace_back(std::forward<F>(f));
  }

  void clear() {
    renderGraphCache.clear();
    pipelineCache.clear();
    renderTextureCache.clear();
    textureViewCache.reset();
    viewCache.clear();
    queueCache.clear();
    debugVisualizers.forAll([](auto &v) { v.clear(); });
    drawableProcessorCache.drawableProcessorsMutex.lock();
    drawableProcessorCache.drawableProcessors.clear();
    drawableProcessorCache.drawableProcessorsMutex.unlock();
    mapReadCopyDstBufferPool.clear();
  }

  WGPUTextureView getTextureView(const TextureContextData &textureData, uint8_t faceIndex, uint8_t mipIndex) {
    shassert((textureData.texture || textureData.externalTexture) && "Invalid texture");
    TextureViewDesc desc{
        .format = deriveTextureViewFormat(textureData),
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = mipIndex,
        .mipLevelCount = 1,
        .baseArrayLayer = uint32_t(faceIndex),
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
    };
    WGPUTextureView result = textureViewCache.getTextureView(frameCounter, textureData, desc);
    shassert(result);
    return result;
  }
};

} // namespace gfx::detail

#endif /* A136DB6F_2BE4_4AC2_B49D_57A7943F990B */
