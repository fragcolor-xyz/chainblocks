#ifndef A136DB6F_2BE4_4AC2_B49D_57A7943F990B
#define A136DB6F_2BE4_4AC2_B49D_57A7943F990B

#include "worker_memory.hpp"
#include "render_graph.hpp"
#include "texture_cache.hpp"
#include "drawable_processor.hpp"

namespace gfx::detail {

struct FrameStats {
  size_t numDrawables{};

  void reset() { numDrawables = 0; }
};

// Storage container for all renderer data
struct RendererStorage {
  WorkerMemory workerMemory;
  WGPUSupportedLimits deviceLimits = {};
  DrawableProcessorCache drawableProcessorCache;
  std::list<TransientPtr> transientPtrCleanupQueue;
  RenderGraphCache renderGraphCache;
  std::unordered_map<const View *, CachedViewDataPtr> viewCache;
  PipelineCache pipelineCache;
  RenderTextureCache renderTextureCache;
  TextureViewCache textureViewCache;

  FrameStats frameStats;

  bool ignoreCompilationErrors{};

  // Within the range [0, maxBufferedFrames)
  size_t frameIndex = 0;

  // Increments forever
  size_t frameCounter = 0;

  RendererStorage(Context &context) : drawableProcessorCache(context) {}

  WGPUTextureView getTextureView(const TextureContextData &textureData, uint8_t faceIndex, uint8_t mipIndex) {
    if (textureData.externalView)
      return textureData.externalView;
    TextureViewDesc desc{
        .format = textureData.format.pixelFormat,
        .dimension = WGPUTextureViewDimension_2D,
        .baseMipLevel = mipIndex,
        .mipLevelCount = 1,
        .baseArrayLayer = uint32_t(faceIndex),
        .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
    };
    return textureViewCache.getTextureView(frameCounter, textureData.texture, desc);
  }
};

} // namespace gfx::detail

#endif /* A136DB6F_2BE4_4AC2_B49D_57A7943F990B */
