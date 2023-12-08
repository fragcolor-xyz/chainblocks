#ifndef FBBE103B_4B4A_462F_BE67_FE59A0C30FB5
#define FBBE103B_4B4A_462F_BE67_FE59A0C30FB5

#pragma clang attribute push(__attribute__((no_sanitize("undefined"))), apply_to = function)
#include "boost/container/stable_vector.hpp"
#pragma clang attribute pop
#include "drawable.hpp"
#include "fwd.hpp"
#include "render_target.hpp"
#include "renderer_types.hpp"
#include "texture.hpp"
#include "texture_view_cache.hpp"
#include "fmt.hpp"
#include "worker_memory.hpp"
#include "pmr/unordered_set.hpp"
#include "pmr/vector.hpp"
#include "gfx_wgpu.hpp"
#include "drawable_processor.hpp"
#include "../core/function.hpp"
#include <compare>
#include <span>

namespace gfx::detail {

namespace graph_build_data {
struct NodeBuildData;
}

struct RenderTargetFormat {
  WGPUTextureFormat format;
  int2 size;

  std::strong_ordering operator<=>(const RenderTargetFormat &other) const = default;
};

// Caches render target textures based on format/size
struct RenderTextureCache {
  struct Item {
    TexturePtr handle;
    size_t lastTouched;
  };

  struct Bin {
    RenderTargetFormat format;
    std::vector<Item> items;
    std::vector<size_t> freeList;

    void reset() {
      freeList.clear();
      for (size_t i = 0; i < items.size(); i++) {
        freeList.push_back(i);
      }
    }

    // return true if texture is already initialized
    //  false if it's an uninitialized texture
    bool getItem(TexturePtr *&outTexture, size_t frameCounter) {
      if (freeList.size() == 0) {
        Item &newItem = items.emplace_back();

        newItem.lastTouched = frameCounter;
        outTexture = &newItem.handle;
        return false;
      } else {
        size_t index = freeList.back();
        freeList.pop_back();
        Item &item = items[index];

        item.lastTouched = frameCounter;
        outTexture = &item.handle;
        return true;
      }
    }
  };

  std::map<RenderTargetFormat, Bin> bins;

  // Cleanup old render targets
  void resetAndClearOldCacheItems(size_t frameCount, size_t frameThreshold) {
    for (auto it = bins.begin(); it != bins.end();) {
      auto &bin = it->second;
      if (cleanupBin(bin, frameCount, frameThreshold) == 0) {
        // Evict
        it = bins.erase(it);
      } else {
        // readd items to free list
        bin.reset();
        ++it;
      }
    }
  }

  const TexturePtr &allocate(const RenderTargetFormat &format, size_t frameCounter) {
    auto it = bins.find(format);
    if (it == bins.end())
      it = bins.insert_or_assign(format, Bin{.format = format}).first;

    auto &bin = it->second;

    TexturePtr *texture;
    if (!bin.getItem(texture, frameCounter)) {
      (*texture) = std::make_shared<Texture>();
      (*texture)
          ->init(TextureDesc{
              .format =
                  TextureFormat{
                      .dimension = TextureDimension::D2,
                      .flags = TextureFormatFlags::RenderAttachment,
                      .pixelFormat = format.format,
                  },
              .resolution = format.size,
          })
          .initWithLabel("auto");
    }

    return *texture;
  }

private:
  static size_t cleanupBin(Bin &bin, size_t frameCounter, size_t frameThreshold) {
    for (auto it = bin.items.begin(); it != bin.items.end();) {
      int64_t delta = int64_t(frameCounter) - int64_t(it->lastTouched);
      if (delta > int64_t(frameThreshold)) {
        it = bin.items.erase(it);
      } else {
        ++it;
      }
    }
    return bin.items.size();
  }
};

struct RenderGraph;
struct RenderGraphNode;
struct RenderGraphEvaluator;
struct RenderGraphEncodeContext {
  WGPURenderPassEncoder encoder;
  const ViewData &viewData;
  RenderGraphEvaluator &evaluator;
  const RenderGraph &graph;
  const RenderGraphNode &node;
  bool ignoreInDebugger;
};

typedef size_t FrameIndex;
typedef size_t SizeIndex;

struct RenderGraphNode {
  struct Attachment {
    FrameIndex frameIndex;
    std::optional<ClearValues> clearValues;

    Attachment(FrameIndex frameIndex) : frameIndex(frameIndex) {}
    Attachment(FrameIndex frameIndex, const ClearValues &clearValues) : frameIndex(frameIndex), clearValues(clearValues) {}

    operator FrameIndex() const { return frameIndex; }
  };

  RenderTargetLayout renderTargetLayout;
  std::vector<Attachment> writesTo;
  std::vector<FrameIndex> readsFrom;
  PipelineStepPtr originalStep;

  int2 outputSize;

  // This points to which data slot to use when resolving view data
  size_t queueDataIndex;

  // Allow updating of node outputs
  // shards::Function<void(RenderGraphNode &)> updateOutputs;

  // Sets up the render pass
  shards::Function<void(WGPURenderPassDescriptor &)> setupPass;

  // Writes the render pass
  shards::Function<void(RenderGraphEncodeContext &)> encode;

  RenderGraphNode() = default;
  RenderGraphNode(RenderGraphNode &&) = default;
  RenderGraphNode &operator=(RenderGraphNode &&) = default;
};

// Keeps a list of nodes that contain GPU commands
struct RenderGraph {
  // References the original texture inside of the render step description
  struct TextureOverrideRef {
    // size_t stepIndex;
    PipelineStepPtr step;
    enum Binding {
      Input,
      Output,
    } bindingType;
    size_t bindingIndex;
  };
  using OutputIndex = size_t;
  using FrameBinding = std::variant<std::monostate, OutputIndex, TextureOverrideRef>;

  struct Frame {
    std::string name;
    WGPUTextureFormat format;
    // Bind this frame texture to an output or texture specified in the graph
    FrameBinding binding;
  };

  struct Output {
    std::string name;
    FrameIndex frameIndex;
  };

  struct SizeParent {
    SizeIndex parentSize;
    float2 sizeScale;
  };

  struct SizeConstraint {
    // All frames listed must have the same size
    std::vector<FrameIndex> frames;
    // Parent size
    std::optional<SizeParent> parent;
  };

  std::vector<SizeConstraint> sizeConstraints;
  std::vector<RenderGraphNode> nodes;
  std::vector<Frame> frames;
  std::vector<Output> outputs;
  std::vector<DrawQueuePtr> autoClearQueues;
};

struct DrawableProcessorCache {
  Context &context;
  std::shared_mutex drawableProcessorsMutex;
  std::map<DrawableProcessorConstructor, std::shared_ptr<IDrawableProcessor>> drawableProcessors;

  DrawableProcessorCache(Context &context) : context(context) {}

  IDrawableProcessor &get(DrawableProcessorConstructor constructor) {
    std::shared_lock<std::shared_mutex> l(drawableProcessorsMutex);
    auto it = drawableProcessors.find(constructor);
    if (it == drawableProcessors.end()) {
      l.unlock();
      std::unique_lock<std::shared_mutex> l1(drawableProcessorsMutex);
      it = drawableProcessors.emplace(constructor, constructor(context)).first;
      return *it->second.get();
    } else {
      return *it->second.get();
    }
  }

  void beginFrame(size_t frameCounter) {
    for (auto &drawableProcessor : drawableProcessors) {
      drawableProcessor.second->reset(frameCounter);
    }
  }
};

// Interface to resolve node evaluation data during runtime
struct IRenderGraphEvaluationData {
  virtual const ViewData &getViewData(size_t queueIndex) = 0;
};

struct RendererStorage;
struct RenderGraphEvaluator {
  using allocator_type = shards::pmr::PolymorphicAllocator<>;

private:
  allocator_type allocator;

  // Keeps track of texture that have been written to at least once
  shards::pmr::unordered_set<Texture *> writtenTextures;

  // Resolved texture handles for indexed by frame index inside nodes
  struct ResolvedFrameTexture {
    TexturePtr texture;
    WGPUTextureView view{};

    ResolvedFrameTexture(TexturePtr texture, WGPUTextureView view) : texture(texture), view(view) {}
  };
  shards::pmr::vector<ResolvedFrameTexture> frameTextures;

  Renderer &renderer;
  RendererStorage &storage;

public:
  bool ignoreInDebugger{};

public:
  RenderGraphEvaluator(allocator_type allocator, Renderer &renderer, RendererStorage &storage);

  RendererStorage &getStorage() { return storage; }

  Renderer &getRenderer() { return renderer; }

  bool isWrittenTo(const TexturePtr &texture) const { return writtenTextures.contains(texture.get()); }

  const TexturePtr &getTexture(FrameIndex frameIndex) {
    if (frameIndex >= frameTextures.size())
      throw std::out_of_range("frameIndex");
    return frameTextures[frameIndex].texture;
  }

  // Describes a render pass so that targets are cleared to their default value or loaded
  // based on if they were already written to previously
  WGPURenderPassDescriptor createRenderPassDescriptor(const RenderGraph &graph, Context &context, const RenderGraphNode &node);

  // Reset all render texture write state
  void reset(size_t frameCounter) {
    writtenTextures.clear();
    frameTextures.clear();
  }

  void getFrameTextures(shards::pmr::vector<ResolvedFrameTexture> &outFrameTextures, const std::span<TextureSubResource> &outputs,
                        const RenderGraph &graph, Context &context);

  void evaluate(const RenderGraph &graph, IRenderGraphEvaluationData &evaluationData, std::span<TextureSubResource> outputs);
};

struct CachedRenderGraph {
  RenderGraph renderGraph;
};
typedef std::shared_ptr<CachedRenderGraph> CachedRenderGraphPtr;

struct RenderGraphCache {
  std::unordered_map<Hash128, CachedRenderGraphPtr> map;

  void clear() { map.clear(); }
};

} // namespace gfx::detail

#endif /* FBBE103B_4B4A_462F_BE67_FE59A0C30FB5 */
