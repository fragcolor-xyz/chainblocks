#ifndef CA95E36A_DF18_4EE1_B394_4094F976B20E
#define CA95E36A_DF18_4EE1_B394_4094F976B20E

#include "gfx/fwd.hpp"
#include "gfx_wgpu.hpp"
#include "texture.hpp"
#include "drawable.hpp"
#include "mesh.hpp"
#include "feature.hpp"
#include "params.hpp"
#include "render_target.hpp"
#include "shader/uniforms.hpp"
#include "shader/textures.hpp"
#include "shader/fmt.hpp"
#include "renderer.hpp"
#include "log.hpp"
#include "graph.hpp"
#include "hasherxxh128.hpp"
#include "sized_item_pool.hpp"
#include "worker_memory.hpp"
#include "pmr/wrapper.hpp"
#include "pmr/map.hpp"
#include "pmr/string.hpp"
#include <cassert>
#include <vector>
#include <map>
#include <compare>
#include <spdlog/spdlog.h>

namespace gfx::detail {

using shader::FieldType;
using shader::TextureBindingLayout;
using shader::UniformBufferLayout;
using shader::UniformLayout;

inline auto getLogger() {
  static auto logger = gfx::getLogger();
  return logger;
}

// Wraps an object that is swapped per frame for double+ buffered rendering
template <typename TInner, size_t MaxSize> struct Swappable {
private:
  std::optional<TInner> elems[MaxSize];

public:
  template <typename... TArgs> Swappable(TArgs... args) {
    for (size_t i = 0; i < MaxSize; i++) {
      elems.emplace(std::forward<TArgs>(args)...);
    }
  }

  TInner &get(size_t frameNumber) {
    assert(frameNumber <= MaxSize);
    return elems[frameNumber].value();
  }

  TInner &operator()(size_t frameNumber) { return get(frameNumber); }
};

typedef uint16_t TextureId;

struct TextureBindings {
  using allocator_type = shards::pmr::PolymorphicAllocator<>;

  shards::pmr::vector<TextureContextData *> textures;

  TextureBindings() = default;
  TextureBindings(allocator_type allocator) : textures(allocator) {}

  template <typename H> void hash(H &hasher) const { hasher(textures.data(), textures.size() * sizeof(TextureId)); }
};

struct PipelineDrawables;

struct BufferBinding {
  UniformBufferLayout layout;
  size_t index;
};

struct RenderTargetLayout {
  struct Target {
    std::string name;
    WGPUTextureFormat format;

    std::strong_ordering operator<=>(const Target &other) const = default;

    template <typename T> void getPipelineHash(T &hasher) const {
      hasher(name);
      hasher(format);
    }
  };

  std::vector<Target> targets;
  std::optional<size_t> depthTargetIndex;

  bool operator==(const RenderTargetLayout &other) const {
    if (!std::equal(targets.begin(), targets.end(), other.targets.begin(), other.targets.end()))
      return false;

    if (depthTargetIndex != other.depthTargetIndex)
      return false;

    return true;
  }

  bool operator!=(const RenderTargetLayout &other) const { return !(*this == other); }

  template <typename T> void getPipelineHash(T &hasher) const {
    hasher(targets);
    hasher(depthTargetIndex);
  }
};

/// <div rustbindgen hide></div>
struct ParameterStorage : public IParameterCollector {
  using allocator_type = shards::pmr::PolymorphicAllocator<>;

  struct KeyLess {
    using is_transparent = std::true_type;
    template <typename T, typename U> bool operator()(const T &a, const U &b) const {
      return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }
  };

  shards::pmr::map<shards::pmr::string, ParamVariant, KeyLess> data;
  shards::pmr::map<shards::pmr::string, TextureParameter, KeyLess> textures;

  using IParameterCollector::setParam;
  using IParameterCollector::setTexture;

  ParameterStorage() = default;
  ParameterStorage(allocator_type allocator) : data(allocator), textures(allocator) {}
  ParameterStorage(ParameterStorage &&other, allocator_type allocator) : data(allocator), textures(allocator) {
    *this = std::move(other);
  }
  ParameterStorage &operator=(ParameterStorage &&) = default;

  void setParam(const char *name, ParamVariant &&value) { data.emplace(name, std::move(value)); }
  void setTexture(const char *name, TextureParameter &&value) { textures.emplace(name, std::move(value)); }

  void setParamIfUnset(const shards::pmr::string &key, const ParamVariant &value) {
    if (!data.contains(key)) {
      data.emplace(key, value);
    }
  }

  void append(const ParameterStorage &other) {
    for (auto &it : other.data) {
      data.emplace(it);
    }
  }
};

struct CompilationError {
  std::string message;

  CompilationError() = default;
  CompilationError(std::string &&message) : message(std::move(message)) {}
};

struct CachedPipeline {
  // The compiled shader module including both vertex/fragment entry points
  WgpuHandle<WGPUShaderModule> shaderModule;

  // The compiled pipeline layout
  WgpuHandle<WGPUPipelineLayout> pipelineLayout;

  // The compiled pipeline
  WgpuHandle<WGPURenderPipeline> pipeline;

  // All compiled bind group layouts used in the pipeline
  std::vector<WgpuHandle<WGPUBindGroupLayout>> bindGroupLayouts;

  // Supported texture bindings
  TextureBindingLayout textureBindingLayout;

  // The output format
  RenderTargetLayout renderTargetLayout;

  // Processor that this pipeline is built for
  IDrawableProcessor *drawableProcessor{};

  std::vector<BufferBinding> viewBuffersBindings;
  std::vector<BufferBinding> drawBufferBindings;

  // Collected parameter generator
  std::vector<FeatureGenerator::PerView> perViewGenerators;
  std::vector<FeatureGenerator::PerObject> perObjectGenerators;

  size_t lastTouched{};

  ParameterStorage baseDrawParameters;

  std::optional<CompilationError> compilationError{};
};
typedef std::shared_ptr<CachedPipeline> CachedPipelinePtr;

struct CachedView {
  float4x4 projectionTransform;
  float4x4 invProjectionTransform;
  float4x4 invViewTransform;
  float4x4 previousViewTransform = linalg::identity;
  float4x4 currentViewTransform = linalg::identity;
  float4x4 viewProjectionTransform;

  size_t lastTouched{};

  void touchWithNewTransform(const float4x4 &viewTransform, const float4x4 &projectionTransform, size_t frameCounter) {
    if (frameCounter > lastTouched) {
      previousViewTransform = currentViewTransform;
      currentViewTransform = viewTransform;
      invViewTransform = linalg::inverse(viewTransform);

      this->projectionTransform = projectionTransform;
      invProjectionTransform = linalg::inverse(projectionTransform);

      viewProjectionTransform = linalg::mul(projectionTransform, currentViewTransform);

      lastTouched = frameCounter;
    }
  }
};
typedef std::shared_ptr<CachedView> CachedViewDataPtr;

struct ViewData {
  ViewPtr view;
  CachedView &cachedView;
  Rect viewport;
  RenderTargetPtr renderTarget;
};

struct VertexStateBuilder {
  std::vector<WGPUVertexAttribute> attributes;
  WGPUVertexBufferLayout vertexLayout = {};

  void build(WGPUVertexState &vertex, const MeshFormat &meshFormat, WGPUShaderModule shaderModule) {
    size_t vertexStride = 0;
    size_t shaderLocationCounter = 0;
    for (auto &attr : meshFormat.vertexAttributes) {
      WGPUVertexAttribute &wgpuAttribute = attributes.emplace_back();
      wgpuAttribute.offset = uint64_t(vertexStride);
      wgpuAttribute.format = getWGPUVertexFormat(attr.type, attr.numComponents);
      wgpuAttribute.shaderLocation = shaderLocationCounter++;
      size_t typeSize = getStorageTypeSize(attr.type);
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
    vertex.module = shaderModule;
  }
};

struct PooledWGPUBuffer {
  WgpuHandle<WGPUBuffer> buffer;
  size_t capacity;

  PooledWGPUBuffer() = default;
  PooledWGPUBuffer(PooledWGPUBuffer &&) = default;
  PooledWGPUBuffer &operator=(PooledWGPUBuffer &&) = default;

  operator WGPUBuffer() const { return buffer; }
};

template <> struct SizedItemOps<PooledWGPUBuffer> {
  using InitFunction = std::function<WGPUBuffer(size_t)>;
  InitFunction initFn;

  static WGPUBuffer defaultInitializer(size_t) { throw std::runtime_error("invalid buffer initializer"); }

  SizedItemOps(InitFunction &&init_ = &defaultInitializer) : initFn(std::move(init_)) {}

  size_t getCapacity(PooledWGPUBuffer &item) const { return item.capacity; }

  void init(PooledWGPUBuffer &item, size_t size) {
    item.capacity = alignTo(size, 512);
    item.buffer.reset(initFn(item.capacity));
  }
};

// Pool of WGPUBuffers with custom initializer
using WGPUBufferPool = SizedItemPool<PooledWGPUBuffer>;

struct CachedDrawable {
  float4x4 previousTransform = linalg::identity;
  float4x4 currentTransform = linalg::identity;

  size_t lastTouched{};

  void touchWithNewTransform(const float4x4 &transform, size_t frameCounter) {
    if (frameCounter > lastTouched) {
      previousTransform = currentTransform;
      currentTransform = transform;

      lastTouched = frameCounter;
    }
  }
};
typedef std::shared_ptr<CachedDrawable> CachedDrawablePtr;

// Data from generators
struct GeneratorData {
  ParameterStorage *viewParameters;
  shards::pmr::vector<ParameterStorage> *drawParameters;
};

} // namespace gfx::detail

#endif /* CA95E36A_DF18_4EE1_B394_4094F976B20E */
