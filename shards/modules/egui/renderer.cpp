#include "renderer.hpp"
#include "egui_render_pass.hpp"
#include "gfx/params.hpp"
#include <gfx/enums.hpp>
#include <gfx/pipeline_step.hpp>
#include <gfx/renderer.hpp>
#include <gfx/shader/blocks.hpp>
#include <gfx/shader/types.hpp>
#include <gfx/mesh.hpp>
#include <gfx/texture.hpp>
#include <gfx/fwd.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/sized_item_pool.hpp>
#include <gfx/feature.hpp>
#include <gfx/view.hpp>
#include <gfx/context.hpp>
#include <gfx/window.hpp>
#include <gfx/gfx_wgpu.hpp>
#include <gfx/log.hpp>
#include "../../core/pool.hpp"
#include <spdlog/spdlog.h>
#include <map>
#include <vector>
#include <memory>
#include <cmath>

namespace gfx {

static auto logger = getLogger();

inline constexpr auto findDrawableInPoolByMeshSize(size_t targetSize) {
  if constexpr (sizeof(size_t) >= 8) {
    if (targetSize > INT64_MAX) {
      throw std::runtime_error("targetSize too large");
    }
  }
  return [targetSize](std::shared_ptr<MeshDrawable> &item) -> int64_t {
    auto &mesh = item->mesh;
    size_t size = mesh->getNumVertices() * mesh->getFormat().getVertexSize();
    if (size < targetSize) {
      return INT64_MAX;
    }
    // Negate so the smallest buffer will be picked first
    return -(int64_t(size) - int64_t(targetSize));
  };
}

struct PoolTraits {
  using T = std::shared_ptr<MeshDrawable>;
  T newItem() {
    auto drawable = std::make_shared<MeshDrawable>();
    drawable->mesh = std::make_shared<Mesh>();
    return drawable;
  }
  void release(T &) {}
  bool canRecycle(T &v) { return v.use_count() == 1; }
  void recycled(T &v) {}
};

struct MeshDrawablePool : public shards::Pool<std::shared_ptr<MeshDrawable>, PoolTraits> {
  std::shared_ptr<MeshDrawable> &allocateBuffer(size_t size) {
    return this->newValue([](auto &buffer) {}, findDrawableInPoolByMeshSize(size));
  }
};

struct TextureManager {
  std::map<uint64_t, TexturePtr> textures;

  TexturePtr get(const egui::RenderOutput &output, const egui::TextureId &id) const {
    if (id.managed) {
      auto it = textures.find(id.id);
      if (it != textures.end())
        return it->second;
      return TexturePtr();
    } else {
      if (id.id >= output.numExternalTextures)
        throw std::logic_error("Invalid texture id");
      return output.externalTextures[id.id];
    }
  }

  void imageCopy(WGPUTextureFormat dstFormat, WGPUTextureFormat srcFormat, uint8_t *dst, const uint8_t *src, size_t numPixels) {
    switch (srcFormat) {
    case WGPUTextureFormat_RGBA8UnormSrgb:
      assert(dstFormat == srcFormat);
      memcpy(dst, src, numPixels * sizeof(uint32_t));
      break;
    case WGPUTextureFormat_R32Float: {
      assert(dstFormat == WGPUTextureFormat_R8Unorm);
      const float *srcFloat = (float *)src;
      for (size_t i = 0; i < numPixels; i++) {
        *dst = uint8_t(*srcFloat * 255);
        dst++;
        srcFloat++;
      }
    } break;
    default:
      assert(false);
    }
  }

  void set(const egui::TextureSet &set) {
    auto it = textures.find(set.id);
    if (it == textures.end()) {
      it = textures.insert_or_assign(set.id.id, std::make_shared<Texture>()).first;
    }
    TexturePtr texture = it->second;

    gfx::TextureFormat fmt{};
    WGPUTextureFormat srcFormat;
    size_t srcPixelSize;
    size_t dstPixelSize;
    switch (set.format) {
    case egui::TextureFormat::R32F:
      //  Need to convert, 32-bit float types are not filterable in the WebGPU spec
      srcFormat = WGPUTextureFormat_R32Float;
      fmt.pixelFormat = WGPUTextureFormat_R8Unorm;
      srcPixelSize = sizeof(float);
      dstPixelSize = sizeof(uint8_t);
      break;
    case egui::TextureFormat::RGBA8:
      srcFormat = fmt.pixelFormat = WGPUTextureFormat_RGBA8UnormSrgb;
      srcPixelSize = dstPixelSize = sizeof(uint32_t);
      break;
    }

    fmt.dimension = TextureDimension::D2;
    fmt.flags = TextureFormatFlags::None;
    SamplerState sampler{
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .filterMode = WGPUFilterMode_Linear,
    };

    TextureDescCPUCopy desc = std::get<TextureDescCPUCopy>(texture->getDesc());
    std::vector<uint8_t> imageData;
    if (set.subRegion && desc.sourceData.getLength() > 0) {
      imageData.resize(desc.sourceData.getLength());
      memcpy(imageData.data(), desc.sourceData.getData(), desc.sourceData.getLength());
    } else {
      if (set.subRegion)
        throw std::logic_error("Can not update a sub-region on a non-existing texture");
      imageData.resize(dstPixelSize * set.size[0] * set.size[1]);
    }

    int2 size{};
    if (set.subRegion) {
      size = texture->getResolution();
      size_t dstPitch = size.x * dstPixelSize;
      size_t srcPitch = set.size[0] * srcPixelSize;
      uint8_t *dstBase = imageData.data() + set.offset[0] * dstPixelSize;
      for (size_t srcY = 0; srcY < set.size[1]; srcY++) {
        size_t dstY = set.offset[1] + srcY;
        uint8_t *dstPtr = dstBase + dstPitch * dstY;
        const uint8_t *srcPtr = set.pixels + srcPitch * srcY;
        imageCopy(fmt.pixelFormat, srcFormat, dstPtr, srcPtr, set.size[0]);
      }
    } else { // Full image
      size.x = set.size[0];
      size.y = set.size[1];
      imageCopy(fmt.pixelFormat, srcFormat, imageData.data(), set.pixels, size.x * size.y);
    }

    desc.sourceData = std::move(imageData);
    desc.format = fmt;
    desc.format.resolution = size;
    texture->init(desc)
        .initWithSamplerState(sampler)
#if SH_GFX_CONTEXT_DATA_LABELS
        .initWithLabel(fmt::format("egui_texture_{}", set.id.id))
#endif
        ;
  }

  void free(egui::TextureId id) { textures.erase(id.id); }
};

struct EguiRendererImpl {
  MeshDrawablePool meshPool;
  TextureManager textures;
  std::vector<egui::TextureId> pendingTextureFrees;
  std::vector<FeaturePtr> uiFeatures;

  MeshFormat meshFormat;

  EguiRendererImpl() {
    uiFeatures.push_back(EguiColorFeature::create());
    meshFormat = {
        .primitiveType = PrimitiveType::TriangleList,
        .windingOrder = WindingOrder::CCW,
        .indexFormat = IndexFormat::UInt32,
        .vertexAttributes = egui::Vertex::getAttributes(),
    };
  }

  void addPendingTextureFree(egui::TextureId id) { pendingTextureFrees.push_back(id); }

  void processPendingTextureFrees() {
    for (auto &id : pendingTextureFrees) {
      SPDLOG_LOGGER_INFO(logger, "textureFree {}", (uint64_t)id.id);
      textures.free(id);
    }
    pendingTextureFrees.clear();
  }

  void applyTextureUpdatesPreRender(const egui::RenderOutput &output) {
    processPendingTextureFrees();

    // Update textures before render
    auto &textureUpdates = output.textureUpdates;
    for (size_t i = 0; i < textureUpdates.numSets; i++) {
      auto &textureSet = textureUpdates.sets[i];
      textures.set(textureSet);
    }
  }

  void applyTextureUpdatesPostRender(const egui::RenderOutput &output) {
    // Store texture id's to free
    // these are executed at the start the next time render() is called
    auto &textureUpdates = output.textureUpdates;
    for (size_t i = 0; i < textureUpdates.numFrees; i++) {
      auto &id = textureUpdates.frees[i];
      addPendingTextureFree(id);
    }
  }

  void render(const egui::RenderOutput &output, const float4x4 &rootTransform, const gfx::DrawQueuePtr &drawQueue,
              bool clipGeometry) {
    meshPool.recycle();

    applyTextureUpdatesPreRender(output);

    // Update meshes & generate drawables
    for (size_t i = 0; i < output.numPrimitives; i++) {
      auto &prim = output.primitives[i];

      auto drawablePtr = meshPool.allocateBuffer(prim.numVertices * sizeof(egui::Vertex));
      auto &drawable = *drawablePtr.get();
      auto &mesh = drawablePtr->mesh;
      mesh->update(meshFormat, prim.vertices, prim.numVertices * sizeof(egui::Vertex), prim.indices,
                   prim.numIndices * sizeof(uint32_t));
#if SH_GFX_CONTEXT_DATA_LABELS
      char buffer[128];
      snprintf(buffer, sizeof(buffer), "egui_%zu/%zu", i, output.numPrimitives);
      mesh->setLabel(buffer);
#endif

      drawable.mesh = mesh;
      drawable.transform = rootTransform;
      drawable.features = uiFeatures;
      uint32_t flags = 0;
      TexturePtr texture = textures.get(output, prim.textureId);
      static FastString fs_color = "color";
      static FastString fs_flags = "flags";
      if (texture) {
        drawable.parameters.set(fs_color, texture);
        if (texture->getFormat().pixelFormat == WGPUTextureFormat_R8Unorm) {
          flags |= uint32_t(0x1);
        }
      }
      drawable.parameters.set(fs_flags, flags);

      drawable.clipRect.reset();
      if (clipGeometry) {
        bool isClippingRectInf = prim.clipRect.min.x == -INFINITY && prim.clipRect.min.y == -INFINITY &&
                                 prim.clipRect.max.x == INFINITY && prim.clipRect.max.y == INFINITY;
        if (!isClippingRectInf) {
          drawable.clipRect = int4(prim.clipRect.min.x, prim.clipRect.min.y, prim.clipRect.max.x, prim.clipRect.max.y);
        }
      }

      drawQueue->add(drawable);
    }

    applyTextureUpdatesPostRender(output);
  }
};

EguiRenderer::EguiRenderer() { impl = std::make_shared<EguiRendererImpl>(); }

void EguiRenderer::applyTextureUpdatesOnly(const egui::RenderOutput &output) {
  impl->applyTextureUpdatesPreRender(output);
  impl->applyTextureUpdatesPostRender(output);
}

void EguiRenderer::render(const egui::RenderOutput &output, const float4x4 &rootTransform, const gfx::DrawQueuePtr &drawQueue,
                          bool clipGeometry) {
  impl->render(output, rootTransform, drawQueue, clipGeometry);
}

void EguiRenderer::renderNoTransform(const egui::RenderOutput &output, const gfx::DrawQueuePtr &drawQueue) {
  render(output, linalg::identity, drawQueue);
}

EguiRenderer *EguiRenderer::create() { return new EguiRenderer(); }

void EguiRenderer::destroy(EguiRenderer *renderer) { delete renderer; }

float EguiRenderer::getDrawScale(Window &window) { return window.getUIScale(); }
} // namespace gfx
