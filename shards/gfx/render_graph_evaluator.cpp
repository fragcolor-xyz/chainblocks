#include "render_graph.hpp"
#include "renderer_storage.hpp"
#include "wgpu_handle.hpp"
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <boost/container/flat_map.hpp>
#include <boost/tti/has_member_data.hpp>

namespace gfx::detail {

BOOST_TTI_TRAIT_HAS_MEMBER_DATA(has_render_step_input, input);
BOOST_TTI_TRAIT_HAS_MEMBER_DATA(has_render_step_output, output);

RenderGraphEvaluator::RenderGraphEvaluator(allocator_type allocator, Renderer &renderer, RendererStorage &storage)
    : allocator(allocator), writtenTextures(allocator), frameTextures(allocator), frameSizes(allocator), renderer(renderer),
      storage(storage) {}

void RenderGraphEvaluator::getFrameTextures(shards::pmr::vector<ResolvedFrameTexture> &outFrameTextures,
                                            const std::span<TextureSubResource> &outputs, const RenderGraph &graph,
                                            Context &context) {
  outFrameTextures.clear();
  outFrameTextures.reserve(graph.frames.size());

  // Entire texture view
  auto resolve = [&](TexturePtr texture) {
    TextureContextData &cd = storage.contextDataStorage.getCreateOrUpdate(context, storage.frameCounter, texture);
    return ResolvedFrameTexture(texture, storage.getTextureView(cd, 0, 0));
  };

  // Subresource texture view
  auto resolveSubResourceView = [&](const TextureSubResource &resource) {
    TextureContextData &cd = storage.contextDataStorage.getCreateOrUpdate(context, storage.frameCounter, resource.texture);
    return ResolvedFrameTexture(resource.texture, storage.getTextureView(cd, resource.faceIndex, resource.mipIndex));
  };

  for (auto &frame : graph.frames) {
    auto &expectedSize = frameSizes[frame.size];
    if (frame.binding.index() != 0) { // Texture or output binding
      auto resolvedBinding = resolveBinding(frame.binding, outputs);
      if (resolvedBinding.isOutput) {
        auto outputSize = resolvedBinding.subResource.getResolution();
        if (expectedSize.size != outputSize) {
          throw std::runtime_error(fmt::format("Invalid output size, expected {}, was {}", expectedSize.size, outputSize));
        }
      } else {
        if (expectedSize.isAutoSize) {
          // Auto-size the texture
          int2 targetSize = resolvedBinding.subResource.getResolutionFromMipResolution(expectedSize.size);
          auto& texture = resolvedBinding.subResource.texture;
          auto desc =  texture->getDesc();
          desc.getFormat().resolution = targetSize;
          texture->init(desc);
        } else {
          shassert(resolvedBinding.subResource.getResolution() == expectedSize.size &&
                   "Manually sized texture does not have it's expected size, this should not happen");
        }
      }
      outFrameTextures.push_back(resolveSubResourceView(resolvedBinding.subResource));
    } else {
      // Create transient texture
      TexturePtr texture = storage.renderTextureCache.allocate(
          RenderTargetFormat{.format = frame.format, .size = expectedSize.size}, storage.frameCounter);
      outFrameTextures.push_back(resolve(texture));
    }
  }
}

WGPURenderPassDescriptor RenderGraphEvaluator::createRenderPassDescriptor(const RenderGraph &graph, Context &context,
                                                                          const RenderGraphNode &node) {
  // NOTE: Allocated in frame memory
  WGPURenderPassDepthStencilAttachment *depthStencilAttachmentPtr{};
  auto &colorAttachments = *allocator.new_object<shards::pmr::vector<WGPURenderPassColorAttachment>>();

  size_t depthIndex = node.renderTargetLayout.depthTargetIndex.value_or(~0);
  for (size_t i = 0; i < node.outputs.size(); i++) {
    const RenderGraphNode::Attachment &nodeAttachment = node.outputs[i];
    ResolvedFrameTexture &resolvedFrameTexture = frameTextures[nodeAttachment.frameIndex];

    if (i == depthIndex) {
      auto &attachment = *(depthStencilAttachmentPtr = allocator.new_object<WGPURenderPassDepthStencilAttachment>());
      attachment = WGPURenderPassDepthStencilAttachment{
          .view = resolvedFrameTexture.view,
      };

      std::visit(
          [&](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
              attachment.depthLoadOp = WGPULoadOp_Load;
            } else if constexpr (std::is_same_v<T, Discard>) {
              attachment.depthLoadOp = WGPULoadOp_Clear;
              attachment.depthClearValue = 1.0f;
            } else {
              attachment.depthLoadOp = WGPULoadOp_Clear;
              attachment.depthClearValue = arg.depth;
            }
          },
          nodeAttachment.loadOp);
      // Insert force-clear
      if (attachment.depthLoadOp == WGPULoadOp_Load && !isWrittenTo(resolvedFrameTexture.texture)) {
        attachment.depthLoadOp = WGPULoadOp_Clear;
        attachment.depthClearValue = 1.0f;
      }
      attachment.depthStoreOp = WGPUStoreOp_Store;

      // TODO: Stencil
      uint32_t clearStencil = 0;
      attachment.stencilClearValue = clearStencil;
      attachment.stencilLoadOp = WGPULoadOp_Undefined;
      attachment.stencilStoreOp = WGPUStoreOp_Undefined;
    } else {
      auto &attachment = colorAttachments.emplace_back(WGPURenderPassColorAttachment {
        .view = resolvedFrameTexture.view,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
      });

      std::visit(
          [&](auto &arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
              attachment.loadOp = WGPULoadOp_Load;
            } else if constexpr (std::is_same_v<T, Discard>) {
              attachment.loadOp = WGPULoadOp_Clear;
              attachment.clearValue = WGPUColor{};
            } else {
              attachment.loadOp = WGPULoadOp_Clear;
              attachment.clearValue = WGPUColor{arg.color.x, arg.color.y, arg.color.z, arg.color.w};
            }
          },
          nodeAttachment.loadOp);
      // Insert force-clear
      if (attachment.loadOp == WGPULoadOp_Load && !isWrittenTo(resolvedFrameTexture.texture)) {
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.clearValue = WGPUColor{};
      }
      attachment.storeOp = WGPUStoreOp_Store;
    }

    writtenTextures.insert(resolvedFrameTexture.texture.get());
  }

  return WGPURenderPassDescriptor{
      .colorAttachmentCount = uint32_t(colorAttachments.size()),
      .colorAttachments = colorAttachments.data(),
      .depthStencilAttachment = depthStencilAttachmentPtr,
  };
}

RenderGraphEvaluator::ResolvedBinding RenderGraphEvaluator::resolveBinding(RenderGraph::FrameBinding binding,
                                                                           const std::span<TextureSubResource> &outputs) {
  return std::visit(
      [&](auto &arg) -> ResolvedBinding {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          throw std::logic_error("binding does not point to a texture");
        } else if constexpr (std::is_same_v<T, RenderGraph::OutputIndex>) {
          return ResolvedBinding(outputs[arg], true);
        } else if constexpr (std::is_same_v<T, RenderGraph::TextureOverrideRef>) {
          return std::visit(
              [&](auto &step) -> ResolvedBinding {
                using T1 = std::decay_t<decltype(step)>;
                if constexpr (has_render_step_input<T1, RenderStepInput>::type::value) {
                  RenderStepInput &input = step.input;
                  if (arg.bindingType == RenderGraph::TextureOverrideRef::Input) {
                    auto tex = std::get_if<RenderStepInput::Texture>(&input.attachments[arg.bindingIndex]);
                    if (!tex)
                      throw std::logic_error("Expected a texture binding");
                    return ResolvedBinding(tex->subResource, false);
                  }
                }
                if constexpr (has_render_step_output<T1, std::optional<RenderStepOutput>>::type::value) {
                  RenderStepOutput &output = *step.output;
                  if (arg.bindingType == RenderGraph::TextureOverrideRef::Output) {
                    auto tex = std::get_if<RenderStepOutput::Texture>(&output.attachments[arg.bindingIndex]);
                    if (!tex)
                      throw std::logic_error("Expected a texture binding");
                    return ResolvedBinding(tex->subResource, false);
                  }
                }
                throw std::logic_error("Invalid bindings");
              },
              *arg.step.get());
        }
      },
      binding);
}

void RenderGraphEvaluator::computeFrameSizes(const RenderGraph &graph, std::span<TextureSubResource> outputs, int2 fallbackSize) {
  int2 refSize{};
  for (auto &output : outputs) {
    refSize = linalg::max(refSize, output.getResolution());
  }
  if (refSize == int2{})
    refSize = fallbackSize;

  frameSizes.resize(graph.sizes.size());
  for (size_t sizeIndex = 0; sizeIndex < graph.sizes.size(); sizeIndex++) {
    auto &size = graph.sizes[sizeIndex];
    frameSizes[sizeIndex] = std::visit(
        [&](auto &size) -> FrameSize {
          using T = std::decay_t<decltype(size)>;
          if constexpr (std::is_same_v<T, RenderGraphFrameSize::Manual>) {
            const RenderGraph::Frame &frame = graph.frames[size.frame];
            ResolvedBinding resolvedBinding = resolveBinding(frame.binding, outputs);
            return FrameSize(resolvedBinding.subResource.getResolution(), float2(1.0f), false);
          } else if constexpr (std::is_same_v<T, RenderGraphFrameSize::RelativeToMain>) {
            double2 inSize = double2(refSize);
            float2 vpScale{1.0f};
            if (size.scale) {
              inSize *= double2(*size.scale);
              vpScale *= *size.scale;
            }
            return FrameSize(int2(linalg::round(inSize)), vpScale, true);
          } else if constexpr (std::is_same_v<T, RenderGraphFrameSize::RelativeToFrame>) {
            auto otherSizeIndex = graph.frames[size.frame].size;
            double2 inSize = double2(frameSizes[otherSizeIndex].size);
            float2 vpScale = frameSizes[otherSizeIndex].viewportScale;
            if (size.scale) {
              inSize *= double2(*size.scale);
              vpScale *= *size.scale;
            }
            return FrameSize(int2(linalg::round(inSize)), vpScale, true);
          } else {
            throw std::logic_error("Invalid variant");
          }
        },
        size);
  }
}

void RenderGraphEvaluator::validateOutputSizes(const RenderGraph &graph) {
  size_t nodeIndex{};
  for (auto &node : graph.nodes) {
    std::optional<int2> outputSize;
    size_t outputIndex{};
    for (auto &output : node.outputs) {
      auto size = frameSizes[graph.frames[output.frameIndex].size];
      if (!outputSize)
        outputSize = size.size;
      else {
        if (outputSize != size.size) {
          throw std::runtime_error(fmt::format("Output sizes do not match, node {}, frame {}, expected: {}, was: {}", nodeIndex,
                                               outputIndex, *outputSize, size.size));
        }
      }
      ++outputIndex;
    }
    ++nodeIndex;
  }
}

void RenderGraphEvaluator::evaluate(const RenderGraph &graph, IRenderGraphEvaluationData &evaluationData,
                                    std::span<TextureSubResource> outputs, int2 fallbackSize) {
  // Evaluate all render steps in the order they are defined
  // potential optimizations:
  // - evaluate steps without dependencies in parralel
  // - group multiple steps into single render passes

  auto &context = renderer.getContext();

  computeFrameSizes(graph, outputs, fallbackSize);

  for (auto &fs : frameSizes) {
    if (fs.size.x == 0 || fs.size.y == 0) {
      SPDLOG_LOGGER_WARN(getLogger(), "Frame is zero size, skipping render");
      return;
    }
  }

  getFrameTextures(frameTextures, outputs, graph, context);
  validateOutputSizes(graph);

  WGPUCommandEncoderDescriptor desc = {};
  WgpuHandle<WGPUCommandEncoder> commandEncoder(wgpuDeviceCreateCommandEncoder(context.wgpuDevice, &desc));

  for (const auto &node : graph.nodes) {
    ViewData viewData(evaluationData.getViewData(node.queueDataIndex));

    WGPURenderPassDescriptor renderPassDesc = createRenderPassDescriptor(graph, context, node);
    std::string nameBuffer = getPipelineStepName(node.originalStep);
    renderPassDesc.label = nameBuffer.c_str();
    if (node.setupPass)
      node.setupPass(renderPassDesc);
    WgpuHandle<WGPURenderPassEncoder> renderPassEncoder(wgpuCommandEncoderBeginRenderPass(commandEncoder, &renderPassDesc));

    int2 targetSize = fallbackSize;
    float2 vpScale(1.0f);
    if (!node.outputs.empty()) {
      auto &outFrame = graph.frames[node.outputs[0].frameIndex];
      targetSize = frameSizes[outFrame.size].size;
      vpScale = frameSizes[outFrame.size].viewportScale;
      if (viewData.viewport) {
        float4 vp4(viewData.viewport->toCorners());
        vp4 *= float4(vpScale.x, vpScale.y, vpScale.x, vpScale.y);
        viewData.viewport = Rect::fromCorners(int4(linalg::floor(vp4)));
      }
    }

    RenderGraphEncodeContext ctx{
        .encoder = renderPassEncoder,
        .viewData = viewData,
        .outputSize = targetSize,
        .viewportScale = vpScale,
        .evaluator = *this,
        .graph = graph,
        .node = node,
        .ignoreInDebugger = ignoreInDebugger,
    };
    if (node.encode)
      node.encode(ctx);
    wgpuRenderPassEncoderEnd(renderPassEncoder);
  }

  WGPUCommandBufferDescriptor cmdBufDesc{};
  WgpuHandle<WGPUCommandBuffer> cmdBuf(wgpuCommandEncoderFinish(commandEncoder, &cmdBufDesc));

  context.submit(cmdBuf);

  // TODO: move this earlier into the process once rendering is multi-threaded
  // TODO(rendergraph)
  for (auto &queue : graph.autoClearQueues) {
    queue->clear();
  }
}

} // namespace gfx::detail
