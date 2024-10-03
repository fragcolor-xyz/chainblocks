#include "./renderer.hpp"
#include <gfx/texture.hpp>
#include <memory>

namespace gfx {

TestRenderer::TestRenderer(std::shared_ptr<TestContext> context) : context(context) {
  debugging = isTestDebuggingEnabled();
  renderer = std::make_shared<Renderer>(*context.get());
}

TestRenderer::~TestRenderer() {
  cleanupRenderTarget();
  renderer->cleanup();
  renderer.reset();
}

void TestRenderer::createRenderTarget(int2 res) {
  // Ignore when debugging
  if (debugging)
    return;

  cleanupRenderTarget();
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
  rtTexture = wgpuDeviceCreateTexture(context->wgpuDevice, &textureDesc);
  shassert(rtTexture);

  // Wrap in texture
  TexturePtr rtTexture = std::make_shared<Texture>();
  rtTexture->init(TextureDescExternal{
      .format =
          TextureFormat{
              .resolution = rtSize,
              .dimension = TextureDimension::D2,
              .flags = TextureFormatFlags::RenderAttachment,
              .pixelFormat = rtFormat,
          },
      .externalTexture = this->rtTexture,
  });

  renderer->setMainOutput(Renderer::MainOutput{
      .texture = rtTexture,
  });
}

void TestRenderer::cleanupRenderTarget() { WGPU_SAFE_RELEASE(wgpuTextureRelease, rtTexture); }

TestFrame TestRenderer::getTestFrame() {
  if (debugging)
    return TestFrame();

  WGPUCommandEncoderDescriptor ceDesc{};
  WgpuHandle<WGPUCommandEncoder> commandEncoder(wgpuDeviceCreateCommandEncoder(context->wgpuDevice, &ceDesc));

  WGPUBufferDescriptor desc{};
  size_t bufferPitch = sizeof(TestFrame::pixel_t) * rtSize.x;
  desc.size = bufferPitch * rtSize.y;
  desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
  WGPUBuffer tempBuffer = wgpuDeviceCreateBuffer(context->wgpuDevice, &desc);

  WGPUImageCopyTexture src{};
  src.texture = rtTexture;
  src.aspect = WGPUTextureAspect_All;
  src.mipLevel = 0;
  WGPUImageCopyBuffer dst{};
  dst.buffer = tempBuffer;
  dst.layout.bytesPerRow = bufferPitch;
  dst.layout.rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED;
  WGPUExtent3D extent{};
  extent.width = rtSize.x;
  extent.height = rtSize.y;
  extent.depthOrArrayLayers = 1;
  wgpuCommandEncoderCopyTextureToBuffer(commandEncoder, &src, &dst, &extent);

  WGPUCommandBufferDescriptor finishDesc{};
  WgpuHandle<WGPUCommandBuffer> copyCommandBuffer(wgpuCommandEncoderFinish(commandEncoder, &finishDesc));
  wgpuQueueSubmit(context->wgpuQueue, 1, &copyCommandBuffer.handle);

  auto mapBufferCallback = [](WGPUBufferMapAsyncStatus status, void *userdata) {};
  wgpuBufferMapAsync(tempBuffer, WGPUMapMode_Read, 0, desc.size, mapBufferCallback, nullptr);
  context->poll();

  uint8_t *bufferData = (uint8_t *)wgpuBufferGetMappedRange(tempBuffer, 0, desc.size);
  TestFrame testFrame(bufferData, rtSize, TestFrameFormat::RGBA8, bufferPitch, false);

  wgpuBufferUnmap(tempBuffer);
  wgpuBufferRelease(tempBuffer);

  return testFrame;
}
} // namespace gfx
