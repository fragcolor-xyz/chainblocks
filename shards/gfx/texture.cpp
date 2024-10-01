#include "texture.hpp"
#include "context.hpp"
#include "error_utils.hpp"
#include "gfx_wgpu.hpp"
#include <tracy/Wrapper.hpp>
#include <magic_enum.hpp>
#include <map>

namespace gfx {
TextureDescGPUOnly TextureDescGPUOnly::getDefault() {
  return TextureDescGPUOnly{.format = {.resolution = int2(512, 512), .pixelFormat = WGPUTextureFormat_RGBA8Unorm}};
}

TextureFormat TextureFormat::deriveFrom(size_t numComponents, StorageType storageType, bool preferSrgb) {
  TextureFormat result{
      .dimension = TextureDimension::D2,
  };

  auto throwUnsupportedPixelFormat = [&]() {
    throw formatException("Unsupported pixel format for {} components: {} ", numComponents, magic_enum::enum_name(storageType));
  };
  if (numComponents == 4) {
    if (storageType == StorageType::UNorm8)
      result.pixelFormat = preferSrgb ? WGPUTextureFormat_RGBA8UnormSrgb : WGPUTextureFormat_RGBA8Unorm;
    else if (storageType == StorageType::Float32)
      result.pixelFormat = WGPUTextureFormat_RGBA32Float;
    else
      throwUnsupportedPixelFormat();
  } else if (numComponents == 2) {
    if (storageType == StorageType::UNorm8)
      result.pixelFormat = WGPUTextureFormat_RG8Unorm;
    else if (storageType == StorageType::Float32)
      result.pixelFormat = WGPUTextureFormat_RG32Float;
    else
      throwUnsupportedPixelFormat();
  } else if (numComponents == 1) {
    if (storageType == StorageType::UNorm8)
      result.pixelFormat = WGPUTextureFormat_R8Unorm;
    else if (storageType == StorageType::Float32)
      result.pixelFormat = WGPUTextureFormat_R32Float;
    else
      throwUnsupportedPixelFormat();
  } else {
    throw formatException("{} component textures not supported", numComponents);
  }
  return result;
}

std::vector<uint8_t> convertToRGBA(const TextureDescCPUCopy &desc) {
  shassert(desc.sourceChannels == 3);
  auto width = desc.format.resolution.x;
  auto height = desc.format.resolution.y;
  std::vector<uint8_t> rgbaData(width * height * 4);

  shassert(desc.sourceData);
  auto srcData = desc.sourceData.getData();
  size_t rowStride = desc.sourceRowStride != 0 ? desc.sourceRowStride : width * 3;
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      size_t srcIndex = y * rowStride + x * 3;
      size_t dstIndex = (y * width + x) * 4;

      rgbaData[dstIndex] = srcData[srcIndex];
      rgbaData[dstIndex + 1] = srcData[srcIndex + 1];
      rgbaData[dstIndex + 2] = srcData[srcIndex + 2];

      rgbaData[dstIndex + 3] = 255;
    }
  }
  return rgbaData;
}

std::shared_ptr<Texture> Texture::makeRenderAttachment(WGPUTextureFormat format, std::string &&label) {
  TextureDescCPUCopy desc{
      .format =
          {
              .flags = TextureFormatFlags::RenderAttachment,
              .pixelFormat = format,
          },
  };
  auto texture = std::make_shared<Texture>(std::move(label));
  texture->init(desc);
  return texture;
}

Texture &Texture::init(const TextureDesc &desc) {
  if (this->desc != desc || desc.isExternal()) {
    // NOTE: Always update when external texture is present
    update();
    this->desc = desc;
  }
  return *this;
}

Texture &Texture::initWithSamplerState(const SamplerState &samplerState) {
  if (this->samplerState != samplerState) {
    this->samplerState = samplerState;
    ++version;
  }
  return *this;
}

Texture &Texture::initWithLabel(std::string &&label) {
  this->label = std::move(label);
  update();
  return *this;
}

std::shared_ptr<Texture> Texture::clone() const {
  auto result = cloneSelfWithId(this, getNextId());
  result->update();
  return result;
}

static void writeTextureData(Context &context, const TextureFormat &format, const int2 &resolution, uint32_t numArrayLayers,
                             WGPUTexture texture, const uint8_t *data, size_t dataLength, uint32_t rowDataLength_) {
  ZoneScoped;

  const TextureFormatDesc &inputFormat = getTextureFormatDescription(format.pixelFormat);
  uint32_t rowDataLength = rowDataLength_ != 0 ? rowDataLength_ : inputFormat.pixelSize * resolution.x;

  WGPUImageCopyTexture dst{
      .texture = texture,
      .mipLevel = 0,
      .aspect = WGPUTextureAspect_All,
  };
  WGPUTextureDataLayout layout{
      .bytesPerRow = rowDataLength,
      .rowsPerImage = uint32_t(resolution.y),
  };
  WGPUExtent3D writeSize{
      .width = uint32_t(resolution.x),
      .height = uint32_t(resolution.y),
      .depthOrArrayLayers = numArrayLayers,
  };
  wgpuQueueWriteTexture(context.wgpuQueue, &dst, data, dataLength, &layout, &writeSize);
}

static void writeTextureData(Context &context, const TextureFormat &format, const int2 &resolution, uint32_t numArrayLayers,
                             WGPUTexture texture, const ImmutableSharedBuffer &isb, uint32_t rowDataLength_) {
  writeTextureData(context, format, resolution, numArrayLayers, texture, isb.getData(), isb.getLength(), rowDataLength_);
}

void Texture::initContextData(Context &context, TextureContextData &contextData) { contextData.init(getLabel()); }

void Texture::updateContextData(Context &context, TextureContextData &contextData) {
  ZoneScoped;

  // WGPUDevice device = context.wgpuDevice;
  // shassert(device);

  // contextData.id = id;
  // contextData.size.width = desc.resolution.x;
  // contextData.size.height = desc.resolution.y;
  // contextData.size.depthOrArrayLayers = 1;
  // contextData.format = desc.format;

  // if (contextData.size.width == 0 || contextData.size.height == 0)
  //   return;

  // if (desc.externalTexture) {
  //   contextData.externalTexture = desc.externalTexture.value();
  // } else {
  //   WGPUTextureDescriptor wgpuDesc = {};
  //   wgpuDesc.usage = WGPUTextureUsage_TextureBinding;
  //   if (desc.source.data) {
  //     wgpuDesc.usage |= WGPUTextureUsage_CopyDst;
  //   }

  //   if (textureFormatFlagsContains(desc.format.flags, TextureFormatFlags::RenderAttachment)) {
  //     wgpuDesc.usage |= WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
  //   }

  //   if (!textureFormatFlagsContains(desc.format.flags, TextureFormatFlags::NoTextureBinding)) {
  //     wgpuDesc.usage |= WGPUTextureUsage_TextureBinding;
  //   }

  //   switch (desc.format.dimension) {
  //   case TextureDimension::D1:
  //     wgpuDesc.dimension = WGPUTextureDimension_1D;
  //     contextData.size.height = 1;
  //     break;
  //   case TextureDimension::D2:
  //     wgpuDesc.dimension = WGPUTextureDimension_2D;
  //     break;
  //   case TextureDimension::Cube:
  //     wgpuDesc.dimension = WGPUTextureDimension_2D;
  //     contextData.size.depthOrArrayLayers = 6;
  //     break;
  //   default:
  //     shassert(false);
  //   }

  //   wgpuDesc.size = contextData.size;
  //   wgpuDesc.format = desc.format.pixelFormat;
  //   wgpuDesc.sampleCount = 1;
  //   wgpuDesc.mipLevelCount = desc.format.mipLevels;
  //   wgpuDesc.label = label.empty() ? "unknown" : label.c_str();

  //   contextData.texture.reset(wgpuDeviceCreateTexture(context.wgpuDevice, &wgpuDesc));
  //   shassert(contextData.texture);

  //   if (desc.source.data)
  //     if (desc.source.numChannels == 3) {
  //       auto rgbaData = convertToRGBA(desc);
  //       writeTextureData(context, desc.format, desc.resolution, contextData.size.depthOrArrayLayers, contextData.texture,
  //                        rgbaData.data(), rgbaData.size(), 0);
  //     } else {
  //       writeTextureData(context, desc.format, desc.resolution, contextData.size.depthOrArrayLayers, contextData.texture,
  //                        desc.source.data, desc.source.rowStride);
  //     }
  // }
}

UniqueId Texture::getNextId() {
  static UniqueIdGenerator gen(UniqueIdTag::Texture);
  return gen.getNext();
}

void Texture::update() { ++version; }

} // namespace gfx
