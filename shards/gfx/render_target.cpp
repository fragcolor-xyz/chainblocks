#include "render_target.hpp"
#include "fwd.hpp"
#include <spdlog/fmt/fmt.h>

namespace gfx {
void RenderTarget::configure(FastString name, WGPUTextureFormat format) {
  auto texture = Texture::makeRenderAttachment(format, fmt::format("{}/{}", label, name));
  attachments.emplace(name, texture);
}

int2 RenderTarget::resizeConditional(int2 referenceSize) {
  computedSize = computeSize(referenceSize);
  for (auto &it : attachments) {
    int2 actualSize = it.second.getResolutionFromMipResolution(computedSize);
    auto desc = it.second.texture->getDesc();
    if (desc.getFormat().resolution != actualSize) {
      desc.getFormat().resolution = actualSize;
      it.second.texture->init(desc);
    }
  }
  return computedSize;
}

void RenderTarget::resizeFixed(int2 size) {
  this->size = RenderTargetSize::Fixed{.size = size};
  resizeConditional(int2());
}

int2 RenderTarget::getSize() const { return computedSize; }
int2 RenderTarget::computeSize(int2 referenceSize) const {
  int2 result{};
  std::visit([&](auto &&arg) { result = arg.apply(referenceSize); }, size);
  return result;
}

const TextureSubResource &RenderTarget::getAttachment(FastString name) const {
  auto it = attachments.find(name);
  if (it != attachments.end())
    return it->second;

  static TextureSubResource None{TexturePtr()};
  return None;
}
} // namespace gfx
