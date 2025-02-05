#ifndef GFX_GFX_WGPU
#define GFX_GFX_WGPU

#ifdef WEBGPU_NATIVE
extern "C" {
#include <webgpu.h>
#include <wgpu.h>
}
#else
extern "C" {
#include <webgpu/webgpu.h>
}
#endif

#ifdef WEBGPU_NATIVE
// Alias Undefined to Clear so wgpu is satisfied
#define WGPULoadOp_Undefined WGPULoadOp_Clear
#define WGPUStoreOp_Undefined WGPUStoreOp_Discard
#endif

struct WGPUAdapterReceiverData {
  bool completed = false;
  WGPURequestAdapterStatus status;
  const char *message;
  WGPUAdapter adapter;
};
WGPUAdapter wgpuInstanceRequestAdapterSync(WGPUInstance instance, const WGPURequestAdapterOptions *options,
                                           WGPUAdapterReceiverData *receiverData);

struct WGPUDeviceReceiverData {
  bool completed = false;
  WGPURequestDeviceStatus status;
  const char *message;
  WGPUDevice device;
};
WGPUDevice wgpuAdapterRequestDeviceSync(WGPUAdapter adapter, const WGPUDeviceDescriptor *descriptor,
                                        WGPUDeviceReceiverData *receiverData);

#define WGPU_SAFE_RELEASE(_fn, _x) \
  if (_x) {                        \
    _fn(_x);                       \
    _x = nullptr;                  \
  }

inline void wgpuShaderModuleWGSLDescriptorSetCode(WGPUShaderModuleWGSLDescriptor &desc, const char *code) { desc.code = code; }

// Default limits as described by the spec (https://www.w3.org/TR/webgpu/#limits)
WGPULimits wgpuGetDefaultLimits();

// workaround for emscripten not implementing limits
void gfxWgpuDeviceGetLimits(WGPUDevice device, WGPUSupportedLimits *outLimits);

// When copying textures into buffers the bytesPerRow should be aligned to this number
inline constexpr size_t WGPU_COPY_BYTES_PER_ROW_ALIGNMENT = 256;

#if !WEBGPU_NATIVE
extern "C" {
WGPUSwapChain gfxWgpuDeviceCreateSwapChain(WGPUDevice device, WGPUSurface surface, WGPUSwapChainDescriptor const *descriptor);
void gfxWgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapModeFlags mode, size_t offset, size_t size, WGPUBufferMapCallback callback, void * userdata);
// Custom function implemented in javascript that reads a mapped buffer directly into the given address
// faster that the default implementation that copies the data into a temporary buffer
void gfxWgpuBufferReadInto(WGPUBuffer buffer, void* dst, size_t offset, size_t size);
}
#endif

#if WEBGPU_NATIVE && !RUST_BINDGEN
#include "rust/gfx/bindings.hpp"
#endif

#endif // GFX_GFX_WGPU
