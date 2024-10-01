#include "context.hpp"
#include "context_data.hpp"
#include "enums.hpp"
#include "error_utils.hpp"
#include <boost/container/small_vector.hpp>
#include "../core/platform.hpp"
#include "../core/assert.hpp"
#include "platform_surface.hpp"
#include "window.hpp"
#include "log.hpp"
#include "texture.hpp"
#include "filesystem.hpp"
#include <tracy/Wrapper.hpp>
#include <magic_enum.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#if SHARDS_GFX_SDL
#include <SDL3/SDL_stdinc.h>
#endif

#if SH_EMSCRIPTEN
#include <emscripten/html5.h>
using WGPUInstanceRequestAdapterCallback = WGPURequestAdapterCallback;
using WGPUAdapterRequestDeviceCallback = WGPURequestDeviceCallback;
#endif

namespace gfx {
static auto logger = getLogger();
static auto wgpuLogger = getWgpuLogger();

#ifdef WEBGPU_NATIVE
static WGPUBackendType getDefaultWgpuBackendType() {
#if SH_WINDOWS
  // Vulkan is more performant on windows for now, see:
  // https://github.com/gfx-rs/wgpu/issues/2719 - Make DX12 the Default API on Windows
  // https://github.com/gfx-rs/wgpu/issues/2720 - Suballocate Buffers in DX12
  return WGPUBackendType_Vulkan;
#elif SH_APPLE
  return WGPUBackendType_Metal;
#elif SH_LINUX || SH_ANDROID
  return WGPUBackendType_Vulkan;
#elif SH_EMSCRIPTEN
  return WGPUBackendType_WebGPU;
#else
#error "No graphics backend defined for platform"
#endif
}
#endif

struct AdapterRequest {
  typedef AdapterRequest Self;

  WGPURequestAdapterStatus status;
  WGPUAdapter adapter;
  bool finished{};
  std::string message;

  static std::shared_ptr<Self> create(WGPUInstance wgpuInstance, const WGPURequestAdapterOptions &options) {
    auto result = std::make_shared<Self>();
    wgpuInstanceRequestAdapter(wgpuInstance, &options, (WGPUInstanceRequestAdapterCallback)&Self::callback,
                               new std::shared_ptr<Self>(result));
    return result;
  };

  static void callback(WGPURequestAdapterStatus status, WGPUAdapter adapter, char const *message, std::shared_ptr<Self> *handle) {
    (*handle)->status = status;
    (*handle)->adapter = adapter;
    if (message)
      (*handle)->message = message;
    (*handle)->finished = true;
    delete handle;
  };
};

struct DeviceRequest {
  typedef DeviceRequest Self;

  WGPURequestDeviceStatus status;
  WGPUDevice device;
  bool finished{};
  std::string message;

  static std::shared_ptr<Self> create(WGPUAdapter wgpuAdapter, const WGPUDeviceDescriptor &deviceDesc) {
    auto result = std::make_shared<Self>();
    wgpuAdapterRequestDevice(wgpuAdapter, &deviceDesc, (WGPUAdapterRequestDeviceCallback)&Self::callback,
                             new std::shared_ptr<Self>(result));
    return result;
  };

  static void callback(WGPURequestDeviceStatus status, WGPUDevice device, char const *message, std::shared_ptr<Self> *handle) {
    (*handle)->status = status;
    (*handle)->device = device;
    if (message)
      (*handle)->message = message;
    (*handle)->finished = true;
    delete handle;
  };
};

struct ContextMainOutput {
  Window *window{};
  WGPUSurface wgpuSurface{};
  WGPUTextureFormat swapchainFormat = WGPUTextureFormat_Undefined;
  int2 currentSize{};
  WGPUTexture wgpuCurrentTexture{};
  TexturePtr texture;

  ContextFlushTextureReferencesRegistry onFlushTextureReferences;

#ifndef WEBGPU_NATIVE
  WGPUSwapChain wgpuSwapChain{};
#endif

  ContextMainOutput(Window &window, ContextFlushTextureReferencesRegistry onFlushTextureReferences)
      : window(&window), onFlushTextureReferences(onFlushTextureReferences) {
    texture = std::make_shared<Texture>();
  }

  ~ContextMainOutput() {
#ifndef WEBGPU_NATIVE
    releaseSwapchain();
#endif
    releaseSurface();
  }

  WGPUSurface initSurface(WGPUInstance instance, void *overrideNativeWindowHandle) {
    ZoneScoped;

    if (!wgpuSurface) {
      void *surfaceHandle = overrideNativeWindowHandle;

#if SH_APPLE
      if (!surfaceHandle) {
        surfaceHandle = window->metalView->layer;
      }
#endif

      WGPUPlatformSurfaceDescriptor surfDesc(*window, surfaceHandle);
      wgpuSurface = wgpuInstanceCreateSurface(instance, &surfDesc);
    }

    return wgpuSurface;
  }

  bool requestFrame(WGPUDevice device, WGPUAdapter adapter) {
    shassert(!wgpuCurrentTexture);

    int2 drawableSize = window->getDrawableSize();
    if (drawableSize != currentSize) {
      resizeSwapchain(device, adapter, drawableSize);
    }

#ifdef WEBGPU_NATIVE
    WGPUSurfaceTexture st{};
    wgpuSurfaceGetCurrentTexture(wgpuSurface, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_Success) {
      SPDLOG_LOGGER_WARN(logger, "Failed to acquire surface texture: {}", magic_enum::enum_name(st.status));
      return false;
    }

    if (st.suboptimal) {
      SPDLOG_LOGGER_DEBUG(logger, "Suboptimal surface configuration");
    }
    wgpuCurrentTexture = st.texture;
#else
    if (!wgpuSwapChain)
      return false;
    wgpuCurrentTexture = wgpuSwapChainGetCurrentTexture(wgpuSwapChain);
#endif

    auto desc = std::get<TextureDescExternal>(texture->getDesc());
    desc.externalTexture = wgpuCurrentTexture;
    desc.format.flags = desc.format.flags | TextureFormatFlags::DontCache;
    texture->init(desc);

    return wgpuCurrentTexture != nullptr;
  }

  const TexturePtr &getCurrentTexture() const {
    shassert(wgpuCurrentTexture);
    return texture;
  }

  void present() {
    shassert(wgpuCurrentTexture);

#if WEBGPU_NATIVE
    wgpuSurfacePresent(wgpuSurface);
#endif

    wgpuTextureRelease(wgpuCurrentTexture);
    wgpuCurrentTexture = nullptr;

    FrameMark;
  }

  void initSwapchain(WGPUDevice device, WGPUAdapter adapter) {
    int2 mainOutputSize = window->getDrawableSize();
    resizeSwapchain(device, adapter, mainOutputSize);
  }

  void resizeSwapchain(WGPUDevice device, WGPUAdapter adapter, const int2 &newSize) {
    if (newSize.x <= 0 || newSize.y <= 0) {
      return;
    }

    WGPUTextureFormat preferredFormat = WGPUTextureFormat_Undefined;
    WGPUSurfaceCapabilities capabilities{};
    wgpuSurfaceGetCapabilities(wgpuSurface, adapter, &capabilities);

    for (size_t i = 0; i < capabilities.formatCount; i++) {
      auto format = capabilities.formats[i];
      if (preferredFormat == WGPUTextureFormat_Undefined) {
        preferredFormat = format;
      }
      SPDLOG_LOGGER_DEBUG(logger, "Surface format: {}", magic_enum::enum_name(format));
    }

    if (preferredFormat == WGPUTextureFormat_Undefined) {
      throw formatException("Failed to reconfigure Surface with format {}", preferredFormat);
    }

    if (preferredFormat != swapchainFormat) {
      SPDLOG_LOGGER_DEBUG(logger, "Surface preferred format changed: {}", magic_enum::enum_name(preferredFormat));
      swapchainFormat = preferredFormat;
    }

    shassert(newSize.x > 0 && newSize.y > 0);
    shassert(device);
    shassert(wgpuSurface);
    shassert(swapchainFormat != WGPUTextureFormat_Undefined);

    SPDLOG_LOGGER_DEBUG(logger, "Configuring surface width: {}, height: {}, format: {}", newSize.x, newSize.y, swapchainFormat);
    currentSize = newSize;

    // Force flush all texture references before resizing
    // this should cause all references to the surface texture to be released
    onFlushTextureReferences->forEach([](ContextFlushTextureReferencesHandler &target) { target.flushTextureReferences(); });

    TextureFormatFlags textureFormatFlags = TextureFormatFlags::RenderAttachment | TextureFormatFlags::NoTextureBinding;

    WGPUTextureFormat viewFormats[1];
    size_t viewFormatCount = 0;

    auto pixelFormatDesc = getTextureFormatDescription(swapchainFormat);
    // When using non-srgb integer storage types, assume that the device interprets it as sRGB
    if (pixelFormatDesc.colorSpace != ColorSpace::Srgb && isIntegerStorageType(pixelFormatDesc.storageType)) {
      auto upgradedFormat = upgradeToSrgbFormat(swapchainFormat);
      if (upgradedFormat) {
        viewFormats[0] = *upgradedFormat;
        viewFormatCount = 1;
        textureFormatFlags = textureFormatFlags | TextureFormatFlags::IsSecretlySrgb;
      } else {
        SPDLOG_LOGGER_ERROR(logger, "Failed to add sRGB view format to surface format {}",
                            magic_enum::enum_name(swapchainFormat));
      }
    }

#if WEBGPU_NATIVE
    WGPUSurfaceConfiguration surfaceConf = {};
    surfaceConf.format = swapchainFormat;
    surfaceConf.alphaMode = WGPUCompositeAlphaMode_Auto;
    surfaceConf.device = device;
    surfaceConf.viewFormats = viewFormats;
    surfaceConf.viewFormatCount = viewFormatCount;

    // Canvas size should't be set when configuring, instead resize the element
    // https://github.com/emscripten-core/emscripten/issues/17416
#if !SH_EMSCRIPTEN
    surfaceConf.width = newSize.x;
    surfaceConf.height = newSize.y;
#endif

    surfaceConf.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;

    std::optional<WGPUPresentMode> presentMode;
    auto trySetPresentMode = [&](WGPUPresentMode mode) {
      if (presentMode)
        return;
      for (int i = 0; i < capabilities.presentModeCount; i++) {
        if (capabilities.presentModes[i] == mode) {
          presentMode = mode;
          break;
        }
      }
    };
    trySetPresentMode(WGPUPresentMode_Mailbox);
    trySetPresentMode(WGPUPresentMode_Immediate);
    trySetPresentMode(WGPUPresentMode_Fifo);
    if (!presentMode) {
      if (capabilities.presentModeCount == 0)
        throw formatException("Failed to set present mode, nothing is supported");
      presentMode = capabilities.presentModes[0];
    }

    surfaceConf.presentMode = *presentMode;
    wgpuSurfaceConfigure(wgpuSurface, &surfaceConf);
#else
    WGPUSwapChainDescriptor desc{
        .label = "<swapchain>",
        .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst,
        .format = swapchainFormat,
        .width = uint32_t(newSize.x),
        .height = uint32_t(newSize.y),
        .presentMode = WGPUPresentMode_Fifo,
    };
    releaseSwapchain();
    wgpuSwapChain = gfxWgpuDeviceCreateSwapChain(device, wgpuSurface, &desc);
#endif

    texture
        ->init(TextureDescExternal{.format{
            .resolution = currentSize,
            .flags = textureFormatFlags,
            .pixelFormat = swapchainFormat,
        }})
        .initWithLabel("<main output>");
  }

  void releaseSurface() {
    if (wgpuCurrentTexture) {
      // Frame was possibly interupted, free texture here
      wgpuTextureRelease(wgpuCurrentTexture);
      wgpuCurrentTexture = nullptr;
    }

    // Force flush all texture references before releasing
    // this should cause all references to the surface texture to be released
    onFlushTextureReferences->forEach([](ContextFlushTextureReferencesHandler &target) { target.flushTextureReferences(); });

    WGPU_SAFE_RELEASE(wgpuSurfaceRelease, wgpuSurface);
  }
#ifndef WEBGPU_NATIVE
  void releaseSwapchain() { WGPU_SAFE_RELEASE(wgpuSwapChainRelease, wgpuSwapChain); }
#endif
};

Context::Context() {}
Context::~Context() { release(); }

void Context::init(Window &window, const ContextCreationOptions &inOptions) {
  options = inOptions;
  mainOutput = std::make_shared<ContextMainOutput>(window, onFlushTextureReferences);

  initCommon();
}

void Context::init(const ContextCreationOptions &inOptions) {
  options = inOptions;

  initCommon();
}

void Context::release() {
  SPDLOG_LOGGER_DEBUG(logger, "release");
  state = ContextState::Uninitialized;

  releaseAdapter();
  mainOutput.reset();

#if WEBGPU_NATIVE
  WGPUGlobalReport report{};
  wgpuGenerateReport(wgpuInstance, &report);
  WGPUHubReport *hubReport{};
  switch (getBackendType()) {
  case WGPUBackendType_Vulkan:
    hubReport = &report.vulkan;
    break;
  case WGPUBackendType_D3D12:
    hubReport = &report.dx12;
    break;
  default:
    break;
  }

  if (hubReport) {
    auto dumpStat = [&](const char *name, auto &v) {
      if (v.numAllocated > 0 || v.numKeptFromUser > 0) {
        SPDLOG_LOGGER_WARN(logger, "Context has {} {} at release ({} released, {} kept)", v.numAllocated, name,
                           v.numReleasedFromUser, v.numKeptFromUser);
      }
    };

    dumpStat("adapters", hubReport->adapters);
    dumpStat("devices", hubReport->devices);
    dumpStat("queues", hubReport->queues);
    dumpStat("pipelineLayouts", hubReport->pipelineLayouts);
    dumpStat("shaderModules", hubReport->shaderModules);
    dumpStat("bindGroupLayouts", hubReport->bindGroupLayouts);
    dumpStat("bindGroups", hubReport->bindGroups);
    dumpStat("commandBuffers", hubReport->commandBuffers);
    dumpStat("renderBundles", hubReport->renderBundles);
    dumpStat("renderPipelines", hubReport->renderPipelines);
    dumpStat("computePipelines", hubReport->computePipelines);
    dumpStat("querySets", hubReport->querySets);
    dumpStat("buffers", hubReport->buffers);
    dumpStat("textures", hubReport->textures);
    dumpStat("textureViews", hubReport->textureViews);
    dumpStat("samplers", hubReport->samplers);
  }
#endif

  WGPU_SAFE_RELEASE(wgpuInstanceRelease, wgpuInstance);
}

Window &Context::getWindow() {
  shassert(mainOutput);
  return *mainOutput->window;
}

void Context::resizeMainOutputConditional(const int2 &newSize) {
  ZoneScoped;

  if (state != ContextState::Ok)
    return;

  shassert(mainOutput);
  if (mainOutput->currentSize != newSize) {
    // Need to sync before to make sure surface is no longer in use
    poll(true);
    mainOutput->resizeSwapchain(wgpuDevice, wgpuAdapter, newSize);
  }
}

TexturePtr Context::getMainOutputTexture() {
  shassert(mainOutput);
  return mainOutput->getCurrentTexture();
}

bool Context::isHeadless() const { return !mainOutput; }

bool Context::beginFrame() {
  ZoneScoped;
  shassert(frameState == ContextFrameState::Ok);

  // Automatically request
  if (state == ContextState::Incomplete) {
    requestDevice();
  }

  if (!isReady())
    tickRequesting();

  if (state != ContextState::Ok)
    return false;

  if (suspended)
    return false;

  // collectContextData();
  if (!isHeadless()) {
    const int maxAttempts = 2;
    bool success = false;

    // Try to request the swapchain texture, automatically recreate swapchain on failure
    for (size_t i = 0; !success && i < maxAttempts; i++) {
      success = mainOutput->requestFrame(wgpuDevice, wgpuAdapter);
      if (!success) {
        SPDLOG_LOGGER_DEBUG(logger, "Failed to get current swapchain texture, forcing recreate");
        mainOutput->initSwapchain(wgpuDevice, wgpuAdapter);
      }
    }

    if (!success)
      return false;
  }

  frameState = ContextFrameState::WaitingForEnd;

  return true;
}

void Context::endFrame() {
  ZoneScoped;
  shassert(frameState == ContextFrameState::WaitingForEnd);

  if (!isHeadless())
    present();

  frameState = ContextFrameState::Ok;
}

void Context::poll(bool wait) {
  ZoneScoped;
#ifdef WEBGPU_NATIVE
  if (wait) {
    while (!wgpuDevicePoll(wgpuDevice, true, nullptr)) {
      // Poll until no more submissions are in flight
    }
  } else {
    wgpuDevicePoll(wgpuDevice, false, nullptr);
  }
#endif
}

void Context::suspend() {
#if SH_ANDROID
  // Also release the surface on suspend
  deviceLost();

  if (mainOutput)
    mainOutput->releaseSurface();
#elif SH_APPLE
  deviceLost();
#endif

  suspended = true;
}

void Context::resume() { suspended = false; }

void Context::submit(WGPUCommandBuffer cmdBuffer) {
  ZoneScoped;
  wgpuQueueSubmit(wgpuQueue, 1, &cmdBuffer);
}

void Context::deviceLost() {
  if (state != ContextState::Incomplete) {
    SPDLOG_LOGGER_DEBUG(logger, "Device lost");
    state = ContextState::Incomplete;

    releaseDevice();
  }
}

void Context::tickRequesting() {
  ZoneScoped;
  SPDLOG_LOGGER_DEBUG(logger, "tickRequesting");
  try {
    if (adapterRequest) {
      if (adapterRequest->finished) {
        if (adapterRequest->status != WGPURequestAdapterStatus_Success) {
          throw formatException("Failed to create adapter: {} {}", magic_enum::enum_name(adapterRequest->status),
                                adapterRequest->message);
        }

        wgpuAdapter = adapterRequest->adapter;
        adapterRequest.reset();

        // Immediately request a device from the adapter
        requestDevice();
      }
    }

    if (deviceRequest) {
      if (deviceRequest->finished) {
        if (deviceRequest->status != WGPURequestDeviceStatus_Success) {
          throw formatException("Failed to create device: {} {}", magic_enum::enum_name(deviceRequest->status),
                                deviceRequest->message);
        }
        wgpuDevice = deviceRequest->device;
        deviceRequest.reset();
        deviceObtained();
      }
    }
  } catch (...) {
    deviceLost();
    throw;
  }
}

void Context::deviceObtained() {
  ZoneScoped;
  state = ContextState::Ok;
  SPDLOG_LOGGER_DEBUG(logger, "wgpuDevice obtained");

  wgpuQueue = wgpuDeviceGetQueue(wgpuDevice);

  if (mainOutput) {
    getOrCreateSurface();
    mainOutput->initSwapchain(wgpuDevice, wgpuAdapter);
  }

  onDeviceStatus->forEach([](ContextDeviceStatusHandler &target) { target.deviceAcquired(); });
}

void Context::requestDevice() {
  ZoneScoped;
  shassert(!wgpuDevice);
  shassert(!wgpuQueue);

  // Request adapter first
  if (!wgpuAdapter) {
    requestAdapter();
    tickRequesting(); // This chains into another requestDevice call once the adapter is obtained
    return;
  }

  state = ContextState::Requesting;

  WGPUDeviceDescriptor deviceDesc{};

  WGPUDeviceLostCallback deviceLostCallback = [](WGPUDeviceLostReason reason, char const *message, void *userdata) {
    SPDLOG_LOGGER_WARN(logger, "Device lost: {} ()", message, magic_enum::enum_name(reason));
    Context *context = static_cast<Context *>(userdata);
    context->deviceLost();
  };
  deviceDesc.deviceLostCallback = deviceLostCallback;
  deviceDesc.deviceLostUserdata = this;
  deviceDesc.defaultQueue.label = "queue";

  // Passed to force full feature set to be enabled
#if WEBGPU_NATIVE
  WGPURequiredLimits requiredLimits = {.limits = wgpuGetDefaultLimits()};
  deviceDesc.requiredLimits = &requiredLimits;

  // Lower default limits to support devices like iOS simulator
  requiredLimits.limits.maxBufferSize = 256 * 1024 * 1024;
  WGPURequiredLimitsExtras extraLimits{
      .chain =
          WGPUChainedStruct{
              .sType = (WGPUSType)WGPUSType_RequiredLimitsExtras,
          },
      .limits =
          {
              .maxPushConstantSize = 0,
              .maxNonSamplerBindings = 1000000,
          },
  };
  requiredLimits.nextInChain = &extraLimits.chain;

#if WEBGPU_TRACE
  // This defines `const char* wgpuTracePath`, as specified by the CMakeLists.txt
#include "wgpu_trace_config.h"
  // Setup tracing
  WGPUDeviceExtras deviceExtras{.chain = {.sType = (WGPUSType)WGPUSType_DeviceExtras}};
  deviceExtras.tracePath = wgpuTracePath;
  deviceDesc.nextInChain = &deviceExtras.chain;
#endif
#endif

  SPDLOG_LOGGER_DEBUG(logger, "Requesting wgpu device");
  deviceRequest = DeviceRequest::create(wgpuAdapter, deviceDesc);
}

void Context::releaseDevice() {
  ZoneScoped;

  onDeviceStatus->forEach([](ContextDeviceStatusHandler &target) { target.deviceLost(); });

  if (mainOutput) {
    mainOutput->releaseSurface();
  }

  if (wgpuDevice) {
    poll(true);
  }

  WGPU_SAFE_RELEASE(wgpuQueueRelease, wgpuQueue);
  WGPU_SAFE_RELEASE(wgpuDeviceRelease, wgpuDevice);
}

WGPUSurface Context::getOrCreateSurface() {
  if (mainOutput)
    return mainOutput->initSurface(wgpuInstance, options.overrideNativeWindowHandle);
  return nullptr;
}

void Context::requestAdapter() {
  ZoneScoped;
  shassert(!wgpuAdapter);

  state = ContextState::Requesting;

  std::optional<WGPUBackendType> backend;
#if SHARDS_GFX_SDL
  if (const char *backendStr = SDL_getenv("SHARDS_GFX_BACKEND")) {
    std::string typeStr = std::string("WGPUBackendType_") + backendStr;
    backend = magic_enum::enum_cast<WGPUBackendType>(typeStr);
    if (backend) {
      SPDLOG_LOGGER_DEBUG(logger, "Using backend {}", magic_enum::enum_name(*backend));
    }
  }
#endif

  if (!wgpuInstance) {
#if WEBGPU_NATIVE
    WGPUInstanceDescriptor desc{};
    WGPUInstanceExtras extras{
        .chain = {.sType = (WGPUSType)WGPUSType_InstanceExtras},
    };

    instanceBackends = WGPUInstanceBackend_Primary;
    if (backend) {
      switch (*backend) {
      case WGPUBackendType_D3D11:
        instanceBackends = WGPUInstanceBackend_DX11;
        break;
      case WGPUBackendType_D3D12:
        instanceBackends = WGPUInstanceBackend_DX12;
        break;
      case WGPUBackendType_Metal:
        instanceBackends = WGPUInstanceBackend_Metal;
        break;
      case WGPUBackendType_Vulkan:
        instanceBackends = WGPUInstanceBackend_Vulkan;
        break;
      default:
        break;
      }
    }
    extras.backends = instanceBackends;

    if (const char *debug = SDL_getenv("SHARDS_GFX_DEBUG")) {
      extras.flags |= WGPUInstanceFlag_Debug;
    }
    if (const char *debug = SDL_getenv("SHARDS_GFX_VALIDATION")) {
      extras.flags |= WGPUInstanceFlag_Validation;
    }
    desc.nextInChain = &extras.chain;
    wgpuInstance = wgpuCreateInstance(&desc);
#else
    // Must be passed nullptr, until supported
    wgpuInstance = wgpuCreateInstance(nullptr);
#endif
  }

#if WEBGPU_NATIVE
  WGPUInstanceEnumerateAdapterOptions enumOpts{.backends = instanceBackends};
  boost::container::small_vector<WGPUAdapter, 32> adapters;
  adapters.resize(wgpuInstanceEnumerateAdapters(wgpuInstance, &enumOpts, nullptr));
  adapters.resize(wgpuInstanceEnumerateAdapters(wgpuInstance, &enumOpts, adapters.data()));

  WGPUAdapter adapterToUse{};
  SPDLOG_LOGGER_DEBUG(logger, "Enumerating {} adapters", adapters.size());

  bool useAnyAdapter = {};
  if (const char *v = SDL_getenv("GFX_ANY_ADAPTER")) {
    useAnyAdapter = true;
  }
  for (size_t i = 0; i < adapters.size(); i++) {
    auto &adapter = adapters[i];
    WGPUAdapterInfo props;
    wgpuAdapterGetInfo(adapter, &props);

    SPDLOG_LOGGER_DEBUG(logger, "WGPUAdapter: {}", i);
    SPDLOG_LOGGER_DEBUG(logger, R"(WGPUAdapterProperties {{
  vendorID: {}
  architecture: {}
  deviceID: {}
  description: {}
  adapterType: {}
  backendType: {}
}})",
                        props.vendorID, props.architecture, props.deviceID, props.description, props.adapterType,
                        props.backendType);
    if (!adapterToUse && (useAnyAdapter || props.adapterType == WGPUAdapterType_DiscreteGPU)) {
      adapterToUse = adapter;
      backendType = props.backendType;
    } else {
      wgpuAdapterRelease(adapter);
    }
  }

  if (adapterToUse) {
    wgpuAdapter = adapterToUse;
    requestDevice();
  } else
#endif
  {
    WGPURequestAdapterOptions requestAdapter = {};
    requestAdapter.powerPreference = WGPUPowerPreference_HighPerformance;
    requestAdapter.compatibleSurface = getOrCreateSurface();
    requestAdapter.forceFallbackAdapter = false;

#ifdef WEBGPU_NATIVE
    requestAdapter.backendType = WGPUBackendType_Null;

    if (requestAdapter.backendType == WGPUBackendType_Null)
      requestAdapter.backendType = getDefaultWgpuBackendType();

    if (backend) {
      requestAdapter.backendType = *backend;
    }
    backendType = requestAdapter.backendType;
#endif

    SPDLOG_LOGGER_DEBUG(logger, "Requesting wgpu adapter");
    adapterRequest = AdapterRequest::create(wgpuInstance, requestAdapter);
  }
}

void Context::releaseAdapter() {
  ZoneScoped;
  releaseDevice();
  WGPU_SAFE_RELEASE(wgpuAdapterRelease, wgpuAdapter);
}

void Context::initCommon() {
  ZoneScoped;
  SPDLOG_LOGGER_DEBUG(logger, "initCommon");

  shassert(!isInitialized());

#ifdef WEBGPU_NATIVE
  wgpuSetLogCallback(
      [](WGPULogLevel level, const char *msg, void *userData) {
        (void)userData;
        switch (level) {
        case WGPULogLevel_Error:
          wgpuLogger->error("{}", msg);
          break;
        case WGPULogLevel_Warn:
          wgpuLogger->warn("{}", msg);
          break;
        case WGPULogLevel_Info:
          // Default logging is too verbose for info
          wgpuLogger->debug("{}", msg);
          break;
        case WGPULogLevel_Debug:
          // Debug too verbose for debug level
          wgpuLogger->trace("{}", msg);
          break;
        case WGPULogLevel_Trace:
          wgpuLogger->trace("{}", msg);
          break;
        default:
          break;
        }
      },
      this);

  if (wgpuLogger->level() == spdlog::level::trace) {
    wgpuSetLogLevel(WGPULogLevel_Trace);
  } else if (wgpuLogger->level() == spdlog::level::debug) {
    wgpuSetLogLevel(WGPULogLevel_Debug);
  } else {
    wgpuSetLogLevel(WGPULogLevel_Info);
  }
#endif

  requestDevice();
}

void Context::present() {
  ZoneScoped;
  shassert(mainOutput);
  mainOutput->present();
}

} // namespace gfx
