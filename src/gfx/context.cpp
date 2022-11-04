#include "context.hpp"
#include "context_data.hpp"
#include "error_utils.hpp"
#include "platform.hpp"
#include "platform_surface.hpp"
#include "window.hpp"
#include "log.hpp"
#include <magic_enum.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <SDL_stdinc.h>
#include <gfx_rust.h>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#if GFX_EMSCRIPTEN
#include <emscripten/html5.h>
#endif

namespace gfx {

static auto logger = getLogger();

static WGPUTextureFormat getDefaultSrgbBackbufferFormat() {
#if GFX_ANDROID
  return WGPUTextureFormat_RGBA8UnormSrgb;
#else
  return WGPUTextureFormat_BGRA8UnormSrgb;
#endif
}

#ifdef WEBGPU_NATIVE
static WGPUBackendType getDefaultWgpuBackendType() {
#if GFX_WINDOWS
  // Vulkan is more performant on windows for now, see:
  // https://github.com/gfx-rs/wgpu/issues/2719 - Make DX12 the Default API on Windows
  // https://github.com/gfx-rs/wgpu/issues/2720 - Suballocate Buffers in DX12
  return WGPUBackendType_Vulkan;
#elif GFX_APPLE
  return WGPUBackendType_Metal;
#elif GFX_LINUX || GFX_ANDROID
  return WGPUBackendType_Vulkan;
#elif GFX_EMSCRIPTEN
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
    wgpuInstanceRequestAdapterEx(wgpuInstance, &options, (WGPURequestAdapterCallback)&Self::callback,
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
    wgpuAdapterRequestDeviceEx(wgpuAdapter, &deviceDesc, (WGPURequestDeviceCallback)&Self::callback,
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

struct VulkanOpenXRBackend : public IBackend {
  WGPUInstance wgpuInstance{};
  WGPUAdapter wgpuAdapter{};
  WGPUDevice wgpuDevice{};
  WGPUQueue wgpuQueue{};

  vk::DispatchLoaderDynamic loader;
  vk::Instance instance;

  VkPhysicalDevice physicalDeviceToUse{};

  WGPUInstance createInstance() {
    // Create instance
    WGPUInstanceDescriptor desc{};
    WGPUInstanceDescriptorVK descVk{};
    if (getDefaultWgpuBackendType() == WGPUBackendType_Vulkan) {
      std::vector<const char *> requiredExtensions = {};

      descVk = {
          .chain{.sType = (WGPUSType)WGPUNativeSTypeEx_InstanceDescriptorVK},
          .requiredExtensions = requiredExtensions.data(),
          .requiredExtensionCount = (uint32_t)requiredExtensions.size(),
      };
      desc.nextInChain = &descVk.chain;
    }

    wgpuInstance = wgpuCreateInstanceEx(&desc);
    if (!wgpuInstance)
      throw std::runtime_error("Failed to create WGPUInstance");

    WGPUInstanceProperties props{};
    WGPUInstancePropertiesVK propsVk{
        .chain = {.sType = WGPUSType(WGPUNativeSTypeEx_InstancePropertiesVK)},
    };
    if (getDefaultWgpuBackendType() == WGPUBackendType_Vulkan) {
      props.nextInChain = &propsVk.chain;
    }
    wgpuInstanceGetPropertiesEx(wgpuInstance, &props);

    loader = vk::DispatchLoaderDynamic(PFN_vkGetInstanceProcAddr(propsVk.getInstanceProcAddr));

    instance = vk::Instance(VkInstance(propsVk.instance));
    loader.init(instance);

    auto devices = instance.enumeratePhysicalDevices(loader);
    for (auto &device : devices) {
      auto properties = device.getProperties(loader);
      auto name = std::string(properties.deviceName.begin(), properties.deviceName.end());
      SPDLOG_INFO("vulkan physical device: {}", name);
      physicalDeviceToUse = device;
      break;
    }
    assert(physicalDeviceToUse);

    return wgpuInstance;
  }

  std::shared_ptr<AdapterRequest> requestAdapter() {
    WGPURequestAdapterOptionsVK optionsVk{
        .chain = {.sType = WGPUSType(WGPUNativeSTypeEx_RequestAdapterOptionsVK)},
        .physicalDevice = physicalDeviceToUse,
    };

    WGPURequestAdapterOptions options{};
    options.nextInChain = &optionsVk.chain;

    return AdapterRequest::create(wgpuInstance, options);
  }

  std::shared_ptr<DeviceRequest> requestDevice() {
    std::vector<const char *> requiredExtensions = {};

    WGPUDeviceDescriptorVK deviceDescVk{
        .chain = {.sType = WGPUSType(WGPUNativeSTypeEx_DeviceDescriptorVK)},
        .requiredExtensions = requiredExtensions.data(),
        .requiredExtensionCount = (uint32_t)requiredExtensions.size(),
    };

    WGPUDeviceDescriptor deviceDesc = {};
    deviceDesc.nextInChain = &deviceDescVk.chain;

    WGPURequiredLimits requiredLimits = {.limits = wgpuGetDefaultLimits()};
    deviceDesc.requiredLimits = &requiredLimits;

    return DeviceRequest::create(wgpuAdapter, deviceDesc);
  }

  virtual void setAdapter(WGPUAdapter adapter) { wgpuAdapter = adapter; }
  virtual void setDevice(WGPUDevice device) { wgpuDevice = device; }
};

struct ContextMainOutput {
  Window *window{};
  WGPUSwapChain wgpuSwapchain{};
  WGPUSurface wgpuWindowSurface{};
  WGPUTextureFormat swapchainFormat = WGPUTextureFormat_Undefined;
  int2 currentSize{};
  WGPUTextureView currentView{};

#if GFX_APPLE
  std::unique_ptr<MetalViewContainer> metalViewContainer;
#endif

  ContextMainOutput(Window &window) { this->window = &window; }
  ~ContextMainOutput() {
    releaseSwapchain();
    releaseSurface();
  }

  WGPUSurface initSurface(WGPUInstance instance, void *overrideNativeWindowHandle) {
    if (!wgpuWindowSurface) {
      void *surfaceHandle = overrideNativeWindowHandle;

#if GFX_APPLE
      if (!surfaceHandle) {
        metalViewContainer = std::make_unique<MetalViewContainer>(window->window);
        surfaceHandle = metalViewContainer->layer;
      }
#endif

      WGPUPlatformSurfaceDescriptor surfDesc(window->window, surfaceHandle);
      wgpuWindowSurface = wgpuInstanceCreateSurface(instance, &surfDesc);
    }

    return wgpuWindowSurface;
  }

  bool requestFrame() {
    assert(!currentView);
    currentView = wgpuSwapChainGetCurrentTextureView(wgpuSwapchain);

    return currentView;
  }

  void present() {
    assert(currentView);

    // Web doesn't have a swapchain, it automatically present the current texture when control
    // is returned to the browser
#ifdef WEBGPU_NATIVE
    wgpuSwapChainPresent(wgpuSwapchain);
#endif

    wgpuTextureViewRelease(currentView);
    currentView = nullptr;
  }

  void initSwapchain(WGPUAdapter adapter, WGPUDevice device) {
    swapchainFormat = wgpuSurfaceGetPreferredFormat(wgpuWindowSurface, adapter);
    int2 mainOutputSize = window->getDrawableSize();
    resizeSwapchain(device, adapter, mainOutputSize);
  }

  void resizeSwapchain(WGPUDevice device, WGPUAdapter adapter, const int2 &newSize) {
    WGPUTextureFormat preferredFormat = wgpuSurfaceGetPreferredFormat(wgpuWindowSurface, adapter);

    // Force the backbuffer to srgb format so we don't have to convert manually in shader
    preferredFormat = getDefaultSrgbBackbufferFormat();

    if (preferredFormat != swapchainFormat) {
      SPDLOG_LOGGER_DEBUG(logger, "swapchain preferred format changed: {}", magic_enum::enum_name(preferredFormat));
      swapchainFormat = preferredFormat;
    }

    assert(newSize.x > 0 && newSize.y > 0);
    assert(device);
    assert(wgpuWindowSurface);
    assert(swapchainFormat != WGPUTextureFormat_Undefined);

    SPDLOG_LOGGER_DEBUG(logger, "resized width: {} height: {}", newSize.x, newSize.y);
    currentSize = newSize;

    releaseSwapchain();

    WGPUSwapChainDescriptor swapchainDesc = {};
    swapchainDesc.format = swapchainFormat;
    swapchainDesc.width = newSize.x;
    swapchainDesc.height = newSize.y;
#if GFX_WINDOWS || GFX_OSX || GFX_LINUX
    swapchainDesc.presentMode = WGPUPresentMode_Immediate;
#else
    swapchainDesc.presentMode = WGPUPresentMode_Fifo;
#endif
    swapchainDesc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
    wgpuSwapchain = wgpuDeviceCreateSwapChain(device, wgpuWindowSurface, &swapchainDesc);
    if (!wgpuSwapchain) {
      throw formatException("Failed to create swapchain");
    }
  }

  void releaseSurface() { WGPU_SAFE_RELEASE(wgpuSurfaceRelease, wgpuWindowSurface); }
  void releaseSwapchain() { WGPU_SAFE_RELEASE(wgpuSwapChainRelease, wgpuSwapchain); }
};

Context::Context() {}
Context::~Context() { release(); }

void Context::init(Window &window, const ContextCreationOptions &inOptions) {
  options = inOptions;
  mainOutput = std::make_shared<ContextMainOutput>(window);

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

  WGPU_SAFE_RELEASE(wgpuInstanceDrop, wgpuInstance);
}

Window &Context::getWindow() {
  assert(mainOutput);
  return *mainOutput->window;
}

void Context::resizeMainOutputConditional(const int2 &newSize) {
  if (state != ContextState::Ok)
    return;

  assert(mainOutput);
  if (mainOutput->currentSize != newSize) {
    mainOutput->resizeSwapchain(wgpuDevice, wgpuAdapter, newSize);
  }
}

int2 Context::getMainOutputSize() const {
  assert(mainOutput);
  return mainOutput->currentSize;
}

WGPUTextureView Context::getMainOutputTextureView() {
  assert(mainOutput);
  assert(mainOutput->wgpuSwapchain);
  return mainOutput->currentView;
}

WGPUTextureFormat Context::getMainOutputFormat() const {
  assert(mainOutput);
  return mainOutput->swapchainFormat;
}

bool Context::isHeadless() const { return !mainOutput; }

void Context::addContextDataInternal(const std::weak_ptr<ContextData> &ptr) {
  assert(!ptr.expired());

  std::shared_ptr<ContextData> sharedPtr = ptr.lock();
  if (sharedPtr) {
    contextDatas.insert_or_assign(sharedPtr.get(), ptr);
  }
}

void Context::removeContextDataInternal(ContextData *ptr) { contextDatas.erase(ptr); }

void Context::collectContextData() {
  for (auto it = contextDatas.begin(); it != contextDatas.end();) {
    if (it->second.expired()) {
      it = contextDatas.erase(it);
    } else {
      it++;
    }
  }
}

void Context::releaseAllContextData() {
  auto contextDatas = std::move(this->contextDatas);
  for (auto &obj : contextDatas) {
    if (!obj.second.expired()) {
      obj.first->releaseContextDataConditional();
    }
  }
}

bool Context::beginFrame() {
  assert(frameState == ContextFrameState::Ok);

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

  collectContextData();
  if (!isHeadless()) {
    const int maxAttempts = 2;
    bool success = false;

    // Try to request the swapchain texture, automatically recreate swapchain on failure
    for (size_t i = 0; !success && i < maxAttempts; i++) {
      success = mainOutput->requestFrame();
      if (!success) {
        SPDLOG_LOGGER_INFO(logger, "Failed to get current swapchain texture, forcing recreate");
        mainOutput->resizeSwapchain(wgpuDevice, wgpuAdapter, mainOutput->currentSize);
      }
    }

    if (!success)
      return false;
  }

  frameState = ContextFrameState::WaitingForEnd;

  return true;
}

void Context::endFrame() {
  assert(frameState == ContextFrameState::WaitingForEnd);

  if (!isHeadless())
    present();

  frameState = ContextFrameState::Ok;
}

void Context::sync() {
#ifdef WEBGPU_NATIVE
  wgpuDevicePoll(wgpuDevice, true, nullptr);
#endif
}

void Context::suspend() {
#if GFX_ANDROID
  // Also release the surface on suspend
  deviceLost();

  if (mainOutput)
    mainOutput->releaseSurface();
#elif GFX_APPLE
  deviceLost();
#endif

  suspended = true;
}

void Context::resume() { suspended = false; }

void Context::submit(WGPUCommandBuffer cmdBuffer) { wgpuQueueSubmit(wgpuQueue, 1, &cmdBuffer); }

void Context::deviceLost() {
  if (state != ContextState::Incomplete) {
    SPDLOG_LOGGER_DEBUG(logger, "Device lost");
    state = ContextState::Incomplete;

    releaseDevice();
  }
}

void Context::tickRequesting() {
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
        backend->setAdapter(wgpuAdapter);

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
        backend->setDevice(wgpuDevice);

        deviceObtained();
      }
    }
  } catch (...) {
    deviceLost();
    throw;
  }
}

void Context::deviceObtained() {
  state = ContextState::Ok;
  SPDLOG_LOGGER_DEBUG(logger, "wgpuDevice obtained");

  auto errorCallback = [](WGPUErrorType type, char const *message, void *userdata) {
    Context &context = *(Context *)userdata;
    std::string msgString(message);
    SPDLOG_LOGGER_ERROR(logger, "{} ({})", message, type);
    if (type == WGPUErrorType_DeviceLost) {
      context.deviceLost();
    }
  };
  wgpuDeviceSetUncapturedErrorCallback(wgpuDevice, errorCallback, this);
  wgpuQueue = wgpuDeviceGetQueue(wgpuDevice);

  if (mainOutput) {
    getOrCreateSurface();
    mainOutput->initSwapchain(wgpuAdapter, wgpuDevice);
  }

  WGPUDeviceLostCallback deviceLostCallback = [](WGPUDeviceLostReason reason, char const *message, void *userdata) {
    SPDLOG_LOGGER_WARN(logger, "Device lost: {} ()", message, magic_enum::enum_name(reason));
  };
  wgpuDeviceSetDeviceLostCallback(wgpuDevice, deviceLostCallback, this);
}

void Context::requestDevice() {
  assert(!wgpuDevice);
  assert(!wgpuQueue);

  // Request adapter first
  if (!wgpuAdapter) {
    requestAdapter();
    tickRequesting(); // This chains into another requestDevice call once the adapter is obtained
    return;
  }

  state = ContextState::Requesting;

  SPDLOG_LOGGER_DEBUG(logger, "Requesting wgpu device");
  deviceRequest = backend->requestDevice();
}

void Context::releaseDevice() {
  releaseAllContextData();

  if (mainOutput) {
    mainOutput->releaseSwapchain();
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
  assert(!wgpuAdapter);

  state = ContextState::Requesting;

  //   WGPURequestAdapterOptions requestAdapter = {};
  //   requestAdapter.powerPreference = WGPUPowerPreference_HighPerformance;
  //   requestAdapter.compatibleSurface = getOrCreateSurface();
  //   requestAdapter.forceFallbackAdapter = false;

  // #ifdef WEBGPU_NATIVE
  //   WGPUAdapterExtras adapterExtras = {};
  //   requestAdapter.nextInChain = &adapterExtras.chain;
  //   adapterExtras.chain.sType = (WGPUSType)WGPUSType_AdapterExtras;

  //   adapterExtras.backend = WGPUBackendType_Null;
  //   if (const char *backendStr = SDL_getenv("GFX_BACKEND")) {
  //     std::string typeStr = std::string("WGPUBackendType_") + backendStr;
  //     auto foundValue = magic_enum::enum_cast<WGPUBackendType>(typeStr);
  //     if (foundValue) {
  //       adapterExtras.backend = foundValue.value();
  //     }
  //   }

  //   if (adapterExtras.backend == WGPUBackendType_Null)
  //     adapterExtras.backend = getDefaultWgpuBackendType();

  //   SPDLOG_LOGGER_INFO(logger, "Using backend {}", magic_enum::enum_name(adapterExtras.backend));
  // #endif

  SPDLOG_LOGGER_DEBUG(logger, "Requesting wgpu adapter");
  adapterRequest = backend->requestAdapter();
}

void Context::releaseAdapter() {
  releaseDevice();
  WGPU_SAFE_RELEASE(wgpuAdapterRelease, wgpuAdapter);
}

void Context::initCommon() {
  SPDLOG_LOGGER_DEBUG(logger, "initCommon");

  assert(!isInitialized());

#ifdef WEBGPU_NATIVE
  wgpuSetLogCallback(
      [](WGPULogLevel level, const char *msg, void *userData) {
        Context &context = *(Context *)userData;
        switch (level) {
        case WGPULogLevel_Error:
          logger->error("{}", msg);
          break;
        case WGPULogLevel_Warn:
          logger->warn("{}", msg);
          break;
        case WGPULogLevel_Info:
          logger->info("{}", msg);
          break;
        case WGPULogLevel_Debug:
          logger->debug("{}", msg);
          break;
        case WGPULogLevel_Trace:
          logger->trace("{}", msg);
          break;
        default:
          break;
        }
      },
      this);

  if (logger->level() <= spdlog::level::debug) {
    wgpuSetLogLevel(WGPULogLevel_Debug);
  } else {
    wgpuSetLogLevel(WGPULogLevel_Info);
  }
#endif

  backend = std::make_shared<VulkanOpenXRBackend>();
  wgpuInstance = backend->createInstance();

  requestDevice();
}

void Context::createInstance() {
  assert(!wgpuInstance);

  // NOTE: Prototype OpenXR binding code
  // the openxr vulkan extensions requires the following:
  // - Specific extensions on instance creation
  // - Specific physical device as returned by OpenXR
  // - Specific extensions on device creation
  // - Need to retrieve the created instance, device handles and queue indices

  // Create instance
  WGPUInstanceDescriptor desc{};
  WGPUInstanceDescriptorVK descVk{};
  if (getDefaultWgpuBackendType() == WGPUBackendType_Vulkan) {
    std::vector<const char *> requiredExtensions = {};

    descVk = {
        .chain{.sType = (WGPUSType)WGPUNativeSTypeEx_InstanceDescriptorVK},
        .requiredExtensions = requiredExtensions.data(),
        .requiredExtensionCount = (uint32_t)requiredExtensions.size(),
    };
    desc.nextInChain = &descVk.chain;
  }

  wgpuInstance = wgpuCreateInstanceEx(&desc);
  if (!wgpuInstance)
    throw std::runtime_error("Failed to create WGPUInstance");

  WGPUInstanceProperties props{};
  WGPUInstancePropertiesVK propsVk{
      .chain = {.sType = WGPUSType(WGPUNativeSTypeEx_InstancePropertiesVK)},
  };
  if (getDefaultWgpuBackendType() == WGPUBackendType_Vulkan) {
    props.nextInChain = &propsVk.chain;
  }
  wgpuInstanceGetPropertiesEx(wgpuInstance, &props);

  vk::DispatchLoaderDynamic loader(PFN_vkGetInstanceProcAddr(propsVk.getInstanceProcAddr));
  vk::Instance instance(VkInstance(propsVk.instance));
  loader.init(instance);

  VkPhysicalDevice physicalDeviceToUse{};
  auto devices = instance.enumeratePhysicalDevices(loader);
  for (auto &device : devices) {
    auto properties = device.getProperties(loader);
    auto name = std::string(properties.deviceName.begin(), properties.deviceName.end());
    SPDLOG_INFO("vulkan physical device: {}", name);
    physicalDeviceToUse = device;
    break;
  }
  assert(physicalDeviceToUse);
}

void Context::present() {
  assert(mainOutput);
  mainOutput->present();
}

} // namespace gfx
