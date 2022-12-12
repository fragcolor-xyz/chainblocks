#pragma once



#include "wgpu_handle.hpp"


#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "linalg.hpp"
#include <optional>

namespace gfx {
  //[t] moved these here because circuar dependencies otherwise
    
  struct IContextCurrentFramePayload{
    WGPUTextureView wgpuTextureView; 
    std::optional<bool> useMatrix = false;
    std::optional<linalg::aliases::float4x4> eyeViewMatrix;
    std::optional<linalg::aliases::float4x4> eyeProjectionMatrix;
  };

  struct IContextMainOutput {
    virtual ~IContextMainOutput() = default;

    // Current output image size
    virtual const int2 &getSize() const = 0;
    // Return the texture format of the images
    virtual WGPUTextureFormat getFormat() const = 0;
    // Requests a new swapchain image to render to. Is an array because can contain e.g. multiple xr eyes
    virtual std::vector<WGPUTextureView> requestFrame() = 0;
    // Returns the currently request frame's texture view
    virtual std::vector<IContextCurrentFramePayload> getCurrentFrame() const = 0;
    // Return the previously requested swapchain image to the chain and allow it to be displayed
    virtual void present() = 0;
  };
  
  struct WGPUVulkanShared {
    //[t] So this vkLoader thingy + vkDevice is a loader for any and all vulkan native functions we might need to call. 
    //[t] The way you call a vulkan function is by having a reference to wgpuVulkanShared first.
    //[t] Then if for example you need to call a vkfunction like vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr)
    //[t] if it's in a specific class like vkDevice, you use wgpuVulkanShared->vkDevice.enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr, wgpuVulkanShared->vkLoader)
    //[t] if it's global, you use vk::enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr, wgpuVulkanShared->vkLoader)
    //[t] note you have to send the wgpuVulkanShared->vkLoader as last argument.
    vk::DispatchLoaderDynamic vkLoader;
    vk::PhysicalDevice vkPhysicalDevice;//[t] VkPhysicalDevice
    vk::Instance vkInstance;//[t] VkInstance
    vk::Device vkDevice;//[t] VkDevice
    vk::Queue vkQueue;//[t] VkQueue, draw queue. The present queue is wgpuPresent afaiks

    uint32_t queueIndex;
    uint32_t queueFamilyIndex;

    WGPUInstance wgpuInstance{};
    WGPUAdapter wgpuAdapter{};
    WGPUDevice wgpuDevice{};
    WGPUQueue wgpuQueue{};
  };

} // namespace gfx
