// Minimal Vulkan context tailored for Android/Adreno compute workloads.

#pragma once

#include <android/native_window.h>
#include <android/hardware_buffer.h>
#include <memory>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>

#include "CommandHelpers.h"

namespace video::processing {

class VulkanContext : public std::enable_shared_from_this<VulkanContext> {
public:
    struct CreateInfo {
        bool enableValidation = false;
        ANativeWindow* window = nullptr;
    };

    static std::shared_ptr<VulkanContext> create(const CreateInfo& info);

    ~VulkanContext();

    VkInstance instance() const { return instance_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkQueue computeQueue() const { return computeQueue_; }
    uint32_t computeQueueFamily() const { return computeQueueFamilyIndex_; }
    VkCommandPool commandPool() const { return commandPool_; }
    VkCommandPool* commandPoolPtr() { return &commandPool_; }
    VkSampler linearSampler() const { return linearSampler_; }

    const VkPhysicalDeviceMemoryProperties& memoryProperties() const { return memoryProps_; }
    VkPhysicalDeviceFeatures features() const { return features_; }
    VkPhysicalDeviceVulkan12Features features12() const { return features12_; }

    const VkPhysicalDeviceProperties& deviceProperties() const { return properties_; }

    AHardwareBuffer_Desc& bufferDesc() { return bufferDesc_; }

    VkDescriptorPool sharedDescriptorPool() const { return descriptorPool_; }

    VkPipelineCache pipelineCache() const { return pipelineCache_; }

private:
    explicit VulkanContext(const CreateInfo& info);

    void createInstance(const CreateInfo& info);
    void setupDebugMessenger();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();
    void createCommandInfrastructure();
    void createSamplerResources();
    void createDescriptorPool();
    void createPipelineCache();

private:
    bool enableValidation_ = false;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceFeatures features_{};
    VkPhysicalDeviceVulkan12Features features12_{};
    VkPhysicalDeviceMemoryProperties memoryProps_{};

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex_ = VK_QUEUE_FAMILY_IGNORED;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

    ANativeWindow* window_ = nullptr;
    AHardwareBuffer_Desc bufferDesc_{};
};

}  // namespace video::processing
