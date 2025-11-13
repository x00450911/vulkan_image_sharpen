#pragma once

#include <vulkan/vulkan.h>

#include <optional>
#include <vector>

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    void initialize(bool enableValidationLayers);
    void cleanup();

    VkInstance getInstance() const { return instance_; }
    VkDevice getDevice() const { return device_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    VkQueue getComputeQueue() const { return computeQueue_; }
    uint32_t getComputeQueueFamilyIndex() const { return computeQueueFamilyIndex_; }
    VkCommandPool getCommandPool() const { return commandPool_; }

    VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) const;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

private:
    void createInstance(bool enableValidationLayers);
    void setupDebugMessenger(bool enableValidationLayers);
    void pickPhysicalDevice();
    void createLogicalDevice(bool enableValidationLayers);
    void createCommandPool();

    bool checkValidationLayerSupport(const std::vector<const char*>& layers) const;
    bool isDeviceSuitable(VkPhysicalDevice device) const;
    std::optional<uint32_t> findComputeQueueFamily(VkPhysicalDevice device) const;

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

private:
    VkInstance instance_{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugMessenger_{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue computeQueue_{VK_NULL_HANDLE};
    uint32_t computeQueueFamilyIndex_{0};
    VkCommandPool commandPool_{VK_NULL_HANDLE};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    bool validationLayersEnabled_{false};
};
