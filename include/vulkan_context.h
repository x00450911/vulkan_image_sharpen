/**
 * @file vulkan_context.h
 * @brief Minimal Vulkan context helper for compute workloads.
 */

#pragma once

#include <vulkan/vulkan.h>

#include <vector>

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    void initialize(bool enableValidationLayers);
    void cleanup();

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkDevice device() const { return m_device; }
    VkQueue computeQueue() const { return m_computeQueue; }
    uint32_t computeQueueFamilyIndex() const { return m_computeQueueFamilyIndex; }
    VkCommandPool commandPool() const { return m_commandPool; }

    VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer commandBuffer) const;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

    const VkPhysicalDeviceProperties& deviceProperties() const { return m_deviceProperties; }

private:
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();

    bool checkValidationLayerSupport() const;
    std::vector<const char*> getRequiredValidationLayers() const;

    void destroyDebugMessenger();

    bool m_validationEnabled = false;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    uint32_t m_computeQueueFamilyIndex = 0;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties m_deviceProperties{};
    VkPhysicalDeviceMemoryProperties m_memoryProperties{};
};
