#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>

struct QueueFamilyIndices {
    uint32_t computeFamily = UINT32_MAX;
    bool isComplete() const { return computeFamily != UINT32_MAX; }
};

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    bool initialize();
    void cleanup();

    VkDevice getDevice() const { return device_; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }
    VkCommandPool getCommandPool() const { return commandPool_; }
    VkQueue getComputeQueue() const { return computeQueue_; }
    VkDescriptorPool getDescriptorPool() const { return descriptorPool_; }

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);

    VkImage createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkDeviceMemory* outMemory = nullptr);
    VkImageView createImageView(VkImage image, VkFormat format);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkShaderModule loadShaderModule(const std::string& filename);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

private:
    bool createInstance();
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    bool createCommandPool();
    bool createDescriptorPool();
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    VkInstance instance_;
    VkDebugUtilsMessengerEXT debugMessenger_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    VkCommandPool commandPool_;
    VkQueue computeQueue_;
    VkDescriptorPool descriptorPool_;

    QueueFamilyIndices queueFamilyIndices_;
};
