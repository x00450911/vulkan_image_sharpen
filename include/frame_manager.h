#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include "vulkan_context.h"

struct FrameBuffer {
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView view;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    uint32_t width;
    uint32_t height;
};

class FrameManager {
public:
    FrameManager(VulkanContext* context);
    ~FrameManager();

    bool initialize(uint32_t width, uint32_t height, uint32_t frameCount);
    void cleanup();

    FrameBuffer& getFrame(uint32_t index) { return frames_[index]; }
    const FrameBuffer& getFrame(uint32_t index) const { return frames_[index]; }
    uint32_t getFrameCount() const { return static_cast<uint32_t>(frames_.size()); }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

    void uploadFrame(uint32_t index, const void* data, size_t size);
    void downloadFrame(uint32_t index, void* data, size_t size);

private:
    VulkanContext* context_;
    std::vector<FrameBuffer> frames_;
    uint32_t width_;
    uint32_t height_;
};
