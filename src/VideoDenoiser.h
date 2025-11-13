#pragma once

#include "FrameProvider.h"
#include "VulkanContext.h"

#include <cstdint>
#include <string>
#include <vector>

struct FrameImage {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkImageView view{VK_NULL_HANDLE};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
};

struct PushConstants {
    float blendFactor;
    float sigmaColor;
    float sigmaTemporal;
    int frameIndex;
    int historyValid;
    int useSpatialFilter;
    float padding0;
    float padding1;
};

class VideoDenoiser {
public:
    explicit VideoDenoiser(VulkanContext& context);
    ~VideoDenoiser();

    void initialize(uint32_t width, uint32_t height);
    void cleanup();

    void processFrame(const std::vector<uint8_t>& rgbaPixels, std::vector<uint8_t>& outDenoisedPixels,
                      float blendFactor, float sigmaColor, float sigmaTemporal, bool useSpatial);

    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

private:
    void createImages();
    void destroyImages();

    void createDescriptorSetLayout();
    void createPipelineLayout();
    void createComputePipeline();
    void createDescriptorPool();
    void allocateDescriptorSet();
    void createBuffers();
    void createCommandResources();

    void updateDescriptorSet(const FrameImage& current, const FrameImage& history, const FrameImage& output);

    void transitionImage(VkCommandBuffer cmd, FrameImage& image, VkImageLayout newLayout,
                         VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                         VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);

    void recordComputePass(VkCommandBuffer cmd, const PushConstants& constants);

    void uploadToImage(VkCommandBuffer cmd, FrameImage& targetImage, const std::vector<uint8_t>& rgbaPixels);
    void readbackFromImage(VkCommandBuffer cmd, FrameImage& sourceImage);

    void submitAndWait(VkCommandBuffer cmd);

private:
    VulkanContext& context_;
    uint32_t width_{0};
    uint32_t height_{0};

    FrameImage currentImage_{};
    FrameImage historyImage_{};
    FrameImage accumulationImage_{};
    VkDeviceSize imageByteSize_{0};

    VkBuffer uploadBuffer_{VK_NULL_HANDLE};
    VkDeviceMemory uploadMemory_{VK_NULL_HANDLE};
    void* uploadMapped_{nullptr};

    VkBuffer readbackBuffer_{VK_NULL_HANDLE};
    VkDeviceMemory readbackMemory_{VK_NULL_HANDLE};
    void* readbackMapped_{nullptr};

    VkDescriptorSetLayout descriptorSetLayout_{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout_{VK_NULL_HANDLE};
    VkPipeline computePipeline_{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool_{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet_{VK_NULL_HANDLE};

    VkFence computeFence_{VK_NULL_HANDLE};
    VkCommandBuffer commandBuffer_{VK_NULL_HANDLE};

    bool initialized_{false};
    bool historyValid_{false};
    uint32_t processedFrames_{0};
};
