// Copyright 2025
// Pipeline for processing AHardwareBuffer-backed RGB frames on Android Adreno GPUs.

#pragma once

#include <android/hardware_buffer.h>
#include <vulkan/vulkan.h>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "CommandHelpers.h"
#include "VulkanContext.h"

namespace video::processing {

struct FrameDimensions {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct RotationConfig {
    float radians = 0.0f;
    float centerX = 0.5f;
    float centerY = 0.5f;
};

struct EdgeSharpenConfig {
    float amount = 0.45f;
    float clampMin = 0.0f;
    float clampMax = 1.0f;
};

struct OverExposureConfig {
    float lumaThreshold = 0.92f;
    float ratioThreshold = 0.015f;
};

struct MultiFrameDenoiseConfig {
    uint32_t historySize = 4;
    float blendFactor = 0.35f;
    float motionThreshold = 0.02f;
};

struct SuperResolutionConfig {
    float upscaleFactor = 1.0f;
    bool enableModel = true;
};

struct PipelineConfig {
    RotationConfig rotation{};
    EdgeSharpenConfig sharpen{};
    OverExposureConfig exposure{};
    MultiFrameDenoiseConfig denoise{};
    SuperResolutionConfig superResolution{};
};

struct PipelineOutputs {
    // Overexposure histogram (per tile).
    std::vector<float> overExposureTiles;
    // Debug value: average luminance result from the last frame.
    float averageLuminance = 0.0f;
};

struct ImportedImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
        AHardwareBuffer* hardwareBuffer = nullptr;
};

    struct ManagedImage {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

class VulkanAndroidFrameProcessor {
public:
    VulkanAndroidFrameProcessor(std::shared_ptr<VulkanContext> context,
                                const FrameDimensions& dimensions,
                                const PipelineConfig& config);
    ~VulkanAndroidFrameProcessor();

    VulkanAndroidFrameProcessor(const VulkanAndroidFrameProcessor&) = delete;
    VulkanAndroidFrameProcessor& operator=(const VulkanAndroidFrameProcessor&) = delete;

    PipelineOutputs processFrame(AHardwareBuffer* input,
                                 std::span<AHardwareBuffer* const> history,
                                 AHardwareBuffer* output);

    void updateConfig(const PipelineConfig& config);

private:
    struct ShaderKey {
        std::string name;
        std::string path;
    };

    struct DescriptorSetAllocation {
        VkDescriptorSet rotationSharpen = VK_NULL_HANDLE;
        VkDescriptorSet exposure = VK_NULL_HANDLE;
        VkDescriptorSet denoise = VK_NULL_HANDLE;
        VkDescriptorSet superResolution = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    };

    struct FrameResources {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        DescriptorSetAllocation descriptors{};
        ImportedImage inputImage{};
        ImportedImage outputImage{};
        ManagedImage intermediateImage{};
        ManagedImage denoiseImage{};
        VkBuffer exposureBuffer = VK_NULL_HANDLE;
        VkDeviceMemory exposureMemory = VK_NULL_HANDLE;
        VkBuffer paramsBuffer = VK_NULL_HANDLE;
        VkDeviceMemory paramsMemory = VK_NULL_HANDLE;
    };

    struct PipelineHandles {
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorUpdateTemplate descriptorTemplate = VK_NULL_HANDLE;
        ShaderKey shader;
    };

    struct ComputeStages {
        PipelineHandles rotationSharpen{};
        PipelineHandles exposure{};
        PipelineHandles denoise{};
        PipelineHandles superResolution{};
    };

    struct ParamsBlock {
        alignas(16) std::array<float, 4> rotationMatrix;
        alignas(16) std::array<float, 4> rotationCenter;
        alignas(16) std::array<float, 4> sharpenParams;
        alignas(16) std::array<float, 4> exposureParams;
        alignas(16) std::array<float, 4> denoiseParams;
        alignas(16) std::array<float, 4> srParams;
    };

private:
    void createDescriptorSetLayouts();
    void createPipelines();
    void destroyPipelines();
    void rebuildPipelinesIfNeeded();

    FrameResources allocateFrameResources();
    void destroyFrameResources(FrameResources& resources);

    void updateParameters(FrameResources& frame, const PipelineConfig& config);
    void recordCommandBuffer(FrameResources& frame,
                             const ImportedImage& input,
                             const ImportedImage& output,
                             std::span<const ImportedImage> historyImages);

    ImportedImage importHardwareBuffer(AHardwareBuffer* buffer);
    void releaseImportedImage(ImportedImage& image);
    VkBuffer createStagingBuffer(VkDeviceSize size, VkDeviceMemory* outMemory);
    ManagedImage createManagedImage(VkFormat format, VkImageUsageFlags usage);
    void destroyManagedImage(ManagedImage& image);

    void updateDescriptorSet(FrameResources& frame,
                             const ImportedImage& input,
                             const ImportedImage& output,
                             std::span<const ImportedImage> historyImages);

    void ensureHistoryCapacity(std::size_t size);

private:
    std::shared_ptr<VulkanContext> context_;
    FrameDimensions dimensions_;
    PipelineConfig config_;
    ComputeStages stages_{};
    std::vector<ImportedImage> historyCache_;
    FrameResources frame_;
    bool pipelinesDirty_ = false;
};

}  // namespace video::processing
