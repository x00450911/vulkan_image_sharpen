/**
 * @file multi_frame_denoiser.h
 * @brief Vulkan-based multi-frame video denoising interface.
 */

#pragma once

#include "vulkan_context.h"

#include <filesystem>
#include <vector>

struct DenoiserConfig {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t historyLength = 4;
    float frameBlendFactor = 0.1f;
    float luminanceSigma = 0.2f;
    std::filesystem::path shaderSpirvPath;
};

class MultiFrameDenoiser {
public:
    static constexpr uint32_t kMaxHistoryImages = 4;

    MultiFrameDenoiser() = default;
    ~MultiFrameDenoiser();

    void initialize(VulkanContext& context, const DenoiserConfig& config);
    void cleanup();

    void processFrame(const float* rgbaPixels, size_t floatCount, std::vector<float>& outputPixels);

    uint32_t width() const { return m_config.width; }
    uint32_t height() const { return m_config.height; }

private:
    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    struct BufferResource {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
    };

    struct alignas(16) DenoiseUniforms {
        uint32_t frameIndex = 0;
        uint32_t historyLength = 1;
        float frameBlendFactor = 0.1f;
        float luminanceSigma = 0.2f;
    };

    void createDescriptorSetLayout();
    void createPipeline();
    void createDescriptorPoolAndSet();
    void createHistoryImages();
    void createAuxiliaryResources();
    void updateUniforms();
    void updateDescriptors(uint32_t writeIndex);

    void uploadFrameToHistory(const float* rgbaPixels, size_t floatCount, uint32_t writeIndex);
    void dispatchDenoise(uint32_t writeIndex);
    void downloadOutput(std::vector<float>& outputPixels);

    ImageResource createStorageImage(VkFormat format, VkImageUsageFlags extraUsage, const char* debugName);
    void destroyImage(ImageResource& image);

    BufferResource createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, const char* debugName);
    void destroyBuffer(BufferResource& buffer);

    void transitionImageLayout(VkCommandBuffer cmd,
                               VkImage image,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout,
                               VkAccessFlags srcAccessMask,
                               VkAccessFlags dstAccessMask,
                               VkPipelineStageFlags srcStage,
                               VkPipelineStageFlags dstStage);

    VulkanContext* m_context = nullptr;
    DenoiserConfig m_config{};

    std::vector<ImageResource> m_historyImages;
    ImageResource m_accumulationImage{};
    ImageResource m_outputImage{};

    BufferResource m_uploadBuffer{};
    BufferResource m_downloadBuffer{};
    BufferResource m_uniformBuffer{};

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    VkShaderModule m_shaderModule = VK_NULL_HANDLE;

    DenoiseUniforms m_uniforms{};
    uint32_t m_frameCounter = 0;
    uint32_t m_historyWriteIndex = 0;
};
