#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include "vulkan_context.h"
#include "frame_manager.h"

class Denoiser {
public:
    Denoiser(VulkanContext* context, FrameManager* frameManager);
    ~Denoiser();

    bool initialize();
    void cleanup();

    void denoiseFrame(uint32_t currentFrameIndex, uint32_t historyFrameCount, float alpha);

private:
    bool createDescriptorSetLayout();
    bool createPipeline();
    bool createDescriptorSets();
    void updateDescriptorSets(uint32_t currentFrameIndex, uint32_t historyFrameCount);

    VulkanContext* context_;
    FrameManager* frameManager_;

    VkDescriptorSetLayout descriptorSetLayout_;
    VkPipelineLayout pipelineLayout_;
    VkPipeline computePipeline_;
    std::vector<VkDescriptorSet> descriptorSets_;

    VkShaderModule denoiseShader_;
    VkShaderModule accumulateShader_;
};
