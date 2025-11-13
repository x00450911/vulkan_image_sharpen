#include "denoiser.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

Denoiser::Denoiser(VulkanContext* context, FrameManager* frameManager)
    : context_(context)
    , frameManager_(frameManager)
    , descriptorSetLayout_(VK_NULL_HANDLE)
    , pipelineLayout_(VK_NULL_HANDLE)
    , computePipeline_(VK_NULL_HANDLE)
    , denoiseShader_(VK_NULL_HANDLE)
    , accumulateShader_(VK_NULL_HANDLE)
{
}

Denoiser::~Denoiser() {
    cleanup();
}

bool Denoiser::initialize() {
    if (!createDescriptorSetLayout()) {
        return false;
    }
    if (!createPipeline()) {
        return false;
    }
    if (!createDescriptorSets()) {
        return false;
    }
    return true;
}

void Denoiser::cleanup() {
    if (context_ && context_->getDevice() != VK_NULL_HANDLE) {
        if (computePipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(context_->getDevice(), computePipeline_, nullptr);
            computePipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context_->getDevice(), pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context_->getDevice(), descriptorSetLayout_, nullptr);
            descriptorSetLayout_ = VK_NULL_HANDLE;
        }
        if (denoiseShader_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context_->getDevice(), denoiseShader_, nullptr);
            denoiseShader_ = VK_NULL_HANDLE;
        }
        if (accumulateShader_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(context_->getDevice(), accumulateShader_, nullptr);
            accumulateShader_ = VK_NULL_HANDLE;
        }
    }
}

bool Denoiser::createDescriptorSetLayout() {
    // Descriptor set layout for denoising shader
    // Binding 0: current frame (read-only)
    // Binding 1: previous denoised frame (read-only)
    // Binding 2: output frame (write-only)
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {
            0,                                          // binding
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           // descriptorType
            1,                                          // descriptorCount
            VK_SHADER_STAGE_COMPUTE_BIT,                // stageFlags
            nullptr                                      // pImmutableSamplers
        },
        {
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr
        },
        {
            2,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr
        }
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(context_->getDevice(), &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout!" << std::endl;
        return false;
    }

    return true;
}

bool Denoiser::createPipeline() {
    // Load shader modules
    try {
        denoiseShader_ = context_->loadShaderModule("shaders/denoise.spv");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load denoise shader: " << e.what() << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = denoiseShader_;
    shaderStageInfo.pName = "main";

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float) * 4 + sizeof(int) * 2; // DenoiseParams size

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(context_->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout!" << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout_;

    if (vkCreateComputePipelines(context_->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline_) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline!" << std::endl;
        return false;
    }

    return true;
}

bool Denoiser::createDescriptorSets() {
    // Allocate descriptor sets (we'll create one per frame for simplicity)
    uint32_t frameCount = frameManager_->getFrameCount();
    descriptorSets_.resize(frameCount);

    std::vector<VkDescriptorSetLayout> layouts(frameCount, descriptorSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = context_->getDescriptorPool();
    allocInfo.descriptorSetCount = frameCount;
    allocInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(context_->getDevice(), &allocInfo, descriptorSets_.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets!" << std::endl;
        return false;
    }

    return true;
}

void Denoiser::updateDescriptorSets(uint32_t currentFrameIndex, uint32_t historyFrameCount) {
    if (currentFrameIndex >= descriptorSets_.size()) {
        return;
    }

    VkDescriptorSet descriptorSet = descriptorSets_[currentFrameIndex];
    
    // Get frame buffers
    const FrameBuffer& currentFrame = frameManager_->getFrame(currentFrameIndex);
    uint32_t previousIndex = (currentFrameIndex > 0) ? currentFrameIndex - 1 : currentFrameIndex;
    const FrameBuffer& previousFrame = frameManager_->getFrame(previousIndex);
    const FrameBuffer& outputFrame = frameManager_->getFrame(currentFrameIndex);

    std::vector<VkWriteDescriptorSet> descriptorWrites;

    // Current frame
    VkDescriptorImageInfo currentImageInfo{};
    currentImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    currentImageInfo.imageView = currentFrame.view;

    VkWriteDescriptorSet currentWrite{};
    currentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    currentWrite.dstSet = descriptorSet;
    currentWrite.dstBinding = 0;
    currentWrite.dstArrayElement = 0;
    currentWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    currentWrite.descriptorCount = 1;
    currentWrite.pImageInfo = &currentImageInfo;
    descriptorWrites.push_back(currentWrite);

    // Previous denoised frame
    VkDescriptorImageInfo previousImageInfo{};
    previousImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    previousImageInfo.imageView = previousFrame.view;

    VkWriteDescriptorSet previousWrite{};
    previousWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    previousWrite.dstSet = descriptorSet;
    previousWrite.dstBinding = 1;
    previousWrite.dstArrayElement = 0;
    previousWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    previousWrite.descriptorCount = 1;
    previousWrite.pImageInfo = &previousImageInfo;
    descriptorWrites.push_back(previousWrite);

    // Output frame
    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputImageInfo.imageView = outputFrame.view;

    VkWriteDescriptorSet outputWrite{};
    outputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    outputWrite.dstSet = descriptorSet;
    outputWrite.dstBinding = 2;
    outputWrite.dstArrayElement = 0;
    outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputWrite.descriptorCount = 1;
    outputWrite.pImageInfo = &outputImageInfo;
    descriptorWrites.push_back(outputWrite);

    vkUpdateDescriptorSets(context_->getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Denoiser::denoiseFrame(uint32_t currentFrameIndex, uint32_t historyFrameCount, float alpha) {
    updateDescriptorSets(currentFrameIndex, historyFrameCount);

    VkCommandBuffer commandBuffer = context_->beginSingleTimeCommands();

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout_,
        0,
        1,
        &descriptorSets_[currentFrameIndex],
        0,
        nullptr
    );

    // Push constants: alpha, noiseThreshold, imageSize
    struct DenoiseParams {
        float alpha;
        float noiseThreshold;
        int32_t imageSize[2];
    } params;

    params.alpha = alpha;
    params.noiseThreshold = 0.1f; // Adjust based on noise characteristics
    params.imageSize[0] = static_cast<int32_t>(frameManager_->getWidth());
    params.imageSize[1] = static_cast<int32_t>(frameManager_->getHeight());

    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(DenoiseParams),
        &params
    );

    // Dispatch compute shader
    uint32_t groupCountX = (frameManager_->getWidth() + 7) / 8;
    uint32_t groupCountY = (frameManager_->getHeight() + 7) / 8;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    context_->endSingleTimeCommands(commandBuffer);
}
