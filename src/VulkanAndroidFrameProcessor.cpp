// Implementation of Vulkan pipeline for processing Android AHardwareBuffer frames.

#include "VulkanAndroidFrameProcessor.h"

#include <android/log.h>
#include <sync/sync.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <numeric>

#include "ShaderLoader.h"

#define LOG_TAG "VulkanFrameProcessor"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace video::processing;

namespace {

constexpr uint32_t kRotationSharpenBindingInput = 0;
constexpr uint32_t kRotationSharpenBindingOutput = 1;
constexpr uint32_t kRotationSharpenBindingParams = 2;

constexpr uint32_t kExposureBindingInput = 0;
constexpr uint32_t kExposureBindingOutput = 1;
constexpr uint32_t kExposureBindingParams = 2;

constexpr uint32_t kDenoiseBindingInput = 0;
constexpr uint32_t kDenoiseBindingHistory = 1;
constexpr uint32_t kDenoiseBindingOutput = 2;
constexpr uint32_t kDenoiseBindingParams = 3;

constexpr uint32_t kSuperResBindingInput = 0;
constexpr uint32_t kSuperResBindingOutput = 1;
constexpr uint32_t kSuperResBindingParams = 2;

constexpr VkFormat kInputFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkImageUsageFlags kInputUsage =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
constexpr VkImageUsageFlags kOutputUsage =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

constexpr VkPipelineStageFlags kWaitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

constexpr uint32_t kLocalSizeX = 16;
constexpr uint32_t kLocalSizeY = 16;

constexpr uint32_t kExposureGridDimension = 8;
constexpr uint32_t kExposureBinCount = kExposureGridDimension * kExposureGridDimension;
constexpr uint32_t kMaxHistoryImages = 4;

VkFormat chooseFormat(const AHardwareBuffer_Desc& desc) {
    switch (desc.format) {
    case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    default:
        ALOGE("Unsupported AHardwareBuffer format %u", desc.format);
        return VK_FORMAT_UNDEFINED;
    }
}

VkExtent2D chooseExtent(const AHardwareBuffer_Desc& desc) {
    return VkExtent2D{static_cast<uint32_t>(desc.width), static_cast<uint32_t>(desc.height)};
}

std::array<float, 4> createRotationMatrix(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return {c, -s, s, c};
}

ParamsBlock makeParamsBlock(const PipelineConfig& config) {
    ParamsBlock params{};
    params.rotationMatrix = createRotationMatrix(config.rotation.radians);
    params.rotationCenter = {config.rotation.centerX, config.rotation.centerY, 0.f, 0.f};
    params.sharpenParams = {config.sharpen.amount, config.sharpen.clampMin,
                            config.sharpen.clampMax, 0.f};
    params.exposureParams = {config.exposure.lumaThreshold, config.exposure.ratioThreshold,
                             static_cast<float>(kExposureGridDimension),
                             static_cast<float>(kExposureBinCount)};
    const uint32_t historyCount =
        std::min<uint32_t>(config.denoise.historySize, kMaxHistoryImages);
    params.denoiseParams = {config.denoise.blendFactor, config.denoise.motionThreshold,
                            static_cast<float>(historyCount), 0.f};
    params.srParams = {config.superResolution.upscaleFactor,
                       config.superResolution.enableModel ? 1.f : 0.f, 0.f, 0.f};
    return params;
}

VkDeviceSize alignUniformBufferSize(VkPhysicalDevice physicalDevice, VkDeviceSize originalSize) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    const VkDeviceSize alignment = props.limits.minUniformBufferOffsetAlignment;
    if (!alignment) {
        return originalSize;
    }
    return (originalSize + alignment - 1) & ~(alignment - 1);
}

VkBufferCreateInfo makeBufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    return info;
}

VkMemoryAllocateInfo makeMemoryAllocateInfo(const VkMemoryRequirements& requirements,
                                            uint32_t memoryTypeIndex) {
    VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    info.allocationSize = requirements.size;
    info.memoryTypeIndex = memoryTypeIndex;
    return info;
}

bool hasFlag(VkMemoryPropertyFlags flags, VkMemoryPropertyFlags mask) {
    return (flags & mask) == mask;
}

uint32_t findMemoryType(const VulkanContext& context, uint32_t typeBits,
                        VkMemoryPropertyFlags required) {
    const VkPhysicalDeviceMemoryProperties& memory = context.memoryProperties();
    for (uint32_t idx = 0; idx < memory.memoryTypeCount; ++idx) {
        if ((typeBits & (1u << idx)) && hasFlag(memory.memoryTypes[idx].propertyFlags, required)) {
            return idx;
        }
    }
    throw std::runtime_error("Unable to find required memory type");
}

std::vector<VkDescriptorSetLayoutBinding> rotationSharpenBindings() {
    return {
        {kRotationSharpenBindingInput, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {kRotationSharpenBindingOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {kRotationSharpenBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
}

std::vector<VkDescriptorSetLayoutBinding> exposureBindings() {
    return {
        {kExposureBindingInput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {kExposureBindingOutput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {kExposureBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
}

std::vector<VkDescriptorSetLayoutBinding> denoiseBindings() {
    return {
        {kDenoiseBindingInput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
        {kDenoiseBindingHistory, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
        {kDenoiseBindingOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
        {kDenoiseBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
    };
}

std::vector<VkDescriptorSetLayoutBinding> superResolutionBindings() {
    return {
        {kSuperResBindingInput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
        {kSuperResBindingOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
        {kSuperResBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
}

VkDescriptorSetLayout createDescriptorSetLayout(const VulkanContext& context,
                                                std::span<const VkDescriptorSetLayoutBinding> bindings) {
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr, &layout));
    return layout;
}

VkPipelineLayout createPipelineLayout(const VulkanContext& context, VkDescriptorSetLayout layout) {
    VkPipelineLayoutCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineInfo.setLayoutCount = 1;
    pipelineInfo.pSetLayouts = &layout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(context.device(), &pipelineInfo, nullptr, &pipelineLayout));
    return pipelineLayout;
}

PipelineHandles makeComputePipeline(const VulkanContext& context, VkDescriptorSetLayout layout,
                                    const char* shaderPath, const char* shaderName) {
    PipelineHandles handles{};
    handles.shader = ShaderKey{shaderName, shaderPath};
    handles.descriptorSetLayout = layout;
    handles.layout = createPipelineLayout(context, layout);

    const auto shaderModule = shader::loadShaderModule(context.device(), shaderPath);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule.get();
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = handles.layout;

    VK_CHECK(vkCreateComputePipelines(context.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                      &handles.pipeline));

    return handles;
}

VkSampler createLinearSampler(const VulkanContext& context) {
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(context.device(), &samplerInfo, nullptr, &sampler));
    return sampler;
}

}  // namespace

namespace video::processing {

VulkanAndroidFrameProcessor::VulkanAndroidFrameProcessor(std::shared_ptr<VulkanContext> context,
                                                         const FrameDimensions& dimensions,
                                                         const PipelineConfig& config)
    : context_(std::move(context)), dimensions_(dimensions), config_(config) {
    createDescriptorSetLayouts();
    createPipelines();
    frame_ = allocateFrameResources();
}

VulkanAndroidFrameProcessor::~VulkanAndroidFrameProcessor() {
    destroyFrameResources(frame_);
    destroyPipelines();
}

void VulkanAndroidFrameProcessor::updateConfig(const PipelineConfig& config) {
    config_ = config;
    updateParameters(frame_, config_);
}

PipelineOutputs VulkanAndroidFrameProcessor::processFrame(
    AHardwareBuffer* input, std::span<AHardwareBuffer* const> history, AHardwareBuffer* output) {
    rebuildPipelinesIfNeeded();

    frame_.inputImage = importHardwareBuffer(input);
    frame_.outputImage = importHardwareBuffer(output);

    ensureHistoryCapacity(std::max<std::size_t>(history.size(), kMaxHistoryImages));
    for (size_t i = 0; i < history.size(); ++i) {
        releaseImportedImage(historyCache_[i]);
        historyCache_[i] = importHardwareBuffer(history[i]);
    }
    for (size_t i = history.size(); i < historyCache_.size(); ++i) {
        if (historyCache_[i].image != VK_NULL_HANDLE) {
            releaseImportedImage(historyCache_[i]);
        }
    }

    std::span<const ImportedImage> historySpan(historyCache_.data(), history.size());

    updateParameters(frame_, config_);
    updateDescriptorSet(frame_, frame_.inputImage, frame_.outputImage, historySpan);
    recordCommandBuffer(frame_, frame_.inputImage, frame_.outputImage, historySpan);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame_.commandBuffer;

    VK_CHECK(vkResetFences(context_->device(), 1, &frame_.fence));
    VK_CHECK(vkQueueSubmit(context_->computeQueue(), 1, &submit, frame_.fence));

    VK_CHECK(vkWaitForFences(context_->device(), 1, &frame_.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetCommandBuffer(frame_.commandBuffer, 0));

    PipelineOutputs outputs{};
    outputs.overExposureTiles.resize(kExposureBinCount);
    std::vector<uint32_t> exposureCounts(kExposureBinCount);
    void* mapped = nullptr;
    const VkDeviceSize exposureBytes =
        static_cast<VkDeviceSize>(kExposureBinCount) * sizeof(uint32_t);

    VK_CHECK(
        vkMapMemory(context_->device(), frame_.exposureMemory, 0, exposureBytes, 0, &mapped));
    std::memcpy(exposureCounts.data(), mapped, exposureBytes);
    vkUnmapMemory(context_->device(), frame_.exposureMemory);

    const float normalization =
        std::max(1.0f, static_cast<float>(dimensions_.width) * static_cast<float>(dimensions_.height));
    float exposureSum = 0.0f;
    for (uint32_t i = 0; i < kExposureBinCount; ++i) {
        const float ratio = static_cast<float>(exposureCounts[i]) / normalization;
        outputs.overExposureTiles[i] = ratio;
        exposureSum += ratio;
    }
    outputs.averageLuminance = exposureSum / static_cast<float>(kExposureBinCount);

    releaseImportedImage(frame_.inputImage);
    releaseImportedImage(frame_.outputImage);
    for (size_t i = 0; i < history.size(); ++i) {
        releaseImportedImage(historyCache_[i]);
    }

    return outputs;
}

void VulkanAndroidFrameProcessor::createDescriptorSetLayouts() {
    stages_.rotationSharpen.descriptorSetLayout =
        createDescriptorSetLayout(*context_, rotationSharpenBindings());
    stages_.exposure.descriptorSetLayout =
        createDescriptorSetLayout(*context_, exposureBindings());
    stages_.denoise.descriptorSetLayout =
        createDescriptorSetLayout(*context_, denoiseBindings());
    stages_.superResolution.descriptorSetLayout =
        createDescriptorSetLayout(*context_, superResolutionBindings());
}

void VulkanAndroidFrameProcessor::createPipelines() {
    stages_.rotationSharpen =
        makeComputePipeline(*context_, stages_.rotationSharpen.descriptorSetLayout,
                            "shaders/rotation_sharpen.comp.spv", "rotation_sharpen");
    stages_.exposure =
        makeComputePipeline(*context_, stages_.exposure.descriptorSetLayout,
                            "shaders/exposure_reduce.comp.spv", "exposure_reduce");
    stages_.denoise =
        makeComputePipeline(*context_, stages_.denoise.descriptorSetLayout,
                            "shaders/denoise.comp.spv", "multi_frame_denoise");
    stages_.superResolution =
        makeComputePipeline(*context_, stages_.superResolution.descriptorSetLayout,
                            "shaders/super_resolution.comp.spv", "super_resolution");
}

void VulkanAndroidFrameProcessor::destroyPipelines() {
    auto destroyHandles = [this](PipelineHandles& handles) {
        if (handles.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(context_->device(), handles.pipeline, nullptr);
            handles.pipeline = VK_NULL_HANDLE;
        }
        if (handles.layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(context_->device(), handles.layout, nullptr);
            handles.layout = VK_NULL_HANDLE;
        }
        if (handles.descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(context_->device(), handles.descriptorSetLayout, nullptr);
            handles.descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (handles.descriptorTemplate != VK_NULL_HANDLE) {
            vkDestroyDescriptorUpdateTemplate(context_->device(), handles.descriptorTemplate,
                                              nullptr);
            handles.descriptorTemplate = VK_NULL_HANDLE;
        }
    };

    destroyHandles(stages_.rotationSharpen);
    destroyHandles(stages_.exposure);
    destroyHandles(stages_.denoise);
    destroyHandles(stages_.superResolution);
}

void VulkanAndroidFrameProcessor::rebuildPipelinesIfNeeded() {
    if (!pipelinesDirty_) {
        return;
    }
    destroyPipelines();
    createDescriptorSetLayouts();
    createPipelines();
    pipelinesDirty_ = false;
}

VulkanAndroidFrameProcessor::FrameResources VulkanAndroidFrameProcessor::allocateFrameResources() {
    FrameResources resources{};

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = context_->commandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context_->device(), &allocInfo, &resources.commandBuffer));

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(context_->device(), &fenceInfo, nullptr, &resources.fence));

    // Exposure buffer
    const VkDeviceSize exposureSize =
        static_cast<VkDeviceSize>(kExposureBinCount) * sizeof(uint32_t);
    resources.exposureBuffer =
        createStagingBuffer(exposureSize, &resources.exposureMemory);

    const VkDeviceSize paramsSize =
        alignUniformBufferSize(context_->physicalDevice(), sizeof(ParamsBlock));
    resources.paramsBuffer =
        createStagingBuffer(paramsSize, &resources.paramsMemory);

    // Descriptor pool
    std::array<VkDescriptorPoolSize, 4> poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
    };

    VkDescriptorPoolCreateInfo poolInfoDesc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfoDesc.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfoDesc.pPoolSizes = poolSizes.data();
    poolInfoDesc.maxSets = 8;
    VK_CHECK(vkCreateDescriptorPool(context_->device(), &poolInfoDesc, nullptr,
                                    &resources.descriptors.descriptorPool));

    std::array<VkDescriptorSetLayout, 4> layouts = {
        stages_.rotationSharpen.descriptorSetLayout,
        stages_.exposure.descriptorSetLayout,
        stages_.denoise.descriptorSetLayout,
        stages_.superResolution.descriptorSetLayout,
    };

    std::array<VkDescriptorSet, 4> descriptorSets{};
    VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setAlloc.descriptorPool = resources.descriptors.descriptorPool;
    setAlloc.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    setAlloc.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_->device(), &setAlloc, descriptorSets.data()));

    resources.descriptors.rotationSharpen = descriptorSets[0];
    resources.descriptors.exposure = descriptorSets[1];
    resources.descriptors.denoise = descriptorSets[2];
    resources.descriptors.superResolution = descriptorSets[3];

    resources.intermediateImage = createManagedImage(kInputFormat, kOutputUsage | kInputUsage);
    resources.denoiseImage = createManagedImage(kInputFormat, kOutputUsage | kInputUsage);

    return resources;
}

void VulkanAndroidFrameProcessor::destroyFrameResources(FrameResources& resources) {
    if (resources.descriptors.descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context_->device(), resources.descriptors.descriptorPool, nullptr);
        resources.descriptors.descriptorPool = VK_NULL_HANDLE;
    }
    if (resources.commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(context_->device(), context_->commandPool(), 1,
                             &resources.commandBuffer);
        resources.commandBuffer = VK_NULL_HANDLE;
    }
    if (resources.fence != VK_NULL_HANDLE) {
        vkDestroyFence(context_->device(), resources.fence, nullptr);
        resources.fence = VK_NULL_HANDLE;
    }
    if (resources.exposureBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_->device(), resources.exposureBuffer, nullptr);
        resources.exposureBuffer = VK_NULL_HANDLE;
    }
    if (resources.exposureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context_->device(), resources.exposureMemory, nullptr);
        resources.exposureMemory = VK_NULL_HANDLE;
    }
    if (resources.paramsBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(context_->device(), resources.paramsBuffer, nullptr);
        resources.paramsBuffer = VK_NULL_HANDLE;
    }
    if (resources.paramsMemory != VK_NULL_HANDLE) {
        vkFreeMemory(context_->device(), resources.paramsMemory, nullptr);
        resources.paramsMemory = VK_NULL_HANDLE;
    }
    destroyManagedImage(resources.intermediateImage);
    destroyManagedImage(resources.denoiseImage);
}

void VulkanAndroidFrameProcessor::updateParameters(FrameResources& frame,
                                                   const PipelineConfig& config) {
    const ParamsBlock params = makeParamsBlock(config);
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(context_->device(), frame.paramsMemory, 0, sizeof(ParamsBlock), 0,
                         &mapped));
    std::memcpy(mapped, &params, sizeof(ParamsBlock));
    vkUnmapMemory(context_->device(), frame.paramsMemory);
}

void VulkanAndroidFrameProcessor::recordCommandBuffer(FrameResources& frame,
                                                      const ImportedImage& input,
                                                      const ImportedImage& output,
                                                      std::span<const ImportedImage> historyImages) {
    VK_CHECK(vkResetCommandBuffer(frame.commandBuffer, 0));

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo));

    const VkImageSubresourceRange colorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    std::vector<VkImageMemoryBarrier> initialBarriers;
    initialBarriers.reserve(4 + historyImages.size());

    auto makeBarrier = [&](VkImage image, VkImageLayout newLayout,
                           VkAccessFlags dstAccess) -> VkImageMemoryBarrier {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = colorRange;
        return barrier;
    };

    initialBarriers.push_back(makeBarrier(input.image, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_ACCESS_SHADER_READ_BIT));
    initialBarriers.push_back(makeBarrier(frame.intermediateImage.image, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_ACCESS_SHADER_WRITE_BIT));
    initialBarriers.push_back(makeBarrier(frame.denoiseImage.image, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_ACCESS_SHADER_WRITE_BIT));
    initialBarriers.push_back(makeBarrier(output.image, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_ACCESS_SHADER_WRITE_BIT));

    for (const ImportedImage& historyImage : historyImages) {
        if (historyImage.image != VK_NULL_HANDLE) {
            initialBarriers.push_back(makeBarrier(historyImage.image, VK_IMAGE_LAYOUT_GENERAL,
                                                  VK_ACCESS_SHADER_READ_BIT));
        }
    }

    if (!initialBarriers.empty()) {
        vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(initialBarriers.size()),
                             initialBarriers.data());
    }

    vkCmdFillBuffer(frame.commandBuffer, frame.exposureBuffer, 0, VK_WHOLE_SIZE, 0);

    VkBufferMemoryBarrier fillBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    fillBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fillBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fillBarrier.buffer = frame.exposureBuffer;
    fillBarrier.offset = 0;
    fillBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &fillBarrier, 0,
                         nullptr);

    const uint32_t groupCountX = (dimensions_.width + kLocalSizeX - 1) / kLocalSizeX;
    const uint32_t groupCountY = (dimensions_.height + kLocalSizeY - 1) / kLocalSizeY;

    // Stage 1: Rotation + Sharpen
    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      stages_.rotationSharpen.pipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            stages_.rotationSharpen.layout, 0, 1,
                            &frame.descriptors.rotationSharpen, 0, nullptr);
    vkCmdDispatch(frame.commandBuffer, groupCountX, groupCountY, 1);

    VkImageMemoryBarrier rotationToExposure{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    rotationToExposure.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    rotationToExposure.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    rotationToExposure.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    rotationToExposure.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    rotationToExposure.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rotationToExposure.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rotationToExposure.image = frame.intermediateImage.image;
    rotationToExposure.subresourceRange = colorRange;

    vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &rotationToExposure);

    // Stage 2: Over-exposure detection
    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, stages_.exposure.pipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            stages_.exposure.layout, 0, 1, &frame.descriptors.exposure, 0,
                            nullptr);
    vkCmdDispatch(frame.commandBuffer, groupCountX, groupCountY, 1);

    // Stage 3: Multi-frame denoise
    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, stages_.denoise.pipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            stages_.denoise.layout, 0, 1, &frame.descriptors.denoise, 0,
                            nullptr);
    vkCmdDispatch(frame.commandBuffer, groupCountX, groupCountY, 1);

    VkImageMemoryBarrier denoiseToSuper{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    denoiseToSuper.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    denoiseToSuper.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    denoiseToSuper.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    denoiseToSuper.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    denoiseToSuper.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    denoiseToSuper.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    denoiseToSuper.image = frame.denoiseImage.image;
    denoiseToSuper.subresourceRange = colorRange;

    vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &denoiseToSuper);

    // Stage 4: Super-resolution
    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      stages_.superResolution.pipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            stages_.superResolution.layout, 0, 1,
                            &frame.descriptors.superResolution, 0, nullptr);
    vkCmdDispatch(frame.commandBuffer, groupCountX, groupCountY, 1);

    VkImageMemoryBarrier finalBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    finalBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    finalBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    finalBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    finalBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    finalBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finalBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    finalBarrier.image = output.image;
    finalBarrier.subresourceRange = colorRange;

    vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &finalBarrier);

    VK_CHECK(vkEndCommandBuffer(frame.commandBuffer));
}

ImportedImage VulkanAndroidFrameProcessor::importHardwareBuffer(AHardwareBuffer* buffer) {
    if (!buffer) {
        throw std::runtime_error("Null AHardwareBuffer");
    }

    AHardwareBuffer_describe(buffer, &context_->bufferDesc());
    AHardwareBuffer_acquire(buffer);

    ImportedImage result{};
    result.hardwareBuffer = buffer;
    result.format = chooseFormat(context_->bufferDesc());
    result.extent = chooseExtent(context_->bufferDesc());

    VkAndroidHardwareBufferFormatPropertiesANDROID formatProps{
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};

    VkAndroidHardwareBufferPropertiesANDROID bufferProps{
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    bufferProps.pNext = &formatProps;

    VK_CHECK(vkGetAndroidHardwareBufferPropertiesANDROID(context_->device(), buffer, &bufferProps));

    VkExternalMemoryImageCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.pNext = &externalInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = result.format;
    imageInfo.extent = {result.extent.width, result.extent.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = kInputUsage | kOutputUsage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(context_->device(), &imageInfo, nullptr, &result.image));

    VkImportAndroidHardwareBufferInfoANDROID importInfo{
        VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
    importInfo.buffer = buffer;

    VkMemoryDedicatedAllocateInfo dedicatedInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicatedInfo.image = result.image;

    importInfo.pNext = &dedicatedInfo;

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &importInfo;
    allocInfo.allocationSize = bufferProps.allocationSize;
    allocInfo.memoryTypeIndex = findMemoryType(*context_, bufferProps.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK(vkAllocateMemory(context_->device(), &allocInfo, nullptr, &result.memory));
    VK_CHECK(vkBindImageMemory(context_->device(), result.image, result.memory, 0));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = result.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = result.format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(context_->device(), &viewInfo, nullptr, &result.view));

    return result;
}

void VulkanAndroidFrameProcessor::releaseImportedImage(ImportedImage& image) {
    if (image.view != VK_NULL_HANDLE) {
        vkDestroyImageView(context_->device(), image.view, nullptr);
        image.view = VK_NULL_HANDLE;
    }
    if (image.image != VK_NULL_HANDLE) {
        vkDestroyImage(context_->device(), image.image, nullptr);
        image.image = VK_NULL_HANDLE;
    }
    if (image.memory != VK_NULL_HANDLE) {
        vkFreeMemory(context_->device(), image.memory, nullptr);
        image.memory = VK_NULL_HANDLE;
    }
    if (image.hardwareBuffer) {
        AHardwareBuffer_release(image.hardwareBuffer);
        image.hardwareBuffer = nullptr;
    }
}

VkBuffer VulkanAndroidFrameProcessor::createStagingBuffer(VkDeviceSize size,
                                                          VkDeviceMemory* outMemory) {
    VkBufferCreateInfo bufferInfo = makeBufferCreateInfo(size,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    VkBuffer buffer = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(context_->device(), &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(context_->device(), buffer, &requirements);

    VkMemoryAllocateInfo allocInfo =
        makeMemoryAllocateInfo(requirements,
                               findMemoryType(*context_, requirements.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK_CHECK(vkAllocateMemory(context_->device(), &allocInfo, nullptr, outMemory));
    VK_CHECK(vkBindBufferMemory(context_->device(), buffer, *outMemory, 0));
    return buffer;
}

VulkanAndroidFrameProcessor::ManagedImage VulkanAndroidFrameProcessor::createManagedImage(
    VkFormat format, VkImageUsageFlags usage) {
    ManagedImage image{};

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {dimensions_.width, dimensions_.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(context_->device(), &imageInfo, nullptr, &image.image));

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(context_->device(), image.image, &requirements);

    VkMemoryAllocateInfo allocInfo =
        makeMemoryAllocateInfo(requirements,
                               findMemoryType(*context_, requirements.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    VK_CHECK(vkAllocateMemory(context_->device(), &allocInfo, nullptr, &image.memory));
    VK_CHECK(vkBindImageMemory(context_->device(), image.image, image.memory, 0));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(context_->device(), &viewInfo, nullptr, &image.view));
    return image;
}

void VulkanAndroidFrameProcessor::destroyManagedImage(ManagedImage& image) {
    if (image.view != VK_NULL_HANDLE) {
        vkDestroyImageView(context_->device(), image.view, nullptr);
        image.view = VK_NULL_HANDLE;
    }
    if (image.image != VK_NULL_HANDLE) {
        vkDestroyImage(context_->device(), image.image, nullptr);
        image.image = VK_NULL_HANDLE;
    }
    if (image.memory != VK_NULL_HANDLE) {
        vkFreeMemory(context_->device(), image.memory, nullptr);
        image.memory = VK_NULL_HANDLE;
    }
}

void VulkanAndroidFrameProcessor::updateDescriptorSet(FrameResources& frame,
                                                      const ImportedImage& input,
                                                      const ImportedImage& output,
                                                      std::span<const ImportedImage> historyImages) {
    VkDescriptorImageInfo inputInfo{};
    inputInfo.imageView = input.view;
    inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    inputInfo.sampler = context_->linearSampler();

    VkDescriptorImageInfo intermediateInfo{};
    intermediateInfo.imageView = frame.intermediateImage.view;
    intermediateInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo denoiseInfo{};
    denoiseInfo.imageView = frame.denoiseImage.view;
    denoiseInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageView = output.view;
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, kMaxHistoryImages> historyInfos{};
    for (uint32_t i = 0; i < kMaxHistoryImages; ++i) {
        if (i < historyImages.size() && historyImages[i].view != VK_NULL_HANDLE) {
            historyInfos[i].imageView = historyImages[i].view;
            historyInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
            historyInfos[i].imageView = frame.intermediateImage.view;
            historyInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    VkDescriptorBufferInfo paramsInfo{};
    paramsInfo.buffer = frame.paramsBuffer;
    paramsInfo.offset = 0;
    paramsInfo.range = sizeof(ParamsBlock);

    VkDescriptorBufferInfo exposureInfo{};
    exposureInfo.buffer = frame.exposureBuffer;
    exposureInfo.offset = 0;
    exposureInfo.range =
        static_cast<VkDeviceSize>(kExposureBinCount) * sizeof(uint32_t);

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(13);

    // Rotation + sharpen set
    VkWriteDescriptorSet writeRotationInput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeRotationInput.dstSet = frame.descriptors.rotationSharpen;
    writeRotationInput.dstBinding = kRotationSharpenBindingInput;
    writeRotationInput.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeRotationInput.descriptorCount = 1;
    writeRotationInput.pImageInfo = &inputInfo;
    writes.push_back(writeRotationInput);

    VkWriteDescriptorSet writeRotationOutput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeRotationOutput.dstSet = frame.descriptors.rotationSharpen;
    writeRotationOutput.dstBinding = kRotationSharpenBindingOutput;
    writeRotationOutput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeRotationOutput.descriptorCount = 1;
    writeRotationOutput.pImageInfo = &intermediateInfo;
    writes.push_back(writeRotationOutput);

    VkWriteDescriptorSet writeRotationParams{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeRotationParams.dstSet = frame.descriptors.rotationSharpen;
    writeRotationParams.dstBinding = kRotationSharpenBindingParams;
    writeRotationParams.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeRotationParams.descriptorCount = 1;
    writeRotationParams.pBufferInfo = &paramsInfo;
    writes.push_back(writeRotationParams);

    // Exposure set
    VkWriteDescriptorSet writeExposureInput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeExposureInput.dstSet = frame.descriptors.exposure;
    writeExposureInput.dstBinding = kExposureBindingInput;
    writeExposureInput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeExposureInput.descriptorCount = 1;
    writeExposureInput.pImageInfo = &intermediateInfo;
    writes.push_back(writeExposureInput);

    VkWriteDescriptorSet writeExposureOutput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeExposureOutput.dstSet = frame.descriptors.exposure;
    writeExposureOutput.dstBinding = kExposureBindingOutput;
    writeExposureOutput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeExposureOutput.descriptorCount = 1;
    writeExposureOutput.pBufferInfo = &exposureInfo;
    writes.push_back(writeExposureOutput);

    VkWriteDescriptorSet writeExposureParams{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeExposureParams.dstSet = frame.descriptors.exposure;
    writeExposureParams.dstBinding = kExposureBindingParams;
    writeExposureParams.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeExposureParams.descriptorCount = 1;
    writeExposureParams.pBufferInfo = &paramsInfo;
    writes.push_back(writeExposureParams);

    // Denoise set
    VkWriteDescriptorSet writeDenoiseInput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeDenoiseInput.dstSet = frame.descriptors.denoise;
    writeDenoiseInput.dstBinding = kDenoiseBindingInput;
    writeDenoiseInput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDenoiseInput.descriptorCount = 1;
    writeDenoiseInput.pImageInfo = &intermediateInfo;
    writes.push_back(writeDenoiseInput);

    VkWriteDescriptorSet writeDenoiseHistory{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeDenoiseHistory.dstSet = frame.descriptors.denoise;
    writeDenoiseHistory.dstBinding = kDenoiseBindingHistory;
    writeDenoiseHistory.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDenoiseHistory.descriptorCount = kMaxHistoryImages;
    writeDenoiseHistory.pImageInfo = historyInfos.data();
    writes.push_back(writeDenoiseHistory);

    VkWriteDescriptorSet writeDenoiseOutput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeDenoiseOutput.dstSet = frame.descriptors.denoise;
    writeDenoiseOutput.dstBinding = kDenoiseBindingOutput;
    writeDenoiseOutput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDenoiseOutput.descriptorCount = 1;
    writeDenoiseOutput.pImageInfo = &denoiseInfo;
    writes.push_back(writeDenoiseOutput);

    VkWriteDescriptorSet writeDenoiseParams{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeDenoiseParams.dstSet = frame.descriptors.denoise;
    writeDenoiseParams.dstBinding = kDenoiseBindingParams;
    writeDenoiseParams.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDenoiseParams.descriptorCount = 1;
    writeDenoiseParams.pBufferInfo = &paramsInfo;
    writes.push_back(writeDenoiseParams);

    // Super-resolution set
    VkWriteDescriptorSet writeSuperInput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSuperInput.dstSet = frame.descriptors.superResolution;
    writeSuperInput.dstBinding = kSuperResBindingInput;
    writeSuperInput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeSuperInput.descriptorCount = 1;
    writeSuperInput.pImageInfo = &denoiseInfo;
    writes.push_back(writeSuperInput);

    VkWriteDescriptorSet writeSuperOutput{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSuperOutput.dstSet = frame.descriptors.superResolution;
    writeSuperOutput.dstBinding = kSuperResBindingOutput;
    writeSuperOutput.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeSuperOutput.descriptorCount = 1;
    writeSuperOutput.pImageInfo = &outputInfo;
    writes.push_back(writeSuperOutput);

    VkWriteDescriptorSet writeSuperParams{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSuperParams.dstSet = frame.descriptors.superResolution;
    writeSuperParams.dstBinding = kSuperResBindingParams;
    writeSuperParams.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeSuperParams.descriptorCount = 1;
    writeSuperParams.pBufferInfo = &paramsInfo;
    writes.push_back(writeSuperParams);

    vkUpdateDescriptorSets(context_->device(), static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

void VulkanAndroidFrameProcessor::ensureHistoryCapacity(std::size_t size) {
    if (historyCache_.size() < size) {
        historyCache_.resize(size);
    }
}

}  // namespace video::processing
