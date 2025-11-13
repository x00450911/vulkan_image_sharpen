#include "VideoDenoiser.h"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SHADER_BINARY_PATH
#error "SHADER_BINARY_PATH must be defined by the build system"
#endif

namespace {
constexpr VkFormat kFrameFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr uint32_t kBytesPerPixel = 4;
constexpr uint32_t kWorkgroupSizeX = 16;
constexpr uint32_t kWorkgroupSizeY = 16;

void vkCheck(VkResult result, const char* errorMessage) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(errorMessage);
    }
}

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open shader binary: " + path);
    }

    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    file.close();

    return buffer;
}

void destroyFrameImage(VkDevice device, FrameImage& image) {
    if (image.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, image.view, nullptr);
        image.view = VK_NULL_HANDLE;
    }
    if (image.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, image.image, nullptr);
        image.image = VK_NULL_HANDLE;
    }
    if (image.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, image.memory, nullptr);
        image.memory = VK_NULL_HANDLE;
    }
    image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

VkImageSubresourceRange defaultColorRange() {
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;
    return range;
}

}  // namespace

VideoDenoiser::VideoDenoiser(VulkanContext& context) : context_(context) {}

VideoDenoiser::~VideoDenoiser() {
    cleanup();
}

void VideoDenoiser::initialize(uint32_t width, uint32_t height) {
    if (initialized_) {
        return;
    }

    width_ = width;
    height_ = height;
    imageByteSize_ = static_cast<VkDeviceSize>(width_) * height_ * kBytesPerPixel;

    createImages();
    createBuffers();
    createDescriptorSetLayout();
    createPipelineLayout();
    createComputePipeline();
    createDescriptorPool();
    allocateDescriptorSet();
    createCommandResources();

    historyValid_ = false;
    processedFrames_ = 0;
    initialized_ = true;
}

void VideoDenoiser::cleanup() {
    VkDevice device = context_.getDevice();

    if (device == VK_NULL_HANDLE) {
        return;
    }

    if (computeFence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device, computeFence_, nullptr);
        computeFence_ = VK_NULL_HANDLE;
    }

    if (commandBuffer_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device, context_.getCommandPool(), 1, &commandBuffer_);
        commandBuffer_ = VK_NULL_HANDLE;
    }

    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }

    if (computePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, computePipeline_, nullptr);
        computePipeline_ = VK_NULL_HANDLE;
    }

    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }

    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }

    if (uploadMapped_ != nullptr) {
        vkUnmapMemory(device, uploadMemory_);
        uploadMapped_ = nullptr;
    }
    if (uploadBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, uploadBuffer_, nullptr);
        uploadBuffer_ = VK_NULL_HANDLE;
    }
    if (uploadMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, uploadMemory_, nullptr);
        uploadMemory_ = VK_NULL_HANDLE;
    }

    if (readbackMapped_ != nullptr) {
        vkUnmapMemory(device, readbackMemory_);
        readbackMapped_ = nullptr;
    }
    if (readbackBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, readbackBuffer_, nullptr);
        readbackBuffer_ = VK_NULL_HANDLE;
    }
    if (readbackMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device, readbackMemory_, nullptr);
        readbackMemory_ = VK_NULL_HANDLE;
    }

    destroyImages();

    initialized_ = false;
    historyValid_ = false;
    processedFrames_ = 0;
}

void VideoDenoiser::processFrame(const std::vector<uint8_t>& rgbaPixels, std::vector<uint8_t>& outDenoisedPixels,
                                 float blendFactor, float sigmaColor, float sigmaTemporal, bool useSpatial) {
    if (!initialized_) {
        throw std::runtime_error("VideoDenoiser::initialize must be called before processing frames.");
    }
    if (rgbaPixels.size() < static_cast<size_t>(imageByteSize_)) {
        throw std::runtime_error("Input frame does not contain enough data.");
    }

    outDenoisedPixels.resize(static_cast<size_t>(imageByteSize_));

    std::memcpy(uploadMapped_, rgbaPixels.data(), static_cast<size_t>(imageByteSize_));

    updateDescriptorSet(currentImage_, historyImage_, accumulationImage_);

    vkCheck(vkResetCommandBuffer(commandBuffer_, 0), "Failed to reset command buffer.");

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkCheck(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "Failed to begin command buffer.");

    uploadToImage(commandBuffer_, currentImage_, rgbaPixels);

    if (!historyValid_) {
        // populate history with first frame to avoid bright start-up
        transitionImage(commandBuffer_, historyImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, VK_ACCESS_TRANSFER_WRITE_BIT);

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = {0, 0, 0};
        copyRegion.imageExtent = {width_, height_, 1};

        vkCmdCopyBufferToImage(commandBuffer_, uploadBuffer_, historyImage_.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        transitionImage(commandBuffer_, historyImage_, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    PushConstants constants{};
    constants.blendFactor = blendFactor;
    constants.sigmaColor = sigmaColor;
    constants.sigmaTemporal = sigmaTemporal;
    constants.frameIndex = static_cast<int>(processedFrames_);
    constants.historyValid = historyValid_ ? 1 : 0;
    constants.useSpatialFilter = useSpatial ? 1 : 0;

    recordComputePass(commandBuffer_, constants);

    readbackFromImage(commandBuffer_, accumulationImage_);

    vkCheck(vkEndCommandBuffer(commandBuffer_), "Failed to end command buffer.");

    submitAndWait(commandBuffer_);

    std::memcpy(outDenoisedPixels.data(), readbackMapped_, static_cast<size_t>(imageByteSize_));

    ++processedFrames_;
    historyValid_ = true;

    std::swap(historyImage_, accumulationImage_);
}

void VideoDenoiser::createImages() {
    VkDevice device = context_.getDevice();

    auto createStorageImage = [&](FrameImage& image) {
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width_;
        imageInfo.extent.height = height_;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = kFrameFormat;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCheck(vkCreateImage(device, &imageInfo, nullptr, &image.image), "Failed to create storage image.");

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image.image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = context_.findMemoryType(
            memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vkCheck(vkAllocateMemory(device, &allocInfo, nullptr, &image.memory), "Failed to allocate image memory.");
        vkCheck(vkBindImageMemory(device, image.image, image.memory, 0), "Failed to bind image memory.");

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kFrameFormat;
        viewInfo.subresourceRange = defaultColorRange();

        vkCheck(vkCreateImageView(device, &viewInfo, nullptr, &image.view), "Failed to create image view.");

        image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    createStorageImage(currentImage_);
    createStorageImage(historyImage_);
    createStorageImage(accumulationImage_);

    VkCommandBuffer cmd = context_.beginSingleTimeCommands();
    FrameImage* images[] = {&currentImage_, &historyImage_, &accumulationImage_};
    for (FrameImage* image : images) {
        transitionImage(cmd, *image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, VK_ACCESS_TRANSFER_WRITE_BIT);

        VkClearColorValue clearValue{};
        VkImageSubresourceRange range = defaultColorRange();
        vkCmdClearColorImage(cmd, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);

        transitionImage(cmd, *image, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    context_.endSingleTimeCommands(cmd);
}

void VideoDenoiser::destroyImages() {
    VkDevice device = context_.getDevice();
    destroyFrameImage(device, currentImage_);
    destroyFrameImage(device, historyImage_);
    destroyFrameImage(device, accumulationImage_);
}

void VideoDenoiser::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    vkCheck(vkCreateDescriptorSetLayout(context_.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout_),
            "Failed to create descriptor set layout.");
}

void VideoDenoiser::createPipelineLayout() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    vkCheck(vkCreatePipelineLayout(context_.getDevice(), &layoutInfo, nullptr, &pipelineLayout_),
            "Failed to create pipeline layout.");
}

void VideoDenoiser::createComputePipeline() {
    const std::vector<char> shaderCode = readBinaryFile(SHADER_BINARY_PATH);

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = shaderCode.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule shaderModule;
    vkCheck(vkCreateShaderModule(context_.getDevice(), &moduleInfo, nullptr, &shaderModule),
            "Failed to create compute shader module.");

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout_;

    vkCheck(vkCreateComputePipelines(context_.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline_),
            "Failed to create compute pipeline.");

    vkDestroyShaderModule(context_.getDevice(), shaderModule, nullptr);
}

void VideoDenoiser::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    vkCheck(vkCreateDescriptorPool(context_.getDevice(), &poolInfo, nullptr, &descriptorPool_),
            "Failed to create descriptor pool.");
}

void VideoDenoiser::allocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout_;

    vkCheck(vkAllocateDescriptorSets(context_.getDevice(), &allocInfo, &descriptorSet_),
            "Failed to allocate descriptor set.");
}

void VideoDenoiser::createBuffers() {
    auto createBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                            VkBuffer& buffer, VkDeviceMemory& memory, void** mappedPtr) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCheck(vkCreateBuffer(context_.getDevice(), &bufferInfo, nullptr, &buffer), "Failed to create buffer.");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context_.getDevice(), buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = context_.findMemoryType(memRequirements.memoryTypeBits, properties);

        vkCheck(vkAllocateMemory(context_.getDevice(), &allocInfo, nullptr, &memory), "Failed to allocate buffer memory.");
        vkCheck(vkBindBufferMemory(context_.getDevice(), buffer, memory, 0), "Failed to bind buffer memory.");

        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            void* mapped = nullptr;
            vkCheck(vkMapMemory(context_.getDevice(), memory, 0, size, 0, &mapped), "Failed to map buffer memory.");
            *mappedPtr = mapped;
        }
    };

    createBuffer(imageByteSize_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 uploadBuffer_, uploadMemory_, &uploadMapped_);

    createBuffer(imageByteSize_, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 readbackBuffer_, readbackMemory_, &readbackMapped_);
}

void VideoDenoiser::createCommandResources() {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = context_.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    vkCheck(vkAllocateCommandBuffers(context_.getDevice(), &allocInfo, &commandBuffer_),
            "Failed to allocate command buffer.");

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCheck(vkCreateFence(context_.getDevice(), &fenceInfo, nullptr, &computeFence_),
            "Failed to create fence.");
}

void VideoDenoiser::updateDescriptorSet(const FrameImage& current, const FrameImage& history, const FrameImage& output) {
    std::array<VkDescriptorImageInfo, 3> imageInfos{};
    imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[0].imageView = current.view;

    imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[1].imageView = history.view;

    imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos[2].imageView = output.view;

    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &imageInfos[i];
    }

    vkUpdateDescriptorSets(context_.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void VideoDenoiser::transitionImage(VkCommandBuffer cmd, FrameImage& image, VkImageLayout newLayout,
                                    VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                                    VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask) {
    if (image.layout == newLayout) {
        return;
    }

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = image.layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = defaultColorRange();
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    image.layout = newLayout;
}

void VideoDenoiser::recordComputePass(VkCommandBuffer cmd, const PushConstants& constants) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);

    const uint32_t groupCountX = (width_ + kWorkgroupSizeX - 1) / kWorkgroupSizeX;
    const uint32_t groupCountY = (height_ + kWorkgroupSizeY - 1) / kWorkgroupSizeY;
    vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
}

void VideoDenoiser::uploadToImage(VkCommandBuffer cmd, FrameImage& targetImage, const std::vector<uint8_t>& /*rgbaPixels*/) {
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags srcAccess = 0;
    if (targetImage.layout == VK_IMAGE_LAYOUT_GENERAL) {
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        srcAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (targetImage.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccess = VK_ACCESS_TRANSFER_READ_BIT;
    }

    transitionImage(cmd, targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcStage, VK_PIPELINE_STAGE_TRANSFER_BIT, srcAccess, VK_ACCESS_TRANSFER_WRITE_BIT);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width_, height_, 1};

    vkCmdCopyBufferToImage(cmd, uploadBuffer_, targetImage.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    transitionImage(cmd, targetImage, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void VideoDenoiser::readbackFromImage(VkCommandBuffer cmd, FrameImage& sourceImage) {
    transitionImage(cmd, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width_, height_, 1};

    vkCmdCopyImageToBuffer(cmd, sourceImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer_, 1, &copyRegion);

    transitionImage(cmd, sourceImage, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void VideoDenoiser::submitAndWait(VkCommandBuffer cmd) {
    VkDevice device = context_.getDevice();
    VkQueue queue = context_.getComputeQueue();

    vkCheck(vkWaitForFences(device, 1, &computeFence_, VK_TRUE, UINT64_MAX), "Failed to wait for fence.");
    vkCheck(vkResetFences(device, 1, &computeFence_), "Failed to reset fence.");

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkCheck(vkQueueSubmit(queue, 1, &submitInfo, computeFence_), "Failed to submit compute work.");
    vkCheck(vkWaitForFences(device, 1, &computeFence_, VK_TRUE, UINT64_MAX), "Failed to wait for compute completion.");
}
