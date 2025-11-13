#include "multi_frame_denoiser.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

std::vector<char> readBinaryFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File is empty: " + path.string());
    }

    std::vector<char> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);
    return buffer;
}

VkExtent3D makeExtent(uint32_t width, uint32_t height) {
    return VkExtent3D{ width, height, 1 };
}

VkImageSubresourceRange makeColorRange() {
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;
    return range;
}

}  // namespace

MultiFrameDenoiser::~MultiFrameDenoiser() {
    cleanup();
}

void MultiFrameDenoiser::initialize(VulkanContext& context, const DenoiserConfig& config) {
    if (m_context != nullptr) {
        throw std::runtime_error("MultiFrameDenoiser is already initialized.");
    }
    if (config.width == 0 || config.height == 0) {
        throw std::runtime_error("Invalid denoiser resolution.");
    }
    if (config.historyLength == 0 || config.historyLength > kMaxHistoryImages) {
        throw std::runtime_error("History length must be between 1 and " + std::to_string(kMaxHistoryImages));
    }
    if (!std::filesystem::exists(config.shaderSpirvPath)) {
        throw std::runtime_error("Shader SPIR-V file does not exist: " + config.shaderSpirvPath.string());
    }

    m_context = &context;
    m_config = config;

    createHistoryImages();
    createAuxiliaryResources();
    createDescriptorSetLayout();
    createPipeline();
    createDescriptorPoolAndSet();

    m_uniforms.historyLength = m_config.historyLength;
    m_uniforms.frameBlendFactor = m_config.frameBlendFactor;
    m_uniforms.luminanceSigma = m_config.luminanceSigma;
}

void MultiFrameDenoiser::cleanup() {
    if (!m_context) {
        return;
    }

    VkDevice device = m_context->device();

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, m_shaderModule, nullptr);
        m_shaderModule = VK_NULL_HANDLE;
    }

    destroyBuffer(m_uniformBuffer);
    destroyBuffer(m_uploadBuffer);
    destroyBuffer(m_downloadBuffer);

    destroyImage(m_accumulationImage);
    destroyImage(m_outputImage);

    for (auto& image : m_historyImages) {
        destroyImage(image);
    }
    m_historyImages.clear();

    m_context = nullptr;
    m_frameCounter = 0;
    m_historyWriteIndex = 0;
}

void MultiFrameDenoiser::processFrame(const float* rgbaPixels, size_t floatCount, std::vector<float>& outputPixels) {
    if (!m_context) {
        throw std::runtime_error("MultiFrameDenoiser not initialized.");
    }

    const size_t expectedFloats = static_cast<size_t>(m_config.width) * m_config.height * 4;
    if (floatCount != expectedFloats) {
        throw std::runtime_error("Input frame size mismatch.");
    }

    uploadFrameToHistory(rgbaPixels, floatCount, m_historyWriteIndex);
    updateUniforms();
    updateDescriptors(m_historyWriteIndex);
    dispatchDenoise(m_historyWriteIndex);
    downloadOutput(outputPixels);

    m_historyWriteIndex = (m_historyWriteIndex + 1) % m_config.historyLength;
    ++m_frameCounter;
}

void MultiFrameDenoiser::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = kMaxHistoryImages;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_context->device(), &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout.");
    }
}

void MultiFrameDenoiser::createPipeline() {
    auto code = readBinaryFile(m_config.shaderSpirvPath);

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = code.size();
    shaderInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    if (vkCreateShaderModule(m_context->device(), &shaderInfo, nullptr, &m_shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module.");
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = m_shaderModule;
    stageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

    if (vkCreatePipelineLayout(m_context->device(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout.");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = m_pipelineLayout;

    if (vkCreateComputePipelines(m_context->device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline.");
    }
}

void MultiFrameDenoiser::createDescriptorPoolAndSet() {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = kMaxHistoryImages + 2;  // history + accumulation + output
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(m_context->device(), &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool.");
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;

    if (vkAllocateDescriptorSets(m_context->device(), &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set.");
    }
}

void MultiFrameDenoiser::createHistoryImages() {
    m_historyImages.resize(m_config.historyLength);
    for (uint32_t i = 0; i < m_config.historyLength; ++i) {
        m_historyImages[i] = createStorageImage(
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            "HistoryImage");
    }
}

void MultiFrameDenoiser::createAuxiliaryResources() {
    m_accumulationImage = createStorageImage(
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        "AccumulationImage");

    m_outputImage = createStorageImage(
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        "OutputImage");

    const VkDeviceSize frameBytes = static_cast<VkDeviceSize>(m_config.width) * m_config.height * 4 * sizeof(float);

    m_uploadBuffer = createBuffer(frameBytes,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  "UploadBuffer");

    m_downloadBuffer = createBuffer(frameBytes,
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    "DownloadBuffer");

    m_uniformBuffer = createBuffer(sizeof(DenoiseUniforms),
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   "UniformBuffer");

    // Clear accumulation and history images to zero.
    VkClearColorValue zeroColor{};
    VkImageSubresourceRange range = makeColorRange();

    auto cmd = m_context->beginSingleTimeCommands();

    auto initializeImage = [&](ImageResource& image) {
        transitionImageLayout(cmd,
                              image.image,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_GENERAL,
                              0,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &zeroColor, 1, &range);
    };

    for (auto& image : m_historyImages) {
        initializeImage(image);
    }
    initializeImage(m_accumulationImage);
    initializeImage(m_outputImage);

    m_context->endSingleTimeCommands(cmd);
}

void MultiFrameDenoiser::updateUniforms() {
    m_uniforms.frameIndex = m_frameCounter;
    m_uniforms.historyLength = m_config.historyLength;
    m_uniforms.frameBlendFactor = (m_frameCounter == 0) ? 0.0f : m_config.frameBlendFactor;
    m_uniforms.luminanceSigma = m_config.luminanceSigma;

    void* data = nullptr;
    vkMapMemory(m_context->device(), m_uniformBuffer.memory, 0, sizeof(DenoiseUniforms), 0, &data);
    std::memcpy(data, &m_uniforms, sizeof(DenoiseUniforms));
    vkUnmapMemory(m_context->device(), m_uniformBuffer.memory);
}

void MultiFrameDenoiser::updateDescriptors(uint32_t writeIndex) {
    std::array<VkDescriptorImageInfo, kMaxHistoryImages> historyInfos{};

    for (uint32_t i = 0; i < kMaxHistoryImages; ++i) {
        uint32_t historyIdx;
        if (i < m_config.historyLength) {
            const int32_t offset = static_cast<int32_t>(i);
            const int32_t base = static_cast<int32_t>(writeIndex);
            historyIdx = static_cast<uint32_t>((base - offset + m_config.historyLength) % m_config.historyLength);
        } else {
            historyIdx = writeIndex;
        }

        historyInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        historyInfos[i].imageView = m_historyImages[historyIdx].view;
        historyInfos[i].sampler = VK_NULL_HANDLE;
    }

    VkDescriptorImageInfo accumulationInfo{};
    accumulationInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    accumulationInfo.imageView = m_accumulationImage.view;

    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputInfo.imageView = m_outputImage.view;

    VkDescriptorBufferInfo uniformInfo{};
    uniformInfo.buffer = m_uniformBuffer.buffer;
    uniformInfo.offset = 0;
    uniformInfo.range = sizeof(DenoiseUniforms);

    std::array<VkWriteDescriptorSet, 4> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = kMaxHistoryImages;
    writes[0].pImageInfo = historyInfos.data();

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &accumulationInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = m_descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &outputInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = m_descriptorSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &uniformInfo;

    vkUpdateDescriptorSets(m_context->device(),
                           static_cast<uint32_t>(writes.size()),
                           writes.data(),
                           0,
                           nullptr);
}

void MultiFrameDenoiser::uploadFrameToHistory(const float* rgbaPixels, size_t floatCount, uint32_t writeIndex) {
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(floatCount * sizeof(float));

    void* data = nullptr;
    vkMapMemory(m_context->device(), m_uploadBuffer.memory, 0, byteSize, 0, &data);
    std::memcpy(data, rgbaPixels, static_cast<size_t>(byteSize));
    vkUnmapMemory(m_context->device(), m_uploadBuffer.memory);

    auto cmd = m_context->beginSingleTimeCommands();

    VkImageSubresourceRange range = makeColorRange();
    auto& target = m_historyImages[writeIndex];

    transitionImageLayout(cmd,
                          target.image,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = makeExtent(m_config.width, m_config.height);

    vkCmdCopyBufferToImage(cmd,
                           m_uploadBuffer.buffer,
                           target.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyRegion);

    transitionImageLayout(cmd,
                          target.image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    m_context->endSingleTimeCommands(cmd);
}

void MultiFrameDenoiser::dispatchDenoise(uint32_t /*writeIndex*/) {
    auto cmd = m_context->beginSingleTimeCommands();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_pipelineLayout,
                            0,
                            1,
                            &m_descriptorSet,
                            0,
                            nullptr);

    const uint32_t groupSizeX = 16;
    const uint32_t groupSizeY = 16;
    const uint32_t groupCountX = (m_config.width + groupSizeX - 1) / groupSizeX;
    const uint32_t groupCountY = (m_config.height + groupSizeY - 1) / groupSizeY;
    vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

    m_context->endSingleTimeCommands(cmd);
}

void MultiFrameDenoiser::downloadOutput(std::vector<float>& outputPixels) {
    auto cmd = m_context->beginSingleTimeCommands();

    transitionImageLayout(cmd,
                          m_outputImage.image,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = makeExtent(m_config.width, m_config.height);

    vkCmdCopyImageToBuffer(cmd,
                           m_outputImage.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_downloadBuffer.buffer,
                           1,
                           &copyRegion);

    transitionImageLayout(cmd,
                          m_outputImage.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_TRANSFER_READ_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    m_context->endSingleTimeCommands(cmd);

    const size_t floatCount = static_cast<size_t>(m_config.width) * m_config.height * 4;
    outputPixels.resize(floatCount);

    void* data = nullptr;
    vkMapMemory(m_context->device(), m_downloadBuffer.memory, 0, m_downloadBuffer.size, 0, &data);
    std::memcpy(outputPixels.data(), data, static_cast<size_t>(m_downloadBuffer.size));
    vkUnmapMemory(m_context->device(), m_downloadBuffer.memory);
}

MultiFrameDenoiser::ImageResource MultiFrameDenoiser::createStorageImage(VkFormat format,
                                                                         VkImageUsageFlags usage,
                                                                         const char* /*debugName*/) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = makeExtent(m_config.width, m_config.height);
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    ImageResource resource{};

    if (vkCreateImage(m_context->device(), &imageInfo, nullptr, &resource.image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image.");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_context->device(), resource.image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = m_context->findMemoryType(memRequirements.memoryTypeBits,
                                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(m_context->device(), &allocInfo, nullptr, &resource.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory.");
    }

    vkBindImageMemory(m_context->device(), resource.image, resource.memory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = resource.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = makeColorRange();

    if (vkCreateImageView(m_context->device(), &viewInfo, nullptr, &resource.view) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view.");
    }

    return resource;
}

void MultiFrameDenoiser::destroyImage(ImageResource& image) {
    if (!m_context) {
        return;
    }

    VkDevice device = m_context->device();

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
}

MultiFrameDenoiser::BufferResource MultiFrameDenoiser::createBuffer(VkDeviceSize size,
                                                                    VkBufferUsageFlags usage,
                                                                    VkMemoryPropertyFlags properties,
                                                                    const char* /*debugName*/) {
    BufferResource buffer{};
    buffer.size = size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_context->device(), &bufferInfo, nullptr, &buffer.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer.");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_context->device(), buffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = m_context->findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_context->device(), &allocInfo, nullptr, &buffer.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory.");
    }

    vkBindBufferMemory(m_context->device(), buffer.buffer, buffer.memory, 0);
    return buffer;
}

void MultiFrameDenoiser::destroyBuffer(BufferResource& buffer) {
    if (!m_context) {
        return;
    }

    VkDevice device = m_context->device();
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

void MultiFrameDenoiser::transitionImageLayout(VkCommandBuffer cmd,
                                               VkImage image,
                                               VkImageLayout oldLayout,
                                               VkImageLayout newLayout,
                                               VkAccessFlags srcAccessMask,
                                               VkAccessFlags dstAccessMask,
                                               VkPipelineStageFlags srcStage,
                                               VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.image = image;
    barrier.subresourceRange = makeColorRange();

    vkCmdPipelineBarrier(cmd,
                         srcStage,
                         dstStage,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
}
