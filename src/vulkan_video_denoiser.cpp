#include "vulkan_video_denoiser.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vk_denoise {

VulkanVideoDenoiser::VulkanVideoDenoiser() {
    // Initialize frame buffers
    for (auto& fb : m_frameBuffers) {
        fb.image = VK_NULL_HANDLE;
        fb.memory = VK_NULL_HANDLE;
        fb.imageView = VK_NULL_HANDLE;
        fb.width = 0;
        fb.height = 0;
    }
    
    m_outputBuffer.image = VK_NULL_HANDLE;
    m_outputBuffer.memory = VK_NULL_HANDLE;
    m_outputBuffer.imageView = VK_NULL_HANDLE;
    
    m_motionBuffer.image = VK_NULL_HANDLE;
    m_motionBuffer.memory = VK_NULL_HANDLE;
    m_motionBuffer.imageView = VK_NULL_HANDLE;
}

VulkanVideoDenoiser::~VulkanVideoDenoiser() {
    cleanup();
}

bool VulkanVideoDenoiser::initialize(uint32_t width, uint32_t height, const DenoiseParams& params) {
    if (m_initialized) {
        std::cerr << "Denoiser already initialized" << std::endl;
        return false;
    }
    
    m_width = width;
    m_height = height;
    m_params = params;
    
    // Initialize Vulkan components
    if (!createInstance()) return false;
    if (!selectPhysicalDevice()) return false;
    if (!createLogicalDevice()) return false;
    if (!createCommandPool()) return false;
    if (!createDescriptorSetLayout()) return false;
    if (!createPipelineLayout()) return false;
    if (!createComputePipeline()) return false;
    if (!createDescriptorPool()) return false;
    if (!createFrameBuffers()) return false;
    if (!createMotionVectorBuffer()) return false;
    if (!createDescriptorSets()) return false;
    if (!allocateCommandBuffers()) return false;
    
    // Create fence for synchronization
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_fence) != VK_SUCCESS) {
        std::cerr << "Failed to create fence" << std::endl;
        return false;
    }
    
    m_initialized = true;
    std::cout << "Vulkan video denoiser initialized successfully" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frame history: " << params.frameCount << " frames" << std::endl;
    
    return true;
}

bool VulkanVideoDenoiser::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Video Denoiser";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;
    
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // No validation layers for production
    createInfo.enabledLayerCount = 0;
    createInfo.enabledExtensionCount = 0;
    
    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        std::cerr << "No Vulkan-capable GPUs found" << std::endl;
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());
    
    // Select first device with compute queue
    for (const auto& device : devices) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                m_physicalDevice = device;
                m_queueFamilyIndex = i;
                
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(device, &props);
                std::cout << "Selected GPU: " << props.deviceName << std::endl;
                
                return true;
            }
        }
    }
    
    std::cerr << "No suitable GPU found" << std::endl;
    return false;
}

bool VulkanVideoDenoiser::createLogicalDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    VkPhysicalDeviceFeatures deviceFeatures{};
    
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;
    createInfo.enabledLayerCount = 0;
    
    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device" << std::endl;
        return false;
    }
    
    vkGetDeviceQueue(m_device, m_queueFamilyIndex, 0, &m_computeQueue);
    return true;
}

bool VulkanVideoDenoiser::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    
    // Binding 0: Current frame (input)
    VkDescriptorSetLayoutBinding currentFrameBinding{};
    currentFrameBinding.binding = 0;
    currentFrameBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    currentFrameBinding.descriptorCount = 1;
    currentFrameBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(currentFrameBinding);
    
    // Binding 1-3: Previous frames (input)
    for (int i = 1; i <= 3; i++) {
        VkDescriptorSetLayoutBinding prevFrameBinding{};
        prevFrameBinding.binding = i;
        prevFrameBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        prevFrameBinding.descriptorCount = 1;
        prevFrameBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(prevFrameBinding);
    }
    
    // Binding 4: Output frame
    VkDescriptorSetLayoutBinding outputBinding{};
    outputBinding.binding = 4;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputBinding.descriptorCount = 1;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(outputBinding);
    
    // Binding 5: Motion vectors
    VkDescriptorSetLayoutBinding motionBinding{};
    motionBinding.binding = 5;
    motionBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    motionBinding.descriptorCount = 1;
    motionBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(motionBinding);
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::createPipelineLayout() {
    // Push constants for parameters
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        return false;
    }
    
    return true;
}

VkShaderModule VulkanVideoDenoiser::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    
    return shaderModule;
}

std::vector<char> VulkanVideoDenoiser::readShaderFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

bool VulkanVideoDenoiser::createComputePipeline() {
    // Load compiled SPIR-V shader
    std::vector<char> computeShaderCode;
    try {
        computeShaderCode = readShaderFile("shaders/temporal_denoise.spv");
    } catch (const std::exception& e) {
        std::cerr << "Error loading shader: " << e.what() << std::endl;
        return false;
    }
    
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);
    if (computeShaderModule == VK_NULL_HANDLE) {
        std::cerr << "Failed to create shader module" << std::endl;
        return false;
    }
    
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = computeShaderStageInfo;
    pipelineInfo.layout = m_pipelineLayout;
    
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        vkDestroyShaderModule(m_device, computeShaderModule, nullptr);
        return false;
    }
    
    vkDestroyShaderModule(m_device, computeShaderModule, nullptr);
    return true;
}

bool VulkanVideoDenoiser::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes;
    
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 6; // 4 frame buffers + 1 output + 1 motion
    poolSizes.push_back(poolSize);
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }
    
    return true;
}

uint32_t VulkanVideoDenoiser::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

bool VulkanVideoDenoiser::createImage(uint32_t width, uint32_t height, VkFormat format,
                                     VkImageUsageFlags usage, VkImage& image,
                                     VkDeviceMemory& memory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateImage(m_device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "Failed to create image" << std::endl;
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, image, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate image memory" << std::endl;
        return false;
    }
    
    vkBindImageMemory(m_device, image, memory, 0);
    return true;
}

bool VulkanVideoDenoiser::createImageView(VkImage image, VkFormat format, VkImageView& imageView) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    if (vkCreateImageView(m_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        std::cerr << "Failed to create image view" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                      VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory" << std::endl;
        return false;
    }
    
    vkBindBufferMemory(m_device, buffer, memory, 0);
    return true;
}

bool VulkanVideoDenoiser::createFrameBuffers() {
    VkDeviceSize bufferSize = m_width * m_height * 4; // RGBA8
    
    // Create staging buffer for uploads
    if (!createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     m_stagingBuffer, m_stagingMemory)) {
        return false;
    }
    
    // Create staging buffer for downloads
    if (!createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     m_outputStagingBuffer, m_outputStagingMemory)) {
        return false;
    }
    
    // Create frame history buffers
    for (int i = 0; i < MAX_FRAME_HISTORY; i++) {
        if (!createImage(m_width, m_height, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        m_frameBuffers[i].image, m_frameBuffers[i].memory)) {
            return false;
        }
        
        if (!createImageView(m_frameBuffers[i].image, VK_FORMAT_R8G8B8A8_UNORM,
                           m_frameBuffers[i].imageView)) {
            return false;
        }
        
        m_frameBuffers[i].width = m_width;
        m_frameBuffers[i].height = m_height;
    }
    
    // Create output buffer
    if (!createImage(m_width, m_height, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    m_outputBuffer.image, m_outputBuffer.memory)) {
        return false;
    }
    
    if (!createImageView(m_outputBuffer.image, VK_FORMAT_R8G8B8A8_UNORM,
                        m_outputBuffer.imageView)) {
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::createMotionVectorBuffer() {
    if (!createImage(m_width, m_height, VK_FORMAT_R16G16_SFLOAT,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    m_motionBuffer.image, m_motionBuffer.memory)) {
        return false;
    }
    
    if (!createImageView(m_motionBuffer.image, VK_FORMAT_R16G16_SFLOAT,
                        m_motionBuffer.imageView)) {
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::createDescriptorSets() {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;
    
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets" << std::endl;
        return false;
    }
    
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    std::vector<VkDescriptorImageInfo> imageInfos;
    
    // Current frame (binding 0)
    VkDescriptorImageInfo currentImageInfo{};
    currentImageInfo.imageView = m_frameBuffers[0].imageView;
    currentImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos.push_back(currentImageInfo);
    
    VkWriteDescriptorSet currentWrite{};
    currentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    currentWrite.dstSet = m_descriptorSet;
    currentWrite.dstBinding = 0;
    currentWrite.dstArrayElement = 0;
    currentWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    currentWrite.descriptorCount = 1;
    currentWrite.pImageInfo = &imageInfos[0];
    descriptorWrites.push_back(currentWrite);
    
    // Previous frames (bindings 1-3)
    for (int i = 1; i <= 3; i++) {
        VkDescriptorImageInfo prevImageInfo{};
        prevImageInfo.imageView = m_frameBuffers[i].imageView;
        prevImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos.push_back(prevImageInfo);
        
        VkWriteDescriptorSet prevWrite{};
        prevWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        prevWrite.dstSet = m_descriptorSet;
        prevWrite.dstBinding = i;
        prevWrite.dstArrayElement = 0;
        prevWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        prevWrite.descriptorCount = 1;
        prevWrite.pImageInfo = &imageInfos[i];
        descriptorWrites.push_back(prevWrite);
    }
    
    // Output frame (binding 4)
    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = m_outputBuffer.imageView;
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos.push_back(outputImageInfo);
    
    VkWriteDescriptorSet outputWrite{};
    outputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    outputWrite.dstSet = m_descriptorSet;
    outputWrite.dstBinding = 4;
    outputWrite.dstArrayElement = 0;
    outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputWrite.descriptorCount = 1;
    outputWrite.pImageInfo = &imageInfos[4];
    descriptorWrites.push_back(outputWrite);
    
    // Motion vectors (binding 5)
    VkDescriptorImageInfo motionImageInfo{};
    motionImageInfo.imageView = m_motionBuffer.imageView;
    motionImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfos.push_back(motionImageInfo);
    
    VkWriteDescriptorSet motionWrite{};
    motionWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    motionWrite.dstSet = m_descriptorSet;
    motionWrite.dstBinding = 5;
    motionWrite.dstArrayElement = 0;
    motionWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    motionWrite.descriptorCount = 1;
    motionWrite.pImageInfo = &imageInfos[5];
    descriptorWrites.push_back(motionWrite);
    
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptorWrites.size()),
                          descriptorWrites.data(), 0, nullptr);
    
    return true;
}

bool VulkanVideoDenoiser::allocateCommandBuffers() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    if (vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffers" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanVideoDenoiser::transitionImageLayout(VkImage image, VkFormat format,
                                               VkImageLayout oldLayout, VkImageLayout newLayout,
                                               VkCommandBuffer cmdBuffer) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        std::cerr << "Unsupported layout transition" << std::endl;
        return false;
    }
    
    vkCmdPipelineBarrier(cmdBuffer, sourceStage, destinationStage, 0,
                        0, nullptr, 0, nullptr, 1, &barrier);
    
    return true;
}

bool VulkanVideoDenoiser::copyBufferToImage(VkBuffer buffer, VkImage image,
                                           uint32_t width, uint32_t height) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    
    transitionImageLayout(image, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         m_commandBuffer);
    
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyBufferToImage(m_commandBuffer, buffer, image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    transitionImageLayout(image, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         m_commandBuffer);
    
    vkEndCommandBuffer(m_commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;
    
    vkResetFences(m_device, 1, &m_fence);
    vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
    vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    
    return true;
}

bool VulkanVideoDenoiser::copyImageToBuffer(VkImage image, VkBuffer buffer,
                                           uint32_t width, uint32_t height) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    
    transitionImageLayout(image, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         m_commandBuffer);
    
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyImageToBuffer(m_commandBuffer, image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    
    transitionImageLayout(image, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         m_commandBuffer);
    
    vkEndCommandBuffer(m_commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;
    
    vkResetFences(m_device, 1, &m_fence);
    vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
    vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    
    return true;
}

void VulkanVideoDenoiser::rotateFrameBuffers() {
    // Rotate frame history (move current to history)
    m_currentFrameIndex = (m_currentFrameIndex + 1) % MAX_FRAME_HISTORY;
}

bool VulkanVideoDenoiser::executeCompute() {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);
    
    // Bind pipeline and descriptor sets
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    
    // Push constants
    PushConstants pushConstants;
    pushConstants.temporalWeight = m_params.temporalWeight;
    pushConstants.spatialSigma = m_params.spatialSigma;
    pushConstants.colorSigma = m_params.colorSigma;
    pushConstants.useMotionCompensation = m_params.useMotionCompensation ? 1 : 0;
    pushConstants.noiseThreshold = m_params.noiseThreshold;
    pushConstants.frameCount = std::min(m_params.frameCount, std::min(m_framesProcessed + 1, MAX_FRAME_HISTORY));
    pushConstants.adaptiveStrength = m_params.adaptiveStrength;
    pushConstants.kernelSize = m_params.kernelSize;
    
    vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pushConstants);
    
    // Dispatch compute shader
    uint32_t groupCountX = (m_width + 15) / 16;
    uint32_t groupCountY = (m_height + 15) / 16;
    vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, 1);
    
    vkEndCommandBuffer(m_commandBuffer);
    
    // Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;
    
    vkResetFences(m_device, 1, &m_fence);
    vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
    vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    
    return true;
}

bool VulkanVideoDenoiser::processFrame(const uint8_t* inputData, uint8_t* outputData) {
    if (!m_initialized) {
        std::cerr << "Denoiser not initialized" << std::endl;
        return false;
    }
    
    VkDeviceSize bufferSize = m_width * m_height * 4;
    
    // Upload input data to staging buffer
    void* mappedData;
    vkMapMemory(m_device, m_stagingMemory, 0, bufferSize, 0, &mappedData);
    memcpy(mappedData, inputData, bufferSize);
    vkUnmapMemory(m_device, m_stagingMemory);
    
    // Copy to current frame buffer
    copyBufferToImage(m_stagingBuffer, m_frameBuffers[m_currentFrameIndex].image,
                     m_width, m_height);
    
    // Execute denoising
    executeCompute();
    
    // Copy result to staging buffer
    copyImageToBuffer(m_outputBuffer.image, m_outputStagingBuffer, m_width, m_height);
    
    // Download output data
    vkMapMemory(m_device, m_outputStagingMemory, 0, bufferSize, 0, &mappedData);
    memcpy(outputData, mappedData, bufferSize);
    vkUnmapMemory(m_device, m_outputStagingMemory);
    
    // Rotate frame buffers for next frame
    rotateFrameBuffers();
    m_framesProcessed++;
    
    return true;
}

bool VulkanVideoDenoiser::processFrameWithMotion(const uint8_t* inputData,
                                                const float* motionVectors,
                                                uint8_t* outputData) {
    // TODO: Implement motion vector upload
    // For now, just process without motion
    return processFrame(inputData, outputData);
}

void VulkanVideoDenoiser::updateParameters(const DenoiseParams& params) {
    m_params = params;
}

void VulkanVideoDenoiser::cleanup() {
    if (!m_initialized) return;
    
    vkDeviceWaitIdle(m_device);
    
    // Cleanup in reverse order
    if (m_fence != VK_NULL_HANDLE) vkDestroyFence(m_device, m_fence, nullptr);
    if (m_commandBuffer != VK_NULL_HANDLE) vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_commandBuffer);
    
    if (m_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    if (m_computePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    if (m_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    if (m_descriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    
    if (m_stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(m_device, m_stagingBuffer, nullptr);
    if (m_stagingMemory != VK_NULL_HANDLE) vkFreeMemory(m_device, m_stagingMemory, nullptr);
    if (m_outputStagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(m_device, m_outputStagingBuffer, nullptr);
    if (m_outputStagingMemory != VK_NULL_HANDLE) vkFreeMemory(m_device, m_outputStagingMemory, nullptr);
    
    for (auto& fb : m_frameBuffers) {
        if (fb.imageView != VK_NULL_HANDLE) vkDestroyImageView(m_device, fb.imageView, nullptr);
        if (fb.image != VK_NULL_HANDLE) vkDestroyImage(m_device, fb.image, nullptr);
        if (fb.memory != VK_NULL_HANDLE) vkFreeMemory(m_device, fb.memory, nullptr);
    }
    
    if (m_outputBuffer.imageView != VK_NULL_HANDLE) vkDestroyImageView(m_device, m_outputBuffer.imageView, nullptr);
    if (m_outputBuffer.image != VK_NULL_HANDLE) vkDestroyImage(m_device, m_outputBuffer.image, nullptr);
    if (m_outputBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(m_device, m_outputBuffer.memory, nullptr);
    
    if (m_motionBuffer.imageView != VK_NULL_HANDLE) vkDestroyImageView(m_device, m_motionBuffer.imageView, nullptr);
    if (m_motionBuffer.image != VK_NULL_HANDLE) vkDestroyImage(m_device, m_motionBuffer.image, nullptr);
    if (m_motionBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(m_device, m_motionBuffer.memory, nullptr);
    
    if (m_commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    if (m_device != VK_NULL_HANDLE) vkDestroyDevice(m_device, nullptr);
    if (m_instance != VK_NULL_HANDLE) vkDestroyInstance(m_instance, nullptr);
    
    m_initialized = false;
}

} // namespace vk_denoise
