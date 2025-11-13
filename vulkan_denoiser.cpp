#include "vulkan_denoiser.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <array>

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> VulkanDenoiser::validationLayers_ = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> VulkanDenoiser::deviceExtensions_ = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// Debug callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

VulkanDenoiser::VulkanDenoiser(uint32_t width, uint32_t height, uint32_t numFrames)
    : instance_(VK_NULL_HANDLE)
    , physicalDevice_(VK_NULL_HANDLE)
    , device_(VK_NULL_HANDLE)
    , computeQueue_(VK_NULL_HANDLE)
    , computeQueueFamilyIndex_(0)
    , descriptorSetLayout_(VK_NULL_HANDLE)
    , pipelineLayout_(VK_NULL_HANDLE)
    , computePipeline_(VK_NULL_HANDLE)
    , descriptorPool_(VK_NULL_HANDLE)
    , commandPool_(VK_NULL_HANDLE)
    , currentFrameIndex_(0)
    , width_(width)
    , height_(height)
    , numFrames_(numFrames)
    , denoisingStrength_(0.5f)
    , temporalWeight_(0.7f)
    , debugMessenger_(VK_NULL_HANDLE)
    , enableValidationLayers_(enableValidationLayers) {
}

VulkanDenoiser::~VulkanDenoiser() {
    Cleanup();
}

bool VulkanDenoiser::Initialize() {
    if (!CreateInstance()) return false;
    if (!SelectPhysicalDevice()) return false;
    if (!CreateLogicalDevice()) return false;
    if (!CreateCommandPool()) return false;
    if (!CreateDescriptorSetLayout()) return false;
    if (!CreateComputePipeline()) return false;
    if (!CreateDescriptorPool()) return false;
    if (!CreateDescriptorSets()) return false;
    if (!CreateFrameBuffers()) return false;
    
    // Create synchronization objects
    computeFences_.resize(numFrames_);
    computeSemaphores_.resize(numFrames_);
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    for (uint32_t i = 0; i < numFrames_; i++) {
        if (vkCreateFence(device_, &fenceInfo, nullptr, &computeFences_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create compute fence" << std::endl;
            return false;
        }
        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &computeSemaphores_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create compute semaphore" << std::endl;
            return false;
        }
    }
    
    // Allocate command buffers
    commandBuffers_.resize(numFrames_);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = numFrames_;
    
    if (vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffers" << std::endl;
        return false;
    }
    
    return true;
}

void VulkanDenoiser::Cleanup() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        
        DestroyFrameBuffers();
        
        for (auto& fence : computeFences_) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device_, fence, nullptr);
            }
        }
        
        for (auto& semaphore : computeSemaphores_) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore, nullptr);
            }
        }
        
        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
        
        if (computePipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, computePipeline_, nullptr);
        }
        
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        }
        
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }
        
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        }
        
        vkDestroyDevice(device_, nullptr);
    }
    
    if (enableValidationLayers_ && debugMessenger_ != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, debugMessenger_, nullptr);
        }
    }
    
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

bool VulkanDenoiser::CreateInstance() {
    if (enableValidationLayers_) {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        
        for (const char* layerName : validationLayers_) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                std::cerr << "Validation layer requested but not available: " << layerName << std::endl;
                return false;
            }
        }
    }
    
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Denoiser";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = nullptr;
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    
    if (enableValidationLayers_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers_) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers_.size());
        createInfo.ppEnabledLayerNames = validationLayers_.data();
        
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }
    
    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
    if (enableValidationLayers_) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, &debugCreateInfo, nullptr, &debugMessenger_);
        }
    }
    
    return true;
}

bool VulkanDenoiser::SelectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());
    
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice_ = device;
                computeQueueFamilyIndex_ = i;
                std::cout << "Selected device: " << deviceProperties.deviceName << std::endl;
                return true;
            }
        }
    }
    
    std::cerr << "Failed to find a suitable GPU" << std::endl;
    return false;
}

bool VulkanDenoiser::CreateLogicalDevice() {
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    VkPhysicalDeviceFeatures deviceFeatures{};
    
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;
    
    if (enableValidationLayers_) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers_.size());
        createInfo.ppEnabledLayerNames = validationLayers_.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }
    
    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device" << std::endl;
        return false;
    }
    
    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);
    return true;
}

bool VulkanDenoiser::CreateCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    
    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanDenoiser::CreateDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    
    // Current frame (input)
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].pImmutableSamplers = nullptr;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Previous frames (input)
    bindings[1].binding = 1;
    bindings[1].descriptorCount = numFrames_ - 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].pImmutableSamplers = nullptr;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Output frame
    bindings[2].binding = 2;
    bindings[2].descriptorCount = 1;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].pImmutableSamplers = nullptr;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Uniform buffer for parameters
    bindings[3].binding = 3;
    bindings[3].descriptorCount = 1;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].pImmutableSamplers = nullptr;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanDenoiser::CreateComputePipeline() {
    // Load compute shader
    VkShaderModule computeShaderModule;
    if (!LoadShaderModule("denoise.comp.spv", computeShaderModule)) {
        std::cerr << "Failed to load compute shader" << std::endl;
        return false;
    }
    
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = computeShaderModule;
    shaderStageInfo.pName = "main";
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    
    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        vkDestroyShaderModule(device_, computeShaderModule, nullptr);
        return false;
    }
    
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout_;
    
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline_) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        vkDestroyShaderModule(device_, computeShaderModule, nullptr);
        return false;
    }
    
    vkDestroyShaderModule(device_, computeShaderModule, nullptr);
    return true;
}

bool VulkanDenoiser::CreateDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = numFrames_ * 2; // Input + output images
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = numFrames_;
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = numFrames_;
    
    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }
    
    return true;
}

bool VulkanDenoiser::CreateDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(numFrames_, descriptorSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = numFrames_;
    allocInfo.pSetLayouts = layouts.data();
    
    descriptorSets_.resize(numFrames_);
    if (vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor sets" << std::endl;
        return false;
    }
    
    // Update descriptor sets will be done in ProcessFrame
    return true;
}

bool VulkanDenoiser::CreateFrameBuffers() {
    frameBuffers_.resize(numFrames_);
    
    for (uint32_t i = 0; i < numFrames_; i++) {
        FrameBuffer& fb = frameBuffers_[i];
        fb.width = width_;
        fb.height = height_;
        fb.frameIndex = i;
        
        // Create input image for frame storage
        if (!CreateImage(width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        fb.image, fb.memory)) {
            return false;
        }
        
        fb.imageView = CreateImageView(fb.image, VK_FORMAT_R8G8B8A8_UNORM);
        
        // Create output image for denoised result
        if (!CreateImage(width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        fb.outputImage, fb.outputMemory)) {
            return false;
        }
        
        fb.outputImageView = CreateImageView(fb.outputImage, VK_FORMAT_R8G8B8A8_UNORM);
        
        // Create staging buffer for CPU-GPU transfers
        VkDeviceSize bufferSize = width_ * height_ * 4; // RGBA
        if (!CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fb.buffer, fb.bufferMemory)) {
            return false;
        }
    }
    
    return true;
}

void VulkanDenoiser::DestroyFrameBuffers() {
    for (auto& fb : frameBuffers_) {
        if (fb.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, fb.imageView, nullptr);
        }
        if (fb.image != VK_NULL_HANDLE) {
            vkDestroyImage(device_, fb.image, nullptr);
        }
        if (fb.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, fb.memory, nullptr);
        }
        if (fb.outputImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, fb.outputImageView, nullptr);
        }
        if (fb.outputImage != VK_NULL_HANDLE) {
            vkDestroyImage(device_, fb.outputImage, nullptr);
        }
        if (fb.outputMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, fb.outputMemory, nullptr);
        }
        if (fb.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, fb.buffer, nullptr);
        }
        if (fb.bufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, fb.bufferMemory, nullptr);
        }
    }
    frameBuffers_.clear();
}

bool VulkanDenoiser::ProcessFrame(const uint8_t* inputData, uint8_t* outputData, uint32_t frameIndex) {
    FrameBuffer* currentFrame = GetFrameBuffer(frameIndex % numFrames_);
    
    // Copy input data to staging buffer
    void* data;
    vkMapMemory(device_, currentFrame->bufferMemory, 0, width_ * height_ * 4, 0, &data);
    memcpy(data, inputData, width_ * height_ * 4);
    vkUnmapMemory(device_, currentFrame->bufferMemory);
    
    // Transition image to transfer destination
    TransitionImageLayout(currentFrame->image, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    
    // Copy buffer to image
    CopyBufferToImage(currentFrame->buffer, currentFrame->image, width_, height_);
    
    // Transition input image to shader read
    TransitionImageLayout(currentFrame->image, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    
    // Initialize output image layout
    TransitionImageLayout(currentFrame->outputImage, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    
    // Wait for previous frame to complete
    vkWaitForFences(device_, 1, &computeFences_[frameIndex % numFrames_], VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &computeFences_[frameIndex % numFrames_]);
    
    // Update descriptor set
    std::vector<VkDescriptorImageInfo> imageInfos(numFrames_);
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    
    VkDescriptorImageInfo currentImageInfo{};
    currentImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    currentImageInfo.imageView = currentFrame->imageView;
    
    VkWriteDescriptorSet currentWrite{};
    currentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    currentWrite.dstSet = descriptorSets_[frameIndex % numFrames_];
    currentWrite.dstBinding = 0;
    currentWrite.dstArrayElement = 0;
    currentWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    currentWrite.descriptorCount = 1;
    currentWrite.pImageInfo = &currentImageInfo;
    descriptorWrites.push_back(currentWrite);
    
    // Previous frames
    std::vector<VkDescriptorImageInfo> prevImageInfos;
    for (uint32_t i = 1; i < numFrames_; i++) {
        uint32_t prevIndex = (frameIndex + numFrames_ - i) % numFrames_;
        FrameBuffer* prevFrame = GetFrameBuffer(prevIndex);
        
        VkDescriptorImageInfo prevImageInfo{};
        prevImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        prevImageInfo.imageView = prevFrame->imageView;
        prevImageInfos.push_back(prevImageInfo);
    }
    
    if (!prevImageInfos.empty()) {
        VkWriteDescriptorSet prevWrite{};
        prevWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        prevWrite.dstSet = descriptorSets_[frameIndex % numFrames_];
        prevWrite.dstBinding = 1;
        prevWrite.dstArrayElement = 0;
        prevWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        prevWrite.descriptorCount = static_cast<uint32_t>(prevImageInfos.size());
        prevWrite.pImageInfo = prevImageInfos.data();
        descriptorWrites.push_back(prevWrite);
    }
    
    // Output image (separate output buffer)
    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputImageInfo.imageView = currentFrame->outputImageView;
    
    VkWriteDescriptorSet outputWrite{};
    outputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    outputWrite.dstSet = descriptorSets_[frameIndex % numFrames_];
    outputWrite.dstBinding = 2;
    outputWrite.dstArrayElement = 0;
    outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputWrite.descriptorCount = 1;
    outputWrite.pImageInfo = &outputImageInfo;
    descriptorWrites.push_back(outputWrite);
    
    // Uniform buffer for parameters
    struct DenoiseParams {
        float denoisingStrength;
        float temporalWeight;
        uint32_t numFrames;
        uint32_t width;
        uint32_t height;
    } params;
    
    params.denoisingStrength = denoisingStrength_;
    params.temporalWeight = temporalWeight_;
    params.numFrames = numFrames_;
    params.width = width_;
    params.height = height_;
    
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    VkDeviceSize bufferSize = sizeof(DenoiseParams);
    if (!CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffer, uniformBufferMemory)) {
        return false;
    }
    
    void* uniformData;
    vkMapMemory(device_, uniformBufferMemory, 0, bufferSize, 0, &uniformData);
    memcpy(uniformData, &params, bufferSize);
    vkUnmapMemory(device_, uniformBufferMemory);
    
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = bufferSize;
    
    VkWriteDescriptorSet uniformWrite{};
    uniformWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uniformWrite.dstSet = descriptorSets_[frameIndex % numFrames_];
    uniformWrite.dstBinding = 3;
    uniformWrite.dstArrayElement = 0;
    uniformWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformWrite.descriptorCount = 1;
    uniformWrite.pBufferInfo = &bufferInfo;
    descriptorWrites.push_back(uniformWrite);
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(descriptorWrites.size()),
                          descriptorWrites.data(), 0, nullptr);
    
    // Record command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    if (vkBeginCommandBuffer(commandBuffers_[frameIndex % numFrames_], &beginInfo) != VK_SUCCESS) {
        std::cerr << "Failed to begin recording command buffer" << std::endl;
        vkDestroyBuffer(device_, uniformBuffer, nullptr);
        vkFreeMemory(device_, uniformBufferMemory, nullptr);
        return false;
    }
    
    RecordDenoisingCommands(commandBuffers_[frameIndex % numFrames_], frameIndex);
    
    if (vkEndCommandBuffer(commandBuffers_[frameIndex % numFrames_]) != VK_SUCCESS) {
        std::cerr << "Failed to record command buffer" << std::endl;
        vkDestroyBuffer(device_, uniformBuffer, nullptr);
        vkFreeMemory(device_, uniformBufferMemory, nullptr);
        return false;
    }
    
    // Submit compute work
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[frameIndex % numFrames_];
    
    if (vkQueueSubmit(computeQueue_, 1, &submitInfo, computeFences_[frameIndex % numFrames_]) != VK_SUCCESS) {
        std::cerr << "Failed to submit compute queue" << std::endl;
        vkDestroyBuffer(device_, uniformBuffer, nullptr);
        vkFreeMemory(device_, uniformBufferMemory, nullptr);
        return false;
    }
    
    // Wait for completion
    vkWaitForFences(device_, 1, &computeFences_[frameIndex % numFrames_], VK_TRUE, UINT64_MAX);
    
    // Transition output image and copy result back
    TransitionImageLayout(currentFrame->outputImage, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    CopyImageToBuffer(currentFrame->outputImage, currentFrame->buffer, width_, height_);
    
    // Read back result
    vkMapMemory(device_, currentFrame->bufferMemory, 0, width_ * height_ * 4, 0, &data);
    memcpy(outputData, data, width_ * height_ * 4);
    vkUnmapMemory(device_, currentFrame->bufferMemory);
    
    // Cleanup uniform buffer
    vkDestroyBuffer(device_, uniformBuffer, nullptr);
    vkFreeMemory(device_, uniformBufferMemory, nullptr);
    
    currentFrameIndex_ = (currentFrameIndex_ + 1) % numFrames_;
    return true;
}

void VulkanDenoiser::RecordDenoisingCommands(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineLayout_, 0, 1, &descriptorSets_[frameIndex % numFrames_], 0, nullptr);
    
    // Dispatch compute shader
    uint32_t groupCountX = (width_ + 15) / 16;
    uint32_t groupCountY = (height_ + 15) / 16;
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
}

FrameBuffer* VulkanDenoiser::GetFrameBuffer(uint32_t index) {
    if (index < frameBuffers_.size()) {
        return &frameBuffers_[index];
    }
    return nullptr;
}

bool VulkanDenoiser::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags properties, VkBuffer& buffer,
                                  VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device_, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }
    
    vkBindBufferMemory(device_, buffer, bufferMemory, 0);
    return true;
}

bool VulkanDenoiser::CreateImage(uint32_t width, uint32_t height, VkFormat format,
                                 VkImageTiling tiling, VkImageUsageFlags usage,
                                 VkMemoryPropertyFlags properties, VkImage& image,
                                 VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateImage(device_, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        return false;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device_, image, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        vkDestroyImage(device_, image, nullptr);
        return false;
    }
    
    vkBindImageMemory(device_, image, imageMemory, 0);
    return true;
}

VkImageView VulkanDenoiser::CreateImageView(VkImage image, VkFormat format) {
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
    
    VkImageView imageView;
    if (vkCreateImageView(device_, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    
    return imageView;
}

uint32_t VulkanDenoiser::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type!");
}

bool VulkanDenoiser::LoadShaderModule(const std::string& filename, VkShaderModule& shaderModule) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return false;
    }
    
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
    
    if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module" << std::endl;
        return false;
    }
    
    return true;
}

void VulkanDenoiser::CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool_;
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
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
    
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue_);
    
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
}

void VulkanDenoiser::CopyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool_;
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
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
    
    vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue_);
    
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
}

void VulkanDenoiser::TransitionImageLayout(VkImage image, VkFormat format,
                                           VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool_;
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
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
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        throw std::invalid_argument("Unsupported layout transition!");
    }
    
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue_);
    
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
}
