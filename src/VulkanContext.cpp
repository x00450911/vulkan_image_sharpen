#include "VulkanContext.h"

#include <android/log.h>
#include <array>
#include <cstring>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace video::processing {

namespace {

constexpr const char* kLogTag = "VulkanContext";

const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> kInstanceExtensions = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
    VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
    VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
    VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
};

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                             void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        __android_log_print(ANDROID_LOG_WARN, kLogTag, "Validation: %s", callbackData->pMessage);
    }
    return VK_FALSE;
}

bool checkValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
    for (const char* layerName : kValidationLayers) {
        bool found = false;
        for (const auto& layer : layers) {
            if (std::strcmp(layer.layerName, layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

std::vector<const char*> collectInstanceExtensions(bool enableValidation) {
    auto extensions = kInstanceExtensions;
    if (enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;

    bool complete() const { return computeFamily.has_value(); }
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, families.data());
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.computeFamily = i;
            break;
        }
    }
    if (!indices.computeFamily.has_value()) {
        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                indices.computeFamily = i;
                break;
            }
        }
    }
    return indices;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> available(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, available.data());

    std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const auto& ext : available) {
        required.erase(ext.extensionName);
    }
    return required.empty();
}

bool isDeviceSuitable(VkPhysicalDevice device) {
    const QueueFamilyIndices indices = findQueueFamilies(device);
    if (!indices.complete()) {
        return false;
    }
    if (!checkDeviceExtensionSupport(device)) {
        return false;
    }
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);
    if (!features.shaderStorageImageExtendedFormats) {
        return false;
    }
    return true;
}

PFN_vkCreateDebugUtilsMessengerEXT loadCreateDebugMessenger(VkInstance instance) {
    return reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
}

PFN_vkDestroyDebugUtilsMessengerEXT loadDestroyDebugMessenger(VkInstance instance) {
    return reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
}

}  // namespace

std::shared_ptr<VulkanContext> VulkanContext::create(const CreateInfo& info) {
    return std::shared_ptr<VulkanContext>(new VulkanContext(info));
}

VulkanContext::VulkanContext(const CreateInfo& info) : enableValidation_(info.enableValidation) {
    if (enableValidation_ && !checkValidationLayerSupport()) {
        __android_log_print(ANDROID_LOG_WARN, kLogTag, "Validation layers not available");
        enableValidation_ = false;
    }
    createInstance(info);
    setupDebugMessenger();
    selectPhysicalDevice();
    createLogicalDevice();
    createCommandInfrastructure();
    createSamplerResources();
    createDescriptorPool();
    createPipelineCache();
}

VulkanContext::~VulkanContext() {
    if (device_ != VK_NULL_HANDLE) {
        if (pipelineCache_ != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        }
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }
        if (linearSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, linearSampler_, nullptr);
        }
        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (enableValidation_ && debugMessenger_ != VK_NULL_HANDLE) {
        auto destroyFn = loadDestroyDebugMessenger(instance_);
        if (destroyFn) {
            destroyFn(instance_, debugMessenger_, nullptr);
        }
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::createInstance(const CreateInfo& info) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "AdrenoVideoPipeline";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Custom";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    const auto extensions = collectInstanceExtensions(enableValidation_);

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (enableValidation_) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();

        VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = debugCallback;
        createInfo.pNext = &debugInfo;
    }

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));
}

void VulkanContext::setupDebugMessenger() {
    if (!enableValidation_) {
        return;
    }
    auto createFn = loadCreateDebugMessenger(instance_);
    if (!createFn) {
        __android_log_print(ANDROID_LOG_WARN, kLogTag, "Failed to load debug messenger function");
        return;
    }
    VkDebugUtilsMessengerCreateInfoEXT createInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    VK_CHECK(createFn(instance_, &createInfo, nullptr, &debugMessenger_));
}

void VulkanContext::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (const VkPhysicalDevice device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice_ = device;
            break;
        }
    }
    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("No suitable Vulkan device located");
    }

    vkGetPhysicalDeviceProperties(physicalDevice_, &properties_);
    vkGetPhysicalDeviceFeatures(physicalDevice_, &features_);
    features12_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &features12_;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &features2);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProps_);
}

void VulkanContext::createLogicalDevice() {
    const QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
    if (!indices.computeFamily.has_value()) {
        throw std::runtime_error("Missing compute queue family");
    }
    computeQueueFamilyIndex_ = indices.computeFamily.value();

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::set<uint32_t> uniqueFamilies = {computeQueueFamilyIndex_};
    const float priority = 1.0f;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        queueInfos.push_back(queueInfo);
    }

    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timelineFeatures.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceSamplerYcbcrConversionFeatures samplerYcbcrFeature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
    samplerYcbcrFeature.samplerYcbcrConversion = VK_TRUE;

    timelineFeatures.pNext = &samplerYcbcrFeature;
    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.pEnabledFeatures = &features_;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();
    createInfo.pNext = &timelineFeatures;

    if (enableValidation_) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    VK_CHECK(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_));
    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);
}

void VulkanContext::createCommandInfrastructure() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
}

void VulkanContext::createSamplerResources() {
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
    VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &linearSampler_));
}

void VulkanContext::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32},
    };

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 128;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));
}

void VulkanContext::createPipelineCache() {
    VkPipelineCacheCreateInfo cacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VK_CHECK(vkCreatePipelineCache(device_, &cacheInfo, nullptr, &pipelineCache_));
}

}  // namespace video::processing
