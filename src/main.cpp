#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define VK_CHECK(call)                                                                            \
    do {                                                                                          \
        VkResult _vk_result = (call);                                                             \
        if (_vk_result != VK_SUCCESS) {                                                           \
            throw std::runtime_error(std::string("Vulkan error at ") + __FILE__ + ":" +           \
                                     std::to_string(__LINE__) + " -> " + std::to_string(_vk_result)); \
        }                                                                                         \
    } while (0)

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels; // Stored as RGBA, row-major, top-left origin.
};

struct PushConstants {
    int32_t width = 0;
    int32_t height = 0;
    float strength = 0.65f;
    float edgeThreshold = 0.045f;
};

struct CommandLineOptions {
    std::string inputPath;
    std::string outputPath;
    std::string shaderPath = "shaders/edge_sharpen.comp.spv";
    float strength = 0.65f;
    float edgeThreshold = 0.045f;
    bool verbose = true;
};

bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(),
                      [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

CommandLineOptions parseCommandLine(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error("Usage: vulkan_edge_sharpen <input.tga> <output.tga> "
                                 "[--strength value] [--edge-threshold value] "
                                 "[--shader path] [--quiet]");
    }

    CommandLineOptions options;
    options.inputPath = argv[1];
    options.outputPath = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--strength" && i + 1 < argc) {
            options.strength = std::stof(argv[++i]);
        } else if (arg == "--edge-threshold" && i + 1 < argc) {
            options.edgeThreshold = std::stof(argv[++i]);
        } else if (arg == "--shader" && i + 1 < argc) {
            options.shaderPath = argv[++i];
        } else if (arg == "--quiet") {
            options.verbose = false;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!ends_with(options.inputPath, ".tga")) {
        throw std::runtime_error("Only uncompressed TGA files are supported for input.");
    }
    if (!ends_with(options.outputPath, ".tga")) {
        throw std::runtime_error("Only uncompressed TGA files are supported for output.");
    }
    if (options.strength < 0.0f) {
        throw std::runtime_error("Sharpen strength must be non-negative.");
    }
    if (options.edgeThreshold < 0.0f) {
        throw std::runtime_error("Edge threshold must be non-negative.");
    }

    return options;
}

Image loadTGA(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open TGA file: " + path);
    }

    std::array<uint8_t, 18> header{};
    file.read(reinterpret_cast<char*>(header.data()), header.size());
    if (!file) {
        throw std::runtime_error("Failed to read TGA header: " + path);
    }

    const uint8_t idLength = header[0];
    const uint8_t colorMapType = header[1];
    const uint8_t imageType = header[2];
    if (colorMapType != 0) {
        throw std::runtime_error("Color-mapped TGA files are not supported: " + path);
    }
    if (imageType != 2) {
        throw std::runtime_error("Only uncompressed true-color TGA files are supported: " + path);
    }

    const uint16_t width = static_cast<uint16_t>(header[12] | (header[13] << 8));
    const uint16_t height = static_cast<uint16_t>(header[14] | (header[15] << 8));
    const uint8_t bitsPerPixel = header[16];
    const uint8_t imageDescriptor = header[17];
    const bool flipVertically = ((imageDescriptor & 0x20u) == 0);

    if (bitsPerPixel != 24 && bitsPerPixel != 32) {
        throw std::runtime_error("Unsupported TGA pixel depth (expected 24 or 32): " + path);
    }

    if (idLength > 0) {
        file.seekg(idLength, std::ios::cur);
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t sourceBytesPerPixel = bitsPerPixel / 8;

    std::vector<uint8_t> source(pixelCount * sourceBytesPerPixel);
    file.read(reinterpret_cast<char*>(source.data()), static_cast<std::streamsize>(source.size()));
    if (!file) {
        throw std::runtime_error("Failed to read TGA pixel data: " + path);
    }

    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(pixelCount * 4u);

    auto convertPixel = [&](size_t srcIndex) {
        uint8_t b = source[srcIndex + 0];
        uint8_t g = source[srcIndex + 1];
        uint8_t r = source[srcIndex + 2];
        uint8_t a = (sourceBytesPerPixel == 4) ? source[srcIndex + 3] : 255u;
        return std::array<uint8_t, 4>{r, g, b, a};
    };

    for (uint32_t y = 0; y < height; ++y) {
        uint32_t srcY = flipVertically ? (height - 1u - y) : y;
        for (uint32_t x = 0; x < width; ++x) {
            size_t srcOffset = static_cast<size_t>(srcY) * width * sourceBytesPerPixel +
                               static_cast<size_t>(x) * sourceBytesPerPixel;
            auto rgba = convertPixel(srcOffset);
            size_t dstOffset = (static_cast<size_t>(y) * width + x) * 4u;
            image.pixels[dstOffset + 0] = rgba[0];
            image.pixels[dstOffset + 1] = rgba[1];
            image.pixels[dstOffset + 2] = rgba[2];
            image.pixels[dstOffset + 3] = rgba[3];
        }
    }

    return image;
}

void saveTGA(const std::string& path, const Image& image) {
    if (image.pixels.size() != static_cast<size_t>(image.width) * image.height * 4u) {
        throw std::runtime_error("Image pixel data size mismatch when saving: " + path);
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open TGA file for writing: " + path);
    }

    std::array<uint8_t, 18> header{};
    header[2] = 2; // uncompressed true-color
    header[12] = static_cast<uint8_t>(image.width & 0xFFu);
    header[13] = static_cast<uint8_t>((image.width >> 8) & 0xFFu);
    header[14] = static_cast<uint8_t>(image.height & 0xFFu);
    header[15] = static_cast<uint8_t>((image.height >> 8) & 0xFFu);
    header[16] = 32; // bits per pixel
    header[17] = 0x20; // origin at top-left

    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    if (!file) {
        throw std::runtime_error("Failed to write TGA header: " + path);
    }

    std::vector<uint8_t> buffer(image.pixels.size());
    for (size_t i = 0; i < image.pixels.size() / 4u; ++i) {
        buffer[i * 4u + 0] = image.pixels[i * 4u + 2]; // B
        buffer[i * 4u + 1] = image.pixels[i * 4u + 1]; // G
        buffer[i * 4u + 2] = image.pixels[i * 4u + 0]; // R
        buffer[i * 4u + 3] = image.pixels[i * 4u + 3]; // A
    }

    file.write(reinterpret_cast<const char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    if (!file) {
        throw std::runtime_error("Failed to write TGA pixel data: " + path);
    }
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    const std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        throw std::runtime_error("File is empty or inaccessible: " + path);
    }
    std::vector<char> buffer(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), fileSize);
    if (!file) {
        throw std::runtime_error("Failed to read file contents: " + path);
    }
    return buffer;
}

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t computeQueueIndex = 0;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
};

uint32_t findComputeQueueFamily(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            return i;
        }
    }
    throw std::runtime_error("Failed to locate a compute-capable queue family.");
}

uint32_t findMemoryType(const VulkanContext& context, uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < context.memoryProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) != 0 &&
            (context.memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable Vulkan memory type.");
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));
    return module;
}

Buffer createBuffer(const VulkanContext& context, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memoryProperties) {
    Buffer buffer{};
    buffer.device = context.device;
    buffer.size = size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(context.device, &bufferInfo, nullptr, &buffer.buffer));

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(context.device, buffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(context, memRequirements.memoryTypeBits, memoryProperties);

    VK_CHECK(vkAllocateMemory(context.device, &allocInfo, nullptr, &buffer.memory));
    VK_CHECK(vkBindBufferMemory(context.device, buffer.buffer, buffer.memory, 0));

    return buffer;
}

void destroyBuffer(Buffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(buffer.device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(buffer.device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.size = 0;
}

std::optional<uint32_t> findPortabilityEnumerationExtension() {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());

    for (const auto& ext : extensions) {
        if (std::string(ext.extensionName) == "VK_KHR_portability_enumeration") {
            return VK_KHR_PORTABILITY_ENUMERATION_BIT_KHR;
        }
    }
    return std::nullopt;
}

VulkanContext createContext(bool verbose) {
    VulkanContext context{};

    std::vector<const char*> enabledExtensions;
    auto portabilityFlag = findPortabilityEnumerationExtension();
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = nullptr;
    if (portabilityFlag.has_value()) {
        enabledExtensions.push_back("VK_KHR_portability_enumeration");
        instanceInfo.flags |= portabilityFlag.value();
    }
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    instanceInfo.ppEnabledExtensionNames = enabledExtensions.empty() ? nullptr : enabledExtensions.data();

    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &context.instance));

    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(context.instance, &deviceCount, nullptr));
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan physical devices detected.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(context.instance, &deviceCount, devices.data()));

    auto pickDevice = [&](const std::string& needle) -> std::optional<VkPhysicalDevice> {
        for (VkPhysicalDevice device : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(device, &props);
            std::string deviceName = props.deviceName;
            if (deviceName.find(needle) != std::string::npos) {
                return device;
            }
        }
        return std::nullopt;
    };

    context.physicalDevice = VK_NULL_HANDLE;
    if (auto preferred = pickDevice("Adreno (TM) 650"); preferred.has_value()) {
        context.physicalDevice = preferred.value();
    } else if (auto adreno = pickDevice("Adreno"); adreno.has_value()) {
        context.physicalDevice = adreno.value();
    } else {
        context.physicalDevice = devices[0];
        if (verbose) {
            std::cerr << "Warning: Adreno GPU not found. Falling back to first available device.\n";
        }
    }

    vkGetPhysicalDeviceProperties(context.physicalDevice, &context.properties);
    vkGetPhysicalDeviceMemoryProperties(context.physicalDevice, &context.memoryProperties);

    context.computeQueueIndex = findComputeQueueFamily(context.physicalDevice);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = context.computeQueueIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;

    VK_CHECK(vkCreateDevice(context.physicalDevice, &deviceInfo, nullptr, &context.device));

    vkGetDeviceQueue(context.device, context.computeQueueIndex, 0, &context.computeQueue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = context.computeQueueIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(context.device, &poolInfo, nullptr, &context.commandPool));

    if (verbose) {
        std::cout << "Using device: " << context.properties.deviceName << "\n";
        std::cout << "API version: " << VK_VERSION_MAJOR(context.properties.apiVersion) << "."
                  << VK_VERSION_MINOR(context.properties.apiVersion) << "."
                  << VK_VERSION_PATCH(context.properties.apiVersion) << "\n";
    }

    return context;
}

void destroyContext(VulkanContext& context) {
    if (context.commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context.device, context.commandPool, nullptr);
        context.commandPool = VK_NULL_HANDLE;
    }
    if (context.device != VK_NULL_HANDLE) {
        vkDestroyDevice(context.device, nullptr);
        context.device = VK_NULL_HANDLE;
    }
    if (context.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context.instance, nullptr);
        context.instance = VK_NULL_HANDLE;
    }
}

struct VulkanSharpenRuntime {
    VulkanContext context;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
};

void destroyRuntime(VulkanSharpenRuntime& runtime) {
    if (runtime.context.device != VK_NULL_HANDLE) {
        if (runtime.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(runtime.context.device, runtime.pipeline, nullptr);
            runtime.pipeline = VK_NULL_HANDLE;
        }
        if (runtime.pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(runtime.context.device, runtime.pipelineLayout, nullptr);
            runtime.pipelineLayout = VK_NULL_HANDLE;
        }
        if (runtime.descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(runtime.context.device, runtime.descriptorPool, nullptr);
            runtime.descriptorPool = VK_NULL_HANDLE;
        }
        if (runtime.descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(runtime.context.device, runtime.descriptorSetLayout, nullptr);
            runtime.descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (runtime.shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(runtime.context.device, runtime.shaderModule, nullptr);
            runtime.shaderModule = VK_NULL_HANDLE;
        }
    }
    destroyContext(runtime.context);
}

VulkanSharpenRuntime createRuntime(const CommandLineOptions& options) {
    VulkanSharpenRuntime runtime{};
    runtime.context = createContext(options.verbose);

    auto shaderCode = readFile(options.shaderPath);
    runtime.shaderModule = createShaderModule(runtime.context.device, shaderCode);

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    setLayoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(runtime.context.device, &setLayoutInfo, nullptr,
                                         &runtime.descriptorSetLayout));

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &runtime.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VK_CHECK(vkCreatePipelineLayout(runtime.context.device, &pipelineLayoutInfo, nullptr,
                                    &runtime.pipelineLayout));

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = runtime.pipelineLayout;
    pipelineInfo.flags = 0;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = runtime.shaderModule;
    pipelineInfo.stage.pName = "main";

    VK_CHECK(vkCreateComputePipelines(runtime.context.device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                      nullptr, &runtime.pipeline));

    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VK_CHECK(vkCreateDescriptorPool(runtime.context.device, &poolInfo, nullptr,
                                    &runtime.descriptorPool));

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = runtime.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &runtime.descriptorSetLayout;

    VK_CHECK(vkAllocateDescriptorSets(runtime.context.device, &allocInfo, &runtime.descriptorSet));

    return runtime;
}

void dispatchSharpen(VulkanSharpenRuntime& runtime, const Image& inputImage,
                     Image& outputImage, const CommandLineOptions& options) {
    const auto pixelCount = static_cast<size_t>(inputImage.width) * inputImage.height;
    std::vector<float> inputData(pixelCount * 4u);
    std::vector<float> outputData(pixelCount * 4u, 0.0f);

    for (size_t i = 0; i < pixelCount; ++i) {
        inputData[i * 4u + 0] = static_cast<float>(inputImage.pixels[i * 4u + 0]) / 255.0f;
        inputData[i * 4u + 1] = static_cast<float>(inputImage.pixels[i * 4u + 1]) / 255.0f;
        inputData[i * 4u + 2] = static_cast<float>(inputImage.pixels[i * 4u + 2]) / 255.0f;
        inputData[i * 4u + 3] = static_cast<float>(inputImage.pixels[i * 4u + 3]) / 255.0f;
    }

    const VkDeviceSize bufferSize = sizeof(float) * inputData.size();
    Buffer inputBuffer = createBuffer(runtime.context, bufferSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Buffer outputBuffer = createBuffer(runtime.context, bufferSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(runtime.context.device, inputBuffer.memory, 0, bufferSize, 0, &mapped));
    std::memcpy(mapped, inputData.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(runtime.context.device, inputBuffer.memory);

    VK_CHECK(vkMapMemory(runtime.context.device, outputBuffer.memory, 0, bufferSize, 0, &mapped));
    std::memset(mapped, 0, static_cast<size_t>(bufferSize));
    vkUnmapMemory(runtime.context.device, outputBuffer.memory);

    VkDescriptorBufferInfo inputInfo{};
    inputInfo.buffer = inputBuffer.buffer;
    inputInfo.offset = 0;
    inputInfo.range = bufferSize;

    VkDescriptorBufferInfo outputInfo{};
    outputInfo.buffer = outputBuffer.buffer;
    outputInfo.offset = 0;
    outputInfo.range = bufferSize;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = runtime.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &inputInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = runtime.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &outputInfo;

    vkUpdateDescriptorSets(runtime.context.device,
                           static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = runtime.context.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(runtime.context.device, &cmdAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, runtime.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            runtime.pipelineLayout, 0, 1, &runtime.descriptorSet, 0, nullptr);

    PushConstants constants{};
    constants.width = static_cast<int32_t>(inputImage.width);
    constants.height = static_cast<int32_t>(inputImage.height);
    constants.strength = options.strength;
    constants.edgeThreshold = options.edgeThreshold;
    vkCmdPushConstants(commandBuffer, runtime.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &constants);

    const uint32_t groupSizeX = 16;
    const uint32_t groupSizeY = 16;
    const uint32_t groupCountX = (inputImage.width + groupSizeX - 1u) / groupSizeX;
    const uint32_t groupCountY = (inputImage.height + groupSizeY - 1u) / groupSizeY;

    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(runtime.context.computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(runtime.context.computeQueue));

    VK_CHECK(vkFreeCommandBuffers(runtime.context.device, runtime.context.commandPool, 1,
                                  &commandBuffer));

    VK_CHECK(vkMapMemory(runtime.context.device, outputBuffer.memory, 0, bufferSize, 0, &mapped));
    std::memcpy(outputData.data(), mapped, static_cast<size_t>(bufferSize));
    vkUnmapMemory(runtime.context.device, outputBuffer.memory);

    destroyBuffer(inputBuffer);
    destroyBuffer(outputBuffer);

    outputImage.width = inputImage.width;
    outputImage.height = inputImage.height;
    outputImage.pixels.resize(pixelCount * 4u);

    for (size_t i = 0; i < pixelCount; ++i) {
        float r = std::clamp(outputData[i * 4u + 0], 0.0f, 1.0f);
        float g = std::clamp(outputData[i * 4u + 1], 0.0f, 1.0f);
        float b = std::clamp(outputData[i * 4u + 2], 0.0f, 1.0f);
        float a = std::clamp(outputData[i * 4u + 3], 0.0f, 1.0f);
        outputImage.pixels[i * 4u + 0] = static_cast<uint8_t>(std::round(r * 255.0f));
        outputImage.pixels[i * 4u + 1] = static_cast<uint8_t>(std::round(g * 255.0f));
        outputImage.pixels[i * 4u + 2] = static_cast<uint8_t>(std::round(b * 255.0f));
        outputImage.pixels[i * 4u + 3] = static_cast<uint8_t>(std::round(a * 255.0f));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CommandLineOptions options = parseCommandLine(argc, argv);
        const Image inputImage = loadTGA(options.inputPath);

        Image outputImage;
        VulkanSharpenRuntime runtime = createRuntime(options);
        dispatchSharpen(runtime, inputImage, outputImage, options);
        saveTGA(options.outputPath, outputImage);

        if (options.verbose) {
            std::cout << "Sharpened image written to " << options.outputPath << "\n";
        }

        destroyRuntime(runtime);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
