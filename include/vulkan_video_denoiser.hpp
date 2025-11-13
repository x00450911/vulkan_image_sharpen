#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <array>
#include <string>
#include <cstdint>

namespace vk_denoise {

// Frame buffer structure
struct FrameBuffer {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView imageView;
    uint32_t width;
    uint32_t height;
};

// Motion vector buffer
struct MotionVectorBuffer {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView imageView;
};

// Denoiser parameters
struct DenoiseParams {
    float temporalWeight = 0.7f;      // 0.0 - 1.0, higher = more temporal filtering
    float spatialSigma = 1.5f;         // Spatial Gaussian sigma
    float colorSigma = 0.15f;          // Color difference sigma
    bool useMotionCompensation = false; // Enable motion compensation
    float noiseThreshold = 0.1f;       // Noise detection threshold
    int frameCount = 4;                // Number of frames to use (1-4)
    float adaptiveStrength = 0.5f;     // Adaptive filtering strength
    int kernelSize = 5;                // Spatial kernel size (3, 5, 7)
};

// Push constants structure matching shader
struct PushConstants {
    float temporalWeight;
    float spatialSigma;
    float colorSigma;
    int useMotionCompensation;
    float noiseThreshold;
    int frameCount;
    float adaptiveStrength;
    int kernelSize;
};

class VulkanVideoDenoiser {
public:
    VulkanVideoDenoiser();
    ~VulkanVideoDenoiser();

    // Initialize Vulkan and create resources
    bool initialize(uint32_t width, uint32_t height, const DenoiseParams& params);
    
    // Process a single frame
    bool processFrame(const uint8_t* inputData, uint8_t* outputData);
    
    // Process frame with motion vectors
    bool processFrameWithMotion(const uint8_t* inputData, 
                               const float* motionVectors,
                               uint8_t* outputData);
    
    // Update denoising parameters
    void updateParameters(const DenoiseParams& params);
    
    // Cleanup resources
    void cleanup();
    
    // Get frame dimensions
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    // Vulkan initialization
    bool createInstance();
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    bool createCommandPool();
    bool createDescriptorSetLayout();
    bool createPipelineLayout();
    bool createComputePipeline();
    bool createDescriptorPool();
    
    // Resource management
    bool createFrameBuffers();
    bool createMotionVectorBuffer();
    bool createDescriptorSets();
    bool allocateCommandBuffers();
    
    // Helper functions
    bool createImage(uint32_t width, uint32_t height, VkFormat format,
                    VkImageUsageFlags usage, VkImage& image,
                    VkDeviceMemory& memory);
    bool createImageView(VkImage image, VkFormat format, VkImageView& imageView);
    bool transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             VkCommandBuffer cmdBuffer);
    bool copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    bool copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkBuffer& buffer,
                     VkDeviceMemory& memory);
    
    VkShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<char> readShaderFile(const std::string& filename);
    
    // Update frame history
    void rotateFrameBuffers();
    
    // Execute compute shader
    bool executeCompute();

private:
    // Vulkan core objects
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    uint32_t m_queueFamilyIndex = 0;
    
    // Pipeline objects
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    
    // Command buffers
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    
    // Frame buffers (ring buffer for temporal filtering)
    static constexpr int MAX_FRAME_HISTORY = 4;
    std::array<FrameBuffer, MAX_FRAME_HISTORY> m_frameBuffers;
    FrameBuffer m_outputBuffer;
    int m_currentFrameIndex = 0;
    
    // Motion vector buffer
    MotionVectorBuffer m_motionBuffer;
    
    // Staging buffers for CPU-GPU transfers
    VkBuffer m_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_stagingMemory = VK_NULL_HANDLE;
    VkBuffer m_outputStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_outputStagingMemory = VK_NULL_HANDLE;
    
    // Frame dimensions
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    
    // Denoising parameters
    DenoiseParams m_params;
    
    // State
    bool m_initialized = false;
    int m_framesProcessed = 0;
};

// Utility class for automatic Vulkan resource cleanup
template<typename T, typename Deleter>
class VulkanResource {
public:
    VulkanResource(T resource, Deleter deleter)
        : m_resource(resource), m_deleter(deleter) {}
    
    ~VulkanResource() {
        if (m_resource != VK_NULL_HANDLE) {
            m_deleter(m_resource);
        }
    }
    
    VulkanResource(const VulkanResource&) = delete;
    VulkanResource& operator=(const VulkanResource&) = delete;
    
    VulkanResource(VulkanResource&& other) noexcept
        : m_resource(other.m_resource), m_deleter(std::move(other.m_deleter)) {
        other.m_resource = VK_NULL_HANDLE;
    }
    
    T get() const { return m_resource; }
    
private:
    T m_resource;
    Deleter m_deleter;
};

} // namespace vk_denoise
