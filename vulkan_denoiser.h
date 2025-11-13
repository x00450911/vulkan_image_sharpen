#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

struct FrameBuffer {
    VkImage image;              // Input frame image
    VkImageView imageView;      // Input frame image view
    VkDeviceMemory memory;      // Input frame memory
    VkImage outputImage;        // Output denoised image
    VkImageView outputImageView;// Output image view
    VkDeviceMemory outputMemory;// Output image memory
    VkBuffer buffer;            // Staging buffer for CPU-GPU transfer
    VkDeviceMemory bufferMemory;// Staging buffer memory
    uint32_t width;
    uint32_t height;
    uint32_t frameIndex;
};

class VulkanDenoiser {
public:
    VulkanDenoiser(uint32_t width, uint32_t height, uint32_t numFrames = 5);
    ~VulkanDenoiser();

    bool Initialize();
    void Cleanup();
    
    // Process a frame and return denoised result
    bool ProcessFrame(const uint8_t* inputData, uint8_t* outputData, uint32_t frameIndex);
    
    // Set denoising parameters
    void SetDenoisingStrength(float strength) { denoisingStrength_ = strength; }
    void SetTemporalWeight(float weight) { temporalWeight_ = weight; }
    
    uint32_t GetWidth() const { return width_; }
    uint32_t GetHeight() const { return height_; }

private:
    // Vulkan initialization
    bool CreateInstance();
    bool SelectPhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateCommandPool();
    bool CreateDescriptorSetLayout();
    bool CreateComputePipeline();
    bool CreateDescriptorPool();
    bool CreateDescriptorSets();
    
    // Frame management
    bool CreateFrameBuffers();
    void DestroyFrameBuffers();
    FrameBuffer* GetFrameBuffer(uint32_t index);
    
    // Memory management
    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                     VkMemoryPropertyFlags properties, VkBuffer& buffer, 
                     VkDeviceMemory& bufferMemory);
    bool CreateImage(uint32_t width, uint32_t height, VkFormat format,
                    VkImageTiling tiling, VkImageUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkImage& image,
                    VkDeviceMemory& imageMemory);
    VkImageView CreateImageView(VkImage image, VkFormat format);
    
    // Utility functions
    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool LoadShaderModule(const std::string& filename, VkShaderModule& shaderModule);
    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void CopyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);
    void TransitionImageLayout(VkImage image, VkFormat format, 
                              VkImageLayout oldLayout, VkImageLayout newLayout);
    
    // Command buffer recording
    void RecordDenoisingCommands(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    
    // Instance and device
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    VkQueue computeQueue_;
    uint32_t computeQueueFamilyIndex_;
    
    // Pipeline
    VkDescriptorSetLayout descriptorSetLayout_;
    VkPipelineLayout pipelineLayout_;
    VkPipeline computePipeline_;
    VkDescriptorPool descriptorPool_;
    std::vector<VkDescriptorSet> descriptorSets_;
    
    // Command pool and buffers
    VkCommandPool commandPool_;
    std::vector<VkCommandBuffer> commandBuffers_;
    
    // Frame buffers
    std::vector<FrameBuffer> frameBuffers_;
    uint32_t currentFrameIndex_;
    
    // Parameters
    uint32_t width_;
    uint32_t height_;
    uint32_t numFrames_;
    float denoisingStrength_;
    float temporalWeight_;
    
    // Synchronization
    std::vector<VkFence> computeFences_;
    std::vector<VkSemaphore> computeSemaphores_;
    
    // Debug
    VkDebugUtilsMessengerEXT debugMessenger_;
    bool enableValidationLayers_;
    
    static const std::vector<const char*> validationLayers_;
    static const std::vector<const char*> deviceExtensions_;
};
