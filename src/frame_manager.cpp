#include "frame_manager.h"
#include <stdexcept>
#include <cstring>

FrameManager::FrameManager(VulkanContext* context)
    : context_(context)
    , width_(0)
    , height_(0)
{
}

FrameManager::~FrameManager() {
    cleanup();
}

bool FrameManager::initialize(uint32_t width, uint32_t height, uint32_t frameCount) {
    width_ = width;
    height_ = height;
    frames_.resize(frameCount);

    VkDeviceSize imageSize = width * height * 4; // RGBA8

    for (uint32_t i = 0; i < frameCount; i++) {
        FrameBuffer& frame = frames_[i];
        frame.image = VK_NULL_HANDLE;
        frame.imageMemory = VK_NULL_HANDLE;
        frame.view = VK_NULL_HANDLE;
        frame.stagingBuffer = VK_NULL_HANDLE;
        frame.stagingMemory = VK_NULL_HANDLE;
        frame.width = width;
        frame.height = height;

        // Create image
        frame.image = context_->createImage(
            width,
            height,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            &frame.imageMemory
        );

        // Create image view
        frame.view = context_->createImageView(frame.image, VK_FORMAT_R8G8B8A8_UNORM);

        // Transition to general layout for compute shader access
        context_->transitionImageLayout(
            frame.image,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL
        );

        // Create staging buffer for CPU-GPU transfers
        frame.stagingBuffer = context_->createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // Get memory requirements for staging buffer
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context_->getDevice(), frame.stagingBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = context_->findMemoryType(
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        if (vkAllocateMemory(context_->getDevice(), &allocInfo, nullptr, &frame.stagingMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate staging buffer memory!");
        }

        vkBindBufferMemory(context_->getDevice(), frame.stagingBuffer, frame.stagingMemory, 0);
    }

    return true;
}

void FrameManager::cleanup() {
    if (context_ && context_->getDevice() != VK_NULL_HANDLE) {
        for (auto& frame : frames_) {
            if (frame.view != VK_NULL_HANDLE) {
                vkDestroyImageView(context_->getDevice(), frame.view, nullptr);
                frame.view = VK_NULL_HANDLE;
            }
            if (frame.image != VK_NULL_HANDLE) {
                vkDestroyImage(context_->getDevice(), frame.image, nullptr);
                frame.image = VK_NULL_HANDLE;
            }
            if (frame.imageMemory != VK_NULL_HANDLE) {
                vkFreeMemory(context_->getDevice(), frame.imageMemory, nullptr);
                frame.imageMemory = VK_NULL_HANDLE;
            }
            if (frame.stagingBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(context_->getDevice(), frame.stagingBuffer, nullptr);
                frame.stagingBuffer = VK_NULL_HANDLE;
            }
            if (frame.stagingMemory != VK_NULL_HANDLE) {
                vkFreeMemory(context_->getDevice(), frame.stagingMemory, nullptr);
                frame.stagingMemory = VK_NULL_HANDLE;
            }
        }
    }
    frames_.clear();
}

void FrameManager::uploadFrame(uint32_t index, const void* data, size_t size) {
    if (index >= frames_.size()) {
        throw std::runtime_error("Frame index out of range!");
    }

    FrameBuffer& frame = frames_[index];
    VkDeviceSize imageSize = frame.width * frame.height * 4;

    if (size > imageSize) {
        size = imageSize;
    }

    // Map staging buffer and copy data
    void* mapped;
    vkMapMemory(context_->getDevice(), frame.stagingMemory, 0, imageSize, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(context_->getDevice(), frame.stagingMemory);

    // Transition image to transfer destination
    context_->transitionImageLayout(
        frame.image,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );

    // Copy buffer to image
    context_->copyBufferToImage(frame.stagingBuffer, frame.image, frame.width, frame.height);

    // Transition back to general layout for compute shader
    context_->transitionImageLayout(
        frame.image,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL
    );
}

void FrameManager::downloadFrame(uint32_t index, void* data, size_t size) {
    if (index >= frames_.size()) {
        throw std::runtime_error("Frame index out of range!");
    }

    FrameBuffer& frame = frames_[index];
    VkDeviceSize imageSize = frame.width * frame.height * 4;

    if (size > imageSize) {
        size = imageSize;
    }

    // Transition image to transfer source
    context_->transitionImageLayout(
        frame.image,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    );

    // Copy image to buffer
    context_->copyImageToBuffer(frame.image, frame.stagingBuffer, frame.width, frame.height);

    // Transition back to general layout
    context_->transitionImageLayout(
        frame.image,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL
    );

    // Map staging buffer and copy data
    void* mapped;
    vkMapMemory(context_->getDevice(), frame.stagingMemory, 0, imageSize, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(context_->getDevice(), frame.stagingMemory);
}
