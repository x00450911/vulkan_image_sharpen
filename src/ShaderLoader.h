// Utility for loading SPIR-V compute shader modules.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "CommandHelpers.h"

namespace shader {

class ShaderModule {
public:
    ShaderModule() = default;
    ShaderModule(VkDevice device, VkShaderModule module)
        : device_(device), module_(module) {}

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    ShaderModule(ShaderModule&& other) noexcept {
        swap(other);
    }
    ShaderModule& operator=(ShaderModule&& other) noexcept {
        if (this != &other) {
            destroy();
            swap(other);
        }
        return *this;
    }

    ~ShaderModule() { destroy(); }

    VkShaderModule get() const { return module_; }

private:
    void swap(ShaderModule& other) noexcept {
        std::swap(device_, other.device_);
        std::swap(module_, other.module_);
    }
    void destroy() {
        if (module_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, module_, nullptr);
        }
        module_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule module_ = VK_NULL_HANDLE;
};

ShaderModule loadShaderModule(VkDevice device, const std::string& path);

}  // namespace shader
