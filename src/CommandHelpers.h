// Common helper utilities for Vulkan error handling.

#pragma once

#include <stdexcept>
#include <string>
#include <vulkan/vulkan.h>

namespace video::processing::detail {

[[noreturn]] inline void throwVulkanError(VkResult result, const char* expr,
                                          const char* file, int line) {
    throw std::runtime_error(std::string("Vulkan error: ") + std::to_string(result) +
                             " while executing " + expr + " at " + file + ":" +
                             std::to_string(line));
}

inline void vkCheck(VkResult result, const char* expr, const char* file, int line) {
    if (result != VK_SUCCESS) {
        throwVulkanError(result, expr, file, line);
    }
}

}  // namespace video::processing::detail

#define VK_CHECK(expr) ::video::processing::detail::vkCheck((expr), #expr, __FILE__, __LINE__)
