#include "ShaderLoader.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace shader {

namespace {

namespace fs = std::filesystem;

bool compileIfNeeded(const fs::path& spirvPath) {
    if (fs::exists(spirvPath)) {
        return true;
    }

    fs::path sourcePath = spirvPath;
    sourcePath.replace_extension("");  // Strip only the final extension (e.g. .spv)
    if (!fs::exists(sourcePath)) {
        return false;
    }

    std::string command = "glslangValidator -V \"" + sourcePath.string() + "\" -o \"" +
                          spirvPath.string() + "\"";
    int result = std::system(command.c_str());
    return result == 0 && fs::exists(spirvPath);
}

std::vector<uint32_t> readFile(const std::string& path) {
    const fs::path spirvPath(path);
    if (!compileIfNeeded(spirvPath)) {
        throw std::runtime_error("Failed to locate or compile shader: " + path);
    }

    std::ifstream file(spirvPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

}  // namespace

ShaderModule loadShaderModule(VkDevice device, const std::string& path) {
    const std::vector<uint32_t> spirv = readFile(path);
    if (spirv.empty()) {
        throw std::runtime_error("Shader file is empty: " + path);
    }

    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));
    return ShaderModule(device, module);
}

}  // namespace shader
