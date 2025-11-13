cmake_minimum_required(VERSION 3.20)

project(VulkanMultiFrameDenoiser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
if (NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "glslangValidator not found. Ensure the Vulkan SDK is installed and Vulkan_GLSLANG_VALIDATOR_EXECUTABLE is set.")
endif

add_executable(denoiser
    src/main.cpp
    src/multi_frame_denoiser.cpp
    src/vulkan_context.cpp
)

target_include_directories(denoiser
    PRIVATE
        ${Vulkan_INCLUDE_DIR}
        include
        external
)

target_link_libraries(denoiser
    PRIVATE
        ${Vulkan_LIBRARIES}
)

add_custom_target(shaders ALL
    COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${CMAKE_SOURCE_DIR}/shaders/denoise.comp -o ${CMAKE_BINARY_DIR}/shaders/denoise.comp.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/denoise.comp
    BYPRODUCTS ${CMAKE_BINARY_DIR}/shaders/denoise.comp.spv
    COMMENT "Compiling compute shaders"
)

add_dependencies(denoiser shaders)
