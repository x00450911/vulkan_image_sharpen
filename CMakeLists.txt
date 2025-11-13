cmake_minimum_required(VERSION 3.16)
project(VulkanMultiFrameDenoiser LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan REQUIRED)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ENV VULKAN_SDK)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "Could not find glslc; please ensure the Vulkan SDK is installed and glslc is on PATH.")
endif()

add_custom_command(
    OUTPUT ${SPIRV_DIR}/denoise.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SPIRV_DIR}
    COMMAND ${GLSLC_EXECUTABLE} -O --target-env=vulkan1.2 -o ${SPIRV_DIR}/denoise.comp.spv ${SHADER_DIR}/denoise.comp
    DEPENDS ${SHADER_DIR}/denoise.comp
    COMMENT "Compiling denoise compute shader"
    VERBATIM
)

add_custom_target(CompileShaders DEPENDS ${SPIRV_DIR}/denoise.comp.spv)

file(GLOB PROJECT_SOURCES
    src/*.cpp
    src/*.h
)

add_executable(vulkan_multi_frame_denoiser
    ${PROJECT_SOURCES}
)

target_include_directories(vulkan_multi_frame_denoiser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(vulkan_multi_frame_denoiser PRIVATE Vulkan::Vulkan)

add_dependencies(vulkan_multi_frame_denoiser CompileShaders)

target_compile_definitions(vulkan_multi_frame_denoiser PRIVATE SHADER_BINARY_PATH="${SPIRV_DIR}/denoise.comp.spv")
