cmake_minimum_required(VERSION 3.10)
project(VulkanDenoiser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Include directories
include_directories(${Vulkan_INCLUDE_DIRS})

# Source files
set(SOURCES
    vulkan_denoiser.cpp
    main.cpp
    video_io.cpp
)

# Create executable
add_executable(vulkan_denoiser ${SOURCES})

# Link libraries
target_link_libraries(vulkan_denoiser ${Vulkan_LIBRARIES})

# Compile shader
find_program(GLSLANG_VALIDATOR glslangValidator)
if(GLSLANG_VALIDATOR)
    add_custom_command(
        OUTPUT denoise.comp.spv
        COMMAND ${GLSLANG_VALIDATOR} -V denoise.comp -o denoise.comp.spv
        DEPENDS denoise.comp
        COMMENT "Compiling compute shader"
    )
    add_custom_target(shader DEPENDS denoise.comp.spv)
    add_dependencies(vulkan_denoiser shader)
else()
    message(WARNING "glslangValidator not found. Shader compilation skipped.")
    message(STATUS "Please compile denoise.comp manually using: glslangValidator -V denoise.comp -o denoise.comp.spv")
endif()

# Copy shader to build directory
add_custom_command(TARGET vulkan_denoiser POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/denoise.comp.spv
    $<TARGET_FILE_DIR:vulkan_denoiser>/denoise.comp.spv
)

# Compiler-specific options
if(MSVC)
    target_compile_options(vulkan_denoiser PRIVATE /W4)
else()
    target_compile_options(vulkan_denoiser PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Debug/Release configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(vulkan_denoiser PRIVATE DEBUG)
endif()
