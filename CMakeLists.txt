cmake_minimum_required(VERSION 3.15)
project(VulkanMultiFrameDenoising)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${VULKAN_INCLUDE_DIRS}
    ${OPENCV_INCLUDE_DIRS}
    include
)

# Source files
set(SOURCES
    src/main.cpp
    src/vulkan_context.cpp
    src/frame_manager.cpp
    src/denoiser.cpp
)

# Shader compilation
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)

file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Compile shaders using glslc
add_custom_target(shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND glslc ${SHADER_DIR}/denoise.comp -o ${SHADER_OUTPUT_DIR}/denoise.spv
    COMMAND glslc ${SHADER_DIR}/accumulate.comp -o ${SHADER_OUTPUT_DIR}/accumulate.spv
    DEPENDS ${SHADER_DIR}/denoise.comp ${SHADER_DIR}/accumulate.comp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} shaders)

target_link_libraries(${PROJECT_NAME}
    ${VULKAN_LIBRARIES}
    glfw
    ${OPENCV_LIBS}
)

# Copy shaders to build directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${SHADER_OUTPUT_DIR} $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)
