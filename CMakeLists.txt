cmake_minimum_required(VERSION 3.22)

project(vulkan_android_video_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vulkan_android_video_pipeline
    src/VulkanAndroidFrameProcessor.cpp
    src/VulkanContext.cpp
    src/ShaderLoader.cpp
    src/CommandHelpers.cpp
)

target_include_directories(vulkan_android_video_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(Vulkan REQUIRED)
target_link_libraries(vulkan_android_video_pipeline
    PRIVATE
        Vulkan::Vulkan
        android
        log
)
