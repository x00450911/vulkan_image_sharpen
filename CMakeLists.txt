cmake_minimum_required(VERSION 3.16)
project(VulkanEdgeSharpen CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

add_executable(vulkan_edge_sharpen
    src/main.cpp
)

target_include_directories(vulkan_edge_sharpen
    PRIVATE
        ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan_edge_sharpen
    PRIVATE
        ${Vulkan_LIBRARIES}
)

target_compile_definitions(vulkan_edge_sharpen
    PRIVATE
        VK_ENABLE_BETA_EXTENSIONS=0
)
