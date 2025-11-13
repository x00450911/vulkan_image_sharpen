cmake_minimum_required(VERSION 3.15)
project(VulkanVideoDenoiser VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build test suite" OFF)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Vulkan_INCLUDE_DIRS})

# Source files
set(DENOISER_SOURCES
    src/vulkan_video_denoiser.cpp
)

set(DENOISER_HEADERS
    include/vulkan_video_denoiser.hpp
)

# Create library
add_library(vulkan_video_denoiser STATIC ${DENOISER_SOURCES} ${DENOISER_HEADERS})

target_link_libraries(vulkan_video_denoiser PUBLIC ${Vulkan_LIBRARIES})

target_include_directories(vulkan_video_denoiser PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler warnings
if(MSVC)
    target_compile_options(vulkan_video_denoiser PRIVATE /W4)
else()
    target_compile_options(vulkan_video_denoiser PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Shader compilation
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)

if(GLSLC)
    message(STATUS "Found glslc: ${GLSLC}")
    
    set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    
    file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})
    
    # Compile compute shader
    set(COMPUTE_SHADER ${SHADER_SOURCE_DIR}/temporal_denoise.comp)
    set(COMPUTE_SHADER_SPV ${SHADER_BINARY_DIR}/temporal_denoise.spv)
    
    add_custom_command(
        OUTPUT ${COMPUTE_SHADER_SPV}
        COMMAND ${GLSLC} -o ${COMPUTE_SHADER_SPV} ${COMPUTE_SHADER}
        DEPENDS ${COMPUTE_SHADER}
        COMMENT "Compiling compute shader"
        VERBATIM
    )
    
    add_custom_target(compile_shaders ALL DEPENDS ${COMPUTE_SHADER_SPV})
    
    # Install shaders
    install(FILES ${COMPUTE_SHADER_SPV} DESTINATION shaders)
else()
    message(WARNING "glslc not found. Shaders will not be compiled automatically.")
    message(WARNING "Please compile shaders manually or install Vulkan SDK.")
endif()

# Build examples
if(BUILD_EXAMPLES)
    add_executable(denoiser_example examples/main.cpp)
    target_link_libraries(denoiser_example PRIVATE vulkan_video_denoiser)
    
    if(GLSLC)
        add_dependencies(denoiser_example compile_shaders)
    endif()
endif()

# Installation
install(TARGETS vulkan_video_denoiser
    EXPORT VulkanVideoDenoiserTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${DENOISER_HEADERS} DESTINATION include)

install(EXPORT VulkanVideoDenoiserTargets
    FILE VulkanVideoDenoiserTargets.cmake
    NAMESPACE VulkanVideoDenoiser::
    DESTINATION lib/cmake/VulkanVideoDenoiser
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VulkanVideoDenoiserConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVideoDenoiserConfig.cmake
    INSTALL_DESTINATION lib/cmake/VulkanVideoDenoiser
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVideoDenoiserConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVideoDenoiserConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVideoDenoiserConfigVersion.cmake
    DESTINATION lib/cmake/VulkanVideoDenoiser
)

# Print summary
message(STATUS "")
message(STATUS "Vulkan Video Denoiser Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "  Vulkan libraries: ${Vulkan_LIBRARIES}")
message(STATUS "")
