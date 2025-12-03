cmake_minimum_required(VERSION 3.18)
project(VideoSuperResolution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Path to the LibTorch distribution (with CUDA). Update as needed.
set(Torch_DIR "" CACHE PATH "Path to LibTorch distribution directory")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

add_executable(video_sr
  src/main.cpp
  src/VideoSuperRes.cpp
)
target_include_directories(video_sr PRIVATE include)
target_link_libraries(video_sr PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if (TORCH_CUDA_ARCH_LIST)
  message(STATUS "Using user-provided TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
else()
  # RTX 4060 Ti uses SM 89 (Ada). Set default if user did not override.
  set_property(TARGET video_sr PROPERTY TORCH_CUDA_ARCH_LIST "8.9")
endif()

if (MSVC)
  set_property(TARGET video_sr PROPERTY VS_STARTUP_PROJECT video_sr)
endif()

set_property(TARGET video_sr PROPERTY CXX_STANDARD 17)
