cmake_minimum_required(VERSION 3.16)
project(gen2_image_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GEN2_ENABLE_OPENMP "Enable OpenMP parallel loops" OFF)

add_library(gen2_pipeline
    src/image.cpp
    src/super_resolution.cpp
    src/mctf_denoiser.cpp
    src/accelerator/accelerator.cpp)

target_include_directories(gen2_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(GEN2_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(gen2_pipeline PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(gen2_demo src/main.cpp)

target_link_libraries(gen2_demo
    PRIVATE
        gen2_pipeline
        pthread)
