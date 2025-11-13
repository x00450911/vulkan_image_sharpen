cmake_minimum_required(VERSION 3.16)

project(MotionVectorDenoise LANGUAGES CXX)

option(BUILD_WARNINGS "Enable extra compiler warnings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(motion_vectors STATIC src/motion_vectors.cpp)
target_include_directories(motion_vectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (BUILD_WARNINGS)
    if (MSVC)
        target_compile_options(motion_vectors PRIVATE /W4 /permissive-)
    else ()
        target_compile_options(motion_vectors PRIVATE -Wall -Wextra -Wpedantic)
    endif ()
endif ()

add_executable(motion_vectors_demo src/main.cpp)
target_link_libraries(motion_vectors_demo PRIVATE motion_vectors)

if (BUILD_WARNINGS)
    if (MSVC)
        target_compile_options(motion_vectors_demo PRIVATE /W4 /permissive-)
    else ()
        target_compile_options(motion_vectors_demo PRIVATE -Wall -Wextra -Wpedantic)
    endif ()
endif ()

