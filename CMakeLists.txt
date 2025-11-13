cmake_minimum_required(VERSION 3.10)
project(MotionVectorDenoising)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable OpenCV
option(ENABLE_OPENCV "Enable OpenCV support" ON)

# Block matching implementation (no dependencies)
add_library(motion_vector_calculator STATIC
    motion_vector_calculator.cpp
    motion_vector_calculator.h
)

target_include_directories(motion_vector_calculator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# OpenCV implementation (optional)
if(ENABLE_OPENCV)
    find_package(OpenCV QUIET)
    
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        add_definitions(-DOPENCV_AVAILABLE)
        
        add_library(motion_vector_opencv STATIC
            motion_vector_opencv.cpp
            motion_vector_opencv.h
        )
        
        target_include_directories(motion_vector_opencv PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
        target_link_libraries(motion_vector_opencv PUBLIC
            ${OpenCV_LIBS}
        )
        
        # Example executable with OpenCV
        add_executable(example_usage
            example_usage.cpp
        )
        
        target_link_libraries(example_usage
            motion_vector_calculator
            motion_vector_opencv
            ${OpenCV_LIBS}
        )
        
    else()
        message(WARNING "OpenCV not found. Building without OpenCV support.")
        message(STATUS "To install OpenCV: sudo apt-get install libopencv-dev")
        
        # Example executable without OpenCV
        add_executable(example_usage
            example_usage.cpp
        )
        
        target_link_libraries(example_usage
            motion_vector_calculator
        )
    endif()
else()
    message(STATUS "OpenCV support disabled")
    
    # Example executable without OpenCV
    add_executable(example_usage
        example_usage.cpp
    )
    
    target_link_libraries(example_usage
        motion_vector_calculator
    )
endif()

# Installation
install(TARGETS motion_vector_calculator
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(ENABLE_OPENCV AND OpenCV_FOUND)
    install(TARGETS motion_vector_opencv
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

install(FILES motion_vector_calculator.h
    DESTINATION include
)

if(ENABLE_OPENCV AND OpenCV_FOUND)
    install(FILES motion_vector_opencv.h
        DESTINATION include
    )
endif()
