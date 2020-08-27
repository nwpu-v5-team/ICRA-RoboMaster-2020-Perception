cmake_minimum_required(VERSION 3.0.2)
project(ICRA-vision)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(CUDA  REQUIRED)
find_package(Threads REQUIRED)


find_library(CUDNN_LIB cudnn HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")
set(CUDA_LIBRARIES ${CUDART_LIB})

set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS "--gpu-architecture=sm_53")
catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES ICRA-vision
        #  CATKIN_DEPENDS roscpp rospy
        #  DEPENDS system_lib
)

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CUDNN_ROOT_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)
message(STATUS "${CUDA_INCLUDE_DIRS}")
message(STATUS "${CUDA_ROOT_DIR}")
set_property(SOURCE ./customNMS.cu  PROPERTY CUDA_STANDARD 11)


cuda_add_executable(ICRA_vision
        src/Detdevice.cpp
        src/detection.cpp
        src/detection.h
        src/target.h
        src/detectionSSD.cpp
        src/detectionSSD.h
        src/bufferManager.h
        src/logger.cpp
        src/logger.h
        src/customNMS.h
        src/customNMS.cu)

set_target_properties(ICRA_vision PROPERTIES LINK_FLAGS "-Wl,--exclude-libs, -Wl,--no-undefined")
target_include_directories(
        ICRA_vision
        PUBLIC
        ${CUDA_INCLUDE_DIRS}

)
target_link_libraries(
        ICRA_vision
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        nvinfer
        nvonnxparser
        -Wl,--unresolved-symbols=ignore-in-shared-libs
        yaml-cpp
)
