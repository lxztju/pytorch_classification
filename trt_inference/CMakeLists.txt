cmake_minimum_required(VERSION 3.5)
project(hello_trt)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)


message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: "${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(3RDPARTY ${CMAKE_CURRENT_SOURCE_DIR}/3rd)
set(TENSORRT_LIB ${3RDPARTY}/TensorRT-8.0.1.6/targets/x86_64-linux-gnu/lib)
file(GLOB TENSORRT_LIBS "${TENSORRT_LIB}/*.so")

set(OpenCV_DIR ${3RDPARTY}/opencv-3.4.16/share/OpenCV)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "CUDA_LIBRARIES: "${CUDA_LIBRARIES})
message(STATUS "TENSORRT_LIBS: "${TENSORRT_LIBS})
message(STATUS "Opencv_LIBS: "${OpenCV_LIBS})
include_directories(
	${CUDA_INCLUDE_DIRS}
    ${3RDPARTY}/TensorRT-8.0.1.6/include
	${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
	)

add_executable(
    trt_cls
    src/TrtCls.cpp
    )

target_link_libraries(
   trt_cls
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
    ${OpenCV_LIBS}
    )
