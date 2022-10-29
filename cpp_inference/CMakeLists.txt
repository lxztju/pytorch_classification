cmake_minimum_required(VERSION 3.5)
project(cpp_deployment)

add_definitions(-D _GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)


set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CLASS_3DPARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party_library)


# 两个必要的依赖包
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party_library/opencv-3.4.16/share/OpenCV)
set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party_library/libtorch/share/cmake/Torch)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "Torch_LIBS: ${TORCH_LIBRARIES}")



## build 
add_subdirectory(classification)