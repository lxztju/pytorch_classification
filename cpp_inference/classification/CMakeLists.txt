cmake_minimum_required(VERSION 3.5)
project(classification)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")



include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

message(STATUS "Torch_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
# link_directories(
#     ${OpenCV_LIBRARIES_DIRS}
#     # ${OpenCV_DIR}/lib
#     ${Torch_LIBRARIES_DIRS}
# )


# link_directories(
#     ${OpenCV_LIBRARIES_DIRS}
# )

# file (GLOB CPP_SRC src/*.cpp)

add_library(classification SHARED src/ImgCls.cpp)
target_link_libraries(
    classification
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)


add_executable(imgcls test_ImgCls.cpp)
target_link_libraries(
    imgcls
    classification
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)