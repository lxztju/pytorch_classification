cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(c)

set(CMAKE_PREFIX_PATH
        /home/luxiangzhe/libtorch_cpu
        /home/luxiangzhe/opencv_3.4.3/build)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(c main.cpp)
target_link_libraries(c ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET c PROPERTY CXX_STANDARD 14)
