cmake_minimum_required(VERSION 2.8.4)
project(Fg-detection-dynamic-background)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(Fg-detection-dynamic-background ${SOURCE_FILES})

find_package( OpenCV REQUIRED )
target_link_libraries( Fg-detection-dynamic-background ${OpenCV_LIBS} )