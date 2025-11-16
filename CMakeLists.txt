cmake_minimum_required(VERSION 3.20)
project(object_tracking LANGUAGES CXX)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(object_tracker src/main.cpp)
target_include_directories(object_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_tracker PRIVATE ${OpenCV_LIBS})
