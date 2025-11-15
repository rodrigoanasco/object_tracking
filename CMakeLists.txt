cmake_minimum_required(VERSION 3.20)
project(object_tracking LANGUAGES CXX)

# Find OpenCV (using OpenCV_DIR from your cache)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(object_tracker src/main.cpp)

target_include_directories(object_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_tracker PRIVATE ${OpenCV_LIBS})
