cmake_minimum_required(VERSION 3.0.2)
project(rtsp_to_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS cv_bridge roscpp sensor_msgs std_msgs
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GST_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/capture_frame_and_tx_from_video.cpp)

## Add cmake target dependencies of the executable
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GST_LIBRARIES}
  jsoncpp
)

