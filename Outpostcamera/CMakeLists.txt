cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_STANDARD 14)
project(robot_vision)
# catkin_make( -DCMAKE_BUILD_TYPE=Debug )
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

#list(APPEND CMAKE_PREFIX_PATH "/home/xdd2/Downloads/libtorch/share/cmake/Torch")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


#list( APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch/")

find_package(OpenCV  REQUIRED)
#include_directories(
#        include ${catkin_INCLUDE_DIRS}${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        message_generation
        cv_bridge
        image_transport
        sensor_msgs
        std_msgs
        geometry_msgs
        dynamic_reconfigure
        )
set(CMAKE_PREFIX_PATH "/usr/local/libtorch/")
find_package(Torch REQUIRED)
message(STATUS "**** :: ${catkin_LIBRARIES}")

#message( STATUS "********** ${CMAKE_PREFIX_PATH}")


# generate_dynamic_reconfigure_options(
#   cfg/color.cfg
# )

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(FILES
        roboPos.msg
        OutpostDetected.msg
        )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
        std_msgs
        geometry_msgs
        )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES robot_vision
        CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
        #  DEPENDS system_lib
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)


#link_directories(${OpenCV_LIBRARY_DIRS})

# add_definitions(-std=c++11 -g -O3)
## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/robot_vision.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/robot_vision_node.cpp)


# add_executable(tag_detector_pub src/image_raw.cpp)
# target_link_libraries(tag_detector_pub  ${OpenCV_LIBS} ${catkin_LIBRARIES})


# add_executable(moving_detect_node src/motion_detect.cpp)
# target_link_libraries(moving_detect_node ${OpenCV_LIBS} ${catkin_LIBRARIES})


#add_executable(play_vedio_node src/play_vedio.cpp)
#target_link_libraries(play_vedio_node ${OpenCV_LIBS} ${catkin_LIBRARIES})

# add_executable(detect_by_color_node src/detect_by_color.cpp)
# target_link_libraries(detect_by_color_node ${OpenCV_LIBS} ${catkin_LIBRARIES})


# add_executable(capture_vedio_node src/capture_vedio.cpp)
# target_link_libraries(capture_vedio_node ${OpenCV_LIBS} ${catkin_LIBRARIES})

# add_executable(cv_test_node src/get_world_pos.cpp)
# target_link_libraries(cv_test_node ${OpenCV_LIBS} ${catkin_LIBRARIES})


# add_executable(get_gazebo_pos
#         src/get_gazebo_pos.cpp
#         )
# add_dependencies(get_gazebo_pos ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(get_gazebo_pos ${OpenCV_LIBS} ${catkin_LIBRARIES})
# add_dependencies(get_gazebo_pos_2_cams ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(get_gazebo_pos_2_cams
#   ${catkin_LIBRARIES}
# )

# add_executable(merge_2_pictures src/merge_2_pictures.cpp)
# add_dependencies(merge_2_pictures ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(merge_2_pictures ${OpenCV_LIBS} ${catkin_LIBRARIES})

#add_executable(get_gazebo_pos2
#        src/get_gazebo_pos2.cpp
#        )
#add_dependencies(get_gazebo_pos2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(get_gazebo_pos2 ${OpenCV_LIBS} ${catkin_LIBRARIES})


find_package(yaml-cpp REQUIRED)

add_executable(cameraDetNode_gc
        src/Det.cpp
        src/Det.h
        src/Window.h
        src/State.h
        src/RosDet.h
        src/RosDet.cpp
        src/CvUtils.h
        src/CvUtils.cpp
        src/CameraDetNode.cpp
    )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS}
                    "/usr/include/eigen3/"
    )
#add_dependencies(cameraDetNode_gc ${catkin_EXPORTED_TARGETS})
#message(STATUS "==>>catkin_LIBRARIES")
target_link_libraries(cameraDetNode_gc
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
        yaml-cpp)

add_executable(cvUtilsTest src/CvUtils.h src/CvUtils.cpp src/CvUtilsTest.cpp)
target_link_libraries(cvUtilsTest ${OpenCV_LIBRARIES} yaml-cpp)
