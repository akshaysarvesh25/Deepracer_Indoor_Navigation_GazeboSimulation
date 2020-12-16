cmake_minimum_required(VERSION 2.8.3)
project(deepracer_joy)

find_package(catkin REQUIRED COMPONENTS)
catkin_package()

install(DIRECTORY config launch 
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if (CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  file(GLOB launch_files ${CMAKE_CURRENT_SOURCE_DIR}/launch/*.launch)
  foreach(_file ${launch_files})
    roslaunch_add_file_check(${_file})
  endforeach()
endif()
