cmake_minimum_required(VERSION 3.0.2)
project(fav_control)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  message_generation
  geometry_msgs
  nav_msgs
  std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/RollControl.cfg
  cfg/PitchControl.cfg
  cfg/YawControl.cfg
  cfg/XControl.cfg
  cfg/YControl.cfg
  cfg/ZControl.cfg
)

# for c code
# make sure configure headers are built before any node using them
# add_dependencies(example_node ${PROJECT_NAME}_gencfg)
# add_dependencies(your_program ${${PROJECT_NAME}_EXPORTED_TARGETS})
# add_dependencies(your_library ${${PROJECT_NAME}_EXPORTED_TARGETS})

# add_message_files(
#   FILES
# )

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES depth_controller
#  CATKIN_DEPENDS other_catkin_pkg
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
  CATKIN_DEPENDS dynamic_reconfigure
)

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)
