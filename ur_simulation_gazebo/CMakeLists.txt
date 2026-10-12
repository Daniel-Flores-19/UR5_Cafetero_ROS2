cmake_minimum_required(VERSION 3.5)
project(ur_simulation_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch world models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/trazo_rviz.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake)
  find_package(ament_cmake_pytest REQUIRED)
  add_launch_test(test/test_gazebo.py
    TIMEOUT
      180
  )
endif()

ament_package()
