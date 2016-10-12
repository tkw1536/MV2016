cmake_minimum_required(VERSION 2.8)
project( camera_calibration )

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable( camera_calibration
 src/camera_calibration.cpp )

target_link_libraries( camera_calibration ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )
