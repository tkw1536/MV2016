cmake_minimum_required(VERSION 2.8)
project( lab_homography_donald )

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable( lab_homography_donald
 src/lab_homography_donald.cpp )

target_link_libraries( lab_homography_donald ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )
