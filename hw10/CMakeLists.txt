cmake_minimum_required(VERSION 2.8)
project( SURF_descriptor )

find_package( OpenCV REQUIRED )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable( SURF_descriptor
 src/visual_featurs_yoda.cpp )

target_link_libraries( SURF_descriptor ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )
