cmake_minimum_required(VERSION 2.8)
project( EssentialMatrix )

find_package( OpenCV REQUIRED )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})
add_executable( essential visual_features_fundamental_matrix_template.cpp )
target_link_libraries( essential ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )
