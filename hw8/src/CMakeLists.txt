cmake_minimum_required(VERSION 3.6)
project(src)

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3) # Access Eigen via pkg-config

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})
add_executable( circle lm_circle.cpp )
target_link_libraries( circle ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )