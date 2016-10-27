cmake_minimum_required(VERSION 2.8)
project( Levenberg_Marquardt )

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3) # Access Eigen via pkg-config

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS}) 
add_executable( hahn1 lm_hahn1.cpp )
target_link_libraries( hahn1 ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )
