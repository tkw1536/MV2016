cmake_minimum_required(VERSION 2.8)
project( eigenfaces )

# Removed Eigen dependency for now.
#find_package(PkgConfig)
#pkg_search_module(Eigen3 REQUIRED eigen3) # Access Eigen via pkg-config

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS system filesystem )

include_directories( include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})  #${Eigen_INCLUDE_DIRS}

add_executable( eigenfaces
 src/eigenfaces.cc )

target_link_libraries( eigenfaces ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} )
