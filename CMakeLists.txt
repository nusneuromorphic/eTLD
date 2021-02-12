cmake_minimum_required(VERSION 2.8)
project(ETLDExample)

set(VLFEATROOT /home/andres/data_extra/repos/vlfeat-0.9.21)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(ETLDDescLib
    src/ETLDDesc.cpp
    src/ECparam.cpp
    src/Matrix.cpp
    )

target_include_directories(ETLDDescLib
    PUBLIC include
    PRIVATE src
    )

add_library(vlfeat
    ${VLFEATROOT}/vl/host.c
    ${VLFEATROOT}/vl/generic.c
    ${VLFEATROOT}/vl/kmeans.c
    ${VLFEATROOT}/vl/kdtree.c
    ${VLFEATROOT}/vl/svm.c
    ${VLFEATROOT}/vl/svmdataset.c
    ${VLFEATROOT}/vl/random.c
    ${VLFEATROOT}/vl/mathop.c
    ${VLFEATROOT}/vl/homkermap.c
    ${VLFEATROOT}/vl/mathop_sse2.c    
    ${VLFEATROOT}/vl/mathop_avx.c    
    )

include_directories(${VLFEATROOT}/vl)

find_package(OpenCV 3.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
find_package(Threads)

add_executable (ETLDDesc example/loadandtrack.cpp)

target_link_libraries (ETLDDesc ETLDDescLib ${OpenCV_LIBRARIES} vlfeat pthread)
