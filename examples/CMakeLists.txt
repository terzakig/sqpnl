project(Example)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

SET(INCLUDE_DIRS ${CMAKE_SOURCE_DIR} )

#list( APPEND CMAKE_MODULE_PATH
#  ${CMAKE_SOURCE_DIR}/cmake
#)

find_package(OpenCV)
if (OpenCV_FOUND)
    list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    list(APPEND LIBS ${OpenCV_LIBS})
    add_definitions(-DHAVE_OPENCV)
    #add_compile_definitions(HAVE_OPENCV)
    set(sources main.cpp )
else(OpenCV_FOUND)
    set(sources main_plain.cpp )
endif(OpenCV_FOUND)

# Append list of libs with SQPnL (static library)
list( APPEND LIBS SQPnL )

add_executable(sqpnl_example ${sources})

target_link_libraries(sqpnl_example ${LIBS} )

#add_subdirectory (robust)
