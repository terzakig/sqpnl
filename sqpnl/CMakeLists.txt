
add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")


find_package(Eigen3 REQUIRED)
list(APPEND INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} )

add_subdirectory(SQPEngine)

SET(SQPNL_INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/SQPEngine)

add_library (SQPnL 
    sqpnl.cpp 
    sqpnl.h
    types.h)

target_link_libraries(SQPnL SQPEngine)
target_include_directories(SQPnL PUBLIC ${SQPNL_INCLUDE_DIRS})