
find_package(Eigen3 3.4 REQUIRED)
list(APPEND INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

add_subdirectory(SQPEngine)

set(SQPNL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/SQPEngine)

add_library (SQPnL 
    sqpnl.cpp 
    sqpnl.h
    types.h)

target_compile_options(SQPnL PRIVATE -Wall)
target_link_libraries(SQPnL SQPEngine)
target_include_directories(SQPnL PUBLIC ${SQPNL_INCLUDE_DIRS})
