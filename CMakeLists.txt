cmake_minimum_required(VERSION 3.5)
project(SQPnL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_EXAMPLES "Build the examples?" ON)

add_subdirectory(sqpnl)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
