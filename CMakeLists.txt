cmake_minimum_required (VERSION 3.5)
project (SQPnL)

option(BUILD_EXAMPLES "Build the examples?" ON)

add_subdirectory (sqpnl)

if(BUILD_EXAMPLES)
    add_subdirectory (examples)
endif()
