cmake_minimum_required(VERSION 3.14)

project(DPLU
        VERSION 0.1
        LANGUAGES CXX
        DESCRIPTION "DPLU - Sparse LU decomposition solver for real and complex systems")

set(CMAKE_CXX_STANDARD 17) # TODO update to version 20/23

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror -std=c++17")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(FetchEigen)
include(FetchSpdlog)

add_subdirectory(tests)