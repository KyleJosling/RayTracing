cmake_minimum_required(VERSION 3.16)

#project(raytracing)
set(CMAKE_CXX_STANDARD 17)

add_executable(raytracing
    src/Renderer.cpp
    src/Camera.cpp
	src/WalnutApp.cpp
)

message(${CMAKE_SOURCE_DIR})

include_directories(raytracing ${CMAKE_SOURCE_DIR}/Walnut/Walnut/src/)
target_link_libraries(raytracing walnut)
