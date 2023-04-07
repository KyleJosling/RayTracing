cmake_minimum_required(VERSION 3.16)

#project(raytracing)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(raytracing
    #src/GpuRenderer.cu
    src/Scene.cu
    src/Renderer.cpp
    src/Camera.cpp
	src/WalnutApp.cpp
)
set_source_files_properties(src/Renderer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/WalnutApp.cpp PROPERTIES LANGUAGE CUDA)

message(${CMAKE_SOURCE_DIR})

include_directories(raytracing 
    ${CMAKE_SOURCE_DIR}/Walnut/Walnut/src/
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(raytracing walnut)
target_link_libraries(raytracing tbb)
target_link_libraries(raytracing ${CUDA_LIBRARIES})
