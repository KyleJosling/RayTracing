cmake_minimum_required(VERSION 3.16)
project(RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_compile_definitions("NDEBUG")

add_subdirectory(Walnut)
add_subdirectory(RayTracing)

# will create compile commands link in the root of the project
# for clangd or whatever
execute_process(
	COMMAND ${CMAKE_COMMAND} -E create_symlink
		${CMAKE_BINARY_DIR}/compile_commands.json
		${CMAKE_SOURCE_DIR}/compile_commands.json
)
