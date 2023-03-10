cmake_minimum_required(VERSION 3.0)
project(vkIsh)
set(CMAKE_CXX_STANDARD 20)

include_directories(external)
include_directories(src)

add_executable (${PROJECT_NAME} src/main.cpp)

find_package(Vulkan REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

add_subdirectory(external/glfw)
target_include_directories(${PROJECT_NAME} PRIVATE external/glfw/include)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

find_library(SHADERC_LIB shaderc_combinedd $ENV{VULKAN_SDK}/Lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${SHADERC_LIB})