cmake_minimum_required(VERSION 3.0)
project(vkIsh)
include_directories(src)

add_executable (vkIsh src/main.cpp)

find_package(Vulkan REQUIRED)
target_link_libraries(vkIsh PRIVATE Vulkan::Vulkan)