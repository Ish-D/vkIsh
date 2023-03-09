cmake_minimum_required(VERSION 3.0)
project(vkIsh)

include_directories(external)
include_directories(src)

add_executable (${PROJECT_NAME} src/main.cpp)

find_package(Vulkan REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

add_subdirectory(external/glfw)
target_include_directories(${PROJECT_NAME} PRIVATE external/glfw/include)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

target_include_directories(${PROJECT_NAME} PRIVATE external/glm)