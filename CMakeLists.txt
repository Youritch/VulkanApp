cmake_minimum_required (VERSION 3.30)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
add_executable(${PROJECT_NAME} main.cpp)

find_package (Vulkan REQUIRED)
find_package(glfw3 3.4 REQUIRED)
## GLFW
include_directories(/opt/homebrew/Cellar/glfw/3.4/include)
include_directories(/opt/homebrew/Cellar/glm/1.0.1/include)
#link_directories(/opt/homebrew/Cellar/glfw/3.4/lib)
target_link_libraries(${PROJECT_NAME} glfw)
## some libs
link_directories(/usr/local/lib)

if (VULKAN_FOUND)
    message (STATUS "Found Vulkan, Including and Linking now")
    include_directories(${Vulkan_INCLUDE_DIRS})
    target_link_libraries (${PROJECT_NAME} ${Vulkan_LIBRARIES})
endif (VULKAN_FOUND)