cmake_minimum_required(VERSION 3.11.0)

set(NAME LveEngine)
project(${NAME} VERSION 0.23.0 LANGUAGES C CXX)

# Everything the engine needs is vendored under external/ (GLFW, GLM, tinyobjloader),
# except the Vulkan SDK, which provides Vulkan and Slang.
# Set VULKAN_SDK_PATH on the command line to override the environment variable.
if (NOT DEFINED VULKAN_SDK_PATH)
  if (DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
  else()
    message(FATAL_ERROR "VULKAN_SDK_PATH is not set and VULKAN_SDK is not in the environment")
  endif()
endif()
message(STATUS "Using Vulkan SDK at: ${VULKAN_SDK_PATH}")

# GLFW is built from source in-tree.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if (MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/external/glm
  ${PROJECT_SOURCE_DIR}/external/tinyobjloader
  ${PROJECT_SOURCE_DIR}/external/glfw/include/GLFW
  ${VULKAN_SDK_PATH}/Include
  ${VULKAN_SDK_PATH}/Include/vulkan
)

target_link_directories(${PROJECT_NAME} PUBLIC ${VULKAN_SDK_PATH}/Lib)
target_link_libraries(${PROJECT_NAME} glfw vulkan-1 slang)
