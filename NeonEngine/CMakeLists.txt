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

# NEON_BUILD_ENGINE=OFF builds only the headless NeonML tool (no GLFW/X11), e.g. on Colab.
option(NEON_BUILD_ENGINE "Build the windowed LveEngine executable" ON)

# The Windows SDK uses Include/ and Lib/, the Linux SDK include/ and lib/.
if (WIN32)
  set(NEON_VULKAN_LIBS vulkan-1 slang)
else()
  set(NEON_VULKAN_LIBS vulkan slang)
endif()
set(NEON_VULKAN_INCLUDES ${VULKAN_SDK_PATH}/Include ${VULKAN_SDK_PATH}/include)
set(NEON_VULKAN_LIBDIRS ${VULKAN_SDK_PATH}/Lib ${VULKAN_SDK_PATH}/lib)
find_package(Threads REQUIRED)

# NeonML: headless ML entry points (gpt-oss inference and training), see src/headless/.
add_executable(NeonML
  ${PROJECT_SOURCE_DIR}/src/headless/neon_ml_main.cpp
  ${PROJECT_SOURCE_DIR}/src/ml_compute.cpp
  ${PROJECT_SOURCE_DIR}/src/ml_json.cpp
  ${PROJECT_SOURCE_DIR}/src/ml_safetensors.cpp
  ${PROJECT_SOURCE_DIR}/src/slang_runtime_compiler.cpp
  ${PROJECT_SOURCE_DIR}/src/gpt_oss_model.cpp
  ${PROJECT_SOURCE_DIR}/src/gpt_oss_app.cpp
  ${PROJECT_SOURCE_DIR}/src/gpt_oss_train.cpp
  ${PROJECT_SOURCE_DIR}/src/memorilla.cpp
  ${PROJECT_SOURCE_DIR}/src/memorilla_train.cpp
)
target_compile_features(NeonML PUBLIC cxx_std_17)
if (MSVC)
  target_compile_options(NeonML PRIVATE /utf-8)
endif()
target_include_directories(NeonML PUBLIC ${PROJECT_SOURCE_DIR}/src ${NEON_VULKAN_INCLUDES})
target_link_directories(NeonML PUBLIC ${NEON_VULKAN_LIBDIRS})
target_link_libraries(NeonML ${NEON_VULKAN_LIBS} Threads::Threads)

if (NOT NEON_BUILD_ENGINE)
  return()
endif()

# GLFW is built from source in-tree.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "/src/headless/")

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

# MctsBench: standalone MCTS-VK experiment harness (bench / sanity / bias / replay / matches).
# Needs only the Vulkan device wrappers, so it builds even while other apps are mid-change.
add_executable(MctsBench
  ${PROJECT_SOURCE_DIR}/src/headless/mcts_main.cpp
  ${PROJECT_SOURCE_DIR}/src/mcts_app.cpp
  ${PROJECT_SOURCE_DIR}/src/mcts_gpu.cpp
  ${PROJECT_SOURCE_DIR}/src/lve_device.cpp
  ${PROJECT_SOURCE_DIR}/src/lve_window.cpp
  ${PROJECT_SOURCE_DIR}/src/lve_buffer.cpp
  ${PROJECT_SOURCE_DIR}/src/lve_descriptors.cpp
)
target_compile_features(MctsBench PUBLIC cxx_std_17)
if (MSVC)
  target_compile_options(MctsBench PRIVATE /utf-8)
endif()
target_include_directories(MctsBench PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/external/glfw/include/GLFW
  ${VULKAN_SDK_PATH}/Include
  ${VULKAN_SDK_PATH}/Include/vulkan
)
target_link_directories(MctsBench PUBLIC ${VULKAN_SDK_PATH}/Lib)
target_link_libraries(MctsBench glfw vulkan-1)
