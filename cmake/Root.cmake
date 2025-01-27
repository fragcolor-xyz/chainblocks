# Silence DOWNLOAD_EXTRACT_TIMESTAMP warning
# because of FetchContent_Declare usage
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

get_filename_component(SHARDS_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
message(STATUS "SHARDS_DIR = ${SHARDS_DIR}")

option(SHARDS_BUILD_TESTS "Enable to build shards tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${SHARDS_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${SHARDS_DIR}/cmake/android-cmake")

include(Sanitizers)
include(Platform)
include(Tidy)
include(Utils)
include(Tools)
include(Modules)

# Rust build support
include(Rust)

# disable targets being added by CTestTargets.cmake
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

add_subdirectory(${SHARDS_DIR}/deps deps)

# Standalone libraries
add_subdirectory(${SHARDS_DIR}/shards/fast_string src/fast_string)
add_subdirectory(${SHARDS_DIR}/shards/log src/log)
add_subdirectory(${SHARDS_DIR}/shards/input src/input)
add_subdirectory(${SHARDS_DIR}/shards/gfx src/gfx)

# Shards core
add_subdirectory(${SHARDS_DIR}/shards/core src/core)

# Rust projects
add_subdirectory(${SHARDS_DIR}/shards/rust src/rust)

# Modules
set(SHARDS_MODULE_ROOT ${SHARDS_DIR}/shards/modules)
macro(add_module NAME)
  add_subdirectory(${SHARDS_MODULE_ROOT}/${NAME} modules/${NAME})
  list(APPEND ADDED_MODULES ${NAME})
endmacro()

# Fixed-order modules (should be here if they have dependencies)
add_module(gfx)
add_module(egui)
add_module(spatial)

message(STATUS "ADDED_MODULES = ${ADDED_MODULES}")

# Automatic scan for remaining modules
file(GLOB SHARDS_MODULE_FOLDERS RELATIVE ${SHARDS_MODULE_ROOT} ${SHARDS_MODULE_ROOT}/*)
foreach(MODULE_FOLDER ${SHARDS_MODULE_FOLDERS})
  list(FIND ADDED_MODULES ${MODULE_FOLDER} ALREADY_ADDED)
  if(EXISTS ${SHARDS_MODULE_ROOT}/${MODULE_FOLDER}/CMakeLists.txt AND ALREADY_ADDED LESS 0)
    add_subdirectory(${SHARDS_MODULE_ROOT}/${MODULE_FOLDER} modules/${MODULE_FOLDER})
  endif()
endforeach()

# Union library that ties the modules together
add_subdirectory(${SHARDS_DIR}/shards/union src/union)

# Shards library and executable bundle
add_subdirectory(${SHARDS_DIR}/shards/cli src/cli)

add_subdirectory(${SHARDS_DIR}/shards/tests src/tests)

# Automatically find subprojects
set(SHARDS_SUBPROJECT_ROOT "${SHARDS_DIR}/external")
file(GLOB SHARDS_SUBPROJECT_FOLDERS RELATIVE ${SHARDS_SUBPROJECT_ROOT} ${SHARDS_SUBPROJECT_ROOT}/*)
foreach(SUBPROJECT_FOLDER ${SHARDS_SUBPROJECT_FOLDERS})
  if(EXISTS ${SHARDS_SUBPROJECT_ROOT}/${SUBPROJECT_FOLDER}/CMakeLists.txt)
    message(STATUS "Adding subproject: ${SUBPROJECT_FOLDER}")
    add_subdirectory(${SHARDS_SUBPROJECT_ROOT}/${SUBPROJECT_FOLDER} external/${SUBPROJECT_FOLDER})
  endif()
endforeach()

# Add manually specified subproject paths
set(SHARDS_SUBPROJECTS "" CACHE FILEPATH "List of paths to subprojects to integrate into the main build")
foreach(SUBPROJECT_PATH ${SHARDS_SUBPROJECTS})
  get_filename_component(SUBPROJECT_PATH_ABS "${SUBPROJECT_PATH}" ABSOLUTE BASE_DIR "${SHARDS_DIR}")
  if(EXISTS ${SUBPROJECT_PATH_ABS}/CMakeLists.txt)
    get_filename_component(FILENAME ${SUBPROJECT_PATH_ABS} NAME)
    message(STATUS "Adding subproject: ${SUBPROJECT_PATH_ABS} (as ${FILENAME})")
    add_subdirectory(${SUBPROJECT_PATH_ABS} external/${FILENAME})
  endif()
endforeach()
