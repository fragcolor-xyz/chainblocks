set(BUILD_SHARED_LIBS OFF)

# Find & cache git using (only considering host tools)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH ON)
find_package(Git REQUIRED)

include(SHExternalProject)
include(FetchContent)

add_subdirectory(filesystem)
add_subdirectory(json)

add_subdirectory(kcp)
add_library(kcp-wrapper INTERFACE)
target_link_libraries(kcp-wrapper INTERFACE kcp)
target_include_directories(kcp-wrapper INTERFACE kcp)

add_subdirectory(magic_enum)
add_subdirectory(nameof)
add_subdirectory(replxx)
add_subdirectory(spdlog)
add_subdirectory(sqlite)

set(DEBUG_RENDERER_IN_DISTRIBUTION ON CACHE BOOL "" FORCE)

# Don't enable these ones
option(USE_AVX2 "Enable AVX2" OFF)
option(USE_AVX512 "Enable AVX512" OFF)

# Required by Jolt as we enable RTTI
set(CPP_RTTI_ENABLED ON CACHE BOOL "Enable C++ RTTI" FORCE)
if(EMSCRIPTEN)
  set(CPP_EXCEPTIONS_ENABLED ON CACHE BOOL "Enable C++ exceptions" FORCE) # wasm requires exceptions to be enabled cos we use exceptions
endif()

# This triggers warnings+errors on valgrind build
option(ENABLE_ALL_WARNINGS "Enable all warnings" OFF)

# Required by Jolt to build with WASM SIMD
option(USE_WASM_SIMD "Enable WASM SIMD" ON)

add_subdirectory(JoltPhysics/Build)

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE OFF CACHE BOOL "" FORCE)
endif()

set(TBB_STRICT OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON CACHE BOOL "" FORCE)
add_subdirectory(oneTBB)

if(NOT EMSCRIPTEN)
  if(DESKTOP_LINUX)
    if(GNU_STATIC_BUILD)
      set(OPENSSL_USE_STATIC_LIBS TRUE)
    endif()

    find_package(OpenSSL REQUIRED)
    add_library(OpenSSL INTERFACE)
    target_link_libraries(OpenSSL INTERFACE OpenSSL::SSL)
    target_include_directories(OpenSSL INTERFACE ${OPENSSL_INCLUDE_DIR})

    message(STATUS "Using system OpenSSL: ${OPENSSL_SSL_LIBRARY}, ${OPENSSL_INCLUDE_DIR}")
  else()
    # Force default config path to system's /etc/ssl
    # Otherwise libressl will use the install prefix
    set(OPENSSLDIR "/etc/ssl" CACHE PATH "" FORCE)

    set(LIBRE_SSL_DIR "" CACHE FILEPATH "Use libressl from this path, not setting this will download the sources automatically")

    if(LIBRE_SSL_DIR)
      set(libressl_SOURCE_DIR ${LIBRE_SSL_DIR})

      message(STATUS "Using libressl from LIBRE_SSL_DIR=${libressl_SOURCE_DIR}")
      add_subdirectory(${libressl_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/libressl)
    else()
      message(STATUS "Fetching libressl")
      FetchContent_Declare(
        libressl
        URL https://develop.fragcolor.com/libressl-3.9.2.tar.gz
        URL_HASH SHA256=7b031dac64a59eb6ee3304f7ffb75dad33ab8c9d279c847f92c89fb846068f97
        ${LIBRESSL_PATCH_COMMAND}
      )
      FetchContent_MakeAvailable(libressl)
    endif()

    # Set source dir property used later by rust openssl-sys
    message(STATUS "libressl_SOURCE_DIR=${libressl_SOURCE_DIR}")
    set_property(GLOBAL PROPERTY libressl_SOURCE_DIR ${libressl_SOURCE_DIR})

    # NOTE: External project doesn't work with Xcode here
    # since ssl/cypto builds into object library referenced by static library
    add_library(OpenSSL INTERFACE)
    add_library(OpenSSL::SSL ALIAS OpenSSL)
    add_library(OpenSSL::Crypto ALIAS crypto)
    target_link_libraries(OpenSSL INTERFACE ssl)
    target_include_directories(OpenSSL INTERFACE ${libressl_SOURCE_DIR}/include)
  endif()
endif()

if(WIN32)
  set(BOOST_STACKTRACE_ENABLE_WINDBG_CACHED ON CACHE BOOL "")
endif()

set(BOOST_DIR "" CACHE FILEPATH "Use boost from this path, not setting this will download the sources automatically")

if(BOOST_DIR)
  message(STATUS "Using boost from BOOST_DIR=${BOOST_DIR}")
  add_subdirectory(${BOOST_DIR} ${CMAKE_CURRENT_BINARY_DIR}/boost)
else()
  message(STATUS "Fetching boost")
  FetchContent_Declare(
    boost
    URL https://develop.fragcolor.com/boost-1.81.0-shards-5.7z

    # URL      https://github.com/shards-lang/boost/releases/download/boost-1.81.0-shards-5/output.7z
    URL_HASH SHA256=a890f672200cf9f018cbd75a433a4dd7837e09a97b0b9287861a497a4a744626
  )
  FetchContent_MakeAvailable(boost)
  message(STATUS "boost_SOURCE_DIR=${boost_SOURCE_DIR}")
endif()

if(NOT EMSCRIPTEN)
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON CACHE BOOL "" FORCE)
  add_subdirectory(SDL3)
endif()

sh_add_external_project(
  NAME wasm3_a
  TARGETS m3
  LIB_RELATIVE_DIRS source/
  CMAKE_ARGS -DBUILD_NATIVE=0
  RELATIVE_INCLUDE_PATHS source
  REPO_ARGS URL ${CMAKE_CURRENT_SOURCE_DIR}/wasm3)

sh_add_external_project(
  NAME snappy_a
  TARGETS snappy
  CMAKE_ARGS -DSNAPPY_BUILD_TESTS=0
  RELATIVE_INCLUDE_PATHS .
  RELATIVE_BINARY_INCLUDE_PATHS .
  REPO_ARGS GIT_REPOSITORY https://github.com/shards-lang/snappy.git
  GIT_TAG 9e3775b)

sh_add_external_project(
  NAME brotli_a
  TARGETS brotlidec-static brotlienc-static brotlicommon-static
  CMAKE_ARGS -DBROTLI_BUNDLED_MODE=1
  RELATIVE_INCLUDE_PATHS c/include
  REPO_ARGS GIT_REPOSITORY https://github.com/shards-lang/brotli.git
  GIT_TAG e83c7b8e8fb8b696a1df6866bc46cbb76d7e0348)

add_subdirectory(Catch2)

sh_add_external_project(
  NAME kissfft_a
  TARGETS kissfft
  LIB_SUFFIX -float
  CMAKE_ARGS -DKISSFFT_PKGCONFIG=0 -DKISSFFT_STATIC=1 -DKISSFFT_TEST=0 -DKISSFFT_TOOLS=0
  RELATIVE_INCLUDE_PATHS .
  REPO_ARGS URL ${SHARDS_DIR}/deps/kissfft)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Draco doesn't expose include paths directly, add them here using a wrapper
set(DRACO_JS_GLUE OFF CACHE BOOL "" FORCE)
add_subdirectory(draco)
add_library(draco-wrapper INTERFACE)
target_link_libraries(draco-wrapper INTERFACE draco::draco)
target_include_directories(draco-wrapper INTERFACE draco/src)
target_include_directories(draco-wrapper INTERFACE ${CMAKE_BINARY_DIR})

set(TF_BUILD_TESTS OFF CACHE BOOL "")
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "")
add_subdirectory(cpp-taskflow)

add_library(xxHash INTERFACE)
target_include_directories(xxHash INTERFACE xxHash)

add_library(linalg INTERFACE)
target_include_directories(linalg INTERFACE linalg ${CMAKE_CURRENT_LIST_DIR})

add_library(miniaudio INTERFACE)
target_include_directories(miniaudio INTERFACE miniaudio)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)

add_library(utf8.h INTERFACE)
target_include_directories(utf8.h INTERFACE utf8.h)

add_library(pdqsort INTERFACE)
target_include_directories(pdqsort INTERFACE pdqsort)

add_library(entt INTERFACE)
target_include_directories(entt INTERFACE entt/src)

# Tracy profiler, default off
option(TRACY_ENABLE "Enable profiling" OFF)

# Enable fibers support by default
option(TRACY_FIBERS "Enable fibers support" ON)
option(TRACY_CALLSTACK_SIZE "Tracy callstack size" 8)

# These settings are required to make tracy from C++ work together with rust's tracy-client
set(TRACY_MANUAL_LIFETIME ON CACHE BOOL "" FORCE)
set(TRACY_DELAYED_INIT ON CACHE BOOL "" FORCE)
add_subdirectory(tracy)

# Add the wrapper include path that contains tracy/Wrapper.hpp that defines the correct TRACY_CALLSTACK based on the stack size configuration
target_include_directories(TracyClient PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tracy_wrapper>)
target_compile_definitions(TracyClient PUBLIC TRACY_CALLSTACK_SIZE=${TRACY_CALLSTACK_SIZE})

if(NOT EMSCRIPTEN)
  # TODO vulkan build support
  # if(WIN32 OR LINUX)
  #   set(GGML_VULKAN ON CACHE BOOL "" FORCE)
  # endif()
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_AVX ON CACHE BOOL "" FORCE)
  set(GGML_AVX2 ON CACHE BOOL "" FORCE)
  set(GGML_F16C ON CACHE BOOL "" FORCE)
  set(GGML_FMA ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE) # omp is very omp...
  set(GGML_METAL_USE_BF16 ON CACHE BOOL "" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
  # Force math library to be empty, sometimes picks up libm.a from mingw64
  if(WIN32)
    set(MATH_LIBRARY "" CACHE FILEPATH "" FORCE)
  endif()
  add_subdirectory(llama.cpp)
endif()
