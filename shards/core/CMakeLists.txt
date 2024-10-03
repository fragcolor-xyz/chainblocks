set(core_SOURCES
  async.cpp
  ops_internal.cpp
  number_types.cpp
  utils.cpp
  coro.cpp
  wire_dsl.cpp
  trait.cpp
  type_info.cpp
  hash.cpp
  taskflow.cpp
  pmr/shared_temp_allocator.cpp
)

if(EMSCRIPTEN)
  list(APPEND core_SOURCES em_proxy.cpp)
endif()

add_library(shards-shared-temp-allocator STATIC pmr/shared_temp_allocator.cpp)
target_include_directories(shards-shared-temp-allocator PUBLIC
  ${SHARDS_DIR}/include
  ${SHARDS_DIR})
target_link_libraries(shards-shared-temp-allocator Tracy::TracyClient spdlog shards-logging Boost::container)

add_library(shards-core STATIC ${core_SOURCES})

target_link_libraries(shards-core shards-shared-temp-allocator)

target_include_directories(shards-core PUBLIC ${SHARDS_DIR}/include)
target_include_directories(shards-core PUBLIC ${SHARDS_DIR})

# Expose tracy client so all dependencies get the TRACY_ENABLE define
target_link_libraries(shards-core Tracy::TracyClient shards-fast-string)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(SHARDS_INLINE_EVERYTHING_DEFAULT OFF)
else()
  set(SHARDS_INLINE_EVERYTHING_DEFAULT ON)
endif()

if(NOT DEFINED SHARDS_INLINE_EVERYTHING)
  set(SHARDS_INLINE_EVERYTHING ${SHARDS_INLINE_EVERYTHING_DEFAULT} CACHE BOOL "When enabled, runtime and core shards are inlined in a single compilation unit")
endif()

if(SHARDS_INLINE_EVERYTHING)
  target_compile_definitions(shards-core PUBLIC SHARDS_INLINE_EVERYTHING=1)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(SHARDS_DEBUG_THREAD_NAMES_DEFAULT ON)
else()
  set(SHARDS_DEBUG_THREAD_NAMES_DEFAULT OFF)
endif()

set(SHARDS_DEBUG_THREAD_NAMES ${SHARDS_DEBUG_THREAD_NAMES_DEFAULT} CACHE BOOL "When enabled, threads are given names based on running wires")

if(SHARDS_DEBUG_THREAD_NAMES)
  target_compile_definitions(shards-core PUBLIC SH_DEBUG_THREAD_NAMES=1)
endif()

set(SHARDS_THREAD_FIBER OFF CACHE BOOL "When enabled, coroutines are evaluated using a single thread (for debugging purposes)")

if(SHARDS_THREAD_FIBER)
  target_compile_definitions(shards-core PUBLIC SH_USE_THREAD_FIBER=1)
endif()

# # Uncomment to enable consistent resumer debugging, this might be useful to debug some specific issues when coroutines cannot be resumed on different threads
if(CMAKE_BUILD_TYPE MATCHES Debug AND NOT SHARDS_THREAD_FIBER AND NOT EMSCRIPTEN)
  # Set to non-zero to enable, Set to 2 to enable more detailed logging
  target_compile_definitions(shards-core PUBLIC SH_DEBUG_CONSISTENT_RESUMER=1)
endif()

if(WIN32)
  target_link_libraries(shards-core Boost::stacktrace_windbg_cached)

  # Target windows SDK (https://docs.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt)
  # 0x0A00 = Windows 10
  target_compile_definitions(shards-core PUBLIC WINVER=0x0A00 _WIN32_WINNT=0x0A00)
elseif(DESKTOP_LINUX)
  target_link_libraries(shards-core Boost::stacktrace_addr2line)
else()
  target_link_libraries(shards-core Boost::stacktrace_basic)
endif()

target_precompile_headers(shards-core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp>")

if(APPLE)
  target_link_libraries(shards-core
    "-framework Foundation"
    "-framework CoreAudio"
    "-framework AudioToolbox"
    "-framework CoreVideo"
    "-framework IOKit"
    "-framework QuartzCore"
    "-framework Accelerate"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework Security"
    "-framework CoreHaptics"
    "-framework GameController"
    "-framework SystemConfiguration"
    -liconv
  )

  if(IOS OR VISIONOS)
    target_link_libraries(shards-core
      "-framework AVFoundation"
      "-framework CoreGraphics"
      "-framework UIKit"
    )

    if(NOT SKIP_CORE_MOTION)
      target_link_libraries(shards-core "-framework CoreMotion")
    endif()
  else()
    target_link_libraries(shards-core
      "-framework Cocoa"
      "-framework Carbon"
      "-framework ForceFeedback"
    )
  endif()
endif()

target_link_libraries(shards-core Boost::filesystem Boost::lockfree Boost::foreach Boost::multiprecision Boost::atomic Boost::thread Boost::tti)

if(NOT EMSCRIPTEN)
  target_link_libraries(shards-core Boost::beast Boost::asio Boost::context)
else()
  target_include_directories(shards-core PUBLIC $<TARGET_PROPERTY:Boost::asio,INTERFACE_INCLUDE_DIRECTORIES>)

  # For usage of boost/beast/core/detail/base64.hpp
  target_include_directories(shards-core PUBLIC $<TARGET_PROPERTY:Boost::beast,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

target_link_libraries(shards-core
  spdlog magic_enum nameof linalg xxHash
  pdqsort entt utf8.h Taskflow stb nlohmann_json
  shards-logging tbb
)

if(DESKTOP)
  target_compile_definitions(shards-core PUBLIC SHARDS_DESKTOP=1)
endif()

if(WIN32)
  target_link_libraries(shards-core ws2_32 mswsock Secur32 Crypt32 NCrypt)
elseif(DESKTOP_LINUX)
  target_link_libraries(shards-core pthread dl rt)
endif()

target_compile_options(shards-core
  PRIVATE -ffast-math -fno-finite-math-only -funroll-loops
)

if(EMSCRIPTEN)
  target_compile_options(shards-core PUBLIC)
  target_link_options(shards-core PUBLIC "-sLLD_REPORT_UNDEFINED")

  if(NOT SHARDS_THREAD_FIBER)
    target_link_options(shards-core PUBLIC "-sASYNCIFY=1")
  endif()

  # Require optimizations when building with asyncify too keep size/build time reasonable
  # if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # target_compile_options(shards-core PUBLIC "-O3")
  # target_link_options(shards-core PUBLIC "-O3")
  # endif()
endif()

duplicate_library_target(shards-core SHARED shards-core-shared)
target_compile_definitions(shards-core-shared
  PUBLIC SHARDS_CORE_DLL=1
  PRIVATE shards_core_EXPORTS=1)

option(USE_UBSAN "Use undefined behaviour sanitizer" OFF)

if(DESKTOP_LINUX OR APPLE AND USE_UBSAN)
  target_compile_options(shards-core PUBLIC -fsanitize=undefined)
  target_link_options(shards-core PUBLIC -fsanitize=undefined)
  add_compile_definitions(SH_USE_UBSAN)
endif()
