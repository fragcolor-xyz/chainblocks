if(UNIX AND NOT (APPLE OR ANDROID OR EMSCRIPTEN))
  set(DESKTOP_LINUX TRUE)
endif()

if(APPLE AND NOT IOS)
  set(MACOSX TRUE)
endif()

if(NOT EMSCRIPTEN AND (WIN32 OR MACOSX OR DESKTOP_LINUX))
  set(DESKTOP TRUE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set(X86 TRUE)
else()
  set(X86 FALSE)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  add_compile_definitions(CPUBITS64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  add_compile_definitions(CPUBITS32)
endif()

# Default arch if ARCH is not set
if(NOT ARCH)
  if(X86 AND NOT (EMSCRIPTEN OR APPLE))
    set(ARCH "sandybridge")
  endif()
endif()

if(ARCH)
  add_compile_options(-march=${ARCH})
endif()

if(USE_FPIC)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  list(APPEND EXTERNAL_CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DSDL_STATIC_PIC=ON)
endif()

# Force ninja to use response files on windows when command line might be too long otherwise
if(CMAKE_HOST_WIN32)
  SET(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "" FORCE)
endif()

if(EMSCRIPTEN)
  add_compile_options(-fdeclspec)

  if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-g1 -Os)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_link_options("SHELL:-s ASSERTIONS=2")
  endif()

  add_compile_definitions(NO_FORCE_INLINE)
  add_link_options(--bind)

  ## if we wanted thread support...
  if(EMSCRIPTEN_PTHREADS)
    add_link_options("SHELL:-s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=6")
    add_compile_options(-pthread -Wno-pthreads-mem-growth)
    add_link_options(-pthread)
    set(HAVE_THREADS ON)
  else()
    add_compile_options(-DBOOST_ASIO_DISABLE_THREADS=1)
  endif()

  if(NODEJS)
    add_link_options(-lnodefs.js)
  else()
    add_link_options(-lidbfs.js)
  endif()
else()
  set(HAVE_THREADS ON)
endif()

if(MSVC OR CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
  set(WINDOWS_ABI "msvc")

  add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
  add_compile_definitions(_SCL_SECURE_NO_WARNINGS=1)
  add_compile_definitions(NOMINMAX=1)

  # We can not keep iterators in memory without freeing with iterator debugging
  # See SHTable/Set iterator internals
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=1)
    list(APPEND EXTERNAL_CMAKE_ARGS -DCMAKE_CXX_FLAGS="-D_ITERATOR_DEBUG_LEVEL=1")
  endif()
else()
  set(WINDOWS_ABI "gnu")
endif()

# mingw32 static runtime
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(WIN32)
    add_link_options(-static)
    list(APPEND EXTERNAL_CMAKE_ARGS -DCMAKE_CXX_FLAGS=-static -DCMAKE_C_FLAGS=-static)
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  # aggressive inlining
  if(NOT SKIP_HEAVY_INLINE)
    # this works with emscripten too but makes the final binary much bigger
    # for now let's keep it disabled
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(INLINING_FLAGS -mllvm -inline-threshold=100000)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(INLINING_FLAGS -finline-limit=100000)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      # using Intel C++
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      # using Visual Studio C++
    endif()
  endif()
endif()

if(USE_LTO)
  add_compile_options(-flto)
  add_link_options(-flto)
endif()

add_compile_options(-Wall)

if(NOT MSVC)
  add_compile_options(-ffast-math -fno-finite-math-only -funroll-loops -Wno-multichar)
endif()

if(WIN32 AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
  set(USE_LLD_DEFAULT ON)
endif()
option(USE_LLD "Override linker tools to use lld & llvm-ar/ranlib" ${USE_LLD_DEFAULT})
if(USE_LLD)
  add_link_options(-fuse-ld=lld)
  SET(CMAKE_AR llvm-ar)
  SET(CMAKE_RANLIB llvm-ranlib)
endif()

if(DESKTOP_LINUX)
  add_link_options(-export-dynamic)
  if(USE_VALGRIND)
    add_compile_definitions(BOOST_USE_VALGRIND SHARDS_NO_BIGINT_SHARDS)
  endif()
endif()

if(USE_GCC_ANALYZER)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fanalyzer)
  endif()
endif()

if(PROFILE_GPROF)
  add_compile_options(-pg -DNO_FORCE_INLINE)
  add_link_options(-pg)
endif()

option(USE_UBSAN "Use undefined behaviour sanitizer" ON)
set(UBSAN_SUPPORTED DESKTOP_LINUX)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(${UBSAN_SUPPORTED} AND USE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
    add_compile_definitions(SH_USE_UBSAN)
  endif()
endif()

if(CODE_COVERAGE)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(--coverage)
    add_link_options(--coverage)
  endif()
endif()

# Move this?
if(NOT SPDLOG_ACTIVE_LEVEL)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO)
    else()
      set(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG)
    endif()
  else()
    set(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE)
  endif()
endif()
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL})

# Move this?
if(VERBOSE_BGFX_LOGGING)
  add_compile_definitions(BGFX_CONFIG_DEBUG)
endif()

add_compile_definitions(BOOST_INTERPROCESS_BOOTSTAMP_IS_LASTBOOTUPTIME=1)

if(ANDROID OR APPLE)
  # This tells FindPackage(Threads) that threads are built in
  if(ANDROID)
    # Bundled in the standard C library
    set(CMAKE_THREAD_LIBS_INIT "-lc")
  else()
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  endif()
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

if(APPLE)
  add_compile_definitions(BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
  add_compile_options(-Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()


if(MSVC OR CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
  set(LIB_PREFIX "")
  set(LIB_SUFFIX ".lib")
else()
  set(LIB_PREFIX "lib")
  set(LIB_SUFFIX ".a")
endif()
