if(UNIX AND NOT(APPLE OR ANDROID OR EMSCRIPTEN))
  set(DESKTOP_LINUX TRUE)
endif()

if(APPLE)
  if(CMAKE_SYSTEM_NAME MATCHES "visionOS")
    set(VISIONOS TRUE)
  endif()

  # Set deployment target BEFORE any language enabling or project() calls
  if(IOS)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "16.3" CACHE STRING "Minimum iOS deployment version" FORCE)
    set(deployment_target_flag "-target ${CMAKE_SYSTEM_PROCESSOR}-apple-ios${CMAKE_OSX_DEPLOYMENT_TARGET}")
  elseif(VISIONOS)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "1.2" CACHE STRING "Minimum visionOS deployment version" FORCE)
    set(deployment_target_flag "-target ${CMAKE_SYSTEM_PROCESSOR}-apple-xros${CMAKE_OSX_DEPLOYMENT_TARGET}")
  else()
    set(MACOSX TRUE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "Minimum macOS deployment version" FORCE)
    set(deployment_target_flag "-target ${CMAKE_SYSTEM_PROCESSOR}-apple-macosx${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()

  # Add the deployment target flag to Swift compiler options
  set(CMAKE_Swift_FLAGS "${CMAKE_Swift_FLAGS} ${deployment_target_flag}" CACHE STRING "Swift compiler flags" FORCE)
  
  set(CMAKE_Swift_COMPILER /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc)
  enable_language(Swift)
  set(CMAKE_Swift_LANGUAGE_VERSION 5)

  # Add link options for Swift targets
  add_link_options("SHELL:$<$<LINK_LANGUAGE:Swift>:${deployment_target_flag}>")

  enable_language(OBJC)

  message(STATUS "MACOSX: ${MACOSX}")
  message(STATUS "CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
  message(STATUS "Swift deployment target flag: ${deployment_target_flag}")
endif()

if(NOT EMSCRIPTEN AND(WIN32 OR MACOSX OR DESKTOP_LINUX))
  set(DESKTOP TRUE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" AND NOT EMSCRIPTEN)
  set(X86 TRUE)
  set(ARM FALSE)
else()
  set(X86 FALSE)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm)|(ARM)|(aarch64)|(AARCH64)")
    set(ARM TRUE)

    # Check for NEON support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)

    if(COMPILER_SUPPORTS_NEON)
      set(ARM_NEON TRUE)
    else()
      set(ARM_NEON FALSE)
    endif()
  else()
    set(ARM FALSE)
    set(ARM_NEON FALSE)
  endif()
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  add_compile_definitions(CPUBITS64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  add_compile_definitions(CPUBITS32)
endif()

# Default arch if ARCH is not set
if(NOT ARCH)
  if(X86)
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
      set(ARCH "pentium4")
    else()
      set(ARCH "broadwell")
    endif()
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

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  # define SH_RELWITHDEBINFO to enable some extra debug asserts
  add_compile_definitions(SH_RELWITHDEBINFO)
endif()

option(SHARDS_MIN_DEBUG_INFO "Use minimal debug info" OFF)

function(fixup_debug_flags VARNAME)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
  set(TMP_VARNAME "${VARNAME}_${CMAKE_BUILD_TYPE_UPPER}")
  string(REPLACE " " ";" TMP_FLAGS ${${TMP_VARNAME}})
  list(REMOVE_ITEM TMP_FLAGS -g)
  list(REMOVE_ITEM TMP_FLAGS -gline-tables-only)
  list(APPEND TMP_FLAGS -gline-tables-only)
  string(JOIN " " TMP_OUT_STR ${TMP_FLAGS})
  set(${TMP_VARNAME} ${TMP_OUT_STR} CACHE STRING "" FORCE)
  message(STATUS "Minimal debug info ${TMP_VARNAME}: ${TMP_FLAGS} (${${TMP_VARNAME}})")
endfunction()

if(SHARDS_MIN_DEBUG_INFO)
  fixup_debug_flags("CMAKE_CXX_FLAGS")
  fixup_debug_flags("CMAKE_C_FLAGS")
endif()

if(EMSCRIPTEN)
  add_compile_options(-fdeclspec)

  # Enable web simd
  add_compile_options(-msimd128)

  # if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  # add_compile_options(-g1 -Os)
  # endif()
  if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_link_options("-sASSERTIONS=2")

    # add_link_options(-gsource-map)
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-sDISABLE_EXCEPTION_CATCHING=0>")
  add_link_options("-sDISABLE_EXCEPTION_CATCHING=0")

  add_compile_definitions(NO_FORCE_INLINE)
  add_link_options(-lembind)

  # # if we wanted thread support...
  if(EMSCRIPTEN_PTHREADS)
    add_link_options("-sUSE_PTHREADS=1")
    add_compile_options(-pthread -Wno-pthreads-mem-growth)
    add_link_options(-pthread)
    set(HAVE_THREADS ON)
  else()
    add_compile_options(-DBOOST_ASIO_DISABLE_THREADS=1)
  endif()

# TODO: move this to application specific code
# if(NODEJS)
# add_link_options(-lnodefs.js)
# else()
# add_link_options(-lidbfs.js)
# endif()
else()
  set(HAVE_THREADS ON)
endif()

if(WIN32)
  add_compile_definitions(_CRT_SECURE_NO_DEPRECATE=1)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS=1)
  add_compile_definitions(_CRT_NONSTDC_NO_WARNINGS=1)
  add_compile_definitions(NOMINMAX=1)

  if(X86 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    # align stack to 16 bytes
    add_compile_options(-mstackrealign)
  endif()
endif()

if(EMSCRIPTEN)
  set(EXTERNAL_BUILD_TYPE "Release")
endif()

if(MSVC OR CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
  set(WINDOWS_ABI "msvc")

  # We can not keep iterators in memory without freeing with iterator debugging
  # See SHTable/Set iterator internals
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=1)
    list(APPEND EXTERNAL_CMAKE_ARGS -DCMAKE_CXX_FLAGS="-D_ITERATOR_DEBUG_LEVEL=1")
  endif()
else()
  set(WINDOWS_ABI "gnu")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  # static releases
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

  if(GNU_STATIC_BUILD)
    add_link_options(-static -static-libgcc -static-libstdc++)
  endif()

  # aggressive inlining
  if(NOT SKIP_HEAVY_INLINE)
    # Adjust inlining based on the compiler
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "[A-Za-z]*Clang")
      set(INLINING_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:-mllvm>
        $<$<COMPILE_LANGUAGE:CXX>:-inline-threshold=2500>
      )
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(INLINING_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:-finline-functions-called-once>
        $<$<COMPILE_LANGUAGE:CXX>:-finline-small-functions>
      )
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      set(INLINING_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:-inline-level=2>
        $<$<COMPILE_LANGUAGE:CXX>:-inline-factor=100>
      )
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      set(INLINING_FLAGS
        $<$<COMPILE_LANGUAGE:CXX>:/Ob2>
      )
    endif()
  endif()
endif()

if(USE_LTO)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX,C>:-flto>)
  add_link_options($<$<COMPILE_LANGUAGE:CXX,C>:-flto>)

  if(APPLE AND CMAKE_SWIFT_COMPILER)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:Swift>:-O>
      $<$<COMPILE_LANGUAGE:Swift>:-wmo>
    )
    add_link_options(
      $<$<COMPILE_LANGUAGE:Swift>:-O>
      $<$<COMPILE_LANGUAGE:Swift>:-wmo>
    )
  endif()
endif()

add_compile_options(
  ${INLINING_FLAGS}
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>
)

if(NOT MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-multichar>
  )
endif()

if(WIN32 AND(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
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
    add_compile_definitions(BOOST_USE_VALGRIND SHARDS_VALGRIND)
  endif()
endif()

if(USE_GCC_ANALYZER)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fanalyzer)
  endif()
endif()

if(PROFILE_GPROF)
  add_compile_options(-pg -DNO_FORCE_INLINE)
  add_link_options(-pg)
endif()

option(USE_ASAN "Use address sanitizer" OFF)

if(USE_ASAN)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-DBOOST_USE_ASAN>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fno-optimize-sibling-calls>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize-address-use-after-scope>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX,C>:-g>
  )
  add_link_options(
    $<$<COMPILE_LANGUAGE:CXX>:-DBOOST_USE_ASAN>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fno-optimize-sibling-calls>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize-address-use-after-scope>
    $<$<COMPILE_LANGUAGE:CXX,C>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX,C>:-g>
  )

  if(USE_ASAN GREATER 1)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX,C>:-O1>
    )
    add_link_options(
      $<$<COMPILE_LANGUAGE:CXX,C>:-O1> # O1
    )
  endif()

  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX,C>:SH_USE_ASAN>)
endif()

option(USE_TSAN "Use thread sanitizer" OFF)

if(USE_TSAN)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=thread>
    $<$<COMPILE_LANGUAGE:CXX,C>:-g>
  )
  add_link_options(
    $<$<COMPILE_LANGUAGE:CXX,C>:-fsanitize=thread>
    $<$<COMPILE_LANGUAGE:CXX,C>:-g>
  )
  if(USE_TSAN GREATER 1)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX,C>:-O1>
    )
    add_link_options(
      $<$<COMPILE_LANGUAGE:CXX,C>:-O1>
    )
  endif()
  add_compile_definitions($<$<COMPILE_LANGUAGE:CXX,C>:SH_USE_TSAN>)
endif()

if(CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
    OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "[A-Za-z]*Clang"
  )
    add_compile_options($<$<COMPILE_LANGUAGE:CXX,C>:--coverage>)
    add_link_options($<$<COMPILE_LANGUAGE:CXX,C>:--coverage>)
  else()
    message(FATAL_ERROR "Code coverage is not supported for the '${CMAKE_CXX_COMPILER_ID}' compiler")
  endif()
endif()

# Move this?
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
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>
  )
endif()

if(MSVC OR CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
  set(LIB_PREFIX "")
  set(LIB_SUFFIX ".lib")
else()
  set(LIB_PREFIX "lib")
  set(LIB_SUFFIX ".a")
endif()
