#### Files to format and tidy
set(
  MY_PROJECT_SOURCE_FILES
  ${MY_PROJECT_SOURCE_FILES}
  ${SHARDS_DIR}/include/shardwrapper.hpp
  ${SHARDS_DIR}/include/dllshard.hpp
  ${SHARDS_DIR}/include/utility.hpp
  ${SHARDS_DIR}/include/shards.hpp
  ${SHARDS_DIR}/include/shards.h
  ${SHARDS_DIR}/include/common_types.hpp
  ${SHARDS_DIR}/include/ops.hpp
  ${SHARDS_DIR}/include/linalg_shim.hpp
  ${SHARDS_DIR}/include/wire_dsl.hpp
  ${SHARDS_DIR}/src/core/runtime.cpp
  ${SHARDS_DIR}/src/core/ops_internal.cpp
  ${SHARDS_DIR}/src/core/runtime.hpp
  ${SHARDS_DIR}/src/core/foundation.hpp
  ${SHARDS_DIR}/src/core/ops_internal.hpp
  ${SHARDS_DIR}/src/core/shards/process.cpp
  ${SHARDS_DIR}/src/core/shards_macros.hpp
  ${SHARDS_DIR}/src/core/coro.cpp
  ${SHARDS_DIR}/src/core/shards/assert.cpp
  ${SHARDS_DIR}/src/core/shards/wires.cpp
  ${SHARDS_DIR}/src/core/shards/logging.cpp
  ${SHARDS_DIR}/src/core/shards/seqs.cpp
  ${SHARDS_DIR}/src/core/shards/shared.hpp
  ${SHARDS_DIR}/src/core/shards/flow.cpp
  ${SHARDS_DIR}/src/core/shards/casting.cpp
  ${SHARDS_DIR}/src/core/shards/core.hpp
  ${SHARDS_DIR}/src/core/shards/core.cpp
  ${SHARDS_DIR}/src/core/shards/math.hpp
  ${SHARDS_DIR}/src/core/shards/linalg.cpp
  ${SHARDS_DIR}/src/core/shards/fs.cpp
  ${SHARDS_DIR}/src/core/shards/json.cpp
  ${SHARDS_DIR}/src/core/shards/network.cpp
  ${SHARDS_DIR}/src/core/shards/struct.cpp
  ${SHARDS_DIR}/src/core/shards/channels.cpp
  ${SHARDS_DIR}/src/core/shards/genetic.hpp
  ${SHARDS_DIR}/src/core/shards/random.cpp
  ${SHARDS_DIR}/src/core/shards/imaging.cpp
  ${SHARDS_DIR}/src/core/shards/http.cpp
  ${SHARDS_DIR}/src/extra/audio.cpp
  ${SHARDS_DIR}/src/extra/runtime.cpp
  ${SHARDS_DIR}/src/extra/desktop.hpp
  ${SHARDS_DIR}/src/extra/desktop.win.cpp
  ${SHARDS_DIR}/src/extra/desktop.capture.win.hpp
  ${SHARDS_DIR}/src/mal/SHCore.cpp
  ${SHARDS_DIR}/src/core/shards/serialization.cpp
  ${SHARDS_DIR}/src/core/shards/time.cpp
  ${SHARDS_DIR}/src/core/shards/os.cpp
  ${SHARDS_DIR}/src/core/shards/strings.cpp
  ${SHARDS_DIR}/src/core/edn/print.hpp
  ${SHARDS_DIR}/src/core/edn/read.hpp
  ${SHARDS_DIR}/src/core/edn/eval.hpp
  ${SHARDS_DIR}/src/core/edn/eval.cpp
  ${SHARDS_DIR}/src/extra/snappy.cpp
  ${SHARDS_DIR}/src/extra/brotli.cpp
  ${SHARDS_DIR}/src/extra/xr.cpp
  ${SHARDS_DIR}/src/core/shards/bigint.cpp
  ${SHARDS_DIR}/src/core/shards/wasm.cpp
  ${SHARDS_DIR}/src/core/shards/edn.cpp
  ${SHARDS_DIR}/src/core/shards/reflection.cpp
  ${SHARDS_DIR}/src/tests/test_runtime.cpp
  ${SHARDS_DIR}/src/core/shccstrings.hpp
  ${SHARDS_DIR}/src/extra/dsp.cpp
  )

#### Header paths for tidy
set(
  MY_PROJECT_HEADER_PATHS
  ${MY_PROJECT_HEADER_PATHS}
  -I${SHARDS_DIR}/include
  -I${SHARDS_DIR}/src/core
  -I${SHARDS_DIR}/src/gfx
  -I${SHARDS_DIR}/deps/spdlog/include
  -I${SHARDS_DIR}/deps/SPSCQueue/include
  -I${SHARDS_DIR}/deps/stb
  -I${SHARDS_DIR}/deps/kcp
  -I${SHARDS_DIR}/deps/xxHash
  -I${SHARDS_DIR}/deps/json/include
  -I${SHARDS_DIR}/deps/magic_enum/include
  -I${SHARDS_DIR}/deps/nameof/include
  -I${SHARDS_DIR}/deps/cpp-taskflow
  -I${SHARDS_DIR}/deps/pdqsort
  -I${SHARDS_DIR}/deps/filesystem/include
  -I${SHARDS_DIR}/deps/miniaudio
  -I${SHARDS_DIR}/deps/wasm3/source
  -I${SHARDS_DIR}/deps/linalg
  )

### setup clang format
find_program(
  CLANG_FORMAT_EXE
  NAMES "clang-format"
  DOC "Path to clang-format executable"
  )
if(NOT CLANG_FORMAT_EXE)
  message(STATUS "clang-format not found.")
else()
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
endif()

#### Format target
foreach(_file ${MY_PROJECT_SOURCE_FILES})
  if(CLANG_FORMAT_EXE)
    add_custom_command(
      OUTPUT ${_file}.formatted
      COMMAND ${CLANG_FORMAT_EXE} -i ${_file}
      )
    list(APPEND formatfied ${_file}.formatted)
  endif()
endforeach()
add_custom_target(format DEPENDS ${formatfied})
###

### setup clang tidy
find_program(
  CLANG_TIDY_EXE
  NAMES "clang-tidy"
  DOC "Path to clang-tidy executable"
  )
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
endif()

#### tidy target
set(tidyfied)
foreach(_file ${MY_PROJECT_SOURCE_FILES})
  if(CLANG_TIDY_EXE)
    add_custom_command(
      OUTPUT ${_file}.tidyfied
      COMMAND ${CLANG_TIDY_EXE} -checks=-*,clang-analyzer-*,performance-*,bugprone-* -fix ${_file} -- -std=c++17 -DDEBUG ${MY_PROJECT_HEADER_PATHS}
      )
    list(APPEND tidyfied ${_file}.tidyfied)
  endif()
endforeach()
add_custom_target(tidy DEPENDS ${tidyfied})
###