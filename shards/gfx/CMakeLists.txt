option(SH_DEBUG "Enable context debug mode by default" OFF)

set(GFX_SOURCES
  culling.cpp
  context.cpp
  loop.cpp
  view.cpp
  gfx_wgpu.cpp
  renderer.cpp
  mesh.cpp
  mesh_utils.cpp
  geom.cpp
  drawable.cpp
  enums.cpp
  params.cpp
  paths.cpp
  utils.cpp
  feature.cpp
  texture.cpp
  material.cpp
  buffer.cpp
  drawables/mesh_drawable.cpp
  drawables/mesh_tree_drawable.cpp
  gizmos/wireframe.cpp
  gizmos/shapes.cpp
  gizmos/gizmo_input.cpp
  gizmos/gizmos.cpp
  gizmos/text.cpp
  shader/generator.cpp
  shader/generator_context.cpp
  shader/blocks.cpp
  shader/types.cpp
  features/velocity.cpp
  features/transform.cpp
  rust_interop.cpp
  view_stack.cpp
  render_target.cpp
  render_graph_evaluator.cpp
  render_step_impl.cpp
  renderer_types.cpp
  pipeline_builder.cpp
  pipeline_step.cpp
  pipeline_hashes.cpp
  unique_id.cpp
  data_cache/data_cache.cpp
)

if(EMSCRIPTEN)
  list(APPEND GFX_SOURCES window_em.cpp)
  list(APPEND GFX_SOURCES gfx_events_em.cpp)
else()
  list(APPEND GFX_SOURCES window.cpp)
endif()

add_library(gfx-core-obj OBJECT ${GFX_SOURCES})

target_bundle_files(gfx-core-obj
  PREFIX bundled
  NAMESPACE bundled
  FILES
  "gizmos/BigBlue_TerminalPlus.bin"
)

# Always optimize files with hot-paths
set_source_files_properties(
  # drawables/mesh_tree_drawable.cpp
  texture_file/texture_file.cpp
  pipeline_hashes.cpp
  PROPERTIES
  COMPILE_FLAGS -O3
  SKIP_PRECOMPILE_HEADERS ON)

target_compile_features(gfx-core-obj PUBLIC cxx_std_20)

set(GFX_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/gfx-generated)
file(MAKE_DIRECTORY ${GFX_GENERATED_DIR})

target_include_directories(gfx-core-obj
  PUBLIC
  ${SHARDS_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${GFX_GENERATED_DIR}
)
target_link_libraries(gfx-core-obj
  tinygltf draco-wrapper
  magic_enum nameof linalg
  spdlog stb xxHash Boost::algorithm Boost::filesystem Boost::container Boost::uuid Boost::tti tbb
  brotlidec-static brotlicommon-static
  shards-logging
  shards-shared-temp-allocator
  shards-fast-string
)
target_precompile_headers(gfx-core-obj PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp>")
target_link_libraries(gfx-core-obj Tracy::TracyClient)

if(HAVE_THREADS)
  set(GFX_THREADING_DEFAULT ON)
endif()

option(GFX_THREADING "Support multithreaded rendering" ${GFX_THREADING_DEFAULT})

if(GFX_THREADING)
  target_compile_definitions(gfx-core-obj PUBLIC GFX_THREADING_SUPPORT=1)
endif()

target_link_libraries(gfx-core-obj Taskflow)

add_subdirectory(gltf)
add_subdirectory(data_format)
add_subdirectory(texture_file)

option(GFX_USE_DATA_PATH "Uses build tree paths to find data (dev only)" ON)

if(NOT EMSCRIPTEN AND(CMAKE_BUILD_TYPE MATCHES Debug))
  set(GFX_CAN_USE_DATA_PATH ON)
endif()

if(GFX_CAN_USE_DATA_PATH AND GFX_USE_DATA_PATH)
  target_compile_definitions(gfx-core-obj PUBLIC GFX_DATA_PATH=\"${SHARDS_DIR}\")
endif()

if(EMSCRIPTEN)
  target_link_options(gfx-core-obj PUBLIC "-sUSE_WEBGPU=1")

  em_link_js_library(gfx-core-obj gfx_wgpu.js)
  em_link_js_library(gfx-core-obj gfx_events.js)

  target_include_directories(gfx-core-obj PUBLIC ${SHARDS_DIR}/deps/SDL3/include)
else()
  target_link_libraries(gfx-core-obj SDL3-static)

  set(HAS_WGPU_NATIVE ON PARENT_SCOPE)
  set(HAS_WGPU_NATIVE ON)
  target_compile_definitions(gfx-core-obj PUBLIC SHARDS_GFX_SDL=1)
endif()

if(ANDROID)
  target_link_libraries(gfx-core-obj android OpenSLES)
endif()

add_library(gfx-core INTERFACE)
target_sources(gfx-core PUBLIC $<TARGET_OBJECTS:gfx-core-obj>)
target_link_libraries(gfx-core INTERFACE gfx-core-obj)

# Define the gfx library target that combines core and the rust side
add_library(gfx INTERFACE)

if(HAS_WGPU_NATIVE)
  # Set GFX_WGPU_TRACE_PATH to the trace path to enable
  if(GFX_WGPU_TRACE_PATH)
    message(STATUS "Tracing WebGPU to: ${GFX_WGPU_TRACE_PATH}")
    string(REPLACE "\\" "\\\\" GFX_WGPU_TRACE_PATH_1 "${GFX_WGPU_TRACE_PATH}")
    file(WRITE ${GFX_GENERATED_DIR}/wgpu_trace_config.h "const char* wgpuTracePath = \"${GFX_WGPU_TRACE_PATH_1}\";")
    target_compile_definitions(gfx-core-obj PUBLIC WEBGPU_TRACE=1)
  endif()

  add_subdirectory(rust)

  # Loose dependency for bindings header
  add_dependencies(gfx-core-obj cargo-gfx-rust)

  # Get include paths from gfx-rust and append them to the core library
  get_target_property(GFX_RUST_INCLUDES gfx-rust INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "gfx-rust includes: ${GFX_RUST_INCLUDES}")
  target_include_directories(gfx-core-obj PUBLIC ${GFX_RUST_INCLUDES})
  target_compile_definitions(gfx-core-obj PUBLIC WEBGPU_NATIVE=1)
else()
  target_link_libraries(gfx INTERFACE gfx-core)
endif()

add_subdirectory(tests)
