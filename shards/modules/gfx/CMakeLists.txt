set(SOURCES
  gfx.cpp
  window.cpp
  renderer.cpp
  mesh.cpp
  drawable.cpp
  material.cpp
  feature.cpp
  shader/translator.cpp
  shader/wgsl.cpp
  shader/composition.cpp
  shader/linalg_shards.cpp
  shader/math_shards.cpp
  shader/core_shards.cpp
  shader/wire_shards.cpp
  shader/flow_shards.cpp
  gltf.cpp
  gltf_data_shards.cpp
  camera.cpp
  texture.cpp
  view.cpp
  steps.cpp
  buffer.cpp
  gizmos/context.cpp
  gizmos/gizmos.cpp
  gizmos/highlight.cpp
  gizmos/shapes.cpp
)

if(HAS_WGPU_NATIVE)
  message(STATUS "Building gfx crate with wgpu-native enabled")
  set(RUST_TARGETS gfx-rust)
else()
  unset(RUST_TARGETS)
endif()

add_shards_module(gfx SOURCES ${SOURCES}
  REGISTER_SHARDS gfx gizmos buffer gltf_data
  RUST_TARGETS ${RUST_TARGETS}
)

target_link_libraries(shards-module-gfx stb)

# Link object libraries since they depend on rust code
# and will be combined into the union lib
target_link_libraries(shards-module-gfx gfx-core gfx-gltf gfx-data-format)

target_precompile_headers(shards-module-gfx PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp>")
