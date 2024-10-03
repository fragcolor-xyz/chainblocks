# Always optimize files with hot-paths
set_source_files_properties(
  gltf.cpp
  PROPERTIES
  COMPILE_FLAGS -O3
  SKIP_PRECOMPILE_HEADERS ON)

add_library(gfx-gltf-obj OBJECT gltf.cpp gltf2.cpp impl.cpp)
target_link_libraries(gfx-gltf-obj gfx-core-obj tinygltf)
target_link_libraries(gfx-gltf-obj Tracy::TracyClient)

add_library(gfx-gltf INTERFACE)
target_sources(gfx-gltf PUBLIC $<TARGET_OBJECTS:gfx-gltf-obj>)
target_link_libraries(gfx-gltf INTERFACE gfx-gltf-obj)