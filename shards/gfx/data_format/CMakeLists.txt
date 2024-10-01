add_library(gfx-data-format-obj OBJECT bfmt.cpp)
target_link_libraries(gfx-data-format-obj Boost::uuid gfx-core-obj tinygltf Tracy::TracyClient)

add_library(gfx-data-format INTERFACE)
target_sources(gfx-data-format PUBLIC $<TARGET_OBJECTS:gfx-data-format-obj>)
target_link_libraries(gfx-data-format INTERFACE gfx-data-format-obj)