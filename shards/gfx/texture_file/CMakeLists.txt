add_library(gfx-texture-file-obj OBJECT texture_file.cpp)
target_link_libraries(gfx-texture-file-obj gfx-core-obj)

add_library(gfx-texture-file INTERFACE)
target_sources(gfx-texture-file PUBLIC $<TARGET_OBJECTS:gfx-texture-file-obj>)
target_link_libraries(gfx-texture-file INTERFACE gfx-texture-file-obj)