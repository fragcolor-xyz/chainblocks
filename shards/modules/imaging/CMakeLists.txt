set(SOURCES
  imaging.cpp
)

add_shards_module(imaging SOURCES ${SOURCES}
  REGISTER_SHARDS imaging)

target_link_libraries(shards-module-imaging stb jpeg-static)
target_include_directories(shards-module-imaging PRIVATE ${CMAKE_SOURCE_DIR}/deps/mozjpeg)
