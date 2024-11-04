set(SOURCES
  riscv.cpp
)

add_shards_module(shriscv SOURCES ${SOURCES}
  REGISTER_SHARDS shriscv
  EXPERIMENTAL)

target_include_directories(shards-module-shriscv PRIVATE ../core)
target_link_libraries(shards-module-shriscv riscv)
