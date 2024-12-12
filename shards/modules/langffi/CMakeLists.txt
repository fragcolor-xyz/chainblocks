# Manually add lang source files as dependencies
set(LANG_PATH ../lang)
file(GLOB_RECURSE LANG_DEPS ${LANG_PATH}/*.rs ${LANG_PATH}/Cargo.toml)
file(GLOB_RECURSE LANG_DEPS_REMOVE ${LANG_PATH}/target/*.rs)
message(DEBUG ${LANG_DEPS_REMOVE})
list(REMOVE_ITEM LANG_DEPS ${LANG_DEPS_REMOVE})

if(TRACY_ENABLE)
  list(APPEND FEATURES tracy)
  # list(APPEND RUST_ENV
  #   TRACY_CLIENT_LIB=TracyClient # The lib file is called libTracyClient.a but this needs to be and -l compatible arg
  #   TRACY_CLIENT_LIB_PATH=$<TARGET_FILE_DIR:Tracy::TracyClient>
  #   TRACY_CLIENT_STATIC=1
  # )
endif()

add_rust_library(
  NAME shards-langffi
  PROJECT_PATH ${CMAKE_CURRENT_LIST_DIR}
  FEATURES ${FEATURES}
  DEPENDS ${SHARDS_DIR}/include/shards/shards.h ${LANG_DEPS}
)

add_shards_module(langffi
  RUST_TARGETS shards-langffi-rust
  REGISTER_SHARDS langffi)