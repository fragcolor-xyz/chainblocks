add_rust_library(NAME shards-onnx
  PROJECT_PATH ${CMAKE_CURRENT_LIST_DIR})
  
add_shards_module(onnx
  REGISTER_SHARDS rust
  RUST_TARGETS shards-onnx-rust
  EXPERIMENTAL)