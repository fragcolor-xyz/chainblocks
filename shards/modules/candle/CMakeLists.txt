# Platform-specific features
if(APPLE)
  list(APPEND ML_RUST_FEATURES "metal") # In reality this will only work on macOS unless metal is initialized before (e.g. Formabble app)
  list(APPEND ML_RUST_FEATURES "accelerate")
endif()

# Add an option to enable or disable CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Try to find CUDA package (optional)
if(ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)

  # Check if CUDA is found
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found. Enabling CUDA support.")
    list(APPEND ML_RUST_FEATURES "cuda")
  else()
    message(STATUS "CUDA not found. Skipping CUDA support.")
  endif()
else()
  message(STATUS "CUDA support is disabled.")
endif()

# Define the Rust library with the collected features
add_rust_library(NAME shards-ml
  PROJECT_PATH ${CMAKE_CURRENT_LIST_DIR}
  FEATURES ${ML_RUST_FEATURES})

# Add the Shards module
add_shards_module(ml
  REGISTER_SHARDS rust
  RUST_TARGETS shards-ml-rust
  EXPERIMENTAL)