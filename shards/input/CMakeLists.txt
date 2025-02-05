add_library(input-obj OBJECT
  input_stack.cpp window_mapping.cpp master.cpp
  gestures.cpp clipboard.cpp
  event_decoder.cpp
)
target_include_directories(input-obj PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(input-obj linalg spdlog Boost::lockfree)
if(NOT EMSCRIPTEN)
  target_link_libraries(input-obj SDL3-static)
else()
  target_include_directories(input-obj PUBLIC ${SHARDS_DIR}/deps/SDL3/include)
endif()
target_link_libraries(input-obj gfx-core) # For window mapping
target_compile_features(input-obj PUBLIC cxx_std_20)

add_library(input INTERFACE)
target_sources(input PUBLIC $<TARGET_OBJECTS:input-obj>)
target_link_libraries(input INTERFACE input-obj)

add_executable(test-input tests/tests.cpp)
target_link_libraries(test-input gfx input Catch2::Catch2WithMain linalg spdlog)
target_compile_features(test-input PUBLIC cxx_std_20)

add_rust_library(
  NAME input-debug-ui
  PROJECT_PATH ${CMAKE_CURRENT_LIST_DIR}/debug_ui
  DEPENDS debug_ui.hpp)
