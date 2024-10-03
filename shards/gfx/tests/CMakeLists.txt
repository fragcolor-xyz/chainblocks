include(Assets)

if(EMSCRIPTEN)
	set(CMAKE_EXECUTABLE_SUFFIX ".html")
endif()

add_executable(test-gfx
  main.cpp
  test_general.cpp
  test_shader.cpp
  test_gltf.cpp
  test_gizmos.cpp
  test_comparison.cpp
  test_rendergraph.cpp
  test_rendergraph_headless.cpp
  test_multithreaded.cpp
  test_caching.cpp
  test_naga.cpp
  test_fast_string.cpp
  test_leaks.cpp
  test_serialization.cpp
)

if(EMSCRIPTEN)
  target_link_options(test-gfx PRIVATE --emrun)
endif()

set(GFX_TEST_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_library(gfx-test-utils
  platform_id.cpp
  context.cpp
  renderer.cpp
  data.cpp)
target_link_libraries(gfx-test-utils gfx-rust gfx-core gfx-texture-file Catch2::Catch2)
add_asset_path_definition(gfx-test-utils PRIVATE GFX_TEST_DATA_PATH ${GFX_TEST_DATA_PATH})

target_link_libraries(test-gfx gfx gfx-test-utils gfx-texture-file gfx-gltf Catch2::Catch2WithMain)
if(WIN32)
  target_link_libraries(test-gfx ntdll)
endif()
target_precompile_headers(test-gfx PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp>")
add_asset_path_definition(test-gfx PUBLIC GFX_TEST_DATA_PATH ${GFX_TEST_DATA_PATH})
