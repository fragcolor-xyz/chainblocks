set(cli_SOURCES
  main.cpp
)

function(setup_shards_target TARGET)
  if(NOT EMSCRIPTEN)
    target_link_libraries(${TARGET} Boost::process)
  endif()

  if(APPLE)
    target_link_libraries(${TARGET} shards-core shards_core_swift)
  else()
    target_link_libraries(${TARGET} shards-core)
  endif()

  set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)

  # Need the lang-ffi bindings
  add_dependencies(${TARGET} cargo-shards-langffi-rust)
endfunction()

function(setup_shards_library_target TARGET)
  target_sources(${TARGET} PRIVATE ${cli_SOURCES})
  target_compile_definitions(${TARGET} PUBLIC NO_MAL_MAIN=1)

  setup_shards_target(${TARGET})
endfunction()

# Executable
add_executable(shards ${cli_SOURCES})
setup_shards_target(shards)
target_link_libraries(shards shards-cpp-union)

# library
add_library(shards-lib)
setup_shards_library_target(shards-lib)
set_target_properties(shards-lib PROPERTIES OUTPUT_NAME "${LIB_PREFIX}shards-static")
target_compile_definitions(shards-lib PRIVATE "shards_EXPORTS=1")

if(EMSCRIPTEN)
  if(EMSCRIPTEN_PTHREADS)
    set_target_properties(shards PROPERTIES SUFFIX "-mt.js")
  else()
    set_target_properties(shards PROPERTIES SUFFIX "-st.js")
  endif()

  target_sources(shards PRIVATE emscripten_api.cpp)

  target_link_options(shards PUBLIC
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=shards"
    "-sEXPORT_ES6=1" # Don't need compatibility with older browsers, wgpu is a limiting factor
    "-sMIN_FIREFOX_VERSION=125"
    "-sMIN_CHROME_VERSION=113"
    "-sMIN_SAFARI_VERSION=-1"
    "-sPTHREAD_POOL_SIZE=12"
    "-sEXPORTED_RUNTIME_METHODS=['FS', 'FETCHFS', 'stringToNewUTF8', 'UTF8ToString', 'ccall', 'cwrap', 'gfxSetup']"
    "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
    "-sOFFSCREENCANVAS_SUPPORT=1"
    "-sSTACK_SIZE=2097152"
    "-sINITIAL_MEMORY=2147483648"
    "-sWASMFS"
    "-lfetchfs.js"
    "-sFORCE_FILESYSTEM"
    "-sASYNCIFY_EXPORTS=['_shardsLoadScript', '_shardsTick', '_shardsFreeInstance']"
  )

  if(TRUE)
    target_link_options(shards PUBLIC

      # "-sJSPI_IMPORTS=['_wasmfs_create_fetch_backend']"
      # "-sJSPI_EXPORTS=['wasmfs_create_fetch_backend, _wasmfs_create_fetch_backend']"
    )
  endif()

  if(SHARDS_DEBUG_SYMBOL_URL)
    # Set cache to something like:
    # SHARDS_DEBUG_SYMBOL_URL:STRING=http://localhost/fbl.debug.wasm
    string(REGEX MATCH "([^/]+)$" DEBUG_SYMBOL_FILENAME ${SHARDS_DEBUG_SYMBOL_URL})
    target_link_options(shards PRIVATE "-gseparate-dwarf=${DEBUG_SYMBOL_FILENAME}")
    target_link_options(shards PRIVATE "-sSEPARATE_DWARF_URL=${SHARDS_DEBUG_SYMBOL_URL}")
  endif()
endif()

# Apple Framework target
if(APPLE)
  add_library(shards-framework SHARED ${cli_SOURCES})
  setup_shards_target(shards-framework)

  # Framework specific settings
  set_target_properties(shards-framework PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    MACOSX_FRAMEWORK_IDENTIFIER com.fragcolor.shards
    MACOSX_FRAMEWORK_BUNDLE_VERSION "1.0"
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING "1.0"
    XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
    XCODE_ATTRIBUTE_ALWAYS_EMBED_SWIFT_STANDARD_LIBRARIES "YES"
    XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
    XCODE_ATTRIBUTE_SWIFT_INSTALL_OBJC_HEADER "NO"
    OUTPUT_NAME "Shards"

    # Debug symbols generation
    XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym"
    XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS "YES"
    XCODE_ATTRIBUTE_DEPLOYMENT_POSTPROCESSING "YES"
    XCODE_ATTRIBUTE_SEPARATE_STRIP "YES"
    XCODE_ATTRIBUTE_STRIP_INSTALLED_PRODUCT "YES"
    XCODE_ATTRIBUTE_STRIP_STYLE "debugging"
  )

  # MacOS specific settings
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(shards-framework PROPERTIES
      XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/../Frameworks @loader_path/Frameworks"
      INSTALL_RPATH "@executable_path/../Frameworks"
      BUILD_WITH_INSTALL_RPATH TRUE
    )

    # Link Swift runtime for MacOS
    target_link_libraries(shards-framework
      "-framework Foundation"
      "-framework CoreFoundation"
      "-lswiftCore"
      "-lswiftFoundation"
    )
  endif()

  # Make headers public for framework
  file(GLOB PUBLIC_HEADERS "*.hpp")
  set_target_properties(shards-framework PROPERTIES
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
  )

  target_link_libraries(shards-framework shards-cpp-union)
endif()

if(EMSCRIPTEN)
  if(EMSCRIPTEN_PTHREADS)
    set_target_properties(shards PROPERTIES SUFFIX "-mt.js")
  else()
    set_target_properties(shards PROPERTIES SUFFIX "-st.js")
  endif()

  target_link_options(shards PUBLIC
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=shards"
    "SHELL:-s NO_EXIT_RUNTIME=1"
  )
endif()
