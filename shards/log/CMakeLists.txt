add_library(shards-logging log.cpp)
target_include_directories(shards-logging PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../..)
target_link_libraries(shards-logging spdlog magic_enum Boost::filesystem Boost::algorithm)

if(NOT EMSCRIPTEN)
  target_link_libraries(shards-logging SDL3-static)
  target_compile_definitions(shards-logging PRIVATE SHARDS_LOG_SDL=1)
endif()

# The compile-time filtered log level
if(NOT SPDLOG_ACTIVE_LEVEL)
  if((CMAKE_BUILD_TYPE STREQUAL "Release") OR(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel"))
    set(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO)
  else()
    set(SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE)
  endif()
endif()

# The default log level for
if(NOT SHARDS_DEFAULT_LOG_LEVEL)
  set(SHARDS_DEFAULT_LOG_LEVEL SPDLOG_LEVEL_INFO)
endif()

target_compile_definitions(shards-logging
  PUBLIC SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL})

# Default log level for all loggers
if(SHARDS_DEFAULT_LOG_LEVEL)
  target_compile_definitions(shards-logging PRIVATE SHARDS_DEFAULT_LOG_LEVEL=${SHARDS_DEFAULT_LOG_LEVEL})
endif()

# Default filter for stdout logger
if(SHARDS_DEFAULT_STDOUT_LOG_LEVEL)
  target_compile_definitions(shards-logging PRIVATE SHARDS_DEFAULT_STDOUT_LOG_LEVEL=${SHARDS_DEFAULT_STDOUT_LOG_LEVEL})
endif()

# Default filter for file logger
if(SHARDS_DEFAULT_FILE_LOG_LEVEL)
  target_compile_definitions(shards-logging PRIVATE SHARDS_DEFAULT_FILE_LOG_LEVEL=${SHARDS_DEFAULT_FILE_LOG_LEVEL})
endif()

# Add options for rotating log variation instead of basic file sink
if(SHARDS_LOG_ROTATING_MAX_FILE_SIZE)
  target_compile_definitions(shards-logging PRIVATE SHARDS_LOG_ROTATING_MAX_FILE_SIZE=${SHARDS_LOG_ROTATING_MAX_FILE_SIZE})
endif()

if(SHARDS_LOG_ROTATING_MAX_FILES)
  target_compile_definitions(shards-logging PRIVATE SHARDS_LOG_ROTATING_MAX_FILES=${SHARDS_LOG_ROTATING_MAX_FILES})
endif()
