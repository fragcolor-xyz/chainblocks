# if(NOT EMSCRIPTEN)
#   set(SOURCES
#     embedd.cpp
#   )

#   add_shards_module(llm SOURCES ${SOURCES}
#     REGISTER_SHARDS llm)

#   target_link_libraries(shards-module-llm llama)
# endif()
