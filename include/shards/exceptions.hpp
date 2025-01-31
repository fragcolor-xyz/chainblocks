#ifndef BD8027B6_C789_4569_9C25_3500A2D0E4BA
#define BD8027B6_C789_4569_9C25_3500A2D0E4BA

#include <cstdlib>
#include <spdlog/spdlog.h>

#define SHARDS_THROW(__e) \
    SPDLOG_CRITICAL("Assertion failed: {}", (__e).what()); \
    std::abort();                                     


#endif /* BD8027B6_C789_4569_9C25_3500A2D0E4BA */
