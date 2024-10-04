#ifndef C37F5D68_A9D1_4A28_9018_523BF07FBDA3
#define C37F5D68_A9D1_4A28_9018_523BF07FBDA3

#include <shards/log/log.hpp>
namespace gfx::data {
inline shards::logging::Logger getLogger() {
  return shards::logging::getOrCreate("data_cache", [](shards::logging::Logger logger) {
    // Set default log level to warn to be less verbose
    logger->set_level(spdlog::level::info);
  });
}
} // namespace gfx::data

#endif /* C37F5D68_A9D1_4A28_9018_523BF07FBDA3 */
