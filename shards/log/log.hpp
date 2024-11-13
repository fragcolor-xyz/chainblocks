#ifndef E8296F1D_E25F_4AC4_AA7C_D680CA0D7ABF
#define E8296F1D_E25F_4AC4_AA7C_D680CA0D7ABF

#include <string>
#include <spdlog/spdlog.h>
#include <shared_mutex>
#include <vector>

namespace shards::logging {
typedef std::shared_ptr<spdlog::logger> Logger;
// Redirects this logger to the same output as the default logger
void initSinks(Logger logger);
// Sets the log level for this logger based on the LOG_<name> environment variable if it is set
void initLogLevel(Logger logger);
// Init flush_on setting
void initFlush(Logger logger);
// Sets the log format for this logger based on the LOG_<name>_FORMAT environment variable if it is set
void initLogFormat(Logger logger);
void redirectAll(const std::vector<spdlog::sink_ptr> &sinks);

// Controls log filter level of the output sinks
spdlog::level::level_enum getSinkLevel();
void setSinkLevel(spdlog::level::level_enum level);

// Setup the default logger if it's not setup already
void setupDefaultLoggerConditional(std::string fileName);

void flush();

// Set default log level and redirect to main logger
// !! Do not call directly since this registers the logger
// !! use getOrCreate instead to prevent race conditions
void __init(Logger logger);

std::shared_mutex& __getRegisterMutex();
template <typename T> Logger getOrCreate(const std::string &name, T init) {
  auto& m = __getRegisterMutex();
  std::shared_lock<std::shared_mutex> l(m);
  auto logger = spdlog::get(name);
  if (!logger) {
    // Swap to write lock
    l.unlock();
    std::unique_lock<std::shared_mutex> ul(m);

    // Check again after acquiring write lock in case another thread created the logger in between
    logger = spdlog::get(name);
    if (logger)
      return logger;

    logger = std::make_shared<spdlog::logger>(name);
    init(logger);
    __init(logger);
  }
  return logger;
}
inline Logger getOrCreate(const std::string &name) {
  return getOrCreate(name, [](Logger logger) {});
}
} // namespace shards::logging

#endif /* E8296F1D_E25F_4AC4_AA7C_D680CA0D7ABF */
