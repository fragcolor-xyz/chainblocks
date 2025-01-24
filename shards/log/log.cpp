#include "log.hpp"
#include "spdlog/fmt/bundled/core.h"
#include <iterator>
#include <spdlog/spdlog.h>
#include <vector>
#if SHARDS_LOG_SDL
#include <SDL3/SDL_stdinc.h>
#endif
#include <magic_enum.hpp>
#include <shared_mutex>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "../core/platform.hpp"

#if SH_ANDROID
#include <spdlog/sinks/android_sink.h>
#elif SH_EMSCRIPTEN
#include <emscripten.h>

struct EmscriptenSink : public spdlog::sinks::base_sink<std::mutex> {
  void sink_it_(const spdlog::details::log_msg &msg) override {
    int lv = EM_LOG_CONSOLE;
    switch (msg.level) {
    case spdlog::level::trace:
    case spdlog::level::debug:
      lv |= EM_LOG_DEBUG;
      break;
    case spdlog::level::critical:
    case spdlog::level::err:
      lv |= EM_LOG_ERROR;
      break;
    case spdlog::level::warn:
      lv |= EM_LOG_WARN;
      break;
    default:
    case spdlog::level::info:
      lv |= EM_LOG_INFO;
      break;
    }

    tmpBuffer.assign(msg.payload.data(), msg.payload.size());
    emscripten_log(lv, "%s", tmpBuffer.data());
  }
  void flush_() override {}
  std::string tmpBuffer;
};

#endif

namespace shards::logging {

std::shared_mutex &__getRegisterMutex() {
  static std::shared_mutex m;
  return m;
}

std::optional<spdlog::level::level_enum> getLogLevelFromEnvVar(std::string inName) {
  std::string varName;
  const char *val{};

#if SHARDS_LOG_SDL
  auto tryReadEnvVar = [&]() {
    varName = inName;
    val = SDL_getenv(varName.c_str());
  };

  tryReadEnvVar();

  if (!val) {
    boost::algorithm::to_lower(inName);
    tryReadEnvVar();
  }

  if (!val) {
    varName.clear();
    boost::algorithm::to_upper(inName);
    tryReadEnvVar();
  }
#endif

  if (val) {
    return magic_enum::enum_cast<spdlog::level::level_enum>(val);
  }
  return std::nullopt;
}

static std::optional<spdlog::level::level_enum> getFlushLogLevel() {
#if SHARDS_LOG_SDL
  auto val = SDL_getenv("LOG_FLUSH_ON");
  if (val) {
    return magic_enum::enum_cast<spdlog::level::level_enum>(val);
  }
#endif
  return std::nullopt;
}

struct Config {
  // -- Compile time settings --
  static constexpr std::optional<spdlog::level::level_enum> DefaultStdOutLogLevel =
#ifdef SHARDS_DEFAULT_STDOUT_LOG_LEVEL
      spdlog::level::level_enum(SHARDS_DEFAULT_STDOUT_LOG_LEVEL);
#else
      std::nullopt;
#endif

  static constexpr std::optional<spdlog::level::level_enum> DefaultFileLogLevel =
#ifdef SHARDS_DEFAULT_FILE_LOG_LEVEL
      spdlog::level::level_enum(SHARDS_DEFAULT_FILE_LOG_LEVEL);
#else
      std::nullopt;
#endif

  static constexpr std::optional<spdlog::level::level_enum> DefaultLoggerLevel =
#ifdef SHARDS_DEFAULT_LOG_LEVEL
      spdlog::level::level_enum(SHARDS_DEFAULT_LOG_LEVEL);
#else
      std::nullopt;
#endif

  static constexpr std::optional<spdlog::level::level_enum> getDefaultLogLevel() {
    if (DefaultLoggerLevel)
      return *DefaultLoggerLevel;

    if (!DefaultStdOutLogLevel && !DefaultFileLogLevel)
      return std::nullopt;

    spdlog::level::level_enum level = spdlog::level::info;
    if (DefaultStdOutLogLevel)
      level = std::min(level, *DefaultStdOutLogLevel);
    if (DefaultFileLogLevel)
      level = std::min(level, *DefaultFileLogLevel);
    return level;
  }
};

struct Sinks {
  std::shared_mutex lock;

  std::shared_ptr<spdlog::sinks::dist_sink_mt> distSink;
  std::shared_ptr<spdlog::sinks::stderr_color_sink_mt> stdErrSink;
  std::shared_ptr<spdlog::sinks::sink> logFileSink;
#if SH_ANDROID
  std::shared_ptr<spdlog::sinks::android_sink_mt> androidSink;
#elif SH_EMSCRIPTEN
  std::shared_ptr<EmscriptenSink> emscriptenSink;
#endif

  bool logLevelOverriden{};

  Sinks() {
    distSink = std::make_shared<spdlog::sinks::dist_sink_mt>();

#if SH_EMSCRIPTEN
    emscriptenSink = std::make_shared<EmscriptenSink>();
    distSink->add_sink(emscriptenSink);
#endif

    initStdErrSink();

    // Setup android logcat output
#if SH_ANDROID
    androidSink = std::make_shared<spdlog::sinks::android_sink_mt>("shards");
    distSink->add_sink(androidSink);
#endif

    if (Config::DefaultStdOutLogLevel) {
      stdErrSink->set_level(Config::DefaultStdOutLogLevel.value());
    }
  }

  void resetStdErrSink() {
    if (auto filter = getLogLevelFromEnvVar("LOG_STDERR_FILTER")) {
      stdErrSink->set_level(*filter);
      logLevelOverriden = true;
    }
  }

  std::unique_lock<std::shared_mutex> lockUnique() { return std::unique_lock<std::shared_mutex>(lock); }
  std::shared_lock<std::shared_mutex> lockShared() { return std::shared_lock<std::shared_mutex>(lock); }

  void initStdErrSink() {
#if !SH_EMSCRIPTEN
    if (stdErrSink)
      distSink->remove_sink(stdErrSink);
    stdErrSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    resetStdErrSink();
    distSink->add_sink(stdErrSink);
#endif
  }

  void initLogFile(std::string fileName) {
    std::string logFilePath = boost::filesystem::absolute(fileName).string();
    if (logFileSink)
      distSink->remove_sink(logFileSink);

#if defined(SHARDS_LOG_ROTATING_MAX_FILE_SIZE) && defined(SHARDS_LOG_ROTATING_MAX_FILES)
    logFileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        logFilePath.c_str(), 
        SHARDS_LOG_ROTATING_MAX_FILE_SIZE,
        SHARDS_LOG_ROTATING_MAX_FILES,
        true);
#else
    logFileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath.c_str(), true);
#endif

    if (Config::DefaultFileLogLevel) {
      logFileSink->set_level(Config::DefaultFileLogLevel.value());
    }
    distSink->add_sink(logFileSink);
  }

  // Reset compile-time settings when environment override is detected
  void overrideLogLevel() {
    if (!logLevelOverriden) {
      for (auto &s : distSink->sinks())
        s->set_level(spdlog::level::trace);
      spdlog::set_level(spdlog::level::info);
      logLevelOverriden = true;
    }
  }
};

Sinks &globalSinks() {
  static Sinks sinks;
  return sinks;
}

void flush() {
  globalSinks().distSink->flush();
}

void __init(Logger logger) {
  spdlog::register_logger(logger);
  initFlush(logger);
  initLogLevel(logger);
  initLogFormat(logger);
  initSinks(logger);
}

void initLogLevel(Logger logger) {
  auto level = getLogLevelFromEnvVar(fmt::format("LOG_{}", logger->name()));
  if (level) {
    globalSinks().overrideLogLevel();
    logger->set_level(level.value());
    return;
  }

  // Use "LOG" var to set global log level
  auto globalLevel = getLogLevelFromEnvVar(fmt::format("LOG"));
  if (globalLevel) {
    globalSinks().overrideLogLevel();
    logger->set_level(globalLevel.value());
    return;
  }

  if (auto ll = Config::getDefaultLogLevel()) {
    logger->set_level(ll.value());
  }
}

void initFlush(Logger logger) {
  if (auto flush = getFlushLogLevel()) {
    logger->flush_on(*flush);
  } else {
    logger->flush_on(spdlog::level::err);
  }
}

void initLogFormat(Logger logger) {
  std::string varName = fmt::format("LOG_{}_FORMAT", logger->name());
#if SHARDS_LOG_SDL
  if (const char *val = SDL_getenv(varName.c_str())) {
    logger->set_pattern(val);
  } else
#endif
  {
    std::string logPattern;
// Use global log format
#if SHARDS_LOG_SDL
    if (const char *val = SDL_getenv("LOG_FORMAT")) {
      logPattern = val;
    } else
#endif
    {
#ifdef __ANDROID
      // Logcat already countains timestamps & log level
      logPattern = "[T-%t][%n][%s::%#] %v";
#else
      logPattern = "[%d/%m %T.%e][T-%t][%n]%^[%l]%$[%s::%#] %v";
#endif
    }

    logger->set_pattern(logPattern);
  }
}

void initSinks(Logger logger) {
  logger->sinks().clear();
  logger->sinks().push_back(globalSinks().distSink);
}

void initAllSinks() {
  spdlog::apply_all([&](Logger logger) { initSinks(logger); });
}

spdlog::level::level_enum getSinkLevel() { return globalSinks().distSink->level(); }

void setSinkLevel(spdlog::level::level_enum level) { globalSinks().distSink->set_level(level); }

static void setupDefaultLogger(const std::string &fileName) {
  auto &sinks = globalSinks();

  {
    auto l = sinks.lockUnique();
    if (!fileName.empty()) {
      sinks.initLogFile(fileName);
    }
    // Reset this sink in case stderr handle changed
    sinks.initStdErrSink();
  }

  auto logger = std::make_shared<spdlog::logger>("shards", sinks.distSink);
  initFlush(logger);
  spdlog::set_default_logger(logger);
  initLogLevel(logger);
  initLogFormat(logger);

  // Sink acts like a filter for all loggers
  setSinkLevel(spdlog::level::trace);

  // Redirect all existing loggers to the global dist sink
  initAllSinks();
}

void setupDefaultLoggerConditional(std::string fileName) {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    setupDefaultLogger(fileName);
  }
}
} // namespace shards::logging
