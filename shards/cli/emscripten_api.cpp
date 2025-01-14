#include <shards/core/runtime.hpp>
#include <shards/core/foundation.hpp>
#include <shards/modules/langffi/bindings.h>
#include <boost/core/span.hpp>
#include <emscripten/wasmfs.h>
#include <emscripten/console.h>
#include <shards/core/em_proxy.hpp>
#include <spdlog/sinks/base_sink.h>

#include <shards/log/log.hpp>

struct Instance {
  std::shared_ptr<SHMesh> mesh{};
  std::shared_ptr<SHWire> wire{};
  std::optional<std::string> error;
};

struct Message {
  std::string msg;
  spdlog::level::level_enum level;
  std::string file;
};

struct MessageRep {
  const char *msg;
  const char *file;
  int line;
  int level;
};

struct LogBuffer {
  std::mutex mutex;
  std::vector<Message> messages;
  std::vector<MessageRep> cachedReps;

  void addMessage(const std::string &msg, spdlog::level::level_enum level, const std::string &file) {
    std::lock_guard<std::mutex> lock(mutex);
    messages.push_back({msg, level, file});
  }

  void updateMessageReps() {
    cachedReps.clear();

    // Convert messages to reps
    cachedReps.reserve(messages.size());
    for (const auto &msg : messages) {
      cachedReps.push_back({msg.msg.c_str(), msg.file.c_str(), 0, static_cast<int>(msg.level)});
    }
  }

  MessageRep *getMessages(int *count) {
    updateMessageReps();
    *count = cachedReps.size();
    return cachedReps.empty() ? nullptr : cachedReps.data();
  }

  void clear() {
    cachedReps.clear();
    messages.clear();
  }
};

static LogBuffer logBuffer;

#include <emscripten.h>

using shards::operator""_swl;
using shards::toSWL;

struct AsyncRunner {
  bool shouldRun = true;
  std::list<std::function<void()>> tasks;
  std::mutex mutex;
  std::condition_variable cv;
  std::optional<std::thread> thread;

  void run() {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
      cv.wait(lock);
      if (!shouldRun)
        break;
      if (tasks.empty())
        continue;
      auto task = std::move(tasks.front());
      tasks.pop_front();
      lock.unlock();
      task();
      lock.lock();
    }
  }

  void post(std::function<void()> task) {
    std::unique_lock<std::mutex> lock(mutex);
    tasks.push_back(std::move(task));
    cv.notify_one();
  }

  void start() {
    thread = std::thread([this]() { run(); });
  }

  void stop() {
    shouldRun = false;
    cv.notify_all();
    if (thread) {
      thread->join();
      thread = std::nullopt;
    }
  }
};

static shards::EmMainProxy emMainProxy;
static AsyncRunner asyncRunner;

class LogBufferSink : public spdlog::sinks::base_sink<std::mutex> {
protected:
  void sink_it_(const spdlog::details::log_msg &msg) override {
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);

    logBuffer.addMessage(fmt::to_string(formatted), msg.level, msg.source.filename ? msg.source.filename : "");
  }

  void flush_() override {}
};

extern "C" {
EMSCRIPTEN_KEEPALIVE void shardsInit() {
  // Add log buffer sink
  shards::logging::setupDefaultLoggerConditional("shards.log");
  auto sink = std::make_shared<LogBufferSink>();
  auto logger = spdlog::default_logger();
  logger->sinks().push_back(sink);

  auto core = shardsInterface(SHARDS_CURRENT_ABI);
  shards_init(core); // to init rust things
  asyncRunner.start();
  shards::EmMainProxy::instance = &emMainProxy;
}

// Needs to be polled on the main thread for audio and other stuff that needs to be run on the main browser thread
EMSCRIPTEN_KEEPALIVE void shardsPollMainProxy() { emMainProxy.poll(); }

EMSCRIPTEN_KEEPALIVE void shardsFSMountHTTP(const char *target_, const char *baseUrl_) {
  std::string target{target_};
  std::string baseUrl{baseUrl_};
  asyncRunner.post([target, baseUrl]() {
    auto backend = wasmfs_create_fetch_backend(baseUrl.c_str());
    wasmfs_create_directory(target.c_str(), 0777, backend);
  });
}

EMSCRIPTEN_KEEPALIVE Instance *shardsLoadScript(const char *code, const char *base_path) {
  shards::logging::setupDefaultLoggerConditional("");
  Instance *instance = new Instance();
  SHLEvalEnv *env{};
  std::vector<SHStringWithLen> includeDirs;

  DEFER({
    if (env)
      shards_free_env(env);
  });

  try {
    instance->mesh = SHMesh::make();

    SHLEvalEnv *env = shards_create_env("shards"_swl);

    auto astRes = shards_read("script"_swl, toSWL(code), toSWL(base_path), includeDirs.data(), includeDirs.size());
    if (astRes.error) {
      SPDLOG_ERROR("Failed to read code: {}", astRes.error->message);
    }
    shards::OwnedVar ast(astRes.ast);
    auto err = shards_eval_env(env, &ast);
    if (err) {
      SPDLOG_ERROR("Failed to eval script at {}:{}: {}", err->line, err->column, err->message);
      instance->error = err->message;
      shards_free_error(err);
      return instance;
    }

    SHLWire shlwire = shards_eval(&ast, "script"_swl);
    DEFER(shards_free_wire(shlwire));
    if (shlwire.error) {
      SPDLOG_ERROR("Failed to evaluate script at {}:{}: {}", shlwire.error->line, shlwire.error->column, shlwire.error->message);
      instance->error = shlwire.error->message;
      return instance;
    }

    auto wire = SHWire::sharedFromRef(*shlwire.wire);
    instance->wire = wire;
    instance->mesh->schedule(wire);
    instance->error.reset();

    return instance;
  } catch (std::exception &ex) {
    SPDLOG_ERROR("Unhandled exception in shardsLoadScript: {}", ex.what());
    return nullptr;
  }
}

EMSCRIPTEN_KEEPALIVE bool shardsTick(Instance *instance) { return instance->mesh->tick(); }

EMSCRIPTEN_KEEPALIVE const char *shardsGetError(Instance *instance) {
  if (instance->error)
    return instance->error->c_str();
  auto &meshErrors = instance->mesh->errors();
  if (meshErrors.size() > 0) {
    return meshErrors[0].c_str();
  }
  return nullptr;
}

EMSCRIPTEN_KEEPALIVE void shardsSetLoggerLevel(const char *loggerName, int level) {
  auto logger = spdlog::get(loggerName);
  if (logger) {
    logger->set_level(spdlog::level::level_enum(level));
  }
}

EMSCRIPTEN_KEEPALIVE void shardsFreeInstance(Instance *instance) { delete instance; }

EMSCRIPTEN_KEEPALIVE void shardsLockLogBuffer() { logBuffer.mutex.lock(); }

EMSCRIPTEN_KEEPALIVE void shardsUnlockLogBuffer() {
  logBuffer.clear(); // Clear messages when unlocking
  logBuffer.mutex.unlock();
}

EMSCRIPTEN_KEEPALIVE MessageRep *shardsGetLogMessages(int *count) { return logBuffer.getMessages(count); }
}