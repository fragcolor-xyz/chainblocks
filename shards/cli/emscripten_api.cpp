#include <shards/core/runtime.hpp>
#include <shards/core/foundation.hpp>
#include <shards/modules/langffi/bindings.h>
#include <boost/core/span.hpp>
#include <emscripten/wasmfs.h>
#include <emscripten/threading.h>
#include <emscripten/console.h>
#include <shards/core/em_proxy.hpp>
#include <spdlog/sinks/base_sink.h>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

#include <shards/log/log.hpp>

struct Instance {
  std::shared_ptr<SHMesh> mesh{};
  std::shared_ptr<SHWire> wire{};
  std::optional<std::string> error;
  int guard = 9999;
  ~Instance() { mesh->terminate(); }
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

// This code only works with JSPI is enabled.
// typedef bool (*AsyncRunnerFrameCallback)(void *);
// EM_JS(void, requestAsyncRunnerJSPI, (void *runner, AsyncRunnerFrameCallback callback), {
//   var wrappedCallback = WebAssembly.promising(getWasmTableEntry(callback));
//   async function tick() {
//     // Start the frame callback. 'await' means we won't call
//     // requestAnimationFrame again until it completes.
//     var keepLooping = await wrappedCallback(runner);
//     if (keepLooping)
//       requestAnimationFrame(tick);
//   }
//   requestAnimationFrame(tick);
// })

struct AsyncRunner {
  bool shouldRun = true;
  std::list<std::function<void()>> tasks;
  std::mutex mutex;
  std::condition_variable cv;
  std::optional<std::thread> thread;

  // static bool step1(void *runner_) {
  //   auto &runner = *(AsyncRunner *)runner_;
  //   runner.step();
  //   return runner.shouldRun;
  // }

  void run() {
    while (shouldRun) {
      step();
    }
  }

  void step() {
    std::unique_lock<std::mutex> lock(mutex);
    if (shouldRun && tasks.empty())
      cv.wait(lock);
    if (!shouldRun)
      return;
    if (tasks.empty())
      return;
    SPDLOG_INFO("Running async task");
    auto task = std::move(tasks.front());
    tasks.pop_front();
    lock.unlock();
    task();
    lock.lock();
  }

  void post(std::function<void()> task) {
    std::unique_lock<std::mutex> lock(mutex);
    tasks.push_back(std::move(task));
    cv.notify_one();
  }

  void start() {
    thread = std::thread([this]() {
      SPDLOG_INFO("Async runner starting");
      run();
      // requestAsyncRunnerJSPI(this, &AsyncRunner::step1);
      SPDLOG_INFO("Async runner stopped");
    });
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

    auto str = std::string(msg.payload.begin(), msg.payload.end());
    logBuffer.addMessage(str, msg.level, msg.source.filename ? msg.source.filename : "");
  }

  void flush_() override {}
};

SHCore *core{};
extern "C" {
EMSCRIPTEN_KEEPALIVE void shardsInit() {
  // Add log buffer sink
  shards::logging::setupDefaultLoggerConditional("shards.log");
  shards::logging::getDistSink()->add_sink(std::make_shared<LogBufferSink>());

  core = shardsInterface(SHARDS_CURRENT_ABI);
  shards_init(core); // to init rust things
  asyncRunner.start();
  shards::EmMainProxy::instance = &emMainProxy;

  SPDLOG_INFO("Test log entry");
}

// Needs to be polled on the main thread for audio and other stuff that needs to be run on the main browser thread
EMSCRIPTEN_KEEPALIVE void shardsPollMainProxy() { emMainProxy.poll(); }

static backend_t fetchBackend{};
static std::atomic_bool fetchBackendReady{false};
EMSCRIPTEN_KEEPALIVE void shardsFSMountHTTP(const char *target_, const char *baseUrl_) {
  std::string target{target_};
  std::string baseUrl{baseUrl_};
  asyncRunner.post([target, baseUrl]() {
    SPDLOG_INFO("Mounting HTTP FS at {} with base URL {}", target, baseUrl);
    fetchBackend = wasmfs_create_fetch_backend(baseUrl.c_str());
    int r = wasmfs_create_directory(target.c_str(), 0777, fetchBackend);
    if (r != 0) {
      SPDLOG_ERROR("Failed to mount HTTP FS at {} with base URL {} ({})", target, baseUrl, r);
    }
    fetchBackendReady = true;
  });
}

EMSCRIPTEN_KEEPALIVE bool shardsFetchBackendReady() { return fetchBackendReady; }

EMSCRIPTEN_KEEPALIVE void shardsPreloadFiles(const char *payload, size_t length) {
  using path = boost::filesystem::path;

  path root = "/tfs";

  auto json = nlohmann::json::parse(payload, payload + length);
  if (!json.is_array()) {
    SPDLOG_ERROR("Payload is not an array");
    return;
  }
  SPDLOG_INFO("Preloading {} files", json.size());
  for (size_t i = 0; i < json.size(); i++) {
    auto &elem = json[i];
    if (!elem.is_object()) {
      SPDLOG_ERROR("Element is not an object");
      continue;
    }
    auto f = elem.find("f");
    if (f != elem.end() && f->is_string()) {
      // Add file
      auto p0 = f->get<std::string>();
      std::replace(p0.begin(), p0.end(), '\\', '/');
      auto p1 = (root / p0).lexically_normal().string();
      SPDLOG_INFO("Preloading file {}", p1);
      int fd = wasmfs_create_file(p1.c_str(), 0777, fetchBackend);
      close(fd);
    }
    auto d = elem.find("d");
    if (d != elem.end() && d->is_string()) {
      // Add directory
      auto p0 = d->get<std::string>();
      std::replace(p0.begin(), p0.end(), '\\', '/');
      auto p1 = (root / p0).lexically_normal().string();
      SPDLOG_INFO("Preloading directory {}", p1);
      int fd = wasmfs_create_directory(p1.c_str(), 0777, fetchBackend);
      close(fd);
    }
  }
}

EMSCRIPTEN_KEEPALIVE void shardsCWDAndList(const char *cwd, const char *dirPath) {
  using path = boost::filesystem::path;
  using namespace boost::filesystem;
  path p0 = cwd;
  current_path(p0);
  path p = current_path();
  SPDLOG_INFO("Current path: {}", p.string());

  path dst = dirPath;
  boost::filesystem::directory_iterator it(dst);
  while (it != boost::filesystem::directory_iterator()) {
    SPDLOG_INFO("Dirent: {}", it->path().string());
    ++it;
  }
}

EMSCRIPTEN_KEEPALIVE void shardsCWDAndTryLoad(const char *cwd, const char *filePath) {
  using path = boost::filesystem::path;
  using namespace boost::filesystem;
  path p0 = cwd;
  current_path(p0);
  path p = current_path();
  SPDLOG_INFO("Current path: {}", p.string());

  FILE *f = fopen(filePath, "rb");
  if (!f) {
    SPDLOG_ERROR("Failed to open file {}", filePath);
    return;
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(size);
  fread(buf.data(), 1, size, f);
  fclose(f);

  SPDLOG_INFO("File size: {}", size);
  SPDLOG_INFO("File content: {}", std::string(buf.data(), size));
}

EMSCRIPTEN_KEEPALIVE void shardsPollMainThread() { emscripten_current_thread_process_queued_calls(); }

EMSCRIPTEN_KEEPALIVE void shardsLoadScript(Instance **outInstance, const char *code, const char *base_path) {
  shards::logging::setupDefaultLoggerConditional("");
  if (!outInstance) {
    SPDLOG_ERROR("shardsLoadScript: outInstance is null");
    return;
  }

  Instance *instance = (*outInstance) = new Instance();
  std::vector<SHStringWithLen> includeDirs;

  try {
    instance->mesh = SHMesh::make();

    auto astRes = shards_read("script"_swl, toSWL(code), toSWL(base_path), includeDirs.data(), includeDirs.size());
    shards::OwnedVar ast{astRes.ast};
    if (astRes.error) {
      SPDLOG_ERROR("Failed to read code: {}", astRes.error->message);
      shards_free_error(astRes.error);
      return;
    }

    core->setRootPath(base_path);

    SHLWire shlwire = shards_eval(&astRes.ast, "script"_swl);
    DEFER(shards_free_wire(shlwire));
    if (shlwire.error) {
      SPDLOG_ERROR("Failed to evaluate script at {}:{}: {}", shlwire.error->line, shlwire.error->column, shlwire.error->message);
      instance->error = shlwire.error->message;
      return;
    }

    auto wire = SHWire::sharedFromRef(*shlwire.wire);
    instance->wire = wire;
    instance->mesh->schedule(wire);
    instance->error.reset();

  } catch (std::exception &ex) {
    SPDLOG_ERROR("Unhandled exception in shardsLoadScript: {}", ex.what());
    instance->error = ex.what();
  }
}

EMSCRIPTEN_KEEPALIVE void shardsTick(Instance *instance, uint32_t *resultPtr) {
  uint32_t result = instance->mesh->tick() ? 1 : 0;
  if (instance->mesh->empty()) {
    SPDLOG_DEBUG("Mesh is empty");
    result = 0; 
  }

  if (resultPtr) {
    *resultPtr = result;
  }
}

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

EMSCRIPTEN_KEEPALIVE void shardsFreeInstance(Instance *instance) {
  if (instance->guard != 9999) {
    SPDLOG_ERROR("Invalid instance");
    return;
  }
  instance->guard = 0;
  delete instance;
}

EMSCRIPTEN_KEEPALIVE void shardsLockLogBuffer() { logBuffer.mutex.lock(); }

EMSCRIPTEN_KEEPALIVE void shardsUnlockLogBuffer() {
  logBuffer.clear(); // Clear messages when unlocking
  logBuffer.mutex.unlock();
}

EMSCRIPTEN_KEEPALIVE MessageRep *shardsGetLogMessages(int *count) { return logBuffer.getMessages(count); }

EMSCRIPTEN_KEEPALIVE int main(int argc, char **argv) { return 0; }
}