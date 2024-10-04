#include "taskflow.hpp"
#include <shards/core/platform.hpp>
#include <shards/core/utils.hpp>
#include <spdlog/spdlog.h>

namespace shards {

struct TaskFlowDebugInterface : tf::WorkerInterface {
  std::string debugName;

  TaskFlowDebugInterface(std::string debugName) : debugName(debugName) {}
  void scheduler_prologue(tf::Worker &worker) {
    pushThreadName(fmt::format("<idle> tf::Executor ({}, {})", debugName, worker.id()));
    SPDLOG_TRACE("TaskFlow: \"{}\" Worker {} starting", debugName, worker.id());
  }
  virtual void scheduler_epilogue(tf::Worker &worker, std::exception_ptr ptr) {
    if (ptr) {
      try {
        std::rethrow_exception(ptr);
      } catch (std::exception &e) {
        SPDLOG_ERROR("TaskFlow: \"{}\" Worker {} exiting with exception: {}", debugName, worker.id(), e.what());
      }
    } else {
      SPDLOG_TRACE("TaskFlow: \"{}\" Worker {} exiting", debugName, worker.id());
    }
  }
};

static size_t getNumTaskflowThreads() {
#if SH_EMSCRIPTEN
  return 4;
#else
  return std::max<size_t>(1, std::thread::hardware_concurrency() - 1);
#endif
}

tf::Executor &TaskFlowInstance::instance() {
  static tf::Executor executor(getNumTaskflowThreads(), std::make_shared<TaskFlowDebugInterface>("global"));
  return executor;
}
} // namespace shards