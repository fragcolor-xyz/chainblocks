#include "shared_temp_allocator.hpp"
#include <shards/log/log.hpp>
#include <tracy/Wrapper.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

static shards::logging::Logger getLogger() {
  static auto logger = shards::logging::getOrCreate("SharedTempAllocator");
  return logger;
}

#define SHARDS_TEMP_ALLOCATOR_TRACE 1
#if SHARDS_TEMP_ALLOCATOR_TRACE
#define STA_TRACE(...) SPDLOG_LOGGER_TRACE(getLogger(), __VA_ARGS__)
#else
#define STA_TRACE(...)
#endif

namespace shards::pmr {
struct SharedTempAllocatorImpl {
  TempAllocator allocator;
  int refCount;

  char debugName[64];

  SharedTempAllocatorImpl() {
    STA_TRACE("[{}] Temp allocator created", std::this_thread::get_id());
    fmt::format_to_n((char*)debugName, std::size(debugName), "STA_{}", std::this_thread::get_id());
  }

  void incRef() {
    if (refCount == 0) {
      STA_TRACE("[{}] Temp allocator reset ({} bytes)", std::this_thread::get_id(), allocator.preallocatedBlock.size());
      TracyPlot(debugName, (int64_t)allocator.preallocatedBlock.size());
      allocator.reset();
    }
    ++refCount;
  }

  void decRef() {
    shassert(refCount >= 0);
    --refCount;
  }
};

static std::optional<SharedTempAllocatorImpl> &getLocalInstance() {
  static thread_local std::optional<SharedTempAllocatorImpl> pool;
  return pool;
}

shards::pmr::memory_resource *SharedTempAllocator::getAllocator() const { return &impl->allocator; }
SharedTempAllocator::SharedTempAllocator() {
  if (!getLocalInstance()) {
    getLocalInstance().emplace();
  }
  impl = &getLocalInstance().value();
  impl->incRef();
}
SharedTempAllocator::~SharedTempAllocator() { reset(); }
void SharedTempAllocator::reset() {
  if (impl) {
    impl->decRef();
    impl = nullptr;
  }
}
} // namespace shards::pmr