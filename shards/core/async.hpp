#ifndef F80CEE03_D5CE_4787_8D65_FB8CC200104A
#define F80CEE03_D5CE_4787_8D65_FB8CC200104A

#include <chrono>
#include <thread>
#include <deque>
#include <future>

#include <shards/shards.h>
#include <shards/utility.hpp>
#include "utils.hpp"
#include "runtime.hpp"

#include <boost/lockfree/queue.hpp>
#include <boost/thread.hpp>

#include <tracy/Wrapper.hpp>

#if SH_EMSCRIPTEN
#include <emscripten.h>
#endif

// Can not create this many workers, create on demand instead
#if SH_EMSCRIPTEN
#define SH_ENABLE_TIDE_POOL 0
#else
#define SH_ENABLE_TIDE_POOL 1
#endif

#if !SH_ENABLE_TIDE_POOL
#include "taskflow.hpp"
#endif

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
#define HAS_ASYNC_SUPPORT 0
#else
#define HAS_ASYNC_SUPPORT 1
#endif

#ifndef NDEBUG
// when debugging, set a bigger stack size to accommodate more stack frames
#define SH_STACK_SIZE 0x400000
#else
// when not debugging, set a smaller stack size to reduce memory usage
#define SH_STACK_SIZE 0x100000
#endif

namespace shards {
#if HAS_ASYNC_SUPPORT
/*
 * The TidePool class is a simple C++ thread pool implementation designed to manage
 * worker threads and distribute tasks to them. The class supports dynamic adjustment
 * of the number of worker threads based on the current workload.
 *
 * Features:
 * - Abstract Work struct representing tasks to be executed by the worker threads.
 * - Dynamic adjustment of the number of worker threads based on the number of tasks in the queue.
 * - Lock-free queue for efficient task scheduling.
 * - Configurable minimum, initial, and maximum number of worker threads.
 * - Asynchronous controller thread that manages worker threads.
 * - Simple scheduling function for adding tasks to the queue.
 *
 * Usage:
 * - Derive custom work classes from the Work struct and implement the call() function.
 * - Create an instance of TidePool.
 * - Schedule tasks using the schedule() function.
 * - The pool will automatically adjust the number of worker threads based on the workload.
 */
struct TidePool {
  struct Work {
    virtual void call() = 0;
  };

#if SH_ENABLE_TIDE_POOL
  struct Worker {
    Worker(boost::lockfree::queue<Work *> &queue, std::atomic_size_t &counter, std::mutex &condMutex,
           std::condition_variable &cond)
        : _queue(queue), _counter(counter), _condMutex(condMutex), _cond(cond) {
      using namespace shards::literals;
      _running = true;
      boost::thread::attributes attrs;
      attrs.set_stack_size(SH_STACK_SIZE);
      _thread = boost::thread(attrs, [this]() {
        pushThreadName("TidePool worker"_ns);
        while (_running) {
          Work *work{};
          if (_queue.pop(work)) {
            // SHLOG_DEBUG("TidePool: calling {}", (void*)work);
            work->call();
            _counter--;
          }
          // wait if the queue is empty
          if (_queue.empty()) {
            std::unique_lock<std::mutex> lock(_condMutex);
            _cond.wait(lock, [this]() { return !_queue.empty() || !_running; });
          }
        }
      });
    }

    boost::thread _thread;
    std::atomic_bool _running;
    boost::lockfree::queue<Work *> &_queue;
    std::atomic_size_t &_counter;
    std::mutex &_condMutex;
    std::condition_variable &_cond;
  };

  static constexpr size_t LowWater = 4;
  static constexpr size_t NumWorkers = 8;
  static constexpr size_t MaxWorkers = 32;

  std::atomic_size_t _scheduledCounter;
  std::atomic_bool _running;
  std::thread _controller;
  boost::lockfree::queue<Work *> _queue{NumWorkers};
  std::deque<Worker> _workers;
  std::mutex _condMutex;
  std::condition_variable _cond;

  TidePool() {
    _running = true;
    _controller = std::thread(&TidePool::controllerWorker, this);
  }

  ~TidePool() { terminate(); }

  void terminate() {
    if (!_running)
      return;

    _running = false;
    // Ensure the thread is properly joined on destruction if it is joinable
    if (_controller.joinable()) {
      _controller.join();
    }
  }

  void schedule(Work *work) {
    _scheduledCounter++;
    _queue.push(work);
    _cond.notify_one();
  }

  void controllerWorker() {
    using namespace shards::literals;
    
    pushThreadName("TidePool controller"_ns);

    // spawn workers first
    for (size_t i = 0; i < NumWorkers; ++i) {
      _workers.emplace_back(_queue, _scheduledCounter, _condMutex, _cond);
    }

    while (_running) {
      shassert(_workers.size() >= NumWorkers);

      if (_scheduledCounter < LowWater && _workers.size() > NumWorkers) {
        // we have less than LowWater scheduled and we have more than NumWorkers workers
        auto &superfluousWorker = _workers.back();
        superfluousWorker._running = false;
        _cond.notify_all(); // we don't know which worker in _cond, so we notify all
        if (superfluousWorker._thread.joinable())
          superfluousWorker._thread.join();
        _workers.pop_back();
        // SHLOG_DEBUG("TidePool: worker removed, count: {}", _workers.size());
      } else if (_scheduledCounter > _workers.size() && _workers.size() < MaxWorkers) {
        // we have more scheduled than workers
        _workers.emplace_back(_queue, _scheduledCounter, _condMutex, _cond);
        // SHLOG_DEBUG("TidePool: worker added, count: {}", _workers.size());
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // stop all workers, we need to go thru them twice to notify them all

    for (auto &worker : _workers) {
      worker._running = false;
    }

    _cond.notify_all();

    for (auto &worker : _workers) {
      if (worker._thread.joinable())
        worker._thread.join();
    }
  }
#else // Dummy implementation
  void schedule(Work *work) {
    tf::Taskflow flow;
    flow.emplace([=]() { work->call(); });
    TaskFlowInstance::instance().run(std::move(flow));
  }
  void terminate() {}
#endif
};

TidePool &getTidePool();

#endif

template <typename FUNC, typename CANCELLATION>
inline SHVar awaitne(SHContext *context, FUNC &&func, CANCELLATION &&cancel) noexcept {
  static_assert(std::is_same_v<decltype(func()), SHVar> || std::is_same_v<decltype(func()), Var>,
                "func must return SHVar or Var");
  ZoneScopedN("awaitne");

  if (context->onWorkerThread) {
    try {
      return func();
    } catch (const std::exception &e) {
      context->cancelFlow(e.what());
    } catch (...) {
      context->cancelFlow("foreign exception failure");
    }
    return SHVar(); // Return a default SHVar in case of an error
  }

#if !HAS_ASYNC_SUPPORT
  return func();
#else
  struct BlockingCall : TidePool::Work {
    BlockingCall(FUNC &&func) : func(std::move(func)), exp(), res(), complete(false) {}

    FUNC &&func;

    std::exception_ptr exp;
    SHVar res;
    std::atomic_bool complete;

    virtual void call() {
      ZoneScopedNC("awaitne-work", 0xFF00FF00);

      try {
        res = func();
      } catch (...) {
        exp = std::current_exception();
      }
      complete = true;
    }
  } call{std::forward<FUNC>(func)};

  context->onWorkerThread = true;
  DEFER(shassert(!context->onWorkerThread && "context still flagged on worker thread"));

  getTidePool().schedule(&call);

  while (!call.complete && context->shouldContinue()) {
    if (shards::suspend(context, 0) != SHWireState::Continue)
      break;
  }

  context->onWorkerThread = false;

  if (unlikely(!call.complete)) {
    cancel();
    while (!call.complete) {
      std::this_thread::yield();
    }
  }

  if (call.exp) {
    try {
      std::rethrow_exception(call.exp);
    } catch (const std::exception &e) {
      context->cancelFlow(e.what());
    } catch (...) {
      context->cancelFlow("foreign exception failure");
    }
  } else if (call.res.flags == SHVAR_FLAGS_ABORT) { // not a bit check we don't expect any other flags
    auto msg = SHSTRVIEW(call.res);
    context->cancelFlow(msg);
    destroyVar(call.res);
  }

  return call.res;
#endif
}

template <typename FUNC, typename CANCELLATION> inline void await(SHContext *context, FUNC &&func, CANCELLATION &&cancel) {
  ZoneScopedN("await");

  if (context->onWorkerThread) {
    // When already on a worker thread, just call the function directly
    func();
    return;
  }

#if !HAS_ASYNC_SUPPORT
  func();
#else
  struct BlockingCall : TidePool::Work {
    BlockingCall(FUNC &&func) : func(std::move(func)), exp(), complete(false) {}

    FUNC &&func;

    std::exception_ptr exp;
    std::atomic_bool complete;

    virtual void call() {
      ZoneScopedNC("await-work", 0xFF00FF00);
      try {
        func();
      } catch (...) {
        exp = std::current_exception();
      }
      complete = true;
    }
  } call{std::forward<FUNC>(func)};

  context->onWorkerThread = true;
  DEFER(shassert(!context->onWorkerThread && "context still flagged on worker thread"));

  getTidePool().schedule(&call);

  while (!call.complete.load(std::memory_order_acquire) && context->shouldContinue()) {
    if (shards::suspend(context, 0) != SHWireState::Continue)
      break;
  }

  context->onWorkerThread = false;

  if (unlikely(!call.complete)) {
    cancel();
    while (!call.complete) {
      std::this_thread::yield();
    }
  }

  if (call.exp) {
    std::rethrow_exception(call.exp);
  }
#endif
}

} // namespace shards

#endif /* F80CEE03_D5CE_4787_8D65_FB8CC200104A */
