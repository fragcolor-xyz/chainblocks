/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#include "boost/context/continuation_fcontext.hpp"
#include "foundation.hpp"
#include "coro.hpp"
#include "utils.hpp"
#include <memory>
#include <mutex>
#include <thread>

namespace shards {
#if SH_USE_THREAD_FIBER

ThreadFiber::~ThreadFiber() {}

void ThreadFiber::init(std::function<void()> fn) {
  isRunning = true;

  // Initial suspend
  thread.emplace([=]() {
    fn();

    // Final suspend
    finished = true;
    switchToCaller();
  });

  // Wait for initial suspend
  std::unique_lock<decltype(mtx)> l(mtx);
  while (isRunning) {
    cv.wait(l);
  }
}

void ThreadFiber::resume() {
  if (isRunning) {
    switchToCaller();
  } else {
    // SPDLOG_TRACE("CALLER> {}", thread->get_id());
    switchToThread();
    // SPDLOG_TRACE("CALLER< {}", thread->get_id());
    if (finished) {
      thread->join();
      thread.reset();
    }
  }
}

void ThreadFiber::suspend() {
  // SPDLOG_TRACE("RUNNER< {}", thread->get_id());
  switchToCaller();
  // SPDLOG_TRACE("RUNNER> {}", thread->get_id());
}

void ThreadFiber::switchToCaller() {
  std::unique_lock<decltype(mtx)> l(mtx);
  isRunning = false;
  cv.notify_all();
  if (!finished) {
    while (!isRunning) {
      cv.wait(l);
    }
  }
}
void ThreadFiber::switchToThread() {
  std::unique_lock<decltype(mtx)> l(mtx);
  isRunning = true;
  cv.notify_all();
  while (isRunning) {
    cv.wait(l);
  }
}

ThreadFiber::operator bool() const { return !finished; }

#else // SH_USE_THREAD_FIBER
#ifndef __EMSCRIPTEN__

Fiber::Fiber(SHStackAllocator allocator) : allocator(allocator) {}
void Fiber::init(std::function<void()> fn) {
  continuation.emplace(boost::context::callcc(std::allocator_arg, allocator, [=](boost::context::continuation &&sink) {
    continuation.emplace(std::move(sink));
    fn();
    return std::move(continuation.value());
  }));
}
void Fiber::resume() {
  assert(continuation);
  continuation = continuation->resume();
}
void Fiber::suspend() {
  assert(continuation);
  continuation = continuation->resume();
}
Fiber::operator bool() const { return continuation.has_value() && (bool)continuation.value(); }

#else // __EMSCRIPTEN__

thread_local emscripten_fiber_t *em_local_coro{nullptr};
thread_local emscripten_fiber_t em_main_coro{};
thread_local uint8_t em_asyncify_main_stack[Fiber::as_stack_size];

[[noreturn]] static void action(void *p) {
  SHLOG_TRACE("EM FIBER ACTION RUN");
  auto coro = reinterpret_cast<Fiber *>(p);
  coro->func();
  // If entry_func returns, the entire program will end, as if main had
  // returned.
  abort();
}

void Fiber::init(const std::function<void()> &func) {
  SHLOG_TRACE("EM FIBER INIT");
  this->func = func;
  c_stack = new (std::align_val_t{16}) uint8_t[stack_size];
  emscripten_fiber_init(&em_fiber, action, this, c_stack, stack_size, asyncify_stack, as_stack_size);
}

NO_INLINE void Fiber::resume() {
  SHLOG_TRACE("EM FIBER SWAP RESUME {}", reinterpret_cast<uintptr_t>(&em_fiber));
  // ensure local thread is setup
  if (!em_main_coro.stack_ptr) {
    SHLOG_DEBUG("Fiber - initialization of new thread");
    emscripten_fiber_init_from_current_context(&em_main_coro, em_asyncify_main_stack, Fiber::as_stack_size);
  }
  // ensure we have a local coro
  if (!em_local_coro) {
    em_local_coro = &em_main_coro;
  }
  // from current to new
  em_parent_fiber = em_local_coro;
  em_local_coro = &em_fiber;
  emscripten_fiber_swap(em_parent_fiber, &em_fiber);
}

NO_INLINE void Fiber::yield() {
  SHLOG_TRACE("EM FIBER SWAP YIELD {}", reinterpret_cast<uintptr_t>(&em_fiber));
  // always yields to main
  assert(em_parent_fiber);
  em_local_coro = em_parent_fiber;
  emscripten_fiber_swap(&em_fiber, em_parent_fiber);
}

#endif
#endif
} // namespace shards