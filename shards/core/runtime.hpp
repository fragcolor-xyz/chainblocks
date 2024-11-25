/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#ifndef SH_CORE_RUNTIME
#define SH_CORE_RUNTIME

// must go first
#include <shards/shards.h>
#include <type_traits>

#if _WIN32
#include <winsock2.h>
#endif

#include <string.h> // memset

#include "pmr/wrapper.hpp"
#include "pmr/unordered_map.hpp"
#include "pmr/shared_temp_allocator.hpp"
#include "shards_macros.hpp"
#include "foundation.hpp"
#include "inline.hpp"
#include "utils.hpp"
#include "object_type.hpp"
#include "platform.hpp"

#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/container/flat_set.hpp>

using SHClock = std::chrono::high_resolution_clock;
using SHTime = decltype(SHClock::now());
using SHDuration = std::chrono::duration<double>;
using SHTimeDiff = decltype(SHClock::now() - SHDuration(0.0));

// For sleep
#if _WIN32
#include <Windows.h>
#else
#include <time.h>
#endif

#if SH_EMSCRIPTEN
#include <emscripten.h>
#include <emscripten/val.h>
#endif

#ifdef SH_USE_TSAN
extern "C" {
void *__tsan_get_current_fiber(void);
void *__tsan_create_fiber(unsigned flags);
void __tsan_destroy_fiber(void *fiber);
void __tsan_switch_to_fiber(void *fiber, unsigned flags);
void __tsan_set_fiber_name(void *fiber, const char *name);
const unsigned __tsan_switch_to_fiber_no_sync = 1 << 0;
}
#define TSANCoroEnter(wire)              \
  {                                      \
    if (!getCoroWireStack2().empty()) {  \
      TracyFiberLeave;                   \
    }                                    \
    TracyFiberEnter(wire->name.c_str()); \
    getCoroWireStack2().push_back(wire); \
  }
#define TSANCoroExit(wire)                                       \
  {                                                              \
    getCoroWireStack2().pop_back();                              \
    TracyFiberLeave;                                             \
    if (!getCoroWireStack2().empty()) {                          \
      TracyFiberEnter(getCoroWireStack2().back()->name.c_str()); \
    }                                                            \
  }
#else
#define TSANCoroEnter(wire)
#define TSANCoroExit(wire)
#endif

#define XXH_INLINE_ALL
#include <xxhash.h>

#ifndef CUSTOM_XXH3_kSecret
// Applications embedding shards can override this and should.
// TODO add our secret
#define CUSTOM_XXH3_kSecret XXH3_kSecret
#endif

#define SH_SUSPEND(_ctx_, _secs_)                             \
  const auto _suspend_state = shards::suspend(_ctx_, _secs_); \
  if (_suspend_state != SHWireState::Continue)                \
  return shards::Var::Empty

struct SHStateSnapshot {
  SHWireState state;
  SHVar flowStorage;
  std::string errorMessage;
};

struct SHContext {
  SHContext(shards::Coroutine *coro, const SHWire *starter) : main(starter), continuation(coro) {
    wireStack.push_back(const_cast<SHWire *>(starter));
  }

  const SHWire *main;
  SHContext *parent{nullptr};
  std::vector<SHWire *> wireStack;
  bool onLastResume{false};
  bool onWorkerThread{false};
  uint64_t stepCounter{};
  volatile void *stackStart{nullptr};

  // Used within the coro& stack! (suspend, etc)
  shards::Coroutine *continuation{nullptr};
  SHDuration next{};

  entt::delegate<void()> meshThreadTask;

  SHWire *rootWire() const { return wireStack.front(); }
  SHWire *currentWire() const { return wireStack.back(); }

  constexpr void stopFlow() { state = SHWireState::Stop; }

  constexpr void stopFlow(const SHVar &lastValue) {
    state = SHWireState::Stop;
    flowStorage = lastValue;
  }

  constexpr void restartFlow(const SHVar &lastValue) {
    state = SHWireState::Restart;
    flowStorage = lastValue;
  }

  constexpr void returnFlow(const SHVar &lastValue) {
    state = SHWireState::Return;
    flowStorage = lastValue;
  }

  void cancelFlow(std::string_view message) {
    SHLOG_DEBUG("Cancelling flow: {}", message);
    state = SHWireState::Error;
    errorMessage = message;
  }

  void pushError(std::string &&message) { errorStack.emplace_back(std::move(message)); }
  void resetErrorStack() { errorStack.clear(); }
  std::string formatErrorStack() {
    // reverse order
    std::string out;
    for (auto it = errorStack.rbegin(); it != errorStack.rend(); ++it) {
      out += *it;
      out += "\n";
    }
    return out;
  }

  SHStateSnapshot takeStateSnapshot() {
    return SHStateSnapshot{
        state,
        std::move(flowStorage),
        std::move(errorMessage),
    };
  }

  void restoreStateSnapshot(SHStateSnapshot &&snapshot) {
    errorMessage = std::move(snapshot.errorMessage);
    state = std::move(snapshot.state);
    flowStorage = std::move(snapshot.flowStorage);
  }

  constexpr void rebaseFlow() { state = SHWireState::Rebase; }

  constexpr void continueFlow() { state = SHWireState::Continue; }

  constexpr bool shouldContinue() const { return state == SHWireState::Continue; }

  constexpr bool shouldReturn() const { return state == SHWireState::Return; }

  constexpr bool shouldStop() const { return state == SHWireState::Stop; }

  constexpr bool failed() const { return state == SHWireState::Error; }

  constexpr const std::string &getErrorMessage() { return errorMessage; }

  constexpr SHWireState getState() const { return state; }

  constexpr SHVar &getFlowStorage() { return flowStorage; }

  void mirror(const SHContext *other) {
    state = other->state;
    flowStorage = other->flowStorage;
    errorMessage = other->errorMessage;
    errorStack = other->errorStack;
  }

private:
  SHWireState state = SHWireState::Continue; // don't make this atomic! it's used a lot!
  // Used when flow is stopped/restart/return
  // to store the previous result
  SHVar flowStorage{};
  std::string errorMessage;
  std::vector<std::string> errorStack;
};

namespace shards {
[[nodiscard]] SHComposeResult composeWire(const std::vector<Shard *> &wire, SHInstanceData data);
[[nodiscard]] SHComposeResult composeWire(const Shards wire, SHInstanceData data);
[[nodiscard]] SHComposeResult composeWire(const SHSeq wire, SHInstanceData data);
[[nodiscard]] SHComposeResult composeWire(const SHWire *wire, SHInstanceData data);

SHVar *findVariable(SHContext *ctx, std::string_view name);

bool validateSetParam(Shard *shard, int index, const SHVar &value);
bool matchTypes(const SHTypeInfo &inputType, const SHTypeInfo &receiverType, bool isParameter, bool strict,
                bool relaxEmptySeqCheck, bool ignoreFixedSeq = false);
void triggerVarValueChange(SHContext *context, const SHVar *name, bool isGlobal, const SHVar *var);
void triggerVarValueChange(SHWire *wire, const SHVar *name, bool isGlobal, const SHVar *var);

void installSignalHandlers();

bool isDebuggerPresent();

#ifdef SH_COMPRESSED_STRINGS
void decompressStrings();
#endif

struct DefaultHelpText {
  static inline const SHOptionalString InputHelpIgnoredOrPass =
      SHCCSTR("Any input type is accepted. The input value will either pass through unchanged or be ignored.");

  static inline const SHOptionalString InputHelpAnyType = SHCCSTR("Input of any type is accepted.");

  static inline const SHOptionalString InputHelpAnyButType =
      SHCCSTR("Input of any type is accepted. For types without inherent value (e.g., None, Bool), a lexicographical comparison "
              "is used.");

  static inline const SHOptionalString InputHelpIgnored = SHCCSTR("The input of this shard is ignored.");

  static inline const SHOptionalString InputHelpPass =
      SHCCSTR("Any input type is accepted. The input value will pass through unchanged.");

  static inline const SHOptionalString OutputHelpPass = SHCCSTR("Outputs the input value, passed through unchanged.");
};

template <typename T> struct AnyStorage {
private:
  std::shared_ptr<entt::any> _anyStorage;
  T *_ptr{};

public:
  AnyStorage() = default;
  AnyStorage(std::shared_ptr<entt::any> &&any) : _anyStorage(any) { _ptr = &entt::any_cast<T &>(*_anyStorage.get()); }
  operator bool() const { return _ptr; }
  T *operator->() const { return _ptr; }
  operator T &() const { return *_ptr; }
};

template <typename TInit, typename T = decltype((*(TInit *)0)()), typename C>
AnyStorage<T> getOrCreateAnyStorage(C *context, const std::string &storageKey, TInit init) {
  std::unique_lock<std::mutex> l(context->anyStorageLock);
  auto ptr = context->anyStorage[storageKey];
  if (!ptr) {
    // recurse into parent if we have one
    shassert(context->parent != context);
    if (context->parent) {
      l.unlock();
      return getOrCreateAnyStorage<TInit, T>(context->parent, storageKey, init);
    } else {
      ptr = std::make_shared<entt::any>(init());
      context->anyStorage[storageKey] = ptr;
      return ptr;
    }
  } else {
    return ptr;
  }
}

template <typename T, typename C> AnyStorage<T> getOrCreateAnyStorage(C *context, const std::string &storageKey) {
  auto v = []() { return std::in_place_type_t<T>(); };
  return getOrCreateAnyStorage<decltype(v), T>(context, storageKey, v);
}

SHRunWireOutput runWire(SHWire *wire, SHContext *context, const SHVar &wireInput);

inline SHRunWireOutput runSubWire(SHWire *wire, SHContext *context, const SHVar &input) {
  // push to wire stack
  context->wireStack.push_back(wire);
  DEFER({ context->wireStack.pop_back(); });
  auto runRes = shards::runWire(wire, context, input);
  return runRes;
}

void run(SHWire *wire, shards::Coroutine *coro);

#ifdef TRACY_ENABLE
// Defined in the gfx rust crate
//   used to initialize tracy on the rust side, since it required special intialization (C++ doesn't)
//   but since we link to the dll, we can use it from C++ too
extern "C" void gfxTracyInit();

struct GlobalTracy {
  GlobalTracy() { gfxTracyInit(); }

  // just need to fool the compiler
  constexpr bool isInitialized() const { return true; }
};

extern GlobalTracy &GetTracy();
#endif

#ifdef TRACY_FIBERS
std::vector<SHWire *> &getCoroWireStack();
#endif

#ifdef SH_VERBOSE_COROUTINES_LOGGING
#define SH_CORO_RESUMED_LOG(_wire)                   \
  {                                                  \
    SHLOG_TRACE("> Resumed wire {}", (_wire)->name); \
  }
#define SH_CORO_SUSPENDED_LOG(_wire)                   \
  {                                                    \
    SHLOG_TRACE("> Suspended wire {}", (_wire)->name); \
  }
#define SH_CORO_EXT_RESUME_LOG(_wire)               \
  {                                                 \
    SHLOG_TRACE("Resuming wire {}", (_wire)->name); \
  }
#define SH_CORO_EXT_SUSPEND_LOG(_wire)                \
  {                                                   \
    SHLOG_TRACE("Suspending wire {}", (_wire)->name); \
  }
#else
#define SH_CORO_RESUMED_LOG(_wire)
#define SH_CORO_SUSPENDED_LOG(_wire)
#define SH_CORO_EXT_RESUME_LOG(_wire)
#define SH_CORO_EXT_SUSPEND_LOG(_wire)
#endif

#if SH_DEBUG_THREAD_NAMES
#define SH_CORO_RESUMED(_wire)                                         \
  {                                                                    \
    shards::pushThreadName(fmt::format("Wire \"{}\"", (_wire)->name)); \
    SH_CORO_RESUMED_LOG(_wire)                                         \
  }
#define SH_CORO_SUSPENDED(_wire)   \
  {                                \
    shards::popThreadName();       \
    SH_CORO_EXT_SUSPEND_LOG(_wire) \
  }
#define SH_CORO_EXT_RESUME(_wire)                                                 \
  {                                                                               \
    shards::pushThreadName(fmt::format("<resuming wire> \"{}\"", (_wire)->name)); \
    TracyCoroEnter(_wire);                                                        \
    SH_CORO_EXT_RESUME_LOG(_wire);                                                \
  }
#define SH_CORO_EXT_SUSPEND(_wire) \
  {                                \
    shards::popThreadName();       \
    TracyCoroExit(_wire);          \
    SH_CORO_EXT_SUSPEND_LOG(_wire) \
  }
#else
#define SH_CORO_RESUMED(_wire) SH_CORO_RESUMED_LOG(_wire)
#define SH_CORO_SUSPENDED(_wire) SH_CORO_SUSPENDED_LOG(_wire)
#define SH_CORO_EXT_RESUME(_wire) \
  {                               \
    TracyCoroEnter(_wire);        \
    SH_CORO_EXT_RESUME_LOG(_wire) \
  }
#define SH_CORO_EXT_SUSPEND(_wire) \
  {                                \
    TracyCoroExit(_wire);          \
    SH_CORO_EXT_SUSPEND_LOG(_wire) \
  }
#endif

inline void prepare(SHWire *wire) {
  shassert(!coroutineValid(wire->coro) && "Wire already prepared!");

  auto runner = [wire]() {
#if SH_USE_THREAD_FIBER
    pushThreadName(fmt::format("<suspended wire> \"{}\"", wire->name));
#endif
    run(wire, &wire->coro);
  };

#if SH_CORO_NEED_STACK_MEM
  if (!wire->stackMem) {
    wire->stackMem = new (std::align_val_t{16}) uint8_t[wire->stackSize];
  }
  wire->coro.emplace(SHStackAllocator{wire->stackSize, wire->stackMem});
#else
  wire->coro.emplace();
#endif

  SH_CORO_EXT_RESUME(wire);
  wire->coro->init(runner);
  SH_CORO_EXT_SUSPEND(wire);
}

inline void start(SHWire *wire, SHVar input = {}) {
  if (wire->state != SHWire::State::Prepared) {
    SHLOG_ERROR("Attempted to start a wire ({}) not ready for running!", wire->name);
    return;
  }

  if (!coroutineValid(wire->coro))
    return; // check if not null and bool operator also to see if alive!

  wire->currentInput = input;
  wire->state = SHWire::State::Starting;
}

inline bool isRunning(SHWire *wire) {
  const auto state = wire->state.load(); // atomic
  return state >= SHWire::State::Starting && state <= SHWire::State::IterationEnded;
}

template <bool IsCleanupContext = false> inline void tick(SHWire *wire, SHDuration now) {
  ZoneScoped;
  ZoneName(wire->name.c_str(), wire->name.size());

  while (true) {
    bool canRun = false;
    if constexpr (IsCleanupContext) {
      canRun = true;
    } else {
      canRun = (isRunning(wire) && now >= wire->context->next) || unlikely(wire->context && wire->context->onLastResume);
    }

    if (canRun) {
      shassert(wire->context && "Wire has no context!");
      shassert(coroutineValid(wire->coro) && "Wire has no coroutine!");

      SH_CORO_EXT_RESUME(wire);
      coroutineResume(wire->coro);
      SH_CORO_EXT_SUSPEND(wire);

      // if we have a task to run, run it and resume coro without yielding to caller
      if (unlikely(wire->context && (bool)wire->context->meshThreadTask)) {
        shassert(wire->context->parent == nullptr && "Mesh thread task should only be called on root context!");
        wire->context->meshThreadTask();
        wire->context->meshThreadTask.reset();
        // And continue in order to resume the coroutine
      } else {
        // Yield to caller if no main thread task
        return;
      }
    } else {
      // We can't run, so we yield to caller
      return;
    }
  }
}

bool stop(SHWire *wire, SHVar *result = nullptr, SHContext *currentContext = nullptr);

inline bool hasEnded(SHWire *wire) { return wire->state > SHWire::State::IterationEnded; }

inline bool isCanceled(SHContext *context) { return context->shouldStop(); }

inline void sleep(double seconds = -1.0) {
  // negative = no sleep
  if (seconds > 0.0) {
#ifdef _WIN32
    HANDLE timer;
    LARGE_INTEGER ft;
    ft.QuadPart = -(int64_t(seconds * 10000000));
    timer = CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
    WaitForSingleObject(timer, INFINITE);
    CloseHandle(timer);
#elif __EMSCRIPTEN__
    unsigned int ms = floor(seconds * 1000.0);
    emscripten_sleep(ms);
#else
    struct timespec delay;
    seconds += 0.5e-9; // add half epsilon
    delay.tv_sec = (decltype(delay.tv_sec))seconds;
    delay.tv_nsec = (seconds - delay.tv_sec) * 1000000000L;
    while (nanosleep(&delay, &delay))
      (void)0;
#endif
  } else if (0.0) {
    // just yield to kernel
    std::this_thread::yield();
  }
}

struct RuntimeCallbacks {
  // TODO, turn them into filters maybe?
  virtual void registerShard(const char *fullName, SHShardConstructor constructor) = 0;
  virtual void registerObjectType(int32_t vendorId, int32_t typeId, SHObjectInfo info) = 0;
  virtual void registerEnumType(int32_t vendorId, int32_t typeId, SHEnumInfo info) = 0;
};

struct CompositionContext {
  pmr::SharedTempAllocator tempAllocator;
  shards::pmr::unordered_map<SHWire *, SHTypeInfo> visitedWires;
  std::vector<std::string> errorStack;

  shards::LayeredMap<std::string_view, SHExposedTypeInfo> inherited;

  CompositionContext() : visitedWires(tempAllocator.getAllocator()) {}
};
}; // namespace shards

struct SHMesh : public std::enable_shared_from_this<SHMesh> {
  static constexpr uint32_t TypeId = 'brcM';
  static inline shards::Type MeshType{{SHType::Object, {.object = {.vendorId = shards::CoreCC, .typeId = TypeId}}}};
  static inline shards::ObjectVar<std::shared_ptr<SHMesh>> MeshVar{"Mesh", shards::CoreCC, TypeId};

  static std::shared_ptr<SHMesh> make(std::string_view label = "") { return std::shared_ptr<SHMesh>(new SHMesh(label)); }

  static std::shared_ptr<SHMesh> *makePtr(std::string_view label = "") { return new std::shared_ptr<SHMesh>(new SHMesh(label)); }

  ~SHMesh() { terminate(); }

  void prettyCompose(const std::shared_ptr<SHWire> &wire, SHInstanceData &data) {
    shards::CompositionContext privateContext;
    data.privateContext = &privateContext;
    try {
      auto validation = shards::composeWire(wire.get(), data);
      shards::arrayFree(validation.exposedInfo);
      shards::arrayFree(validation.requiredInfo);
    } catch (const std::exception &e) {
      // build a reverse stack error log from privateContext.errorStack
      std::string errors;
      for (auto it = privateContext.errorStack.rbegin(); it != privateContext.errorStack.rend(); ++it) {
        errors += *it;
        if (++it == privateContext.errorStack.rend())
          break;
        errors += "\n";
      }
      SHLOG_ERROR("Wire {} failed to compose:\n{}", wire->name, errors);
      throw;
    }
  }

  void compose(const std::shared_ptr<SHWire> &wire, SHVar input = shards::Var::Empty) {
    ZoneScoped;

    SHLOG_TRACE("Composing wire {}", wire->name);

    if (wire->warmedUp) {
      SHLOG_ERROR("Attempted to Pre-composing a wire multiple times, wire: {}", wire->name);
      throw shards::SHException("Multiple wire Pre-composing");
    }

    wire->mesh = shared_from_this();

    wire->isRoot = true;
    // remove when done here
    DEFER(wire->isRoot = false);

    // compose the wire
    SHInstanceData data = instanceData;
    data.wire = wire.get();
    data.inputType = shards::deriveTypeInfo(input, data);
    DEFER({ shards::freeDerivedInfo(data.inputType); });
    prettyCompose(wire, data);

    SHLOG_TRACE("Wire {} composed", wire->name);
  }

  struct EmptyObserver {
    void before_compose(SHWire *wire) {}
    void before_tick(SHWire *wire) {}
    void before_stop(SHWire *wire) {}
    void before_prepare(SHWire *wire) {}
    void before_start(SHWire *wire) {}
  };

  template <class Observer>
  void schedule(Observer &observer, const std::shared_ptr<SHWire> &wire, SHVar input = shards::Var::Empty, bool compose = true) {
    ZoneScoped;

    SHLOG_TRACE("Scheduling wire {}", wire->name);

    if (wire->warmedUp || _scheduledSet.count(wire.get())) {
      SHLOG_ERROR("Attempted to schedule a wire multiple times, wire: {}", wire->name);
      throw shards::SHException("Multiple wire schedule");
    }

    wire->mesh = shared_from_this();

    observer.before_compose(wire.get());
    if (compose) {
      wire->isRoot = true;
      // remove when done here
      DEFER(wire->isRoot = false);

      // compose the wire
      SHInstanceData data = instanceData;
      data.wire = wire.get();
      data.inputType = shards::deriveTypeInfo(input, data);
      DEFER({ shards::freeDerivedInfo(data.inputType); });
      prettyCompose(wire, data);

      SHLOG_TRACE("Wire {} composed", wire->name);
    } else {
      SHLOG_TRACE("Wire {} skipped compose", wire->name);
    }

    observer.before_prepare(wire.get());
    shards::prepare(wire.get());

    // wire might fail on warmup during prepare
    if (wire->state == SHWire::State::Failed) {
      throw shards::SHException(fmt::format("Wire {} failed during prepare", wire->name));
    }

    observer.before_start(wire.get());
    shards::start(wire.get(), input);

    _pendingUnschedule.erase(wire);
    _pendingSchedule.insert(wire);
    _scheduledSet.insert(wire.get());

    SHLOG_TRACE("Wire {} scheduled", wire->name);
  }

  void schedule(const std::shared_ptr<SHWire> &wire, SHVar input = shards::Var::Empty, bool compose = true) {
    EmptyObserver obs;
    schedule(obs, wire, input, compose);
  }

  template <class Observer> bool tick(Observer &observer) {
    ZoneScoped;

    auto noErrors = true;
    _errors.clear();
    _failedWires.clear();

    // schedule next
    _scheduled.insert(_pendingSchedule.begin(), _pendingSchedule.end());
    _pendingSchedule.clear();

    if (shards::GetGlobals().SigIntTerm > 0) {
      terminate();
    } else {
      SHDuration now = SHClock::now().time_since_epoch();
      auto it = _scheduled.begin();
      while (it != _scheduled.end()) {
        auto wire = (*it).get();
        if (wire->paused) {
          ++it;
          continue; // simply skip
        }

        observer.before_tick(wire);
        shards::tick(wire->tickingWire(), now);

        if (unlikely(!shards::isRunning(wire))) {
          if (wire->finishedError.size() > 0) {
            _errors.emplace_back(wire->finishedError);
          }

          if (wire->state == SHWire::State::Failed) {
            _failedWires.emplace_back(wire);
            noErrors = false;
          }

          observer.before_stop(wire);
          if (!shards::stop(wire)) {
            noErrors = false;
          }

          // stop should have done the following:
          SHLOG_TRACE("Wire {} ended while ticking", wire->name);
          shassert(wire->mesh.expired() && "Wire still has a mesh!");

          _pendingUnschedule.insert(wire->shared_from_this());
          _scheduledSet.erase(wire);
        }

        ++it;
      }

      // unschedule at the end
      for (const auto &item : _pendingUnschedule) {
        _scheduled.erase(item);
      }
      _pendingUnschedule.clear();
    }

    return noErrors;
  }

  bool tick() {
    EmptyObserver obs;
    return tick(obs);
  }

  friend struct SHWire;
  void clear() {
    auto it = _scheduled.begin();
    while (it != _scheduled.end()) {
      SHLOG_TRACE("Mesh {} stopping scheduled wire {}", label, (*it)->name);
      shards::stop((*it).get());
      ++it;
    }
    auto it2 = _pendingSchedule.begin();
    while (it2 != _pendingSchedule.end()) {
      SHLOG_TRACE("Mesh {} stopping pending wire {}", label, (*it2)->name);
      shards::stop((*it2).get());
      ++it2;
    }

    // release all wires
    _scheduled.clear();
    _scheduledSet.clear();
    _pendingSchedule.clear();
    _pendingUnschedule.clear();

    // find dangling variables and notice
    for (auto var : variables) {
      if (var.second.refcount > 0) {
        SHLOG_ERROR("Found a dangling global variable: {}", var.first);
      }
    }
    variables.clear();
  }

  void terminate() {
    clear();

    // whichever shard uses refs must clean them
    refs.clear();

    // finally clear storage
    anyStorage.clear();
  }

  void remove(const std::shared_ptr<SHWire> &wire) {
    shards::stop(wire.get());
    // stop cause wireOnCleanup to be called
    shassert(wire->mesh.expired() && "Wire still has a mesh!");
    // queue for unschedule
    _pendingUnschedule.insert(wire);
    _scheduledSet.erase(wire.get());
  }

  bool empty() { return _scheduledSet.empty(); }

  bool isCleared() { return _scheduled.empty() && variables.empty(); }

  size_t scheduledSetCount() { return _scheduledSet.size(); }
  size_t scheduledCount() { return _scheduled.size(); }

  const std::vector<std::string> &errors() { return _errors; }

  const std::vector<SHWire *> &failedWires() { return _failedWires; }

  SHInstanceData instanceData{};

  std::mutex anyStorageLock;
  std::unordered_map<std::string, std::shared_ptr<entt::any>> anyStorage;
  SHMesh *parent{nullptr};

  // up to the users to call .update on this, we internally use just "trigger", which is instant
  mutable entt::dispatcher dispatcher{};

  SHVar &getVariable(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name); // copy on write
    return variables[key];
  }

  constexpr auto &getVariables() { return variables; }

  void setMetadata(SHVar *var, SHExposedTypeInfo info) {
    auto it = variablesMetadata.find(var);
    if (it != variablesMetadata.end()) {
      if (info != it->second) {
        SHLOG_WARNING("Metadata for global variable {} already exists and is different!", info.name);
      }
    }
    variablesMetadata[var] = info;
  }

  void releaseMetadata(SHVar *var) {
    if (var->refcount == 0) {
      variablesMetadata.erase(var);
    }
  }

  std::optional<SHExposedTypeInfo> getMetadata(SHVar *var) {
    auto it = variablesMetadata.find(var);
    if (it != variablesMetadata.end()) {
      return it->second;
    } else {
      return std::nullopt;
    }
  }

  void addRef(const SHStringWithLen name, SHVar *var) {
    shassert(((var->flags & SHVAR_FLAGS_REF_COUNTED) == SHVAR_FLAGS_REF_COUNTED && var->refcount > 0) ||
             (var->flags & SHVAR_FLAGS_EXTERNAL) == SHVAR_FLAGS_EXTERNAL);
    auto key = shards::OwnedVar::Foreign(name); // copy on write
    refs[key] = var;
  }

  std::optional<std::reference_wrapper<SHVar>> getVariableIfExists(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name);
    auto it = variables.find(key);
    if (it != variables.end()) {
      return it->second;
    } else {
      return std::nullopt;
    }
  }

  SHVar *getRefIfExists(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name);
    auto it = refs.find(key);
    if (it != refs.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  bool hasRef(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name);
    return refs.count(key) > 0;
  }

  void setLabel(std::string_view label) { this->label = label; }
  std::string_view getLabel() const { return label; }

  void unschedule(const std::shared_ptr<SHWire> &wire) {
    _pendingUnschedule.insert(wire);
    _scheduledSet.erase(wire.get());
  }

private:
  SHMesh(std::string_view label) : label(label) {}

  std::unordered_map<shards::OwnedVar, SHVar, std::hash<shards::OwnedVar>, std::equal_to<shards::OwnedVar>,
                     boost::alignment::aligned_allocator<std::pair<const shards::OwnedVar, SHVar>, 16>>
      variables;

  std::unordered_map<SHVar *, SHExposedTypeInfo> variablesMetadata;

  // variables with lifetime managed externally
  std::unordered_map<shards::OwnedVar, SHVar *, std::hash<shards::OwnedVar>, std::equal_to<shards::OwnedVar>,
                     boost::alignment::aligned_allocator<std::pair<const shards::OwnedVar, SHVar *>, 16>>
      refs;

  struct WireLess {
    bool operator()(const std::shared_ptr<SHWire> &a, const std::shared_ptr<SHWire> &b) const {
      shassert(a && b && "WireLess should not be called with a null pointer");
      return *a < *b;
    }
  };
  // Notice, has to be stable_vector to ensure iterator stability
  using WirePtr = std::shared_ptr<SHWire>;
  using ScheduledSet = boost::container::flat_set<WirePtr, WireLess>;
  ScheduledSet _scheduled;
  std::unordered_set<SHWire *> _scheduledSet;
  ScheduledSet _pendingSchedule;
  ScheduledSet _pendingUnschedule;

  std::vector<std::string> _errors;
  std::vector<SHWire *> _failedWires;
  std::string label;
};

namespace shards {
inline bool stop(SHWire *wire, SHVar *result, SHContext *currentContext) {
  if (wire->state == SHWire::State::Stopped) {
    // Clone the results if we need them
    if (result)
      cloneVar(*result, wire->finishedOutput);

    return true;
  }

  bool stopping = false; // <- expected
  // if exchange fails, we are already stopping
  if (!const_cast<SHWire *>(wire)->stopping.compare_exchange_strong(stopping, true))
    return true;
  DEFER({ wire->stopping = false; });

  SHLOG_TRACE("stopping wire: {}, has-coro: {}, state: {}", wire->name, bool(wire->coro),
              magic_enum::enum_name<SHWire::State>(wire->state));

  if (coroutineValid(wire->coro)) {
    // Run until exit if alive, need to propagate to all suspended shards!
    if (coroutineValid(wire->coro) && wire->state > SHWire::State::Stopped && wire->state < SHWire::State::Failed) {
      // set abortion flag, we always have a context in this case
      wire->context->stopFlow(shards::Var::Empty);
      wire->context->onLastResume = true;

      // BIG Warning: wire->context existed in the coro stack!!!
      // after this resume wire->context is trash!

      // Another issue, if we resume from current context to current context we dead lock here!!
      if (currentContext && currentContext == wire->context) {
        SHLOG_WARNING("Trying to stop wire {} from the same context it's running in!", wire->name);
      } else {
        shards::tick<true>(wire->tickingWire(), SHDuration{});
      }
    }

    // delete also the coro ptr
    wire->coro.reset();
  } else {
    auto mesh = wire->mesh.lock();

    // if we had a coro this will run inside it!
    wire->cleanup(true);

    // let's not forget to call events, those are called inside coro handler for the above case
    if (mesh) {
      mesh->dispatcher.trigger(SHWire::OnStopEvent{wire});
    }
  }

  // return true if we ended, as in we did our job
  auto res = wire->state == SHWire::State::Ended;

  wire->state = SHWire::State::Stopped;
  wire->currentInput.reset();

  // Clone the results if we need them
  if (result)
    cloneVar(*result, wire->finishedOutput);

  return res;
}

inline SHContext *getRootContext(SHContext *current) {
  while (current->parent) {
    current = current->parent;
  }
  return current;
}

template <typename DELEGATE> auto callOnMeshThread(SHContext *context, DELEGATE &func) -> decltype(func.action(), void()) {
#if SH_EMSCRIPTEN
  // Context always runs on the mesh thread already
  func.action();
#else
  if (context) {
    if (unlikely(context->onWorkerThread)) {
      throw ActivationError("Trying to callOnMeshThread from a worker thread!");
    }

    // shassert(!context->onLastResume && "Trying to callOnMeshThread from a wire that is about to stop!");
    shassert(context->continuation && "Context has no continuation!");
    shassert(context->currentWire() && "Context has no current wire!");
    shassert(!context->meshThreadTask && "Context already has a mesh thread task!");

    // ok this is the key, we want to go back to the root context and execute there to ensure we are calling from mesh thread
    // indeed and not from any nested coroutine (Step etc)
    auto rootContext = getRootContext(context);

    rootContext->meshThreadTask.connect<&DELEGATE::action>(func);

    // after suspend context might be invalid!
    auto currentWire = context->currentWire();
    SH_CORO_SUSPENDED(currentWire);
    coroutineResume(*rootContext->continuation); // on root context!
    SH_CORO_RESUMED(currentWire);

    shassert(context->currentWire() == currentWire && "Context changed wire during callOnMeshThread!");
    shassert(!rootContext->meshThreadTask && "Context still has a mesh thread task!");
  } else {
    SHLOG_WARNING("NO Context, not running on mesh thread");
    func.action();
  }
#endif
}

template <typename L, typename V = std::enable_if_t<std::is_invocable_v<L>>> void callOnMeshThread(SHContext *context, L &&func) {
  struct Action {
    L &lambda;
    std::exception_ptr exp;
    void action() {
      try {
        lambda();
      } catch (...) {
        exp = std::current_exception();
      }
    }
  } l{func};
  callOnMeshThread(context, l);
  if (l.exp) {
    std::rethrow_exception(l.exp);
  }
}

#ifdef __EMSCRIPTEN__
template <typename T> inline T emscripten_wait(SHContext *context, emscripten::val promise) {
  const static emscripten::val futs = emscripten::val::global("ShardsBonder");
  emscripten::val fut = futs.new_(promise);
  fut.call<void>("run");

  while (!fut["finished"].as<bool>()) {
    suspend(context, 0.0);
  }

  if (fut["hadErrors"].as<bool>()) {
    throw ActivationError("A javascript async task has failed, check the "
                          "console for more informations.");
  }

  return fut["result"].as<T>();
}
#endif

#if SH_IOS
// such as: auto &container = entt::locator<UIViewControllerContainer>::value();
// and: entt::locator<UIViewControllerContainer>::emplace(viewControllerContainer);
// and entt::locator<UIViewControllerContainer>::has_value()
struct UIViewControllerContainer {
  void *viewController{nullptr};
};
#endif
} // namespace shards
#endif // SH_CORE_RUNTIME
