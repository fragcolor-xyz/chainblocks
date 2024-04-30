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

#include "shards_macros.hpp"
#include "foundation.hpp"
#include "inline.hpp"
#include "utils.hpp"
#include "object_type.hpp"

#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>

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

#ifdef __EMSCRIPTEN__
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

struct SHFlow {
  struct SHWire *wire;
  SHBool paused;
};

struct SHStateSnapshot {
  SHWireState state;
  SHVar flowStorage;
  std::string errorMessage;
};

struct SHContext {
  SHContext(shards::Coroutine *coro, const SHWire *starter, SHFlow *flow) : main(starter), flow(flow), continuation(coro) {
    wireStack.push_back(const_cast<SHWire *>(starter));
  }

  const SHWire *main;
  SHFlow *flow;
  SHContext *parent{nullptr};
  std::vector<SHWire *> wireStack;
  bool onLastResume{false};
  bool onWorkerThread{false};
  uint64_t stepCounter{};

  std::mutex anyStorageLock;
  std::unordered_map<std::string, std::shared_ptr<entt::any>> anyStorage;

  // Used within the coro& stack! (suspend, etc)
  shards::Coroutine *continuation{nullptr};
  SHDuration next{};

  entt::delegate<void()> meshThreadTask;

  SHWire *rootWire() const { return wireStack.front(); }
  SHWire *currentWire() const { return wireStack.back(); }

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
    state = SHWireState::Error;
    errorMessage = message;
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

  constexpr SHVar getFlowStorage() const { return flowStorage; }

private:
  SHWireState state = SHWireState::Continue; // don't make this atomic! it's used a lot!
  // Used when flow is stopped/restart/return
  // to store the previous result
  SHVar flowStorage{};
  std::string errorMessage;
};

namespace shards {
[[nodiscard]] SHComposeResult composeWire(const std::vector<Shard *> &wire, SHValidationCallback callback, void *userData,
                                          SHInstanceData data);
[[nodiscard]] SHComposeResult composeWire(const Shards wire, SHValidationCallback callback, void *userData, SHInstanceData data);
[[nodiscard]] SHComposeResult composeWire(const SHSeq wire, SHValidationCallback callback, void *userData, SHInstanceData data);
[[nodiscard]] SHComposeResult composeWire(const SHWire *wire, SHValidationCallback callback, void *userData, SHInstanceData data);

bool validateSetParam(Shard *shard, int index, const SHVar &value, SHValidationCallback callback, void *userData);
bool matchTypes(const SHTypeInfo &inputType, const SHTypeInfo &receiverType, bool isParameter, bool strict,
                bool relaxEmptySeqCheck);
void triggerVarValueChange(SHContext *context, const SHVar *name, bool isGlobal, const SHVar *var);
void triggerVarValueChange(SHWire *wire, const SHVar *name, bool isGlobal, const SHVar *var);

void installSignalHandlers();

bool isDebuggerPresent();

#ifdef SH_COMPRESSED_STRINGS
void decompressStrings();
#endif

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

FLATTEN ALWAYS_INLINE inline SHVar activateShard(Shard *blk, SHContext *context, const SHVar &input) {
  ZoneScoped;
  ZoneName(blk->name(blk), blk->nameLength);

  SHVar output;
  if (!activateShardInline(blk, context, input, output))
    output = blk->activate(blk, context, &input);
  return output;
}

SHRunWireOutput runWire(SHWire *wire, SHContext *context, const SHVar &wireInput);

inline SHRunWireOutput runSubWire(SHWire *wire, SHContext *context, const SHVar &input) {
  // push to wire stack
  context->wireStack.push_back(wire);
  DEFER({ context->wireStack.pop_back(); });
  auto runRes = shards::runWire(wire, context, input);
  return runRes;
}

void run(SHWire *wire, SHFlow *flow, shards::Coroutine *coro);

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

#if SH_DEBUG_THREAD_NAMES
#define SH_CORO_RESUMED(_wire) \
  { shards::pushThreadName(fmt::format("Wire \"{}\"", (_wire)->name)); }
#define SH_CORO_SUSPENDED(_wire) \
  { shards::popThreadName(); }
#define SH_CORO_EXT_RESUME(_wire)                                                 \
  {                                                                               \
    shards::pushThreadName(fmt::format("<resuming wire> \"{}\"", (_wire)->name)); \
    TracyCoroEnter(wire);                                                         \
  }
#define SH_CORO_EXT_SUSPEND(_wire) \
  {                                \
    shards::popThreadName();       \
    TracyCoroExit(_wire);          \
  }
#else
#define SH_CORO_RESUMED(_wire)
#define SH_CORO_SUSPENDED(_)
#define SH_CORO_EXT_RESUME(_) \
  { TracyCoroEnter(wire); }
#define SH_CORO_EXT_SUSPEND(_) \
  { TracyCoroExit(_wire); }
#endif

inline void prepare(SHWire *wire, SHFlow *flow) {
  shassert(!coroutineValid(wire->coro) && "Wire already prepared!");

  auto runner = [wire, flow]() {
#if SH_USE_THREAD_FIBER
    pushThreadName(fmt::format("<suspended wire> \"{}\"", wire->name));
#endif
    run(wire, flow, &wire->coro);
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

inline void sleep(double seconds = -1.0, bool runCallbacks = true) {
  // negative = no sleep, just run callbacks

  // Run callbacks first if needed
  // Take note of how long it took and subtract from sleep time! if some time is
  // left sleep
  if (runCallbacks) {
    SHDuration sleepTime(seconds);
    auto pre = SHClock::now();
    for (auto &shInfo : GetGlobals().RunLoopHooks) {
      if (shInfo.second) {
        shInfo.second();
      }
    }
    auto post = SHClock::now();

    SHDuration shsTime = post - pre;
    SHDuration realSleepTime = sleepTime - shsTime;
    if (seconds != -1.0 && realSleepTime.count() > 0.0) {
      // Sleep actual time minus stuff we did in shs
      seconds = realSleepTime.count();
    } else {
      // Don't yield to kernel at all in this case!
      seconds = -1.0;
    }
  }

  if (seconds >= 0.0) {
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
  }

  if (runCallbacks)
    FrameMarkNamed("Main Shards Yield");
}

struct RuntimeCallbacks {
  // TODO, turn them into filters maybe?
  virtual void registerShard(const char *fullName, SHShardConstructor constructor) = 0;
  virtual void registerObjectType(int32_t vendorId, int32_t typeId, SHObjectInfo info) = 0;
  virtual void registerEnumType(int32_t vendorId, int32_t typeId, SHEnumInfo info) = 0;
};
}; // namespace shards

using VisitedWires = std::unordered_map<SHWire *, SHTypeInfo>;
struct SHMesh : public std::enable_shared_from_this<SHMesh> {
  static constexpr uint32_t TypeId = 'brcM';
  static inline shards::Type MeshType{{SHType::Object, {.object = {.vendorId = shards::CoreCC, .typeId = TypeId}}}};
  static inline shards::ObjectVar<std::shared_ptr<SHMesh>> MeshVar{"Mesh", shards::CoreCC, TypeId};

  static std::shared_ptr<SHMesh> make(std::string_view label = "") { return std::shared_ptr<SHMesh>(new SHMesh(label)); }

  static std::shared_ptr<SHMesh> *makePtr(std::string_view label = "") { return new std::shared_ptr<SHMesh>(new SHMesh(label)); }

  ~SHMesh() { terminate(); }

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
    VisitedWires visitedWires;
    data.visitedWires = &visitedWires;
    auto validation = shards::composeWire(
        wire.get(),
        [](const Shard *errorShard, SHStringWithLen errorTxt, bool nonfatalWarning, void *userData) {
          auto blk = const_cast<Shard *>(errorShard);
          if (!nonfatalWarning) {
            throw shards::ComposeError(std::string(errorTxt.string, errorTxt.len) +
                                       ", input shard: " + std::string(blk->name(blk)));
          } else {
            SHLOG_INFO("Validation warning: {} input shard: {}", errorTxt, blk->name(blk));
          }
        },
        this, data);
    shards::arrayFree(validation.exposedInfo);
    shards::arrayFree(validation.requiredInfo);
    shards::freeDerivedInfo(data.inputType);

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
  void schedule(Observer observer, const std::shared_ptr<SHWire> &wire, SHVar input = shards::Var::Empty, bool compose = true) {
    ZoneScoped;

    SHLOG_TRACE("Scheduling wire {}", wire->name);

    if (wire->warmedUp || scheduled.count(wire) > 0) {
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
      auto validation = shards::composeWire(
          wire.get(),
          [](const Shard *errorShard, SHStringWithLen errorTxt, bool nonfatalWarning, void *userData) {
            auto blk = const_cast<Shard *>(errorShard);
            if (!nonfatalWarning) {
              throw shards::ComposeError(std::string(errorTxt.string, errorTxt.len) +
                                         ", input shard: " + std::string(blk->name(blk)));
            } else {
              SHLOG_INFO("Validation warning: {} input shard: {}", errorTxt, blk->name(blk));
            }
          },
          this, data);
      shards::arrayFree(validation.exposedInfo);
      shards::arrayFree(validation.requiredInfo);
      shards::freeDerivedInfo(data.inputType);

      SHLOG_TRACE("Wire {} composed", wire->name);
    } else {
      SHLOG_TRACE("Wire {} skipped compose", wire->name);
    }

    observer.before_prepare(wire.get());
    // create a flow as well
    auto &flow = _flowPool.emplace_back();
    flow.wire = wire.get();
    shards::prepare(wire.get(), &flow);

    // wire might fail on warmup during prepare
    if (wire->state == SHWire::State::Failed) {
      throw shards::SHException(fmt::format("Wire {} failed during prepare", wire->name));
    }

    observer.before_start(wire.get());
    shards::start(wire.get(), input);

    scheduled.insert(wire);

    SHLOG_TRACE("Wire {} scheduled", wire->name);
  }

  void schedule(const std::shared_ptr<SHWire> &wire, SHVar input = shards::Var::Empty, bool compose = true) {
    EmptyObserver obs;
    schedule(obs, wire, input, compose);
  }

  void wireCleanedUp(SHWire *wire) {
    scheduled.erase(wire->shared_from_this());

    auto it = std::find_if(_flowPool.begin(), _flowPool.end(), [wire](const auto &f) { return f.wire == wire; });
    if (it != _flowPool.end()) { // Remove from flow pool, while keeping the iteration state
      size_t idxToRemove = _flowPool.index_of(it);
      size_t itIdx = _flowPool.index_of(_flowPoolIt);
      _flowPool.erase(it);
      if (idxToRemove <= itIdx) {
        if (itIdx >= _flowPool.size())
          _flowPoolIt = _flowPool.end();
        else
          _flowPoolIt = _flowPool.begin() + itIdx;
      }
    }
  }

  template <class Observer> bool tick(Observer observer) {
    ZoneScoped;

    auto noErrors = true;
    _errors.clear();
    _failedWires.clear();

    if (shards::GetGlobals().SigIntTerm > 0) {
      terminate();
    } else {
      SHDuration now = SHClock::now().time_since_epoch();
      _flowPoolIt = _flowPool.begin();
      while (_flowPoolIt != _flowPool.end()) {
        auto startFlowPoolIt = _flowPoolIt;
        auto &flow = *_flowPoolIt;
        if (flow.paused) {
          ++_flowPoolIt;
          continue; // simply skip
        }

        observer.before_tick(flow.wire);

        shards::tick(flow.wire, now);

        if (unlikely(!shards::isRunning(flow.wire))) {
          if (flow.wire->finishedError.size() > 0) {
            _errors.emplace_back(flow.wire->finishedError);
          }

          if (flow.wire->state == SHWire::State::Failed) {
            _failedWires.emplace_back(flow.wire);
            noErrors = false;
          }

          observer.before_stop(flow.wire);
          if (!shards::stop(flow.wire)) {
            noErrors = false;
          }

          // stop should have done the following:
          SHLOG_TRACE("Wire {} ended while ticking", flow.wire->name);
          shassert(scheduled.count(flow.wire->shared_from_this()) == 0 && "Wire still in scheduled!");
          shassert(flow.wire->mesh.expired() && "Wire still has a mesh!");
        }

        // Wire removal can change the iterator, in that case don't increment
        if (_flowPoolIt == startFlowPoolIt) {
          ++_flowPoolIt;
        }
      }
    }

    return noErrors;
  }

  bool tick() {
    EmptyObserver obs;
    return tick(obs);
  }

  friend struct SHWire;
  void clear() {
    // clear all wires!
    boost::container::small_vector<std::shared_ptr<SHWire>, 16> toStop;

    // scheduled might not be the full picture!
    for (auto flow : _flowPool) {
      toStop.emplace_back(flow.wire->shared_from_this());
    }
    _flowPool.clear();

    // now add scheduled, notice me might have duplicates!
    for (auto wire : scheduled) {
      toStop.emplace_back(wire);
    }

    // remove dupes
    std::sort(toStop.begin(), toStop.end());
    toStop.erase(std::unique(toStop.begin(), toStop.end()), toStop.end());

    for (auto wire : toStop) {
      shards::stop(wire.get());
      // stop should have done the following:
      shassert(scheduled.count(wire) == 0 && "Wire still in scheduled!");
      shassert(wire->mesh.expired() && "Wire still has a mesh!");
    }

    // release all wires
    scheduled.clear();

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
    // stop should have done the following:
    // shassert(visitedWires.count(wire.get()) == 0 && "Wire still in visitedWires!");
    shassert(scheduled.count(wire) == 0 && "Wire still in scheduled!");
    shassert(wire->mesh.expired() && "Wire still has a mesh!");

    // Erase-Remove Idiom
    _flowPool.erase(
        std::remove_if(_flowPool.begin(), _flowPool.end(), [wire](const auto &flow) { return flow.wire == wire.get(); }),
        _flowPool.end());
  }

  bool empty() { return _flowPool.empty(); }

  const std::vector<std::string> &errors() { return _errors; }

  const std::vector<SHWire *> &failedWires() { return _failedWires; }

  std::unordered_set<std::shared_ptr<SHWire>> scheduled;

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

  void releaseAllRefs() {
    for (auto ref : refs) {
      shards::releaseVariable(ref.second);
    }
    refs.clear();
  }

  bool hasRef(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name);
    return refs.count(key) > 0;
  }

  void setLabel(std::string_view label) { this->label = label; }
  std::string_view getLabel() const { return label; }

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

  boost::container::stable_vector<SHFlow> _flowPool;
  decltype(_flowPool)::iterator _flowPoolIt = _flowPool.end();

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
        shards::tick<true>(wire, SHDuration{});
      }
    }

    // delete also the coro ptr
    wire->coro.reset();
  } else {
    // if we had a coro this will run inside it!
    wire->cleanup(true);

    // let's not forget to call events, those are called inside coro handler for the above case
    std::shared_ptr<SHMesh> mesh = wire->mesh.lock();
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
}

template <typename L, typename V = std::enable_if_t<std::is_invocable_v<L>>> void callOnMeshThread(SHContext *context, L &&func) {
  struct Action {
    L &lambda;
    void action() { lambda(); }
  } l{func};
  callOnMeshThread(context, l);
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
} // namespace shards

#endif // SH_CORE_RUNTIME
