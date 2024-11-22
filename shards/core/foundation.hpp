/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#ifndef SH_CORE_FOUNDATION
#define SH_CORE_FOUNDATION

// must go first
#include <stdlib.h>
#if _WIN32
#include <winsock2.h>
#endif

#include <shards/shards.h>

#include <shards/shards.hpp>
#include "ops_internal.hpp"

#include "spdlog/spdlog.h"
#include "type_matcher.hpp"
#include "type_info.hpp"
#include "trait.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <random>

#ifdef SHARDS_TRACKING
#include "tracking.hpp"
#endif

#include <shards/shardwrapper.hpp>

// Needed specially for win32/32bit
#include <boost/align/aligned_allocator.hpp>

#include "oneapi/tbb/concurrent_unordered_map.h"

#include "coro.hpp"

#if SH_EMSCRIPTEN
#include <emscripten.h>
#endif

#ifdef NDEBUG
#define SH_COMPRESSED_STRINGS 1
#endif

#ifdef SH_COMPRESSED_STRINGS
#define SHCCSTR(_str_) ::shards::getCompiledCompressedString(::shards::constant<::shards::crc32(_str_)>::value)
#else
#define SHCCSTR(_str_) ::shards::setCompiledCompressedString(::shards::constant<::shards::crc32(_str_)>::value, _str_)
#endif

#define SHLOG_TRACE SPDLOG_TRACE
#define SHLOG_DEBUG SPDLOG_DEBUG
#define SHLOG_INFO SPDLOG_INFO
#define SHLOG_WARNING SPDLOG_WARN
#define SHLOG_ERROR SPDLOG_ERROR
#define SHLOG_FATAL(...)          \
  {                               \
    SPDLOG_CRITICAL(__VA_ARGS__); \
    std::abort();                 \
  }

#include "assert.hpp"

namespace shards {
#ifdef SH_COMPRESSED_STRINGS
SHOptionalString getCompiledCompressedString(uint32_t crc);
#else
SHOptionalString setCompiledCompressedString(uint32_t crc, const char *str);
#endif

SHString getString(uint32_t crc);
void setString(uint32_t crc, SHString str);
void stringGrow(SHStringPayload *str, uint32_t newCap);
void stringFree(SHStringPayload *str);
[[nodiscard]] SHComposeResult composeWire(const Shards wire, SHInstanceData data);
// caller does not handle return
SHWireState activateShards(SHSeq shards, SHContext *context, const SHVar &wireInput, SHVar &output) noexcept;
// caller handles return
SHWireState activateShards2(SHSeq shards, SHContext *context, const SHVar &wireInput, SHVar &output) noexcept;
// caller does not handle return
SHWireState activateShards(Shards shards, SHContext *context, const SHVar &wireInput, SHVar &output) noexcept;
// caller handles return
SHWireState activateShards2(Shards shards, SHContext *context, const SHVar &wireInput, SHVar &output) noexcept;
SHVar *findVariable(SHContext *ctx, std::string_view name);
SHVar *referenceGlobalVariable(SHContext *ctx, std::string_view name);
SHVar *referenceVariable(SHContext *ctx, std::string_view name);
SHVar *referenceWireVariable(SHWire *wire, std::string_view name);
void releaseVariable(SHVar *variable);
void setSharedVariable(std::string_view name, const SHVar &value);
void unsetSharedVariable(std::string_view name);
SHVar getSharedVariable(std::string_view name);
SHWireState suspend(SHContext *context, double seconds, bool sleepOnWorker = false);
entt::id_type findId(SHContext *ctx) noexcept;

Shard *createShard(std::string_view name);
void registerShards();
void registerShard(std::string_view name, SHShardConstructor constructor, std::string_view fullTypeName = std::string_view());
void registerObjectType(int32_t vendorId, int32_t typeId, SHObjectInfo info);
void registerEnumType(int32_t vendorId, int32_t typeId, SHEnumInfo info);
const SHObjectInfo *findObjectInfo(int32_t vendorId, int32_t typeId);
int64_t findObjectTypeId(std::string_view name);
const SHEnumInfo *findEnumInfo(int32_t vendorId, int32_t typeId);
int64_t findEnumId(std::string_view name);
void registerWire(SHWire *wire);
void unregisterWire(SHWire *wire);

void imageIncRef(SHImage *ptr);
void imageDecRef(SHImage *ptr);
SHImage *imageNew(uint32_t dataLen);
SHImage *imageClone(SHImage *ptr);
uint32_t imageDeriveDataLength(SHImage *ptr);
uint32_t imageGetPixelSize(SHImage *img);
uint32_t imageGetRowStride(SHImage *img);

struct RuntimeObserver {
  virtual void registerShard(const char *fullName, SHShardConstructor constructor) {}
  virtual void registerObjectType(int32_t vendorId, int32_t typeId, SHObjectInfo info) {}
  virtual void registerEnumType(int32_t vendorId, int32_t typeId, SHEnumInfo info) {}
};

inline void cloneVar(SHVar &dst, const SHVar &src);
inline void destroyVar(SHVar &src);

struct InternalCore;
using OwnedVar = TOwnedVar<InternalCore>;

void decRef(ShardPtr shard);
void incRef(ShardPtr shard);

template <std::size_t Size = 512> struct bumping_memory_resource {
  char buffer[Size];
  std::size_t _used = 0;
  char *_ptr;

  explicit bumping_memory_resource() : _ptr(&buffer[0]) {}

  void *allocate(std::size_t size) {
    _used += size;
    if (_used > Size)
      throw shards::SHException("Stack allocator memory exhausted");

    auto ret = _ptr;
    _ptr += size;
    return ret;
  }

  void deallocate(void *) noexcept {}
};

template <typename T, typename Resource = bumping_memory_resource<sizeof(T) * 32>> struct stack_allocator {
  Resource _res;
  using value_type = T;

  stack_allocator() {}

  stack_allocator(const stack_allocator &) = default;

  template <typename U> stack_allocator(const stack_allocator<U, Resource> &other) : _res(other._res) {}

  T *allocate(std::size_t n) { return static_cast<T *>(_res.allocate(sizeof(T) * n)); }
  void deallocate(T *ptr, std::size_t) { _res.deallocate(ptr); }

  friend bool operator==(const stack_allocator &lhs, const stack_allocator &rhs) { return lhs._res == rhs._res; }

  friend bool operator!=(const stack_allocator &lhs, const stack_allocator &rhs) { return lhs._res != rhs._res; }
};

struct TypeInfo {
  TypeInfo() {}

  TypeInfo(const SHVar &var, const SHInstanceData &data, std::vector<SHExposedTypeInfo> *expInfo = nullptr,
           bool resolveContextVariables = true) {
    _info = deriveTypeInfo(var, data, expInfo, resolveContextVariables);
  }

  TypeInfo(const SHTypeInfo &info) { _info = cloneTypeInfo(info); }

  TypeInfo(TypeInfo &&info) {
    _info = info._info;
    info._info = {};
  }

  TypeInfo &operator=(const SHTypeInfo &info) {
    freeTypeInfo(_info);
    _info = cloneTypeInfo(info);
    return *this;
  }

  TypeInfo &operator=(TypeInfo &&info) {
    freeTypeInfo(_info);
    _info = info._info;
    info._info = {};
    return *this;
  }

  ~TypeInfo() { freeTypeInfo(_info); }

  operator const SHTypeInfo &() { return _info; }

  const SHTypeInfo *operator->() const { return &_info; }
  const SHTypeInfo &operator*() const { return _info; }
  SHTypeInfo *operator->() { return &_info; }
  SHTypeInfo &operator*() { return _info; }

private:
  SHTypeInfo _info{};
};
} // namespace shards

struct SHSetImpl : public std::unordered_set<shards::OwnedVar, std::hash<SHVar>, std::equal_to<SHVar>,
                                             boost::alignment::aligned_allocator<shards::OwnedVar, 16>> {
#if SHARDS_TRACKING
  SHSetImpl() {}
  ~SHSetImpl() {}
#endif
};

template <typename K, typename V>
struct SHAlignedMap : public boost::container::flat_map<
                          K, V, std::less<K>,
                          boost::container::stable_vector<std::pair<const K, V>,
                                                          boost::alignment::aligned_allocator<std::pair<const K, V>, 16>>> {};

struct SHTableImpl : public SHAlignedMap<shards::OwnedVar, shards::OwnedVar> {
#if SHARDS_TRACKING
  SHTableImpl() {}
  ~SHTableImpl() {}
#endif
};

typedef void(__cdecl *SHSetWireError)(const SHWire *, void *errorData, struct SHStringWithLen msg);

struct SHWire : public std::enable_shared_from_this<SHWire> {
  enum State { Stopped, Prepared, Starting, Iterating, IterationEnded, Failed, Ended };

  // Triggered whenever the main wire of a context starts
  struct OnStartEvent {
    const SHWire *wire;
  };

  // Triggered directly on each wire
  struct OnCleanupEvent {
    const SHWire *wire;
  };

  struct OnStopEvent {
    const SHWire *wire;
  };

  struct OnErrorEvent {
    const SHWire *wire;
    const Shard *shard; // can be nullptr
    shards::OwnedVar error;
  };

  // Triggered whenever a new wire is detached from this wire
  struct OnDetachedEvent {
    const SHWire *wire;
    const SHWire *childWire;
  };

  mutable entt::dispatcher dispatcher;

  // Storage of data used only during compose
  struct ComposeData {
    // List of output types used for this wire
    std::vector<SHTypeInfo> outputTypes;
    // name, isMutable
    std::unordered_map<std::string_view, SHExposedTypeInfo> declaredVariables;
  };
  std::shared_ptr<ComposeData> composeData;

  ComposeData &getComposeData() {
    if (!composeData) {
      composeData = std::make_shared<ComposeData>();
    }
    return *composeData;
  }

  // Attributes
  bool looped{false};
  bool unsafe{false};
  bool pure{false};

  std::string name{"unnamed"};
  entt::id_type id{entt::null};
  uint64_t debugId{0};            // used for debugging
  shards::OwnedVar astObject;     // optional, used for debugging
  std::shared_ptr<SHWire> parent; // used in doppelganger pool, we keep track of the template wire
  int priority{0};                // used in scheduler

  // The wire's running coroutine
  shards::Coroutine coro;

  std::atomic<State> state{Stopped};
  std::atomic_bool composing{false};
  std::atomic_bool stopping{false};

  // no ideal to be owned, as it triggers deep copy, but for now we need to keep it
  // at some point we should find where it needs to be retained and retain it there
  shards::OwnedVar currentInput{};
  SHVar previousOutput{};

  // notice we preserve those even over stop/reset!
  // they are cleared only on a following prepare
  shards::OwnedVar finishedOutput{};
  std::string finishedError{};

  SHVar composedHash{};
  bool warmedUp{false};
  bool isRoot{false};
  bool detached{false};
  std::unordered_set<void *> wireUsers;

  // we need to clone this, as might disappear, since outside wire
  mutable shards::TypeInfo inputType{};
  // flag if we changed inputType to None or Any on purpose
  mutable bool ignoreInputTypeCheck{false};
  // this one is a shard inside the wire, so won't disappear
  mutable SHTypeInfo outputType{};

  // used in wires.cpp to store exposed/required types from compose operations
  mutable std::optional<SHComposeResult> composeResult;
  // used sometimes in wires.cpp and .hpp when capturing variables is needed
  mutable std::unordered_map<std::string_view, SHExposedTypeInfo> requirements;

  SHContext *context{nullptr};

  SHWire *resumer{nullptr};   // used in SwitchTo shard
  SHWire *childWire{nullptr}; // used in SwitchTo shard

  std::weak_ptr<SHMesh> mesh;

  std::vector<Shard *> shards;

  // used only in the case of external variables
  std::unordered_map<uint64_t, shards::TypeInfo> typesCache;

#if SH_CORO_NEED_STACK_MEM
  // this is the eventual coroutine stack memory buffer
  uint8_t *stackMem{nullptr};
  size_t stackSize{SH_BASE_STACK_SIZE};
#endif

  ~SHWire() {
    SHLOG_TRACE("Destroying wire {}", name);
    destroy();
  }

  void warmup(SHContext *context);

  void cleanup(bool force = false);

  // Also the wire takes ownership of the shard!
  void addShard(Shard *blk) {
    shassert(!blk->owned);
    blk->owned = true;
    shards::incRef(blk);
    shards.push_back(blk);
  }

  // Also removes ownership of the shard
  void removeShard(Shard *blk) {
    auto findIt = std::find(shards.begin(), shards.end(), blk);
    if (findIt != shards.end()) {
      shards.erase(findIt);
      blk->owned = false;
      shards::decRef(blk);
    } else {
      throw shards::SHException("removeShard: shard not found!");
    }
  }

  SHWireRef newRef() { return reinterpret_cast<SHWireRef>(new std::shared_ptr<SHWire>(shared_from_this())); }

  static std::shared_ptr<SHWire> &sharedFromRef(SHWireRef ref) {
    shassert(ref && "sharedFromRef - ref was nullptr");
    return *reinterpret_cast<std::shared_ptr<SHWire> *>(ref);
  }

  static void deleteRef(SHWireRef ref) {
    auto pref = reinterpret_cast<std::shared_ptr<SHWire> *>(ref);
    // if your screen is spammed by the under, don't you dare think of removing this line...
    // likely a red flag and not a red herring, stop going into kernel land...
    SHLOG_TRACE("{} wire deleteRef - use_count: {}", (*pref)->name, pref->use_count());
    delete pref;
  }

  // WARNING: This creates a reference to the current shared_ptr in memory
  static SHWireRef weakRef(const std::shared_ptr<SHWire> &shared) {
    return reinterpret_cast<SHWireRef>(&const_cast<std::shared_ptr<SHWire> &>(shared));
  }

  static SHWireRef addRef(SHWireRef ref) {
    auto cref = sharedFromRef(ref);
    // if your screen is spammed by the under, don't you dare think of removing this line...
    // likely a red flag and not a red herring, stop going into kernel land...
    SHLOG_TRACE("{} wire addRef - use_count: {}", cref->name, cref.use_count());
    auto res = new std::shared_ptr<SHWire>(cref);
    return reinterpret_cast<SHWireRef>(res);
  }

  static std::shared_ptr<SHWire> make(std::string_view wire_name) { return std::shared_ptr<SHWire>(new SHWire(wire_name)); }

  static std::shared_ptr<SHWire> *makePtr(std::string_view wire_name) {
    return new std::shared_ptr<SHWire>(new SHWire(wire_name));
  }

  SHVar &getVariable(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name); // copy on write
    return variables[key];
  }

  constexpr auto &getVariables() { return variables; }

  SHVar *getExternalVariable(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name); // copy on write
    return externalVariables[key].var;
  }

  constexpr auto &getExternalVariables() { return externalVariables; }

  std::optional<std::reference_wrapper<SHVar>> getVariableIfExists(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name);
    auto it = variables.find(key);
    if (it != variables.end()) {
      return it->second;
    } else {
      return std::nullopt;
    }
  }

  SHVar *getExternalVariableIfExists(const SHStringWithLen name) {
    auto key = shards::OwnedVar::Foreign(name);
    auto it = externalVariables.find(key);
    if (it != externalVariables.end()) {
      return it->second.var;
    } else {
      return nullptr;
    }
  }

  void addTrait(SHTrait trait);
  const std::vector<shards::Trait> &getTraits() const { return traits; }

  // less operator, compare by priority fall back to wire id
  bool operator<(const SHWire &other) const {
    if (priority != other.priority) {
      return priority < other.priority;
    } else {
      return uniqueId < other.uniqueId;
    }
  }

  bool paused{false};

  SHWire *tickingWire() {
    SHWire *wire = this;
    while (wire->childWire) {
      wire = wire->childWire;
    }
    return wire;
  }

  // regenerate the id, SHMesh uses flat_set which is sorted by id
  // this allows us to regenerate when acquiring from pools in order to keep adding new wires at the end of the list
  uint64_t regenerateId() { return uniqueId = idCounter.fetch_add(1); }

private:
  SHWire(std::string_view wire_name) : name(wire_name) {
    SHLOG_TRACE("Creating wire: {}", name);
    regenerateId();
  }

  std::unordered_map<shards::OwnedVar, SHVar, std::hash<shards::OwnedVar>, std::equal_to<shards::OwnedVar>,
                     boost::alignment::aligned_allocator<std::pair<const shards::OwnedVar, SHVar>, 16>>
      variables;

  // variables with lifetime managed externally
  std::unordered_map<shards::OwnedVar, SHExternalVariable, std::hash<shards::OwnedVar>, std::equal_to<shards::OwnedVar>,
                     boost::alignment::aligned_allocator<std::pair<const shards::OwnedVar, SHExternalVariable>, 16>>
      externalVariables;

  std::vector<shards::Trait> traits;

  void destroy();

  uint64_t uniqueId;
  static inline std::atomic_uint64_t idCounter{0};
};

namespace shards {
// LayeredMap Class Definition
template <typename K, typename V> class LayeredMap {
  std::vector<std::unordered_map<K, V>> layers;
  std::vector<bool> blocker_tags;

public:
  // Nested iterator class
  class iterator {
    using OuterIt = typename std::vector<std::unordered_map<K, V>>::const_iterator;
    using InnerIt = typename std::unordered_map<K, V>::const_iterator;

    OuterIt current_layer;
    OuterIt end_layer;
    InnerIt current_it;

    // Helper to advance to the next valid element
    void advance_to_next() {
      while (current_layer != end_layer) {
        if (current_it == current_layer->end()) {
          ++current_layer;
          if (current_layer != end_layer) {
            current_it = current_layer->begin();
          }
          continue;
        }
        break;
      }
    }

  public:
    // Iterator traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<const K, V>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    // Constructor for begin iterator
    iterator(OuterIt start, OuterIt end) : current_layer(start), end_layer(end) {
      if (current_layer != end_layer) {
        current_it = current_layer->begin();
        advance_to_next();
      }
    }

    // Constructor for specific element (used in find)
    iterator(OuterIt layer, OuterIt end, InnerIt it) : current_layer(layer), end_layer(end), current_it(it) {}

    // Dereference operators
    reference operator*() const { return *current_it; }
    pointer operator->() const { return &(*current_it); }

    // Pre-increment
    iterator &operator++() {
      if (current_layer == end_layer)
        return *this;
      ++current_it;
      advance_to_next();
      return *this;
    }

    // Equality operators
    bool operator==(const iterator &other) const {
      if (current_layer == other.current_layer) {
        if (current_layer == end_layer)
          return true;
        return current_it == other.current_it;
      }
      return false;
    }

    bool operator!=(const iterator &other) const { return !(*this == other); }
  };

  // Push a new layer
  // If blocked is true, the previous layer will be blocked from being accessed
  void pushLayer(bool blocked = false) {
    if (!layers.empty()) {
      // Mark the current layer as blocked
      blocker_tags.back() = blocked;
    }
    // Push a new layer and set its blocked status
    layers.emplace_back();
    blocker_tags.push_back(false);
  }

  // Pop the topmost layer
  void popLayer() {
    if (!layers.empty()) {
      layers.pop_back();
      blocker_tags.pop_back();
      if (!blocker_tags.empty()) {
        blocker_tags.back() = false;
      }
    } else {
      throw std::runtime_error("No layers to pop!");
    }
  }

  // Insert only if the key does not exist in any accessible layer
  void insert(const K &key, const V &value) {
    auto it = find(key);
    if (it == end()) {
      if (!layers.empty()) {
        layers.back()[key] = value;
      } else {
        throw std::runtime_error("No layers to insert into!");
      }
    } else {
      // update existing value
      const_cast<V &>(it->second) = value;
    }
  }

  // Erase the key from the topmost accessible layer where it exists
  void erase(const K &key) {
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
      size_t index = std::distance(it, layers.rend()) - 1;
      if (blocker_tags[index]) {
        // If the current layer is blocked, do not go further down
        break;
      }
      if (it->erase(key)) {
        break;
      }
    }
  }

  // Find the key and return an iterator to it
  iterator find(const K &key) const {
    size_t num_layers = layers.size();
    for (size_t i = num_layers; i > 0; --i) {
      size_t index = i - 1;
      if (blocker_tags[index]) {
        // If a blocker is found and it's not the topmost layer, stop searching
        break;
      }
      const auto &layer = layers[index];
      auto found = layer.find(key);
      if (found != layer.end()) {
        // Calculate the iterator to the found layer
        auto layer_it = layers.begin() + index;
        return iterator(layer_it, layers.end(), found);
      }
    }
    return end();
  }

  // Calculate the size considering blocker_tags
  size_t size() const {
    size_t total = 0;
    for (size_t i = layers.size(); i > 0; --i) {
      size_t index = i - 1;
      if (blocker_tags[index]) {
        break;
      }
      total += layers[index].size();
    }
    return total;
  }

  // Begin iterator
  iterator begin() const {
    // Determine the range of layers to iterate over, respecting blockers
    auto start = layers.end();
    auto end_it = layers.end();

    // Iterate from top to bottom to find the first blocker
    for (size_t i = layers.size(); i > 0; --i) {
      size_t index = i - 1;
      if (blocker_tags[index]) {
        start = layers.begin() + index;
        end_it = layers.end();
        break;
      }
    }

    // If no blocker is found, start from the first layer
    if (start == layers.end()) {
      start = layers.begin();
    }

    return iterator(start, end_it);
  }

  // End iterator
  iterator end() const { return iterator(layers.end(), layers.end()); }
};

using SHHashSet = SHSetImpl;
using SHHashSetIt = SHHashSet::iterator;

using SHMap = SHTableImpl;
using SHMapIt = SHMap::iterator;

struct EventDispatcher {
  entt::dispatcher dispatcher;
  std::string name;
  SHTypeInfo type;

  entt::dispatcher *operator->() { return &dispatcher; }
};

struct CrashHandlerBase {
  virtual void crash() {}
};

struct Globals {
public:
  std::unordered_map<std::string, OwnedVar> Settings;

  CrashHandlerBase *CrashHandler{nullptr};

  int SigIntTerm{0};
  std::unordered_map<std::string_view, SHShardConstructor> ShardsRegister;
  std::unordered_map<std::string_view, std::string_view> ShardNamesToFullTypeNames;
  std::unordered_map<int64_t, SHObjectInfo> ObjectTypesRegister;
  std::unordered_map<std::string_view, int64_t> ObjectTypesRegisterByName;
  std::unordered_map<int64_t, SHEnumInfo> EnumTypesRegister;
  std::unordered_map<std::string_view, int64_t> EnumTypesRegisterByName;

  std::unordered_map<std::string, std::shared_ptr<SHWire>> GlobalWires;

  std::list<std::weak_ptr<RuntimeObserver>> Observers;

  std::string RootPath;
  std::string ExePath;

  oneapi::tbb::concurrent_unordered_map<uint32_t, SHOptionalString> *CompressedStrings{nullptr};

  SHTableInterface TableInterface{
      .tableGetIterator =
          [](SHTable table, SHTableIterator *outIter) {
            if (outIter == nullptr)
              SHLOG_FATAL("tableGetIterator - outIter was nullptr");
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            shards::SHMapIt *mapIt = reinterpret_cast<shards::SHMapIt *>(outIter);
            *mapIt = map->begin();
          },
      .tableNext =
          [](SHTable table, SHTableIterator *inIter, SHVar *outKey, SHVar *outVar) {
            if (inIter == nullptr)
              SHLOG_FATAL("tableGetIterator - inIter was nullptr");
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            shards::SHMapIt *mapIt = reinterpret_cast<shards::SHMapIt *>(inIter);
            if ((*mapIt) != map->end()) {
              *outKey = (*(*mapIt)).first;
              *outVar = (*(*mapIt)).second;
              (*mapIt)++;
              return true;
            } else {
              return false;
            }
          },
      .tableSize =
          [](SHTable table) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            return uint64_t(map->size());
          },
      .tableContains =
          [](SHTable table, SHVar key) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            // the following is safe cos count takes a const ref
            auto k = reinterpret_cast<shards::OwnedVar *>(&key);
            return map->count(*k) > 0;
          },
      .tableAt =
          [](SHTable table, SHVar key) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            // the following is safe cos []] takes a const ref
            auto k = reinterpret_cast<shards::OwnedVar *>(&key);
            SHVar &vRef = (*map)[*k];
            return &vRef;
          },
      .tableGet =
          [](SHTable table, SHVar key) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            // the following is safe cos []] takes a const ref
            auto k = reinterpret_cast<shards::OwnedVar *>(&key);
            auto it = (*map).find(*k);
            if (it != (*map).end()) {
              return (SHVar *)&it->second;
            } else {
              return (SHVar *)nullptr;
            }
          },
      .tableRemove =
          [](SHTable table, SHVar key) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            // the following is safe cos erase takes a const ref
            auto k = reinterpret_cast<shards::OwnedVar *>(&key);
            map->erase(*k);
          },
      .tableClear =
          [](SHTable table) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            map->clear();
          },
      .tableFree =
          [](SHTable table) {
            shards::SHMap *map = reinterpret_cast<shards::SHMap *>(table.opaque);
            delete map;
          },
  };

  SHSetInterface SetInterface{
      .setGetIterator =
          [](SHSet shset, SHSetIterator *outIter) {
            if (outIter == nullptr)
              SHLOG_FATAL("setGetIterator - outIter was nullptr");
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            shards::SHHashSetIt *setIt = reinterpret_cast<shards::SHHashSetIt *>(outIter);
            *setIt = set->begin();
          },
      .setNext =
          [](SHSet shset, SHSetIterator *inIter, SHVar *outVar) {
            if (inIter == nullptr)
              SHLOG_FATAL("setGetIterator - inIter was nullptr");
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            shards::SHHashSetIt *setIt = reinterpret_cast<shards::SHHashSetIt *>(inIter);
            if ((*setIt) != set->end()) {
              *outVar = (*(*setIt));
              (*setIt)++;
              return true;
            } else {
              return false;
            }
          },
      .setSize =
          [](SHSet shset) {
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            return uint64_t(set->size());
          },
      .setContains =
          [](SHSet shset, SHVar value) {
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            return set->count(value) > 0;
          },
      .setInclude =
          [](SHSet shset, SHVar value) {
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            auto [_, inserted] = set->emplace(value);
            return inserted;
          },
      .setExclude =
          [](SHSet shset, SHVar value) {
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            return set->erase(value) > 0;
          },
      .setClear =
          [](SHSet shset) {
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            set->clear();
          },
      .setFree =
          [](SHSet shset) {
            shards::SHHashSet *set = reinterpret_cast<shards::SHHashSet *>(shset.opaque);
            delete set;
          },
  };
};

void parseArguments(int argc, const char **argv);
Globals &GetGlobals();
EventDispatcher &getEventDispatcher(const std::string &name);

template <typename T> inline void arrayGrow(T &arr, size_t addlen, size_t min_cap = 4) {
  // safety check to make sure this is not a borrowed foreign array!
  shassert((arr.cap == 0 && arr.elements == nullptr) || (arr.cap > 0 && arr.elements != nullptr));

  size_t min_len = arr.len + addlen;

  // compute the minimum capacity needed
  if (min_len > min_cap)
    min_cap = min_len;

  if (min_cap <= arr.cap)
    return;

  // increase needed capacity to guarantee O(1) amortized
  if (min_cap < 2 * arr.cap)
    min_cap = 2 * arr.cap;

  // realloc would be nice here but clashes with our alignment requirements, in the end this is the fastest way without a custom
  // allocator
  auto newbuf = new (std::align_val_t{16}) uint8_t[sizeof(arr.elements[0]) * min_cap];
  if (arr.elements) {
    memcpy(newbuf, arr.elements, sizeof(arr.elements[0]) * arr.len);
    ::operator delete[](arr.elements, std::align_val_t{16});
  }
  arr.elements = (decltype(arr.elements))newbuf;

  // also memset to 0 new memory in order to make cloneVars valid on new items
  size_t size = sizeof(arr.elements[0]) * (min_cap - arr.len);
  memset(arr.elements + arr.len, 0x0, size);

  if (min_cap > UINT32_MAX) {
    // this is the case for now for many reasons, but should be just fine
    SHLOG_FATAL("Int array overflow, we don't support more then UINT32_MAX.");
  }
  arr.cap = uint32_t(min_cap);
}

template <typename T, typename V> inline void arrayPush(T &arr, const V &val) {
  if ((arr.len + 1) > arr.cap) {
    shards::arrayGrow(arr, 1);
  }
  shassert(arr.elements && "array elements cannot be null");
  arr.elements[arr.len++] = val;
}

template <typename T> inline void arrayResize(T &arr, uint32_t size) {
  if (arr.cap < size) {
    shards::arrayGrow(arr, size - arr.len);
  }
  arr.len = size;
}

template <typename T> inline void arrayShuffle(T &arr) {
  // Check if the array is empty
  if (arr.len == 0)
    return;

  // Random number generator
  static thread_local std::random_device rd;
  static thread_local std::mt19937 gen(rd());

  // Fisher-Yates shuffle
  for (uint32_t i = arr.len - 1; i > 0; i--) {
    std::uniform_int_distribution<uint32_t> dis(0, i);
    uint32_t j = dis(gen);
    std::swap(arr.elements[i], arr.elements[j]);
  }
}

template <typename T, typename V> inline void arrayInsert(T &arr, uint32_t index, const V &val) {
  if ((arr.len + 1) > arr.cap) {
    arrayGrow(arr, 1);
  }
  memmove(&arr.elements[index + 1], &arr.elements[index], sizeof(V) * (arr.len - index));
  arr.len++;
  arr.elements[index] = val;
}

template <typename T, typename V> inline V arrayPop(T &arr) {
  shassert(arr.len > 0);
  arr.len--;
  return arr.elements[arr.len];
}

template <typename T> inline void arrayDelFast(T &arr, uint32_t index) {
  shassert(arr.len > 0);
  arr.len--;
  // this allows eventual destroyVar/cloneVar magic
  // avoiding allocations even in nested seqs
  std::swap(arr.elements[index], arr.elements[arr.len]);
}

template <typename T> inline void arrayDel(T &arr, uint32_t index) {
  shassert(arr.len > 0);
  // this allows eventual destroyVar/cloneVar magic
  // avoiding allocations even in nested seqs
  arr.len--;
  while (index < arr.len) {
    std::swap(arr.elements[index], arr.elements[index + 1]);
    index++;
  }
}

template <typename T> inline void arrayFree(T &arr) {
  if (arr.elements) {
    ::operator delete[](arr.elements, std::align_val_t{16});
  }
  memset(&arr, 0x0, sizeof(T));
}

template <typename T> class PtrIterator {
public:
  typedef T value_type;

  PtrIterator(T *ptr) : ptr(ptr) {}

  PtrIterator &operator+=(std::ptrdiff_t s) {
    ptr += s;
    return *this;
  }
  PtrIterator &operator-=(std::ptrdiff_t s) {
    ptr -= s;
    return *this;
  }

  PtrIterator operator+(std::ptrdiff_t s) {
    PtrIterator it(ptr);
    return it += s;
  }
  PtrIterator operator-(std::ptrdiff_t s) {
    PtrIterator it(ptr);
    return it -= s;
  }

  PtrIterator operator++() {
    ++ptr;
    return *this;
  }
  PtrIterator operator++(int s) {
    ptr += s;
    return *this;
  }

  PtrIterator operator--() {
    --ptr;
    return *this;
  }
  PtrIterator operator--(int s) {
    ptr -= s;
    return *this;
  }

  std::ptrdiff_t operator-(PtrIterator const &other) const { return ptr - other.ptr; }

  T &operator*() const { return *ptr; }

  bool operator==(const PtrIterator &rhs) const { return ptr == rhs.ptr; }
  bool operator!=(const PtrIterator &rhs) const { return ptr != rhs.ptr; }
  bool operator>(const PtrIterator &rhs) const { return ptr > rhs.ptr; }
  bool operator<(const PtrIterator &rhs) const { return ptr < rhs.ptr; }
  bool operator>=(const PtrIterator &rhs) const { return ptr >= rhs.ptr; }
  bool operator<=(const PtrIterator &rhs) const { return ptr <= rhs.ptr; }

private:
  T *ptr;
};

template <typename S, typename T, typename Allocator = std::allocator<T>> class IterableArray {
public:
  typedef S seq_type;
  typedef T value_type;
  typedef size_t size_type;

  typedef PtrIterator<T> iterator;
  typedef PtrIterator<const T> const_iterator;

  IterableArray() : _seq({}), _owned(true) {}

  // Not OWNING!
  IterableArray(const seq_type &seq) : _seq(seq), _owned(false) {}

  // implicit converter
  IterableArray(const SHVar &v) : _seq(v.payload.seqValue), _owned(false) { shassert(v.valueType == SHType::Seq); }

  IterableArray(size_t s) : _seq({}), _owned(true) { arrayResize(_seq, s); }

  IterableArray(size_t s, T v) : _seq({}), _owned(true) {
    arrayResize(_seq, s);
    for (size_t i = 0; i < s; i++) {
      _seq[i] = v;
    }
  }

  IterableArray(iterator first, iterator last) : _seq({}), _owned(true) {
    size_t size = last - first;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = *(first + i);
    }
  }

  IterableArray(const_iterator first, const_iterator last) : _seq({}), _owned(true) {
    size_t size = last - first;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = *(first + i);
    }
  }

  IterableArray(const IterableArray &other) : _seq({}), _owned(true) {
    size_t size = other._seq.len;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = other._seq.elements[i];
    }
  }

  IterableArray &operator=(SHVar &var) {
    shassert(var.valueType == SHType::Seq);
    _seq = var.payload.seqValue;
    _owned = false;
    return *this;
  }

  IterableArray &operator=(IterableArray &other) {
    _seq = other._seq;
    _owned = other._owned;
    return *this;
  }

  IterableArray &operator=(const IterableArray &other) {
    _seq = {};
    _owned = true;
    size_t size = other._seq.len;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = other._seq.elements[i];
    }
    return *this;
  }

  ~IterableArray() {
    if (_owned) {
      arrayFree(_seq);
    }
  }

  bool isOwned() const { return _owned; }

private:
  seq_type _seq;
  bool _owned;

public:
  iterator begin() { return iterator(&_seq.elements[0]); }
  const_iterator begin() const { return const_iterator(&_seq.elements[0]); }
  const_iterator cbegin() const { return const_iterator(&_seq.elements[0]); }
  iterator end() { return iterator(&_seq.elements[0] + size()); }
  const_iterator end() const { return const_iterator(&_seq.elements[0] + size()); }
  const_iterator cend() const { return const_iterator(&_seq.elements[0] + size()); }
  T &operator[](int index) { return _seq.elements[index]; }
  const T &operator[](int index) const { return _seq.elements[index]; }
  T &front() { return _seq.elements[0]; }
  const T &front() const { return _seq.elements[0]; }
  T &back() { return _seq.elements[size() - 1]; }
  const T &back() const { return _seq.elements[size() - 1]; }
  T *data() { return _seq; }
  size_t size() const { return _seq.len; }
  bool empty() const { return _seq.elements == nullptr || size() == 0; }
  void resize(size_t nsize) { arrayResize(_seq, nsize); }
  void push_back(const T &value) { arrayPush(_seq, value); }
  void clear() { arrayResize(_seq, 0); }
  operator seq_type() const { return _seq; }
};

using IterableSeq = IterableArray<SHSeq, SHVar>;
using IterableExposedInfo = IterableArray<SHExposedTypesInfo, SHExposedTypeInfo>;
using IterableTypesInfo = IterableArray<SHTypesInfo, SHTypeInfo>;
} // namespace shards

// needed by some c++ library
namespace std {
template <typename T> class iterator_traits<shards::PtrIterator<T>> {
public:
  using difference_type = ptrdiff_t;
  using size_type = size_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = random_access_iterator_tag;
};

inline void swap(SHVar &v1, SHVar &v2) {
  auto t = v1;
  v1 = v2;
  v2 = t;
}
} // namespace std

namespace shards {
NO_INLINE void _destroyVarSlow(SHVar &var);
NO_INLINE void _cloneVarSlow(SHVar &dst, const SHVar &src);

ALWAYS_INLINE inline void destroyVar(SHVar &var) {
  // if var.flags contains SHVAR_FLAGS_FOREIGN, then the var should not be destroyed
  if ((var.flags & SHVAR_FLAGS_FOREIGN) == SHVAR_FLAGS_FOREIGN) {
    return;
  }

  if (var.valueType >= SHType::EndOfBlittableTypes) {
    _destroyVarSlow(var);
  }

  // For blittable types, just zero out the payload and set type to None
  memset(&var.payload, 0x0, sizeof(SHVarPayload));
  var.valueType = SHType::None;
}

ALWAYS_INLINE inline void cloneVar(SHVar &dst, const SHVar &src) {
  if (src.valueType < SHType::EndOfBlittableTypes && dst.valueType < SHType::EndOfBlittableTypes) {
    dst.valueType = src.valueType;
    memcpy(&dst.payload, &src.payload, sizeof(SHVarPayload));
  } else if (src.valueType < SHType::EndOfBlittableTypes) {
    destroyVar(dst);
    dst.valueType = src.valueType;
    memcpy(&dst.payload, &src.payload, sizeof(SHVarPayload));
  } else {
    _cloneVarSlow(dst, src);
  }
}

inline void weakObjectVar(SHVar &dst, const SHVar &src) {
  shassert(src.valueType == SHType::Object);
  shassert(src.objectInfo && "ObjectInfo is null");
  shassert(src.objectInfo->weakReference && "Weak clone function is null");

  destroyVar(dst);
  dst.valueType = SHType::Object;
  dst.payload.objectValue = src.payload.objectValue;
  dst.payload.objectVendorId = src.payload.objectVendorId;
  dst.payload.objectTypeId = src.payload.objectTypeId;
  dst.flags |= SHVAR_FLAGS_USES_OBJINFO | SHVAR_FLAGS_WEAK_OBJECT;
  dst.objectInfo = src.objectInfo;
  dst.objectInfo->weakReference(dst.payload.objectValue);
}

struct InternalCore {
  // need to emulate dllshard Core a bit
  static SHTable tableNew() {
    SHTable res;
    res.api = &shards::GetGlobals().TableInterface;
    res.opaque = new shards::SHMap();
    return res;
  }

  static SHSet setNew() {
    SHSet res;
    res.api = &shards::GetGlobals().SetInterface;
    res.opaque = new shards::SHHashSet();
    return res;
  }

  static SHVar *referenceVariable(SHContext *context, SHStringWithLen name) {
    return shards::referenceVariable(context, std::string_view(name.string, name.len));
  }

  static void stringGrow(SHStringPayload *str, uint32_t size) { shards::stringGrow(str, size); }
  static void stringFree(SHStringPayload *str) { shards::stringFree(str); }

  static void releaseVariable(SHVar *variable) { shards::releaseVariable(variable); }

  static void cloneVar(SHVar &dst, const SHVar &src) { shards::cloneVar(dst, src); }

  static void destroyVar(SHVar &var) { shards::destroyVar(var); }

  template <typename T> static void arrayFree(T &arr) { shards::arrayFree<T>(arr); }

  static void seqPush(SHSeq *s, const SHVar *v) { arrayPush(*s, *v); }

  static void seqResize(SHSeq *s, uint32_t size) { arrayResize(*s, size); }

  static void expTypesFree(SHExposedTypesInfo &arr) { arrayFree(arr); }

  static void log(SHStringWithLen msg) {
    std::string_view msgView(msg.string, size_t(msg.len));
    SHLOG_INFO(msgView);
  }

  static void registerEnumType(int32_t vendorId, int32_t enumId, SHEnumInfo info) {
    shards::registerEnumType(vendorId, enumId, info);
  }

  static void registerObjectType(int32_t vendorId, int32_t objectId, SHObjectInfo info) {
    shards::registerObjectType(vendorId, objectId, info);
  }

  static SHComposeResult composeShards(Shards shards, SHInstanceData data) { return shards::composeWire(shards, data); }

  static SHWireState runShards(Shards shards, SHContext *context, const SHVar &input, SHVar &output) {
    return shards::activateShards(shards, context, input, output);
  }

  static SHWireState runShards2(Shards shards, SHContext *context, const SHVar &input, SHVar &output) {
    return shards::activateShards2(shards, context, input, output);
  }

  static SHWireState suspend(SHContext *ctx, double seconds) { return shards::suspend(ctx, seconds); }
};

typedef TParamVar<InternalCore> ParamVar;

template <Parameters &Params, size_t NPARAMS, Type &InputType, Type &OutputType>
struct SimpleShard : public TSimpleShard<InternalCore, Params, NPARAMS, InputType, OutputType> {};

#define DECL_ENUM_INFO_WITH_VENDOR(_ENUM_, _NAME_, _HELP_, _VENDOR_CC_, _CC_)                             \
  struct _NAME_##EnumInfoWrapper {                                                                        \
    static inline const char Name[] = #_NAME_;                                                            \
    static inline const SHOptionalString Help = SHCCSTR(_HELP_);                                          \
  };                                                                                                      \
  using _NAME_##EnumInfo = shards::TEnumInfo<shards::InternalCore, _ENUM_, _NAME_##EnumInfoWrapper::Name, \
                                             _NAME_##EnumInfoWrapper::Help, _VENDOR_CC_, _CC_, false>

#define DECL_ENUM_FLAGS_INFO_WITH_VENDOR(_ENUM_, _NAME_, _HELP_, _VENDOR_CC_, _CC_)                       \
  struct _NAME_##EnumInfoWrapper {                                                                        \
    static inline const char Name[] = #_NAME_;                                                            \
    static inline const SHOptionalString Help = SHCCSTR(_HELP_);                                          \
  };                                                                                                      \
  using _NAME_##EnumInfo = shards::TEnumInfo<shards::InternalCore, _ENUM_, _NAME_##EnumInfoWrapper::Name, \
                                             _NAME_##EnumInfoWrapper::Help, _VENDOR_CC_, _CC_, true>

#define DECL_ENUM_INFO(_ENUM_, _NAME_, _HELP_, _CC_) DECL_ENUM_INFO_WITH_VENDOR(_ENUM_, _NAME_, _HELP_, shards::CoreCC, _CC_)
#define DECL_ENUM_FLAGS_INFO(_ENUM_, _NAME_, _HELP_, _CC_) \
  DECL_ENUM_FLAGS_INFO_WITH_VENDOR(_ENUM_, _NAME_, _HELP_, shards::CoreCC, _CC_)

#define SH_CONCAT1(_a_, _b_) _a_##_b_
#define SH_CONCAT(_a_, _b_) SH_CONCAT1(_a_, _b_)
#define REGISTER_ENUM(_ENUM_INFO_) \
  static shards::EnumRegisterImpl SH_GENSYM(__registeredEnum) = shards::EnumRegisterImpl::registerEnum<_ENUM_INFO_>()

#define ENUM_HELP(_ENUM_, _VALUE_, _STR_)         \
  namespace shards {                              \
  template <> struct TEnumHelp<_ENUM_, _VALUE_> { \
    static inline SHOptionalString help = _STR_;  \
  };                                              \
  }

template <typename E> static E getFlags(SHVar var) {
  E flags{};
  switch (var.valueType) {
  case SHType::Enum:
    flags = E(var.payload.enumValue);
    break;
  case SHType::Seq: {
    shassert(var.payload.seqValue.len == 0 || var.payload.seqValue.elements[0].valueType == SHType::Enum);
    for (uint32_t i = 0; i < var.payload.seqValue.len; i++) {
      // note: can't use namespace magic_enum::bitwise_operators because of
      // potential conflicts with other implementations
      flags = static_cast<E>(static_cast<magic_enum::underlying_type_t<E>>(flags) |
                             static_cast<magic_enum::underlying_type_t<E>>(var.payload.seqValue.elements[i].payload.enumValue));
    }
  } break;
  default:
    break;
  }
  return flags;
};

typedef TShardsVar<InternalCore> ShardsVar;

typedef TTableVar<InternalCore> TableVar;
typedef TSeqVar<InternalCore> SeqVar;

inline SeqVar &makeSeq(SHVar &var) { return makeSeq<InternalCore>(var); }
inline TableVar &makeTable(SHVar &var) { return makeTable<InternalCore>(var); }

inline TableVar &asTable(SHVar &var) {
  shassert(var.valueType == SHType::Table);
  return *reinterpret_cast<TableVar *>(&var);
}

inline const TableVar &asTable(const SHVar &var) {
  shassert(var.valueType == SHType::Table);
  return *reinterpret_cast<const TableVar *>(&var);
}

inline SeqVar &asSeq(SHVar &var) {
  shassert(var.valueType == SHType::Seq);
  return *reinterpret_cast<SeqVar *>(&var);
}

inline const SeqVar &asSeq(const SHVar &var) {
  shassert(var.valueType == SHType::Seq);
  return *reinterpret_cast<const SeqVar *>(&var);
}

inline OwnedVar makeImage(uint32_t dataLen = 0) {
  SHVar var{};
  var.valueType = SHType::Image;
  var.payload.imageValue = imageNew(dataLen);
  return std::move(reinterpret_cast<OwnedVar &>(var));
}

struct ParamsInfo {
  /*
   DO NOT USE ANYMORE, THIS IS DEPRECATED AND MESSY
   IT WAS USEFUL WHEN DYNAMIC ARRAYS WERE STB ARRAYS
   BUT NOW WE CAN JUST USE DESIGNATED/AGGREGATE INITIALIZERS
 */
  ParamsInfo(const ParamsInfo &other) {
    shards::arrayResize(_innerInfo, 0);
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      shards::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }
  }

  ParamsInfo &operator=(const ParamsInfo &other) {
    _innerInfo = {};
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      shards::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }
    return *this;
  }

  template <typename... Types> explicit ParamsInfo(SHParameterInfo first, Types... types) {
    std::vector<SHParameterInfo> vec = {first, types...};
    _innerInfo = {};
    for (auto pi : vec) {
      shards::arrayPush(_innerInfo, pi);
    }
  }

  template <typename... Types> explicit ParamsInfo(const ParamsInfo &other, SHParameterInfo first, Types... types) {
    _innerInfo = {};

    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      shards::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }

    std::vector<SHParameterInfo> vec = {first, types...};
    for (auto pi : vec) {
      shards::arrayPush(_innerInfo, pi);
    }
  }

  static SHParameterInfo Param(SHString name, SHOptionalString help, SHTypesInfo types, bool variableSetter = false) {
    SHParameterInfo res = {name, help, types, variableSetter};
    return res;
  }

  ~ParamsInfo() { shards::arrayFree(_innerInfo); }

  explicit operator SHParametersInfo() const { return _innerInfo; }

  SHParametersInfo _innerInfo{};
};

struct ExposedInfo {
  /*
   DO NOT USE ANYMORE, THIS IS DEPRECATED AND MESSY
   IT WAS USEFUL WHEN DYNAMIC ARRAYS WERE STB ARRAYS
   BUT NOW WE CAN JUST USE DESIGNATED/AGGREGATE INITIALIZERS
 */
  ExposedInfo() {}

  ExposedInfo(const ExposedInfo &other) {
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      push_back(other._innerInfo.elements[i]);
    }
  }

  ExposedInfo &operator=(const ExposedInfo &other) {
    shards::arrayResize(_innerInfo, 0);
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      push_back(other._innerInfo.elements[i]);
    }
    return *this;
  }

  template <typename... Types> explicit ExposedInfo(const ExposedInfo &other, Types... types) {
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      push_back(other._innerInfo.elements[i]);
    }

    std::vector<SHExposedTypeInfo> vec = {types...};
    for (auto pi : vec) {
      push_back(pi);
    }
  }

  template <typename... Types> explicit ExposedInfo(const SHExposedTypesInfo other, Types... types) {
    for (uint32_t i = 0; i < other.len; i++) {
      push_back(other.elements[i]);
    }

    std::vector<SHExposedTypeInfo> vec = {types...};
    for (auto pi : vec) {
      push_back(pi);
    }
  }

  template <typename... Types> explicit ExposedInfo(const SHExposedTypeInfo &first, Types... types) {
    std::vector<SHExposedTypeInfo> vec = {first, types...};
    for (auto pi : vec) {
      push_back(pi);
    }
  }

  constexpr static SHExposedTypeInfo Variable(SHString name, SHOptionalString help, SHTypeInfo type, bool isMutable = false) {
    SHExposedTypeInfo res = {name, help, type, isMutable};
    return res;
  }

  constexpr static SHExposedTypeInfo ProtectedVariable(SHString name, SHOptionalString help, SHTypeInfo type,
                                                       bool isMutable = false) {
    SHExposedTypeInfo res = {name, help, type, isMutable, true};
    return res;
  }

  constexpr static SHExposedTypeInfo GlobalVariable(SHString name, SHOptionalString help, SHTypeInfo type,
                                                    bool isMutable = false) {
    SHExposedTypeInfo res = {name, help, type, isMutable, false, true};
    return res;
  }

  ~ExposedInfo() { shards::arrayFree(_innerInfo); }

  void push_back(const SHExposedTypeInfo &info) { shards::arrayPush(_innerInfo, info); }

  void clear() { shards::arrayResize(_innerInfo, 0); }

  explicit operator SHExposedTypesInfo() const { return _innerInfo; }

  SHExposedTypesInfo _innerInfo{};
};

using ShardsCollection = std::variant<const SHWire *, ShardPtr, Shards, SHVar>;

struct ShardInfo {
  ShardInfo(const std::string_view &name, const Shard *shard, const SHWire *wire) : name(name), shard(shard), wire(wire) {}
  std::string_view name;
  const Shard *shard;
  const SHWire *wire;
};

struct WireNode {
  WireNode(const SHWire *wire, const SHWire *previous) : wire(wire), previous(previous) {}

  const SHWire *wire;
  const SHWire *previous;

  std::vector<SHVar> eventsSent;
  std::vector<SHVar> eventsReceived;

  std::vector<SHVar> channelsProduced;
  std::vector<SHVar> channelsConsumed;

  std::vector<SHVar> channelsBroadcasted;
  std::vector<SHVar> channelsListened;
};

void gatherShards(const ShardsCollection &coll, std::vector<ShardInfo> &out);

void gatherWires(const ShardsCollection &coll, std::vector<WireNode> &out);

inline void printWireGraph(const SHWire *wire) {
  std::vector<shards::WireNode> nodes;
  shards::gatherWires(wire, nodes);
  for (auto &node : nodes) {
    SHLOG_DEBUG("Wire: {}, parent: {}", node.wire->name, node.previous ? node.previous->name : "none");
    // eventsSent
    if (!node.eventsSent.empty()) {
      SHLOG_DEBUG("  Events Sent: ");
      for (auto &event : node.eventsSent) {
        SHLOG_DEBUG("    {}", event);
      }
    }
    // eventsReceived
    if (!node.eventsReceived.empty()) {
      SHLOG_DEBUG("  Events Received: ");
      for (auto &event : node.eventsReceived) {
        SHLOG_DEBUG("    {}", event);
      }
    }
    // Channels Produced
    if (!node.channelsProduced.empty()) {
      SHLOG_DEBUG("  Channels Produced: ");
      for (auto &channel : node.channelsProduced) {
        SHLOG_DEBUG("    {}", channel);
      }
    }
    // Channels Consumed
    if (!node.channelsConsumed.empty()) {
      SHLOG_DEBUG("  Channels Consumed: ");
      for (auto &channel : node.channelsConsumed) {
        SHLOG_DEBUG("    {}", channel);
      }
    }
    // Channels Broadcasted
    if (!node.channelsBroadcasted.empty()) {
      SHLOG_DEBUG("  Channels Broadcasted: ");
      for (auto &channel : node.channelsBroadcasted) {
        SHLOG_DEBUG("    {}", channel);
      }
    }
    // Channels Listened
    if (!node.channelsListened.empty()) {
      SHLOG_DEBUG("  Channels Listened: ");
      for (auto &channel : node.channelsListened) {
        SHLOG_DEBUG("    {}", channel);
      }
    }
  }
}

struct VariableResolver {
  // this an utility to resolve nested variables, like we do in Const, Match etc

  std::vector<SHVar *> _vals;
  std::vector<SHVar *> _refs;

  void warmup(const SHVar &base, SHVar &slot, SHContext *context) {
    if (base.valueType == SHType::ContextVar) {
      auto sv = SHSTRVIEW(base);
      _refs.emplace_back(referenceVariable(context, sv));
      _vals.emplace_back(&slot);
      // also destroy the previous value to avoid leaks
      destroyVar(slot);
    } else if (base.valueType == SHType::Seq) {
      for (uint32_t i = 0; i < base.payload.seqValue.len; i++) {
        warmup(base.payload.seqValue.elements[i], slot.payload.seqValue.elements[i], context);
      }
    } else if (base.valueType == SHType::Table) {
      ForEach(base.payload.tableValue, [&](auto key, auto &val) {
        auto vptr = slot.payload.tableValue.api->tableAt(slot.payload.tableValue, key);
        warmup(val, *vptr, context);
      });
    }
  }

  void cleanup(SHContext *context) {
    if (_refs.size() > 0) {
      for (auto val : _vals) {
        // we do this to avoid double freeing, we don't really own this value
        *val = shards::Var::Empty;
      }
      for (auto ref : _refs) {
        releaseVariable(ref);
      }
      _refs.clear();
      _vals.clear();
    }
  }

  void reassign() {
    size_t len = _refs.size();
    for (size_t i = 0; i < len; i++) {
      *_vals[i] = *_refs[i];
    }
  }
};

inline bool isObjectType(const SHVar &var, const SHTypeInfo &type) {
  shassert(type.basicType == SHType::Object);
  if (var.valueType != SHType::Object)
    return false;
  if (var.payload.objectVendorId != type.object.vendorId || var.payload.objectTypeId != type.object.typeId)
    return false;
  return true;
}

template <typename T> T &varAsObjectChecked(const SHVar &var, const shards::Type &type) {
  SHTypeInfo typeInfo(type);
  if (var.valueType != SHType::Object) {
    throw std::runtime_error(fmt::format("Invalid type, expected: {} got: {}", type, magic_enum::enum_name(var.valueType)));
  }
  if (var.payload.objectVendorId != typeInfo.object.vendorId) {
    throw std::runtime_error(fmt::format("Invalid object vendor id, expected: {} got: {}", type,
                                         Type::Object(var.payload.objectVendorId, var.payload.objectTypeId)));
  }
  if (var.payload.objectTypeId != typeInfo.object.typeId) {
    throw std::runtime_error(fmt::format("Invalid object type id, expected: {} got: {}", type,
                                         Type::Object(var.payload.objectVendorId, var.payload.objectTypeId)));
  }
  return *reinterpret_cast<T *>(var.payload.objectValue);
}

inline const SHExposedTypeInfo *findExposedVariablePtr(const SHExposedTypesInfo &exposed, std::string_view variableName) {
  for (const auto &entry : exposed) {
    if (variableName == entry.name) {
      return &entry;
    }
  }
  return nullptr;
}

inline const SHExposedTypeInfo *findExposedVariablePtr(const shards::LayeredMap<std::string_view, SHExposedTypeInfo> &exposed,
                                                       std::string_view variableName) {
  auto it = exposed.find(variableName);
  if (it != exposed.end()) {
    return &it->second;
  }
  return nullptr;
}

inline std::optional<SHExposedTypeInfo> findExposedVariable(const SHExposedTypesInfo &exposed, std::string_view variableName) {
  auto ptr = findExposedVariablePtr(exposed, variableName);
  if (ptr) {
    return *ptr;
  }
  return std::nullopt;
}

inline std::optional<SHExposedTypeInfo>
findExposedVariable(const shards::LayeredMap<std::string_view, SHExposedTypeInfo> &exposed, std::string_view variableName) {
  auto it = exposed.find(variableName);
  if (it != exposed.end()) {
    return it->second;
  }
  return std::nullopt;
}

inline std::optional<SHExposedTypeInfo> findExposedVariable(const SHExposedTypesInfo &exposed, const SHVar &var) {
  shassert(var.valueType == SHType::ContextVar);
  return findExposedVariable(exposed, SHSTRVIEW(var));
}

// Collects all ContextVar references
void collectRequiredVariables(const SHInstanceData &data, ExposedInfo &out, const SHVar &var);

inline bool collectRequiredVariables(const SHInstanceData &data, ExposedInfo &out, const SHVar &var, SHTypesInfo validTypes,
                                     const char *debugTag) {
  std::vector<SHExposedTypeInfo> expInfo;
  TypeInfo ti(var, data, &expInfo, false);
  for (auto &type : validTypes) {
    if (TypeMatcher{
            .isParameter = true, .relaxEmptyTableCheck = true, .relaxEmptySeqCheck = expInfo.empty(), .checkVarTypes = true}
            .match(ti, type)) {
      for (auto &it : expInfo) {
        out.push_back(it);
      }
      return true;
    }
  }
  auto msg = fmt::format("No matching variable found for parameter {}, was: {}, expected any of {}", debugTag, (SHTypeInfo &)ti,
                         validTypes);
  SHLOG_ERROR("{}", msg);
  throw ComposeError(msg);
}

template <typename... TArgs>
inline void collectAllRequiredVariables(const SHInstanceData &data, ExposedInfo &out, TArgs &&...args) {
  (collectRequiredVariables(data, out, std::forward<TArgs>(args)), ...);
}

inline SHStringWithLen swlDuplicate(SHStringWithLen in) {
  if (in.len == 0) {
    return SHStringWithLen{};
  }
  SHStringWithLen cpy{
      .string = new char[in.len],
      .len = in.len,
  };
  memcpy(const_cast<char *>(cpy.string), in.string, in.len);
  return cpy;
}
inline SHStringWithLen swlFromStringView(std::string_view in) { return swlDuplicate(toSWL(in)); }
inline void swlFree(SHStringWithLen &in) {
  if (in.len > 0) {
    delete[] in.string;
    in.string = nullptr;
    in.len = 0;
  }
}

}; // namespace shards

#endif // SH_CORE_FOUNDATION
