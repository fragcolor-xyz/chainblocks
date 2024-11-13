#ifndef C51922C9_A0A1_463B_86AB_B786E4F226F9
#define C51922C9_A0A1_463B_86AB_B786E4F226F9

#include "runtime.hpp"
#include "serialization.hpp"
#include "pmr/shared_temp_allocator.hpp"
#include <tracy/Wrapper.hpp>
#include <tracy/TracyC.h>
#include <type_traits>
#include <concepts>
#include <deque>
#include <unordered_set>
#include <string>
#include <mutex>

namespace shards {

template <typename TData>
concept WireData = requires(TData data) {
  { data.wire } -> std::assignable_from<std::shared_ptr<SHWire>>;
};

template <typename TData>
concept WireDataDeps = requires(TData data) {
  { data.wires.clear() };
  { data.wires.push_back(std::shared_ptr<SHWire>()) };
};

template <typename T> struct WireDoppelgangerPool {
  struct BatchOperation {
    pmr::SharedTempAllocator temp;
    struct Item {
      T *poolItemPtr;
      int newItemIndex;
      Item(T *poolItemPtr, int newItemIndex) : poolItemPtr(poolItemPtr), newItemIndex(newItemIndex) {}
    };
    pmr::vector<Item> items;

    BatchOperation() : items(temp.getAllocator()) {};
    BatchOperation(BatchOperation &&) = default;
    BatchOperation &operator=(BatchOperation &&) = default;
  };

  WireDoppelgangerPool(SHWireRef master) {
    _master = SHWire::sharedFromRef(master);

    // Never call this from setParam or earlier...
    auto vwire = shards::Var(master);
    std::stringstream stream;
    Writer w(stream);
    pmr::SharedTempAllocator tempAlloc;
    Serialization serializer{tempAlloc.getAllocator(), true};
    serializer.serialize(vwire, w); // need to serialize the wire template pointers in this case!
    _wireStr = stream.str();
  }

  WireDoppelgangerPool(WireDoppelgangerPool &&other) noexcept
      : _pool(std::move(other._pool)), _avail(std::move(other._avail)), _wireStr(std::move(other._wireStr)),
        _master(std::move(other._master)) {
    // No need to lock since we're moving and the other object is being destroyed
  }

  // notice users should stop wires themselves, we might want wires to persist
  // after this object lifetime
  void stopAll()
    requires WireData<T>
  {
    std::unique_lock<LockableBase(std::mutex)> _l(_poolMutex);
    for (auto &item : _pool) {
      stop(item->wire.get());
      _avail.emplace(item.get());
    }
  }

  // Starts a batch acquire operation (which NEEDS to be completed with acquireFromBatch for each index in the batch)
  // Large batch operations are faster since synchronization only happens during acquireBatch and acquireFromBatch can run in
  // parallel without locks
  BatchOperation acquireBatch(size_t numWires) {
    ZoneScoped;

    BatchOperation operation;
    std::unique_lock<LockableBase(std::mutex)> lock(_poolMutex);
    operation.items.reserve(numWires);
    for (size_t i = 0; i < numWires; ++i) {
      if (_avail.size() == 0) {
        int newIdx = (int)_pool.size();
        auto &itemPtr = _pool.emplace_back(std::make_shared<T>());
        operation.items.emplace_back(itemPtr.get(), newIdx);
      } else {
        auto res = _avail.extract(_avail.begin()).value();
        operation.items.emplace_back(res, -1);
      }
    }

    return operation;
  }

  // Used to get a composed wire from a batch operation, this can run on a worker thread without any locks
  template <class Composer, typename Anything = void>
  T *acquireFromBatch(BatchOperation &batch, size_t index, Composer &composer, Anything *anything = nullptr)
    requires WireData<T>
  {
    shassert(index < batch.items.size() && "Index out of bounds");
    auto &item = batch.items[index];
    auto &poolItem = *item.poolItemPtr;

    pmr::SharedTempAllocator tempAlloc;
    if (!poolItem.wire) {
      Serialization serializer{tempAlloc.getAllocator(), true};
      std::shared_ptr<SHWire> wire;
      {
        ZoneScopedN("Deserialize");
        std::stringstream stream(_wireStr);
        Reader r(stream);
        SHVar vwire{};
        serializer.deserialize(r, vwire); // need to deserialize the wire template pointers in this case!
        wire = SHWire::sharedFromRef(vwire.payload.wireValue);
        destroyVar(vwire);
      }

      // ensure things are setup correctly
      shassert(wire != nullptr && "Wire is null");
      shassert(wire->mesh.expired() && "Wire already has a mesh!");
      // keep a reference to the master wire
      wire->parent = _master;
      // regenerate the id for the new wire, in order to keep adding new wires at the end of the list of mesh scheduled wires
      wire->regenerateId();
      poolItem.wire = wire;
      poolItem.wire->name = fmt::format("{}-{}", wire->name, item.newItemIndex);
      if constexpr (WireDataDeps<T>) {
        poolItem.wires.clear();
        for (auto &w : serializer.wires) {
          poolItem.wires.push_back(SHWire::sharedFromRef(w.second));
        }
      }

      ZoneScopedN("Compose");
      composer.compose(poolItem.wire.get(), anything, false);
    } else {
      composer.compose(poolItem.wire.get(), anything, true);
    }

    return &poolItem;
  }

  std::pair<T *, bool> acquireFromBatch(BatchOperation &batch, size_t index)
    requires WireData<T>
  {
    shassert(index < batch.items.size() && "Index out of bounds");
    auto &item = batch.items[index];
    auto &poolItem = *item.poolItemPtr;
    bool cached = true;
    pmr::SharedTempAllocator tempAlloc;

    if (!poolItem.wire) {
      cached = false;
      Serialization serializer{tempAlloc.getAllocator(), true};
      std::shared_ptr<SHWire> wire;
      {
        ZoneScopedN("Deserialize");
        std::stringstream stream(_wireStr);
        Reader r(stream);
        SHVar vwire{};
        serializer.deserialize(r, vwire); // need to deserialize the wire template pointers in this case!
        wire = SHWire::sharedFromRef(vwire.payload.wireValue);
        destroyVar(vwire);
      }

      // ensure things are setup correctly
      shassert(wire != nullptr && "Wire is null");
      shassert(wire->mesh.expired() && "Wire already has a mesh!");
      // keep a reference to the master wire
      wire->parent = _master;
      // regenerate the id for the new wire, in order to keep adding new wires at the end of the list of mesh scheduled wires
      wire->regenerateId();
      poolItem.wire = wire;
      poolItem.wire->name = fmt::format("{}-{}", wire->name, item.newItemIndex);
      if constexpr (WireDataDeps<T>) {
        poolItem.wires.clear();
        for (auto &w : serializer.wires) {
          poolItem.wires.push_back(SHWire::sharedFromRef(w.second));
        }
      }
    }

    return {&poolItem, cached};
  }

  template <class Composer, typename Anything = void>
  T *acquire(Composer &composer, Anything *anything = nullptr)
    requires WireData<T>
  {
    ZoneScoped;

    auto batch = acquireBatch(1);
    return acquireFromBatch(batch, 0, composer, anything);
  }

  std::pair<T *, bool> acquire()
    requires WireData<T>
  {
    ZoneScoped;

    auto batch = acquireBatch(1);
    return acquireFromBatch(batch, 0);
  }

  void release(T *wire) {
    shassert(wire != nullptr && "Releasing a null wire?");

    std::unique_lock<LockableBase(std::mutex)> _l(_poolMutex);

    _avail.emplace(wire);
  }

private:
  WireDoppelgangerPool() = delete;

  struct Writer {
    std::stringstream &stream;
    Writer(std::stringstream &stream) : stream(stream) {}
    void operator()(const uint8_t *buf, size_t size) { stream.write((const char *)buf, size); }
  };

  struct Reader {
    std::stringstream &stream;
    Reader(std::stringstream &stream) : stream(stream) {}
    void operator()(uint8_t *buf, size_t size) { stream.read((char *)buf, size); }
  };

  // keep our pool in a deque in order to keep them alive
  // so users don't have to worry about lifetime
  // just release when possible
  TracyLockable(std::mutex, _poolMutex);
  std::deque<std::shared_ptr<T>> _pool;
  std::unordered_set<T *> _avail;
  std::string _wireStr;

  std::shared_ptr<SHWire> _master; // keep master in any case, for debugging purposes
};
} // namespace shards

#endif /* C51922C9_A0A1_463B_86AB_B786E4F226F9 */