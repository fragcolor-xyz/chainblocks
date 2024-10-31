#ifndef SH_CORE_SHARDS_CHANNELS
#define SH_CORE_SHARDS_CHANNELS

#include <shards/core/shared.hpp>
#include <atomic>
#include <oneapi/tbb/concurrent_queue.h>
#include <memory>
#include <mutex>
#include <variant>

namespace shards {
namespace channels {

struct ChannelShared {
  shards::TypeInfo type;
  std::atomic_bool closed;

  virtual void clear() = 0;
};

struct DummyChannel : public ChannelShared {
  virtual ~DummyChannel() = default;
  virtual void clear() override {}
};

struct MPMCChannel : public ChannelShared {
  MPMCChannel() : ChannelShared() {}

  virtual ~MPMCChannel() {}

  virtual void clear() override {
    // move all to recycle
    OwnedVar value{};
    while (_data.try_pop(value)) {
      _recycle.push(std::move(value));
    }
  }

  bool try_unrecycle(OwnedVar &value) { return _recycle.try_pop(value); }
  // must call try_unrecycle before pushing here, otherwise recycle will grow !!
  void push_unsafe(OwnedVar &&value) { _data.push(std::move(value)); }

  void recycle(OwnedVar &&value) { _recycle.push(std::move(value)); }

  void push_clone(const SHVar &value) {
    // in this case try check recycle bin
    OwnedVar valueClone{};
    _recycle.try_pop(valueClone);
    valueClone = value;
    _data.push(std::move(valueClone));
  }

  template <bool Recycle = true, typename Func> bool try_pop(Func &&func) {
    OwnedVar value{};
    if (_data.try_pop(value)) {
      func(std::forward<OwnedVar>(value));
      if constexpr (Recycle) {
        _recycle.push(std::move(value));
      }
      return true;
    }
    return false;
  }

private:
  // A single source to steal data from
  oneapi::tbb::concurrent_queue<OwnedVar> _data;
  oneapi::tbb::concurrent_queue<OwnedVar> _recycle;
};

struct Broadcast;
class BroadcastChannel : public ChannelShared {
public:
  BroadcastChannel() : ChannelShared() {}

  virtual ~BroadcastChannel() {}

  MPMCChannel &subscribe() {
    // we automatically cleanup based on the closed flag of the inner channel
    std::unique_lock<std::mutex> lock(subMutex);
    return subscribers.emplace_back();
  }

  virtual void clear() override {
    std::unique_lock<std::mutex> lock(subMutex);
    for (auto &sub : subscribers) {
      sub.clear();
    }
    subscribers.clear();
  }

protected:
  friend struct Broadcast;
  std::mutex subMutex;
  std::list<MPMCChannel> subscribers;
};

using Channel = std::variant<DummyChannel, MPMCChannel, BroadcastChannel>;
std::shared_ptr<Channel> get(const std::string &name);

// Atomic flag wrapper
struct AtomicFlag {
  std::atomic_bool value{false};
};

using FlagPtr = std::shared_ptr<AtomicFlag>;
FlagPtr getFlag(const std::string &name);

} // namespace channels
} // namespace shards

#endif