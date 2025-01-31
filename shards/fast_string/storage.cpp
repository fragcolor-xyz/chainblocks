#include "storage.hpp"
#include <shared_mutex>
#include <mutex>
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/stable_vector.hpp>
#include <boost/container/allocator.hpp>
#include <boost/container/scoped_allocator.hpp>
#include <tracy/Wrapper.hpp>

namespace shards::fast_string {

static constexpr size_t Megabyte = 1 << 20;
static constexpr size_t InitialPoolSize = Megabyte * 8;

using Alloc = boost::container::pmr::monotonic_buffer_resource;
struct CountingAllocator : public boost::container::pmr::memory_resource {
  boost::container::pmr::monotonic_buffer_resource *baseAllocatorPtr{};
  size_t totalRequestedBytes{};

  CountingAllocator(boost::container::pmr::monotonic_buffer_resource *base) : baseAllocatorPtr(base) {}
  __attribute__((always_inline)) void *do_allocate(size_t _Bytes, size_t _Align) override {
    totalRequestedBytes += _Bytes;
    return baseAllocatorPtr->allocate(_Bytes, _Align);
  }
  __attribute__((always_inline)) void do_deallocate(void *_Ptr, size_t _Bytes, size_t _Align) override {
    // Using monotonic_buffer_resource, so safe to no-op
  }
  __attribute__((always_inline)) bool do_is_equal(const memory_resource &_That) const noexcept override { return &_That == this; }
};

struct KeyLess {
  using is_transparent = std::true_type;
  template <typename T, typename U> bool operator()(const T &a, const U &b) const {
    return std::string_view(a) < std::string_view(b);
  }
};

struct Pool {
  using Pair = std::pair<std::string, uint64_t>;

  using MapAllocator = boost::container::pmr::polymorphic_allocator<Pair>;
  using SvAllocator = boost::container::pmr::polymorphic_allocator<std::string_view>;

  Alloc allocator;
  CountingAllocator countingAllocator;
  boost::container::flat_map<std::string, uint64_t, KeyLess, boost::container::stable_vector<Pair, MapAllocator>> map;
  boost::container::vector<std::string_view, SvAllocator> reverse;
  std::shared_mutex lock;

  Pool()
      : allocator(InitialPoolSize, boost::container::pmr::new_delete_resource()), countingAllocator(&allocator),
        map(&countingAllocator), reverse(&countingAllocator) {}
};

struct Storage {
  Pool pool;

  template <typename F> auto write(F callback) {
    std::unique_lock<std::shared_mutex> lock(pool.lock);
    return callback(lock);
  }

  template <typename F> auto read(F callback) {
    std::shared_lock<std::shared_mutex> lock(pool.lock);
    return callback(lock);
  }

  uint64_t store(std::string_view str) {
    return read([&](auto &readLock) -> uint64_t {
      auto it = pool.map.find(str);
      if (it != pool.map.end()) {
        return it->second;
      } else {
        readLock.unlock();
        return write([&](auto &writeLock) -> uint64_t {
          auto it = pool.map.find(str);
          if (it != pool.map.end()) {
            // Double check to make sure it wasn't added in the meantime
            return it->second;
          } else {
            size_t index = pool.reverse.size();
            auto it = pool.map.emplace(std::string(str), index).first;
            pool.reverse.emplace_back(it->first);

            TracyPlot("FastString memory", int64_t(pool.countingAllocator.totalRequestedBytes));
            return (uint64_t)index;
          }
        });
      }
    });
  }

  std::string_view load(uint64_t id) {
    return read([&](auto &readLock) {
      if (id >= pool.reverse.size()) {
        SHARDS_THROW(std::logic_error("Invalid fast string"));
      }
      return pool.reverse[id];
    });
  }
};

Storage *storage{};
void init() {
  static Storage instance;
  storage = &instance;
}
uint64_t store(std::string_view sv) {
  ZoneScopedN("fast_string::store");
  if (!storage)
    init();
  return storage->store(sv);
}
std::string_view load(uint64_t id) {
  ZoneScopedN("fast_string::load");
  return storage->load(id);
}
} // namespace shards::fast_string