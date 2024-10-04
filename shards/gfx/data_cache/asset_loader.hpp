#ifndef A51F655D_FBA0_4D5A_B7EA_A7F8C1081154
#define A51F655D_FBA0_4D5A_B7EA_A7F8C1081154

#include "../texture.hpp"
#include "../mesh.hpp"
#include "data_cache.hpp"
#include "types.hpp"
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <variant>

namespace gfx::data {

struct AssetLoaderEntry {
  // LoadedAssetData loadedData;
  std::shared_ptr<AssetLoadRequest> request;
  // AssetCategory category;
  uint8_t lockIndex;

  AssetLoaderEntry(uint8_t lockIndex) : lockIndex(lockIndex) {}
};

// Keeps alive asset requests, and the results for a while in a caching kind of way
struct AssetLoader {
  static inline constexpr size_t LockCount = 0x10;
  static inline constexpr size_t LockMask = (LockCount - 1);

private:
  using Map = oneapi::tbb::concurrent_unordered_map<AssetKey, std::shared_ptr<AssetLoaderEntry>>;
  Map map;
  std::shared_ptr<DataCache> cache;
  std::array<std::shared_mutex, LockCount> locks;
  std::atomic_uint8_t lockCounter;

public:
  AssetLoader(std::shared_ptr<DataCache> cache) : cache(cache) {}

  void gc() {
    // map.begin();
    // for (auto &[key, entry] : map) {
    // }
  }

  void clear() {
    map.clear();
  }

  std::shared_ptr<AssetLoaderEntry> getOrInsert(const AssetKey &key) {
    uint8_t lockIndex = (lockCounter.fetch_add(1) & LockMask);
    std::unique_lock lock(locks[lockIndex]);

    auto insertOp = map.insert(std::make_pair(key, std::make_shared<AssetLoaderEntry>(lockIndex)));
    if (!insertOp.second) {
      // Failed to insert, already in list
      return insertOp.first->second;
    }
    auto entry = insertOp.first->second;
    entry->request = cache->load(key);
    return entry;
  }

  std::shared_lock<std::shared_mutex> lockEntryShared(const AssetLoaderEntry &entry) {
    return std::shared_lock(locks[entry.lockIndex]);
  }
  std::unique_lock<std::shared_mutex> lockEntryExclusive(const AssetLoaderEntry &entry) {
    return std::unique_lock(locks[entry.lockIndex]);
  }
};
} // namespace gfx::data

#endif /* A51F655D_FBA0_4D5A_B7EA_A7F8C1081154 */
