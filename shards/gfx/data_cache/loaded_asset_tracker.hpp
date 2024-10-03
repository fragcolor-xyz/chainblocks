#ifndef DEA617E6_4F30_48E5_9EC1_D990C1F8FDE2
#define DEA617E6_4F30_48E5_9EC1_D990C1F8FDE2

#include "../fwd.hpp"
#include "types.hpp"
#include <map>
#include <variant>
#include <shared_mutex>

namespace gfx::data {

using LoadedAssetHandle = std::variant<std::weak_ptr<Texture>, std::weak_ptr<Mesh>, std::weak_ptr<IDrawable>>;
struct LoadedAssetTracker {
private:
  static inline constexpr size_t DefaultIterationBudget = 100;

  using Map = std::map<AssetKey, LoadedAssetHandle>;
  struct IteratorState {
    Map::iterator it;
    Map::iterator end;
  };
  std::optional<IteratorState> iterator;
  Map map;
  std::shared_mutex mutex;

public:
  void gc(size_t iterationBudget = DefaultIterationBudget) {
    std::unique_lock lock(mutex);
    if (iterator) {
      gc(iterator, iterationBudget);
    } else {
      iterator = IteratorState{map.begin(), map.end()};
      gc(iterator, iterationBudget);
    }
  }

  void insert(const AssetKey &key, LoadedAssetHandle handle) {
    std::unique_lock lock(mutex);
    map[key] = handle;
    iterator.reset();
  }

  std::optional<LoadedAssetHandle> find(const AssetKey &key) {
    std::shared_lock lock(mutex);
    auto it = map.find(key);
    if (it == map.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  template <typename T> std::shared_ptr<T> find(const AssetKey &key) {
    auto handle = find(key);
    if (!handle) {
      return nullptr;
    }
    if (auto ptr = std::get_if<std::weak_ptr<T>>(&*handle)) {
      if (auto locked = ptr->lock()) {
        return locked;
      }
    }
    return nullptr;
  }

private:
  template <typename T> bool isExpired(std::weak_ptr<T> &handle) { return handle.expired(); }
  void gc(std::optional<IteratorState> &state, size_t iterationBudget) {
    for (size_t i = 0; i < iterationBudget && state->it != state->end; i++) {
      std::visit(
          [&](auto &&arg) {
            if (isExpired(arg)) {
              state->it = map.erase(state->it);
            } else {
              ++state->it;
            }
          },
          state->it->second);
      if (state->it == state->end) {
        state.reset();
        break;
      }
    }
  }
};

const std::shared_ptr<LoadedAssetTracker> &getStaticLoadedAssetTracker();
} // namespace gfx::data

#endif /* DEA617E6_4F30_48E5_9EC1_D990C1F8FDE2 */
