#ifndef GFX_CONTEXT_DATA
#define GFX_CONTEXT_DATA

#include "../core/assert.hpp"
#include "unique_id.hpp"
#include <memory>
#include <mutex>
#include <concepts>
#include <boost/container/flat_map.hpp>

namespace gfx {
struct Context;

#define SH_GFX_CONTEXT_DATA_LABELS 1
#define SH_GFX_CONTEXT_DATA_LOG_LIFETIME 1
struct ContextData {
  size_t version = ~0;
  size_t lastTouched = ~0;

#if !SH_GFX_CONTEXT_DATA_LABELS
  std::string_view getLabel() const { return "<unknown>"; }
#endif
};

struct ContextDataRequest {
  size_t version = ~0;
  size_t lastTouched = ~0;
};

template <typename T>
concept TContextData = std::is_base_of_v<ContextData, T>;

template <typename T>
concept TWithContextDataKeepAlive = requires(T t, typename T::ContextDataType &cd) {
  { cd.keepAliveRef } -> std::assignable_from<std::shared_ptr<T>>;
  { t.keepAlive() } -> std::convertible_to<bool>;
};

template <typename T>
concept TAsyncContextData = requires(T t) {
  { t.isLoadedAsync() } -> std::convertible_to<bool>;
  { t.pollAsync() } -> std::convertible_to<bool>;
};

// This is the source of the context data on the client side that provides
// functions to populate the gpu-side context data and timestamps to coordinate when to update
template <typename T>
concept TWithContextData = requires(T t, typename T::ContextDataType &cd, Context &ctx) {
  typename T::ContextDataType;
  { t.initContextData(ctx, cd) };
  { t.updateContextData(ctx, cd) };
  { t.getVersion() } -> std::convertible_to<uint64_t>;
  { t.getId() } -> std::convertible_to<UniqueId>;
} && TContextData<typename T::ContextDataType>;

} // namespace gfx

#endif // GFX_CONTEXT_DATA
