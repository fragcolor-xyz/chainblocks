#ifndef A16CC8A4_FBC4_4500_BE1D_F565963C9C16
#define A16CC8A4_FBC4_4500_BE1D_F565963C9C16

#include <shards.hpp>
#include <runtime.hpp>
#include <array>

namespace shards {

// Defines a reference to a required variable
// Warmup references the variable, cleanup releases the refence
// use  getRequiredVariable to get the type
template <typename T, const SHTypeInfo &VariableType, const char *VariableName, bool Required = true>
struct RequiredContextVariable {
private:
  SHVar *variable{};

public:
  RequiredContextVariable() {}

  void warmup(SHContext *context) {
    assert(!variable);
    variable = shards::referenceVariable(context, VariableName);

    if constexpr (Required) {
      // Ensure correct runtime type
      (void)varAsObjectChecked<T>(*variable, VariableType);
    }
  }

  void cleanup() {
    if (variable) {
      shards::releaseVariable(variable);
      variable = nullptr;
    }
  }

  operator bool() const { return variable; }
  T *operator->() const {
    assert(variable); // Missing call to warmup?
    return reinterpret_cast<T *>(variable->payload.objectValue);
  }
  operator T &() const {
    assert(variable); // Missing call to warmup?
    return *reinterpret_cast<T *>(variable->payload.objectValue);
  }

  static constexpr SHExposedTypeInfo getExposedTypeInfo() {
    SHExposedTypeInfo typeInfo{
        .name = VariableName,
        .exposedType = VariableType,
        .global = true,
    };
    return typeInfo;
  }
};

namespace detail {
template <size_t N> struct StaticExposedArray : public std::array<SHExposedTypeInfo, N> {
  operator SHExposedTypesInfo() const {
    return SHExposedTypesInfo{.elements = const_cast<SHExposedTypeInfo *>(this->data()), .len = N, .cap = 0};
  }
};
} // namespace detail

// Returns a storage type that is convertible to SHExposedTypesInfo
// Example:
//   SHExposedTypesInfo requiredVariables() {
//     static auto e = exposedTypesOf(SomeExposedTypeInfo, SHExposedTypeInfo{...});
//     return e;
//   }
//
//   RequiredContextVariable<GraphicsContext, GraphicsContext::Type, Base::graphicsContextVarName> graphicsContext;
//   SHExposedTypesInfo requiredVariables() {
//     static auto e = exposedTypesOf(decltype(graphicsContext)::getExposedTypeInfo());
//     return e;
//   }
template <typename... TArgs> constexpr auto exposedTypesOf(TArgs... args) {
  detail::StaticExposedArray<sizeof...(TArgs)> result;
  size_t index{};
  (
      [&](SHExposedTypeInfo typeInfo) {
        result[index] = typeInfo;
        ++index;
      }(SHExposedTypeInfo(args)),
      ...);
  return result;
}
} // namespace shards

#endif /* A16CC8A4_FBC4_4500_BE1D_F565963C9C16 */
