#ifndef B0328A63_0B69_4191_94D5_38783B9F20C9
#define B0328A63_0B69_4191_94D5_38783B9F20C9

#include "stddef.h"
#include "foundation.hpp"
#include "self_macro.h"
#include "exposed_type_utils.hpp"
#include <vector>
#include <type_traits>
#include <shards/shardwrapper.hpp>

// Shards parameter Macros
// The general usage will be:
//   struct MyShard {
//     PARAM_PARAMVAR(_varNameToDefine, "PublicName", "Help text", {shards::CoreInfo::StringType, OtherTypes... })
//     PARAM_IMPL(PARAM_IMPL_FOR(_varNameToDefine))
//
//     PARAM_REQUIRED_VARIABLES();
//     SHTypeInfo compose(SHInstanceData &data) {
//       PARAM_COMPOSE_REQUIRED_VARIABLES(data);
//       return outputTypes().elements[0];
//     }
//     void warmup(SHContext *ctx) {
//       PARAM_WARMUP(ctx);
//     }
//     void cleanup(SHContext *ctx) {
//       PARAM_CLEANUP(ctx);
//     }
//   };
// CAUTION: 
//  - If you use PARAM_PARAMVAR, you must also add the compose(), warmup() and cleanup() functions

namespace shards {
#define PARAM_EXT(_type, _name, _paramInfo)                              \
  static inline shards::ParameterInfo _name##ParameterInfo = _paramInfo; \
  _type _name{};

#define PARAM(_type, _name, _displayName, _help, ...)                                                     \
  static inline shards::ParameterInfo _name##ParameterInfo = {_displayName, SHCCSTR(_help), __VA_ARGS__}; \
  _type _name{};

#define PARAM_VAR(_name, _displayName, _help, ...) PARAM(shards::OwnedVar, _name, _displayName, _help, __VA_ARGS__)

// Defines a ParamVar variable
// CAUTION: you must manually add the Var type to the list of supported types (e.g. CoreInfo::StringVarType)
#define PARAM_PARAMVAR(_name, _displayName, _help, ...) PARAM(shards::ParamVar, _name, _displayName, _help, __VA_ARGS__)

struct IterableParam {
  // Return address of param inside shard
  void *(*resolveParamInShard)(void *shardPtr);
  const shards::ParameterInfo *paramInfo;

  void (*setParam)(void *varPtr, SHVar var){};
  SHVar (*getParam)(void *varPtr){};
  void (*collectRequirements)(const shards::IterableParam &param, const SHExposedTypesInfo &exposed, ExposedInfo &out,
                              void *varPtr){};
  void (*warmup)(void *varPtr, SHContext *ctx){};
  void (*cleanup)(void *varPtr, SHContext *ctx){};

  template <typename T> T &get(void *obj) const { return *(T *)resolveParamInShard(obj); }

  template <typename T> static IterableParam create(void *(*resolveParamInShard)(void *), const ParameterInfo *paramInfo) {
    return createWithVarInterface<T>(resolveParamInShard, paramInfo);
  }

  template <typename T>
  static IterableParam createWithVarInterface(void *(*resolveParamInShard)(void *), const ParameterInfo *paramInfo) {
    IterableParam result{
        .resolveParamInShard = resolveParamInShard,
        .paramInfo = paramInfo,
        .setParam = [](void *varPtr, SHVar var) { *((T *)varPtr) = var; },
        .getParam = [](void *varPtr) -> SHVar { return *((T *)varPtr); },
        .collectRequirements =
            [](const shards::IterableParam &param, const SHExposedTypesInfo &exposed, ExposedInfo &out, void *varPtr) {
              collectRequiredVariables(exposed, out, *((T *)varPtr), SHTypesInfo(param.paramInfo->_types),
                                       param.paramInfo->_name);
            }};

    bool canPossiblyHaveContextVariables = false;
    for (auto &type : paramInfo->_types._types) {
      if (hasContextVariables(type)) {
        canPossiblyHaveContextVariables = true;
        break;
      }
    }
    if (!canPossiblyHaveContextVariables) {
      result.collectRequirements = nullptr;
    }

    if constexpr (has_warmup<T>::value) {
      result.warmup = [](void *varPtr, SHContext *ctx) { ((T *)varPtr)->warmup(ctx); };
    } else {
      result.warmup = [](void *, SHContext *) {};
    }

    if constexpr (has_warmup<T>::value) {
      result.cleanup = [](void *varPtr, SHContext *ctx) { ((T *)varPtr)->cleanup(ctx); };
    } else {
      result.cleanup = [](void *, SHContext *ctx) {};
    }

    return result;
  }
};

// Usage:
//  PARAM_IMPL(
//    PARAM_IMPL_FOR(Param0),
//    PARAM_IMPL_FOR(Param1),
//    PARAM_IMPL_FOR(Param2))
// Side effects:
//  - typedefs Self to the current class
//  - creates a static member named iterableParams
#define PARAM_IMPL(...)                                                         \
  SELF_MACRO_DEFINE_SELF(Self, public)                                          \
  static const shards::IterableParam *getIterableParams(size_t &outNumParams) { \
    static shards::IterableParam result[] = {__VA_ARGS__};                      \
    outNumParams = std::extent<decltype(result)>::value;                        \
    return result;                                                              \
  }                                                                             \
  PARAM_PARAMS()                                                                \
  PARAM_GET_SET()

// Usage:
//  PARAM_IMPL_DERIVED(BaseClass,
//    PARAM_IMPL_FOR(Param0),
//    PARAM_IMPL_FOR(Param1),
//    PARAM_IMPL_FOR(Param2))
// Side effects:
//  - same as PARAM_IMPL
// NOTE: Make sure you DO NOT call base PARAM_WARMUP, PARAM_COMPOSE_REQUIRED_VARIABLES, etc. in the base class
//   and only use PARAM_REQUIRED_VARIABLES once in the hierarchy to save storage.
//   so only call PARAM_WARMUP/CLEANUP/COMPOSE in the final class!!
#define PARAM_IMPL_DERIVED(BaseClass, ...)                                      \
  SELF_MACRO_DEFINE_SELF(Self, public)                                          \
  static const shards::IterableParam *getIterableParams(size_t &outNumParams) { \
    static std::vector<shards::IterableParam> combined = []() {                 \
      static shards::IterableParam addParams[] = {__VA_ARGS__};                 \
      size_t numAddParams = std::extent<decltype(addParams)>::value;            \
                                                                                \
      size_t numBaseParams{};                                                   \
      auto *baseParams = BaseClass::getIterableParams(numBaseParams);           \
      std::vector<shards::IterableParam> result;                                \
      result.resize(numBaseParams + numAddParams);                              \
      for (size_t i = 0; i < numBaseParams; ++i)                                \
        result[i] = baseParams[i];                                              \
      for (size_t i = 0; i < numAddParams; ++i)                                 \
        result[numBaseParams + i] = addParams[i];                               \
      return result;                                                            \
    }();                                                                        \
    outNumParams = combined.size();                                             \
    return combined.data();                                                     \
  }                                                                             \
  PARAM_PARAMS()                                                                \
  PARAM_GET_SET()

#define PARAM_IMPL_FOR(_name)                                                                                       \
  shards::IterableParam::create<decltype(_name)>([](void *obj) -> void * { return (void *)&((Self *)obj)->_name; }, \
                                                 &_name##ParameterInfo)

// Implements parameters()
#define PARAM_PARAMS()                                                                  \
  static SHParametersInfo parameters() {                                                \
    static SHParametersInfo result = []() {                                             \
      SHParametersInfo result{};                                                        \
      size_t numParams;                                                                 \
      const shards::IterableParam *params = getIterableParams(numParams);               \
      shards::arrayResize(result, numParams);                                           \
      for (size_t i = 0; i < numParams; i++) {                                          \
        result.elements[i] = *const_cast<shards::ParameterInfo *>(params[i].paramInfo); \
      }                                                                                 \
      return result;                                                                    \
    }();                                                                                \
    return result;                                                                      \
  }

#define PARAM_REQUIRED_VARIABLES()        \
  shards::ExposedInfo _requiredVariables; \
  SHExposedTypesInfo requiredVariables() { return (SHExposedTypesInfo)_requiredVariables; }

// Implements collection of required variables
// Usage:
//  PARAM_REQUIRED_VARIABLES();
//  SHTypeInfo compose(SHInstanceData &data) {
//    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
//    return outputTypes().elements[0];
//  }
#define PARAM_COMPOSE_REQUIRED_VARIABLES(__data)                                                                          \
  {                                                                                                                       \
    size_t numParams;                                                                                                     \
    const shards::IterableParam *params = getIterableParams(numParams);                                                   \
    _requiredVariables.clear();                                                                                           \
    for (size_t i = 0; i < numParams; i++) {                                                                              \
      if (params[i].collectRequirements) {                                                                                \
        params[i].collectRequirements(params[i], __data.shared, _requiredVariables, params[i].resolveParamInShard(this)); \
      }                                                                                                                   \
    }                                                                                                                     \
  }

// Implements setParam()/getParam()
#define PARAM_GET_SET()                                                       \
  void setParam(int index, const SHVar &value) {                              \
    size_t numParams;                                                         \
    const shards::IterableParam *params = getIterableParams(numParams);       \
    if (index < int(numParams)) {                                             \
      params[index].setParam(params[index].resolveParamInShard(this), value); \
    } else {                                                                  \
      throw shards::InvalidParameterIndex();                                  \
    }                                                                         \
  }                                                                           \
  SHVar getParam(int index) {                                                 \
    size_t numParams;                                                         \
    const shards::IterableParam *params = getIterableParams(numParams);       \
    if (index < int(numParams)) {                                             \
      return params[index].getParam(params[index].resolveParamInShard(this)); \
    } else {                                                                  \
      throw shards::InvalidParameterIndex();                                  \
    }                                                                         \
  }

// Implements warmup(ctx) for parameters
// call from warmup manually with context
#define PARAM_WARMUP(_ctx)                                              \
  {                                                                     \
    size_t numParams;                                                   \
    const shards::IterableParam *params = getIterableParams(numParams); \
    for (size_t i = 0; i < numParams; i++)                              \
      params[i].warmup(params[i].resolveParamInShard(this), _ctx);      \
  }

// implements cleanup() for parameters
// call from cleanup manually
#define PARAM_CLEANUP(_ctx)                                               \
  {                                                                       \
    size_t numParams;                                                     \
    const shards::IterableParam *params = getIterableParams(numParams);   \
    for (size_t i = 0; i < numParams; i++) {                              \
      size_t iRev = (numParams - 1) - i;                                  \
      params[iRev].cleanup(params[iRev].resolveParamInShard(this), _ctx); \
    }                                                                     \
  }

} // namespace shards

#endif /* B0328A63_0B69_4191_94D5_38783B9F20C9 */
