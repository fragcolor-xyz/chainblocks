#include "type_info.hpp"
#include "foundation.hpp"
#include "trait.hpp"
#include <shards/iterator.hpp>

namespace shards {
void freeTypeInfo(SHTypeInfo info) {
  switch (info.basicType) {
  case SHType::Object:
    if (info.object.extInfo) {
      info.object.extInfo->release(info.object.extInfoData);
    }
    break;
  case SHType::ContextVar: {
    for (uint32_t i = 0; info.contextVarTypes.len > i; i++) {
      freeTypeInfo(info.contextVarTypes.elements[i]);
    }
    shards::arrayFree(info.contextVarTypes);
  } break;
  case SHType::Seq: {
    for (uint32_t i = 0; info.seqTypes.len > i; i++) {
      freeTypeInfo(info.seqTypes.elements[i]);
    }
    shards::arrayFree(info.seqTypes);
  } break;
  case SHType::Table: {
    for (uint32_t i = 0; info.table.types.len > i; i++) {
      freeTypeInfo(info.table.types.elements[i]);
    }
    for (uint32_t i = 0; info.table.keys.len > i; i++) {
      destroyVar(info.table.keys.elements[i]);
    }
    shards::arrayFree(info.table.types);
    shards::arrayFree(info.table.keys);
  } break;
  default:
    break;
  }
}

SHTypeInfo cloneTypeInfo(const SHTypeInfo &other) {
  SHTypeInfo varType;
  memcpy(&varType, &other, sizeof(SHTypeInfo));
  switch (varType.basicType) {
  case SHType::Object:
    if (other.object.extInfo) {
      other.object.extInfo->reference(other.object.extInfoData);
    }
    break;
  case SHType::ContextVar: {
    varType.contextVarTypes = {};
    for (uint32_t i = 0; i < other.contextVarTypes.len; i++) {
      auto cloned = cloneTypeInfo(other.contextVarTypes.elements[i]);
      shards::arrayPush(varType.contextVarTypes, cloned);
    }
    break;
  }
  case SHType::Seq: {
    varType.seqTypes = {};
    for (uint32_t i = 0; i < other.seqTypes.len; i++) {
      auto cloned = cloneTypeInfo(other.seqTypes.elements[i]);
      shards::arrayPush(varType.seqTypes, cloned);
    }
    break;
  }
  case SHType::Table: {
    varType.table = {};
    for (uint32_t i = 0; i < other.table.types.len; i++) {
      auto cloned = cloneTypeInfo(other.table.types.elements[i]);
      shards::arrayPush(varType.table.types, cloned);
    }
    for (uint32_t i = 0; i < other.table.keys.len; i++) {
      auto idx = varType.table.keys.len;
      shards::arrayResize(varType.table.keys, idx + 1);
      cloneVar(varType.table.keys.elements[idx], other.table.keys.elements[i]);
    }
    break;
  }
  default:
    break;
  }
  return varType;
}

SHTypeInfo deriveTypeInfo(const SHVar &value, const SHInstanceData &data, std::vector<SHExposedTypeInfo> *expInfo,
                          bool resolveContextVariables) {
  ZoneScopedN("deriveTypeInfo");

  SHTypeInfo varType{};
  varType.basicType = value.valueType;
  varType.innerType = value.innerType;
  switch (value.valueType) {
  case SHType::Object: {
    varType.object.vendorId = value.payload.objectVendorId;
    varType.object.typeId = value.payload.objectTypeId;
    break;
  }
  case SHType::Enum: {
    varType.enumeration.vendorId = value.payload.enumVendorId;
    varType.enumeration.typeId = value.payload.enumTypeId;
    break;
  }
  case SHType::Seq: {
    std::unordered_set<SHTypeInfo> types;
    for (uint32_t i = 0; i < value.payload.seqValue.len; i++) {
      auto derived = deriveTypeInfo(value.payload.seqValue.elements[i], data, expInfo, resolveContextVariables);
      if (!types.count(derived)) {
        shards::arrayPush(varType.seqTypes, derived);
        types.insert(derived);
      } else {
        freeTypeInfo(derived);
      }
    }
    varType.fixedSize = value.payload.seqValue.len;
    // if the len is 0 we should make it a [Any] seq!
    if (value.payload.seqValue.len == 0) {
      shards::arrayPush(varType.seqTypes, SHTypeInfo{SHType::Any});
    }
    break;
  }
  case SHType::Table: {
    auto &t = value.payload.tableValue;
    SHTableIterator tit;
    t.api->tableGetIterator(t, &tit);
    SHVar k;
    SHVar v;
    while (t.api->tableNext(t, &tit, &k, &v)) {
      auto derived = deriveTypeInfo(v, data, expInfo, resolveContextVariables);
      shards::arrayPush(varType.table.types, derived);
      auto idx = varType.table.keys.len;
      shards::arrayResize(varType.table.keys, idx + 1);
      cloneVar(varType.table.keys.elements[idx], k);
    }
    break;
  }
  case SHType::ContextVar: {
    if (expInfo) {
      auto sv = SHSTRVIEW(value);
      const auto varName = sv;
      shassert(data.privateContext && "Private context should be valid");
      auto inherited = reinterpret_cast<CompositionContext *>(data.privateContext);
      auto info = findExposedVariable(inherited->inherited, varName);
      if (info) {
        expInfo->push_back(*info);
        if (resolveContextVariables) {
          return cloneTypeInfo(info->exposedType);
        } else {
          shards::arrayPush(varType.contextVarTypes, cloneTypeInfo(info->exposedType));
          return varType;
        }
      } else {
        SHLOG_ERROR("Could not find variable {} when deriving type info", varName);
        throw std::runtime_error(fmt::format("Could not find variable {} when deriving type info", varName));
      }
    }
  } break;
  default:
    break;
  };
  return varType;
}
} // namespace shards
