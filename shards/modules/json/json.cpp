/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include "nlohmann/json.hpp"
#include <shards/core/module.hpp>
#include <shards/core/runtime.hpp>
#include <shards/shards.h>
#include <shards/core/shared.hpp>
#include <shards/utility.hpp>
#include <magic_enum.hpp>

using json = nlohmann::json;

void from_json(const json &j, SHWireRef &wire);
void to_json(json &j, const SHWireRef &wire);

void _releaseMemory(SHVar &var) {
  // Used by Shard and Wire from_json
  switch (var.valueType) {
  case SHType::Path:
  case SHType::ContextVar:
  case SHType::String:
    delete[] var.payload.stringValue;
    break;
  case SHType::Image:
    shards::imageDecRef(var.payload.imageValue);
    break;
  case SHType::Audio:
    delete[] var.payload.audioValue.samples;
    break;
  case SHType::Bytes:
    delete[] var.payload.bytesValue;
    break;
  case SHType::Seq:
    for (uint32_t i = 0; i < var.payload.seqValue.len; i++) {
      _releaseMemory(var.payload.seqValue.elements[i]);
    }
    shards::arrayFree(var.payload.seqValue);
    break;
  case SHType::Table: {
    auto map = (shards::SHMap *)var.payload.tableValue.opaque;
    delete map;
  } break;
  case SHType::Wire: {
    SHWire::deleteRef(var.payload.wireValue);
  } break;
  case SHType::Object: {
    if ((var.flags & SHVAR_FLAGS_USES_OBJINFO) == SHVAR_FLAGS_USES_OBJINFO && var.objectInfo && var.objectInfo->release) {
      var.objectInfo->release(var.payload.objectValue);
    }
  } break;
  default:
    break;
  }
  var = {};
}

void to_json(json &j, const SHVar &var) {
  auto valType = magic_enum::enum_name(var.valueType);
  switch (var.valueType) {
  case SHType::Any:
  case SHType::EndOfBlittableTypes:
  case SHType::None: {
    j = json{{"type", valType}};
    break;
  }
  case SHType::Bool: {
    j = json{{"type", valType}, {"value", var.payload.boolValue}};
    break;
  }
  case SHType::Int: {
    j = json{{"type", valType}, {"value", var.payload.intValue}};
    break;
  }
  case SHType::Int2: {
    auto vec = {var.payload.int2Value[0], var.payload.int2Value[1]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Int3: {
    auto vec = {var.payload.int3Value[0], var.payload.int3Value[1], var.payload.int3Value[2]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Int4: {
    auto vec = {var.payload.int4Value[0], var.payload.int4Value[1], var.payload.int4Value[2], var.payload.int4Value[3]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Int8: {
    auto vec = {var.payload.int8Value[0], var.payload.int8Value[1], var.payload.int8Value[2], var.payload.int8Value[3],
                var.payload.int8Value[4], var.payload.int8Value[5], var.payload.int8Value[6], var.payload.int8Value[7]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Int16: {
    std::stringstream ss;
    for (int i = 0; i < 16; i++) {
        ss << std::setfill('0') << std::setw(2) << std::hex 
           << (static_cast<int>(var.payload.int16Value[i]) & 0xFF);
    }
    j = json{{"type", valType}, {"value", ss.str()}};
    break;
  }
  case SHType::Float: {
    j = json{{"type", valType}, {"value", var.payload.floatValue}};
    break;
  }
  case SHType::Float2: {
    auto vec = {var.payload.float2Value[0], var.payload.float2Value[1]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Float3: {
    auto vec = {var.payload.float3Value[0], var.payload.float3Value[1], var.payload.float3Value[2]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Float4: {
    auto vec = {var.payload.float4Value[0], var.payload.float4Value[1], var.payload.float4Value[2], var.payload.float4Value[3]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case SHType::Path:
  case SHType::ContextVar:
  case SHType::String: {
    j = json{{"type", valType}, {"value", SHSTRVIEW(var)}};
    break;
  }
  case SHType::Color: {
    j = json{{"type", valType},
             {"value", {var.payload.colorValue.r, var.payload.colorValue.g, var.payload.colorValue.b, var.payload.colorValue.a}}};
    break;
  }
  case SHType::Image: {
    if (var.payload.imageValue->data) {
      auto binsize = shards::imageDeriveDataLength(var.payload.imageValue);
      std::vector<uint8_t> buffer;
      buffer.resize(binsize);
      memcpy(&buffer[0], var.payload.imageValue->data, binsize);
      j = json{{"type", valType},
               {"width", var.payload.imageValue->width},
               {"height", var.payload.imageValue->height},
               {"channels", var.payload.imageValue->channels},
               {"flags", var.payload.imageValue->flags},
               {"data", buffer}};
    } else {
      j = json{{"type", 0}, {"value", int(SHWireState::Continue)}};
    }
    break;
  }
  case SHType::Audio: {
    if (var.payload.audioValue.samples) {
      auto size = var.payload.audioValue.nsamples * var.payload.audioValue.channels;
      std::vector<float> buffer;
      buffer.resize(size);
      memcpy(&buffer[0], var.payload.audioValue.samples, size * sizeof(float));
      j = json{{"type", valType},
               {"sampleRate", var.payload.audioValue.sampleRate},
               {"nsamples", var.payload.audioValue.nsamples},
               {"channels", var.payload.audioValue.channels},
               {"samples", buffer}};
    } else {
      j = json{{"type", 0}, {"value", int(SHWireState::Continue)}};
    }
    break;
  }
  case SHType::Bytes: {
    std::vector<uint8_t> buffer;
    buffer.resize(var.payload.bytesSize);
    if (var.payload.bytesSize > 0)
      memcpy(&buffer[0], var.payload.bytesValue, var.payload.bytesSize);
    j = json{{"type", valType}, {"data", buffer}};
    break;
  }
  case SHType::Enum: {
    j = json{{"type", valType},
             {"value", int32_t(var.payload.enumValue)},
             {"vendorId", var.payload.enumVendorId},
             {"typeId", var.payload.enumTypeId}};
    break;
  }
  case SHType::Seq: {
    std::vector<json> items;
    for (uint32_t i = 0; i < var.payload.seqValue.len; i++) {
      auto &v = var.payload.seqValue.elements[i];
      items.emplace_back(v);
    }
    j = json{{"type", valType}, {"values", items}};
    break;
  }
  case SHType::Table: {
    std::vector<json> items;
    auto &t = var.payload.tableValue;
    SHTableIterator tit;
    t.api->tableGetIterator(t, &tit);
    SHVar k;
    SHVar v;
    while (t.api->tableNext(t, &tit, &k, &v)) {
      json entry{{"key", k}, {"value", v}};
      items.emplace_back(entry);
    }
    j = json{{"type", valType}, {"values", items}};
    break;
  }
  case SHType::ShardRef: {
    auto blk = var.payload.shardValue;
    std::vector<json> params;
    auto paramsDesc = blk->parameters(blk);
    for (uint32_t i = 0; i < paramsDesc.len; i++) {
      auto &desc = paramsDesc.elements[i];
      auto value = blk->getParam(blk, i);
      json param_obj = {{"name", desc.name}, {"value", value}};
      params.push_back(param_obj);
    }
    if (blk->getState) {
      j = json{{"type", valType}, {"name", blk->name(blk)}, {"params", params}, {"state", blk->getState(blk)}};
    } else {
      j = json{{"type", valType}, {"name", blk->name(blk)}, {"params", params}};
    }
    break;
  }
  case SHType::Wire: {
    json jwire = var.payload.wireValue;
    j = json{{"type", valType}, {"value", jwire}};
    break;
  }
  case SHType::Object: {
    j = json{{"type", valType}, {"vendorId", var.payload.objectVendorId}, {"typeId", var.payload.objectTypeId}};
    if ((var.flags & SHVAR_FLAGS_USES_OBJINFO) == SHVAR_FLAGS_USES_OBJINFO && var.objectInfo && var.objectInfo->serialize) {
      uint64_t len = 0;
      uint8_t *data = nullptr;
      SHPointer handle = nullptr;
      if (!var.objectInfo->serialize(var.payload.objectValue, &data, &len, &handle)) {
        throw shards::SHException("Failed to serialize custom object variable!");
      }
      std::vector<uint8_t> buf;
      buf.resize(len);
      memcpy(&buf.front(), data, len);
      j.emplace("data", buf);
      var.objectInfo->free(handle);
    } else {
      throw shards::ActivationError("Custom object cannot be serialized.");
    }
    break;
  }
  case SHType::Type:
    throw shards::ActivationError("Type can not be serialized to json.");
  case SHType::Trait:
    throw shards::ActivationError("Trait can not be serialized to json.");
  };
}

void from_json(const json &j, SHVar &var) {
  auto valName = j.at("type").get<std::string>();
  auto valType = magic_enum::enum_cast<SHType>(valName);
  if (!valType.has_value()) {
    throw shards::ActivationError("Failed to parse SHVar value type.");
  }
  switch (valType.value()) {
  case SHType::Any:
  case SHType::EndOfBlittableTypes:
  case SHType::None: {
    var = {};
    var.valueType = valType.value();
    break;
  }
  case SHType::Bool: {
    var.valueType = SHType::Bool;
    var.payload.boolValue = j.at("value").get<bool>();
    break;
  }
  case SHType::Int: {
    var.valueType = SHType::Int;
    var.payload.intValue = j.at("value").get<int64_t>();
    break;
  }
  case SHType::Int2: {
    var.valueType = SHType::Int2;
    var.payload.int2Value[0] = j.at("value")[0].get<int64_t>();
    var.payload.int2Value[1] = j.at("value")[1].get<int64_t>();
    break;
  }
  case SHType::Int3: {
    var.valueType = SHType::Int3;
    var.payload.int3Value[0] = j.at("value")[0].get<int32_t>();
    var.payload.int3Value[1] = j.at("value")[1].get<int32_t>();
    var.payload.int3Value[2] = j.at("value")[2].get<int32_t>();
    break;
  }
  case SHType::Int4: {
    var.valueType = SHType::Int4;
    var.payload.int4Value[0] = j.at("value")[0].get<int32_t>();
    var.payload.int4Value[1] = j.at("value")[1].get<int32_t>();
    var.payload.int4Value[2] = j.at("value")[2].get<int32_t>();
    var.payload.int4Value[3] = j.at("value")[3].get<int32_t>();
    break;
  }
  case SHType::Int8: {
    var.valueType = SHType::Int8;
    for (auto i = 0; i < 8; i++) {
      var.payload.int8Value[i] = j.at("value")[i].get<int16_t>();
    }
    break;
  }
  case SHType::Int16: {
    var.valueType = SHType::Int16;
    auto hexStr = j.at("value").get<std::string>();
    if (hexStr.length() != 32) { // 16 bytes = 32 hex chars
        throw shards::ActivationError("Int16 hex string must be exactly 32 characters long (16 bytes)");
    }
    for (int i = 0; i < 16; i++) {
        std::string byteStr = hexStr.substr(i * 2, 2);
        var.payload.int16Value[i] = static_cast<int8_t>(std::stoi(byteStr, nullptr, 16));
    }
    break;
  }
  case SHType::Float: {
    var.valueType = SHType::Float;
    var.payload.floatValue = j.at("value").get<double>();
    break;
  }
  case SHType::Float2: {
    var.valueType = SHType::Float2;
    var.payload.float2Value[0] = j.at("value")[0].get<double>();
    var.payload.float2Value[1] = j.at("value")[1].get<double>();
    break;
  }
  case SHType::Float3: {
    var.valueType = SHType::Float3;
    var.payload.float3Value[0] = j.at("value")[0].get<float>();
    var.payload.float3Value[1] = j.at("value")[1].get<float>();
    var.payload.float3Value[2] = j.at("value")[2].get<float>();
    break;
  }
  case SHType::Float4: {
    var.valueType = SHType::Float4;
    var.payload.float4Value[0] = j.at("value")[0].get<float>();
    var.payload.float4Value[1] = j.at("value")[1].get<float>();
    var.payload.float4Value[2] = j.at("value")[2].get<float>();
    var.payload.float4Value[3] = j.at("value")[3].get<float>();
    break;
  }
  case SHType::ContextVar: {
    var.valueType = SHType::ContextVar;
    auto strVal = j.at("value").get<std::string>();
    const auto strLen = strVal.length();
    var.payload.stringValue = new char[strLen + 1];
    var.payload.stringLen = uint32_t(strLen);
    memcpy((void *)var.payload.stringValue, strVal.c_str(), strLen);
    ((char *)var.payload.stringValue)[strLen] = 0;
    break;
  }
  case SHType::String: {
    var.valueType = SHType::String;
    auto strVal = j.at("value").get<std::string>();
    const auto strLen = strVal.length();
    var.payload.stringValue = new char[strLen + 1];
    var.payload.stringLen = uint32_t(strLen);
    memcpy((void *)var.payload.stringValue, strVal.c_str(), strLen);
    ((char *)var.payload.stringValue)[strLen] = 0;
    break;
  }
  case SHType::Path: {
    var.valueType = SHType::Path;
    auto strVal = j.at("value").get<std::string>();
    const auto strLen = strVal.length();
    var.payload.stringValue = new char[strLen + 1];
    var.payload.stringLen = uint32_t(strLen);
    memcpy((void *)var.payload.stringValue, strVal.c_str(), strLen);
    ((char *)var.payload.stringValue)[strLen] = 0;
    break;
  }
  case SHType::Color: {
    var.valueType = SHType::Color;
    var.payload.colorValue.r = j.at("value")[0].get<uint8_t>();
    var.payload.colorValue.g = j.at("value")[1].get<uint8_t>();
    var.payload.colorValue.b = j.at("value")[2].get<uint8_t>();
    var.payload.colorValue.a = j.at("value")[3].get<uint8_t>();
    break;
  }
  case SHType::Image: {
    var.valueType = SHType::Image;
    SHImage tmpImage{};
    tmpImage.width = j.at("width").get<int32_t>();
    tmpImage.height = j.at("height").get<int32_t>();
    tmpImage.channels = j.at("channels").get<int32_t>();
    tmpImage.flags = j.at("flags").get<int32_t>();
    auto binsize = shards::imageDeriveDataLength(&tmpImage);
    SHImage &outImage = *(var.payload.imageValue = shards::imageNew(binsize));
    auto buffer = j.at("data").get<std::vector<uint8_t>>();
    memcpy(outImage.data, &buffer[0], binsize);
    break;
  }
  case SHType::Bytes: {
    var.valueType = SHType::Bytes;
    auto buffer = j.at("data").get<std::vector<uint8_t>>();
    var.payload.bytesValue = new uint8_t[buffer.size()];
    memcpy(var.payload.bytesValue, &buffer[0], buffer.size());
    break;
  }
  case SHType::Audio: {
    var.valueType = SHType::Audio;
    var.payload.audioValue.sampleRate = j.at("sampleRate").get<float>();
    var.payload.audioValue.nsamples = j.at("nsamples").get<uint16_t>();
    var.payload.audioValue.channels = j.at("channels").get<uint16_t>();
    auto size = var.payload.audioValue.nsamples * var.payload.audioValue.channels;
    var.payload.audioValue.samples = new float[size];
    auto buffer = j.at("samples").get<std::vector<float>>();
    memcpy(var.payload.audioValue.samples, &buffer[0], size * sizeof(float));
    break;
  }
  case SHType::Enum: {
    var.valueType = SHType::Enum;
    var.payload.enumValue = SHEnum(j.at("value").get<int32_t>());
    var.payload.enumVendorId = SHEnum(j.at("vendorId").get<int32_t>());
    var.payload.enumTypeId = SHEnum(j.at("typeId").get<int32_t>());
    break;
  }
  case SHType::Seq: {
    var.valueType = SHType::Seq;
    auto items = j.at("values").get<std::vector<json>>();
    var.payload.seqValue = {};
    for (const auto &item : items) {
      shards::arrayPush(var.payload.seqValue, item.get<SHVar>());
    }
    break;
  }
  case SHType::Table: {
    auto map = new shards::SHMap();
    var.valueType = SHType::Table;
    var.payload.tableValue.api = &shards::GetGlobals().TableInterface;
    var.payload.tableValue.opaque = map;
    auto items = j.at("values").get<std::vector<json>>();
    for (const auto &item : items) {
      auto key = item.at("key").get<SHVar>();
      auto value = item.at("value").get<SHVar>();
      (*map)[key] = value;
      _releaseMemory(key);   // key is copied over
      _releaseMemory(value); // value is copied over
    }
    break;
  }
  case SHType::ShardRef: {
    var.valueType = SHType::ShardRef;
    auto blkname = j.at("name").get<std::string>();
    auto blk = shards::createShard(blkname.c_str());
    if (!blk) {
      auto errmsg = "Failed to create shard of type: " + std::string("blkname");
      throw shards::ActivationError(errmsg.c_str());
    }

    shards::incRef(blk);
    var.payload.shardValue = blk;

    // Setup
    blk->setup(blk);

    // Set params
    auto jparams = j.at("params");
    auto blkParams = blk->parameters(blk);
    for (auto jparam : jparams) {
      auto paramName = jparam.at("name").get<std::string>();
      auto value = jparam.at("value").get<SHVar>();
      if (value.valueType != SHType::None) {
        for (uint32_t i = 0; blkParams.len > i; i++) {
          auto &paramInfo = blkParams.elements[i];
          if (paramName == paramInfo.name) {
            blk->setParam(blk, i, &value);
            break;
          }
        }
      }
      // Assume shard copied memory internally so we can clean up here!!!
      _releaseMemory(value);
    }

    if (blk->setState) {
      auto state = j.at("state").get<SHVar>();
      blk->setState(blk, &state);
      _releaseMemory(state);
    }
    break;
  }
  case SHType::Wire: {
    var.valueType = SHType::Wire;
    var.payload.wireValue = j.at("value").get<SHWireRef>();
    break;
  }
  case SHType::Object: {
    var.valueType = SHType::Object;
    var.payload.objectVendorId = SHEnum(j.at("vendorId").get<int32_t>());
    var.payload.objectTypeId = SHEnum(j.at("typeId").get<int32_t>());
    var.payload.objectValue = nullptr;
    const SHObjectInfo *objectInfo = shards::findObjectInfo(var.payload.objectVendorId, var.payload.objectTypeId);
    if (objectInfo) {
      auto data = j.at("data").get<std::vector<uint8_t>>();
      var.payload.objectValue = objectInfo->deserialize(&data.front(), data.size());
      var.flags |= SHVAR_FLAGS_USES_OBJINFO;
      var.objectInfo = const_cast<SHObjectInfo *>(objectInfo);
      if (objectInfo->reference)
        objectInfo->reference(var.payload.objectValue);
    } else {
      throw shards::ActivationError("Failed to find object type in registry.");
    }
    break;
  }
  case SHType::Type:
    throw shards::ActivationError("Type can not be deserialized from json.");
  case SHType::Trait:
    throw shards::ActivationError("Trait can not be serialized to json.");
  }
}

void to_json(json &j, const SHWireRef &wireref) {
  auto wire = SHWire::sharedFromRef(wireref);
  std::vector<json> shards;
  for (auto blk : wire->shards) {
    SHVar blkVar{};
    blkVar.valueType = SHType::ShardRef;
    blkVar.payload.shardValue = blk;
    shards.emplace_back(blkVar);
  }
  j = {
      {"shards", shards},
      {"name", wire->name},
      {"looped", wire->looped},
      {"unsafe", wire->unsafe},
      {"version", SHARDS_CURRENT_ABI},
  };
}

void from_json(const json &j, SHWireRef &wireref) {
  auto wireName = j.at("name").get<std::string>();
  auto wire = SHWire::make(wireName);

  wire->looped = j.at("looped").get<bool>();
  wire->unsafe = j.at("unsafe").get<bool>();

  auto blks = j.at("shards").get<std::vector<SHVar>>();
  for (auto blk : blks) {
    wire->addShard(blk.payload.shardValue);
    wireref = wire->newRef();
  }
}

namespace shards {
struct ToJson {
  std::string _output;
  int64_t _indent = 0;
  bool _pure{true};

  static SHParametersInfo parameters() {
    static Parameters params{{"Pure",
                              SHCCSTR("If the input string is generic pure json rather then "
                                      "shards flavored json."),
                              {CoreInfo::BoolType}},
                             {"Indent", SHCCSTR("How many spaces to use as json prettify indent."), {CoreInfo::IntType}}};
    return params;
  }

  void setParam(int index, const SHVar &value) {
    if (index == 0)
      _pure = value.payload.boolValue;
    else
      _indent = value.payload.intValue;
  }

  SHVar getParam(int index) {
    if (index == 0)
      return Var(_pure);
    else
      return Var(_indent);
  }
  static SHOptionalString help() { return SHCCSTR("This shard takes its input and converts it into a JSON string."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("If Pure is set to false, this shard accepts an input of any type. If Pure is set to true, this shard only "
                   "accepts standard JSON types (tables, sequences, strings, numbers, booleans and none).");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the input converted to a JSON string."); }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  void anyDump(json &j, const SHVar &input) {
    switch (input.valueType) {
    case SHType::Table: {
      std::unordered_map<std::string, json> table;
      auto &tab = input.payload.tableValue;
      ForEach(tab, [&](auto &key, auto &val) {
        json sj{};
        anyDump(sj, val);
        if (key.valueType != SHType::String)
          throw shards::ActivationError("Table keys must be strings.");
        std::string keyStr(key.payload.stringValue, key.payload.stringLen);
        table.emplace(std::move(keyStr), sj);
        return true;
      });
      j = table;
    } break;
    case SHType::Seq: {
      std::vector<json> array;
      auto &seq = input.payload.seqValue;
      for (uint32_t i = 0; i < seq.len; i++) {
        json js{};
        anyDump(js, seq.elements[i]);
        array.emplace_back(js);
      }
      j = array;
    } break;
    case SHType::String: {
      j = SHSTRVIEW(input);
    } break;
    case SHType::Int: {
      j = input.payload.intValue;
    } break;
    case SHType::Float: {
      j = input.payload.floatValue;
    } break;
    case SHType::Bool: {
      j = input.payload.boolValue;
    } break;
    case SHType::None: {
      j = nullptr;
    } break;
    default: {
      SHLOG_ERROR("Unexpected type for pure JSON conversion: {}", type2Name(input.valueType));
      throw ActivationError("Type not supported for pure JSON conversion");
    }
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!_pure) {
      json j = input;
      if (_indent == 0)
        _output = j.dump();
      else
        _output = j.dump(_indent);
    } else {
      json j{};
      anyDump(j, input);
      if (_indent == 0)
        _output = j.dump();
      else
        _output = j.dump(_indent);
    }
    return Var(_output);
  }
};

struct FromJson {
  SHVar _output{};
  bool _pure{true};

  static SHOptionalString help() {
    return SHCCSTR("This shard parses a JSON string and outputs its contents as the appropriate type.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("A JSON string to be parsed."); }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the contents of the input JSON string as the appropriate type.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() {
    static Parameters params{{"Pure",
                              SHCCSTR("If the input string is generic pure json rather then "
                                      "shards flavored json."),
                              {CoreInfo::BoolType}}};
    return params;
  }

  void setParam(int index, const SHVar &value) { _pure = value.payload.boolValue; }

  SHVar getParam(int index) { return Var(_pure); }

  void cleanup(SHContext *context) { _releaseMemory(_output); }

  void anyParse(json &j, SHVar &storage) {
    if (j.is_array()) {
      storage.valueType = SHType::Seq;
      for (json::iterator it = j.begin(); it != j.end(); ++it) {
        const auto len = storage.payload.seqValue.len;
        arrayResize(storage.payload.seqValue, len + 1);
        anyParse(*it, storage.payload.seqValue.elements[len]);
      }
    } else if (j.is_number_integer()) {
      storage.valueType = SHType::Int;
      storage.payload.intValue = j.get<int64_t>();
    } else if (j.is_number_float()) {
      storage.valueType = SHType::Float;
      storage.payload.floatValue = j.get<double>();
    } else if (j.is_string()) {
      storage.valueType = SHType::String;
      auto strVal = j.get<std::string>();
      const auto strLen = strVal.length();
      storage.payload.stringValue = new char[strLen + 1];
      storage.payload.stringLen = strLen;
      memcpy((void *)storage.payload.stringValue, strVal.c_str(), strLen);
      ((char *)storage.payload.stringValue)[strLen] = 0;
    } else if (j.is_boolean()) {
      storage.valueType = SHType::Bool;
      storage.payload.boolValue = j.get<bool>();
    } else if (j.is_object()) {
      storage.valueType = SHType::Table;
      auto map = new shards::SHMap();
      storage.payload.tableValue.api = &shards::GetGlobals().TableInterface;
      storage.payload.tableValue.opaque = map;
      for (auto &[key, value] : j.items()) {
        anyParse(value, (*map)[Var(key)]);
      }
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    _releaseMemory(_output); // release previous

    try {
      json j = json::parse(SHSTRVIEW(input));

      if (_pure) {
        anyParse(j, _output);
      } else {
        _output = j.get<SHVar>();
      }
    } catch (const json::exception &ex) {
      // re-throw with our type to allow Maybe etc
      throw ActivationError(ex.what());
    }

    return _output;
  }
};

SHARDS_REGISTER_FN(json) {
  REGISTER_SHARD("FromJson", FromJson);
  REGISTER_SHARD("ToJson", ToJson);
}
}; // namespace shards
