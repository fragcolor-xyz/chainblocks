/* SPDX-License-Identifier: BSD 3-Clause "New" or "Revised" License */
/* Copyright © 2019 Giovanni Petrantoni */

#include "nlohmann/json.hpp"
#include "shared.hpp"

using json = nlohmann::json;

void from_json(const json &j, CBChainPtr &chain);
void to_json(json &j, const CBChainPtr &chain);

void _releaseMemory(CBVar &var) {
  // Used by Block and Chain from_json
  switch (var.valueType) {
  case ContextVar:
  case String:
    delete[] var.payload.stringValue;
    break;
  case Image:
    delete[] var.payload.imageValue.data;
    break;
  case Bytes:
    delete[] var.payload.bytesValue;
    break;
  case Seq:
    for (auto i = 0; i < stbds_arrlen(var.payload.seqValue); i++) {
      _releaseMemory(var.payload.seqValue[i]);
    }
    stbds_arrfree(var.payload.seqValue);
    break;
  case Table:
    for (auto i = 0; i < stbds_shlen(var.payload.tableValue); i++) {
      _releaseMemory(var.payload.tableValue[i].value);
    }
    stbds_shfree(var.payload.tableValue);
    break;
  default:
    break;
  }
}

void to_json(json &j, const CBVar &var) {
  auto valType = int(var.valueType);
  switch (var.valueType) {
  case Any:
  case Object:
  case Chain: {
    json jchain = (CBChainPtr)var.payload.chainValue;
    j = json{{"type", valType}, {"value", jchain}};
    break;
  }
  case EndOfBlittableTypes: {
    j = json{{"type", 0}, {"value", int(Continue)}};
    break;
  }
  case None: {
    j = json{{"type", 0}, {"value", int(var.payload.chainState)}};
    break;
  }
  case Bool: {
    j = json{{"type", valType}, {"value", var.payload.boolValue}};
    break;
  }
  case Int: {
    j = json{{"type", valType}, {"value", var.payload.intValue}};
    break;
  }
  case Int2: {
    auto vec = {var.payload.int2Value[0], var.payload.int2Value[1]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Int3: {
    auto vec = {var.payload.int3Value[0], var.payload.int3Value[1],
                var.payload.int3Value[2]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Int4: {
    auto vec = {var.payload.int4Value[0], var.payload.int4Value[1],
                var.payload.int4Value[2], var.payload.int4Value[3]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Int8: {
    auto vec = {var.payload.int8Value[0], var.payload.int8Value[1],
                var.payload.int8Value[2], var.payload.int8Value[3],
                var.payload.int8Value[4], var.payload.int8Value[5],
                var.payload.int8Value[6], var.payload.int8Value[7]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Int16: {
    auto vec = {var.payload.int16Value[0],  var.payload.int16Value[1],
                var.payload.int16Value[2],  var.payload.int16Value[3],
                var.payload.int16Value[4],  var.payload.int16Value[5],
                var.payload.int16Value[6],  var.payload.int16Value[7],
                var.payload.int16Value[8],  var.payload.int16Value[9],
                var.payload.int16Value[10], var.payload.int16Value[11],
                var.payload.int16Value[12], var.payload.int16Value[13],
                var.payload.int16Value[14], var.payload.int16Value[15]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Float: {
    j = json{{"type", valType}, {"value", var.payload.floatValue}};
    break;
  }
  case Float2: {
    auto vec = {var.payload.float2Value[0], var.payload.float2Value[1]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Float3: {
    auto vec = {var.payload.float3Value[0], var.payload.float3Value[1],
                var.payload.float3Value[2]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case Float4: {
    auto vec = {var.payload.float4Value[0], var.payload.float4Value[1],
                var.payload.float4Value[2], var.payload.float4Value[3]};
    j = json{{"type", valType}, {"value", vec}};
    break;
  }
  case ContextVar:
  case String: {
    j = json{{"type", valType}, {"value", var.payload.stringValue}};
    break;
  }
  case Color: {
    j = json{{"type", valType},
             {"value",
              {var.payload.colorValue.r, var.payload.colorValue.g,
               var.payload.colorValue.b, var.payload.colorValue.a}}};
    break;
  }
  case Image: {
    if (var.payload.imageValue.data) {
      auto binsize = var.payload.imageValue.width *
                     var.payload.imageValue.height *
                     var.payload.imageValue.channels;
      std::vector<uint8_t> buffer;
      buffer.resize(binsize);
      memcpy(&buffer[0], var.payload.imageValue.data, binsize);
      j = json{{"type", valType},
               {"width", var.payload.imageValue.width},
               {"height", var.payload.imageValue.height},
               {"channels", var.payload.imageValue.channels},
               {"data", buffer}};
    } else {
      j = json{{"type", 0}, {"value", int(Continue)}};
    }
    break;
  }
  case Bytes: {
    std::vector<uint8_t> buffer;
    buffer.resize(var.payload.bytesSize);
    if (var.payload.bytesSize > 0)
      memcpy(&buffer[0], var.payload.imageValue.data, var.payload.bytesSize);
    j = json{{"type", valType}, {"data", buffer}};
    break;
  }
  case Enum: {
    j = json{{"type", valType},
             {"value", int32_t(var.payload.enumValue)},
             {"vendorId", var.payload.enumVendorId},
             {"typeId", var.payload.enumTypeId}};
    break;
  }
  case Seq: {
    std::vector<json> items;
    for (int i = 0; i < stbds_arrlen(var.payload.seqValue); i++) {
      auto &v = var.payload.seqValue[i];
      items.emplace_back(v);
    }
    j = json{{"type", valType}, {"values", items}};
    break;
  }
  case Table: {
    std::vector<json> items;
    for (int i = 0; i < stbds_shlen(var.payload.tableValue); i++) {
      auto &v = var.payload.tableValue[i];
      items.push_back(json{{"key", v.key}, {"value", v.value}});
    }
    j = json{{"type", valType}, {"values", items}};
    break;
  }
  case Block: {
    auto blk = var.payload.blockValue;
    std::vector<json> params;
    auto paramsDesc = blk->parameters(blk);
    for (int i = 0; i < stbds_arrlen(paramsDesc); i++) {
      auto &desc = paramsDesc[i];
      auto value = blk->getParam(blk, i);

      json param_obj = {{"name", desc.name}, {"value", value}};

      params.push_back(param_obj);
    }

    j = json{{"type", valType}, {"name", blk->name(blk)}, {"params", params}};
    break;
  }
  };
}

void from_json(const json &j, CBVar &var) {
  auto valType = CBType(j.at("type").get<int>());
  switch (valType) {
  case Any:
  case Object:
  case Chain: {
    var.valueType = Chain;
    var.payload.chainValue = j.at("value").get<CBChainPtr>();
    break;
  }
  case EndOfBlittableTypes: {
    var = {};
    break;
  }
  case None: {
    var.valueType = None;
    var.payload.chainState = CBChainState(j.at("value").get<int>());
    break;
  }
  case Bool: {
    var.valueType = Bool;
    var.payload.boolValue = j.at("value").get<bool>();
    break;
  }
  case Int: {
    var.valueType = Int;
    var.payload.intValue = j.at("value").get<int64_t>();
    break;
  }
  case Int2: {
    var.valueType = Int2;
    var.payload.int2Value[0] = j.at("value")[0].get<int64_t>();
    var.payload.int2Value[1] = j.at("value")[1].get<int64_t>();
    break;
  }
  case Int3: {
    var.valueType = Int3;
    var.payload.int3Value[0] = j.at("value")[0].get<int32_t>();
    var.payload.int3Value[1] = j.at("value")[1].get<int32_t>();
    var.payload.int3Value[2] = j.at("value")[2].get<int32_t>();
    break;
  }
  case Int4: {
    var.valueType = Int4;
    var.payload.int4Value[0] = j.at("value")[0].get<int32_t>();
    var.payload.int4Value[1] = j.at("value")[1].get<int32_t>();
    var.payload.int4Value[2] = j.at("value")[2].get<int32_t>();
    var.payload.int4Value[3] = j.at("value")[3].get<int32_t>();
    break;
  }
  case Int8: {
    var.valueType = Int8;
    for (auto i = 0; i < 8; i++) {
      var.payload.int8Value[i] = j.at("value")[i].get<int16_t>();
    }
    break;
  }
  case Int16: {
    var.valueType = Int16;
    for (auto i = 0; i < 16; i++) {
      var.payload.int16Value[i] = j.at("value")[i].get<int8_t>();
    }
    break;
  }
  case Float: {
    var.valueType = Float;
    var.payload.floatValue = j.at("value").get<double>();
    break;
  }
  case Float2: {
    var.valueType = Float2;
    var.payload.float2Value[0] = j.at("value")[0].get<double>();
    var.payload.float2Value[1] = j.at("value")[1].get<double>();
    break;
  }
  case Float3: {
    var.valueType = Float3;
    var.payload.float3Value[0] = j.at("value")[0].get<float>();
    var.payload.float3Value[1] = j.at("value")[1].get<float>();
    var.payload.float3Value[2] = j.at("value")[2].get<float>();
    break;
  }
  case Float4: {
    var.valueType = Float4;
    var.payload.float4Value[0] = j.at("value")[0].get<float>();
    var.payload.float4Value[1] = j.at("value")[1].get<float>();
    var.payload.float4Value[2] = j.at("value")[2].get<float>();
    var.payload.float4Value[3] = j.at("value")[3].get<float>();
    break;
  }
  case ContextVar: {
    var.valueType = ContextVar;
    auto strVal = j.at("value").get<std::string>();
    var.payload.stringValue = new char[strVal.length() + 1];
    memset((void *)var.payload.stringValue, 0x0, strVal.length() + 1);
    memcpy((void *)var.payload.stringValue, strVal.c_str(), strVal.length());
    break;
  }
  case String: {
    var.valueType = String;
    auto strVal = j.at("value").get<std::string>();
    var.payload.stringValue = new char[strVal.length() + 1];
    memset((void *)var.payload.stringValue, 0x0, strVal.length() + 1);
    memcpy((void *)var.payload.stringValue, strVal.c_str(), strVal.length());
    break;
  }
  case Color: {
    var.valueType = Color;
    var.payload.colorValue.r = j.at("value")[0].get<uint8_t>();
    var.payload.colorValue.g = j.at("value")[1].get<uint8_t>();
    var.payload.colorValue.b = j.at("value")[2].get<uint8_t>();
    var.payload.colorValue.a = j.at("value")[3].get<uint8_t>();
    break;
  }
  case Image: {
    var.valueType = Image;
    var.payload.imageValue.width = j.at("width").get<int32_t>();
    var.payload.imageValue.height = j.at("height").get<int32_t>();
    var.payload.imageValue.channels = j.at("channels").get<int32_t>();
    auto binsize = var.payload.imageValue.width *
                   var.payload.imageValue.height *
                   var.payload.imageValue.channels;
    var.payload.imageValue.data = new uint8_t[binsize];
    auto buffer = j.at("data").get<std::vector<uint8_t>>();
    memcpy(var.payload.imageValue.data, &buffer[0], binsize);
    break;
  }
  case Bytes: {
    var.valueType = Bytes;
    auto buffer = j.at("data").get<std::vector<uint8_t>>();
    var.payload.bytesValue = new uint8_t[buffer.size()];
    memcpy(var.payload.bytesValue, &buffer[0], buffer.size());
    break;
  }
  case Enum: {
    var.valueType = Enum;
    var.payload.enumValue = CBEnum(j.at("value").get<int32_t>());
    var.payload.enumVendorId = CBEnum(j.at("vendorId").get<int32_t>());
    var.payload.enumTypeId = CBEnum(j.at("typeId").get<int32_t>());
    break;
  }
  case Seq: {
    var.valueType = Seq;
    auto items = j.at("values").get<std::vector<json>>();
    var.payload.seqValue = nullptr;
    for (const auto &item : items) {
      stbds_arrpush(var.payload.seqValue, item.get<CBVar>());
    }
    break;
  }
  case Table: {
    var.valueType = Seq;
    auto items = j.at("values").get<std::vector<json>>();
    var.payload.tableValue = nullptr;
    stbds_sh_new_arena(var.payload.tableValue);
    for (const auto &item : items) {
      auto key = item.at("key").get<std::string>();
      auto value = item.at("value").get<CBVar>();
      stbds_shput(var.payload.tableValue, key.c_str(), value);
    }
    break;
  }
  case Block: {
    var.valueType = Block;
    auto blkname = j.at("name").get<std::string>();
    auto blk = chainblocks::createBlock(blkname.c_str());
    if (!blk) {
      auto errmsg = "Failed to create block of type: " + std::string("blkname");
      throw chainblocks::CBException(errmsg.c_str());
    }
    var.payload.blockValue = blk;

    // Setup
    blk->setup(blk);

    // Set params
    auto jparams = j.at("params");
    auto blkParams = blk->parameters(blk);
    for (auto jparam : jparams) {
      auto paramName = jparam.at("name").get<std::string>();
      auto value = jparam.at("value").get<CBVar>();

      if (value.valueType != None) {
        for (auto i = 0; stbds_arrlen(blkParams) > i; i++) {
          auto &paramInfo = blkParams[i];
          if (paramName == paramInfo.name) {
            blk->setParam(blk, i, value);
            break;
          }
        }
      }

      // Assume block copied memory internally so we can clean up here!!!
      _releaseMemory(value);
    }
    break;
  }
  }
}

void to_json(json &j, const CBChainPtr &chain) {
  std::vector<json> blocks;
  for (auto blk : chain->blocks) {
    std::vector<json> params;
    auto paramsDesc = blk->parameters(blk);
    for (int i = 0; stbds_arrlen(paramsDesc) > i; i++) {
      auto &desc = paramsDesc[i];
      auto value = blk->getParam(blk, i);

      json param_obj = {{"name", desc.name}, {"value", value}};

      params.push_back(param_obj);
    }

    json block_obj = {{"name", blk->name(blk)}, {"params", params}};

    blocks.push_back(block_obj);
  }

  j = {
      {"blocks", blocks},        {"name", chain->name},
      {"looped", chain->looped}, {"unsafe", chain->unsafe},
      {"version", 0.1},
  };
}

void from_json(const json &j, CBChainPtr &chain) {
  auto chainName = j.at("name").get<std::string>();
  auto findIt = chainblocks::GlobalChains.find(chainName);
  if (findIt != chainblocks::GlobalChains.end()) {
    chain = findIt->second;
    // Need to clean it up for rewrite!
    chain->cleanup();
  } else {
    chain = new CBChain(chainName.c_str());
    chainblocks::GlobalChains[chainName] = chain;
  }

  chain->looped = j.at("looped").get<bool>();
  chain->unsafe = j.at("unsafe").get<bool>();

  auto jblocks = j.at("blocks");
  for (auto jblock : jblocks) {
    auto blkname = jblock.at("name").get<std::string>();
    auto blk = chainblocks::createBlock(blkname.c_str());
    if (!blk) {
      auto errmsg = "Failed to create block of type: " + std::string(blkname);
      throw chainblocks::CBException(errmsg.c_str());
    }

    // Setup
    blk->setup(blk);

    // Set params
    auto jparams = jblock.at("params");
    auto blkParams = blk->parameters(blk);
    for (auto jparam : jparams) {
      auto paramName = jparam.at("name").get<std::string>();
      auto value = jparam.at("value").get<CBVar>();

      if (value.valueType != None) {
        for (auto i = 0; stbds_arrlen(blkParams) > i; i++) {
          auto &paramInfo = blkParams[i];
          if (paramName == paramInfo.name) {
            blk->setParam(blk, i, value);
            break;
          }
        }
      }

      // Assume block copied memory internally so we can clean up here!!!
      _releaseMemory(value);
    }

    // From now on this chain owns the block
    chain->addBlock(blk);
  }
}

namespace chainblocks {
struct ToJson {
  std::string _output;
  static CBTypesInfo inputTypes() { return CBTypesInfo(SharedTypes::anyInfo); }
  static CBTypesInfo outputTypes() { return CBTypesInfo(SharedTypes::anyInfo); }
  CBVar activate(CBContext *context, const CBVar &input) {
    json j = input;
    _output = j.dump();
    return Var(_output);
  }
};

struct FromJson {
  CBVar _output;
  static CBTypesInfo inputTypes() { return CBTypesInfo(SharedTypes::strInfo); }
  static CBTypesInfo outputTypes() { return CBTypesInfo(SharedTypes::anyInfo); }
  CBVar activate(CBContext *context, const CBVar &input) {
    _releaseMemory(_output); // release previous
    json j = json::parse(input.payload.stringValue);
    _output = j.get<CBVar>();
    return _output;
  }
};

RUNTIME_CORE_BLOCK(ToJson);
RUNTIME_BLOCK_inputTypes(ToJson);
RUNTIME_BLOCK_outputTypes(ToJson);
RUNTIME_BLOCK_activate(ToJson);
RUNTIME_BLOCK_END(ToJson);

RUNTIME_CORE_BLOCK(FromJson);
RUNTIME_BLOCK_inputTypes(FromJson);
RUNTIME_BLOCK_outputTypes(FromJson);
RUNTIME_BLOCK_activate(FromJson);
RUNTIME_BLOCK_END(FromJson);

void registerJsonBlocks() {
  REGISTER_CORE_BLOCK(ToJson);
  REGISTER_CORE_BLOCK(FromJson);
}
}; // namespace chainblocks
