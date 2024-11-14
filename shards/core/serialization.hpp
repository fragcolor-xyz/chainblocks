#ifndef F5B6AA09_CDF7_4DFD_B508_854541459119
#define F5B6AA09_CDF7_4DFD_B508_854541459119

#include "runtime.hpp"
#include "trait.hpp"
#include <linalg.h>
#include <stdexcept>
#include <cassert>
#include <string_view>
#include <unordered_map>
#include <tracy/Wrapper.hpp>
#include "pmr/vector.hpp"
#include "pmr/unordered_map.hpp"

// Enable for more detailed breakdown of what types are being (de)serialized in tracy
#define SH_SERIALIZATION_TRACY_DETAILED 0

namespace shards {

struct BufferRefWriter {
  std::vector<uint8_t> &_buffer;
  BufferRefWriter(std::vector<uint8_t> &stream, bool clear = true) : _buffer(stream) {
    if (clear) {
      _buffer.clear();
    }
  }
  void operator()(const uint8_t *buf, size_t size) { _buffer.insert(_buffer.end(), buf, buf + size); }

  // expose data and size
  const uint8_t *data() { return _buffer.data(); }
  size_t size() { return _buffer.size(); }
};

struct BufferWriter {
  std::vector<uint8_t> _buffer;
  BufferRefWriter _inner;
  BufferWriter() : _inner(_buffer) {}
  void operator()(const uint8_t *buf, size_t size) { _inner(buf, size); }
};

struct BytesReader {
  const uint8_t *buffer;
  size_t offset;
  size_t max;

  BytesReader(std::string_view sv) : buffer((uint8_t *)sv.data()), offset(0), max(sv.size()) {}
  BytesReader(const uint8_t *buf, size_t size) : buffer(buf), offset(0), max(size) {}
  void operator()(uint8_t *buf, size_t size) {
    auto newOffset = offset + size;
    if (newOffset > max) {
      throw std::runtime_error("Overflow requested");
    }
    memcpy(buf, buffer + offset, size);
    offset = newOffset;
  }
};

struct BufferRefReader {
  BytesReader _inner;
  BufferRefReader(const std::vector<uint8_t> &stream) : _inner(stream.data(), stream.size()) {}
  void operator()(uint8_t *buf, size_t size) { _inner(buf, size); }
};

struct VarReader {
  BytesReader _inner;
  VarReader(const SHVar &var) : _inner(var.payload.bytesValue, var.payload.bytesSize) { assert(var.valueType == SHType::Bytes); }
  void operator()(uint8_t *buf, size_t size) { _inner(buf, size); }
};

template <typename T, typename V>
std::enable_if_t<std::is_integral_v<V> || std::is_floating_point_v<V> || std::is_same_v<V, bool>, void> //
serde(T &stream, V &v) {
  stream((uint8_t *)&v, sizeof(V));
}
template <typename T, typename V, int M> void serde(T &stream, linalg::vec<V, M> &v) {
  for (int i = 0; i < M; i++)
    serde(stream, v[i]);
}
template <typename T, typename V, int N, int M> void serde(T &stream, linalg::mat<V, M, N> &v) {
  for (int m = 0; m < N; m++) {
    serde(stream, v[m]);
  }
}
template <typename T> void serde(T &stream, std::monostate &v) {}
template <typename T, typename V> void serdeConst(T &stream, const V &v) { serde(stream, const_cast<V &>(v)); }
template <typename T, typename As, typename E> std::enable_if_t<std::is_enum_v<E>> serdeAs(T &stream, E &v) {
  As a = As(v);
  serde(stream, a);
  v = E(a);
}
struct Serialization {
  Serialization() = default;
  Serialization(bool private_internal) : private_internal(private_internal) {}
  Serialization(pmr::PolymorphicAllocator<> tempAllocator, bool private_internal = false)
      : tempAllocator(tempAllocator), private_internal(private_internal), wires(tempAllocator) {}

  pmr::PolymorphicAllocator<> tempAllocator;
  bool private_internal{false}; // turn this flag on when you want to serialize private data (like wire pointers)

  pmr::unordered_map<SHVar, SHWireRef> wires;

  void reset() {
    for (auto &ref : wires) {
      SHWire::deleteRef(ref.second);
    }
    wires.clear();
  }

  ~Serialization() { reset(); }

  template <class BinaryReader> void deserialize(BinaryReader &read, SHStringPayload &output, bool recycle) {
    auto availChars = recycle ? output.cap : 0;
    read((uint8_t *)&output.len, sizeof(uint32_t));

#if 0
      if (output.payload.len <= 7) {
        // small string, just use the stack
        output.payload.stringValue = output.shortString;
        output.payload.cap = 7;
      } else
#endif
    {
      if (availChars > 0 && availChars < output.len) {
        // we need more chars then what we have, realloc
        delete[] output.elements;
        output.elements = new char[output.len + 1];
      } else if (availChars == 0) {
        // just alloc
        output.elements = new char[output.len + 1];
      } // else recycling

      // record actualSize
      output.cap = std::max(availChars, output.len);
    }

    read((uint8_t *)output.elements, output.len);
    const_cast<char *>(output.elements)[output.len] = 0;
  }

  template <class BinaryWriter> size_t serialize(const SHStringPayload &input, BinaryWriter &write) {
    size_t total{};
    uint32_t len = input.len > 0 || input.elements == nullptr ? input.len : uint32_t(strlen(input.elements));
    write((const uint8_t *)&len, sizeof(uint32_t));
    total += sizeof(uint32_t);
    write((const uint8_t *)input.elements, len);
    total += len;
    return total;
  }

  template <class BinaryReader> void deserialize(BinaryReader &read, SHVar &output) {
    // we try to recycle memory so pass a empty None as first timer!
    SHType nextType;
    read((uint8_t *)&nextType, sizeof(output.valueType));

    // stop trying to recycle, types differ
    auto recycle = true;
    if (output.valueType != nextType) {
      destroyVar(output);
      recycle = false;
    }

    output.valueType = nextType;

#if SH_SERIALIZATION_TRACY_DETAILED
    ZoneScopedN("deserialize");
    const char *c = type2Name_raw(nextType);
    ZoneText(c, strlen(c));
#endif

    switch (output.valueType) {
    case SHType::None:
      break;
    case SHType::Enum:
      read((uint8_t *)&output.payload, sizeof(int32_t) * 3);
      break;
    case SHType::Bool:
      read((uint8_t *)&output.payload, sizeof(SHBool));
      break;
    case SHType::Int:
      read((uint8_t *)&output.payload, sizeof(SHInt));
      break;
    case SHType::Int2:
      read((uint8_t *)&output.payload, sizeof(SHInt2));
      break;
    case SHType::Int3:
      read((uint8_t *)&output.payload, sizeof(SHInt3));
      break;
    case SHType::Int4:
      read((uint8_t *)&output.payload, sizeof(SHInt4));
      break;
    case SHType::Int8:
      read((uint8_t *)&output.payload, sizeof(SHInt8));
      break;
    case SHType::Int16:
      read((uint8_t *)&output.payload, sizeof(SHInt16));
      break;
    case SHType::Float:
      read((uint8_t *)&output.payload, sizeof(SHFloat));
      break;
    case SHType::Float2:
      read((uint8_t *)&output.payload, sizeof(SHFloat2));
      break;
    case SHType::Float3:
      read((uint8_t *)&output.payload, sizeof(SHFloat3));
      break;
    case SHType::Float4:
      read((uint8_t *)&output.payload, sizeof(SHFloat4));
      break;
    case SHType::Color:
      read((uint8_t *)&output.payload, sizeof(SHColor));
      break;
    case SHType::Bytes: {
      auto availBytes = recycle ? output.payload.bytesCapacity : 0;
      read((uint8_t *)&output.payload.bytesSize, sizeof(output.payload.bytesSize));

#if 0
      if (output.payload.bytesSize <= 8) {
        // short array
        output.payload.bytesValue = output.shortBytes;
        output.payload.bytesCapacity = 8;
      } else
#endif
      {
        if (availBytes > 0 && availBytes < output.payload.bytesSize) {
          // not enough space, ideally realloc, but for now just delete
          delete[] output.payload.bytesValue;
          // and re alloc
          output.payload.bytesValue = new uint8_t[output.payload.bytesSize];
        } else if (availBytes == 0) {
          // just alloc
          output.payload.bytesValue = new uint8_t[output.payload.bytesSize];
        } // else got enough space to recycle!

        // record actualSize for further recycling usage
        output.payload.bytesCapacity = std::max(availBytes, output.payload.bytesSize);
      }

      read((uint8_t *)output.payload.bytesValue, output.payload.bytesSize);
      break;
    }
    case SHType::String:
    case SHType::Path:
    case SHType::ContextVar: {
      deserialize(read, output.payload.string, recycle);
      break;
    }
    case SHType::Seq: {
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));
      // notice we assume all elements up to capacity are memset to 0x0
      // or are valid SHVars we can overwrite
      shards::arrayResize(output.payload.seqValue, len);
      for (uint32_t i = 0; i < len; i++) {
        deserialize(read, output.payload.seqValue.elements[i]);
      }
      break;
    }
    case SHType::Table: {
      SHMap *map = nullptr;

      if (recycle) {
        if (output.payload.tableValue.api && output.payload.tableValue.opaque) {
          map = (SHMap *)output.payload.tableValue.opaque;
          map->clear();
        } else {
          destroyVar(output);
          output.valueType = nextType;
        }
      }

      if (!map) {
        map = new SHMap();
        output.payload.tableValue.api = &GetGlobals().TableInterface;
        output.payload.tableValue.opaque = map;
      }

      uint64_t len;
      read((uint8_t *)&len, sizeof(uint64_t));
      for (uint64_t i = 0; i < len; i++) {
        shards::OwnedVar keyBuf{};
        deserialize(read, keyBuf);
        auto &dst = (*map)[std::move(keyBuf)];
        deserialize(read, dst);
      }
      break;
    }
    case SHType::Image: {
      SHImage tmpImage{};
      read((uint8_t *)&tmpImage.channels, sizeof(tmpImage.channels));
      read((uint8_t *)&tmpImage.flags, sizeof(tmpImage.flags));
      read((uint8_t *)&tmpImage.width, sizeof(tmpImage.width));
      read((uint8_t *)&tmpImage.height, sizeof(tmpImage.height));

      size_t imageDataLen = imageDeriveDataLength(&tmpImage);

      output.valueType = SHType::Image;
      output.payload.imageValue = imageNew(imageDataLen);
      SHImage &image = *output.payload.imageValue;
      image.channels = tmpImage.channels;
      image.flags = tmpImage.flags;
      image.width = tmpImage.width;
      image.height = tmpImage.height;

      read((uint8_t *)image.data, imageDataLen);
      break;
    }
    case SHType::Audio: {
      size_t currentSize = 0;
      if (recycle) {
        currentSize = output.payload.audioValue.nsamples * output.payload.audioValue.channels * sizeof(float);
      }

      read((uint8_t *)&output.payload.audioValue.nsamples, sizeof(output.payload.audioValue.nsamples));
      read((uint8_t *)&output.payload.audioValue.channels, sizeof(output.payload.audioValue.channels));
      read((uint8_t *)&output.payload.audioValue.sampleRate, sizeof(output.payload.audioValue.sampleRate));

      size_t size = output.payload.audioValue.nsamples * output.payload.audioValue.channels * sizeof(float);

      if (currentSize > 0 && currentSize < size) {
        // delete first & alloc
        delete[] output.payload.audioValue.samples;
        output.payload.audioValue.samples = new float[output.payload.audioValue.nsamples * output.payload.audioValue.channels];
      } else if (currentSize == 0) {
        // just alloc
        output.payload.audioValue.samples = new float[output.payload.audioValue.nsamples * output.payload.audioValue.channels];
      }

      read((uint8_t *)output.payload.audioValue.samples, size);
      break;
    }
    case SHType::ShardRef: {
      Shard *blk;
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));
      pmr::vector<char> buf(tempAllocator);
      buf.resize(len + 1);
      read((uint8_t *)&buf[0], len);
      buf[len] = 0;
      blk = createShard(&buf[0]);
      if (!blk) {
        throw shards::SHException("Shard not found! name: " + std::string(&buf[0]));
      }

      // validate the hash of the shard
      uint32_t crc;
      read((uint8_t *)&crc, sizeof(uint32_t));
      if (blk->hash(blk) != crc) {
        throw shards::SHException("Shard hash mismatch, the serialized version is "
                                  "probably different: " +
                                  std::string(&buf[0]));
      }

      blk->setup(blk);

      auto params = blk->parameters(blk).len;
      for (uint32_t i = 0; i < params; i++) {
        SHVar tmp{};
        deserialize(read, tmp);
        blk->setParam(blk, i, &tmp);
        destroyVar(tmp);
      }

      if (blk->setState) {
        SHVar state{};
        deserialize(read, state);
        blk->setState(blk, &state);
        destroyVar(state);
      }

      if (private_internal) {
        // also get line and column
        read((uint8_t *)&blk->line, sizeof(uint32_t));
        read((uint8_t *)&blk->column, sizeof(uint32_t));
        // read shard id
        read((uint8_t *)&blk->id, sizeof(uint64_t));
      }

      incRef(blk);
      output.payload.shardValue = blk;
      break;
    }
    case SHType::Wire: {
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));
      pmr::vector<char> buf(tempAllocator);
      buf.resize(len + 1);
      read((uint8_t *)&buf[0], len);
      buf[len] = 0;

      SHVar hash{};
      deserialize(read, hash);

      // search if we already have this wire!
      auto cit = wires.find(hash);
      if (cit != wires.end()) {
        SHLOG_TRACE("Skipping deserializing wire: {}", SHWire::sharedFromRef(cit->second)->name);
        output.payload.wireValue = SHWire::addRef(cit->second);
        break;
      }

      auto wire = SHWire::make(&buf[0]);
      output.payload.wireValue = wire->newRef();
      wires.emplace(hash, SHWire::addRef(output.payload.wireValue));
      SHLOG_TRACE("Deserializing wire: {}", wire->name);
      read((uint8_t *)&wire->looped, 1);
      read((uint8_t *)&wire->unsafe, 1);
      read((uint8_t *)&wire->pure, 1);
      // shards len
      read((uint8_t *)&len, sizeof(uint32_t));
      // shards
      for (uint32_t i = 0; i < len; i++) {
        SHVar shardVar{};
        deserialize(read, shardVar);
        ShardPtr shard = shardVar.payload.shardValue;
        if (shardVar.valueType != SHType::ShardRef)
          throw shards::SHException("Expected a shard ref!");
        wire->addShard(shardVar.payload.shardValue);
        // shard's owner is now the wire, remove original reference from deserialize
        decRef(shard);
      }
      // traits len
      read((uint8_t *)&len, sizeof(uint32_t));
      // traits
      for (uint32_t i = 0; i < len; i++) {
        SHTrait trait{};
        deserialize(read, trait);
        wire->addTrait(trait);
        freeTrait(trait);
      }

      if (private_internal) {
        // read the pointer address of the parent wire
        size_t parentAddress;
        read((uint8_t *)&parentAddress, sizeof(size_t));
        auto parentWire = reinterpret_cast<SHWire *>(parentAddress);
        if (parentWire) {
          wire->parent = parentWire->shared_from_this();
        }
      }
      break;
    }
    case SHType::Object: {
      int64_t id;
      read((uint8_t *)&id, sizeof(int64_t));
      uint64_t len;
      read((uint8_t *)&len, sizeof(uint64_t));
      if (len > 0) {
        auto it = GetGlobals().ObjectTypesRegister.find(id);
        if (it != GetGlobals().ObjectTypesRegister.end()) {
          auto &info = it->second;
          if (!info.deserialize) {
            throw shards::SHException("Object variable deserializer missing");
          }
          auto size = size_t(len);
          pmr::vector<uint8_t> data(tempAllocator);
          data.resize(size);
          read((uint8_t *)&data[0], size);
          output.payload.objectValue = info.deserialize(&data[0], size);
          int32_t vendorId = (int32_t)((id & 0xFFFFFFFF00000000) >> 32);
          int32_t typeId = (int32_t)(id & 0x00000000FFFFFFFF);
          output.payload.objectVendorId = vendorId;
          output.payload.objectTypeId = typeId;
          output.flags |= SHVAR_FLAGS_USES_OBJINFO;
          output.objectInfo = &info;
          // up ref count also, not included in deserialize op!
          if (info.reference) {
            info.reference(output.payload.objectValue);
          }
        } else {
          throw shards::SHException("Failed to find object type in registry.");
        }
      } else {
        // set this var to None
        output.valueType = SHType::None;
      }
      break;
    }
    case SHType::Type:
      if (output.payload.typeValue)
        freeDerivedInfo(*output.payload.typeValue);
      else
        output.payload.typeValue = new SHTypeInfo{};
      deserialize(read, *output.payload.typeValue);
      break;

    case SHType::Trait:
      if (output.payload.traitValue)
        freeTrait(*output.payload.traitValue);
      else
        output.payload.traitValue = new SHTrait{};

      deserialize(read, *output.payload.traitValue);
      break;
    default:
      throw shards::SHException("Unknown type during deserialization!");
    }
  }

  template <class BinaryWriter> size_t serialize(const SHVar &input, BinaryWriter &write) {
    size_t total = 0;
    write((const uint8_t *)&input.valueType, sizeof(input.valueType));
    total += sizeof(input.valueType);

#if SH_SERIALIZATION_TRACY_DETAILED
    ZoneScopedN("serialize");
    const char *c = type2Name_raw(input.valueType);
    ZoneText(c, strlen(c));
#endif

    switch (input.valueType) {
    case SHType::None:
      break;
    case SHType::Enum:
      write((const uint8_t *)&input.payload, sizeof(int32_t) * 3);
      total += sizeof(int32_t) * 3;
      break;
    case SHType::Bool:
      write((const uint8_t *)&input.payload, sizeof(SHBool));
      total += sizeof(SHBool);
      break;
    case SHType::Int:
      write((const uint8_t *)&input.payload, sizeof(SHInt));
      total += sizeof(SHInt);
      break;
    case SHType::Int2:
      write((const uint8_t *)&input.payload, sizeof(SHInt2));
      total += sizeof(SHInt2);
      break;
    case SHType::Int3:
      write((const uint8_t *)&input.payload, sizeof(SHInt3));
      total += sizeof(SHInt3);
      break;
    case SHType::Int4:
      write((const uint8_t *)&input.payload, sizeof(SHInt4));
      total += sizeof(SHInt4);
      break;
    case SHType::Int8:
      write((const uint8_t *)&input.payload, sizeof(SHInt8));
      total += sizeof(SHInt8);
      break;
    case SHType::Int16:
      write((const uint8_t *)&input.payload, sizeof(SHInt16));
      total += sizeof(SHInt16);
      break;
    case SHType::Float:
      write((const uint8_t *)&input.payload, sizeof(SHFloat));
      total += sizeof(SHFloat);
      break;
    case SHType::Float2:
      write((const uint8_t *)&input.payload, sizeof(SHFloat2));
      total += sizeof(SHFloat2);
      break;
    case SHType::Float3:
      write((const uint8_t *)&input.payload, sizeof(SHFloat3));
      total += sizeof(SHFloat3);
      break;
    case SHType::Float4:
      write((const uint8_t *)&input.payload, sizeof(SHFloat4));
      total += sizeof(SHFloat4);
      break;
    case SHType::Color:
      write((const uint8_t *)&input.payload, sizeof(SHColor));
      total += sizeof(SHColor);
      break;
    case SHType::Bytes:
      write((const uint8_t *)&input.payload.bytesSize, sizeof(input.payload.bytesSize));
      total += sizeof(input.payload.bytesSize);
      write((const uint8_t *)input.payload.bytesValue, input.payload.bytesSize);
      total += input.payload.bytesSize;
      break;
    case SHType::Path:
    case SHType::String:
    case SHType::ContextVar: {
      total += serialize(input.payload.string, write);
      break;
    }
    case SHType::Seq: {
      uint32_t len = input.payload.seqValue.len;
      write((const uint8_t *)&len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      for (uint32_t i = 0; i < len; i++) {
        total += serialize(input.payload.seqValue.elements[i], write);
      }
      break;
    }
    case SHType::Table: {
      if (input.payload.tableValue.api && input.payload.tableValue.opaque) {
        auto &t = input.payload.tableValue;
        uint64_t len = (uint64_t)t.api->tableSize(t);
        write((const uint8_t *)&len, sizeof(uint64_t));
        total += sizeof(uint64_t);
        SHTableIterator tit;
        t.api->tableGetIterator(t, &tit);
        SHVar k;
        SHVar v;
        while (t.api->tableNext(t, &tit, &k, &v)) {
          total += serialize(k, write);
          total += serialize(v, write);
        }
      } else {
        uint64_t none = 0;
        write((const uint8_t *)&none, sizeof(uint64_t));
        total += sizeof(uint64_t);
      }
      break;
    }
    case SHType::Image: {
      SHImage &image = *input.payload.imageValue;
      write((const uint8_t *)&image.channels, sizeof(image.channels));
      total += sizeof(image.channels);
      write((const uint8_t *)&image.flags, sizeof(image.flags));
      total += sizeof(image.flags);
      write((const uint8_t *)&image.width, sizeof(image.width));
      total += sizeof(image.width);
      write((const uint8_t *)&image.height, sizeof(image.height));
      total += sizeof(image.height);
      auto imageDataLength = imageDeriveDataLength(&image);
      write((const uint8_t *)image.data, imageDataLength);
      total += imageDataLength;
      break;
    }
    case SHType::Audio: {
      write((const uint8_t *)&input.payload.audioValue.nsamples, sizeof(input.payload.audioValue.nsamples));
      total += sizeof(input.payload.audioValue.nsamples);

      write((const uint8_t *)&input.payload.audioValue.channels, sizeof(input.payload.audioValue.channels));
      total += sizeof(input.payload.audioValue.channels);

      write((const uint8_t *)&input.payload.audioValue.sampleRate, sizeof(input.payload.audioValue.sampleRate));
      total += sizeof(input.payload.audioValue.sampleRate);

      auto size = input.payload.audioValue.nsamples * input.payload.audioValue.channels * sizeof(float);

      write((const uint8_t *)input.payload.audioValue.samples, size);
      total += size;
      break;
    }
    case SHType::ShardRef: {
      auto blk = input.payload.shardValue;
      // name
      auto name = blk->name(blk);
      uint32_t len = uint32_t(strlen(name));
      write((const uint8_t *)&len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      write((const uint8_t *)name, len);
      total += len;

      // serialize the hash of the shard as well
      auto crc = blk->hash(blk);
      write((const uint8_t *)&crc, sizeof(uint32_t));
      total += sizeof(uint32_t);

      // params
      auto params = blk->parameters(blk);
      for (uint32_t i = 0; i < params.len; i++) {
        auto idx = int32_t(i);
        auto pval = blk->getParam(blk, idx);
        total += serialize(pval, write);
      }

      // optional state
      if (blk->getState) {
        auto state = blk->getState(blk);
        total += serialize(state, write);
      }

      if (private_internal) {
        // line and column
        write((const uint8_t *)&blk->line, sizeof(uint32_t));
        total += sizeof(uint32_t);
        write((const uint8_t *)&blk->column, sizeof(uint32_t));
        total += sizeof(uint32_t);
        // serialize shard id
        write((const uint8_t *)&blk->id, sizeof(uint64_t));
        total += sizeof(uint64_t);
      }
      break;
    }
    case SHType::Wire: {
      auto sc = SHWire::sharedFromRef(input.payload.wireValue);
      auto wire = sc.get();

      { // Name
        uint32_t len = uint32_t(wire->name.size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
        write((const uint8_t *)wire->name.c_str(), len);
        total += len;
      }

      SHVar hash;
      { // Hash
        hash = shards::hash(input);
        total += serialize(hash, write);
      }

      // stop here if we had it already
      if (wires.count(hash) > 0) {
        SHLOG_TRACE("Skipping serializing wire: {}", wire->name);
        break;
      }

      SHLOG_TRACE("Serializing wire: {}", wire->name);
      wires.emplace(hash, SHWire::addRef(input.payload.wireValue));

      { // Looped & Unsafe
        write((const uint8_t *)&wire->looped, 1);
        total += 1;
        write((const uint8_t *)&wire->unsafe, 1);
        total += 1;
        write((const uint8_t *)&wire->pure, 1);
        total += 1;
      }
      { // Shards len
        uint32_t len = uint32_t(wire->shards.size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
      }
      // Shards
      for (auto shard : wire->shards) {
        SHVar shardVar{};
        shardVar.valueType = SHType::ShardRef;
        shardVar.payload.shardValue = shard;
        total += serialize(shardVar, write);
      }

      { // traits len
        uint32_t len = uint32_t(wire->getTraits().size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
      }
      for (auto &trait : wire->getTraits()) {
        total += serialize((SHTrait &)trait, write);
      }

      if (private_internal) {
        // just serialize the address of the wire that will be parent when deserialized
        size_t parentAddress = reinterpret_cast<size_t>(wire);
        write((const uint8_t *)&parentAddress, sizeof(size_t));
        total += sizeof(size_t);
      }
      break;
    }
    case SHType::Object: {
      int64_t id = (int64_t)input.payload.objectVendorId << 32 | input.payload.objectTypeId;
      write((const uint8_t *)&id, sizeof(int64_t));
      total += sizeof(int64_t);
      if ((input.flags & SHVAR_FLAGS_USES_OBJINFO) == SHVAR_FLAGS_USES_OBJINFO && input.objectInfo &&
          input.objectInfo->serialize) {
        uint64_t len = 0;
        uint8_t *data = nullptr;
        SHPointer handle = nullptr;
        if (!input.objectInfo->serialize(input.payload.objectValue, &data, &len, &handle)) {
          throw shards::SHException("Failed to serialize custom object variable!");
        }
        uint64_t ulen = uint64_t(len);
        write((const uint8_t *)&ulen, sizeof(uint64_t));
        total += sizeof(uint64_t);
        write((const uint8_t *)&data[0], len);
        total += len;
        input.objectInfo->free(handle);
      } else {
        uint64_t empty = 0;
        write((const uint8_t *)&empty, sizeof(uint64_t));
        total += sizeof(uint64_t);
      }
      break;
    }
    case SHType::Type:
      total += serialize(*input.payload.typeValue, write);
      break;
    case SHType::Trait:
      total += serialize(*input.payload.traitValue, write);
      break;
    default:
      throw shards::SHException("Unknown type during serialization!");
    }
    return total;
  }

  template <class BinaryReader> void deserialize(BinaryReader &read, SHTypeInfo &output) {
    read((uint8_t *)&output.basicType, sizeof(uint8_t));
    switch (output.basicType) {
    case SHType::None:
    case SHType::Any:
    case SHType::Bool:
    case SHType::Int:
    case SHType::Int2:
    case SHType::Int3:
    case SHType::Int4:
    case SHType::Int8:
    case SHType::Int16:
    case SHType::Float:
    case SHType::Float2:
    case SHType::Float3:
    case SHType::Float4:
    case SHType::Color:
    case SHType::Bytes:
    case SHType::String:
    case SHType::Path:
    case SHType::ContextVar:
    case SHType::Image:
    case SHType::Wire:
    case SHType::ShardRef:
    case SHType::Audio:
    case SHType::Type:
      // No extra data
      break;
    case SHType::Table: {
      uint32_t len = 0;
      read((uint8_t *)&len, sizeof(uint32_t));
      for (uint32_t i = 0; i < len; i++) {
        SHVar key{};
        deserialize(read, key);
        arrayPush(output.table.keys, key);
      }
      len = 0;
      read((uint8_t *)&len, sizeof(uint32_t));
      for (uint32_t i = 0; i < len; i++) {
        SHTypeInfo info{};
        deserialize(read, info);
        arrayPush(output.table.types, info);
      }
    } break;
    case SHType::Seq: {
      uint32_t len = 0;
      read((uint8_t *)&len, sizeof(uint32_t));
      for (uint32_t i = 0; i < len; i++) {
        SHTypeInfo info{};
        deserialize(read, info);
        arrayPush(output.seqTypes, info);
      }
    } break;
    case SHType::Object:
      read((uint8_t *)&output.object, sizeof(output.object));
      break;
    case SHType::Enum:
      read((uint8_t *)&output.enumeration, sizeof(output.enumeration));
      break;
    default:
      throw shards::SHException("Invalid type to deserialize");
    }
  }

  template <class BinaryWriter> size_t serialize(const SHTypeInfo &input, BinaryWriter &write) {
    size_t total{};
    write((const uint8_t *)&input.basicType, sizeof(uint8_t));
    total += sizeof(uint8_t);
    switch (input.basicType) {
    case SHType::None:
    case SHType::Any:
    case SHType::Bool:
    case SHType::Int:
    case SHType::Int2:
    case SHType::Int3:
    case SHType::Int4:
    case SHType::Int8:
    case SHType::Int16:
    case SHType::Float:
    case SHType::Float2:
    case SHType::Float3:
    case SHType::Float4:
    case SHType::Color:
    case SHType::Bytes:
    case SHType::String:
    case SHType::Path:
    case SHType::ContextVar:
    case SHType::Image:
    case SHType::Wire:
    case SHType::ShardRef:
    case SHType::Audio:
    case SHType::Type:
      // No extra data
      break;
    case SHType::Table:
      write((const uint8_t *)&input.table.keys.len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      for (uint32_t i = 0; i < input.table.keys.len; i++) {
        total += serialize(input.table.keys.elements[i], write);
      }
      write((const uint8_t *)&input.table.types.len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      for (uint32_t i = 0; i < input.table.types.len; i++) {
        total += serialize(input.table.types.elements[i], write);
      }
      break;
    case SHType::Seq:
      write((const uint8_t *)&input.seqTypes.len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      for (uint32_t i = 0; i < input.seqTypes.len; i++) {
        total += serialize(input.seqTypes.elements[i], write);
      }
      break;
    case SHType::Object:
      write((const uint8_t *)&input.object, sizeof(input.object));
      total += sizeof(input.object);
      break;
    case SHType::Enum:
      write((const uint8_t *)&input.enumeration, sizeof(input.enumeration));
      total += sizeof(input.enumeration);
      break;
    default:
      throw shards::SHException("Invalid type to serialize");
    }
    return total;
  }

  template <class BinaryReader> void deserialize(BinaryReader &read, SHTraitVariable &output) {
    SHStringPayload spl{};
    deserialize(read, spl, false);
    output.name = SHStringWithLen{
        .string = spl.elements,
        .len = spl.len,
    };
    deserialize(read, output.type);
  }

  template <class BinaryWriter> size_t serialize(const SHTraitVariable &input, BinaryWriter &write) {
    size_t total{};
    SHStringPayload spl{
        .elements = const_cast<char *>(input.name.string),
        .len = uint32_t(input.name.len),
    };
    total += serialize(spl, write);
    total += serialize(input.type, write);
    return total;
  }

  template <class BinaryReader> void deserialize(BinaryReader &read, SHTrait &output) {
    read((uint8_t *)output.id, sizeof(SHTrait::id));

    // TODO: Don't really need name to be serialized
    SHStringPayload spl{};
    deserialize(read, spl, false);
    output.name = SHStringWithLen{
        .string = spl.elements,
        .len = spl.len,
    };

    uint32_t len;
    read((uint8_t *)&len, sizeof(uint32_t));
    shards::arrayResize(output.variables, len);
    for (uint32_t i = 0; i < len; i++) {
      deserialize(read, output.variables.elements[i]);
    }
  }

  template <class BinaryWriter> size_t serialize(const SHTrait &input, BinaryWriter &write) {
    size_t total{};
    write((const uint8_t *)input.id, sizeof(SHTrait::id));
    total += sizeof(SHTrait::id);

    // TODO: Don't really need name to be serialized
    SHStringPayload spl{
        .elements = const_cast<char *>(input.name.string),
        .len = uint32_t(input.name.len),
    };
    total += serialize(spl, write);

    write((const uint8_t *)&input.variables.len, sizeof(uint32_t));
    total += sizeof(uint32_t);
    for (uint32_t i = 0; i < input.variables.len; i++) {
      total += serialize(input.variables.elements[i], write);
    }
    return total;
  }
};

} // namespace shards

#endif /* F5B6AA09_CDF7_4DFD_B508_854541459119 */
