/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

/*
Utility to auto load auto discover shards from DLLs
All that is needed is to declare a shards::registerShards
At runtime just dlopen the dll, that's it!
*/

#ifndef SH_DLLSHARD_HPP
#define SH_DLLSHARD_HPP

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include <cassert>
#include <functional>

#include "utility.hpp"
#include "shardwrapper.hpp"
#include "common_types.hpp"

#define SH_NO_INTERNAL_CORE
#include "../shards/core/object_type.hpp"

namespace shards {
// this must be defined in the external
extern void registerExternalShards();

struct CoreLoader {
  SHCore *_core{nullptr};

  CoreLoader() {
    SHShardsInterface ifaceproc;
#ifdef _WIN32
    auto handle = GetModuleHandle(NULL);
    ifaceproc = (SHShardsInterface)GetProcAddress(handle, "shardsInterface");
#else
    auto handle = dlopen(NULL, RTLD_NOW);
    ifaceproc = (SHShardsInterface)dlsym(handle, "shardsInterface");
#endif

    if (!ifaceproc) {
      // try again.. see if we are libshards
#ifdef _WIN32
      handle = GetModuleHandleA("libshards.dll");
      if (handle)
        ifaceproc = (SHShardsInterface)GetProcAddress(handle, "shardsInterface");
#else
      handle = dlopen("libshards.so", RTLD_NOW);
      if (handle)
        ifaceproc = (SHShardsInterface)dlsym(handle, "shardsInterface");
#endif
    }
    assert(ifaceproc);
    _core = ifaceproc(SHARDS_CURRENT_ABI);
    assert(_core);
    _core->log("loading external shards..."_swl);
    registerExternalShards();
  }
};

class Core {
public:
  static void registerShard(const char *fullName, SHShardConstructor constructor) {
    sCore._core->registerShard(fullName, constructor);
  }

  static void registerObjectType(int32_t vendorId, int32_t typeId, SHObjectInfo info) {
    sCore._core->registerObjectType(vendorId, typeId, info);
  }

  static void registerEnumType(int32_t vendorId, int32_t typeId, SHEnumInfo info) {
    sCore._core->registerEnumType(vendorId, typeId, info);
  }

  static SHVar *referenceVariable(SHContext *context, struct SHStringWithLen name) {
    return sCore._core->referenceVariable(context, name);
  }

  static void releaseVariable(SHVar *variable) { return sCore._core->releaseVariable(variable); }

  static SHWireState suspend(SHContext *context, double seconds) { return sCore._core->suspend(context, seconds); }

  static void cloneVar(SHVar &dst, const SHVar &src) { sCore._core->cloneVar(&dst, &src); }

  static void destroyVar(SHVar &var) { sCore._core->destroyVar(&var); }

#define SH_ARRAY_INTERFACE(_arr_, _val_, _short_)                                                                 \
  static void _short_##Free(_arr_ &seq) { sCore._core->_short_##Free(&seq); };                                    \
                                                                                                                  \
  static void _short_##Resize(_arr_ &seq, uint64_t size) { sCore._core->_short_##Resize(&seq, size); };           \
                                                                                                                  \
  static void _short_##Push(_arr_ &seq, const _val_ &value) { sCore._core->_short_##Push(&seq, &value); };        \
                                                                                                                  \
  static void _short_##Insert(_arr_ &seq, uint64_t index, const _val_ &value) {                                   \
    sCore._core->_short_##Insert(&seq, index, &value);                                                            \
  };                                                                                                              \
                                                                                                                  \
  static _val_ _short_##Pop(_arr_ &seq) { return sCore._core->_short_##Pop(&seq); };                              \
                                                                                                                  \
  static void _short_##FastDelete(_arr_ &seq, uint64_t index) { sCore._core->_short_##FastDelete(&seq, index); }; \
                                                                                                                  \
  static void _short_##SlowDelete(_arr_ &seq, uint64_t index) { sCore._core->_short_##SlowDelete(&seq, index); }

  SH_ARRAY_INTERFACE(SHSeq, SHVar, seq);
  SH_ARRAY_INTERFACE(SHTypesInfo, SHTypeInfo, types);
  SH_ARRAY_INTERFACE(SHParametersInfo, SHParameterInfo, params);
  SH_ARRAY_INTERFACE(Shards, ShardPtr, shards);
  SH_ARRAY_INTERFACE(SHExposedTypesInfo, SHExposedTypeInfo, expTypes);
  SH_ARRAY_INTERFACE(SHStrings, SHString, strings);

  static SHComposeResult composeWire(SHWireRef wire, SHInstanceData data) { return sCore._core->composeWire(wire, data); }

  static SHRunWireOutput runWire(SHWireRef wire, SHContext *context, const SHVar &input) {
    return sCore._core->runWire(wire, context, &input);
  }

  static SHComposeResult composeShards(Shards shards, SHInstanceData data) { return sCore._core->composeShards(shards, data); }

  static SHWireState runShards(Shards shards, SHContext *context, const SHVar &input, SHVar &output) {
    return sCore._core->runShards(shards, context, &input, &output);
  }

  static SHWireState runShards2(Shards shards, SHContext *context, const SHVar &input, SHVar &output) {
    return sCore._core->runShards2(shards, context, &input, &output);
  }

  static void log(struct SHStringWithLen msg) { sCore._core->log(msg); }

  static void abortWire(SHContext *context, struct SHStringWithLen msg) { sCore._core->abortWire(context, msg); }

  static const char *rootPath() { return sCore._core->getRootPath(); }

  static SHVar asyncActivate(SHContext *context, std::function<SHVar()> activate, std::function<void()> cancellation) {
    auto tup = std::make_tuple(activate, cancellation);
    return sCore._core->asyncActivate(
        context, &tup,
        [](auto ctx, auto data) {
          auto f = reinterpret_cast<decltype(tup) *>(data);
          return std::get<0>(*f)();
        },
        [](auto ctx, auto data) {
          auto f = reinterpret_cast<decltype(tup) *>(data);
          return std::get<1>(*f)();
        });
  }

  static SHVar *referenceWireVariable(SHWireRef wire, struct SHStringWithLen name) {
    return sCore._core->referenceWireVariable(wire, name);
  }

  static void setExternalVariable(SHWireRef wire, struct SHStringWithLen name, struct SHExternalVariable *pVar) {
    sCore._core->setExternalVariable(wire, name, pVar);
  }

  static void removeExternalVariable(SHWireRef wire, struct SHStringWithLen name) {
    sCore._core->removeExternalVariable(wire, name);
  }

  static SHVar *allocExternalVariable(SHWireRef wire, struct SHStringWithLen name, const struct SHTypeInfo *type) {
    return sCore._core->allocExternalVariable(wire, name, type);
  }

  static void freeExternalVariable(SHWireRef wire, struct SHStringWithLen name) { sCore._core->freeExternalVariable(wire, name); }

  static SHVar *getMeshVariable(SHMeshRef mesh, struct SHStringWithLen name) { return sCore._core->getMeshVariable(mesh, name); }

  static void setMeshVariableType(SHMeshRef mesh, struct SHStringWithLen name, const struct SHExposedTypeInfo *type) {
    sCore._core->setMeshVariableType(mesh, name, type);
  }

  static SHWireState getState(SHContext *context) { return sCore._core->getState(context); }

  static SHVar hashVar(const SHVar &var) { return sCore._core->hashVar(&var); }

  static bool isEqualVar(const SHVar &v1, const SHVar &v2) { return sCore._core->isEqualVar(&v1, &v2); }

  static int compareVar(const SHVar &v1, const SHVar &v2) { return sCore._core->compareVar(&v1, &v2); }

  static bool isEqualType(const SHTypeInfo &t1, const SHTypeInfo &t2) { return sCore._core->isEqualType(&t1, &t2); }

  static SHTypeInfo deriveTypeInfo(const SHVar &v, const SHInstanceData *data, bool mutable_) {
    return sCore._core->deriveTypeInfo(&v, data, mutable_);
  }

  static void freeDerivedTypeInfo(SHTypeInfo &t) { sCore._core->freeDerivedTypeInfo(&t); }

  static const SHEnumInfo *findEnumInfo(int32_t vendorId, int32_t typeId) { return sCore._core->findEnumInfo(vendorId, typeId); }

  static int64_t findEnumId(SHStringWithLen name) { return sCore._core->findEnumId(name); }

  static const SHObjectInfo *findObjectInfo(int32_t vendorId, int32_t typeId) {
    return sCore._core->findObjectInfo(vendorId, typeId);
  }

  static int64_t findObjectTypeId(SHStringWithLen name) { return sCore._core->findObjectTypeId(name); }

  static SHString type2Name(SHType type) { return sCore._core->type2Name(type); }

  // Additional array interfaces
  SH_ARRAY_INTERFACE(SHEnums, SHEnum, enums);
  SH_ARRAY_INTERFACE(SHTraitVariables, SHTraitVariable, traitVariables);
  SH_ARRAY_INTERFACE(SHTraits, SHTrait, traits);

  // Image manipulation
  static void imageIncRef(struct SHImage *image) { sCore._core->imageIncRef(image); }

  static void imageDecRef(struct SHImage *image) { sCore._core->imageDecRef(image); }

  static struct SHImage *imageNew(uint32_t dataLen) { return sCore._core->imageNew(dataLen); }

  static struct SHImage *imageClone(struct SHImage *image) { return sCore._core->imageClone(image); }

  static uint32_t imageDeriveDataLength(struct SHImage *image) { return sCore._core->imageDeriveDataLength(image); }

  // Wire manipulation
  static void setWireLooped(SHWireRef wire, SHBool looped) { sCore._core->setWireLooped(wire, looped); }

  static void setWireUnsafe(SHWireRef wire, SHBool unsafe) { sCore._core->setWireUnsafe(wire, unsafe); }

  static void setWirePure(SHWireRef wire, SHBool pure) { sCore._core->setWirePure(wire, pure); }

  static void setWireStackSize(SHWireRef wire, uint64_t stackSize) { sCore._core->setWireStackSize(wire, stackSize); }

  static void setWirePriority(SHWireRef wire, int priority) { sCore._core->setWirePriority(wire, priority); }

  static void setWireTraits(SHWireRef wire, SHSeq traits) { sCore._core->setWireTraits(wire, traits); }

  static void addShard(SHWireRef wire, ShardPtr shard) { sCore._core->addShard(wire, shard); }

  static void removeShard(SHWireRef wire, ShardPtr shard) { sCore._core->removeShard(wire, shard); }

  static bool isWireRunning(SHWireRef wire) { return sCore._core->isWireRunning(wire); }

  static SHVar stopWire(SHWireRef wire) { return sCore._core->stopWire(wire); }

  // Mesh operations
  static SHMeshRef createMesh() { return sCore._core->createMesh(); }

  static void destroyMesh(SHMeshRef mesh) { sCore._core->destroyMesh(mesh); }

  static SHVar createMeshVar() { return sCore._core->createMeshVar(); }

  static void setMeshLabel(SHMeshRef mesh, SHStringWithLen label) { sCore._core->setMeshLabel(mesh, label); }

  static bool compose(SHMeshRef mesh, SHWireRef wire) { return sCore._core->compose(mesh, wire); }

  static void schedule(SHMeshRef mesh, SHWireRef wire, SHBool compose) { sCore._core->schedule(mesh, wire, compose); }

  static void unschedule(SHMeshRef mesh, SHWireRef wire) { sCore._core->unschedule(mesh, wire); }

  static bool tick(SHMeshRef mesh) { return sCore._core->tick(mesh); }

  static bool isEmpty(SHMeshRef mesh) { return sCore._core->isEmpty(mesh); }

  static void terminate(SHMeshRef mesh) { sCore._core->terminate(mesh); }

  static void sleep(double seconds) { sCore._core->sleep(seconds); }

  static uint64_t getStepCount(SHContext *context) { return sCore._core->getStepCount(context); }

private:
  static inline CoreLoader sCore{};
};

using ParamVar = TParamVar<Core>;
using ShardsVar = TShardsVar<Core>;
using OwnedVar = TOwnedVar<Core>;
using SeqVar = TSeqVar<Core>;
using TableVar = TTableVar<Core>;

template <typename E, std::vector<uint8_t> (*Serializer)(const E &) = nullptr,
          E (*Deserializer)(const std::string_view &) = nullptr, void (*BeforeDelete)(const E &) = nullptr,
          bool ThreadSafe = false>
class ObjectVar : public TObjectVar<Core, E, Serializer, Deserializer, BeforeDelete, ThreadSafe> {
public:
  ObjectVar(const char *name, int32_t vendorId, int32_t objectId)
      : TObjectVar<Core, E, Serializer, Deserializer, BeforeDelete, ThreadSafe>(name, vendorId, objectId) {}
};

template <typename T> class PtrIterator {
public:
  typedef T value_type;

  PtrIterator(T *ptr) : ptr(ptr) {}

  PtrIterator &operator+=(std::ptrdiff_t s) {
    ptr += s;
    return *this;
  }
  PtrIterator &operator-=(std::ptrdiff_t s) {
    ptr -= s;
    return *this;
  }

  PtrIterator operator+(std::ptrdiff_t s) {
    PtrIterator it(ptr);
    return it += s;
  }
  PtrIterator operator-(std::ptrdiff_t s) {
    PtrIterator it(ptr);
    return it -= s;
  }

  PtrIterator operator++() {
    ++ptr;
    return *this;
  }
  PtrIterator operator++(int s) {
    ptr += s;
    return *this;
  }

  PtrIterator operator--() {
    --ptr;
    return *this;
  }
  PtrIterator operator--(int s) {
    ptr -= s;
    return *this;
  }

  std::ptrdiff_t operator-(PtrIterator const &other) const { return ptr - other.ptr; }

  T &operator*() const { return *ptr; }

  bool operator==(const PtrIterator &rhs) const { return ptr == rhs.ptr; }
  bool operator!=(const PtrIterator &rhs) const { return ptr != rhs.ptr; }
  bool operator>(const PtrIterator &rhs) const { return ptr > rhs.ptr; }
  bool operator<(const PtrIterator &rhs) const { return ptr < rhs.ptr; }
  bool operator>=(const PtrIterator &rhs) const { return ptr >= rhs.ptr; }
  bool operator<=(const PtrIterator &rhs) const { return ptr <= rhs.ptr; }

private:
  T *ptr;
};

template <typename S, typename T, void (*arrayResize)(S &, uint64_t), void (*arrayFree)(S &), void (*arrayPush)(S &, const T &),
          typename Allocator = std::allocator<T>>
class IterableArray {
public:
  typedef S seq_type;
  typedef T value_type;
  typedef size_t size_type;

  typedef PtrIterator<T> iterator;
  typedef PtrIterator<const T> const_iterator;

  IterableArray() : _seq({}), _owned(true) {}

  // Not OWNING!
  IterableArray(const seq_type &seq) : _seq(seq), _owned(false) {}

  // implicit converter
  IterableArray(const SHVar &v) : _seq(v.payload.seqValue), _owned(false) { assert(v.valueType == SHType::Seq); }

  IterableArray(size_t s) : _seq({}), _owned(true) { arrayResize(_seq, s); }

  IterableArray(size_t s, T v) : _seq({}), _owned(true) {
    arrayResize(_seq, s);
    for (size_t i = 0; i < s; i++) {
      _seq[i] = v;
    }
  }

  IterableArray(const_iterator first, const_iterator last) : _seq({}), _owned(true) {
    size_t size = last - first;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = *first++;
    }
  }

  IterableArray(const IterableArray &other) : _seq({}), _owned(true) {
    size_t size = other._seq.len;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = other._seq.elements[i];
    }
  }

  IterableArray &operator=(SHVar &var) {
    assert(var.valueType == SHType::Seq);
    _seq = var.payload.seqValue;
    _owned = false;
    return *this;
  }

  IterableArray &operator=(IterableArray &other) {
    _seq = other._seq;
    _owned = other._owned;
    return *this;
  }

  IterableArray &operator=(const IterableArray &other) {
    _seq = {};
    _owned = true;
    size_t size = other._seq.len;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = other._seq.elements[i];
    }
    return *this;
  }

  ~IterableArray() {
    if (_owned) {
      arrayFree(_seq);
    }
  }

private:
  seq_type _seq;
  bool _owned;

public:
  iterator begin() { return iterator(&_seq.elements[0]); }
  const_iterator begin() const { return const_iterator(&_seq.elements[0]); }
  const_iterator cbegin() const { return const_iterator(&_seq.elements[0]); }
  iterator end() { return iterator(&_seq.elements[0] + size()); }
  const_iterator end() const { return const_iterator(&_seq.elements[0] + size()); }
  const_iterator cend() const { return const_iterator(&_seq.elements[0] + size()); }
  // those (T&) casts are a bit unsafe
  // but needed when overriding SHVar with utility classes
  T &operator[](int index) { return (T &)_seq.elements[index]; }
  const T &operator[](int index) const { return (T &)_seq.elements[index]; }
  T &front() { return (T &)_seq.elements[0]; }
  const T &front() const { return (T &)_seq.elements[0]; }
  T &back() { return (T &)_seq.elements[size() - 1]; }
  const T &back() const { return (T &)_seq.elements[size() - 1]; }
  T *data() { return (T *)_seq; }
  size_t size() const { return _seq.len; }
  bool empty() const { return _seq.elements == nullptr || size() == 0; }
  void resize(size_t nsize) { arrayResize(_seq, nsize); }
  void push_back(const T &value) { arrayPush(_seq, value); }
  void clear() { arrayResize(_seq, 0); }
  operator seq_type() const { return _seq; }
};

using IterableSeq = IterableArray<SHSeq, SHVar, &Core::seqResize, &Core::seqFree, &Core::seqPush>;
using IterableExposedInfo =
    IterableArray<SHExposedTypesInfo, SHExposedTypeInfo, &Core::expTypesResize, &Core::expTypesFree, &Core::expTypesPush>;

template <typename E, const char *Name_, const SHOptionalString &Help_, int32_t VendorId_, int32_t TypeId_, bool IsFlags_ = false>
class EnumInfo : public TEnumInfo<Core, E, Name_, Help_, VendorId_, TypeId_, IsFlags_> {
public:
  EnumInfo(const char *name, int32_t vendorId, int32_t enumId)
      : TEnumInfo<Core, E, Name_, Help_, VendorId_, TypeId_, IsFlags_>(name, vendorId, enumId) {}
};

inline void registerShard(const char *fullName, SHShardConstructor constructor, std::string_view _) {
  Core::registerShard(fullName, constructor);
}

inline void abortWire(SHContext *ctx, struct SHStringWithLen msg) { Core::abortWire(ctx, msg); }
inline void abortWire(SHContext *ctx, std::string_view msg) { Core::abortWire(ctx, toSWL(msg)); }
inline void log(std::string_view msg) { Core::log(toSWL(msg)); }
}; // namespace shards

#endif
