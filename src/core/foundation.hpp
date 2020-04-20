/* SPDX-License-Identifier: BSD 3-Clause "New" or "Revised" License */
/* Copyright © 2019-2020 Giovanni Petrantoni */

#ifndef CB_CORE_HPP
#define CB_CORE_HPP

#ifdef CB_USE_TSAN
extern "C" {
void *__tsan_get_current_fiber(void);
void *__tsan_create_fiber(unsigned flags);
void __tsan_destroy_fiber(void *fiber);
void __tsan_switch_to_fiber(void *fiber, unsigned flags);
void __tsan_set_fiber_name(void *fiber, const char *name);
const unsigned __tsan_switch_to_fiber_no_sync = 1 << 0;
}
#endif

#include "chainblocks.h"
#include "ops_internal.hpp"
#include <chainblocks.hpp>

// Included 3rdparty
#include "easylogging++.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <list>
#include <set>
#include <type_traits>
#include <variant>

#include "blockwrapper.hpp"

// Needed specially for win32/32bit
#include <boost/align/aligned_allocator.hpp>

// For coroutines/context switches
#include <boost/context/continuation.hpp>
typedef boost::context::continuation CBCoro;

#define TRACE_LINE DLOG(TRACE) << "#trace#"

CBValidationResult validateConnections(const CBlocks chain,
                                       CBValidationCallback callback,
                                       void *userData, CBInstanceData data);

namespace chainblocks {
constexpr uint32_t FragCC = 'frag'; // 1718772071

CBChainState activateBlocks(CBSeq blocks, CBContext *context,
                            const CBVar &chainInput, CBVar &output);
CBChainState activateBlocks(CBlocks blocks, CBContext *context,
                            const CBVar &chainInput, CBVar &output);
CBVar *referenceGlobalVariable(CBContext *ctx, const char *name);
CBVar *referenceVariable(CBContext *ctx, const char *name);
void releaseVariable(CBVar *variable);
CBChainState suspend(CBContext *context, double seconds);
void registerEnumType(int32_t vendorId, int32_t enumId, CBEnumInfo info);

CBlock *createBlock(const char *name);
void registerCoreBlocks();
void registerBlock(const char *fullName, CBBlockConstructor constructor);
void registerObjectType(int32_t vendorId, int32_t typeId, CBObjectInfo info);
void registerEnumType(int32_t vendorId, int32_t typeId, CBEnumInfo info);
void registerRunLoopCallback(const char *eventName, CBCallback callback);
void unregisterRunLoopCallback(const char *eventName);
void registerExitCallback(const char *eventName, CBCallback callback);
void unregisterExitCallback(const char *eventName);
void callExitCallbacks();
void registerChain(CBChain *chain);
void unregisterChain(CBChain *chain);

struct RuntimeObserver {
  virtual void registerBlock(const char *fullName,
                             CBBlockConstructor constructor) {}
  virtual void registerObjectType(int32_t vendorId, int32_t typeId,
                                  CBObjectInfo info) {}
  virtual void registerEnumType(int32_t vendorId, int32_t typeId,
                                CBEnumInfo info) {}
};

ALWAYS_INLINE inline void cloneVar(CBVar &dst, const CBVar &src);
ALWAYS_INLINE inline void destroyVar(CBVar &src);

struct InternalCore;
} // namespace chainblocks

struct CBChain {
  enum State {
    Stopped,
    Prepared,
    Starting,
    Iterating,
    IterationEnded,
    Failed,
    Ended
  };

  CBChain(const char *chain_name)
      : looped(false), unsafe(false), name(chain_name), coro(nullptr),
        rootTickInput(CBVar()), finishedOutput(CBVar()), ownedOutput(false),
        composedHash(0), context(nullptr), node(nullptr) {
    LOG(TRACE) << "CBChain(): " << name;
#ifdef CB_USE_TSAN
    tsan_coro = nullptr;
#endif
  }

  ~CBChain() {
    clear();
    chainblocks::destroyVar(rootTickInput);
    LOG(TRACE) << "~CBChain() " << name;

#ifdef CB_USE_TSAN
    if (tsan_coro) {
      __tsan_destroy_fiber(tsan_coro);
    }
#endif
  }

  void clear();

  void warmup(CBContext *context) {
    for (auto &blk : blocks) {
      if (blk->warmup)
        blk->warmup(blk, context);
    }
  }

  // Also the chain takes ownership of the block!
  void addBlock(CBlock *blk) {
    assert(!blk->owned);
    blk->owned = true;
    blocks.push_back(blk);
  }

  // Also removes ownership of the block
  void removeBlock(CBlock *blk) {
    auto findIt = std::find(blocks.begin(), blocks.end(), blk);
    if (findIt != blocks.end()) {
      blocks.erase(findIt);
      blk->owned = false;
    } else {
      throw chainblocks::CBException("removeBlock: block not found!");
    }
  }

  // Attributes
  bool looped;
  bool unsafe;

  std::string name;

  CBCoro *coro;
#ifdef CB_USE_TSAN
  void *tsan_coro;
#endif

  std::atomic<State> state{Stopped};

  CBVar rootTickInput{};
  CBVar previousOutput{};
  CBVar finishedOutput{};
  bool ownedOutput;

  std::size_t composedHash;

  CBContext *context;
  CBNode *node;
  CBFlow *flow;
  std::vector<CBlock *> blocks;
  std::unordered_map<std::string, CBVar, std::hash<std::string>,
                     std::equal_to<std::string>,
                     boost::alignment::aligned_allocator<
                         std::pair<const std::string, CBVar>, 16>>
      variables;

  static std::shared_ptr<CBChain> sharedFromRef(CBChainRef ref) {
    return *reinterpret_cast<std::shared_ptr<CBChain> *>(ref);
  }

  static void deleteRef(CBChainRef ref) {
    auto pref = reinterpret_cast<std::shared_ptr<CBChain> *>(ref);
    delete pref;
  }

  CBChainRef newRef() {
    return reinterpret_cast<CBChainRef>(new std::shared_ptr<CBChain>(this));
  }

  static CBChainRef weakRef(std::shared_ptr<CBChain> &shared) {
    return reinterpret_cast<CBChainRef>(&shared);
  }

  static CBChainRef addRef(CBChainRef ref) {
    auto cref = sharedFromRef(ref);
    auto res = new std::shared_ptr<CBChain>();
    *res = cref;
    return reinterpret_cast<CBChainRef>(res);
  }
};

namespace chainblocks {
using OwnedVar = TOwnedVar<InternalCore>;

using CBMap = std::unordered_map<
    std::string, OwnedVar, std::hash<std::string>, std::equal_to<std::string>,
    boost::alignment::aligned_allocator<std::pair<const std::string, OwnedVar>,
                                        16>>;
using CBMapIt = std::unordered_map<
    std::string, OwnedVar, std::hash<std::string>, std::equal_to<std::string>,
    boost::alignment::aligned_allocator<std::pair<const std::string, OwnedVar>,
                                        16>>::iterator;

struct Globals {
  static inline std::unordered_map<std::string, CBBlockConstructor>
      BlocksRegister;
  static inline std::unordered_map<int64_t, CBObjectInfo> ObjectTypesRegister;
  static inline std::unordered_map<int64_t, CBEnumInfo> EnumTypesRegister;

  // map = ordered! we need that for those
  static inline std::map<std::string, CBCallback> RunLoopHooks;
  static inline std::map<std::string, CBCallback> ExitHooks;

  static inline std::unordered_map<std::string, std::shared_ptr<CBChain>>
      GlobalChains;

  static inline std::list<std::weak_ptr<RuntimeObserver>> Observers;

  static inline std::string RootPath;

  static inline CBTableInterface TableInterface{
      .tableForEach =
          [](CBTable table, CBTableForEachCallback cb, void *data) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            for (auto &entry : *map) {
              if (!cb(entry.first.c_str(), &entry.second, data))
                break;
            }
          },
      .tableSize =
          [](CBTable table) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            return map->size();
          },
      .tableContains =
          [](CBTable table, const char *key) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            return map->count(key) > 0;
          },
      .tableAt =
          [](CBTable table, const char *key) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            CBVar &vref = (*map)[key];
            return &vref;
          },
      .tableRemove =
          [](CBTable table, const char *key) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            map->erase(key);
          },
      .tableClear =
          [](CBTable table) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            map->clear();
          },
      .tableFree =
          [](CBTable table) {
            chainblocks::CBMap *map =
                reinterpret_cast<chainblocks::CBMap *>(table.opaque);
            delete map;
          }};
};

template <typename T>
NO_INLINE void arrayGrow(T &arr, size_t addlen, size_t min_cap = 4);

template <typename T, typename V>
ALWAYS_INLINE inline void arrayPush(T &arr, const V &val) {
  if ((arr.len + 1) > arr.cap) {
    arrayGrow(arr, 1);
  }
  arr.elements[arr.len++] = val;
}

template <typename T>
ALWAYS_INLINE inline void arrayResize(T &arr, uint32_t size) {
  if (arr.cap < size) {
    arrayGrow(arr, size - arr.len);
  }
  arr.len = size;
}

template <typename T, typename V>
ALWAYS_INLINE inline void arrayInsert(T &arr, uint32_t index, const V &val) {
  if ((arr.len + 1) > arr.cap) {
    arrayGrow(arr, 1);
  }
  memmove(&arr.elements[index + 1], &arr.elements[index],
          sizeof(V) * (arr.len - index));
  arr.len++;
  arr.elements[index] = val;
}

template <typename T, typename V> ALWAYS_INLINE inline V arrayPop(T &arr) {
  assert(arr.len > 0);
  arr.len--;
  return arr.elements[arr.len];
}

template <typename T>
ALWAYS_INLINE inline void arrayDelFast(T &arr, uint32_t index) {
  assert(arr.len > 0);
  arr.len--;
  // this allows eventual destroyVar/cloneVar magic
  // avoiding allocations even in nested seqs
  std::swap(arr.elements[index], arr.elements[arr.len]);
}

template <typename T>
ALWAYS_INLINE inline void arrayDel(T &arr, uint32_t index) {
  assert(arr.len > 0);
  // this allows eventual destroyVar/cloneVar magic
  // avoiding allocations even in nested seqs
  arr.len--;
  while (index < arr.len) {
    std::swap(arr.elements[index], arr.elements[index + 1]);
    index++;
  }
}

template <typename T> inline void arrayFree(T &arr) {
  if (arr.elements)
    ::operator delete (arr.elements, std::align_val_t{16});
  memset(&arr, 0x0, sizeof(T));
}

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

  std::ptrdiff_t operator-(PtrIterator const &other) const {
    return ptr - other.ptr;
  }

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

template <typename S, typename T, typename Allocator = std::allocator<T>>
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
  IterableArray(const CBVar &v) : _seq(v.payload.seqValue), _owned(false) {
    assert(v.valueType == Seq);
  }

  IterableArray(size_t s) : _seq({}), _owned(true) { arrayResize(_seq, s); }

  IterableArray(size_t s, T v) : _seq({}), _owned(true) {
    arrayResize(_seq, s);
    for (size_t i = 0; i < s; i++) {
      _seq[i] = v;
    }
  }

  IterableArray(const_iterator first, const_iterator last)
      : _seq({}), _owned(true) {
    size_t size = last - first;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq[i] = *first++;
    }
  }

  IterableArray(const IterableArray &other) : _seq({}), _owned(true) {
    size_t size = other._seq.len;
    arrayResize(_seq, size);
    for (size_t i = 0; i < size; i++) {
      _seq.elements[i] = other._seq.elements[i];
    }
  }

  IterableArray &operator=(IterableArray &other) {
    std::swap(_seq, other._seq);
    std::swap(_owned, other._owned);
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
  const_iterator end() const {
    return const_iterator(&_seq.elements[0] + size());
  }
  const_iterator cend() const {
    return const_iterator(&_seq.elements[0] + size());
  }
  T &operator[](int index) { return _seq.elements[index]; }
  const T &operator[](int index) const { return _seq.elements[index]; }
  T &front() { return _seq.elements[0]; }
  const T &front() const { return _seq.elements[0]; }
  T &back() { return _seq.elements[size() - 1]; }
  const T &back() const { return _seq.elements[size() - 1]; }
  T *data() { return _seq; }
  size_t size() const { return _seq.len; }
  bool empty() const { return _seq.elements == nullptr || size() == 0; }
  void resize(size_t nsize) { arrayResize(_seq, nsize); }
  void push_back(const T &value) { arrayPush(_seq, value); }
  void clear() { arrayResize(_seq, 0); }
  operator seq_type() const { return _seq; }
};

using IterableSeq = IterableArray<CBSeq, CBVar>;
using IterableExposedInfo =
    IterableArray<CBExposedTypesInfo, CBExposedTypeInfo>;
} // namespace chainblocks

// needed by some c++ library
namespace std {
template <typename T> class iterator_traits<chainblocks::PtrIterator<T>> {
public:
  using difference_type = ptrdiff_t;
  using size_type = size_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = random_access_iterator_tag;
};

inline void swap(CBVar &v1, CBVar &v2) {
  auto t = v1;
  v1 = v2;
  v2 = t;
}
} // namespace std

namespace chainblocks {
NO_INLINE void _destroyVarSlow(CBVar &var);
NO_INLINE void _cloneVarSlow(CBVar &dst, const CBVar &src);

ALWAYS_INLINE inline void destroyVar(CBVar &var) {
  switch (var.valueType) {
  case Table:
  case Seq:
    _destroyVarSlow(var);
    break;
  case CBType::Path:
  case CBType::String:
  case ContextVar:
    delete[] var.payload.stringValue;
    break;
  case Image:
    delete[] var.payload.imageValue.data;
    break;
  case Bytes:
    delete[] var.payload.bytesValue;
    break;
  case CBType::Array:
    delete[] var.payload.arrayValue;
    break;
  case Object:
    if ((var.flags & CBVAR_FLAGS_USES_OBJINFO) == CBVAR_FLAGS_USES_OBJINFO &&
        var.objectInfo && var.objectInfo->release) {
      // in this case the custom object needs actual destruction
      var.objectInfo->release(var.payload.objectValue);
    }
    break;
  case CBType::Chain:
    CBChain::deleteRef(var.payload.chainValue);
    break;
  default:
    break;
  };

  memset(&var.payload, 0x0, sizeof(CBVarPayload));
  var.valueType = CBType::None;
}

ALWAYS_INLINE inline void cloneVar(CBVar &dst, const CBVar &src) {
  if (src.valueType < EndOfBlittableTypes &&
      dst.valueType < EndOfBlittableTypes) {
    dst.valueType = src.valueType;
    memcpy(&dst.payload, &src.payload, sizeof(CBVarPayload));
  } else if (src.valueType < EndOfBlittableTypes) {
    destroyVar(dst);
    dst.valueType = src.valueType;
    memcpy(&dst.payload, &src.payload, sizeof(CBVarPayload));
  } else {
    _cloneVarSlow(dst, src);
  }
}

struct Serialization {
  static inline void varFree(CBVar &output) {
    switch (output.valueType) {
    case CBType::None:
    case CBType::EndOfBlittableTypes:
    case CBType::Any:
    case CBType::Enum:
    case CBType::Bool:
    case CBType::Int:
    case CBType::Int2:
    case CBType::Int3:
    case CBType::Int4:
    case CBType::Int8:
    case CBType::Int16:
    case CBType::Float:
    case CBType::Float2:
    case CBType::Float3:
    case CBType::Float4:
    case CBType::Color:
    case CBType::StackIndex:
      break;
    case CBType::Bytes:
      delete[] output.payload.bytesValue;
      break;
    case CBType::Array:
      delete[] output.payload.arrayValue;
      break;
    case CBType::Path:
    case CBType::String:
    case CBType::ContextVar: {
      delete[] output.payload.stringValue;
      break;
    }
    case CBType::Seq: {
      for (uint32_t i = 0; i < output.payload.seqValue.cap; i++) {
        varFree(output.payload.seqValue.elements[i]);
      }
      chainblocks::arrayFree(output.payload.seqValue);
      break;
    }
    case CBType::Table: {
      output.payload.tableValue.api->tableFree(output.payload.tableValue);
      break;
    }
    case CBType::Image: {
      delete[] output.payload.imageValue.data;
      break;
    }
    case CBType::Block: {
      auto blk = output.payload.blockValue;
      if (!blk->owned) {
        // destroy only if not owned
        blk->destroy(blk);
      }
      break;
    }
    case CBType::Chain: {
      CBChain::deleteRef(output.payload.chainValue);
      break;
    }
    case CBType::Object: {
      if ((output.flags & CBVAR_FLAGS_USES_OBJINFO) ==
              CBVAR_FLAGS_USES_OBJINFO &&
          output.objectInfo && output.objectInfo->release) {
        output.objectInfo->release(output.payload.objectValue);
      }
      break;
    }
    }

    memset(&output, 0x0, sizeof(CBVar));
  }

  template <class BinaryReader>
  static inline void deserialize(BinaryReader &read, CBVar &output) {
    // we try to recycle memory so pass a empty None as first timer!
    CBType nextType;
    read((uint8_t *)&nextType, sizeof(output.valueType));

    // stop trying to recycle, types differ
    auto recycle = true;
    if (output.valueType != nextType) {
      varFree(output);
      recycle = false;
    }

    output.valueType = nextType;

    switch (output.valueType) {
    case CBType::None:
    case CBType::EndOfBlittableTypes:
    case CBType::Any:
      break;
    case CBType::Enum:
      read((uint8_t *)&output.payload, sizeof(int32_t) * 3);
      break;
    case CBType::Bool:
      read((uint8_t *)&output.payload, sizeof(CBBool));
      break;
    case CBType::Int:
      read((uint8_t *)&output.payload, sizeof(CBInt));
      break;
    case CBType::Int2:
      read((uint8_t *)&output.payload, sizeof(CBInt2));
      break;
    case CBType::Int3:
      read((uint8_t *)&output.payload, sizeof(CBInt3));
      break;
    case CBType::Int4:
      read((uint8_t *)&output.payload, sizeof(CBInt4));
      break;
    case CBType::Int8:
      read((uint8_t *)&output.payload, sizeof(CBInt8));
      break;
    case CBType::Int16:
      read((uint8_t *)&output.payload, sizeof(CBInt16));
      break;
    case CBType::Float:
      read((uint8_t *)&output.payload, sizeof(CBFloat));
      break;
    case CBType::Float2:
      read((uint8_t *)&output.payload, sizeof(CBFloat2));
      break;
    case CBType::Float3:
      read((uint8_t *)&output.payload, sizeof(CBFloat3));
      break;
    case CBType::Float4:
      read((uint8_t *)&output.payload, sizeof(CBFloat4));
      break;
    case CBType::Color:
      read((uint8_t *)&output.payload, sizeof(CBColor));
      break;
    case CBType::StackIndex:
      read((uint8_t *)&output.payload, sizeof(int64_t));
      break;
    case CBType::Bytes: {
      auto availBytes = recycle ? output.payload.bytesCapacity : 0;
      read((uint8_t *)&output.payload.bytesSize,
           sizeof(output.payload.bytesSize));

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
      output.payload.bytesCapacity =
          std::max(availBytes, output.payload.bytesSize);

      read((uint8_t *)output.payload.bytesValue, output.payload.bytesSize);
      break;
    }
    case CBType::Array: {
      read((uint8_t *)&output.innerType, sizeof(output.innerType));

      auto availArray = recycle ? output.payload.arrayCapacity : 0;
      read((uint8_t *)&output.payload.arrayLen,
           sizeof(output.payload.arrayLen));

      if (availArray > 0 && availArray < output.payload.arrayLen) {
        // not enough space, ideally realloc, but for now just delete
        delete[] output.payload.arrayValue;
        // and re alloc
        output.payload.arrayValue = new CBVarPayload[output.payload.arrayLen];
      } else if (availArray == 0) {
        // just alloc
        output.payload.arrayValue = new CBVarPayload[output.payload.arrayLen];
      } // else got enough space to recycle!

      // record actualSize for further recycling usage
      output.payload.arrayCapacity =
          std::max(availArray, output.payload.arrayLen);

      read((uint8_t *)output.payload.arrayValue,
           output.payload.arrayLen * sizeof(CBVarPayload));
      break;
    }
    case CBType::Path:
    case CBType::String:
    case CBType::ContextVar: {
      auto availChars = recycle ? output.payload.stringCapacity : 0;
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));

      if (availChars > 0 && availChars < len) {
        // we need more chars then what we have, realloc
        delete[] output.payload.stringValue;
        output.payload.stringValue = new char[len + 1];
      } else if (availChars == 0) {
        // just alloc
        output.payload.stringValue = new char[len + 1];
      } // else recycling

      // record actualSize
      output.payload.stringCapacity = std::max(availChars, len);

      read((uint8_t *)output.payload.stringValue, len);
      const_cast<char *>(output.payload.stringValue)[len] = 0;
      output.payload.stringLen = len;
      break;
    }
    case CBType::Seq: {
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));
      // notice we assume all elements up to capacity are memset to 0x0
      // or are valid CBVars we can overwrite
      chainblocks::arrayResize(output.payload.seqValue, len);
      for (uint32_t i = 0; i < len; i++) {
        deserialize(read, output.payload.seqValue.elements[i]);
      }
      break;
    }
    case CBType::Table: {
      CBMap *map;
      if (recycle) {
        if (output.payload.tableValue.api && output.payload.tableValue.opaque) {
          map = (CBMap *)output.payload.tableValue.opaque;
          map->clear();
        } else {
          map = new CBMap();
          output.payload.tableValue.api = &Globals::TableInterface;
          output.payload.tableValue.opaque = map;
        }
      }

      uint64_t len;
      std::string keyBuf;
      read((uint8_t *)&len, sizeof(uint64_t));
      for (uint64_t i = 0; i < len; i++) {
        uint32_t klen;
        read((uint8_t *)&klen, sizeof(uint32_t));
        keyBuf.resize(klen + 1);
        read((uint8_t *)keyBuf.c_str(), len);
        const_cast<char *>(keyBuf.c_str())[klen] = 0;
        auto dst = (*map)[keyBuf.c_str()];
        deserialize(read, dst);
      }
      break;
    }
    case CBType::Image: {
      size_t currentSize = 0;
      if (recycle) {
        currentSize = output.payload.imageValue.channels *
                      output.payload.imageValue.height *
                      output.payload.imageValue.width;
      }

      read((uint8_t *)&output.payload.imageValue.channels,
           sizeof(output.payload.imageValue.channels));
      read((uint8_t *)&output.payload.imageValue.flags,
           sizeof(output.payload.imageValue.flags));
      read((uint8_t *)&output.payload.imageValue.width,
           sizeof(output.payload.imageValue.width));
      read((uint8_t *)&output.payload.imageValue.height,
           sizeof(output.payload.imageValue.height));

      size_t size = output.payload.imageValue.channels *
                    output.payload.imageValue.height *
                    output.payload.imageValue.width;

      if (currentSize > 0 && currentSize < size) {
        // delete first & alloc
        delete[] output.payload.imageValue.data;
        output.payload.imageValue.data = new uint8_t[size];
      } else if (currentSize == 0) {
        // just alloc
        output.payload.imageValue.data = new uint8_t[size];
      }

      read((uint8_t *)output.payload.imageValue.data, size);
      break;
    }
    case CBType::Block: {
      CBlock *blk;
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));
      std::vector<char> buf;
      buf.resize(len + 1);
      read((uint8_t *)&buf[0], len);
      buf[len] = 0;
      blk = createBlock(&buf[0]);
      if (!blk) {
        throw CBException("Block not found! name: " + std::string(&buf[0]));
      }
      blk->setup(blk);
      // TODO we need some block hashing to validate maybe?
      auto params = blk->parameters(blk);
      for (uint32_t i = 0; i < params.len; i++) {
        CBVar tmp{};
        deserialize(read, tmp);
        blk->setParam(blk, int(i), tmp);
        varFree(tmp);
      }
      if (blk->setState) {
        CBVar state{};
        deserialize(read, state);
        blk->setState(blk, state);
        varFree(state);
      }
      output.payload.blockValue = blk;
      break;
    }
    case CBType::Chain: {
      uint32_t len;
      read((uint8_t *)&len, sizeof(uint32_t));
      std::vector<char> buf;
      buf.resize(len + 1);
      read((uint8_t *)&buf[0], len);
      buf[len] = 0;
      auto chain = new CBChain(&buf[0]);
      read((uint8_t *)&chain->looped, 1);
      read((uint8_t *)&chain->unsafe, 1);
      // blocks len
      read((uint8_t *)&len, sizeof(uint32_t));
      // blocks
      for (uint32_t i = 0; i < len; i++) {
        CBVar blockVar{};
        deserialize(read, blockVar);
        assert(blockVar.valueType == Block);
        chain->addBlock(blockVar.payload.blockValue);
        // blow's owner is the chain
      }
      // variables len
      read((uint8_t *)&len, sizeof(uint32_t));
      auto varsLen = len;
      for (uint32_t i = 0; i < varsLen; i++) {
        read((uint8_t *)&len, sizeof(uint32_t));
        buf.resize(len + 1);
        read((uint8_t *)&buf[0], len);
        buf[len] = 0;
        CBVar tmp{};
        deserialize(read, tmp);
        chain->variables[&buf[0]] = tmp;
      }
      output.payload.chainValue = chain->newRef();
      break;
    }
    case CBType::Object: {
      int64_t id;
      read((uint8_t *)&id, sizeof(int64_t));
      uint64_t len;
      read((uint8_t *)&len, sizeof(uint64_t));
      if (len > 0) {
        auto it = Globals::ObjectTypesRegister.find(id);
        if (it != Globals::ObjectTypesRegister.end()) {
          auto &info = it->second;
          auto size = size_t(len);
          std::vector<uint8_t> data;
          data.resize(size);
          read((uint8_t *)&data[0], size);
          output.payload.objectValue = info.deserialize(&data[0], size);
          int32_t vendorId = (int32_t)((id & 0xFFFFFFFF00000000) >> 32);
          int32_t typeId = (int32_t)(id & 0x00000000FFFFFFFF);
          output.payload.objectVendorId = vendorId;
          output.payload.objectTypeId = typeId;
          output.flags |= CBVAR_FLAGS_USES_OBJINFO;
          output.objectInfo = &info;
          // up ref count also, not included in deserialize op!
          if (info.reference) {
            info.reference(output.payload.objectValue);
          }
        }
      }
      break;
    }
    }
  }

  template <class BinaryWriter>
  static inline size_t serialize(const CBVar &input, BinaryWriter &write) {
    size_t total = 0;
    write((const uint8_t *)&input.valueType, sizeof(input.valueType));
    total += sizeof(input.valueType);
    switch (input.valueType) {
    case CBType::None:
    case CBType::EndOfBlittableTypes:
    case CBType::Any:
      break;
    case CBType::Enum:
      write((const uint8_t *)&input.payload, sizeof(int32_t) * 3);
      total += sizeof(int32_t) * 3;
      break;
    case CBType::Bool:
      write((const uint8_t *)&input.payload, sizeof(CBBool));
      total += sizeof(CBBool);
      break;
    case CBType::Int:
      write((const uint8_t *)&input.payload, sizeof(CBInt));
      total += sizeof(CBInt);
      break;
    case CBType::Int2:
      write((const uint8_t *)&input.payload, sizeof(CBInt2));
      total += sizeof(CBInt2);
      break;
    case CBType::Int3:
      write((const uint8_t *)&input.payload, sizeof(CBInt3));
      total += sizeof(CBInt3);
      break;
    case CBType::Int4:
      write((const uint8_t *)&input.payload, sizeof(CBInt4));
      total += sizeof(CBInt4);
      break;
    case CBType::Int8:
      write((const uint8_t *)&input.payload, sizeof(CBInt8));
      total += sizeof(CBInt8);
      break;
    case CBType::Int16:
      write((const uint8_t *)&input.payload, sizeof(CBInt16));
      total += sizeof(CBInt16);
      break;
    case CBType::Float:
      write((const uint8_t *)&input.payload, sizeof(CBFloat));
      total += sizeof(CBFloat);
      break;
    case CBType::Float2:
      write((const uint8_t *)&input.payload, sizeof(CBFloat2));
      total += sizeof(CBFloat2);
      break;
    case CBType::Float3:
      write((const uint8_t *)&input.payload, sizeof(CBFloat3));
      total += sizeof(CBFloat3);
      break;
    case CBType::Float4:
      write((const uint8_t *)&input.payload, sizeof(CBFloat4));
      total += sizeof(CBFloat4);
      break;
    case CBType::Color:
      write((const uint8_t *)&input.payload, sizeof(CBColor));
      total += sizeof(CBColor);
      break;
    case CBType::StackIndex:
      write((const uint8_t *)&input.payload, sizeof(int64_t));
      total += sizeof(int64_t);
      break;
    case CBType::Bytes:
      write((const uint8_t *)&input.payload.bytesSize,
            sizeof(input.payload.bytesSize));
      total += sizeof(input.payload.bytesSize);
      write((const uint8_t *)input.payload.bytesValue, input.payload.bytesSize);
      total += input.payload.bytesSize;
      break;
    case CBType::Array:
      write((const uint8_t *)&input.innerType, sizeof(input.innerType));
      total += sizeof(input.innerType);
      write((const uint8_t *)&input.payload.arrayLen,
            sizeof(input.payload.arrayLen));
      total += sizeof(input.payload.arrayLen);
      write((const uint8_t *)input.payload.arrayValue,
            input.payload.arrayLen * sizeof(CBVarPayload));
      total += input.payload.arrayLen * sizeof(CBVarPayload);
      break;
    case CBType::Path:
    case CBType::String:
    case CBType::ContextVar: {
      uint32_t len = input.payload.stringLen > 0
                         ? input.payload.stringLen
                         : strlen(input.payload.stringValue);
      write((const uint8_t *)&len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      write((const uint8_t *)input.payload.stringValue, len);
      total += len;
      break;
    }
    case CBType::Seq: {
      uint32_t len = input.payload.seqValue.len;
      write((const uint8_t *)&len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      for (uint32_t i = 0; i < len; i++) {
        total += serialize(input.payload.seqValue.elements[i], write);
      }
      break;
    }
    case CBType::Table: {
      if (input.payload.tableValue.api && input.payload.tableValue.opaque) {
        auto &t = input.payload.tableValue;
        uint64_t len = (uint64_t)t.api->tableSize(t);
        write((const uint8_t *)&len, sizeof(uint64_t));
        total += sizeof(uint64_t);
        struct iterdata {
          BinaryWriter *write;
          size_t *total;
        } data;
        data.write = &write;
        data.total = &total;
        t.api->tableForEach(
            t,
            [](const char *key, CBVar *value, void *_data) {
              auto data = (iterdata *)_data;
              uint32_t klen = strlen(key);
              (*data->write)((const uint8_t *)&klen, sizeof(uint32_t));
              *data->total += sizeof(uint32_t);
              (*data->write)((const uint8_t *)key, klen);
              *data->total += klen;
              *data->total += serialize(*value, *data->write);
              return true;
            },
            &data);
      } else {
        uint64_t none = 0;
        write((const uint8_t *)&none, sizeof(uint64_t));
        total += sizeof(uint64_t);
      }
      break;
    }
    case CBType::Image: {
      write((const uint8_t *)&input.payload.imageValue.channels,
            sizeof(input.payload.imageValue.channels));
      total += sizeof(input.payload.imageValue.channels);
      write((const uint8_t *)&input.payload.imageValue.flags,
            sizeof(input.payload.imageValue.flags));
      total += sizeof(input.payload.imageValue.flags);
      write((const uint8_t *)&input.payload.imageValue.width,
            sizeof(input.payload.imageValue.width));
      total += sizeof(input.payload.imageValue.width);
      write((const uint8_t *)&input.payload.imageValue.height,
            sizeof(input.payload.imageValue.height));
      total += sizeof(input.payload.imageValue.height);
      auto size = input.payload.imageValue.channels *
                  input.payload.imageValue.height *
                  input.payload.imageValue.width;
      write((const uint8_t *)input.payload.imageValue.data, size);
      total += size;
      break;
    }
    case CBType::Block: {
      auto blk = input.payload.blockValue;
      // name
      auto name = blk->name(blk);
      uint32_t len = strlen(name);
      write((const uint8_t *)&len, sizeof(uint32_t));
      total += sizeof(uint32_t);
      write((const uint8_t *)name, len);
      total += len;
      // params
      auto params = blk->parameters(blk);
      for (uint32_t i = 0; i < params.len; i++) {
        auto pval = blk->getParam(blk, int(i));
        total += serialize(pval, write);
      }
      // optional state
      if (blk->getState) {
        auto state = blk->getState(blk);
        total += serialize(state, write);
      }
      break;
    }
    case CBType::Chain: {
      auto sc = reinterpret_cast<std::shared_ptr<CBChain> *>(
          input.payload.chainValue);
      auto chain = sc->get();
      { // Name
        uint32_t len = uint32_t(chain->name.size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
        write((const uint8_t *)chain->name.c_str(), len);
        total += len;
      }
      { // Looped & Unsafe
        write((const uint8_t *)&chain->looped, 1);
        total += 1;
        write((const uint8_t *)&chain->unsafe, 1);
        total += 1;
      }
      { // Blocks len
        uint32_t len = uint32_t(chain->blocks.size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
      }
      // Blocks
      for (auto block : chain->blocks) {
        CBVar blockVar{};
        blockVar.valueType = CBType::Block;
        blockVar.payload.blockValue = block;
        total += serialize(blockVar, write);
      }
      { // Variables len
        uint32_t len = uint32_t(chain->variables.size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
      }
      // Variables
      for (auto &var : chain->variables) {
        uint32_t len = uint32_t(var.first.size());
        write((const uint8_t *)&len, sizeof(uint32_t));
        total += sizeof(uint32_t);
        write((const uint8_t *)var.first.c_str(), len);
        total += len;
        // Serialization discards anything cept payload
        // That is what we want anyway!
        total += serialize(var.second, write);
      }
      break;
    }
    case CBType::Object: {
      int64_t id = (int64_t)input.payload.objectVendorId << 32 |
                   input.payload.objectTypeId;
      write((const uint8_t *)&id, sizeof(int64_t));
      total += sizeof(int64_t);
      if ((input.flags & CBVAR_FLAGS_USES_OBJINFO) ==
              CBVAR_FLAGS_USES_OBJINFO &&
          input.objectInfo && input.objectInfo->serialize) {
        size_t len = 0;
        uint8_t *data = nullptr;
        CBPointer handle = nullptr;
        if (!input.objectInfo->serialize(input.payload.objectValue, &data, &len,
                                         &handle)) {
          throw chainblocks::CBException(
              "Failed to serialize custom object variable!");
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
    }
    return total;
  }
};

struct InternalCore {
  // need to emulate dllblock Core a bit
  static CBVar *referenceVariable(CBContext *context, const char *name) {
    return chainblocks::referenceVariable(context, name);
  }

  static void releaseVariable(CBVar *variable) {
    chainblocks::releaseVariable(variable);
  }

  static CBSeq *getStack(CBContext *context);

  static void cloneVar(CBVar &dst, const CBVar &src) {
    chainblocks::cloneVar(dst, src);
  }

  static void destroyVar(CBVar &var) { chainblocks::destroyVar(var); }

  template <typename T> static void arrayFree(T &arr) {
    chainblocks::arrayFree<T>(arr);
  }

  static void expTypesFree(CBExposedTypesInfo &arr) { arrayFree(arr); }

  [[noreturn]] static void throwException(const char *msg) {
    throw chainblocks::CBException(msg);
  }

  static void log(const char *msg) { LOG(INFO) << msg; }

  static void registerEnumType(int32_t vendorId, int32_t enumId,
                               CBEnumInfo info) {
    chainblocks::registerEnumType(vendorId, enumId, info);
  }

  static CBValidationResult validateBlocks(CBlocks blocks,
                                           CBValidationCallback callback,
                                           void *userData,
                                           CBInstanceData data) {
    return validateConnections(blocks, callback, userData, data);
  }

  static CBChainState runBlocks(CBlocks blocks, CBContext *context, CBVar input,
                                CBVar *output) {
    return chainblocks::activateBlocks(blocks, context, input, *output);
  }

  static CBChainState suspend(CBContext *ctx, double seconds) {
    return chainblocks::suspend(ctx, seconds);
  }
};

typedef TParamVar<InternalCore> ParamVar;

template <Parameters &Params, size_t NPARAMS, Type &InputType, Type &OutputType>
struct SimpleBlock : public TSimpleBlock<InternalCore, Params, NPARAMS,
                                         InputType, OutputType> {};

template <typename E> class EnumInfo : public TEnumInfo<InternalCore, E> {
public:
  EnumInfo(const char *name, int32_t vendorId, int32_t enumId)
      : TEnumInfo<InternalCore, E>(name, vendorId, enumId) {}
};

typedef TBlocksVar<InternalCore> BlocksVar;

struct ParamsInfo {
  /*
   DO NOT USE ANYMORE, THIS IS DEPRECATED AND MESSY
   IT WAS USEFUL WHEN DYNAMIC ARRAYS WERE STB ARRAYS
   BUT NOW WE CAN JUST USE DESIGNATED/AGGREGATE INITIALIZERS
 */
  ParamsInfo(const ParamsInfo &other) {
    chainblocks::arrayResize(_innerInfo, 0);
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      chainblocks::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }
  }

  ParamsInfo &operator=(const ParamsInfo &other) {
    _innerInfo = {};
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      chainblocks::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }
    return *this;
  }

  template <typename... Types>
  explicit ParamsInfo(CBParameterInfo first, Types... types) {
    std::vector<CBParameterInfo> vec = {first, types...};
    _innerInfo = {};
    for (auto pi : vec) {
      chainblocks::arrayPush(_innerInfo, pi);
    }
  }

  template <typename... Types>
  explicit ParamsInfo(const ParamsInfo &other, CBParameterInfo first,
                      Types... types) {
    _innerInfo = {};

    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      chainblocks::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }

    std::vector<CBParameterInfo> vec = {first, types...};
    for (auto pi : vec) {
      chainblocks::arrayPush(_innerInfo, pi);
    }
  }

  static CBParameterInfo Param(const char *name, const char *help,
                               CBTypesInfo types) {
    CBParameterInfo res = {name, help, types};
    return res;
  }

  ~ParamsInfo() {
    if (_innerInfo.elements)
      chainblocks::arrayFree(_innerInfo);
  }

  explicit operator CBParametersInfo() const { return _innerInfo; }

  CBParametersInfo _innerInfo{};
};

struct ExposedInfo {
  /*
   DO NOT USE ANYMORE, THIS IS DEPRECATED AND MESSY
   IT WAS USEFUL WHEN DYNAMIC ARRAYS WERE STB ARRAYS
   BUT NOW WE CAN JUST USE DESIGNATED/AGGREGATE INITIALIZERS
 */
  ExposedInfo() { _innerInfo = {}; }

  ExposedInfo(const ExposedInfo &other) {
    _innerInfo = {};
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      chainblocks::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }
  }

  ExposedInfo &operator=(const ExposedInfo &other) {
    chainblocks::arrayResize(_innerInfo, 0);
    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      chainblocks::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }
    return *this;
  }

  template <typename... Types>
  explicit ExposedInfo(const ExposedInfo &other, Types... types) {
    _innerInfo = {};

    for (uint32_t i = 0; i < other._innerInfo.len; i++) {
      chainblocks::arrayPush(_innerInfo, other._innerInfo.elements[i]);
    }

    std::vector<CBExposedTypeInfo> vec = {types...};
    for (auto pi : vec) {
      chainblocks::arrayPush(_innerInfo, pi);
    }
  }

  template <typename... Types>
  explicit ExposedInfo(const CBExposedTypesInfo other, Types... types) {
    _innerInfo = {};

    for (uint32_t i = 0; i < other.len; i++) {
      chainblocks::arrayPush(_innerInfo, other.elements[i]);
    }

    std::vector<CBExposedTypeInfo> vec = {types...};
    for (auto pi : vec) {
      chainblocks::arrayPush(_innerInfo, pi);
    }
  }

  template <typename... Types>
  explicit ExposedInfo(const CBExposedTypeInfo first, Types... types) {
    std::vector<CBExposedTypeInfo> vec = {first, types...};
    _innerInfo = {};
    for (auto pi : vec) {
      chainblocks::arrayPush(_innerInfo, pi);
    }
  }

  static CBExposedTypeInfo Variable(const char *name, const char *help,
                                    CBTypeInfo type, bool isMutable = false,
                                    bool isTableField = false) {
    CBExposedTypeInfo res = {name, help, type, isMutable, isTableField, false};
    return res;
  }

  static CBExposedTypeInfo GlobalVariable(const char *name, const char *help,
                                          CBTypeInfo type,
                                          bool isMutable = false,
                                          bool isTableField = false) {
    CBExposedTypeInfo res = {name, help, type, isMutable, isTableField, true};
    return res;
  }

  ~ExposedInfo() {
    if (_innerInfo.elements)
      chainblocks::arrayFree(_innerInfo);
  }

  explicit operator CBExposedTypesInfo() const { return _innerInfo; }

  CBExposedTypesInfo _innerInfo;
};

struct CachedStreamBuf : std::streambuf {
  std::vector<char> data;
  void reset() { data.clear(); }

  int overflow(int c) override {
    data.push_back(static_cast<char>(c));
    return 0;
  }

  void done() { data.push_back('\0'); }

  const char *str() { return &data[0]; }
};

struct VarStringStream {
  CachedStreamBuf cache;
  CBVar previousValue{};

  ~VarStringStream() { destroyVar(previousValue); }

  void write(const CBVar &var) {
    if (var != previousValue) {
      cache.reset();
      std::ostream stream(&cache);
      stream << var;
      cache.done();
      cloneVar(previousValue, var);
    }
  }

  void tryWriteHex(const CBVar &var) {
    if (var != previousValue) {
      cache.reset();
      std::ostream stream(&cache);
      if (var.valueType == Int) {
        stream << "0x" << std::hex << var.payload.intValue << std::dec;
      } else {
        stream << var;
      }
      cache.done();
      cloneVar(previousValue, var);
    }
  }

  const char *str() { return cache.str(); }
};

using BlocksCollection = std::variant<CBChain *, CBlockPtr, CBlocks, CBVar>;

struct CBlockInfo {
  CBlockInfo(const std::string_view &name, const CBlock *block)
      : name(name), block(block) {}
  std::string_view name;
  const CBlock *block;
};

void gatherBlocks(const BlocksCollection &coll, std::vector<CBlockInfo> &out);
}; // namespace chainblocks

#endif
