#ifndef C599F883_4CF5_416C_AC0D_AF07729F5C2B
#define C599F883_4CF5_416C_AC0D_AF07729F5C2B

#include <shards/shards.hpp>
#include <boost/tti/has_static_member_function.hpp>
#include <boost/tti/has_member_function.hpp>

#ifndef SH_NO_INTERNAL_CORE
#include "foundation.hpp"
#endif

namespace shards {
namespace detail {
BOOST_TTI_TRAIT_HAS_STATIC_MEMBER_FUNCTION(HasStaticMember_match, match);
BOOST_TTI_TRAIT_HAS_STATIC_MEMBER_FUNCTION(HasStaticMember_hash, hash);
} // namespace detail

template <bool Atomic> struct RefCount {
  uint32_t count;

  void reference() { ++count; }
  bool release() {
    --count;
    return count == 0;
  }
};

template <> struct RefCount<true> {
  std::atomic<uint32_t> count;

  void reference() { ++count; }
  bool release() { return count.fetch_sub(1) == 1; }
};

template <class SH_CORE, typename E, std::vector<uint8_t> (*Serializer)(const E &) = nullptr,
          E (*Deserializer)(const std::string_view &) = nullptr, void (*BeforeDelete)(const E &) = nullptr,
          bool IsThreadSafe = false>
class TObjectVar {
private:
  SHObjectInfo info;
  int32_t vendorId;
  int32_t typeId;

  struct ObjectRef {
    E shared;
    RefCount<IsThreadSafe> refcount;
  };

public:
  TObjectVar(const char *name, int32_t vendorId, int32_t typeId) : vendorId(vendorId), typeId(typeId) {
    info = {};
    info.name = name;
    info.isThreadSafe = IsThreadSafe;
    info.reference = [](SHPointer ptr) {
      auto p = reinterpret_cast<ObjectRef *>(ptr);
      p->refcount.reference();
    };
    info.release = [](SHPointer ptr) {
      auto p = reinterpret_cast<ObjectRef *>(ptr);
      if (p->refcount.release()) {
        delete p;
      }
    };
    if (Serializer != nullptr && Deserializer != nullptr) {
      info.serialize = [](SHPointer obj, uint8_t **outData, uint64_t *outLen, SHPointer *customHandle) {
        auto tobj = reinterpret_cast<E *>(obj);
        auto holder = new std::vector<uint8_t>();
        *holder = Serializer(*tobj);
        *customHandle = holder;
        *outData = holder->data();
        *outLen = holder->size();
        return true;
      };
      info.free = [](SHPointer handle) {
        auto holder = reinterpret_cast<std::vector<uint8_t> *>(handle);
        delete holder;
      };
      info.deserialize = [](uint8_t *data, uint64_t len) {
        auto r = new ObjectRef();
        r->shared = Deserializer(std::string_view((char *)data, len));
        // don't bump ref count, deserializer is supposed to do that
        return (SHPointer)r;
      };
    }
    SH_CORE::registerObjectType(vendorId, typeId, info);
  }

  // the following methods are generally used by the shard
  // that creates the object, that's it.
  // other shards use regular referenceVariable etc.

  E *New() {
    auto r = new ObjectRef();
    r->refcount.count = 1;
    return &r->shared;
  }

  std::tuple<E &, TOwnedVar<SH_CORE>> NewOwnedVar() {
    E *ptr = New();
    SHVar v = Get(ptr);
    return {*ptr, std::move((TOwnedVar<SH_CORE> &)v)};
  }

  Type GetObjectType() { return Type::Object(vendorId, typeId); }
  operator Type() { return GetObjectType(); }

  template <typename ST> E *Emplace(std::shared_ptr<ST> &&obj) {
    auto r = new ObjectRef{.shared = E(std::move(obj)), .refcount = RefCount<IsThreadSafe>{.count = 1}};
    return &r->shared;
  }

  void Release(E *obj) {
    auto r = reinterpret_cast<ObjectRef *>(obj);
    if (r->refcount.release()) {
      if constexpr (BeforeDelete != nullptr) {
        BeforeDelete(*obj);
      }
      delete r;
    }
  }

  uint32_t GetRefCount(E *obj) {
    auto r = reinterpret_cast<ObjectRef *>(obj);
    return r->refcount.count;
  }

  SHVar Get(E *obj) {
    SHVar res;
    res.valueType = SHType::Object;
    res.payload.objectValue = obj;
    res.payload.objectVendorId = vendorId;
    res.payload.objectTypeId = typeId;
    res.flags = SHVAR_FLAGS_USES_OBJINFO;
    res.objectInfo = &info;
    return res;
  }
};

template <typename T> struct ExtendedObjectTypeTraits {
  static inline const SHTypeInfo &getTypeInfo() { return (const SHTypeInfo &)T::Type; }
};

template <class SH_CORE, typename T, typename Traits = ExtendedObjectTypeTraits<T>> class TExtendedObjectType {
  struct Extended {
    T data;
    RefCount<true> refCount;

    Extended() { refCount.count = 1; }
    Extended(const Extended &other) : data(other.data) { refCount.count = 1; }
  };

public:
  static inline const SHExtendedObjectTypeInfo &getTypeInfo() {
    static SHExtendedObjectTypeInfo info = []() {
      SHExtendedObjectTypeInfo info{
          .reference =
              [](SHPointer ptr) {
                auto p = reinterpret_cast<Extended *>(ptr);
                p->refCount.reference();
              },
          .release =
              [](SHPointer ptr) {
                auto p = reinterpret_cast<Extended *>(ptr);
                if (p->refCount.release()) {
                  delete p;
                }
              },
      };
      if constexpr (detail::HasStaticMember_match<T, bool(const T &, const T &)>::value) {
        info.match = [](const void *self, const void *other) {
          auto tself = reinterpret_cast<const T *>(self);
          auto tother = reinterpret_cast<const T *>(other);
          return T::match(*tself, *tother);
        };
      }
      if constexpr (detail::HasStaticMember_hash<T, void(const T &, void *outDigest, size_t digestSize)>::value) {
        info.hash = [](const void *self, void *outDigest, uint32_t digestSize) {
          auto tself = reinterpret_cast<const T *>(self);
          return T::hash(*tself, outDigest, size_t(digestSize));
        };
      }
      return info;
    }();
    return info;
  }

  // Create a new extended type into the target TypeInfo
  // returns the extened type info created, if one was already set,
  //  the new extended type is constructed from the old one using the copy constructor
  static inline T &makeExtended(TypeInfo &dst) {
    TypeInfo old;
    std::swap(old, dst);
    return makeExtended(dst, &(SHTypeInfo&)old);
  }
  static inline T &makeExtended(TypeInfo &dst, const SHTypeInfo *original) {
    if (dst->basicType != SHType::Object) {
      dst = SHTypeInfo{};
    }

    Extended *e{};
    if (original && original->basicType == SHType::Object && original->object.extInfo) {
      e = new Extended(*reinterpret_cast<Extended *>(original->object.extInfoData));
    } else {
      e = new Extended();
    }

    const Type &type = Traits::getTypeInfo();
    dst->basicType = SHType::Object;
    dst->object = type->object;
    dst->object.extInfo = &getTypeInfo();
    dst->object.extInfoData = e;

    return e->data;
  }

  static T &fromTypeInfo(const SHTypeInfo &ti) {
    shassert(ti.basicType == SHType::Object && ti.object.extInfo == &getTypeInfo());
    return ((Extended *)ti.object.extInfoData)->data;
  }
};

#ifndef SH_NO_INTERNAL_CORE
template <typename E, std::vector<uint8_t> (*Serializer)(const E &) = nullptr,
          E (*Deserializer)(const std::string_view &) = nullptr, void (*BeforeDelete)(const E &) = nullptr,
          bool ThreadSafe = false>
class ObjectVar : public TObjectVar<InternalCore, E, Serializer, Deserializer, BeforeDelete, ThreadSafe> {
public:
  ObjectVar(const char *name, int32_t vendorId, int32_t objectId)
      : TObjectVar<InternalCore, E, Serializer, Deserializer, BeforeDelete, ThreadSafe>(name, vendorId, objectId) {}
};
#endif

template<typename TObjectType, int32_t _ObjectId, int32_t _VendorId, const char* _Name>
struct TObjectTypeInfo {
  static inline int32_t ObjectId = _ObjectId;
  static inline ::shards::Type Type = ::shards::Type::Object(_VendorId, _ObjectId);
  static inline SHTypeInfo RawType = Type;
  static inline ::shards::Type VarType = ::shards::Type::VariableOf(Type);

  static inline shards::ObjectVar<TObjectType, nullptr, nullptr, nullptr, true> ObjectVar{_Name, RawType.object.vendorId, RawType.object.typeId};

  static TObjectType& fromVarChecked(const SHVar& var) {
    return shards::varAsObjectChecked<TObjectType>(var, RawType);
  }

  static shards::OwnedVar newOwnedVar() {
    return std::get<1>(ObjectVar.NewOwnedVar());
  }
};

}; // namespace shards

#endif /* C599F883_4CF5_416C_AC0D_AF07729F5C2B */
