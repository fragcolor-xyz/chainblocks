/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#ifndef SH_SHARDS_H
#define SH_SHARDS_H

#include <stdbool.h>
#include <stdint.h>

#if defined(__cplusplus) && !defined(RUST_BINDGEN)
#define SH_ENUM_CLASS class
#define SH_ENUM_DECL
#else
#define SH_ENUM_CLASS
#define SH_ENUM_DECL enum
#endif

#define SH_USE_ENUMS 1
// All the available types
#if defined(__cplusplus) || defined(SH_USE_ENUMS)
enum SH_ENUM_CLASS SHType : uint8_t {
  // Blittables
  None,
  Any,
  Enum,
  Bool,
  Int,    // A 64bits int
  Int2,   // A vector of 2 64bits ints
  Int3,   // A vector of 3 32bits ints
  Int4,   // A vector of 4 32bits ints
  Int8,   // A vector of 8 16bits ints
  Int16,  // A vector of 16 8bits ints
  Float,  // A 64bits float
  Float2, // A vector of 2 64bits floats
  Float3, // A vector of 3 32bits floats
  Float4, // A vector of 4 32bits floats
  Color,  // A vector of 4 uint8

  // Internal use only
  EndOfBlittableTypes = 50, // anything below this is not blittable (ish)

  // Non Blittables
  Bytes, // pointer + size
  String,
  Path,       // An OS filesystem path
  ContextVar, // A string label to find from SHContext variables
  Image,
  Seq,
  Table,
  Wire,
  ShardRef, // a shard, useful for future introspection shards!
  Object = 60,
  // Array, // Notice: of just blittable types - Reserved for future use - 61
  // Set, // Reserved for future use - 62
  Audio = 63,
  Type, // Describes a type
  Trait // A wire trait
};

enum SH_ENUM_CLASS SHWireState : uint8_t {
  Continue, // Nothing happened, continue
  Return,   // Control flow, end this wire/flow and return previous output
  Rebase,   // Continue but put the local wire initial input as next input
  Restart,  // Restart the current wire from the top (non inline wires)
  Stop,     // Stop the flow execution
  Error,    // Stop the flow execution and raise an error
};

#else
typedef uint8_t SHType;
typedef uint8_t SHWireState;
#endif

// Enum defined in shards/inlined.hpp
typedef uint32_t SHInlineShards;

typedef void *SHArray;

// This (32bit sizes) is for obvious packing reason, sizeof(SHVar) == 32
// But also for compatibility, shards supports 32bit systems
// So the max array size should be INT32_SIZE indeed
// Also most of array index operators in c++ use int
#define SH_ARRAY_DECL(_seq_, _val_) \
  typedef struct _seq_ {            \
    _val_ *elements;                \
    uint32_t len;                   \
    uint32_t cap;                   \
  } _seq_

// Forward declarations

struct SHVarPayload;
SH_ARRAY_DECL(SHPayloadArray, struct SHVarPayload);

struct SHVar;
SH_ARRAY_DECL(SHSeq, struct SHVar);

struct SHTableImpl;

struct SHTableInterface;
// 64 bytes should be huge and well enough space for an iterator...
typedef char SHTableIterator[64];
struct SHTable {
  struct SHTableImpl *opaque;
  struct SHTableInterface *api;
};

#if !defined(NDEBUG) && !defined(SH_RELWITHDEBINFO)
const uint32_t SH_MIN_STACK_SIZE = 2 * 1024 * 1024; // 2 MB for dev builds
#else
const uint32_t SH_MIN_STACK_SIZE = 128 * 1024; // 128 KB for non-dev builds
#endif

struct SHWire;
struct SHWireRefOpaque;
typedef struct SHWireRefOpaque *SHWireRef;

struct SHWireProvider;

struct SHContext;

struct SHMesh;
struct SHMeshRefOpaque;
typedef struct SHMeshRefOpaque *SHMeshRef;

struct Shard;
typedef struct Shard *ShardPtr;
SH_ARRAY_DECL(Shards, ShardPtr);

struct SHTypeInfo;
SH_ARRAY_DECL(SHTypesInfo, struct SHTypeInfo);

struct SHParameterInfo;
SH_ARRAY_DECL(SHParametersInfo, struct SHParameterInfo);

struct SHExposedTypeInfo;
SH_ARRAY_DECL(SHExposedTypesInfo, struct SHExposedTypeInfo);

typedef void *SHPointer;
typedef int64_t SHInt;
typedef double SHFloat;
typedef bool SHBool;
typedef int32_t SHEnum;
SH_ARRAY_DECL(SHEnums, SHEnum);

typedef const char *SHString;
SH_ARRAY_DECL(SHStrings, SHString);
// used in order to compress/omit/localize the strings at runtime
// specially for help functions
// if string is null, crc is checked and used to retrieve a string
typedef struct _SHOptionalString {
  SHString string;
  uint32_t crc;
} SHOptionalString;
SH_ARRAY_DECL(SHOptionalStrings, SHOptionalString);

// used in hot paths to avoid extra strlen calls
typedef struct SHStringWithLen {
  SHString string;
  uint64_t len;
} SHStringWithLen;

#if defined(__clang__) || defined(__GNUC__)
#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#ifdef __clang__
#define shufflevector __builtin_shufflevector
#else
#define shufflevector __builtin_shuffle
#endif

typedef int64_t SHInt2 __attribute__((vector_size(16)));
typedef int32_t SHInt3 __attribute__((vector_size(16)));
typedef int32_t SHInt4 __attribute__((vector_size(16)));
typedef int16_t SHInt8 __attribute__((vector_size(16)));
typedef int8_t SHInt16 __attribute__((vector_size(16)));

typedef double SHFloat2 __attribute__((vector_size(16)));
typedef float SHFloat3 __attribute__((vector_size(16)));
typedef float SHFloat4 __attribute__((vector_size(16)));

#define NO_INLINE __attribute__((noinline))

#if defined(NDEBUG) && !defined(NO_FORCE_INLINE)
#define ALWAYS_INLINE __attribute__((always_inline))
#define FLATTEN __attribute__((flatten))
#else
#define ALWAYS_INLINE
#define FLATTEN
#endif

#else // TODO
#error "Unsupported compiler"

typedef int64_t SHInt2[2];
typedef int32_t SHInt3[3];
typedef int32_t SHInt4[4];
typedef int16_t SHInt8[8];
typedef int8_t SHInt16[16];

typedef double SHFloat2[2];
typedef float SHFloat3[3];
typedef float SHFloat4[4];

#define ALWAYS_INLINE
#define NO_INLINE
#define FLATTEN

#endif

#ifndef _WIN32
#if defined(__i386__)
#define __cdecl __attribute__((__cdecl__))
#else
#define __cdecl
#endif
#endif

struct SHColor {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
};

#define SHIMAGE_FLAGS_NONE (0)
#define SHIMAGE_FLAGS_BGRA (1 << 0)
#define SHIMAGE_FLAGS_PREMULTIPLIED_ALPHA (1 << 1)
#define SHIMAGE_FLAGS_16BITS_INT (1 << 2)
#define SHIMAGE_FLAGS_32BITS_FLOAT (1 << 3)

struct SHImage;
typedef void(__cdecl *SHImageFreeProc)(struct SHImage *image);

struct SHImage {
  uint32_t refCount;
  uint16_t width;
  uint16_t height;
  uint16_t rowStride;
  uint8_t channels;
  uint8_t flags;
  uint8_t *data;
  SHImageFreeProc free;
} __attribute__((aligned(16)));

struct SHAudio {
  uint32_t sampleRate; // set to 0 if unknown/not relevant
  uint16_t nsamples;
  uint16_t channels;
  float *samples;
};

#define SH_FLOW_CONTINUE (0)
#define SH_FLOW_ERROR (1 << 0)
#define SH_FLOW_CHANGE_STATE (1 << 1)

struct SHError {
  uint8_t code; // 0 if no error, 1 if error so far
  struct SHStringWithLen message;

#ifdef __cplusplus
  static const SHError Success;
#endif
};

#define SH_ERROR_NONE (0)

#ifdef __cplusplus
constexpr const SHError SHError::Success = {SH_ERROR_NONE, {nullptr, 0}};
#endif

// table interface
typedef void(__cdecl *SHTableGetIterator)(struct SHTable table, SHTableIterator *outIter);
typedef SHBool(__cdecl *SHTableNext)(struct SHTable table, SHTableIterator *inIter, struct SHVar *outKey, struct SHVar *outValue);
typedef uint64_t(__cdecl *SHTableSize)(struct SHTable table);
typedef SHBool(__cdecl *SHTableContains)(struct SHTable table, struct SHVar key);
typedef struct SHVar *(__cdecl *SHTableAt)(struct SHTable table, struct SHVar key);
typedef struct SHVar *(__cdecl *SHTableGet)(struct SHTable table, struct SHVar key);
typedef void(__cdecl *SHTableRemove)(struct SHTable table, struct SHVar key);
typedef void(__cdecl *SHTableClear)(struct SHTable table);
typedef void(__cdecl *SHTableFree)(struct SHTable table);

struct SHTableInterface {
  SHTableGetIterator tableGetIterator;
  SHTableNext tableNext;
  SHTableSize tableSize;
  SHTableContains tableContains;
  SHTableAt tableAt;
  SHTableGet tableGet;
  SHTableRemove tableRemove;
  SHTableClear tableClear;
  SHTableFree tableFree;
};

#ifdef SH_NO_ANON
#define SH_STRUCT(_name_) struct _name_
#define SH_UNION(_name_) union _name_
#define SH_UNION_NAME(_name_) _name_
#else
#define SH_STRUCT(_name_) struct
#define SH_UNION(_name_) union
#define SH_UNION_NAME(_name_)
#endif

SH_ARRAY_DECL(SHTraits, struct SHTrait);

typedef struct SHTableTypeInfo {
  // If tableKeys is populated, it is expected that
  // tableTypes will be populated as well and that at the same
  // key index there is the key's type
  SHSeq keys;
  // If tableKeys is not populated, len == 0 and tableKeys is populated len
  // > 0 it is assumed that tableTypes contains a sequence with the possible
  // types in the table
  SHTypesInfo types;
} SHTableTypeInfo;

typedef struct SHExtendedObjectTypeInfo SHExtendedObjectTypeInfo;
typedef struct SHObjectTypeInfo {
  int32_t vendorId;
  int32_t typeId;
  // Provides additional object behaviors, such as comparision, special types
  const SHExtendedObjectTypeInfo *extInfo;
  // User data passed that can be used by the extended object type functions
  void *extInfoData;
} SHObjectTypeInfo;

typedef struct SHEnumTypeInfo {
  int32_t vendorId;
  int32_t typeId;
} SHEnumTypeInfo;

struct SHTypeInfo {
#if defined(__cplusplus) || defined(SH_USE_ENUMS)
  SH_ENUM_DECL SHType basicType;
#else
  SHType basicType;
#endif

  SH_UNION(Details) {
    SHObjectTypeInfo object;

    SHEnumTypeInfo enumeration;

    SHTypesInfo seqTypes;
    SHTypesInfo setTypes;

    SHTableTypeInfo table;

    SHTypesInfo contextVarTypes;

    SH_STRUCT(Path) {
      // if is File, the extension allowed
      SHStrings extensions;
      // expects a path to a file
      SHBool isFile;
      // expects an existing path
      SHBool existing;
      // expects a relative path (relative to the SHCore.getRootPath)
      SHBool relative;
    }
    path;

    // Not meant for wire runtime checks
    // rather validation/compose checks of params and mutations
    SH_STRUCT(Integers) {
      int64_t min;
      int64_t max;
      SHBool valid;
    }
    integers;

    // Not meant for wire runtime checks
    // rather validation/compose checks of params and mutations
    SH_STRUCT(Real) {
      double min;
      double max;
      SHBool valid;
    }
    real;
  }
  SH_UNION_NAME(details);

  // used for Seq and Array only for now, to allow optimizations if the size is
  // known at compose time.
  // Should not be considered when hashing this type
  uint32_t fixedSize;
  // Used by Array type, which is still not implemented properly and unstable.
  SH_ENUM_DECL SHType innerType;
  // used internally to make our live easy when types are recursive (aka Self is
  // inside the seqTypes or so)
  // Should not be considered when hashing this type
  SHBool recursiveSelf;
};

typedef struct SHTraitVariable {
  // Name of the variable
  SHStringWithLen name;
  // Expected variable type
  struct SHTypeInfo type;
} SHTraitVariable;
SH_ARRAY_DECL(SHTraitVariables, struct SHTraitVariable);

typedef struct SHTrait {
  // Unique Id of the trait, which is also the hash of the items
  uint64_t id[2];
  // Friendly name given to this trait, unhashed
  SHStringWithLen name;
  // List of variables
  SHTraitVariables variables;
} SHTrait;

struct SHShardComposeResult {
  struct SHError error;
  struct SHTypeInfo result;
};

// if outData is NULL will just give you a valid outLen
// still must check result is true!
typedef SHBool(__cdecl *SHObjectSerializer)(SHPointer, uint8_t **outData, uint64_t *outLen, SHPointer *customHandle);
typedef void(__cdecl *SHObjectSerializerFree)(SHPointer customHandle);
typedef SHPointer(__cdecl *SHObjectDeserializer)(uint8_t *data, uint64_t len);
typedef void(__cdecl *SHObjectReference)(SHPointer);
typedef void(__cdecl *SHObjectRelease)(SHPointer);
typedef void(__cdecl *SHObjectWeakReference)(SHPointer);
typedef void(__cdecl *SHObjectWeakRelease)(SHPointer);
typedef SHPointer(__cdecl *SHObjectUpgradeWeak)(SHPointer);
typedef uint64_t(__cdecl *SHObjectHash)(SHPointer);

struct SHObjectInfo {
  SHString name;

  SHObjectSerializer serialize;
  SHObjectSerializerFree free;
  SHObjectDeserializer deserialize;

  SHObjectReference reference;
  SHObjectRelease release;
  SHObjectWeakReference weakReference;
  SHObjectWeakRelease weakRelease;
  SHObjectUpgradeWeak upgradeWeak;

  SHObjectHash hash;

  bool isThreadSafe;
};

typedef struct SHTypeInfo SHTypeInfo;

typedef bool(__cdecl *SHExtTypeMatch)(const void *self, const void *other);
typedef void(__cdecl *SHExtTypeHash)(const void *self, void *outDigest, uint32_t digestSize);
typedef void(__cdecl *SHExtTypeReference)(void *self);
typedef void(__cdecl *SHExtTypeRelease)(void *self);

typedef struct SHExtendedObjectTypeInfo {
  SHExtTypeMatch match;
  SHExtTypeHash hash;
  SHExtTypeReference reference;
  SHExtTypeRelease release;
} SHExtendedObjectTypeInfo;

struct SHEnumInfo {
  SHString name;
  SHOptionalString help;
  SHStrings labels;
  SHEnums values;
  SHOptionalStrings descriptions;
};

struct SHParameterInfo {
  SHString name;
  SHOptionalString help;
  SHTypesInfo valueTypes;
  SHBool variableSetter;
};

struct SHExposedTypeInfo {
  SHString name;
  SHOptionalString help;
  struct SHTypeInfo exposedType;

  // the following are actually used only when exposing.

  // generally those are from internal shards like Set
  // means they can change in-place
  SHBool isMutable;

  // a protected variable is ignored by Get/Set etc.
  // can only be used directly as reference from params
  SHBool isProtected;

  // If the exposed variable should be available to all wires in the mesh
  SHBool global;

  // If the variable is market as tracked, apps building on top will can use this feature
  SHBool tracked;

  // If the variable is declared in this shard (not inherited from an inner shard)
  SHBool declared;
};

typedef struct SHStringPayload {
  char *elements;
  // this field is an optional optimization
  // if 0 strlen should be called to find the string length
  // if not 0 should be assumed valid
  uint32_t len;
  // this is mostly used internal
  // useful when serializing, recycling memory
  uint32_t cap;
} SHStringPayload;

// # Of SHVars and memory

// ## Specifically String and Seq types

// ### Shards need to maintain their own garbage, in a way so that any reciver
// of SHVar/s will not have to worry about it.

// #### In the case of getParam:
//   * the callee should allocate and preferably cache any String or Seq that
// needs to return.
//   * the callers will just read and should not modify the contents.
// #### In the case of activate:
//   * The input var memory is owned by the previous shard.
//   * The output var memory is owned by the activating shard.
//   * The activating shard will have to manage any SHVar that puts in context
// #### In the case of setParam:
//   * If the shard needs to store the String or Seq data it will then need to
// deep copy it.
//   * Callers should free up any allocated memory.

// ### What about exposed/requiredVariables, parameters and input/outputTypes:
// * Same for them, they are just read only basically

struct SHVarPayload {
  union {
    SHBool boolValue;

    struct {
      SHPointer objectValue;
      int32_t objectVendorId;
      int32_t objectTypeId;
    };

    SHInt intValue;
    SHInt2 int2Value;
    SHInt3 int3Value;
    SHInt4 int4Value;
    SHInt8 int8Value;
    SHInt16 int16Value;

    SHFloat floatValue;
    SHFloat2 float2Value;
    SHFloat3 float3Value;
    SHFloat4 float4Value;

    // notice, this is assumed not mutable!
    // unless specified in the SHTypeInfo of this value
    // and only when used as variable (shared)
    SHSeq seqValue;

    struct SHTable tableValue;

    SHStringPayload string;

    // WARNING: Keep this in sync with SHStringPayload
    struct {
      SHString stringValue;
      // this field is an optional optimization
      // if 0 strlen should be called to find the string length
      // if not 0 should be assumed valid
      uint32_t stringLen;
      // this is mostly used internal
      // useful when serializing, recycling memory
      uint32_t stringCapacity;
    };

    struct SHColor colorValue;

    struct SHImage *imageValue;

    struct SHAudio audioValue;

    SHWireRef wireValue;

    ShardPtr shardValue;

    struct {
      SHEnum enumValue;
      int32_t enumVendorId;
      int32_t enumTypeId;
    };

    struct {
      uint8_t *bytesValue;
      uint32_t bytesSize;
      // this is mostly used internal
      // useful when serializing, recycling memory
      uint32_t bytesCapacity;
    };

    SHPayloadArray arrayValue;

    struct SHTypeInfo *typeValue;
    struct SHTrait *traitValue;
  };
} __attribute__((aligned(16)));

// SHVar flags
#define SHVAR_FLAGS_NONE (0)
#define SHVAR_FLAGS_USES_OBJINFO (1 << 0) // 1
#define SHVAR_FLAGS_REF_COUNTED (1 << 1)  // 2
// this marks a variable external and even if references are counted
// it won't be destroyed automatically
#define SHVAR_FLAGS_EXTERNAL (1 << 2) // 3
// this marks the variable tracked, can be set inside a (Set) shard
#define SHVAR_FLAGS_TRACKED (1 << 3) // 4
// this marks the variable as a foreign variable, to prevent destruction
// when used inside seq and table
#define SHVAR_FLAGS_FOREIGN (1 << 4) // 5
// used internally to cancel and abort a wire flow, must be String type and receiver must abort and destroy the var
// used only in async.hpp and rust side async activation so far
#define SHVAR_FLAGS_ABORT (1 << 5) // 6
// this marks a weak object reference
#define SHVAR_FLAGS_WEAK_OBJECT (1 << 6) // 7

// Additional flags available
// #define SHVAR_FLAGS_RESERVED_0 (1 << 7) // 8
// #define SHVAR_FLAGS_RESERVED_1 (1 << 8) // 9
// #define SHVAR_FLAGS_RESERVED_2 (1 << 9) // 10
// #define SHVAR_FLAGS_RESERVED_3 (1 << 10) // 11
// #define SHVAR_FLAGS_RESERVED_4 (1 << 11) // 12
// #define SHVAR_FLAGS_RESERVED_5 (1 << 12) // 13
// #define SHVAR_FLAGS_RESERVED_6 (1 << 13) // 14

// 2 custom flags reserved for apps or anything building on top of shards
#define SHVAR_FLAGS_CUSTOM_0 (1 << 14) // 15
#define SHVAR_FLAGS_CUSTOM_1 (1 << 15) // 16

struct SHVar {
  struct SHVarPayload payload;
  union {
    struct SHObjectInfo *objectInfo;
    uint64_t version;
    char shortString[8];
    uint8_t shortBytes[8];
  };
#if defined(__cplusplus) || defined(SH_USE_ENUMS)
  SH_ENUM_DECL SHType valueType;
  SH_ENUM_DECL SHType innerType;
#else
  SHType valueType;
  SHType innerType;
#endif
  uint16_t flags;
  uint32_t refcount;
} __attribute__((aligned(16)));

enum SH_ENUM_CLASS SHRunWireOutputState { Running, Restarted, Returned, Stopped, Failed };

struct SHRunWireOutput {
  struct SHVar output;
  SH_ENUM_DECL SHRunWireOutputState state;
} __attribute__((aligned(16)));

struct SHComposeResult {
  struct SHTypeInfo outputType;

  // Allow external shards to fail
  // Internally we use exceptions
  SHBool failed;
  struct SHVar failureMessage; // destroyVar after use if failed

  SHExposedTypesInfo exposedInfo;
  SHExposedTypesInfo requiredInfo;

  // used when the last shard of the flow is
  // Restart/Stop/Return etc
  bool flowStopper;
};

struct SHInstanceData {
  // Used to optimize activations,
  // replacing function pointers during compose
  struct Shard *shard;

  // The current wire we are composing
  struct SHWire *wire;

  // Info related to our activation
  struct SHTypeInfo inputType;
  SHExposedTypesInfo shared;

  // if this activation might happen in a worker thread
  // for example cos this shard is within an Await shard
  // useful to fail during compose if we don't wish this
  bool onWorkerThread;

  // basically what the next shard can get as input
  struct SHTypesInfo outputTypes;

  // Internally used
  void *requiredVariables;
  void *privateContext;
};

typedef struct Shard *(__cdecl *SHShardConstructor)();
typedef void(__cdecl *SHCallback)();

typedef SHString(__cdecl *SHNameProc)(struct Shard *);
typedef uint32_t(__cdecl *SHHashProc)(struct Shard *);
typedef SHOptionalString(__cdecl *SHHelpProc)(struct Shard *);
typedef const struct SHTable *(__cdecl *SHPropertiesProc)(struct Shard *);

// Construction/Destruction
typedef void(__cdecl *SHSetupProc)(struct Shard *);
typedef void(__cdecl *SHDestroyProc)(struct Shard *);

typedef SHTypesInfo(__cdecl *SHInputTypesProc)(struct Shard *);
typedef SHTypesInfo(__cdecl *SHOutputTypesProc)(struct Shard *);

typedef SHExposedTypesInfo(__cdecl *SHExposedVariablesProc)(struct Shard *);
typedef SHExposedTypesInfo(__cdecl *SHRequiredVariablesProc)(struct Shard *);

typedef SHParametersInfo(__cdecl *SHParametersProc)(struct Shard *);
typedef struct SHError(__cdecl *SHSetParamProc)(struct Shard *, int, const struct SHVar *);
typedef struct SHVar(__cdecl *SHGetParamProc)(struct Shard *, int);

typedef struct SHShardComposeResult(__cdecl *SHComposeProc)(struct Shard *, struct SHInstanceData *data);
typedef struct SHShardComposeResult(__cdecl *SHComposeV2Proc)(struct Shard *, struct SHInstanceData *data);

// The core of the shard processing, avoid syscalls here
typedef const struct SHVar *(__cdecl *SHActivateProc)(struct Shard *, struct SHContext *, const struct SHVar *);

// Generally when stop() is called
// Note that context may be null in some cases
typedef struct SHError(__cdecl *SHCleanupProc)(struct Shard *, struct SHContext *);

typedef struct SHError(__cdecl *SHWarmupProc)(struct Shard *, struct SHContext *);

// Genetic programming optional mutation procedure
typedef void(__cdecl *SHMutateProc)(struct Shard *, struct SHTable options);
// Genetic programming optional crossover (inplace/3way) procedure
typedef void(__cdecl *SHCrossoverProc)(struct Shard *, const struct SHVar *state0, const struct SHVar *state1);

// Used for serialization, to deep serialize internal shard state
typedef struct SHVar(__cdecl *SHGetStateProc)(struct Shard *);
typedef void(__cdecl *SHSetStateProc)(struct Shard *, const struct SHVar *state);
typedef void(__cdecl *SHResetStateProc)(struct Shard *);

struct Shard {
  // \-- Internal stuff, do not directly use! --/

  // magic tricks to make some shards inline
  SHInlineShards inlineShardId;

  // used to manage the lifetime of this shard, should be set to 0
  uint32_t refCount;

  // flag to ensure shards are unique when flows/wires
  SHBool owned;

  // name length, used for profiling and more
  uint32_t nameLength;

  // some debug/utility info
  uint32_t line;
  uint32_t column;

  // internal use only, to optionally identify the shard
  uint64_t id;

  // \-- The interface to fill --/

  SHNameProc name;             // Returns the name of the shard, do not free the string,
                               // generally const
  SHHashProc hash;             // Returns the hash of the shard, useful for serialization
  SHHelpProc help;             // Returns the help text of the shard, do not free the
                               // string, generally const
  SHHelpProc inputHelp;        // optional help text for the input
  SHHelpProc outputHelp;       // optional help text for the output
  SHPropertiesProc properties; // optional properties

  SHSetupProc setup;     // A one time constructor setup for the shard
  SHDestroyProc destroy; // A one time finalizer for the shard, shards should
                         // also free all the memory in here!

  SHInputTypesProc inputTypes;
  SHOutputTypesProc outputTypes;

  SHExposedVariablesProc exposedVariables;
  SHRequiredVariablesProc requiredVariables;

  // Optional call used during validation to fixup "Any" input
  // type and provide valid output and exposed variable types
  SHComposeProc compose;
  SHComposeV2Proc composeV2;

  SHParametersProc parameters;
  SHSetParamProc setParam; // Set a parameter, the shard will copy the value, so
                           // if you allocated any memory you should free it
  SHGetParamProc getParam; // Gets a parameter, the shard is the owner of any
                           // allocated stuff, DO NOT free them

  SHWarmupProc warmup; // Called before running the wire, once
  SHActivateProc activate;
  // Called every time you stop a coroutine or sometimes
  // internally to clean up the shard
  SHCleanupProc cleanup;

  // Optional genetic programming helpers
  // getState/setState are also used during serialization
  // assume all the following to be called out of wire
  // likely the wire will be stopped after cleanup() called
  // so any state (in fact) should be kept
  SHMutateProc mutate;
  SHCrossoverProc crossover;
  // intended as persistent state during shard lifetime
  // persisting cleanups
  SHGetStateProc getState;
  SHSetStateProc setState;
  SHResetStateProc resetState;
};

struct SHWireProviderUpdate {
  SHString error;      // if any or nullptr
  struct SHWire *wire; // or nullptr if error
};

typedef void(__cdecl *SHProviderReset)(struct SHWireProvider *provider);

typedef SHBool(__cdecl *SHProviderReady)(struct SHWireProvider *provider);
typedef void(__cdecl *SHProviderSetup)(struct SHWireProvider *provider, SHString path, struct SHInstanceData data);

typedef SHBool(__cdecl *SHProviderUpdated)(struct SHWireProvider *provider);
typedef struct SHWireProviderUpdate(__cdecl *SHProviderAcquire)(struct SHWireProvider *provider);

typedef void(__cdecl *SHProviderReleaseWire)(struct SHWireProvider *provider, struct SHWire *wire);

struct SHWireProvider {
  SHProviderReset reset;

  SHProviderReady ready;
  SHProviderSetup setup;

  SHProviderUpdated updated;
  SHProviderAcquire acquire;

  SHProviderReleaseWire release;

  void *userData;
};

typedef void(__cdecl *SHRegisterShard)(SHString fullName, SHShardConstructor constructor);

typedef void(__cdecl *SHRegisterObjectType)(int32_t vendorId, int32_t typeId, struct SHObjectInfo info);

typedef void(__cdecl *SHRegisterEnumType)(int32_t vendorId, int32_t typeId, struct SHEnumInfo info);

typedef void(__cdecl *SHRegisterRunLoopCallback)(SHString eventName, SHCallback callback);

typedef void(__cdecl *SHRegisterExitCallback)(SHString eventName, SHCallback callback);

typedef void(__cdecl *SHUnregisterRunLoopCallback)(SHString eventName);

typedef void(__cdecl *SHUnregisterExitCallback)(SHString eventName);

typedef struct SHVar *(__cdecl *SHReferenceVariable)(struct SHContext *context, struct SHStringWithLen name);
typedef struct SHVar *(__cdecl *SHReferenceWireVariable)(SHWireRef wire, struct SHStringWithLen name);

typedef struct SHExternalVariable {
  struct SHVar *var;
  // Optional, if null, the type is derived  from the var
  const struct SHTypeInfo *type;
} SHExternalVariable;

// This copies the SHExternalVariable, although the var/type fields should be kept alive by the caller
typedef void(__cdecl *SHSetExternalVariable)(SHWireRef wire, struct SHStringWithLen name, struct SHExternalVariable *pVar);
typedef void(__cdecl *SHRemoveExternalVariable)(SHWireRef wire, struct SHStringWithLen name);

typedef struct SHVar *(__cdecl *SHAllocExternalVariable)(SHWireRef wire, struct SHStringWithLen name,
                                                         const struct SHTypeInfo *type);

typedef void(__cdecl *SHFreeExternalVariable)(SHWireRef wire, struct SHStringWithLen name);

typedef void(__cdecl *SHReleaseVariable)(struct SHVar *variable);

typedef void(__cdecl *SHAbortWire)(struct SHContext *context, struct SHStringWithLen errorText);

#if defined(__cplusplus) || defined(SH_USE_ENUMS)
typedef SH_ENUM_DECL SHWireState(__cdecl *SHSuspend)(struct SHContext *context, double seconds);
typedef SH_ENUM_DECL SHWireState(__cdecl *SHGetState)(struct SHContext *context);
#else
typedef SHWireState(__cdecl *SHSuspend)(struct SHContext *context, double seconds);
typedef SHWireState(__cdecl *SHGetState)(struct SHContext *context);
#endif

typedef void(__cdecl *SHCloneVar)(struct SHVar *dst, const struct SHVar *src);

typedef void(__cdecl *SHDestroyVar)(struct SHVar *var);

typedef struct SHVar(__cdecl *SHHashVar)(const struct SHVar *var);

typedef SHBool(__cdecl *SHValidateSetParam)(struct Shard *shard, int index, const struct SHVar *param);

typedef struct SHComposeResult(__cdecl *SHComposeShards)(Shards shards, struct SHInstanceData data);

#if defined(__cplusplus) || defined(SH_USE_ENUMS)
typedef SH_ENUM_DECL SHWireState(__cdecl *SHRunShards)(Shards shards, struct SHContext *context, const struct SHVar *input,
                                                       struct SHVar *output);
#else
typedef SHWireState(__cdecl *SHRunShards)(Shards shards, struct SHContext *context, const struct SHVar *input,
                                          struct SHVar *output);
#endif

#if defined(__cplusplus) || defined(SH_USE_ENUMS)
typedef SH_ENUM_DECL SHWireState(__cdecl *SHRunShardsHashed)(Shards shards, struct SHContext *context, const struct SHVar *input,
                                                             struct SHVar *output, struct SHVar *outHash);
#else
typedef SHWireState(__cdecl *SHRunShardsHashed)(Shards shards, struct SHContext *context, const struct SHVar *input,
                                                struct SHVar *output, struct SHVar *outHash);
#endif

typedef void(__cdecl *SHLog)(struct SHStringWithLen msg);
typedef void(__cdecl *SHLogLevel)(int level, struct SHStringWithLen msg);

typedef struct Shard *(__cdecl *SHCreateShard)(struct SHStringWithLen name);
typedef void(__cdecl *SHReleaseShard)(struct Shard *shard);

typedef SHWireRef(__cdecl *SHCreateWire)(struct SHStringWithLen name);
typedef void(__cdecl *SHSetWireLooped)(SHWireRef wire, SHBool looped);
typedef void(__cdecl *SHSetWireUnsafe)(SHWireRef wire, SHBool unsafe);
typedef void(__cdecl *SHSetWirePure)(SHWireRef wire, SHBool pure);
typedef void(__cdecl *SHSetWireStackSize)(SHWireRef wire, uint64_t stackSize);
typedef void(__cdecl *SHSetWirePriority)(SHWireRef wire, int priority);
typedef void(__cdecl *SHSetWireTraits)(SHWireRef wire, SHSeq traits);
typedef void(__cdecl *SHAddShard)(SHWireRef wire, ShardPtr shard);
typedef void(__cdecl *SHRemShard)(SHWireRef wire, ShardPtr shard);
typedef void(__cdecl *SHDestroyWire)(SHWireRef wire);
typedef SHBool(__cdecl *SHIsWireRunning)(SHWireRef wire);
typedef struct SHVar(__cdecl *SHStopWire)(SHWireRef wire);
typedef struct SHComposeResult(__cdecl *SHComposeWire)(SHWireRef wire, struct SHInstanceData data);
typedef struct SHRunWireOutput(__cdecl *SHRunWire)(SHWireRef wire, struct SHContext *context, const struct SHVar *input);
typedef SHWireRef(__cdecl *SHGetGlobalWire)(struct SHStringWithLen name);
typedef void(__cdecl *SHSetGlobalWire)(struct SHStringWithLen name, SHWireRef wire);
typedef void(__cdecl *SHUnsetGlobalWire)(struct SHStringWithLen name);

typedef SHMeshRef(__cdecl *SHCreateMesh)();
typedef void(__cdecl *SHDestroyMesh)(SHMeshRef mesh);
typedef struct SHVar(__cdecl *SHCreateMeshVar)();
typedef void(__cdecl *SHSetMeshLabel)(SHMeshRef mesh, SHStringWithLen label);
typedef SHBool(__cdecl *SHCompose)(SHMeshRef mesh, SHWireRef wire);
typedef void(__cdecl *SHUnSchedule)(SHMeshRef mesh, SHWireRef wire);
typedef void(__cdecl *SHSchedule)(SHMeshRef mesh, SHWireRef wire, SHBool compose);
typedef SHBool(__cdecl *SHTick)(SHMeshRef mesh);
typedef void(__cdecl *SHTerminate)(SHMeshRef mesh);
typedef void(__cdecl *SHSleep)(double seconds);
typedef struct SHVar *(__cdecl *SHGetMeshVariable)(SHMeshRef mesh, struct SHStringWithLen name);
typedef void(__cdecl *SHSetMeshVariableType)(SHMeshRef mesh, struct SHStringWithLen name, const struct SHExposedTypeInfo *type);

#define SH_ARRAY_TYPE(_array_, _value_)                                          \
  typedef void(__cdecl * _array_##Free)(_array_ *);                              \
  typedef void(__cdecl * _array_##Push)(_array_ *, const _value_ *);             \
  typedef void(__cdecl * _array_##Insert)(_array_ *, uint32_t, const _value_ *); \
  typedef _value_(__cdecl *_array_##Pop)(_array_ *);                             \
  typedef void(__cdecl * _array_##Resize)(_array_ *, uint32_t);                  \
  typedef void(__cdecl * _array_##FastDelete)(_array_ *, uint32_t);              \
  typedef void(__cdecl * _array_##SlowDelete)(_array_ *, uint32_t)

SH_ARRAY_TYPE(SHSeq, struct SHVar);
SH_ARRAY_TYPE(SHTypesInfo, struct SHTypeInfo);
SH_ARRAY_TYPE(SHParametersInfo, struct SHParameterInfo);
SH_ARRAY_TYPE(Shards, ShardPtr);
SH_ARRAY_TYPE(SHExposedTypesInfo, struct SHExposedTypeInfo);
SH_ARRAY_TYPE(SHEnums, SHEnum);
SH_ARRAY_TYPE(SHStrings, SHString);
SH_ARRAY_TYPE(SHTraitVariables, SHTraitVariable);
SH_ARRAY_TYPE(SHTraits, SHTrait);

#define SH_ARRAY_PROCS(_array_, _short_)   \
  _array_##Free _short_##Free;             \
  _array_##Push _short_##Push;             \
  _array_##Insert _short_##Insert;         \
  _array_##Pop _short_##Pop;               \
  _array_##Resize _short_##Resize;         \
  _array_##FastDelete _short_##FastDelete; \
  _array_##SlowDelete _short_##SlowDelete

typedef struct SHTable(__cdecl *SHTableNew)();

typedef SHString(__cdecl *SHGetRootPath)();
typedef void(__cdecl *SHSetRootPath)(SHString);

typedef struct SHVar(__cdecl *SHAsyncActivateProc)(struct SHContext *context, void *userData);

typedef void(__cdecl *SHAsyncCancelProc)(struct SHContext *context, void *userData);

typedef struct SHVar(__cdecl *SHRunAsyncActivate)(struct SHContext *context, void *userData, SHAsyncActivateProc call,
                                                  SHAsyncCancelProc cancel_call);

typedef SHStrings(__cdecl *SHGetShards)();

struct SHWireInfo {
  const struct SHStringWithLen name;
  SHBool looped;
  SHBool unsafe;
  const struct SHWire *wire;
  const Shards shards;
  SHBool isRunning;
  SHBool failed;
  const struct SHStringWithLen failureMessage;
  struct SHVar *finalOutput;
};

typedef struct SHWireInfo(__cdecl *SHGetWireInfo)(SHWireRef wireref);

// id is generally a crc32 value
// use this when the cache is compiled
typedef SHOptionalString(__cdecl *SHReadCachedString)(uint32_t id);
// the string is not copied! should be constant.
// use this when the cache is not precompiled
typedef SHOptionalString(__cdecl *SHWriteCachedString)(uint32_t id, const char *str);
typedef void(__cdecl *SHDecompressStrings)(void);

typedef SHBool(__cdecl *SHIsEqualVar)(const struct SHVar *v1, const struct SHVar *v2);
typedef int(__cdecl *SHCompareVar)(const struct SHVar *v1, const struct SHVar *v2);
typedef SHBool(__cdecl *SHIsEqualType)(const struct SHTypeInfo *t1, const struct SHTypeInfo *t2);

typedef struct SHTypeInfo(__cdecl *SHDeriveTypeInfo)(const struct SHVar *v, const struct SHInstanceData *data, bool mutable_);
typedef void(__cdecl *SHFreeDerivedTypeInfo)(struct SHTypeInfo *t);

typedef void *(__cdecl *SHAlloc)(uint32_t size);
typedef void(__cdecl *SHFree)(void *ptr);

typedef const struct SHEnumInfo *(__cdecl *SHFindEnumInfo)(int32_t vendorId, int32_t typeId);
typedef int64_t(__cdecl *SHFindEnumId)(SHStringWithLen name);
typedef const struct SHObjectInfo *(__cdecl *SHFindObjectInfo)(int32_t vendorId, int32_t typeId);
typedef int64_t(__cdecl *SHFindObjectTypeId)(SHStringWithLen name);
typedef SHString(__cdecl *SHType2Name)(SH_ENUM_DECL SHType type);

typedef void(__cdecl *SHStringGrow)(SHStringPayload *str, uint32_t newCap);
typedef void(__cdecl *SHStringFree)(SHStringPayload *str);

typedef uint64_t(__cdecl *SHGetStepCount)(struct SHContext *context);

typedef void(__cdecl *SHImageIncRefProc)(struct SHImage *);
typedef void(__cdecl *SHImageDecRefProc)(struct SHImage *);
typedef struct SHImage *(__cdecl *SHImageNewProc)(uint32_t dataLen);
typedef struct SHImage *(__cdecl *SHImageCloneProc)(struct SHImage *);
typedef uint32_t(__cdecl *SHImageDeriveDataLengthProc)(struct SHImage *);

struct SHLError {
  char *message;
  uint32_t line;
  uint32_t column;
};

struct SHLAst {
  /// of Program ast object, ref counted, count at 0 when returned, receiver must clone it!
  struct SHVar ast;
  struct SHLError *error;
};

struct SHLWire {
  SHWireRef *wire;
  struct SHLError *error;
};

struct SHLEvalEnv;

typedef struct SHLAst(__cdecl *SHReadProc)(struct SHStringWithLen name, struct SHStringWithLen code,
                                           struct SHStringWithLen basePath, const struct SHStringWithLen *includeDirs,
                                           uint32_t numIncludeDirs);
typedef struct SHLAst(__cdecl *SHLoadAstProc)(const uint8_t *bytes, uint32_t size);
typedef struct SHLEvalEnv *(__cdecl *SHCreateEvalEnv)(struct SHStringWithLen namespace_);
typedef void(__cdecl *SHFreeEvalEnv)(struct SHLEvalEnv *env);
typedef struct SHLError *(__cdecl *SHEvalProc)(struct SHLEvalEnv *env, const struct SHVar *ast);
typedef struct SHLWire(__cdecl *SHTransformEnv)(struct SHLEvalEnv *env, struct SHStringWithLen name);
typedef void(__cdecl *SHFreeWire)(struct SHLWire *wire);

typedef struct _SHCore {
  // Aligned allocator
  SHAlloc alloc;
  SHFree free;

  // String interface
  SHStringGrow stringGrow;
  SHStringFree stringFree;

  // SHTable interface
  SHTableNew tableNew;

  // Utility to use shards within shards
  SHComposeShards composeShards;
  // caller not handling return state
  SHRunShards runShards;
  // caller handles return state
  SHRunShards runShards2;
  // caller not handling return state
  SHRunShardsHashed runShardsHashed;
  // caller handles return state
  SHRunShardsHashed runShardsHashed2;

  // Logging
  SHLog log;
  SHLogLevel logLevel;

  // Wire creation
  SHCreateShard createShard;
  SHReleaseShard releaseShard;
  SHValidateSetParam validateSetParam;

  SHCreateWire createWire;
  SHSetWireLooped setWireLooped;
  SHSetWireUnsafe setWireUnsafe;
  SHSetWirePure setWirePure;
  SHSetWireStackSize setWireStackSize;
  SHSetWirePriority setWirePriority;
  SHSetWireTraits setWireTraits;
  SHAddShard addShard;
  SHRemShard removeShard;
  SHDestroyWire destroyWire;
  SHIsWireRunning isWireRunning;
  SHStopWire stopWire; // must destroyVar once done
  SHComposeWire composeWire;
  SHRunWire runWire;
  SHGetWireInfo getWireInfo;
  SHGetGlobalWire getGlobalWire;
  SHSetGlobalWire setGlobalWire;
  SHUnsetGlobalWire unsetGlobalWire;

  // Wire scheduling
  SHCreateMesh createMesh;
  SHDestroyMesh destroyMesh;
  SHCreateMeshVar createMeshVar;
  SHSetMeshLabel setMeshLabel;
  SHCompose compose;
  SHSchedule schedule;
  SHUnSchedule unschedule;
  SHTick tick;    // returns false if we had a failure
  SHTick isEmpty; // returns true if we have no wires to tick
  SHTerminate terminate;
  SHSleep sleep;
  SHGetStepCount getStepCount;

  // Environment utilities
  SHGetRootPath getRootPath;
  SHSetRootPath setRootPath;

  // async execution
  SHRunAsyncActivate asyncActivate;

  // Shards discovery (free after use, only the array, not the strings)
  SHGetShards getShards;

  // Adds a shard to the runtime database
  SHRegisterShard registerShard;
  // Adds a custom object type to the runtime database
  SHRegisterObjectType registerObjectType;
  // Adds a custom enumeration type to the runtime database
  SHRegisterEnumType registerEnumType;

  // To be used within shards, to manipulate variables
  SHReferenceVariable referenceVariable;
  SHReferenceWireVariable referenceWireVariable;
  SHReleaseVariable releaseVariable;

  // To add/rem external variables to wires
  SHSetExternalVariable setExternalVariable;
  SHRemoveExternalVariable removeExternalVariable;
  // Alternatives of the above that allocate memory for the variables
  SHAllocExternalVariable allocExternalVariable;
  SHFreeExternalVariable freeExternalVariable;

  SHGetMeshVariable getMeshVariable;
  SHSetMeshVariableType setMeshVariableType;

  // To be used within shards, to suspend the coroutine
  SHSuspend suspend;
  SHGetState getState;
  // To be used within shards, to abort the wire
  // after this call you should return immediately from activate
  SHAbortWire abortWire;

  // Utility to deal with SHVars
  SHCloneVar cloneVar;
  SHDestroyVar destroyVar;
  SHHashVar hashVar;

  // Compressed strings utility
  SHReadCachedString readCachedString;
  SHWriteCachedString writeCachedString;
  SHDecompressStrings decompressStrings;

  // Image object manipulation
  SHImageIncRefProc imageIncRef;
  SHImageDecRefProc imageDecRef;
  SHImageNewProc imageNew;
  SHImageCloneProc imageClone;
  SHImageDeriveDataLengthProc imageDeriveDataLength;

  // equality utilities
  SHIsEqualVar isEqualVar;
  SHCompareVar compareVar;
  SHIsEqualType isEqualType;
  SHDeriveTypeInfo deriveTypeInfo;
  SHFreeDerivedTypeInfo freeDerivedTypeInfo;

  SHFindEnumInfo findEnumInfo;
  SHFindEnumId findEnumId;
  SHFindObjectInfo findObjectInfo;
  SHFindObjectTypeId findObjectTypeId;
  SHType2Name type2Name;

  // Language parsing and evaluation
  SHReadProc read;
  SHLoadAstProc loadAst;
  SHCreateEvalEnv createEvalEnv;
  SHFreeEvalEnv freeEvalEnv;
  SHEvalProc eval;
  SHTransformEnv transformEnv;
  SHFreeWire freeWire;

  // Utility to deal with SHSeqs
  SH_ARRAY_PROCS(SHSeq, seq);

  // Utility to deal with SHTypesInfo
  SH_ARRAY_PROCS(SHTypesInfo, types);

  // Utility to deal with SHParamsInfo
  SH_ARRAY_PROCS(SHParametersInfo, params);

  // Utility to deal with Shards
  SH_ARRAY_PROCS(Shards, shards);

  // Utility to deal with SHExposedTypeInfo
  SH_ARRAY_PROCS(SHExposedTypesInfo, expTypes);

  // Utility to deal with SHEnums
  SH_ARRAY_PROCS(SHEnums, enums);

  // Utility to deal with SHStrings
  SH_ARRAY_PROCS(SHStrings, strings);

  // Utility to deal with SHTraitVariables
  SH_ARRAY_PROCS(SHTraitVariables, traitVariables);

  // Utility to deal with SHTraits
  SH_ARRAY_PROCS(SHTraits, traits);
} SHCore;

typedef SHCore *(__cdecl *SHShardsInterface)(uint32_t abi_version);

#ifdef SHARDS_CORE_DLL
#ifdef _WIN32
#define SHARDS_EXPORT __declspec(dllexport)
#define SHARDS_IMPORT __declspec(dllimport)
#else
#define SHARDS_EXPORT __attribute__((visibility("default")))
#define SHARDS_IMPORT
#endif
#else
#define SHARDS_EXPORT
#define SHARDS_IMPORT
#endif

#ifdef shards_core_EXPORTS
#define SHARDS_API SHARDS_EXPORT
#else
#define SHARDS_API SHARDS_IMPORT
#endif

#define SHARDS_CURRENT_ABI 0x20200102
#define SHARDS_CURRENT_ABI_STR "0x20200102"

#if defined(__cplusplus)
extern "C" {
#endif
SHARDS_API SHCore *__cdecl shardsInterface(uint32_t abi_version);
#if defined(__cplusplus)
};
#endif

#ifndef shassert
#define shassert assert
#endif

#ifndef NDEBUG
#define SH_DEBUG_MODE 1
#else
#define SH_DEBUG_MODE 0
#endif

#define sh_debug_only(__CODE__) \
  if (SH_DEBUG_MODE) {          \
    __CODE__;                   \
  }

#define SH_CAT_IMPL(s1, s2) s1##s2
#define SH_CAT(s1, s2) SH_CAT_IMPL(s1, s2)

#ifdef __COUNTER__
#define SH_GENSYM(str) SH_CAT(str, __COUNTER__)
#else
#define SH_GENSYM(str) SH_CAT(str, __LINE__)
#endif

#endif
