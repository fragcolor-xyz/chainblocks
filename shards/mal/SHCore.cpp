/* SPDX-License-Identifier: MPL-2.0 AND BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include <optional>
#include <shards/core/foundation.hpp>
#include <shards/wire_dsl.hpp>
#include "Environment.h"
#include "MAL.h"
#include "StaticList.h"
#include "Types.h"
#include "SHLisp.h"
#include <shards/shards.h>
#include <shards/core/shared.hpp>
#include <shards/core/runtime.hpp>
#include <algorithm>
#include <boost/lockfree/queue.hpp>
#include <stdexcept>
#ifdef SHARDS_DESKTOP
#include <boost/process/environment.hpp>
#endif
#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <ostream>
#include <set>
#include <thread>
#include <shards/log/log.hpp>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#ifdef SH_COMPRESSED_STRINGS
#include <shards/core/shccstrings.hpp>
#endif

using namespace shards;

#define ARG(type, name) type *name = VALUE_CAST(type, *argsBegin++)

static StaticList<malBuiltIn *> handlers;
#define FUNCNAME(uniq) sh_builtIn##uniq
#define HRECNAME(uniq) sh_handler##uniq
#define BUILTIN_DEF(uniq, symbol)                                                                         \
  static malBuiltIn::ApplyFunc FUNCNAME(uniq);                                                            \
  static StaticList<malBuiltIn *>::Node HRECNAME(uniq)(handlers, new malBuiltIn(symbol, FUNCNAME(uniq))); \
  malValuePtr FUNCNAME(uniq)(const MalString &name, malValueIter argsBegin, malValueIter argsEnd)

#define BUILTIN(symbol) BUILTIN_DEF(__LINE__, symbol)

#define BUILTIN_ISA(symbol, type)                        \
  BUILTIN(symbol) {                                      \
    CHECK_ARGS_IS(1);                                    \
    return mal::boolean(DYNAMIC_CAST(type, *argsBegin)); \
  }

#ifndef SH_CORE_ONLY
extern "C" void runRuntimeTests();
#endif

class malSHWire;
class malShard;
class malSHMesh;
class malSHVar;
typedef RefCountedPtr<malSHWire> malSHWirePtr;
typedef RefCountedPtr<malShard> malShardPtr;
typedef RefCountedPtr<malSHMesh> malSHMeshPtr;
typedef RefCountedPtr<malSHVar> malSHVarPtr;

void registerKeywords(malEnvPtr env);
malSHVarPtr varify(const malValuePtr &arg, bool consumeShard = true);

namespace fs = boost::filesystem;

namespace shards {
Shard *createShardInnerCall();
} // namespace shards

struct Observer;
void setupObserver(std::shared_ptr<Observer> &obs, const malEnvPtr &env);

static std::map<MalString, malValuePtr> builtIns;
static std::mutex observersMutex;
static std::map<malEnv *, std::shared_ptr<Observer>> observers;

namespace shards {
struct EdnEval {
  static malEnvPtr &GetThreadEnv() {
    thread_local malEnvPtr threadEnv;
    if (!threadEnv) {
      threadEnv = malEnvPtr(new malEnv());
      // path should not matter as malinit should have been called already
      malinit(threadEnv, nullptr, nullptr);
    }
    return threadEnv;
  }

  static inline Parameters params{
      {"Global",
       SHCCSTR("If true the script will be evaluated in the global root thread environment, if false "
               "a new child environment will be created."),
       {CoreInfo::BoolType}},
      {"Prefix",
       SHCCSTR("The prefix (similar to namespace) to add when setting and getting variables when evaluating this script."),
       CoreInfo::StringStringVarOrNone}};

  SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      global = value.payload.boolValue;
      break;
    case 1:
      prefix = value;
      break;
    default:
      throw std::runtime_error("Invalid parameter index");
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(global);
    case 1:
      return prefix;
    default:
      throw std::runtime_error("Invalid parameter index");
    }
  }

  SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  void warmup(SHContext *context) { prefix.warmup(context); }

  void cleanup(SHContext *context) { prefix.cleanup(); }

  SHVar activate(SHContext *context, const SHVar &input);

private:
  OwnedVar output{};
  bool global{true};
  ParamVar prefix;
};
} // namespace shards

class malRoot {
public:
  void reference(malValue *val) { m_refs.emplace_back(val); }
  void reference(const malValuePtr &val) { m_refs.emplace_back(val); }

protected:
  std::vector<malValuePtr> m_refs;
};

class malSHWire : public malValue, public malRoot {
public:
  malSHWire(const MalString &name) {
    SHLOG_TRACE("Created a SHWire - {}", name);
    auto wire = SHWire::make(name);
    m_wire = wire->newRef();
    {
      shards::GetGlobals().GlobalWires[name] = wire;
    }
  }

  malSHWire(const std::shared_ptr<SHWire> &wire) : m_wire(wire->newRef()) { SHLOG_TRACE("Loaded a SHWire - {}", wire->name); }

  malSHWire(const malSHWire &that, const malValuePtr &meta) = delete;

  ~malSHWire() {
    // unref all we hold  first
    m_refs.clear();

    // remove from globals
    auto cp = SHWire::sharedFromRef(m_wire);
    {
      auto it = shards::GetGlobals().GlobalWires.find(cp.get()->name);
      if (it != shards::GetGlobals().GlobalWires.end() && it->second == cp) {
        shards::GetGlobals().GlobalWires.erase(it);
      }
    }

    // delref
    SHLOG_TRACE("Deleting a SHWire - {}", cp->name);
    SHWire::deleteRef(m_wire);
  }

  virtual MalString print(bool readably) const {
    std::ostringstream stream;
    auto cp = SHWire::sharedFromRef(m_wire);
    stream << "SHWire: " << cp->name;
    return stream.str();
  }

  SHWireRef value() const { return m_wire; }

  virtual bool doIsEqualTo(const malValue *rhs) const { return m_wire == static_cast<const malSHWire *>(rhs)->m_wire; }

  virtual malValuePtr doWithMeta(malValuePtr meta) const { throw shards::SHException("Meta not supported on shards wires."); }

private:
  SHWireRef m_wire;
};

class malShard : public malValue, public malRoot {
public:
  malShard(const MalString &name) : m_shard(shards::createShard(name.c_str())) { incRef(m_shard); }

  malShard(Shard *shard) : m_shard(shard) { incRef(m_shard); }

  malShard(const malShard &that, const malValuePtr &meta) = delete;

  malShard(const malShard &that) = delete;

  malShard(malShard &&that) = delete;

  ~malShard() {
    // unref all we hold  first
    m_refs.clear();

    // could be nullptr
    if (m_shard) {
      decRef(m_shard);
    }
  }

  virtual MalString print(bool readably) const {
    assert(m_shard);
    std::ostringstream stream;
    stream << "Shard: " << m_shard->name(m_shard);
    return stream.str();
  }

  Shard *value() const {
    if (!m_shard) {
      throw shards::SHException("Attempted to use a null shard, shards are unique, "
                                "probably was already used.");
    }
    return m_shard;
  }

  void consume() {
    decRef(m_shard);
    m_shard = nullptr;
  }

  virtual bool doIsEqualTo(const malValue *rhs) const { return m_shard == static_cast<const malShard *>(rhs)->m_shard; }

  virtual malValuePtr doWithMeta(malValuePtr meta) const { throw shards::SHException("Meta not supported on shards shards."); }

private:
  Shard *m_shard;
};

class malSHMesh : public malValue, public malRoot {
public:
  malSHMesh() : m_mesh(SHMesh::make()) { SHLOG_TRACE("Created a Mesh"); }

  malSHMesh(const malSHMesh &that, const malValuePtr &meta) = delete;

  ~malSHMesh() {
    // Delete all refs first
    m_refs.clear();
    m_mesh->terminate();
    SHLOG_TRACE("Deleted a Mesh");
  }

  virtual MalString print(bool readably) const {
    std::ostringstream stream;
    stream << "Mesh";
    return stream.str();
  }

  SHMesh *value() const { return m_mesh.get(); }

  std::shared_ptr<SHMesh> &sharedRef() { return m_mesh; }

  void schedule(malSHWire *wire, bool compose = true) {
    auto cp = SHWire::sharedFromRef(wire->value());
    m_mesh->schedule(cp, Var::Empty, compose);
    reference(wire);
  }

  void compose(malSHWire *wire, bool compose = true) {
    auto cp = SHWire::sharedFromRef(wire->value());
    m_mesh->compose(cp, Var::Empty);
    // reference(wire); // don't reference in this case..?
  }

  virtual bool doIsEqualTo(const malValue *rhs) const { return m_mesh == static_cast<const malSHMesh *>(rhs)->m_mesh; }

  virtual malValuePtr doWithMeta(malValuePtr meta) const { throw shards::SHException("Meta not supported on shards meshs."); }

private:
  std::shared_ptr<SHMesh> m_mesh;
};

class malSHVar : public malValue, public malRoot {
public:
  malSHVar(const SHVar &var, bool owned) : m_owned(owned) { m_var = var; }

  malSHVar(const malSHVar &that, const malValuePtr &meta) : malValue(meta), m_owned(true) {
    m_var = SHVar();
    shards::cloneVar(m_var, that.m_var);
  }

  static malSHVar *newCloned(const SHVar &var) {
    SHVar clonedVar{};
    cloneVar(clonedVar, var);
    return new malSHVar(clonedVar, true);
  }

  ~malSHVar() {
    // unref all we hold  first
    m_refs.clear();

    if (m_owned)
      shards::destroyVar(m_var);
  }

  virtual MalString print(bool readably) const {
    std::ostringstream stream;
    stream << m_var;
    return stream.str();
  }

  virtual bool doIsEqualTo(const malValue *rhs) const { return m_var == static_cast<const malSHVar *>(rhs)->m_var; }

  SHVar &value() { return m_var; }
  const SHVar &value() const { return m_var; }

  WITH_META(malSHVar);

private:
  SHVar m_var;
  bool m_owned;
};

class malSHTypeInfo : public malValue, public malRoot {
public:
  malSHTypeInfo(const SHTypeInfo &var, bool owned) : m_owned(owned) { m_typeInfo = var; }

  malSHTypeInfo(const malSHTypeInfo &that, const malValuePtr &meta) : malValue(meta), m_owned(true) {
    m_typeInfo = shards::cloneTypeInfo(that.m_typeInfo);
  }

  static malSHTypeInfo *newCloned(const SHTypeInfo &var) { return new malSHTypeInfo(cloneTypeInfo(var), true); }

  ~malSHTypeInfo() {
    // unref all we hold  first
    m_refs.clear();

    if (m_owned)
      shards::freeDerivedInfo(m_typeInfo);
  }

  virtual MalString print(bool readably) const {
    std::ostringstream stream;
    stream << m_typeInfo;
    return stream.str();
  }

  virtual bool doIsEqualTo(const malValue *rhs) const {
    return m_typeInfo == static_cast<const malSHTypeInfo *>(rhs)->m_typeInfo;
  }

  SHTypeInfo &value() { return m_typeInfo; }
  const SHTypeInfo &value() const { return m_typeInfo; }

  WITH_META(malSHTypeInfo);

private:
  SHTypeInfo m_typeInfo;
  bool m_owned;
};

void installSHCore(const malEnvPtr &env, const char *exePath, const char *scriptPath) {
  // Setup logging first
  // read env var for log file
  auto logFile = std::getenv("SHARDS_LOG_FILE");
  if (logFile) {
    logging::setupDefaultLoggerConditional(logFile);
  } else {
    logging::setupDefaultLoggerConditional("shards.log");
  }

  std::shared_ptr<Observer> obs;
  setupObserver(obs, env);

  static bool initDoneOnce = false;
  if (!initDoneOnce) {
    shards::GetGlobals().ExePath = exePath;
    shards::GetGlobals().RootPath = scriptPath;

    SHLOG_DEBUG("Exe path: {}", exePath);
    SHLOG_DEBUG("Script path: {}", scriptPath);

#ifndef __EMSCRIPTEN__
    shards::installSignalHandlers();
#endif

    // initialize shards
    (void)shardsInterface(SHARDS_CURRENT_ABI);

    REGISTER_SHARD("EDN.Eval", EdnEval);

    initDoneOnce = true;

#if defined(SHARDS_WITH_RUST_SHARDS) && !defined(NDEBUG)
    // TODO fix running rust tests...
    runRuntimeTests();
#endif
  }

  // Wire params
  env->set(":Looped", mal::keyword(":Looped"));
  env->set(":Unsafe", mal::keyword(":Unsafe"));
  env->set(":LStack", mal::keyword(":LStack"));
  env->set(":SStack", mal::keyword(":SStack"));

  // inject static handlers
  for (auto it = handlers.begin(), end = handlers.end(); it != end; ++it) {
    malBuiltIn *handler = *it;
    env->set(handler->name(), handler);
  }

  // inject discovered loaded values
  // TODO
  // this is not efficient as it likely will repeat inserting the first time!
  // See Observer down
  for (auto &v : builtIns) {
    // SPDLOG_TRACE("env->set({})", v.first);
    env->set(v.first, v.second.ptr());
  }

  rep("(def! inc (fn* [a] (+ a 1)))", env);
  rep("(def! dec (fn* (a) (- a 1)))", env);
  rep("(def! zero? (fn* (n) (= 0 n)))", env);
  rep("(def! identity (fn* (x) x))", env);
  rep("(def! gensym (let* [counter (atom 0)] (fn* [] (symbol (str \"G__\" "
      "(swap! counter inc))))))",
      env);
  rep("(defmacro! or (fn* [& xs] (if (< (count xs) 2) (first xs) (let* [r "
      "(gensym)] `(let* (~r ~(first xs)) (if ~r ~r (or ~@(rest xs))))))))",
      env);
  rep("(defmacro! and (fn* (& xs) (cond (empty? xs) true (= 1 (count xs)) "
      "(first xs) true (let* (condvar (gensym)) `(let* (~condvar ~(first xs)) "
      "(if ~condvar (and ~@(rest xs)) ~condvar))))))",
      env);
  rep("(def! _alias_add_implicit (fn* [special added] (fn* [x & xs] (list "
      "special x (cons added xs)))))",
      env);
  rep("(defmacro! let  (_alias_add_implicit 'let* 'do))", env);
  rep("(defmacro! when (_alias_add_implicit 'if   'do))", env);
  rep("(defmacro! def  (_alias_add_implicit 'def! 'do))", env);
  rep("(defmacro! deflocal  (_alias_add_implicit 'deflocal! 'do))", env);
  rep("(defmacro! fn   (_alias_add_implicit 'fn*  'do))", env);
  rep("(defmacro! defn (_alias_add_implicit 'def! 'fn))", env);
  rep("(def! partial (fn* [pfn & args] (fn* [& args-inner] (apply pfn (concat "
      "args args-inner)))))",
      env);
  rep("(def! reduce (fn* (f init xs) (if (empty? xs) init (reduce f (f init (first xs)) (rest xs)))))", env);
  rep("(def! foldr (let* [rec (fn* [f xs acc index] (if (< index 0) acc (rec f xs (f (nth xs index) acc) (- index 1))))] (fn* [f "
      "init xs] (rec f xs init (- (count xs) 1)))))",
      env);
  rep("(def || Await)", env);
  rep("(defn for [from to pfn] (map (fn* [n] (apply pfn [n])) (range from to)))", env);
#if defined(_WIN32)
  rep("(def platform \"windows\")", env);
#elif defined(__ANDROID__)
  rep("(def platform \"android\")", env);
#elif defined(__EMSCRIPTEN__)
  rep("(def platform \"emscripten\")", env);
#elif defined(__linux__)
  rep("(def platform \"linux\")", env);
#elif defined(__APPLE__)
#if TARGET_OS_IOS
  rep("(def platform \"ios\")", env);
#else
  rep("(def platform \"macos\")", env);
#endif
#endif
  rep("(defmacro! defwire (fn* [name & shards] `(do (def! ~(symbol (str name)) (DefWire ~(str name))) (ImplWire ~(symbol (str "
      "name)) (wireify (vector ~@shards))))))",
      env);
  rep("(defmacro! defloop (fn* [name & shards] `(do (def! ~(symbol (str name)) (DefWire ~(str name))) (ImplWire ~(symbol (str "
      "name)) :Looped (wireify (vector ~@shards))))))",
      env);
  rep("(defmacro! defpure (fn* [name & shards] `(do (def! ~(symbol (str name)) (DefWire ~(str name))) (ImplWire ~(symbol (str "
      "name)) :Pure (wireify (vector ~@shards))))))",
      env);
  rep("(defmacro! patch (fn* [name & shards] `(do (ImplWire ~(symbol (str name)) :Pure (wireify (vector ~@shards))))))", env);
  rep("(defmacro! deftrait (fn* [name & shards] `(def! ~(symbol (str name)) (hash-map ~@shards))))", env);
  rep("(defmacro! defmesh (fn* [name] `(def ~(symbol (str name)) (Mesh))))", env);
  rep("(defmacro! | (fn* [& shards] `(SubFlow (wireify (vector ~@shards)))))", env);
  rep("(defmacro! |# (fn* [& shards] `(Hashed (wireify (vector ~@shards)))))", env);
  rep("(defmacro! || (fn* [& shards] `(Await (wireify (vector ~@shards)))))", env);
  rep("(defmacro! Setup (fn* [& shards] `(Once (wireify (vector ~@shards)))))", env);
  rep("(defmacro! defshards (fn* [name args & shards] `(defn ~(symbol (str name)) ~args (wireify (vector ~@shards)))))", env);
  rep("(defmacro! $ (fn* [prefix sym] (symbol (str prefix \"/\" sym))))", env);

  // Type definitions and built-in constructors
  env->set(MalString("None"), malValuePtr(new malSHTypeInfo(CoreInfo::NoneType, false)));
  env->set(MalString("Any"), malValuePtr(new malSHTypeInfo(CoreInfo::AnyType, false)));
  env->set(MalString("Bool"), malValuePtr(new malSHTypeInfo(CoreInfo::BoolType, false)));
  rep("(def! Enum enum)", env);
  rep("(def! Int int)", env);
  rep("(def! Int2 int2)", env);
  rep("(def! Int3 int3)", env);
  rep("(def! Int4 int4)", env);
  rep("(def! Int8 int8)", env);
  rep("(def! Int16 int16)", env);
  rep("(def! Float float)", env);
  rep("(def! Float2 float2)", env);
  rep("(def! Float3 float3)", env);
  rep("(def! Float4 float4)", env);
  rep("(def! Color color)", env);
  rep("(def! Bytes bytes)", env);
  rep("(def! String string)", env);
  rep("(def! Path path)", env);
  rep("(def! ContextVar context-var)", env);
  env->set(MalString("Image"), malValuePtr(new malSHTypeInfo(CoreInfo::ImageType, false)));
  env->set(MalString("Shard"), malValuePtr(new malSHTypeInfo(CoreInfo::ShardRefType, false)));
  env->set(MalString("Audio"), malValuePtr(new malSHTypeInfo(CoreInfo::AudioType, false)));
  env->set(MalString("Type"), malValuePtr(new malSHTypeInfo(CoreInfo::TypeType, false)));
}

struct WireLoadResult {
  bool hasError;
  const char *errorMsg;
  SHWire *wire;
};

struct WatchedFile {
  using LWT = decltype(fs::last_write_time(fs::path()));
  std::string path;
  LWT lastWrite{};

  WatchedFile(std::string path) : path(path) { clean(); }

  // Returns true if time stamp changed
  bool isChanged() const {
    fs::path p(path);
    boost::system::error_code errorCode;
    bool changed = fs::is_regular_file(p, errorCode) && lastWrite != fs::last_write_time(p, errorCode);
    return !errorCode.failed() && changed;
  }

  // Force the file to be marked as changed
  void markChanged() { lastWrite = 0; }

  // Clears any changes on this file
  void clean() {
    boost::system::error_code errorCode;
    lastWrite = fs::last_write_time(fs::path(path), errorCode);
  }
};

struct WireFileWatcher {
  std::atomic_bool running;
  std::thread worker;
  std::string rootFilePath;
  malValuePtr autoexec;
  std::string path;
  boost::lockfree::queue<WireLoadResult> results;
  std::unordered_map<SHWire *, std::tuple<malEnvPtr, malValuePtr>> liveWires;
  boost::lockfree::queue<SHWire *> garbage;
  std::weak_ptr<SHMesh> mesh;
  std::string localRoot;
  decltype(fs::last_write_time(fs::path())) lastWrite{};
  malEnvPtr rootEnv{new malEnv()};
  SHTypeInfo inputTypeInfo;
  shards::IterableExposedInfo shared;

  // std::vector<std::string> paths;
  std::string rootPath;
  std::vector<WatchedFile> watchedFiles;

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  decltype(std::chrono::high_resolution_clock::now()) tStart{std::chrono::high_resolution_clock::now()};
#endif

  void mergeTrackedDependencies(const std::set<std::string> &paths) {
    for (auto &it : paths) {
      SHLOG_INFO("Tracked Path> {}", it);
      watchedFiles.emplace_back(it);
    }
  }

  bool checkForChanges() {
    bool anyChanges = false;
    for (auto &file : watchedFiles) {
      // Make sure not to early out here so time stamps get updated for all files
      if (file.isChanged())
        anyChanges = true;
    }
    return anyChanges;
  }

  void checkOnce() {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // check only every 2 seconds
    auto tnow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = tnow - tStart;
    if (dt.count() < 2.0) {
      return;
    }
    tStart = tnow;
#endif

    try {
      if (checkForChanges()) {
        fs::path p(rootFilePath);

        std::ifstream lsp(p.c_str());
        std::string str((std::istreambuf_iterator<char>(lsp)), std::istreambuf_iterator<char>());

        auto fileNameOnly = p.filename();
        str = "(Wire \"" + fileNameOnly.string() + "\" " + str + ")";

        SHLOG_DEBUG("Processing {}", fileNameOnly.string());

        malEnvPtr env(new malEnv(rootEnv));
        env->trackDependencies = true;
        auto test = malenv();

        auto res = maleval(str.c_str(), env);

        // Rebuild the list of watched files
        // They'll all be marked as clean
        watchedFiles.clear();
        watchedFiles.emplace_back(rootFilePath);
        mergeTrackedDependencies(env->getTrackedDependencies());

        auto var = varify(res);
        if (var->value().valueType != SHType::Wire) {
          SHLOG_ERROR("Script did not return a SHWire");
          return;
        }

        auto wireref = var->value().payload.wireValue;
        auto wire = SHWire::sharedFromRef(wireref);

        SHInstanceData data{};
        data.inputType = inputTypeInfo;
        data.shared = shared;
        data.wire = wire.get();
        data.wire->mesh = mesh;

        SHLOG_TRACE("Processed {}", fileNameOnly.string());

        // run compose to infer types and specialize
        auto composeResult = composeWire(wire.get(), data);
        shards::arrayFree(composeResult.exposedInfo);

        SHLOG_TRACE("Validated {}", fileNameOnly.string());

        liveWires[wire.get()] = std::make_tuple(env, res);

        WireLoadResult result = {false, "", wire.get()};
        results.push(result);
      }

      // Collect garbage
      if (!garbage.empty()) {
        SHWire *gwire;
        if (garbage.pop(gwire)) {
          auto &data = liveWires[gwire];
          SHLOG_TRACE("Collecting hot wire {} env refcount: {}", gwire->name, std::get<0>(data)->refCount());
          liveWires.erase(gwire);
        }
      }

#if !(defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__))
      // sleep some, don't run callbacks here tho!
      shards::sleep(0.2);
#endif
    } catch (malEmptyInputException &) {
      WireLoadResult result = {true, "empty input exception", nullptr};
      results.push(result);
    } catch (malValuePtr &mv) {
      SHLOG_ERROR(mv->print(true));
      WireLoadResult result = {true, "script error", nullptr};
      results.push(result);
    } catch (MalString &s) {
      SHLOG_ERROR(s);
      WireLoadResult result = {true, "parse error", nullptr};
      results.push(result);
    } catch (const std::exception &e) {
      SHLOG_ERROR(e.what());
      WireLoadResult result = {true, "exception", nullptr};
      results.push(result);
    } catch (...) {
      WireLoadResult result = {true, "unknown error (...)", nullptr};
      results.push(result);
    }
  }

  explicit WireFileWatcher(const std::string &file, std::string currentPath, const SHInstanceData &data,
                           const malValuePtr &autoexec)
      : running(true), autoexec(autoexec), path(currentPath), results(2), garbage(2), inputTypeInfo(data.inputType),
        shared(data.shared) {

    rootFilePath = absolute(fs::path(file), currentPath).string();

    // Setup initial list of marked files
    watchedFiles.emplace_back(rootFilePath);
    watchedFiles.back().markChanged();

    mesh = data.wire->mesh;
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    localRoot = fs::path(path).string();
    try {
      malinit(rootEnv, localRoot.c_str(), localRoot.c_str());
      if (this->autoexec != nullptr) {
        EVAL(this->autoexec, rootEnv);
      }
    } catch (malEmptyInputException &) {
      SHLOG_ERROR("empty input exception.");
      throw;
    } catch (malValuePtr &mv) {
      SHLOG_ERROR("script error: {}", mv->print(true));
      throw;
    } catch (MalString &s) {
      SHLOG_ERROR("parse error: {}", s);
      throw;
    } catch (const std::exception &e) {
      SHLOG_ERROR("Failed to init WireFileWatcher: {}", e.what());
      throw;
    } catch (...) {
      SHLOG_ERROR("Failed to init WireFileWatcher.");
      throw;
    }
#else
    worker = std::thread([this] {
      localRoot = fs::path(path).string();
      try {
        malinit(rootEnv, localRoot.c_str(), localRoot.c_str());
        if (this->autoexec != nullptr) {
          EVAL(this->autoexec, rootEnv);
        }
      } catch (malEmptyInputException &) {
        SHLOG_ERROR("empty input exception.");
        throw;
      } catch (malValuePtr &mv) {
        SHLOG_ERROR("script error: {}", mv->print(true));
        throw;
      } catch (MalString &s) {
        SHLOG_ERROR("parse error: {}", s);
        throw;
      } catch (const std::exception &e) {
        SHLOG_ERROR("Failed to init WireFileWatcher: {}", e.what());
        throw;
      } catch (...) {
        SHLOG_ERROR("Failed to init WireFileWatcher.");
        throw;
      }

      while (running) {
        checkOnce();
      }

      // Final collect garbage before returing!
      if (!garbage.empty()) {
        SHWire *gwire;
        if (garbage.pop(gwire)) {
          auto &data = liveWires[gwire];
          SHLOG_TRACE("Collecting hot wire {} env refcount: {}", gwire->name, std::get<0>(data)->refCount());
          liveWires.erase(gwire);
        }
      }
    });
#endif
  }

  ~WireFileWatcher() {
    running = false;
// join is bad under emscripten, will lock the main thread
#ifndef __EMSCRIPTEN__
    worker.join();
#elif defined(__EMSCRIPTEN_PTHREADS__)
    worker.detach();
#endif
  }
};

class malWireProvider : public malValue, public shards::WireProvider {
public:
  malWireProvider(const MalString &name) : shards::WireProvider(), _filename(name), _watcher(nullptr) {}

  malWireProvider(const MalString &name, const malValuePtr &ast)
      : shards::WireProvider(), _filename(name), _autoexec(ast), _watcher(nullptr) {}

  malWireProvider(const malSHVar &that, const malValuePtr &meta) = delete;

  virtual ~malWireProvider() {}

  MalString print(bool readably) const override { return "WireProvider"; }

  bool doIsEqualTo(const malValue *rhs) const override { return false; }

  malValuePtr doWithMeta(malValuePtr meta) const override {
    throw shards::SHException("Meta not supported on shards wire providers.");
  }

  void reset() override { _watcher.reset(nullptr); }

  bool ready() override { return bool(_watcher); }

  void setup(const char *path, const SHInstanceData &data) override {
    _watcher.reset(new WireFileWatcher(_filename, path, data, _autoexec));
  }

  bool updated() override {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    _watcher->checkOnce();
#endif
    return !_watcher->results.empty();
  }

  SHWireProviderUpdate acquire() override {
    assert(!_watcher->results.empty());
    SHWireProviderUpdate update{};
    WireLoadResult result;
    // consume all of them!
    while (_watcher->results.pop(result)) {
      if (result.hasError) {
        update.wire = nullptr;
        update.error = result.errorMsg;
      } else {
        update.wire = result.wire;
        update.error = nullptr;
      }
    }
    return update;
  }

  void release(SHWire *wire) override { _watcher->garbage.push(wire); }

private:
  MalString _filename;
  malValuePtr _autoexec;
  std::unique_ptr<WireFileWatcher> _watcher;
};

SHType keywordToType(malKeyword *typeKeyword) {
  auto value = typeKeyword->value();
  if (value == ":None")
    return SHType::None;
  else if (value == ":Any")
    return SHType::Any;
  else if (value == ":Enum")
    return SHType::Enum;
  else if (value == ":Bool")
    return SHType::Bool;
  else if (value == ":Int")
    return SHType::Int;
  else if (value == ":Int2")
    return SHType::Int2;
  else if (value == ":Int3")
    return SHType::Int3;
  else if (value == ":Int4")
    return SHType::Int4;
  else if (value == ":Int8")
    return SHType::Int8;
  else if (value == ":Int16")
    return SHType::Int16;
  else if (value == ":Float")
    return SHType::Float;
  else if (value == ":Float2")
    return SHType::Float2;
  else if (value == ":Float3")
    return SHType::Float3;
  else if (value == ":Float4")
    return SHType::Float4;
  else if (value == ":Color")
    return SHType::Color;
  else if (value == ":Wire")
    return SHType::Wire;
  else if (value == ":Shard")
    return SHType::ShardRef;
  else if (value == ":String")
    return SHType::String;
  else if (value == ":ContextVar")
    return SHType::ContextVar;
  else if (value == ":Image")
    return SHType::Image;
  else if (value == ":Seq")
    return SHType::Seq;
  else if (value == ":Table")
    return SHType::Table;
  else
    throw shards::SHException("Could not infer type for special '.' shard. Need to return a proper "
                              "type keyword!");
}

malValuePtr typeToKeyword(SHType type) {
  switch (type) {
  case SHType::Type:
    return mal::keyword(":Type");
  case SHType::EndOfBlittableTypes:
    return mal::keyword(":EndOfBlittableTypes");
  case SHType::None:
    return mal::keyword(":None");
  case SHType::Any:
    return mal::keyword(":Any");
  case SHType::Object:
    return mal::keyword(":Object");
  case SHType::Enum:
    return mal::keyword(":Enum");
  case SHType::Bool:
    return mal::keyword(":Bool");
  case SHType::Int:
    return mal::keyword(":Int");
  case SHType::Int2:
    return mal::keyword(":Int2");
  case SHType::Int3:
    return mal::keyword(":Int3");
  case SHType::Int4:
    return mal::keyword(":Int4");
  case SHType::Int8:
    return mal::keyword(":Int8");
  case SHType::Int16:
    return mal::keyword(":Int16");
  case SHType::Float:
    return mal::keyword(":Float");
  case SHType::Float2:
    return mal::keyword(":Float2");
  case SHType::Float3:
    return mal::keyword(":Float3");
  case SHType::Float4:
    return mal::keyword(":Float4");
  case SHType::Color:
    return mal::keyword(":Color");
  case SHType::Wire:
    return mal::keyword(":Wire");
  case SHType::ShardRef:
    return mal::keyword(":Shard");
  case SHType::String:
    return mal::keyword(":String");
  case SHType::ContextVar:
    return mal::keyword(":ContextVar");
  case SHType::Path:
    return mal::keyword(":Path");
  case SHType::Image:
    return mal::keyword(":Image");
  case SHType::Audio:
    return mal::keyword(":Audio");
  case SHType::Bytes:
    return mal::keyword(":Bytes");
  case SHType::Seq:
    return mal::keyword(":Seq");
  case SHType::Table:
    return mal::keyword(":Table");
  case SHType::Trait:
    return mal::keyword(":Trait");
  };
  return mal::keyword(":None");
}

BUILTIN("Mesh") {
  auto mesh = new malSHMesh();
  return malValuePtr(mesh);
}

#define WRAP_TO_CONST(_var_)                      \
  auto constShard = shards::createShard("Const"); \
  constShard->setup(constShard);                  \
  constShard->setParam(constShard, 0, &_var_);    \
  auto mshard = new malShard(constShard);         \
  mshard->line = arg->line;                       \
  result.emplace_back(mshard);

// Helper to generate const shards automatically inferring types
std::vector<malShardPtr> shardify(const malValuePtr &arg) {
  // shards clone vars internally so there is no need to keep ref!
  std::vector<malShardPtr> result;
  if (arg == mal::nilValue()) {
    // Wrap none into const
    SHVar var{};
    WRAP_TO_CONST(var);
  } else if (const malString *v = DYNAMIC_CAST(malString, arg)) {
    SHVar strVar{};
    strVar.valueType = SHType::String;
    auto &s = v->ref();
    strVar.payload.stringValue = s.c_str();
    strVar.payload.stringLen = s.size();
    WRAP_TO_CONST(strVar);
  } else if (const malNumber *v = DYNAMIC_CAST(malNumber, arg)) {
    auto value = v->value();
    SHVar var{};
    if (v->isInteger()) {
      var.valueType = SHType::Int;
      var.payload.intValue = value;
    } else {
      var.valueType = SHType::Float;
      var.payload.floatValue = value;
    }
    WRAP_TO_CONST(var);
  } else if (arg == mal::trueValue()) {
    SHVar var{};
    var.valueType = SHType::Bool;
    var.payload.boolValue = true;
    WRAP_TO_CONST(var);
  } else if (arg == mal::falseValue()) {
    SHVar var{};
    var.valueType = SHType::Bool;
    var.payload.boolValue = false;
    WRAP_TO_CONST(var);
  } else if (malSHVar *v = DYNAMIC_CAST(malSHVar, arg)) {
    WRAP_TO_CONST(v->value());
  } else if (malSHWire *v = DYNAMIC_CAST(malSHWire, arg)) {
    auto val = shards::Var(v->value());
    WRAP_TO_CONST(val);
  } else if (malSHMesh *v = DYNAMIC_CAST(malSHMesh, arg)) {
    auto val = shards::Var::Object(&v->sharedRef(), CoreCC, 'brcM');
    WRAP_TO_CONST(val);
  } else if (malShard *v = DYNAMIC_CAST(malShard, arg)) {
    result.emplace_back(v);
  } else if (DYNAMIC_CAST(malVector, arg)) {
    auto shv = varify(arg);
    WRAP_TO_CONST(shv->value());
  } else if (const malSequence *v = DYNAMIC_CAST(malSequence, arg)) {
    auto count = v->count();
    for (auto i = 0; i < count; i++) {
      auto val = v->item(i);
      auto blks = shardify(val);
      result.insert(result.end(), blks.begin(), blks.end());
    }
  } else if (const malHash *v = DYNAMIC_CAST(malHash, arg)) {
    SHVar var{};
    shards::SHMap shMap;
    std::vector<malSHVarPtr> vars;
    for (auto [k, v] : v->m_map) {
      auto shv = varify(v);
      vars.emplace_back(shv);
      // key is either a quoted string or a keyword (starting with ':')
      auto ks = k[0] == '"' ? unescape(k) : k.substr(1);
      shMap[Var(ks)] = shv->value();
    }
    var.valueType = SHType::Table;
    var.payload.tableValue.api = &shards::GetGlobals().TableInterface;
    var.payload.tableValue.opaque = &shMap;
    WRAP_TO_CONST(var);
  } else if (malAtom *v = DYNAMIC_CAST(malAtom, arg)) {
    return shardify(v->deref());
  }
  // allow anything else, makes more sense, in order to write inline funcs
  // but print a diagnostic message
  else {
    if (malValue *v = DYNAMIC_CAST(malValue, arg)) {
      SHLOG_INFO("Ignoring value inside a wire definition: {} ignore this "
                 "warning if this was intentional.",
                 v->print(true));
    } else {
      throw shards::SHException("Invalid argument for wire, not a value");
    }
  }
  return result;
}

std::vector<malShardPtr> wireify(malValueIter begin, malValueIter end);

malSHVarPtr varify(const malValuePtr &arg, bool consumeShard) {
  // Returns clones in order to proper cleanup (nested) allocations
  if (arg == mal::nilValue()) {
    SHVar var{};
    auto v = new malSHVar(var, false);
    v->line = arg->line;
    return malSHVarPtr(v);
  } else if (malString *v = DYNAMIC_CAST(malString, arg)) {
    auto &s = v->ref();
    SHVar var{};
    var.valueType = SHType::String;
    var.payload.stringValue = s.c_str();
    var.payload.stringLen = s.size();
    // notice, we don't clone in this case
    auto svar = new malSHVar(var, false);
    svar->reference(v);
    svar->line = arg->line;
    return malSHVarPtr(svar);
  } else if (const malNumber *v = DYNAMIC_CAST(malNumber, arg)) {
    auto value = v->value();
    SHVar var{};
    if (v->isInteger()) {
      var.valueType = SHType::Int;
      var.payload.intValue = value;
    } else {
      var.valueType = SHType::Float;
      var.payload.floatValue = value;
    }
    auto res = new malSHVar(var, false);
    res->line = arg->line;
    return malSHVarPtr(res);
  } else if (malSequence *v = DYNAMIC_CAST(malSequence, arg)) {
    SHVar tmp{};
    tmp.valueType = SHType::Seq;
    tmp.payload.seqValue = {};
    auto count = v->count();
    std::vector<malSHVarPtr> vars;
    for (auto i = 0; i < count; i++) {
      auto val = v->item(i);
      auto subVar = varify(val);
      vars.push_back(subVar);
      shards::arrayPush(tmp.payload.seqValue, subVar->value());
    }
    auto mvar = malSHVar::newCloned(tmp);
    shards::arrayFree(tmp.payload.seqValue);
    for (auto &rvar : vars) {
      mvar->reference(rvar.ptr());
    }
    mvar->line = arg->line;
    return malSHVarPtr(mvar);
  } else if (const malHash *v = DYNAMIC_CAST(malHash, arg)) {
    shards::SHMap shMap;
    std::vector<malSHVarPtr> vars;
    for (auto [k, v] : v->m_map) {
      auto shv = varify(v);
      vars.emplace_back(shv);
      // key is either a quoted string or a keyword (starting with ':')
      auto ks = k[0] == '"' ? unescape(k) : k.substr(1);
      shMap[Var(ks)] = shv->value();
    }
    SHVar tmp{};
    tmp.valueType = SHType::Table;
    tmp.payload.tableValue.api = &shards::GetGlobals().TableInterface;
    tmp.payload.tableValue.opaque = &shMap;
    auto mvar = malSHVar::newCloned(tmp);
    for (auto &rvar : vars) {
      mvar->reference(rvar.ptr());
    }
    mvar->line = arg->line;
    return malSHVarPtr(mvar);
  } else if (arg == mal::trueValue()) {
    SHVar var{};
    var.valueType = SHType::Bool;
    var.payload.boolValue = true;
    auto v = new malSHVar(var, false);
    v->line = arg->line;
    return malSHVarPtr(v);
  } else if (arg == mal::falseValue()) {
    SHVar var{};
    var.valueType = SHType::Bool;
    var.payload.boolValue = false;
    auto v = new malSHVar(var, false);
    v->line = arg->line;
    return malSHVarPtr(v);
  } else if (malSHVar *v = DYNAMIC_CAST(malSHVar, arg)) {
    auto res = malSHVar::newCloned(v->value());
    res->line = arg->line;
    return malSHVarPtr(res);
  } else if (malWireProvider *v = DYNAMIC_CAST(malWireProvider, arg)) {
    SHVar var = *v;
    auto providerVar = new malSHVar(var, false);
    providerVar->reference(v);
    providerVar->line = arg->line;
    return malSHVarPtr(providerVar);
  } else if (malShard *v = DYNAMIC_CAST(malShard, arg)) {
    auto shard = v->value();
    SHVar var{};
    var.valueType = SHType::ShardRef;
    var.payload.shardValue = shard;
    incRef(shard);
    auto bvar = new malSHVar(var, true);
    if (consumeShard)
      v->consume();
    bvar->reference(v);
    bvar->line = arg->line;
    return malSHVarPtr(bvar);
  } else if (malSHWire *v = DYNAMIC_CAST(malSHWire, arg)) {
    auto wire = v->value();
    SHVar var{};
    var.valueType = SHType::Wire;
    var.payload.wireValue = wire;
    auto cvar = new malSHVar(var, false);
    cvar->reference(v);
    cvar->line = arg->line;
    return malSHVarPtr(cvar);
  } else if (malSHMesh *v = DYNAMIC_CAST(malSHMesh, arg)) {
    auto var = shards::Var::Object(&v->sharedRef(), CoreCC, 'brcM');
    auto cvar = new malSHVar(var, false);
    cvar->reference(v);
    cvar->line = arg->line;
    return malSHVarPtr(cvar);
  } else if (malAtom *v = DYNAMIC_CAST(malAtom, arg)) {
    return varify(v->deref());
  } else {
    throw shards::SHException(fmt::format("Invalid variable: {}", arg->print(true)));
  }
}

int findParamIndex(const std::string &name, SHParametersInfo params) {
  for (uint32_t i = 0; params.len > i; i++) {
    auto param = params.elements[i];
    if (param.name == name)
      return (int)i;
  }
  return -1;
}

void validationCallback(const Shard *errorShard, SHStringWithLen errorTxt, bool nonfatalWarning, void *userData) {
  auto shard = const_cast<Shard *>(errorShard);
  if (!nonfatalWarning) {
    SHLOG_ERROR("Parameter validation failed: {} shard: {}", errorTxt, shard->name(shard));
  } else {
    SHLOG_INFO("Parameter validation warning: {} shard: {}", errorTxt, shard->name(shard));
  }
}

void setShardParameters(malShard *malshard, malValueIter begin, malValueIter end) {
  auto shard = malshard->value();
  auto argsBegin = begin;
  auto argsEnd = end;
  auto params = shard->parameters(shard);
  auto pindex = 0;
  while (argsBegin != argsEnd) {
    auto arg = *argsBegin++;
    if (const malKeyword *v = DYNAMIC_CAST(malKeyword, arg)) {
      // In this case it's a keyworded call from now on
      pindex = -1;

      // Jump to next value straight, expecting a value
      auto value = *argsBegin++;
      auto paramName = v->value().substr(1);
      auto idx = findParamIndex(paramName, params);
      if (unlikely(idx == -1)) {
        auto err = fmt::format("Parameter not found: {} shard: {}", paramName, shard->name(shard));
        SHLOG_ERROR(err);
        throw shards::SHException(err);
      } else {
        auto var = varify(value);
        if (!validateSetParam(shard, idx, var->value())) {
          auto err = fmt::format("Failed parameter: {} line: {} shard: {}", paramName, value->line, shard->name(shard));
          SHLOG_ERROR(err);
          throw shards::SHException(err);
        }
        shard->setParam(shard, idx, &var->value());

        // keep ref
        malshard->reference(var.ptr());
      }
    } else if (pindex == -1) {
      // We expected a keyword! fail
      SHLOG_ERROR("Keyword expected, shard: {}", shard->name(shard));
      throw shards::SHException("Keyword expected");
    } else {
      auto var = varify(arg);
      if (!validateSetParam(shard, pindex, var->value())) {
        auto err = fmt::format("Failed parameter index: {} line: {} shard: {}", pindex, arg->line, shard->name(shard));
        SHLOG_ERROR(err);
        throw shards::SHException(err);
      }
      shard->setParam(shard, pindex, &var->value());

      // keep ref
      malshard->reference(var.ptr());

      pindex++;
    }
  }
}

malValuePtr newEnum(int32_t vendor, int32_t type, SHEnum value) {
  SHVar var{};
  var.valueType = SHType::Enum;
  var.payload.enumVendorId = vendor;
  var.payload.enumTypeId = type;
  var.payload.enumValue = value;
  return malValuePtr(new malSHVar(var, false));
}

struct Observer : public shards::RuntimeObserver {
  virtual ~Observer() {}

  malEnvPtr _env;

  void registerShard(const char *fullName, SHShardConstructor constructor) override {
    // do some reflection
    auto shard = constructor();
    assert(shard && "Shard constructor returned NULL");
    // add params keywords if any
    assert(shard->parameters && "Shard is missing parameters call");
    auto params = shard->parameters(shard);
    for (uint32_t i = 0; i < params.len; i++) {
      auto param = params.elements[i];
      _env->set(":" + MalString(param.name), mal::keyword(":" + MalString(param.name)));
#ifndef NDEBUG
      // add coverage for parameters setting
      auto val = shard->getParam(shard, i);
      auto res = shard->setParam(shard, i, &val);
      if (res.code != 0) {
        SHLOG_WARNING("Debug validation failed to set parameter: {} code: {} shard: {}", param.name, res.code,
                      shard->name(shard));
      }
#endif
    }
    // define the new built-in
    MalString mname(fullName);
    malBuiltIn::ApplyFunc *func = [](const MalString &name, malValueIter argsBegin, malValueIter argsEnd) {
      auto shard = shards::createShard(name.c_str());
      shard->setup(shard);
      auto malshard = new malShard(shard);
      setShardParameters(malshard, argsBegin, argsEnd);
      return malValuePtr(malshard);
    };
    auto bi = new malBuiltIn(mname, func);
    builtIns.emplace(mname, bi);
    _env->set(fullName, bi);
    // destroy our sample shard
    shard->destroy(shard);
  }

  void registerEnumType(int32_t vendorId, int32_t typeId, SHEnumInfo info) override {
    for (uint32_t i = 0; i < info.labels.len; i++) {
      auto enumName = MalString(info.name) + "." + MalString(info.labels.elements[i]);
      auto enumValue = newEnum(vendorId, typeId, info.values.elements[i]);
      builtIns[enumName] = enumValue;
      _env->set(enumName, enumValue);
    }

    // Set type keyword
    SHTypeInfo typeInfo{
        .basicType = SHType::Enum,
        .enumeration{.vendorId = vendorId, .typeId = typeId},
    };
    _env->set(MalString(info.name), malValuePtr(new malSHTypeInfo(typeInfo, true)));
  }
};

malShard *makeVarShard(malSHVar *v, const char *shardName) {
  auto b = shards::createShard(shardName);
  b->setup(b);
  b->setParam(b, 0, &v->value());
  auto blk = new malShard(b);
  blk->line = v->line;
  return blk;
}

malShard *makeVarShard(SHVar &v, size_t line, const char *shardName) {
  auto b = shards::createShard(shardName);
  b->setup(b);
  b->setParam(b, 0, &v);
  auto blk = new malShard(b);
  blk->line = line;
  return blk;
}

BUILTIN(">>") { return mal::nilValue(); }

BUILTIN(">>!") { return mal::nilValue(); }

BUILTIN("&>") { return mal::nilValue(); }

BUILTIN(">==") { return mal::nilValue(); }

BUILTIN("??") { return mal::nilValue(); }

BUILTIN("=!") { return mal::nilValue(); }

std::vector<malShardPtr> wireify(malValueIter begin, malValueIter end) {
  enum State { Get, Set, SetGlobal, Update, Ref, RefOverwrite, Push, PushNoClear, AddGetDefault };
  State state = Get;
  std::vector<malShardPtr> res;
  while (begin != end) {
    auto next = *begin++;
    if (auto *v = DYNAMIC_CAST(malSHVar, next)) {
      if (v->value().valueType == SHType::ContextVar) {
        if (state == Get) {
          auto sv = SHSTRVIEW(v->value());
          std::string fullVar(sv);
          std::vector<std::string> components;
          std::string component;
          std::istringstream input_stream(fullVar);
          while (std::getline(input_stream, component, ':')) {
            components.push_back(component);
          }

          shards::Var mainVar(components[0]);
          res.emplace_back(makeVarShard(mainVar, v->line, "Get"));

          // iterate skipping the first one
          for (size_t i = 1; i < components.size(); i++) {
            // if component is an integer we pass by number (seq)
            // if not by string (table)
            char *endptr;
            auto value = std::strtol(components[i].c_str(), &endptr, 10);
            if (*endptr != '\0') {
              // Parsing failed
              shards::Var extraVar(components[i]);
              res.emplace_back(makeVarShard(extraVar, v->line, "Take"));
            } else {
              // Parsing succeeded
              shards::Var extraVar((int64_t(value)));
              res.emplace_back(makeVarShard(extraVar, v->line, "Take"));
            }
          }
        } else if (state == Set) {
          res.emplace_back(makeVarShard(v, "Set"));
          state = Get;
        } else if (state == SetGlobal) {
          auto blk = makeVarShard(v, "Set");
          // set :Global true
          auto val = shards::Var(true);
          blk->value()->setParam(blk->value(), 2, &val);
          res.emplace_back(blk);
          state = Get;
        } else if (state == Update) {
          res.emplace_back(makeVarShard(v, "Update"));
          state = Get;
        } else if (state == Ref) {
          res.emplace_back(makeVarShard(v, "Ref"));
          state = Get;
        } else if (state == RefOverwrite) {
          auto blk = makeVarShard(v, "Ref");
          // set :Overwrite false
          auto val = shards::Var(true);
          blk->value()->setParam(blk->value(), 3, &val);
          res.emplace_back(blk);
          state = Get;
        } else if (state == Push) {
          res.emplace_back(makeVarShard(v, "Push"));
          state = Get;
        } else if (state == PushNoClear) {
          auto blk = makeVarShard(v, "Push");
          // set :Clear false
          auto val = shards::Var(false);
          blk->value()->setParam(blk->value(), 3, &val);
          res.emplace_back(blk);
          state = Get;
        } else {
          throw shards::SHException("Expected a variable");
        }
      } else {
        auto blks = shardify(next);
        res.insert(res.end(), blks.begin(), blks.end());
      }
    } else if (auto *v = DYNAMIC_CAST(malBuiltIn, next)) {
      if (v->name() == ">=") {
        state = Set;
      } else if (v->name() == ">==") {
        state = SetGlobal;
      } else if (v->name() == ">") {
        state = Update;
      } else if (v->name() == "&>" || v->name() == "=") {
        state = Ref;
      } else if (v->name() == "=!") {
        state = RefOverwrite;
      } else if (v->name() == ">>") {
        state = Push;
      } else if (v->name() == ">>!") {
        state = PushNoClear;
      } else if (v->name() == "??") {
        state = AddGetDefault;
      } else {
        throw shards::SHException("Unexpected a token");
      }
    } else {
      if (state == AddGetDefault) {
        auto blk = res.back();
        if (strcmp(blk->value()->name(blk->value()), "Get") != 0) {
          throw shards::SHException("There should be a variable preceding ||");
        }
        // set :Default
        auto v = varify(next);
        blk->value()->setParam(blk->value(), 3, &v->value());
        state = Get;
      } else {
        auto blks = shardify(next);
        res.insert(res.end(), blks.begin(), blks.end());
      }
    }
  }
  return res;
}

BUILTIN("Wire") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malString, wireName);
  auto mwire = new malSHWire(wireName->value());
  auto wireref = mwire->value();
  auto wire = SHWire::sharedFromRef(wireref);
  while (argsBegin != argsEnd) {
    auto pbegin = argsBegin;
    auto arg = *argsBegin++;
    // Option keywords or shards
    if (const malKeyword *v = DYNAMIC_CAST(malKeyword, arg)) {
      if (v->value() == ":Looped") {
        wire->looped = true;
      } else if (v->value() == ":Unsafe") {
        wire->unsafe = true;
      }
#if SH_CORO_NEED_STACK_MEM
      else if (v->value() == ":LStack") {
        wire->stackSize = 4 * 1024 * 1024;  // 4mb
      } else if (v->value() == ":SStack") { // default is 128kb
        wire->stackSize = 32 * 1024;        // 32kb
      }
#endif
      else if (v->value() == ":Pure") {
        wire->pure = true;
      }
    } else {
      auto blks = wireify(pbegin, argsEnd);
      for (auto blk : blks) {
        wire->addShard(blk->value());
        blk->consume();
        mwire->reference(blk.ptr());
      }
      break;
    }
  }
  return malValuePtr(mwire);
}

BUILTIN("ImplWire") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malSHWire, mwire);
  auto wireref = mwire->value();
  auto wire = SHWire::sharedFromRef(wireref);
  // if the wire has already been implemented, clean it up first
  if (wire->shards.size() > 0) {
    wire->cleanup(true);
    for (auto it = wire->shards.rbegin(); it != wire->shards.rend(); ++it) {
      decRef(*it);
    }
    wire->shards.clear();
  }
  while (argsBegin != argsEnd) {
    auto pbegin = argsBegin;
    auto arg = *argsBegin++;
    // Option keywords or shards
    if (const malKeyword *v = DYNAMIC_CAST(malKeyword, arg)) {
      if (v->value() == ":Looped") {
        wire->looped = true;
      } else if (v->value() == ":Unsafe") {
        wire->unsafe = true;
      }
#if SH_CORO_NEED_STACK_MEM
      else if (v->value() == ":LStack") {
        wire->stackSize = 4 * 1024 * 1024;  // 4mb
      } else if (v->value() == ":SStack") { // default is 128kb
        wire->stackSize = 32 * 1024;        // 32kb
      }
#endif
      else if (v->value() == ":Pure") {
        wire->pure = true;
      }
    } else {
      auto blks = wireify(pbegin, argsEnd);
      for (auto blk : blks) {
        wire->addShard(blk->value());
        blk->consume();
        mwire->reference(blk.ptr());
      }
      break;
    }
  }
  return malValuePtr(mwire);
}

BUILTIN("DefWire") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malString, wireName);
  auto mwire = new malSHWire(wireName->value());
  return malValuePtr(mwire);
}

static MalString printValues(malValueIter begin, malValueIter end, const MalString &sep, bool readably) {
  MalString out;

  if (begin != end) {
    out += (*begin)->print(readably);
    ++begin;
  }

  for (; begin != end; ++begin) {
    out += sep;
    out += (*begin)->print(readably);
  }

  return out;
}

SHVar shards::EdnEval::activate(SHContext *context, const SHVar &input) {
  auto env = global ? GetThreadEnv() : malEnvPtr(new malEnv(GetThreadEnv()));

  auto p = prefix.get();
  if (p.valueType == SHType::String) {
    auto sv = SHSTRVIEW(p);
    std::string s(sv.data(), sv.size());
    malEnv::setPrefix(s);
  }
  DEFER(malEnv::unsetPrefix());

  try {
    auto sv = SHSTRVIEW(input);
    std::string s(sv);
    auto malRes = EVAL(READ(s), env);
    auto malVar = varify(malRes);
    output = malVar->value();
  } catch (const MalString &exStr) {
    throw ActivationError(exStr);
  }

  return output;
}

BUILTIN("LOG") {
  SHLOG_INFO(printValues(argsBegin, argsEnd, " ", false));
  return mal::nilValue();
}

BUILTIN("Wire@") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malString, value);
  if (argsBegin != argsEnd) {
    return malValuePtr(new malWireProvider(value->ref(), *argsBegin));
  } else {
    return malValuePtr(new malWireProvider(value->ref()));
  }
}

BUILTIN("Wire*") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malString, value);
  if (argsBegin != argsEnd) {
    return malValuePtr(new malWireProvider(value->ref(), *argsBegin));
  } else {
    return malValuePtr(new malWireProvider(value->ref()));
  }
}

BUILTIN("eval-wire") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malString, value);

  malEnvPtr env(new malEnv());
  auto res = maleval(value->ref().c_str(), env);
  auto var = varify(res);
  if (var->value().valueType != SHType::Wire) {
    SHLOG_ERROR("Script did not return a SHWire");
    return mal::nilValue();
  }

  auto wireref = var->value().payload.wireValue;
  auto wire = SHWire::sharedFromRef(wireref);

  return malValuePtr(new malSHWire(wire));
}

static malValuePtr readVar(const SHVar &v) {
  if (v.valueType == SHType::String) {
    auto sv = SHSTRVIEW(v);
    MalString s(sv);
    return mal::string(s);
  } else if (v.valueType == SHType::Int) {
    return mal::number(double(v.payload.intValue), true);
  } else if (v.valueType == SHType::Float) {
    return mal::number(v.payload.floatValue, false);
  } else if (v.valueType == SHType::Seq) {
    auto vec = new malValueVec();
    for (auto &sv : v) {
      vec->emplace_back(readVar(sv));
    }
    return malValuePtr(new malList(vec));
  } else if (v.valueType == SHType::Table) {
    malHash::Map map;
    auto &t = v.payload.tableValue;
    SHTableIterator tit;
    t.api->tableGetIterator(t, &tit);
    SHVar k;
    SHVar v;
    while (t.api->tableNext(t, &tit, &k, &v)) {
      assert(k.valueType == SHType::String && "Table key is not a string");
      auto sv = SHSTRVIEW(k);
      MalString ks(sv);
      map[escape(ks)] = readVar(v);
    }
    return mal::hash(map);
  } else {
    // just return a variable in this case, but a copy, not cloning inner
    return malValuePtr(new malSHVar(v, false));
  }
}

BUILTIN("read-var") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malSHVar, value);
  auto &v = value->value();
  return readVar(v);
}

BUILTIN("->") {
  auto vec = new malValueVec();
  auto blks = wireify(argsBegin, argsEnd);
  for (auto blk : blks) {
    malShard *pblk = blk.ptr();
    vec->emplace_back(pblk);
  }
  return malValuePtr(new malList(vec));
}

BUILTIN("wireify") {
  CHECK_ARGS_IS(1);
  ARG(malSequence, value);
  auto vec = new malValueVec();
  auto blks = wireify(value->begin(), value->end());
  for (auto blk : blks) {
    malShard *pblk = blk.ptr();
    vec->emplace_back(pblk);
  }
  return malValuePtr(new malList(vec));
}

BUILTIN("unquote") {
  CHECK_ARGS_IS(1);
  auto vec = new malValueVec();
  auto await = new malShard("Await");
  await->value()->setup(await->value());
  auto first = *argsBegin++;
  if (malShard *shrd = DYNAMIC_CAST(malShard, first)) {
    vec->emplace_back(shrd);
  } else if (malSequence *seq = DYNAMIC_CAST(malSequence, first)) {
    auto blks = wireify(seq->begin(), seq->end());
    for (auto blk : blks) {
      malShard *pblk = blk.ptr();
      vec->emplace_back(pblk);
    }
  } else {
    throw shards::SHException("Expected Shard or Shards sequence");
  }

  auto shardsVP = malValuePtr(new malList(vec));
  auto shardsVar = varify(shardsVP);
  await->value()->setParam(await->value(), 0, &shardsVar->value());
  await->reference(shardsVP);

  return malValuePtr(await);
}

BUILTIN("schedule") {
  CHECK_ARGS_IS(2);
  ARG(malSHMesh, mesh);
  ARG(malSHWire, wire);
  mesh->schedule(wire);
  return mal::nilValue();
}

BUILTIN("schedule-fast") {
  CHECK_ARGS_IS(2);
  ARG(malSHMesh, mesh);
  ARG(malSHWire, wire);
  mesh->schedule(wire, false);
  return mal::nilValue();
}

BUILTIN("pre-compose") {
  CHECK_ARGS_IS(2);
  ARG(malSHMesh, mesh);
  ARG(malSHWire, wire);
  mesh->compose(wire);
  return mal::nilValue();
}

// used in wires without a mesh (manually prepared etc)
thread_local std::shared_ptr<SHMesh> TLSRootSHMesh{SHMesh::make()};

BUILTIN("prepare") {
  CHECK_ARGS_IS(1);
  ARG(malSHWire, wirevar);
  auto wire = SHWire::sharedFromRef(wirevar->value());
  SHInstanceData data{};
  data.wire = wire.get();
  wire->mesh = TLSRootSHMesh->shared_from_this();
  auto composeResult = composeWire(wire.get(), data);
  wire->composedHash = Var(1, 1); // just to mark as composed
  shards::arrayFree(composeResult.exposedInfo);
  shards::prepare(wire.get());
  return mal::nilValue();
}

BUILTIN("set-global-wire") {
  CHECK_ARGS_IS(1);
  auto first = *argsBegin;
  if (const malSHWire *v = DYNAMIC_CAST(malSHWire, first)) {
    auto wire = SHWire::sharedFromRef(v->value());
    shards::GetGlobals().GlobalWires[wire->name] = wire;
    return mal::trueValue();
  }
  return mal::falseValue();
}

BUILTIN("unset-global-wire") {
  CHECK_ARGS_IS(1);
  auto first = *argsBegin;
  if (const malSHWire *v = DYNAMIC_CAST(malSHWire, first)) {
    auto wire = SHWire::sharedFromRef(v->value());
    auto it = shards::GetGlobals().GlobalWires.find(wire->name);
    if (it != shards::GetGlobals().GlobalWires.end()) {
      shards::GetGlobals().GlobalWires.erase(it);
      return mal::trueValue();
    }
  }
  return mal::falseValue();
}

BUILTIN("run-empty-forever") {
  while (true) {
    shards::sleep(1.0);
  }
}

BUILTIN("start") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malSHWire, wirevar);
  auto wire = SHWire::sharedFromRef(wirevar->value());
  if (argsBegin != argsEnd) {
    ARG(malSHVar, inputVar);
    shards::start(wire.get(), inputVar->value());
  } else {
    shards::start(wire.get());
  }
  return mal::nilValue();
}

BUILTIN("tick") {
  CHECK_ARGS_IS(1);
  auto first = *argsBegin;
  if (const malSHWire *v = DYNAMIC_CAST(malSHWire, first)) {
    auto wire = SHWire::sharedFromRef(v->value());
    SHDuration now = SHClock::now().time_since_epoch();
    shards::tick(wire->tickingWire(), now);
    return mal::boolean(true);
  } else if (const malSHMesh *v = DYNAMIC_CAST(malSHMesh, first)) {
    auto noErrors = v->value()->tick();
    return mal::boolean(noErrors);
  } else {
    throw shards::SHException("tick Expected Mesh or Wire");
  }
  return mal::nilValue();
}

BUILTIN("stop") {
  CHECK_ARGS_IS(1);
  ARG(malSHWire, wirevar);
  auto wire = SHWire::sharedFromRef(wirevar->value());
  SHVar res{};
  shards::stop(wire.get(), &res);
  return malValuePtr(new malSHVar(res, true));
}

bool run(SHMesh *mesh, SHWire *wire, double sleepTime, int times, bool dec) {
  SHDuration dsleep(sleepTime);
  auto logTicks = 0;

  if (mesh) {
    auto now = SHClock::now();
    auto next = now + dsleep;
    while (!mesh->empty()) {
      const auto noErrors = mesh->tick();

      // other wires might be not in error tho...
      // so return only if empty
      if (!noErrors && mesh->empty()) {
        return false;
      }

      if (dec) {
        times--;
        if (times == 0) {
          mesh->terminate();
          break;
        }
      }
      // We on purpose run terminate (eventually)
      // before sleep
      // cos during sleep some shards
      // swap states and invalidate stuff
      if (sleepTime <= 0.0) {
        shards::sleep(-1.0);
      } else {
        // remove the time we took to tick from sleep
        now = SHClock::now();
        SHDuration realSleepTime = next - now;
        if (unlikely(realSleepTime.count() <= 0.0)) {
          // tick took too long!!!
          if (++logTicks >= 1000) {
            logTicks = 0;
            SHLOG_WARNING("Mesh tick took too long: {}ms", -realSleepTime.count() * 1000.0);
          }
          next = now + dsleep;
        } else {
          ++logTicks;
          next = next + dsleep;
          shards::sleep(realSleepTime.count());
        }
      }
    }
  } else {
    SHDuration now = SHClock::now().time_since_epoch();
    do {
      shards::tick(wire->tickingWire(), now);
      if (dec) {
        times--;
        if (times == 0) {
          shards::stop(wire);
          break;
        }
      }
      shards::sleep(sleepTime);
      now = SHClock::now().time_since_epoch();
    } while (!shards::isRunning(wire));
  }

  return true;
}

BUILTIN("run") {
  CHECK_ARGS_AT_LEAST(1);
  SHMesh *mesh = nullptr;
  SHWire *wire = nullptr;
  auto first = *argsBegin++;
  if (const malSHWire *v = DYNAMIC_CAST(malSHWire, first)) {
    auto cs = SHWire::sharedFromRef(v->value());
    wire = cs.get();
  } else if (const malSHMesh *v = DYNAMIC_CAST(malSHMesh, first)) {
    mesh = v->value();
  } else {
    throw shards::SHException("tick Expected Mesh or Wire");
  }

  auto sleepTime = -1.0;
  if (argsBegin != argsEnd) {
    ARG(malNumber, argSleepTime);
    sleepTime = argSleepTime->value();
  }

  int times = 1;
  bool dec = false;
  if (argsBegin != argsEnd) {
    ARG(malNumber, argTimes);
    times = argTimes->value();
    dec = true;
  }

  auto noErrors = run(mesh, wire, sleepTime, times, dec);

  spdlog::default_logger()->flush();

  return mal::boolean(noErrors);
}

BUILTIN("run-many") {
  CHECK_ARGS_AT_LEAST(1);
  ARG(malSequence, mMeshes);

  std::vector<SHMesh *> meshes;

  auto count = mMeshes->count();
  for (auto i = 0; i < count; i++) {
    auto mMesh = mMeshes->item(i);
    if (const malSHMesh *v = DYNAMIC_CAST(malSHMesh, mMesh)) {
      meshes.push_back(v->value());
    } else {
      throw shards::SHException("run-many expects Mesh only");
    }
  }

  auto sleepTime = -1.0;
  if (argsBegin != argsEnd) {
    ARG(malNumber, argSleepTime);
    sleepTime = argSleepTime->value();
  }

  int times = 1;
  bool dec = false;
  if (argsBegin != argsEnd) {
    ARG(malNumber, argTimes);
    times = argTimes->value();
    dec = true;
  }

  std::vector<std::future<bool>> futures;

  // Run all other meshes on another thread
  for (size_t i = 1; i < meshes.size(); i++) {
    auto fut = std::async(std::launch::async, [=]() { return run(meshes[i], nullptr, sleepTime, times, dec); });
    futures.push_back(std::move(fut));
  }

  assert(futures.size() == size_t(count - 1));

  // Run this first mesh on this thread
  run(meshes[0], nullptr, sleepTime, times, dec);

  // wait for all futures to finish
  bool noErrors = true;
  for (auto &fut : futures) {
    if (!fut.get())
      noErrors = false;
  }

  spdlog::default_logger()->flush();

  return mal::boolean(noErrors);
}

BUILTIN("sleep") {
  CHECK_ARGS_IS(1);
  ARG(malNumber, sleepTime);
  shards::sleep(sleepTime->value());
  return mal::nilValue();
}

BUILTIN("override-root-path") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);

  shards::GetGlobals().RootPath = value->ref();
  fs::current_path(value->ref());
  return mal::nilValue();
}

BUILTIN("path") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);
  auto &s = value->ref();
  SHVar var{};
  var.valueType = SHType::Path;
  var.payload.stringValue = s.c_str();
  var.payload.stringLen = s.size();
  auto mvar = new malSHVar(var, false);
  mvar->reference(value);
  return malValuePtr(mvar);
}

BUILTIN("enum") {
  CHECK_ARGS_IS(3);
  ARG(malNumber, value0);
  ARG(malNumber, value1);
  ARG(malNumber, value2);
  SHVar var{};
  var.valueType = SHType::Enum;
  var.payload.enumVendorId = static_cast<int32_t>(value0->value());
  var.payload.enumTypeId = static_cast<int32_t>(value1->value());
  var.payload.enumValue = static_cast<SHEnum>(value2->value());
  return malValuePtr(new malSHVar(var, false));
}

BUILTIN("string") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);
  SHVar var{};
  var.valueType = SHType::String;
  auto &s = value->ref();
  var.payload.stringValue = s.c_str();
  var.payload.stringLen = s.size();
  auto mvar = new malSHVar(var, false);
  mvar->reference(value);
  return malValuePtr(mvar);
}

std::string hex2Bytes(const std::string &s) {
  std::string bytes;
  for (size_t i = 0; i < s.size(); i += 2) {
    auto c = s[i];
    auto c2 = s[i + 1];
    auto v = 0;
    if (c >= '0' && c <= '9') {
      v = (c - '0') << 4;
    } else if (c >= 'a' && c <= 'f') {
      v = (c - 'a' + 10) << 4;
    } else if (c >= 'A' && c <= 'F') {
      v = (c - 'A' + 10) << 4;
    } else {
      throw shards::SHException("Invalid hex string");
    }
    if (c2 >= '0' && c2 <= '9') {
      v |= (c2 - '0');
    } else if (c2 >= 'a' && c2 <= 'f') {
      v |= (c2 - 'a' + 10);
    } else if (c2 >= 'A' && c2 <= 'F') {
      v |= (c2 - 'A' + 10);
    } else {
      throw shards::SHException("Invalid hex string");
    }
    bytes.push_back(v);
  }
  return bytes;
}

BUILTIN("bytes") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);
  SHVar var{};
  var.valueType = SHType::Bytes;
  auto &s = value->ref();
  if (s.size() > 2 && s[0] == '0' && s[1] == 'x') {
    auto bytes = hex2Bytes(s.substr(2));
    var.payload.bytesValue = (uint8_t *)bytes.data();
    var.payload.bytesSize = bytes.size();
    auto mvar = malSHVar::newCloned(var);
    return malValuePtr(mvar);
  } else {
    var.payload.bytesValue = (uint8_t *)s.data();
    var.payload.bytesSize = s.size();
    auto mvar = new malSHVar(var, false);
    mvar->reference(value);
    return malValuePtr(mvar);
  }
}

BUILTIN("context-var") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);
  SHVar var{};
  var.valueType = SHType::ContextVar;
  auto &s = value->ref();
  var.payload.stringValue = s.c_str();
  var.payload.stringLen = s.size();
  auto mvar = new malSHVar(var, false);
  mvar->reference(value);
  return malValuePtr(mvar);
}

template <SHType T> struct GetComponentType {};
template <> struct GetComponentType<SHType::Float2> {
  typedef double Type;
};
template <> struct GetComponentType<SHType::Float3> {
  typedef float Type;
};
template <> struct GetComponentType<SHType::Float4> {
  typedef float Type;
};
template <> struct GetComponentType<SHType::Int2> {
  typedef int64_t Type;
};
template <> struct GetComponentType<SHType::Int3> {
  typedef int32_t Type;
};
template <> struct GetComponentType<SHType::Int4> {
  typedef int32_t Type;
};
template <> struct GetComponentType<SHType::Int8> {
  typedef int16_t Type;
};
template <> struct GetComponentType<SHType::Int16> {
  typedef int8_t Type;
};
template <> struct GetComponentType<SHType::Color> {
  typedef uint8_t Type;
  static constexpr uint8_t getDefaultValue(size_t index) { return index == 3 ? 255 : 0; }
};

// Generates a vector constant or generating shards
//  passing a single argument broadcasts it to all the components
//  Setting AllowDefaultComponents will allow any number of arguments and fill in the remainder from the default value (0 0 0 1)
template <size_t VectorSize, SHType Type, bool AllowDefaultComponents = false>
malValuePtr makeVector(const MalString &name, malValueIter argsBegin, malValueIter argsEnd, const char *shardName) {
  typedef typename GetComponentType<Type>::Type ValueType;
  constexpr bool IsFloat = std::is_floating_point_v<ValueType>;

  size_t numArgs = std::distance(argsBegin, argsEnd);
  bool broadcast = false;
  if (numArgs == 1) {
    broadcast = true;
  } else {
    if constexpr (AllowDefaultComponents) {
      if (numArgs > VectorSize) {
        throw STRF("Too many arguments to vector constructor: %zu, %zu required", numArgs, VectorSize);
      }
    } else {
      if (numArgs != VectorSize)
        throw STRF("Not enough arguments to vector constructor: %zu, %zu required", numArgs, VectorSize);
    }
  }
  size_t inputSize = broadcast ? 1 : numArgs;

  bool containsVariables = false;
  auto it = argsBegin;
  for (size_t i = 0; i < inputSize; ++i, ++it) {
    if (malNumber *number = dynamic_cast<malNumber *>(it->ptr())) {
      // ignore
    } else if (malSHVar *number = dynamic_cast<malSHVar *>(it->ptr())) {
      containsVariables = true;
    } else {
      throw STRF("vector constructor expects a number or a symbol");
    }
  }

  if (containsVariables) {
    auto b = shards::createShard(shardName);
    b->setup(b);

    auto it = argsBegin;
    for (size_t i = 0; i < inputSize; ++i, ++it) {
      SHVar var{};
      if (malNumber *number = dynamic_cast<malNumber *>(it->ptr())) {
        if constexpr (IsFloat) {
          var.valueType = SHType::Float;
          var.payload.floatValue = SHFloat(number->value());
        } else {
          var.valueType = SHType::Int;
          var.payload.intValue = SHInt(number->value());
        }
      } else if (malSHVar *inVar = dynamic_cast<malSHVar *>(it->ptr())) {
        var = inVar->value();
      } else {
        throw STRF("vector constructor expects a number or a symbol");
      }
      b->setParam(b, i, &var);
    }

    auto blk = new malShard(b);
    return blk;
  } else {

    SHVar var{};
    var.valueType = Type;
    ValueType *out = reinterpret_cast<ValueType *>(&var.payload.floatValue);

    it = argsBegin;
    for (size_t i = 0; i < inputSize; ++i, ++it) {
      auto v = VALUE_CAST(malNumber, it->ptr());
      out[i] = ValueType(v->value());
    }

    if (broadcast) {
      for (size_t i = 1; i < VectorSize; ++i) {
        out[i] = out[0];
      }
    } else {
      if constexpr (AllowDefaultComponents) {
        for (size_t i = inputSize; i < VectorSize; ++i) {
          out[i] = GetComponentType<Type>::getDefaultValue(i);
        }
      }
    }

    return malValuePtr(new malSHVar(var, false));
  }
}

BUILTIN("int") {
  CHECK_ARGS_IS(1);
  ARG(malNumber, value);
  SHVar var{};
  var.valueType = SHType::Int;
  var.payload.intValue = value->value();
  return malValuePtr(new malSHVar(var, false));
}

BUILTIN("int2") { return makeVector<2, SHType::Int2>(name, argsBegin, argsEnd, "MakeInt2"); }

BUILTIN("int3") { return makeVector<3, SHType::Int3>(name, argsBegin, argsEnd, "MakeInt3"); }

BUILTIN("int4") { return makeVector<4, SHType::Int4>(name, argsBegin, argsEnd, "MakeInt4"); }

BUILTIN("int8") { return makeVector<8, SHType::Int8>(name, argsBegin, argsEnd, "MakeInt8"); }

BUILTIN("int16") { return makeVector<16, SHType::Int16>(name, argsBegin, argsEnd, "MakeInt16"); }

BUILTIN("color") { return makeVector<4, SHType::Color, true>(name, argsBegin, argsEnd, "MakeColor"); }

BUILTIN("float") {
  CHECK_ARGS_IS(1);
  ARG(malNumber, value);
  SHVar var{};
  var.valueType = SHType::Float;
  var.payload.floatValue = value->value();
  return malValuePtr(new malSHVar(var, false));
}

BUILTIN("float2") { return makeVector<2, SHType::Float2>(name, argsBegin, argsEnd, "MakeFloat2"); }

BUILTIN("float3") { return makeVector<3, SHType::Float3>(name, argsBegin, argsEnd, "MakeFloat3"); }

BUILTIN("float4") { return makeVector<4, SHType::Float4>(name, argsBegin, argsEnd, "MakeFloat4"); }

BUILTIN("import") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);

  auto filepath = fs::path(value->value());

  auto lib_name_str = filepath.string();
  auto lib_name = lib_name_str.c_str();

  // FIXME we are leaking libs
#if _WIN32
  SHLOG_INFO("Importing DLL: {}", lib_name);
  auto handle = LoadLibraryA(lib_name);
  if (!handle) {
    SHLOG_ERROR("LoadLibrary failed.");
  }
#elif defined(__linux__) || defined(__APPLE__)
  SHLOG_INFO("Importing Shared Library: {}", lib_name);
  auto handle = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    SHLOG_ERROR("dlerror: {}", dlerror());
  }
#else
  void *handle = nullptr;
#endif

  if (!handle) {
    return mal::falseValue();
  }
  return mal::trueValue();
}

BUILTIN("getenv") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);
#ifdef SHARDS_DESKTOP
  auto envs = boost::this_process::environment();
  auto env_value = envs[value->value()].to_string();
  return mal::string(env_value);
#else
  return mal::string(getenv(value->ref().c_str()));
#endif
}

BUILTIN("setenv") {
  CHECK_ARGS_IS(2);
  ARG(malString, key);
  ARG(malString, value);
#ifdef SHARDS_DESKTOP
  auto envs = boost::this_process::environment();
  envs[key->value()] = value->value();
  return mal::trueValue();
#else
  if (!setenv(key->ref().c_str(), value->ref().c_str(), 1))
    return mal::trueValue();
  else
    return mal::falseValue();
#endif
}

BUILTIN("settings-try-set-min") {
  CHECK_ARGS_IS(2);
  ARG(malString, key);
  ARG(malSHVar, value);
  auto &current = GetGlobals().Settings[key->value()];
  if (current.valueType != SHType::None) {
    if (value->value() < current) {
      current = value->value();
      return mal::trueValue();
    } else {
      return mal::falseValue();
    }
  } else {
    current = value->value();
    return mal::trueValue();
  }
}

BUILTIN("settings-try-set-max") {
  CHECK_ARGS_IS(2);
  ARG(malString, key);
  ARG(malSHVar, value);
  auto &current = GetGlobals().Settings[key->value()];
  if (current.valueType != SHType::None) {
    if (value->value() > current) {
      current = value->value();
      return mal::trueValue();
    } else {
      return mal::falseValue();
    }
  } else {
    current = value->value();
    return mal::trueValue();
  }
}

BUILTIN("hasShard?") {
  CHECK_ARGS_IS(1);
  ARG(malString, value);
  if (builtIns.find(value->value()) != builtIns.end()) {
    return mal::trueValue();
  } else {
    return mal::falseValue();
  }
}

BUILTIN("shards") {
  malValueVec v;
  for (auto [name, _] : shards::GetGlobals().ShardsRegister) {
    v.emplace_back(mal::string(std::string(name)));
  }
  malValueVec *items = new malValueVec(v);
  return mal::list(items);
}

BUILTIN("enums") {
  malValueVec v;
  for (auto [_, ti] : shards::GetGlobals().EnumTypesRegister) {
    v.emplace_back(mal::string(std::string(ti.name)));
  }
  malValueVec *items = new malValueVec(v);
  return mal::list(items);
}

static bool containsNone(const SHTypesInfo &types) {
  for (size_t i = 0; i < types.len; i++) {
    if (types.elements[i].basicType == SHType::None)
      return true;
  }
  return false;
}

static malValuePtr richTypeInfo(const SHTypesInfo &types, bool ignoreNone = true) {
  malValueVec vec;
  for (uint32_t j = 0; j < types.len; j++) {
    auto &type = types.elements[j];
    if (ignoreNone && type.basicType == SHType::None)
      continue;
    malHash::Map map;
    std::stringstream ss;
    ss << type;
    map[":name"] = mal::string(ss.str());
    map[":type"] = mal::number(double(type.basicType), true);
    vec.emplace_back(mal::hash(map));
  }
  return mal::list(vec.begin(), vec.end());
}

BUILTIN("shard-info") {
  CHECK_ARGS_IS(1);
  ARG(malString, blkname);
  const auto blkIt = builtIns.find(blkname->ref());
  SHLOG_TRACE("shard-info {}", blkname->value());
  if (blkIt == builtIns.end()) {
    return mal::nilValue();
  } else {
    malHash::Map map;
    auto shard = createShard(blkname->ref().c_str());
    DEFER(shard->destroy(shard));

    {
      auto help = shard->help(shard);
      map[":help"] = mal::string(help.string ? help.string : getString(help.crc));
    }

    auto params = shard->parameters(shard);
    malValueVec pvec;
    for (uint32_t i = 0; i < params.len; i++) {
      malHash::Map pmap;
      pmap[":name"] = mal::string(params.elements[i].name);
      SHLOG_TRACE(" param {}", params.elements[i].name);
      auto &help = params.elements[i].help;
      pmap[":help"] = mal::string(help.string ? help.string : getString(help.crc));
      pmap[":types"] = richTypeInfo(params.elements[i].valueTypes);
      pmap[":optional"] = mal::boolean(containsNone(params.elements[i].valueTypes));
      {
        std::ostringstream ss;
        auto param = shard->getParam(shard, (int)i);
        if (param.valueType == SHType::String) {
          auto sv = SHSTRVIEW(param);
          ss << "\"" << sv << "\"";
        } else {
          ss << param;
        }
        pmap[":default"] = mal::string(ss.str());
      }
      pvec.emplace_back(mal::hash(pmap));
    }
    map[":parameters"] = mal::list(pvec.begin(), pvec.end());

    {
      map[":inputTypes"] = richTypeInfo(shard->inputTypes(shard));
      auto help = shard->inputHelp(shard);
      map[":inputHelp"] = mal::string(help.string ? help.string : getString(help.crc));
    }

    {
      map[":outputTypes"] = richTypeInfo(shard->outputTypes(shard));
      auto help = shard->outputHelp(shard);
      map[":outputHelp"] = mal::string(help.string ? help.string : getString(help.crc));
    }

    {
      auto properties = shard->properties(shard);
      if (unlikely(properties != nullptr)) {
        malHash::Map pmap;
        ForEach(*properties, [&](auto &key, auto &val) {
          assert(key.valueType == SHType::String && "property key is not a string");
          auto sv = SHSTRVIEW(key);
          MalString ks(sv);
          pmap[escape(ks)] = readVar(val);
        });
        map[":properties"] = mal::hash(pmap);
      }
    }

    return mal::hash(map);
  }
}

static SHEnumInfo *findEnumByName(const std::string &name) {
  for (auto &it : GetGlobals().EnumTypesRegister) {
    if (it.second.name == name)
      return &it.second;
  }
  return nullptr;
}

BUILTIN("enum-info") {
  CHECK_ARGS_IS(1);
  ARG(malString, enumName);

  SHEnumInfo *enumInfo = findEnumByName(enumName->value());
  assert(enumInfo);

  malHash::Map map;
  map[":name"] = mal::string(enumInfo->name);

  malValueVec values;
  assert(enumInfo->descriptions.len == enumInfo->labels.len);
  for (size_t i = 0; i < enumInfo->labels.len; i++) {
    malHash::Map elemMap;
    SHString label = enumInfo->labels.elements[i];
    SHOptionalString description = enumInfo->descriptions.elements[i];
    elemMap[":label"] = mal::string(label);
    elemMap[":description"] = mal::string(description.string ? description.string : getString(description.crc));
    values.emplace_back(mal::hash(elemMap));
  }
  map[":values"] = mal::list(values.begin(), values.end());

  return mal::hash(map);
}

#ifndef SH_COMPRESSED_STRINGS
BUILTIN("export-strings") {
  assert(shards::GetGlobals().CompressedStrings);
  malValueVec strs;
  for (auto &[crc, str] : *shards::GetGlobals().CompressedStrings) {
    if (crc != 0) {
      malValueVec record;
      record.emplace_back(mal::number(crc, true));
      record.emplace_back(mal::string(str.string));
      strs.emplace_back(mal::list(record.begin(), record.end()));
    }
  }
  return mal::list(strs.begin(), strs.end());
}
#else
static std::unordered_map<uint32_t, std::string> strings_storage;
#endif

BUILTIN("decompress-strings") {
#ifdef SH_COMPRESSED_STRINGS
  if (!shards::GetGlobals().CompressedStrings) {
    throw shards::SHException("String storage was null");
  }
  // run the script to populate compressed strings
  auto bytes = Var(__shards_compressed_strings);
  auto wire = ::shards::Wire("decompress strings").let(bytes).shard("Brotli.Decompress").shard("FromBytes");
  auto mesh = SHMesh::make();
  mesh->schedule(wire);
  mesh->tick();
  if (wire->finishedOutput.valueType != SHType::Seq) {
    throw shards::SHException("Failed to decompress strings!");
  }

  for (uint32_t i = 0; i < wire->finishedOutput.payload.seqValue.len; i++) {
    auto pair = wire->finishedOutput.payload.seqValue.elements[i];
    if (pair.valueType != SHType::Seq || pair.payload.seqValue.len != 2) {
      throw shards::SHException("Failed to decompress strings!");
    }
    auto crc = pair.payload.seqValue.elements[0];
    auto str = pair.payload.seqValue.elements[1];
    if (crc.valueType != SHType::Int || str.valueType != SHType::String) {
      throw shards::SHException("Failed to decompress strings!");
    }
    auto emplaced = strings_storage.emplace(uint32_t(crc.payload.intValue), str.payload.stringValue);
    auto &s = emplaced.first->second;
    auto &val = (*shards::GetGlobals().CompressedStrings)[uint32_t(crc.payload.intValue)];
    val.string = s.c_str();
    val.crc = uint32_t(crc.payload.intValue);
  }
#endif
  return mal::nilValue();
}

BUILTIN_ISA("Var?", malSHVar);
BUILTIN_ISA("Mesh?", malSHMesh);
BUILTIN_ISA("Wire?", malSHWire);
BUILTIN_ISA("Shard?", malShard);

static SHTypeInfo deriveTypeInfoFromMal(malValuePtr value);

static SHTypeInfo deriveBasicBuiltinTypeInfo(const malBuiltIn &builtin) {
  auto &name = builtin.name();
  if (name == "int") {
    return SHTypeInfo{.basicType = SHType::Int};
  } else if (name == "int2") {
    return SHTypeInfo{.basicType = SHType::Int2};
  } else if (name == "int3") {
    return SHTypeInfo{.basicType = SHType::Int3};
  } else if (name == "int4") {
    return SHTypeInfo{.basicType = SHType::Int4};
  } else if (name == "int8") {
    return SHTypeInfo{.basicType = SHType::Int8};
  } else if (name == "int16") {
    return SHTypeInfo{.basicType = SHType::Int16};
  } else if (name == "float") {
    return SHTypeInfo{.basicType = SHType::Float};
  } else if (name == "float2") {
    return SHTypeInfo{.basicType = SHType::Float2};
  } else if (name == "float3") {
    return SHTypeInfo{.basicType = SHType::Float3};
  } else if (name == "float4") {
    return SHTypeInfo{.basicType = SHType::Float4};
  } else if (name == "color") {
    return SHTypeInfo{.basicType = SHType::Color};
  } else if (name == "bytes") {
    return SHTypeInfo{.basicType = SHType::Bytes};
  } else if (name == "path") {
    return SHTypeInfo{.basicType = SHType::Path};
  } else if (name == "string") {
    return SHTypeInfo{.basicType = SHType::String};
  } else if (name == "Wire") {
    return SHTypeInfo{.basicType = SHType::Wire};
  } else {
    throw std::logic_error(fmt::format("Can not convert basic type: {}", name));
  }
}

static SHTypeInfo deriveSeqType(const malSequence &seq) {
  SHTypeInfo type{};
  type.basicType = SHType::Seq;
  SHTypesInfo seqTypes{};
  try {
    for (int i = 0; i < seq.count(); i++) {
      auto item = seq.item(i);
      arrayPush(seqTypes, deriveTypeInfoFromMal(item));
    }
  } catch (...) {
    type.seqTypes = seqTypes;
    freeDerivedInfo(type);
    throw;
  }

  type.seqTypes = seqTypes;
  return type;
}

static SHTypeInfo deriveMapType(const malHash &hash) {
  SHTypeInfo type{};
  type.basicType = SHType::Table;
  try {
    for (auto &[key, value] : hash.m_map) {
      Var newKey{};
      (OwnedVar &)newKey = (Var)key.substr(1);
      arrayPush(type.table.keys, newKey);
      arrayPush(type.table.types, deriveTypeInfoFromMal(value));
    }
  } catch (...) {
    freeDerivedInfo(type);
    throw;
  }

  return type;
}

static SHTypeInfo deriveTypeInfoFromMal(malValuePtr value) {
  std::optional<SHTypeInfo> type;
  if (const malSHVar *v = DYNAMIC_CAST(malSHVar, value)) {
    const SHVar &var = v->value();
    if (var.valueType == SHType::Type) {
      return cloneTypeInfo(*var.payload.typeValue);
    }
  } else if (const malBuiltIn *v = DYNAMIC_CAST(malBuiltIn, value)) {
    type = deriveBasicBuiltinTypeInfo(*v);
  } else if (const malSequence *v = DYNAMIC_CAST(malSequence, value)) {
    type = deriveSeqType(*v);
  } else if (const malHash *v = DYNAMIC_CAST(malHash, value)) {
    type = deriveMapType(*v);
  } else if (const malShard *ms = DYNAMIC_CAST(malShard, value)) {
    auto *shard = ms->value();
    auto outputTypes = shard->outputTypes(shard);
    if (outputTypes.len != 1) {
      throw std::logic_error(fmt::format("Can retrieve single output type from shard {}", shard->name(shard)));
    }
    type = cloneTypeInfo(outputTypes.elements[0]);
  } else if (const malSHTypeInfo *ti = DYNAMIC_CAST(malSHTypeInfo, value)) {
    return cloneTypeInfo(ti->value());
  }

  if (type)
    return type.value();

  throw std::logic_error(fmt::format("Can not convert value to type info: {}", value->print(true)));
}

BUILTIN("var-type") {
  CHECK_ARGS_IS(1);
  auto &arg = *argsBegin;

  auto innerType = deriveTypeInfoFromMal(arg);

  SHTypesInfo types{};
  arrayPush(types, innerType);

  SHVar var{
      .payload{
          .typeValue =
              new SHTypeInfo{
                  SHType::ContextVar,
                  {
                      .contextVarTypes = types,
                  },
              },
      },
      .valueType = SHType::Type,
  };
  return malValuePtr(new malSHVar(var, true));
}

BUILTIN("type") {
  CHECK_ARGS_IS(1);
  auto &arg = *argsBegin;

  auto typeInfo = deriveTypeInfoFromMal(arg);

  SHVar var{.payload{.typeValue = new SHTypeInfo(typeInfo)}, .valueType = SHType::Type};
  return malValuePtr(new malSHVar(var, true));
}

// Gives the input type of the given shard
BUILTIN("input-type") {
  CHECK_ARGS_IS(1);
  auto &arg = *argsBegin;

  if (const malShard *ms = DYNAMIC_CAST(malShard, arg)) {
    auto *shard = ms->value();
    auto inputTypes = shard->inputTypes(shard);
    if (inputTypes.len != 1) {
      throw std::logic_error(fmt::format("Can retrieve single input type from shard {}", shard->name(shard)));
    }

    SHVar var{.payload{.typeValue = new SHTypeInfo(cloneTypeInfo(inputTypes.elements[0]))}, .valueType = SHType::Type};
    return malValuePtr(new malSHVar(var, true));
  }

  throw std::logic_error(fmt::format("Can not convert value to type info: {}", arg->print(true)));
}

extern "C" {
SHLISP_API __cdecl void *shLispCreate(const char *path) {
  try {
    auto env = new malEnvPtr(new malEnv);
    malinit(*env, path, path);
    return (void *)env;
  } catch (std::exception &ex) {
    SHLOG_FATAL("Failed to create lisp environment: {}", ex.what());
    return nullptr;
  }
}

SHLISP_API __cdecl void *shLispCreateSub(void *parent) {
  assert(parent != nullptr);
  auto env = new malEnvPtr(new malEnv(*reinterpret_cast<malEnvPtr *>(parent)));
  return (void *)env;
}

SHLISP_API __cdecl void shLispSetPrefix(const char *envNamespace) {
  if (envNamespace)
    malEnv::setPrefix(envNamespace);
  else
    malEnv::unsetPrefix();
}

SHLISP_API __cdecl void shLispDestroy(void *env) {
  auto penv = (malEnvPtr *)env;
  {
    std::scoped_lock obsLock(observersMutex);
    observers.erase((malEnv *)penv->ptr());
  }
  delete penv;
}

SHLISP_API __cdecl SHBool shLispEval(void *env, const char *str, SHVar *output) {
  auto penv = (malEnvPtr *)env;
  try {
    malValuePtr res;
    if (penv) {
      res = maleval(str, *penv);
    } else {
      auto cenv = malenv();
      res = maleval(str, cenv);
    }
    if (output) {
      auto mvar = varify(res, false); // don't consume if it's a shard
      auto scriptVal = mvar->value();
      shards::cloneVar(*output, scriptVal);
    }
    return true;
  } catch (malEmptyInputException &) {
    return false;
  } catch (malValuePtr &mv) {
    SHLOG_WARNING("Eval error: {} line: {}", mv->print(true), std::to_string(mv->line));
    return false;
  } catch (std::exception &e) {
    SHLOG_WARNING("Eval error: {}", e.what());
    return false;
  } catch (MalString &str) {
    SHLOG_WARNING("Eval error: {}", str.c_str());
    return false;
  } catch (...) {
    return false;
  }
}

SHLISP_API __cdecl SHBool shLispAddToEnv(void *env, const char *str, SHVar input, bool reference) {
  auto penv = (malEnvPtr *)env;
  try {
    if (!reference) {
      auto var = malValuePtr(malSHVar::newCloned(input));
      (*penv)->set(str, var);
    } else {
      auto var = malValuePtr(new malSHVar(input, false));
      (*penv)->set(str, var);
    }
    return true;
  } catch (malEmptyInputException &) {
    return false;
  } catch (malValuePtr &mv) {
    SHLOG_WARNING("AddToEnv error: {} line: {}", mv->print(true), std::to_string(mv->line));
    return false;
  } catch (std::exception &e) {
    SHLOG_WARNING("AddToEnv error: {}", e.what());
    return false;
  } catch (MalString &str) {
    SHLOG_WARNING("AddToEnv error: {}", str.c_str());
    return false;
  } catch (...) {
    return false;
  }
}
}

void setupObserver(std::shared_ptr<Observer> &obs, const malEnvPtr &env) {
  obs = std::make_shared<Observer>();
  obs->_env = env;
  {
    shards::GetGlobals().Observers.emplace_back(obs);
  }
  {
    std::scoped_lock obsLock(observersMutex);
    observers[env.ptr()] = obs;
  }
}

namespace mal {
malValuePtr contextVar(const MalString &token) {
  SHVar tmp{};
  tmp.valueType = SHType::ContextVar;
  tmp.payload.stringValue = token.c_str();
  tmp.payload.stringLen = token.size();
  return malValuePtr(malSHVar::newCloned(tmp));
}
} // namespace mal
