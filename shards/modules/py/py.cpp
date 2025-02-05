#ifndef CB_PYTHON_HPP
#define CB_PYTHON_HPP

// must be on top
#ifndef __kernel_entry
#define __kernel_entry
#endif
#include <boost/process.hpp>

#include <shards/core/platform.hpp>
#include <shards/core/module.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

/*
 * This block will always work.. even if no python is present
 * We always try to dyn link and we provide always interface here
 * no headers required, no linking required
 * Python 3++ only, tested and developed with python 3.8
 */

namespace shards {
namespace Python {
inline bool hasLib(const char *lib_name) {
#if _WIN32
  HMODULE mod = GetModuleHandleA(lib_name);
  if (mod == 0) {
    mod = LoadLibraryA(lib_name);
    if (mod == 0)
      return false;
  }
  return true;
#elif defined(__linux__) || defined(__APPLE__)
  // has to be global or py modules would fail
  void *mod = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
  if (mod == 0)
    return false;
  return true;
#else
  return false;
#endif
}

inline void *dynLoad(const char *lib_name, const char *sym_name) {
  thread_local void *mod = nullptr;
#if _WIN32
  if (mod == nullptr) {
    mod = GetModuleHandleA(lib_name);
    if (mod == 0) {
      mod = LoadLibraryA(lib_name);
      if (mod == 0)
        return 0;
    }
  }

  return (void *)GetProcAddress(mod, sym_name);
#elif defined(__linux__) || defined(__APPLE__)
  // has to be global or py modules would fail
  if (mod == nullptr) {
    mod = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (mod == nullptr) {
      return nullptr;
    }
  }

  void *sym = dlsym(mod, sym_name);
  return sym;
#else
  return nullptr;
#endif
}

struct PyObject;
struct PyTypeObject;

typedef void(__cdecl *PyDealloc)(PyObject *p);

struct PyObject {
  ssize_t refcount;
  PyTypeObject *type;
};

struct PyObjectVarHead : public PyObject {
  ssize_t size;
};

struct PyTypeObject : public PyObjectVarHead {
  const char *name;
  ssize_t basicsize;
  ssize_t itemsize;
  PyDealloc dealloc;
};

typedef PyObject *(__cdecl *PyCFunc)(PyObject *, PyObject *);

struct PyMethodDef {
  const char *name;
  PyCFunc func;
  int flags;
  const char *doc;
};

using PyObj = std::shared_ptr<PyObject>;

struct PyThreadState {
  void *a;
  void *b;
  void *c;
  void *frame;
  //...
};

struct PyInterpreterState;

typedef int PyGILState_STATE;

struct Env {
  typedef void(__cdecl *Py_InitializeEx)(int handlers);
  typedef PyObject *(__cdecl *PyUnicode_DecodeFSDefault)(const char *str);
  typedef PyObject *(__cdecl *PyImport_Import)(PyObject *name);
  typedef PyObject *(__cdecl *PyObject_GetAttrString)(PyObject *obj, const char *str);
  typedef int(__cdecl *PyCallable_Check)(PyObject *obj);
  typedef PyObject *(__cdecl *PyObject_CallObject)(PyObject *obj, PyObject *args);

  typedef PyObject *(__cdecl *PyDict_New)();

  typedef int(__cdecl *PyErr_Occurred)();
  typedef void(__cdecl *PyErr_Print)();
  typedef PyObject *(__cdecl *PySys_GetObject)(const char *name);

  typedef void(__cdecl *PyList_Append)(PyObject *list, PyObject *item);
  typedef ssize_t(__cdecl *PyList_Size)(PyObject *list);
  typedef void *(__cdecl *PyList_SetItem)(PyObject *list, ssize_t idx, PyObject *item);
  typedef PyObject *(__cdecl *PyList_GetItem)(PyObject *list, ssize_t pos);

  typedef PyObject *(__cdecl *PyTuple_New)(ssize_t n);
  typedef ssize_t(__cdecl *PyTuple_Size)(PyObject *tup);
  typedef void *(__cdecl *PyTuple_SetItem)(PyObject *tuple, ssize_t idx, PyObject *item);
  typedef PyObject *(__cdecl *PyTuple_GetItem)(PyObject *tup, ssize_t pos);

  typedef int(__cdecl *PyType_IsSubtype)(PyTypeObject *a, PyTypeObject *b);
  typedef PyObject *(__cdecl *PyUnicode_AsUTF8String)(PyObject *unicode);
  typedef int(__cdecl *PyBytes_AsStringAndSize)(PyObject *obj, char **buffer, ssize_t *len);
  typedef PyObject *(__cdecl *Py_BuildValue)(const char *format, ...);
  typedef double(__cdecl *PyFloat_AsDouble)(PyObject *floatObj);
  typedef long long(__cdecl *PyLong_AsLongLong)(PyObject *longObj);
  typedef PyObject *(__cdecl *PyCFunction_NewEx)(PyMethodDef *mdef, PyObject *self, PyObject *mod);
  typedef void(__cdecl *PyCapsuleDtor)(PyObject *p);
  typedef PyObject *(__cdecl *PyCapsule_New)(void *ptr, const char *name, PyCapsuleDtor dtor);
  typedef void *(__cdecl *PyCapsule_GetPointer)(PyObject *cap, const char *name);
  typedef int(__cdecl *PyObject_SetAttrString)(PyObject *obj, const char *attrName, PyObject *item);
  typedef int(__cdecl *PyDict_SetItemString)(PyObject *obj, const char *attrName, PyObject *item);
  typedef int(__cdecl *PyArg_ParseTuple)(PyObject *args, const char *fmt, ...);
  typedef void(__cdecl *PyErr_Clear)();
  typedef int(__cdecl *PyObject_IsTrue)(PyObject *obj);
  typedef PyObject *(__cdecl *PyBool_FromLong)(long b);
  typedef void(__cdecl *PySys_SetArgvEx)(int, void *, int);
  typedef PyThreadState *(__cdecl *PyThreadState_Get)();
  typedef PyObject *(__cdecl *PyImport_AddModule)(const char *);
  typedef PyObject *(__cdecl *PyModule_GetDict)(PyObject *);
  typedef PyObject *(__cdecl *PyCode_NewEmpty)(const char *, const char *, int);
  typedef void *(__cdecl *PyFrame_New)(void *, void *, void *, void *);
  typedef PyThreadState *(__cdecl *PyThreadState_New)(PyInterpreterState *state);
  typedef PyThreadState *(__cdecl *PyThreadState_Swap)(PyThreadState *state);
  typedef PyThreadState *(__cdecl *Py_NewInterpreter)();
  typedef void(__cdecl *Py_EndInterpreter)(PyThreadState *state);

  typedef PyTypeObject *PyTuple_Type;
  typedef PyTypeObject *PyLong_Type;
  typedef PyTypeObject *PyUnicode_Type;
  typedef PyTypeObject *PyFloat_Type;
  typedef PyTypeObject *PyCapsule_Type;
  typedef PyTypeObject *PyObject_Type;
  typedef PyTypeObject *PyBool_Type;
  typedef PyTypeObject *PyList_Type;
  typedef PyTypeObject *PyBytes_Type;
  typedef PyObject *_Py_NoneStruct;

  typedef PyObject *(__cdecl *PyBytes_FromStringAndSize)(const char *v, ssize_t len);
  typedef PyObject *(__cdecl *PyUnicode_FromStringAndSize)(const char *u, ssize_t size);

  typedef void(__cdecl *PyEval_InitThreads)();
  typedef PyGILState_STATE(__cdecl *PyGILState_Ensure)();
  typedef void(__cdecl *PyGILState_Release)(PyGILState_STATE);

  typedef PyThreadState *(__cdecl *PyEval_SaveThread)();
  typedef void(__cdecl *PyEval_RestoreThread)(PyThreadState *);

  static inline PyUnicode_DecodeFSDefault _makeStr;
  static inline PyImport_Import _import;
  static inline PyObject_GetAttrString _getAttr;
  static inline PyCallable_Check _callable;
  static inline PyObject_CallObject _call;
  static inline PyTuple_New _tupleNew;
  static inline PyTuple_SetItem _tupleSetItem;
  static inline PyDict_New _dictNew;
  static inline PyErr_Occurred _errOccurred;
  static inline PyErr_Print _errPrint;
  static inline PyList_Append _listAppend;
  static inline PyTuple_Size _tupleSize;
  static inline PyTuple_GetItem _borrow_tupleGetItem;
  static inline PyType_IsSubtype _typeIsSubType;
  static inline PyUnicode_AsUTF8String _unicodeToString;
  static inline PyBytes_AsStringAndSize _bytesAsStringAndSize;
  static inline Py_BuildValue _buildValue;
  static inline PyFloat_AsDouble _asDouble;
  static inline PyLong_AsLongLong _asLong;
  static inline PyCapsule_New _newCapsule;
  static inline PyCapsule_GetPointer _capsuleGet;
  static inline PyObject_SetAttrString _setAttr;
  static inline PyCFunction_NewEx _newFunc;
  static inline PyArg_ParseTuple _argsParse;
  static inline PyErr_Clear _errClear;
  static inline PyObject_IsTrue _isTrue;
  static inline PyBool_FromLong _bool;
  static inline PyList_Size _listSize;
  static inline PyList_GetItem _borrow_listGetItem;
  static inline PyList_SetItem _listSetItem;
  static inline PyThreadState_Get _tsGet;
  static inline PyImport_AddModule _borrow_addModule;
  static inline PyModule_GetDict _borrow_modGetDict;
  static inline PyCode_NewEmpty _pyCodeNew;
  static inline PyFrame_New _frameNew;
  static inline PyDict_SetItemString _setDictItem;
  static inline PyThreadState_Swap _swapState;
  static inline Py_NewInterpreter _newInterpreter;
  static inline PySys_GetObject _sysGetObj;
  static inline Py_EndInterpreter _endInterpreter;

  static inline PyUnicode_Type _unicode_type;
  static inline PyTuple_Type _tuple_type;
  static inline PyFloat_Type _float_type;
  static inline PyLong_Type _long_type;
  static inline PyCapsule_Type _capsule_type;
  static inline PyBool_Type _bool_type;
  static inline PyList_Type _list_type;
  static inline PyBytes_Type _bytes_type;

  static inline _Py_NoneStruct _py_none;

  static inline PyBytes_FromStringAndSize _bytesFromStringAndSize;
  static inline PyUnicode_FromStringAndSize _unicodeFromStringAndSize;

  static inline PyEval_InitThreads _initThreads;
  static inline PyGILState_Ensure _gilEnsure;
  static inline PyGILState_Release _gilRelease;

  static inline PyEval_SaveThread _saveThread;
  static inline PyEval_RestoreThread _restoreThread;

  static PyObject *__cdecl methodPause(PyObject *self, PyObject *args) {
    auto ctxObj = make_pyshared(_getAttr(self, "__shcontext__"));
    if (!isCapsule(ctxObj)) {
      SHLOG_ERROR("internal error, __shcontext__ is not a capsule!");
      throw SHException("pause from python failed!");
    }

    auto ctx = getPtr(ctxObj);
    if (!ctx) {
      SHLOG_ERROR("internal error, __shcontext__ was null!");
      throw SHException("pause from python failed!");
    }

    // find a double arg
    double time = 0.0;
    if (_argsParse(args, "d", &time) == -1) {
      SHInt longtime = 0;
      if (_argsParse(args, "L", &longtime) != -1) {
        time = double(longtime);
      }
    }

    auto state = _saveThread();

    suspend((SHContext *)ctx, time);

    _restoreThread(state);

    _py_none->refcount++;
    return _py_none;
  }

  static inline PyMethodDef pauseMethod{"pause", &methodPause, (1 << 0), nullptr};

  static PyObj make_pyshared(PyObject *p) {
    std::shared_ptr<PyObject> res(p, [](auto p) {
      if (!p)
        return;

      p->refcount--;

      if (p->refcount == 0) {
        SHLOG_TRACE("PyObj::Dealloc");
        if (p->type)
          p->type->dealloc(p);
      }
    });
    return res;
  }

  static PyObj make_pyborrow(PyObject *p) {
    std::shared_ptr<PyObject> res(p, [](auto p) {});
    return res;
  }

  static bool ensure(void *ptr, const char *name) {
    if (!ptr) {
      SHLOG_ERROR("Failed to initialize python environment, could not find procedure: {}", name);
      return false;
    }
    return true;
  }

  static inline std::vector<std::string> Path;

  static void init() {
    try {
      // let's hack and find python paths...
      boost::process::ipstream opipe;
      boost::process::child cmd("python -c \"import sys; print(sys.path)\"", boost::process::std_out > opipe);
      cmd.join();
      if (cmd.exit_code() == 0) {
        std::stringstream ss;
        auto s = opipe.rdbuf();
        ss << s;
        auto paths_str = ss.str();
        std::replace(paths_str.begin(), paths_str.end(), '\'', '\"');
        auto jpaths = nlohmann::json::parse(paths_str);
        std::vector<std::string> paths = jpaths;
        for (auto &path : paths) {
          SHLOG_DEBUG("PY PATH: {}", path);
        }
        Path = paths;
      }
    } catch (const std::exception &ex) {
      SHLOG_ERROR("Error while probing python: {}", ex.what());
    }

    static const auto version_patterns = {"3.13", "313", "3.12", "312", "3.11", "311", "3.10", "310",
                                          "3.9",  "39",  "3.8",  "38",  "3.7",  "37",  "3",    ""};
    std::vector<std::string> candidates;
    // prefer this order
#ifdef _WIN32
    for (auto &pattern : version_patterns) {
      candidates.emplace_back(std::string("python") + pattern + ".dll");
    }
    for (auto &pattern : version_patterns) {
      candidates.emplace_back(std::string("libpython") + pattern + ".dll");
      candidates.emplace_back(std::string("libpython") + pattern + "m.dll");
    }
#elif defined(__APPLE__)
    for (auto &pattern : version_patterns) {
      candidates.emplace_back(std::string("libpython") + pattern + ".dylib");
      candidates.emplace_back(std::string("libpython") + pattern + "m.dylib");
    }
#else
    for (auto &pattern : version_patterns) {
      candidates.emplace_back(std::string("libpython") + pattern + ".so");
      candidates.emplace_back(std::string("libpython") + pattern + ".so.1");
      candidates.emplace_back(std::string("libpython") + pattern + "m.so");
      candidates.emplace_back(std::string("libpython") + pattern + "m.so.1");
    }
#endif

    SHLOG_TRACE("Probing python versions: {}", candidates.size());

    auto pos = std::find_if(std::begin(candidates), std::end(candidates), [](auto &&str) {
      SHLOG_TRACE("Probing: {}", str);
      return hasLib(str.c_str());
    });
    if (pos != std::end(candidates)) {
      auto idx = std::distance(std::begin(candidates), pos);
      auto dll = candidates[idx].c_str();

      SHLOG_TRACE("Selected: {}", dll);

      auto init = (Py_InitializeEx)dynLoad(dll, "Py_InitializeEx");
      if (!ensure((void *)init, "Py_InitializeEx"))
        return;
      init(0);

      SHLOG_TRACE("PyInit called fine!");

#define DLIMPORT(_proc_, _name_)          \
  _proc_ = (_name_)dynLoad(dll, #_name_); \
  if (!ensure((void *)_proc_, #_name_))   \
  return

      DLIMPORT(_makeStr, PyUnicode_DecodeFSDefault);
      DLIMPORT(_import, PyImport_Import);
      DLIMPORT(_getAttr, PyObject_GetAttrString);
      DLIMPORT(_callable, PyCallable_Check);
      DLIMPORT(_call, PyObject_CallObject);
      DLIMPORT(_tupleNew, PyTuple_New);
      DLIMPORT(_tupleSetItem, PyTuple_SetItem);
      DLIMPORT(_dictNew, PyDict_New);
      DLIMPORT(_errOccurred, PyErr_Occurred);
      DLIMPORT(_errPrint, PyErr_Print);
      DLIMPORT(_listAppend, PyList_Append);
      DLIMPORT(_tupleSize, PyTuple_Size);
      DLIMPORT(_borrow_tupleGetItem, PyTuple_GetItem);
      DLIMPORT(_typeIsSubType, PyType_IsSubtype);
      DLIMPORT(_unicodeToString, PyUnicode_AsUTF8String);
      DLIMPORT(_bytesAsStringAndSize, PyBytes_AsStringAndSize);
      DLIMPORT(_buildValue, Py_BuildValue);
      DLIMPORT(_asDouble, PyFloat_AsDouble);
      DLIMPORT(_asLong, PyLong_AsLongLong);
      DLIMPORT(_newCapsule, PyCapsule_New);
      DLIMPORT(_capsuleGet, PyCapsule_GetPointer);
      DLIMPORT(_setAttr, PyObject_SetAttrString);
      DLIMPORT(_newFunc, PyCFunction_NewEx);
      DLIMPORT(_argsParse, PyArg_ParseTuple);
      DLIMPORT(_errClear, PyErr_Clear);
      DLIMPORT(_isTrue, PyObject_IsTrue);
      DLIMPORT(_bool, PyBool_FromLong);
      DLIMPORT(_listSize, PyList_Size);
      DLIMPORT(_listSetItem, PyList_SetItem);
      DLIMPORT(_borrow_listGetItem, PyList_GetItem);
      DLIMPORT(_tsGet, PyThreadState_Get);
      DLIMPORT(_borrow_addModule, PyImport_AddModule);
      DLIMPORT(_borrow_modGetDict, PyModule_GetDict);
      DLIMPORT(_pyCodeNew, PyCode_NewEmpty);
      DLIMPORT(_frameNew, PyFrame_New);
      DLIMPORT(_setDictItem, PyDict_SetItemString);
      DLIMPORT(_swapState, PyThreadState_Swap);
      DLIMPORT(_newInterpreter, Py_NewInterpreter);
      DLIMPORT(_sysGetObj, PySys_GetObject);
      DLIMPORT(_endInterpreter, Py_EndInterpreter);

      DLIMPORT(_unicode_type, PyUnicode_Type);
      DLIMPORT(_tuple_type, PyTuple_Type);
      DLIMPORT(_float_type, PyFloat_Type);
      DLIMPORT(_long_type, PyLong_Type);
      DLIMPORT(_capsule_type, PyCapsule_Type);
      DLIMPORT(_list_type, PyList_Type);
      DLIMPORT(_bool_type, PyBool_Type);
      DLIMPORT(_bytes_type, PyBytes_Type);

      DLIMPORT(_py_none, _Py_NoneStruct);

      DLIMPORT(_bytesFromStringAndSize, PyBytes_FromStringAndSize);
      DLIMPORT(_unicodeFromStringAndSize, PyUnicode_FromStringAndSize);

      DLIMPORT(_initThreads, PyEval_InitThreads);
      DLIMPORT(_gilEnsure, PyGILState_Ensure);
      DLIMPORT(_gilRelease, PyGILState_Release);

      DLIMPORT(_saveThread, PyEval_SaveThread);
      DLIMPORT(_restoreThread, PyEval_RestoreThread);

      _initThreads();
      _savedThreadState = _saveThread();

      SHLOG_TRACE("Python symbols loaded");

      _ok = true;

      SHLOG_INFO("Python found, Py blocks will work!");
    } else {
      SHLOG_INFO("Python not found, Py blocks won't work.");
    }
  }

  static PyObj string(const char *name) { return make_pyshared(_makeStr(name)); }

  static PyObj import(const char *name) {
    auto pyname = string(name);
    return make_pyshared(_import(pyname.get()));
  }

  static PyObj getAttr(const PyObj &obj, const char *attr_name) { return make_pyshared(_getAttr(obj.get(), attr_name)); }

  static void setAttr(const PyObj &obj, const char *attr_name, const PyObj &item) {
    if (_setAttr(obj.get(), attr_name, item.get()) == -1) {
      printErrors();
      throw SHException("Failed to set attribute (Py)");
    }
  }

  static bool isCallable(const PyObj &obj) { return obj.get() && _callable(obj.get()); }

  template <typename... Ts> static PyObj call(const PyObj &obj, Ts... vargs) {
    constexpr std::size_t n = sizeof...(Ts);
    std::array<PyObject *, n> args{vargs...};
    auto tuple = make_pyshared(_tupleNew(n));
    for (size_t i = 0; i < n; i++) {
      _tupleSetItem(tuple.get(), ssize_t(i), args[i]);
    }
    return make_pyshared(_call(obj.get(), tuple.get()));
  }

  static PyObj call(const PyObj &obj) { return make_pyshared(_call(obj.get(), nullptr)); }

  static PyObj dict() { return make_pyshared(_dictNew()); }

  static bool ok() { return _ok; }

  static PyObject *var2Py(const SHVar &var) {
    switch (var.valueType) {
    case SHType::Int: {
      PyObject *result = _buildValue("L", var.payload.intValue);
      assert(result != nullptr);
      return result;
    }
    case SHType::Int2: {
      PyObject *result = _buildValue("(LL)", var.payload.int2Value[0], var.payload.int2Value[1]);
      assert(result != nullptr);
      return result;
    }
    case SHType::Int3: {
      PyObject *result = _buildValue("(lll)", var.payload.int3Value[0], var.payload.int3Value[1], var.payload.int3Value[2]);
      assert(result != nullptr);
      return result;
    }
    case SHType::Int4: {
      PyObject *result = _buildValue("(llll)", var.payload.int4Value[0], var.payload.int4Value[1], var.payload.int4Value[2],
                                     var.payload.int4Value[3]);
      assert(result != nullptr);
      return result;
    }
    case SHType::Float: {
      PyObject *result = _buildValue("d", var.payload.floatValue);
      assert(result != nullptr);
      return result;
    }
    case SHType::Float2: {
      PyObject *result = _buildValue("(dd)", var.payload.float2Value[0], var.payload.float2Value[1]);
      assert(result != nullptr);
      return result;
    }
    case SHType::Float3: {
      PyObject *result = _buildValue("(fff)", var.payload.float3Value[0], var.payload.float3Value[1], var.payload.float3Value[2]);
      assert(result != nullptr);
      return result;
    }
    case SHType::Float4: {
      PyObject *result = _buildValue("(dddd)", var.payload.float4Value[0], var.payload.float4Value[1], var.payload.float4Value[2],
                                     var.payload.float4Value[3]);
      assert(result != nullptr);
      return result;
    }
    case SHType::String: {
      auto str = SHSTRVIEW(var);
      PyObject *result = _unicodeFromStringAndSize(str.data(), str.size());
      if (result == nullptr) {
        Env::printErrors();
        SHLOG_ERROR("Failed to create Python unicode object. Size: {}", str.size());
        throw SHException("Failed to convert SHVar String to Python unicode object");
      }
      return result;
    }
    case SHType::Bool: {
      return _bool(long(var.payload.boolValue));
    }
    case SHType::None: {
      _py_none->refcount++;
      return _py_none;
    }
    case SHType::Bytes: {
      PyObject *result = _bytesFromStringAndSize((const char *)var.payload.bytesValue, var.payload.bytesSize);
      if (result == nullptr) {
        Env::printErrors();
        SHLOG_ERROR("Failed to create Python bytes object. Size: {}", var.payload.bytesSize);
        if (var.payload.bytesValue == nullptr) {
          SHLOG_ERROR("Bytes value pointer is null");
        }
        throw SHException("Failed to convert SHVar Bytes to Python bytes object");
      }
      return result;
    }
    default:
      SHLOG_ERROR("Unsupported type {}", type2Name(var.valueType));
      throw SHException("Failed to convert SHVar into PyObject, type not supported!");
    }
  }

  static std::tuple<SHVar, PyObj> py2Var(const PyObj &obj) {
    SHVar res{};
    if (isLong(obj)) {
      res.valueType = SHType::Int;
      res.payload.intValue = _asLong(obj.get());
      return {res, PyObj()};
    } else if (isFloat(obj)) {
      res.valueType = SHType::Float;
      res.payload.floatValue = _asDouble(obj.get());
      return {res, PyObj()};
    } else if (isBool(obj)) {
      res.valueType = SHType::Bool;
      res.payload.boolValue = _isTrue(obj.get()) == 1;
      return {res, PyObj()};
    } else if (isString(obj)) {
      res.valueType = SHType::String;
      auto tstr = toStringView(obj);
      auto str = std::get<0>(tstr);
      res.payload.stringValue = str.data();
      res.payload.stringLen = str.size();
      return {res, std::get<1>(tstr)};
    } else if (isBytes(obj)) {
      res.valueType = SHType::Bytes;
      auto bytes = toBytesView(obj);
      auto str = std::get<0>(bytes);
      res.payload.bytesValue = (uint8_t *)str.data();
      res.payload.bytesSize = str.size();
      return {res, std::get<1>(bytes)};
    } else if (isTuple(obj)) {
      auto tupSize = _tupleSize(obj.get());
      if (tupSize == 0) {
        // None
        return {res, PyObj()};
      } else {
        auto first = _borrow_tupleGetItem(obj.get(), 0);
        assert(first);
        if (isLong(first)) {
          switch (tupSize) {
          case 1:
            res.valueType = SHType::Int;
            res.payload.intValue = _asLong(obj.get());
            return {res, PyObj()};
          case 2: {
            res.valueType = SHType::Int2;
            res.payload.int2Value[0] = _asLong(first);
            auto second = _borrow_tupleGetItem(obj.get(), 1);
            res.payload.int2Value[1] = _asLong(second);
            return {res, PyObj()};
          }
          case 3: {
            res.valueType = SHType::Int3;
            res.payload.int3Value[0] = _asLong(first);
            for (int i = 1; i < 3; i++) {
              auto next = _borrow_tupleGetItem(obj.get(), i);
              res.payload.int3Value[i] = _asLong(next);
            }
            return {res, PyObj()};
          }
          case 4: {
            res.valueType = SHType::Int4;
            res.payload.int4Value[0] = _asLong(first);
            for (int i = 1; i < 4; i++) {
              auto next = _borrow_tupleGetItem(obj.get(), i);
              res.payload.int4Value[i] = _asLong(next);
            }
            return {res, PyObj()};
          }
          default:
            throw SHException("Failed to convert python value to SHVar value, invalid tuple size!");
          }
        } else if (isFloat(first)) {
          switch (tupSize) {
          case 1:
            res.valueType = SHType::Float;
            res.payload.floatValue = _asDouble(obj.get());
            return {res, PyObj()};
          case 2: {
            res.valueType = SHType::Float2;
            res.payload.float2Value[0] = _asDouble(first);
            auto second = _borrow_tupleGetItem(obj.get(), 1);
            res.payload.float2Value[1] = _asDouble(second);
            return {res, PyObj()};
          }
          case 3: {
            res.valueType = SHType::Float3;
            res.payload.float3Value[0] = _asDouble(first);
            for (int i = 1; i < 3; i++) {
              auto next = _borrow_tupleGetItem(obj.get(), i);
              res.payload.float3Value[i] = _asDouble(next);
            }
            return {res, PyObj()};
          }
          case 4: {
            res.valueType = SHType::Float4;
            res.payload.float4Value[0] = _asDouble(first);
            for (int i = 1; i < 4; i++) {
              auto next = _borrow_tupleGetItem(obj.get(), i);
              res.payload.float4Value[i] = _asDouble(next);
            }
            return {res, PyObj()};
          }
          default:
            throw SHException("Failed to convert python value to SHVar value, invalid tuple size!");
          }
        } else {
          throw SHException("Failed to convert python value to SHVar value, invalid tuple!");
        }
      }
    } else {
      throw SHException("Failed to convert python value to SHVar value!");
    }
  }

  static void printErrors() {
    if (_errOccurred()) {
      _errPrint();
      _errClear();
    }
  }

  static void clearError() { _errClear(); }

  static bool isTuple(const PyObject *obj) { return obj->type == _tuple_type || _typeIsSubType(obj->type, _tuple_type); }

  static bool isTuple(const PyObj &obj) { return obj.get() && isTuple(obj.get()); }

  static bool isList(const PyObject *obj) { return obj->type == _list_type || _typeIsSubType(obj->type, _list_type); }

  static bool isList(const PyObj &obj) { return obj.get() && isList(obj.get()); }

  static bool isString(const PyObject *obj) { return obj->type == _unicode_type || _typeIsSubType(obj->type, _unicode_type); }

  static bool isString(const PyObj &obj) { return obj.get() && isString(obj.get()); }

  static bool isLong(const PyObject *obj) { return obj->type == _long_type || _typeIsSubType(obj->type, _long_type); }

  static bool isLong(const PyObj &obj) { return obj.get() && isLong(obj.get()); }

  static bool isFloat(const PyObject *obj) { return obj->type == _float_type || _typeIsSubType(obj->type, _float_type); }

  static bool isFloat(const PyObj &obj) { return obj.get() && isFloat(obj.get()); }

  static bool isBool(const PyObject *obj) { return obj->type == _bool_type || _typeIsSubType(obj->type, _bool_type); }

  static bool isBool(const PyObj &obj) { return obj.get() && isBool(obj.get()); }

  static bool isCapsule(const PyObject *obj) { return obj->type == _capsule_type || _typeIsSubType(obj->type, _capsule_type); }

  static bool isCapsule(const PyObj &obj) { return obj.get() && isCapsule(obj.get()); }

  static bool isNone(const PyObject *obj) { return obj == _py_none; }

  static bool isNone(const PyObj &obj) { return obj.get() && isNone(obj.get()); }

  static bool isBytes(const PyObject *obj) { return obj->type == _bytes_type || _typeIsSubType(obj->type, _bytes_type); }

  static bool isBytes(const PyObj &obj) { return obj.get() && isBytes(obj.get()); }

  static ssize_t tupleSize(const PyObj &obj) { return _tupleSize(obj.get()); }

  static PyObj tupleGetItem(const PyObj &tup, ssize_t idx) { return make_pyborrow(_borrow_tupleGetItem(tup.get(), idx)); }

  static ssize_t listSize(const PyObj &obj) { return _listSize(obj.get()); }

  static PyObj listGetItem(const PyObj &l, ssize_t idx) { return make_pyborrow(_borrow_listGetItem(l.get(), idx)); }

  static std::tuple<std::string_view, PyObj> toStringView(PyObject *obj) {
    char *str;
    ssize_t len;
    auto utf = make_pyshared(_unicodeToString(obj));
    auto res = _bytesAsStringAndSize(utf.get(), &str, &len);
    if (res == -1) {
      printErrors();
      throw SHException("String conversion failed!");
    }
    return {std::string_view(str, len), utf};
  }

  static std::tuple<std::string_view, PyObj> toStringView(const PyObj &obj) { return toStringView(obj.get()); }

  static std::tuple<std::span<const uint8_t>, PyObj> toBytesView(PyObject *obj) {
    char *data;
    ssize_t len;
    auto res = _bytesAsStringAndSize(obj, &data, &len);
    if (res == -1) {
      printErrors();
      throw SHException("Bytes conversion failed!");
    }
    return {std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(data), len), make_pyborrow(obj)};
  }

  static std::tuple<std::span<const uint8_t>, PyObj> toBytesView(const PyObj &obj) { return toBytesView(obj.get()); }

  static PyObj capsule(void *ptr) { return make_pyshared(_newCapsule(ptr, nullptr, nullptr)); }

  static void *getPtr(const PyObj &obj) { return _capsuleGet(obj.get(), nullptr); }

  static PyObj none() {
    _py_none->refcount++;
    return make_pyborrow(_py_none);
  }

  static PyObj func(PyMethodDef &def, const PyObj &self) { return make_pyshared(_newFunc(&def, self.get(), nullptr)); }

  static void setDictItem(const PyObj &dict, const char *name, const PyObj &item) { _setDictItem(dict.get(), name, item.get()); }

  static PyObject *intVal(int i) { return _buildValue("i", i); }

  using ToTypesFailed = SHException;

  static SHType toSHType(const std::string_view &str) {
    if (str == "Int") {
      return SHType::Int;
    } else if (str == "Int2") {
      return SHType::Int2;
    } else if (str == "Int3") {
      return SHType::Int3;
    } else if (str == "Int4") {
      return SHType::Int4;
    } else if (str == "Float") {
      return SHType::Float;
    } else if (str == "Float2") {
      return SHType::Float2;
    } else if (str == "Float3") {
      return SHType::Float3;
    } else if (str == "Float4") {
      return SHType::Float4;
    } else if (str == "String") {
      return SHType::String;
    } else if (str == "Any") {
      return SHType::Any;
    } else if (str == "None") {
      return SHType::None;
    } else if (str == "Bool") {
      return SHType::Bool;
    } else if (str == "Bytes") {
      return SHType::Bytes;
    } else {
      throw ToTypesFailed("Unsupported toSHType type.");
    }
  }

  static void extractTypes(const PyObj &obj, Types &out_types, std::list<SHTypeInfo> &innerInfos) {
    innerInfos.clear();
    std::vector<SHTypeInfo> types;
    if (Env::isList(obj)) {
      auto size = Env::listSize(obj);
      for (ssize_t i = 0; i < size; i++) {
        auto item = Env::listGetItem(obj, i);
        if (Env::isString(item)) {
          auto tstr = Env::toStringView(item);
          auto &str = std::get<0>(tstr);
          if (str.size() > 3 && str.substr(str.size() - 3, 3) == "Seq") {
            auto &inner = innerInfos.emplace_back(SHTypeInfo{Env::toSHType(str.substr(0, str.size() - 3))});
            auto seqType = types.emplace_back(SHTypeInfo{SHType::Seq});
            seqType.seqTypes = {&inner, 1, 0};
          } else {
            types.emplace_back(SHTypeInfo{Env::toSHType(str)});
          }
        } else {
          printErrors();
          throw ToTypesFailed("Failed to transform python object to Types within tuple.");
        }
      }
    } else {
      printErrors();
      throw ToTypesFailed("Failed to transform python object to Types, object is not a list!");
    }
    out_types = types;
  }

  static PyObject *incRefGet(const PyObj &obj) {
    auto res = obj.get();
    res->refcount++;
    return res;
  }

  static PyThreadState *saveThread() { return _saveThread(); }

  static void restoreThread(PyThreadState *state) { _restoreThread(state); }

  static void saveThreadState() { _savedThreadState = _saveThread(); }

  static void restoreThreadState() { _restoreThread(_savedThreadState); }

private:
  static inline bool _ok{false};
  static inline PyThreadState *_savedThreadState{nullptr};
};

struct Context {
  Context() {
    if (!Env::ok()) {
      return;
    }

    _gstate = Env::_gilEnsure();
  }

  ~Context() {
    if (!Env::ok()) {
      return;
    }

    Env::_gilRelease(_gstate);
  }

private:
  PyGILState_STATE _gstate;
};

struct Py {
  Py() {
    // Try lazy init
    if (!Env::ok()) {
      Env::init();
    }
  }

  ~Py() {
    Context ctx;

    // Reset all PyObjs
    _self.reset();
    _module.reset();
    _inputTypes.reset();
    _outputTypes.reset();
    _activate.reset();
    _parameters.reset();
    _setParam.reset();
    _getParam.reset();
    _compose.reset();
    _currentResult.reset();
    _pyParamResult.reset();

    // Clear vector of PyObjs
    _seqCacheObjs.clear();
  }

  Parameters params{{"Module",
                     SHCCSTR("The module name to load (must be in the script "
                             "path, .py extension added internally!)"),
                     {CoreInfo::StringType}}};

  SHParametersInfo parameters() {
    Context ctx;

    // clear cache first
    _paramNames.clear();
    _paramHelps.clear();

    if (Env::isCallable(_parameters)) {
      std::vector<ParameterInfo> otherParams;
      auto pyParams = Env::call(_parameters, Env::incRefGet(_self));
      if (Env::isList(pyParams)) {
        auto psize = Env::listSize(pyParams);
        for (ssize_t i = 0; i < psize; i++) {
          auto pyParam = Env::listGetItem(pyParams, i);
          if (!Env::isTuple(pyParam)) {
            throw SHException("Malformed python block parameters, list of tuple expected");
          }
          auto tupSize = Env::tupleSize(pyParam);
          if (tupSize == 2) {
            // has no help
            auto pyname = Env::tupleGetItem(pyParam, 0);
            auto tnameview = Env::toStringView(pyname);
            auto nameview = std::get<0>(tnameview);
            auto &name = _paramNames.emplace_back(nameview);
            auto pytypes = Env::tupleGetItem(pyParam, 1);
            Types types;
            Env::extractTypes(pytypes, types, _paramsInners);
            otherParams.emplace_back(name.c_str(), SHOptionalString{}, types);
          } else if (tupSize == 3) {
            // has help
            auto pyname = Env::tupleGetItem(pyParam, 0);
            auto tnameview = Env::toStringView(pyname);
            auto nameview = std::get<0>(tnameview);
            auto &name = _paramNames.emplace_back(nameview);
            auto pyhelp = Env::tupleGetItem(pyParam, 0);
            auto thelpview = Env::toStringView(pyhelp);
            auto helpview = std::get<0>(thelpview);
            auto &help = _paramHelps.emplace_back(helpview);
            auto pytypes = Env::tupleGetItem(pyParam, 2);
            Types types;
            Env::extractTypes(pytypes, types, _paramsInners);
            otherParams.emplace_back(name.c_str(), SHOptionalString{help.c_str()}, types);
          } else {
            throw SHException("Malformed python block parameters, list of tuple (name, help, "
                              "types) or (name, types) expected");
          }
        }
      } else {
        Env::printErrors();
        throw SHException("Failed to fetch python block parameters");
      }
      _params = Parameters(params, otherParams);
    } else {
      _params = Parameters(params);
    }
    return _params;
  }

  void setParam(int index, const SHVar &value) {
    Context ctx;

    if (index == 0) {
      // Handle here
      _scriptName = SHSTRVIEW(value);
      if (_scriptName != "")
        reloadScript();
    } else {
      if (!Env::isCallable(_setParam)) {
        SHLOG_ERROR("Script: {} cannot call setParam, is it missing?", _scriptName);
        throw SHException("Python block setParam is not callable!");
      }
      Env::call(_setParam, Env::incRefGet(_self), Env::intVal(index - 1), Env::var2Py(value));
    }
  }

  SHVar getParam(int index) {
    Context ctx;

    if (index == 0) {
      return Var(_scriptName);
    } else {
      _pyParamResult = Env::none();

      if (!Env::isCallable(_getParam)) {
        SHLOG_ERROR("Script: {} cannot call getParam, is it missing?", _scriptName);
        throw SHException("Python block getParam is not callable!");
      }

      auto res = Env::call(_getParam, Env::incRefGet(_self), Env::intVal(index - 1));

      auto cbres = Env::py2Var(res);
      _pyParamResult = std::get<1>(cbres);
      return std::get<0>(cbres);
    }
  }

  void reloadScript() {
    if (!Env::ok()) {
      SHLOG_ERROR("Script: {} cannot be loaded, no python support.", _scriptName);
      throw SHException("Failed to load python script!");
    }

    auto path = Env::_sysGetObj("path");
    // fix sys.path
    for (auto &item : Env::Path) {
      auto pyp = Env::string(item.c_str());
      Env::_listAppend(path, pyp.get());
    }

    namespace fs = std::filesystem;

    auto scriptName = _scriptName;
    std::replace(scriptName.begin(), scriptName.end(), '.', '/');
    fs::path scriptPath(scriptName);

    if (GetGlobals().RootPath.size() > 0) {
      fs::path cbpath(GetGlobals().RootPath);
      auto absRoot = fs::absolute(cbpath / scriptPath.parent_path());
      absRoot.make_preferred();
      auto pyAbsRoot = Env::string(absRoot.string().c_str());
      Env::_listAppend(path, pyAbsRoot.get());
    }

    auto absRoot = fs::absolute(fs::current_path() / scriptPath.parent_path());
    absRoot.make_preferred();
    auto pyAbsRoot = Env::string(absRoot.string().c_str());
    Env::_listAppend(path, pyAbsRoot.get());

    auto moduleName = scriptPath.stem().string();
    _module = Env::import(moduleName.c_str());
    if (!_module.get()) {
      SHLOG_ERROR("Script: {} failed to load!", _scriptName);
      Env::printErrors();
      throw SHException("Failed to load python script!");
    }

    _inputTypes = Env::getAttr(_module, "inputTypes");
    if (!Env::isCallable(_inputTypes)) {
      SHLOG_ERROR("Script: {} has no callable inputTypes.", _scriptName);
      throw SHException("Failed to reload python script!");
    }

    _outputTypes = Env::getAttr(_module, "outputTypes");
    if (!Env::isCallable(_outputTypes)) {
      SHLOG_ERROR("Script: {} has no callable outputTypes.", _scriptName);
      throw SHException("Failed to reload python script!");
    }

    _activate = Env::getAttr(_module, "activate");
    if (!Env::isCallable(_activate)) {
      SHLOG_ERROR("Script: {} has no callable activate.", _scriptName);
      throw SHException("Failed to reload python script!");
    }

    // Optional stuff
    _parameters = Env::getAttr(_module, "parameters");
    _setParam = Env::getAttr(_module, "setParam");
    _getParam = Env::getAttr(_module, "getParam");
    _compose = Env::getAttr(_module, "compose");

    auto setup = Env::getAttr(_module, "setup");

    if (Env::isCallable(setup)) {
      _self = Env::call(setup);
      if (Env::isNone(_self)) {
        SHLOG_ERROR("Script: {} setup must return a valid object.", _scriptName);
        throw SHException("Failed to reload python script!");
      }

      auto pause1 = Env::func(Env::pauseMethod, _self);
      Env::setAttr(_self, "pause", pause1);
    }

    Env::clearError();
  }

  SHTypesInfo inputTypes() {
    Context ctx;

    if (Env::isCallable(_inputTypes)) {
      PyObj pytype;
      if (_self.get())
        pytype = Env::call(_inputTypes, Env::incRefGet(_self));
      else
        pytype = Env::call(_inputTypes);
      try {
        Env::extractTypes(pytype, _inputTypesStorage, _inputInners);
      } catch (Env::ToTypesFailed &ex) {
        SHLOG_ERROR(ex.what());
        SHLOG_ERROR("Script: {} inputTypes method should return a tuple of strings "
                    "or a string.",
                    _scriptName);
        throw SHException("Failed call inputTypes on python script!");
      }
    }
    return _inputTypesStorage;
  }

  SHTypesInfo outputTypes() {
    Context ctx;

    if (Env::isCallable(_outputTypes)) {
      PyObj pytype;
      if (_self.get())
        pytype = Env::call(_outputTypes, Env::incRefGet(_self));
      else
        pytype = Env::call(_outputTypes);
      try {
        Env::extractTypes(pytype, _outputTypesStorage, _outputInners);
      } catch (Env::ToTypesFailed &ex) {
        SHLOG_ERROR(ex.what());
        SHLOG_ERROR("Script: {} outputTypes method should return a tuple of strings "
                    "or a string.",
                    _scriptName);
        throw SHException("Failed call outputTypes on python script!");
      }
    }
    return _outputTypesStorage;
  }

  void cleanup() {
    Context ctx;

    _seqCache.clear();
    _seqCacheObjs.clear();
    _currentResult = Env::none();
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    Context ctx;

    PyObj res;
    if (_self.get()) {
      auto pyctx = Env::capsule(context);
      Env::setAttr(_self, "__shcontext__", pyctx);
      SHLOG_TRACE("Self refcount: {}", int(_self->refcount));
      res = Env::call(_activate, Env::incRefGet(_self), Env::var2Py(input));
    } else {
      res = Env::call(_activate, Env::var2Py(input));
    }

    if (!res.get()) {
      Env::printErrors();
      throw SHException("Python script activation failed.");
    }

    if (Env::isList(res)) {
      _currentResult = res;
      ssize_t size = Env::listSize(res);
      _seqCache.resize(size);
      _seqCacheObjs.resize(size);
      for (ssize_t i = 0; i < size; i++) {
        auto cbres = Env::py2Var(Env::listGetItem(res, i));
        _seqCache[i] = std::get<0>(cbres);
        _seqCacheObjs[i] = std::get<1>(cbres);
      }
      return Var(_seqCache);
    } else {
      auto cbres = Env::py2Var(res);
      _currentResult = std::get<1>(cbres);
      return std::get<0>(cbres);
    }
  }

  static const SHTable *properties() {
    static TableVar experimental{{Var("dynamic-parameters"), Var(true)}};
    return &experimental.payload.tableValue;
  }

private:
  Types _inputTypesStorage;
  std::list<SHTypeInfo> _inputInners;
  Types _outputTypesStorage;
  std::list<SHTypeInfo> _outputInners;
  Parameters _params;
  std::list<SHTypeInfo> _paramsInners;
  std::list<std::string> _paramNames;
  std::list<std::string> _paramHelps;

  PyObj _self;
  PyObj _module;

  // Needed defs
  PyObj _inputTypes;
  PyObj _outputTypes;
  PyObj _activate;

  // Optional defs
  PyObj _parameters;
  PyObj _setParam;
  PyObj _getParam;
  PyObj _compose; // TODO

  PyObj _currentResult;
  PyObj _pyParamResult;
  std::vector<SHVar> _seqCache;
  std::vector<PyObj> _seqCacheObjs;

  std::string _scriptName;
};

SHARDS_REGISTER_FN(py) { REGISTER_SHARD("Py", Py); }
} // namespace Python
} // namespace shards

#endif /* CB_PYTHON_HPP */