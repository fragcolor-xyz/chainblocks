#ifndef CB_PYTHON_HPP
#define CB_PYTHON_HPP

#include "shared.hpp"
#include <filesystem>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

/*
 * This block will always work.. even if no python is present
 * We always try to dyn link and we provide always interface here
 * no headers required, no linking required
 * Python 3++ only, tested and developed with python 3.8
 */

namespace chainblocks {
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
  void *mod = dlopen(lib_name, RTLD_LOCAL | RTLD_NODELETE);
  if (mod == 0)
    return false;
  dlclose(mod);
  return true;
#else
  return false;
#endif
}

inline void *dynLoad(const char *lib_name, const char *sym_name) {
#if _WIN32
  HMODULE mod = GetModuleHandleA(lib_name);
  if (mod == 0) {
    mod = LoadLibraryA(lib_name);
    if (mod == 0)
      return 0;
  }

  return (void *)GetProcAddress(mod, sym_name);
#elif defined(__linux__) || defined(__APPLE__)
  void *mod = dlopen(lib_name, RTLD_LOCAL | RTLD_NODELETE);
  if (mod == 0)
    return 0;
  void *sym = dlsym(mod, sym_name);
  dlclose(mod);
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

struct Env {
  static inline std::vector<const char *> lib_names = {
      "python38.dll",       "python37.dll",      "python3.dll",
      "python.dll",         "libpython38.so.1",  "libpython38.so",
      "libpython38m.so.1",  "libpython38m.so",   "libpython37.so.1",
      "libpython37.so",     "libpython37m.so.1", "libpython37m.so",
      "libpython3.so.1",    "libpython3.so",     "libpython3m.so.1",
      "libpython3m.so",     "libpython.so",      "libpython.so.1",
      "libpythonm.so",      "libpythonm.so.1",   "libpython38.dylib",
      "libpython38m.dylib", "libpython37.dylib", "libpython37m.dylib",
      "libpython3.dylib",   "libpython3m.dylib", "libpython.dylib",
      "libpythonm.dylib",
  };

  typedef void(__cdecl *Py_Initialize)();
  typedef PyObject *(__cdecl *PyUnicode_DecodeFSDefault)(const char *str);
  typedef PyObject *(__cdecl *PyImport_Import)(PyObject *name);
  typedef PyObject *(__cdecl *PyObject_GetAttrString)(PyObject *obj,
                                                      const char *str);
  typedef int(__cdecl *PyCallable_Check)(PyObject *obj);
  typedef PyObject *(__cdecl *PyObject_CallObject)(PyObject *obj,
                                                   PyObject *args);
  typedef PyObject *(__cdecl *PyTuple_New)(ssize_t n);
  typedef void *(__cdecl *PyTuple_SetItem)(PyObject *tuple, ssize_t idx,
                                           PyObject *item);
  typedef PyObject *(__cdecl *PyDict_New)();
  typedef int(__cdecl *PyErr_Occurred)();
  typedef void(__cdecl *PyErr_Print)();
  typedef PyObject *(__cdecl *PySys_GetObject)(const char *name);
  typedef void(__cdecl *PyList_Append)(PyObject *obj, PyObject *item);
  typedef ssize_t(__cdecl *PyTuple_Size)(PyObject *tup);
  typedef PyObject *(__cdecl *PyTuple_GetItem)(PyObject *tup, ssize_t pos);
  typedef int(__cdecl *PyType_IsSubtype)(PyTypeObject *a, PyTypeObject *b);
  typedef PyObject *(__cdecl *PyUnicode_AsUTF8String)(PyObject *unicode);
  typedef int(__cdecl *PyBytes_AsStringAndSize)(PyObject *obj, char **buffer,
                                                ssize_t *len);
  typedef PyObject *(__cdecl *Py_BuildValue)(const char *format, ...);
  typedef double(__cdecl *PyFloat_AsDouble)(PyObject *floatObj);
  typedef long long(__cdecl *PyLong_AsLongLong)(PyObject *longObj);
  typedef PyObject *(__cdecl *PyCFunction_NewEx)(PyMethodDef *mdef,
                                                 PyObject *self, PyObject *mod);
  typedef void(__cdecl *PyCapsuleDtor)(PyObject *p);
  typedef PyObject *(__cdecl *PyCapsule_New)(void *ptr, const char *name,
                                             PyCapsuleDtor dtor);
  typedef void *(__cdecl *PyCapsule_GetPointer)(PyObject *cap,
                                                const char *name);
  typedef int(__cdecl *PyObject_SetAttrString)(PyObject *obj,
                                               const char *attrName,
                                               PyObject *item);
  typedef int(__cdecl *PyArg_ParseTuple)(PyObject *args, const char *fmt, ...);
  typedef void(__cdecl *PyErr_Clear)();
  typedef int(__cdecl *PyObject_IsTrue)(PyObject *obj);
  typedef PyObject *(__cdecl *PyBool_FromLong)(long b);

  typedef PyTypeObject *PyTuple_Type;
  typedef PyTypeObject *PyLong_Type;
  typedef PyTypeObject *PyUnicode_Type;
  typedef PyTypeObject *PyFloat_Type;
  typedef PyTypeObject *PyCapsule_Type;
  typedef PyTypeObject *PyObject_Type;
  typedef PyObject *_Py_NoneStruct;

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

  static inline PyUnicode_Type _unicode_type;
  static inline PyTuple_Type _tuple_type;
  static inline PyFloat_Type _float_type;
  static inline PyLong_Type _long_type;
  static inline PyCapsule_Type _capsule_type;

  static inline _Py_NoneStruct _py_none;

  static PyObject __cdecl *methodPause(PyObject *self, PyObject *args) {
    auto ctxObj = make_pyshared(_getAttr(self, "__cbcontext__"));
    if (!isCapsule(ctxObj)) {
      LOG(ERROR) << "internal error, __cbcontext__ is not a capsule!";
      throw CBException("pause from python failed!");
    }

    auto ctx = getPtr(ctxObj);
    if (!ctx) {
      LOG(ERROR) << "internal error, __cbcontext__ was null!";
      throw CBException("pause from python failed!");
    }

    // find a double arg
    double time = 0.0;
    if (_argsParse(args, "d", &time) == -1) {
      CBInt longtime = 0;
      if (_argsParse(args, "L", &longtime) != -1) {
        time = double(longtime);
      }
    }

    // FIXME TODO handle chain state
    suspend((CBContext *)ctx, time);

    return _py_none;
  }

  static inline PyMethodDef pauseMethod{"pause", &methodPause, (1 << 0),
                                        nullptr};

  static PyObj make_pyshared(PyObject *p, bool inc_refcount = false) {
    if (inc_refcount && p)
      p->refcount++;
    std::shared_ptr<PyObject> res(p, [](auto p) {
      if (!p)
        return;

      p->refcount--;

      if (p->refcount == 0) {
        LOG(TRACE) << "PyObj::Dealloc";
        if (p->type)
          p->type->dealloc(p);
      }
    });
    return res;
  }

  static bool ensure(void *ptr, const char *name) {
    if (!ptr) {
      LOG(ERROR) << "Failed to initialize python environment, could not find "
                    "procedure: "
                 << name;
      return false;
    }
    return true;
  }

  static void init() {
    auto pos = std::find_if(std::begin(lib_names), std::end(lib_names),
                            [](auto &&str) { return hasLib(str); });
    if (pos != std::end(lib_names)) {
      auto idx = std::distance(std::begin(lib_names), pos);
      auto dll = lib_names[idx];

      auto init = (Py_Initialize)dynLoad(dll, "Py_Initialize");
      if (!ensure((void *)init, "Py_Initialize"))
        return;
      init();

#define DLIMPORT(_proc_, _name_)                                               \
  _proc_ = (_name_)dynLoad(dll, #_name_);                                      \
  if (!ensure((void *)_proc_, #_name_))                                        \
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

      DLIMPORT(_unicode_type, PyUnicode_Type);
      DLIMPORT(_tuple_type, PyTuple_Type);
      DLIMPORT(_float_type, PyFloat_Type);
      DLIMPORT(_long_type, PyLong_Type);
      DLIMPORT(_capsule_type, PyCapsule_Type);

      DLIMPORT(_py_none, _Py_NoneStruct);

      // Also add local path to "path"
      // TODO Add more paths?
      auto borrow_sysGetObj = (PySys_GetObject)dynLoad(dll, "PySys_GetObject");
      if (!ensure((void *)borrow_sysGetObj, "PySys_GetObject"))
        return;
      auto absRoot = std::filesystem::absolute(Globals::RootPath);
      auto pyAbsRoot = string(absRoot.string().c_str());
      auto path = borrow_sysGetObj("path");
      _listAppend(path, pyAbsRoot.get());

      _ok = true;

      LOG(INFO) << "Python found, Py blocks will work!";
    } else {
      LOG(INFO) << "Python not found, Py blocks won't work.";
    }
  }

  static PyObj string(const char *name) {
    return make_pyshared(_makeStr(name));
  }

  static PyObj import(const char *name) {
    auto pyname = string(name);
    return make_pyshared(_import(pyname.get()));
  }

  static PyObj getAttr(const PyObj &obj, const char *attr_name) {
    return make_pyshared(_getAttr(obj.get(), attr_name));
  }

  static void setAttr(const PyObj &obj, const char *attr_name,
                      const PyObj &item) {
    if (_setAttr(obj.get(), attr_name, item.get()) == -1) {
      printErrors();
      throw CBException("Failed to set attribute (Py)");
    }
  }

  static bool isCallable(const PyObj &obj) {
    return obj.get() && _callable(obj.get());
  }

  template <typename... Ts> static PyObj call(const PyObj &obj, Ts... vargs) {
    // this trick should allocate a shared single tuple for n kind of call
    constexpr std::size_t n = sizeof...(Ts);
    std::array<PyObject *, n> args{vargs...};
    const static auto tuple = _tupleNew(n);
    for (size_t i = 0; i < n; i++) {
      _tupleSetItem(tuple, ssize_t(i), args[i]);
    }
    return make_pyshared(_call(obj.get(), tuple), true);
  }

  static PyObj call(const PyObj &obj) {
    return make_pyshared(_call(obj.get(), nullptr), true);
  }

  static PyObj dict() { return make_pyshared(_dictNew()); }

  static bool ok() { return _ok; }

  static PyObject *var2Py(const CBVar &var) {
    switch (var.valueType) {
    case Int: {
      return _buildValue("L", var.payload.intValue);
    } break;
    case Int2: {
      return _buildValue("(LL)", var.payload.int2Value[0],
                         var.payload.int2Value[1]);
    } break;
    case Int3: {
      return _buildValue("(lll)", var.payload.int3Value[0],
                         var.payload.int3Value[1], var.payload.int3Value[2]);
    } break;
    case Int4: {
      return _buildValue("(llll)", var.payload.int4Value[0],
                         var.payload.int4Value[1], var.payload.int4Value[2],
                         var.payload.int4Value[3]);
    } break;
    case Float: {
      return _buildValue("d", var.payload.floatValue);
    } break;
    case Float2: {
      return _buildValue("(dd)", var.payload.float2Value[0],
                         var.payload.float2Value[1]);
    } break;
    case Float3: {
      return _buildValue("(fff)", var.payload.float3Value[0],
                         var.payload.float3Value[1],
                         var.payload.float3Value[2]);
    } break;
    case Float4: {
      return _buildValue("(dddd)", var.payload.float4Value[0],
                         var.payload.float4Value[1], var.payload.float4Value[2],
                         var.payload.float4Value[3]);
    } break;
    case String: {
      return _buildValue("u", var.payload.stringValue);
    } break;
    case Bool: {
      return _bool(long(var.payload.boolValue));
    } break;
    case None: {
      return _py_none;
    } break;
    default:
      LOG(ERROR) << "Unsupported type " << type2Name(var.valueType);
      throw CBException(
          "Failed to convert CBVar into PyObject, type not supported!");
    }
  }

  static CBVar py2Var(const PyObj &obj) {
    CBVar res{};
    if (isLong(obj)) {
      res.valueType = Int;
      res.payload.intValue = _asLong(obj.get());
    } else if (isFloat(obj)) {
      res.valueType = Float;
      res.payload.floatValue = _asDouble(obj.get());
    }
    return res;
  }

  static void printErrors() {
    if (_errOccurred()) {
      _errPrint();
      _errClear();
    }
  }

  static void clearError() { _errClear(); }

  static bool isTuple(const PyObj &obj) {
    return obj.get() && (obj.get()->type == _tuple_type ||
                         _typeIsSubType(obj.get()->type, _tuple_type));
  }

  static bool isString(const PyObject *obj) {
    return obj->type == _unicode_type ||
           _typeIsSubType(obj->type, _unicode_type);
  }

  static bool isString(const PyObj &obj) {
    return obj.get() && isString(obj.get());
  }

  static bool isLong(const PyObject *obj) {
    return obj->type == _long_type || _typeIsSubType(obj->type, _long_type);
  }

  static bool isLong(const PyObj &obj) {
    return obj.get() && isLong(obj.get());
  }

  static bool isFloat(const PyObject *obj) {
    return obj->type == _float_type || _typeIsSubType(obj->type, _float_type);
  }

  static bool isFloat(const PyObj &obj) {
    return obj.get() && isFloat(obj.get());
  }

  static bool isCapsule(const PyObject *obj) {
    return obj->type == _capsule_type ||
           _typeIsSubType(obj->type, _capsule_type);
  }

  static bool isCapsule(const PyObj &obj) {
    return obj.get() && isCapsule(obj.get());
  }

  static bool isNone(const PyObject *obj) { return obj == _py_none; }

  static bool isNone(const PyObj &obj) {
    return obj.get() && isNone(obj.get());
  }

  static ssize_t tupleSize(const PyObj &obj) { return _tupleSize(obj.get()); }

  static PyObject *tupleGetItem(const PyObj &tup, ssize_t idx) {
    return _borrow_tupleGetItem(tup.get(), idx);
  }

  static std::string_view toStringView(PyObject *obj) {
    char *str;
    ssize_t len;
    auto utf = make_pyshared(_unicodeToString(obj));
    auto res = _bytesAsStringAndSize(utf.get(), &str, &len);
    if (res == -1) {
      printErrors();
      throw CBException("String conversion failed!");
    }
    return std::string_view(str, len);
  }

  static std::string_view toStringView(const PyObj &obj) {
    return toStringView(obj.get());
  }

  static PyObj capsule(void *ptr) {
    return make_pyshared(_newCapsule(ptr, nullptr, nullptr));
  }

  static void *getPtr(const PyObj &obj) {
    return _capsuleGet(obj.get(), nullptr);
  }

  static PyObj none() { return PyObj(_py_none); }

  static PyObj func(PyMethodDef &def, const PyObj &self) {
    return make_pyshared(_newFunc(&def, self.get(), nullptr));
  }

  static PyObject *intVal(int i) { return _buildValue("i", i); }

  using ToTypesFailed = CBException;

  static CBType toCBType(const std::string_view &str) {
    if (str == "Int") {
      return CBType::Int;
    } else if (str == "Int2") {
      return CBType::Int2;
    } else if (str == "Int3") {
      return CBType::Int3;
    } else if (str == "Int4") {
      return CBType::Int4;
    } else if (str == "Float") {
      return CBType::Float;
    } else if (str == "Float2") {
      return CBType::Float2;
    } else if (str == "Float3") {
      return CBType::Float3;
    } else if (str == "Float4") {
      return CBType::Float4;
    } else if (str == "String") {
      return CBType::String;
    } else if (str == "Any") {
      return CBType::Any;
    } else if (str == "None") {
      return CBType::None;
    } else if (str == "Bool") {
      return CBType::Bool;
    } else {
      throw ToTypesFailed("Unsupported toCBType type.");
    }
  }

  static Types toTypes(const PyObj &obj) {
    std::vector<CBTypeInfo> types;
    if (Env::isTuple(obj)) {
      // multiple types
      auto size = Env::tupleSize(obj);
      for (ssize_t i = 0; i < size; i++) {
        auto item = Env::tupleGetItem(obj, i);
        if (Env::isString(item)) {
          auto str = Env::toStringView(item);
          types.emplace_back(CBTypeInfo{Env::toCBType(str)});
        } else {
          throw ToTypesFailed(
              "Failed to transform python object to Types within tuple.");
        }
      }
    } else if (Env::isString(obj)) {
      // single type will just be string
      auto str = Env::toStringView(obj);
      types.emplace_back(CBTypeInfo{Env::toCBType(str)});
    } else {
      printErrors();
      throw ToTypesFailed("Failed to transform python object to Types.");
    }
    return Types(types);
  }

private:
  static inline bool _ok{false};
};

struct Py {
  void setup() {
    if (!Env::ok()) {
      Env::init();
    }
  }

  Parameters params{{"Module",
                     "The module name to load (must be in the script path, .py "
                     "extension added internally!)",
                     {CoreInfo::StringType}}};

  CBParametersInfo parameters() {
    if (Env::isCallable(_parameters)) {
      auto pyparams = Env::call(_parameters, _self.get());
      if (Env::isTuple(pyparams)) {
        // this in turn could be either already string (string) type or more
        // tuples
      } else {
        Env::printErrors();
        throw CBException("Failed to fetch python block parameters");
      }
      std::vector<ParameterInfo> otherParams;
      _params = Parameters(params, otherParams);
    } else {
      _params = Parameters(params);
    }
    return params;
  }

  void setParam(int index, CBVar value) {
    if (index == 0) {
      // Handle here
      _scriptName = value.payload.stringValue;
      reloadScript();
    } else {
      if (!Env::isCallable(_setParam)) {
        LOG(ERROR) << "Script: " << _scriptName
                   << " cannot call setParam, is it missing?";
        throw CBException("Python block setParam is not callable!");
      }
      Env::call(_setParam, _self.get(), Env::intVal(index), Env::var2Py(value));
    }
  }

  CBVar getParam(int index) {
    if (index == 0) {
      return Var(_scriptName);
    } else {
      // To the script
    }
  }

  void reloadScript() {
    if (!Env::ok()) {
      LOG(ERROR) << "Script: " << _scriptName
                 << " cannot be loaded, no python support.";
      throw CBException("Failed to load python script!");
    }

    _module = Env::import(_scriptName.c_str());
    if (!_module.get()) {
      LOG(ERROR) << "Script: " << _scriptName << " failed to load!.";
      Env::printErrors();
      throw CBException("Failed to load python script!");
    }

    _inputTypes = Env::getAttr(_module, "inputTypes");
    if (!Env::isCallable(_inputTypes)) {
      LOG(ERROR) << "Script: " << _scriptName << " has no callable inputTypes.";
      throw CBException("Failed to reload python script!");
    }

    _outputTypes = Env::getAttr(_module, "outputTypes");
    if (!Env::isCallable(_outputTypes)) {
      LOG(ERROR) << "Script: " << _scriptName
                 << " has no callable outputTypes.";
      throw CBException("Failed to reload python script!");
    }

    _activate = Env::getAttr(_module, "activate");
    if (!Env::isCallable(_activate)) {
      LOG(ERROR) << "Script: " << _scriptName << " has no callable activate.";
      throw CBException("Failed to reload python script!");
    }

    // Optional stuff
    _parameters = Env::getAttr(_module, "parameters");
    _setParam = Env::getAttr(_module, "setParam");
    _getParam = Env::getAttr(_module, "getParam");

    auto setup = Env::getAttr(_module, "setup");

    if (Env::isCallable(setup)) {
      _self = Env::call(setup);
      if (Env::isNone(_self)) {
        LOG(ERROR) << "Script: " << _scriptName
                   << " setup must return a valid object.";
        throw CBException("Failed to reload python script!");
      }

      auto pause1 = Env::func(Env::pauseMethod, _self);
      Env::setAttr(_self, "pause", pause1);
    }

    Env::clearError();
  }

  CBTypesInfo inputTypes() {
    PyObj pytype;
    if (_self.get())
      pytype = Env::call(_inputTypes, _self.get());
    else
      pytype = Env::call(_inputTypes);
    try {
      _inputTypesStorage = Env::toTypes(pytype);
    } catch (Env::ToTypesFailed &ex) {
      LOG(ERROR) << ex.what();
      LOG(ERROR) << "Script: " << _scriptName
                 << " inputTypes method should return a tuple of strings "
                    "or a string.";
      throw CBException("Failed call inputTypes on python script!");
    }
    return _inputTypesStorage;
  }

  CBTypesInfo outputTypes() {
    PyObj pytype;
    if (_self.get())
      pytype = Env::call(_outputTypes, _self.get());
    else
      pytype = Env::call(_outputTypes);
    try {
      _outputTypesStorage = Env::toTypes(pytype);
    } catch (Env::ToTypesFailed &ex) {
      LOG(ERROR) << ex.what();
      LOG(ERROR) << "Script: " << _scriptName
                 << " outputTypes method should return a tuple of strings "
                    "or a string.";
      throw CBException("Failed call outputTypes on python script!");
    }
    return _outputTypesStorage;
  }

  CBVar activate(CBContext *context, const CBVar &input) {
    PyObj pyres;
    if (_self.get()) {
      auto pyctx = Env::capsule(context);
      Env::setAttr(_self, "__cbcontext__", pyctx);
      pyres = Env::call(_activate, _self.get(), Env::var2Py(input));
    } else {
      pyres = Env::call(_activate, Env::var2Py(input));
    }
    if (!pyres.get()) {
      Env::printErrors();
      throw CBException("Python script activation failed.");
    }
    return Env::py2Var(pyres);
  }

private:
  Types _inputTypesStorage;
  Types _outputTypesStorage;
  Parameters _params;

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

  std::string _scriptName;
};

void registerBlocks() { REGISTER_CBLOCK("Py", Py); }
} // namespace Python
} // namespace chainblocks

#endif /* CB_PYTHON_HPP */
