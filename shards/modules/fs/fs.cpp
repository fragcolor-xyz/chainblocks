/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include <boost/filesystem/operations.hpp>
#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/utility.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <fcntl.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <boost/filesystem.hpp>
#include <system_error>
namespace fs = boost::filesystem;
using ErrorCode = boost::system::error_code;

namespace shards {
namespace FS {

void portable_fsync(const boost::filesystem::path &filePath) {
#if defined(_WIN32) || defined(_WIN64)
  HANDLE hFile = CreateFileW(filePath.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) {
    SHLOG_ERROR("Error opening file: {} ({})", GetLastError(), filePath);
    return;
  }

  if (!FlushFileBuffers(hFile)) {
    SHLOG_ERROR("Error in FlushFileBuffers: {}", GetLastError());
  }

  CloseHandle(hFile);

#else
  int fd = ::open(filePath.c_str(), O_RDWR);
  if (fd == -1) {
    SHLOG_ERROR("Error opening file: {}", strerror(errno));
    return;
  }

  if (::fsync(fd) == -1) {
    SHLOG_ERROR("Error in fsync: {}", strerror(errno));
  }

  ::close(fd);
#endif
}

struct FileNotFoundException : public std::runtime_error {
  FileNotFoundException(const std::string &err) : std::runtime_error(err) {}
};

struct Iterate {
  SHSeq _storage = {};
  std::vector<std::string> _strings;

  void destroy() {
    if (_storage.elements) {
      shards::arrayFree(_storage);
    }
  }

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringSeqType; }

  bool _recursive = true;

  static inline ParamsInfo params = ParamsInfo(ParamsInfo::Param(
      "Recursive", SHCCSTR("If the iteration should be recursive, following sub-directories."), CoreInfo::BoolType));
  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _recursive = bool(Var(value));
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_recursive);
    default:
      return Var::Empty;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    shards::arrayResize(_storage, 0);
    _strings.clear();

    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));

    if (!fs::exists(p)) {
      throw ActivationError(fmt::format("FS.Iterate, path {} does not exist.", p));
    }

    if (_recursive) {
      auto dIterator = fs::recursive_directory_iterator(p);
      for (auto &subP : dIterator) {
        auto &path = subP.path();
        auto str = path.string();
#ifdef _WIN32
        boost::replace_all(str, "\\", "/");
#endif
        _strings.push_back(str);
      }
    } else {
      auto dIterator = fs::directory_iterator(p);
      for (auto &subP : dIterator) {
        auto &path = subP.path();
        auto str = path.string();
#ifdef _WIN32
        boost::replace_all(str, "\\", "/");
#endif
        _strings.push_back(str);
      }
    }

    shards::arrayResize(_storage, 0);
    for (auto &sRef : _strings) {
      shards::arrayPush(_storage, Var(sRef));
    }

    return Var(_storage);
  }
};

struct Join {
  fs::path _buffer;
  std::string _result;

  static SHTypesInfo inputTypes() { return CoreInfo::StringSeqType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    _buffer.clear();

    for (auto &v : input) {
      fs::path path(SHSTRING_PREFER_SHSTRVIEW(v));
      _buffer /= path;
    }
    _result = _buffer.string();
    return Var(_result);
  }
};

struct Extension {
  std::string _output;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    _output.clear();
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    auto ext = p.extension();
    _output.assign(ext.string());
    return Var(_output);
  }
};

struct IsFile {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    if (fs::exists(p) && !fs::is_directory(p)) {
      return Var::True;
    } else {
      return Var::False;
    }
  }
};

struct IsDirectory {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    if (fs::exists(p) && fs::is_directory(p)) {
      return Var::True;
    } else {
      return Var::False;
    }
  }
};

struct Remove {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    if (fs::exists(p)) {
      return Var(fs::remove(p));
    } else {
      return Var::False;
    }
  }
};

struct RemoveAll {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    if (fs::exists(p)) {
      return Var(SHInt(fs::remove_all(p)));
    } else {
      return Var(0);
    }
  }
};

struct Filename {
  std::string _output;
  bool _noExt = false;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  static inline ParamsInfo params = ParamsInfo(
      ParamsInfo::Param("NoExtension", SHCCSTR("If the extension should be stripped from the result."), CoreInfo::BoolType));
  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _noExt = bool(Var(value));
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_noExt);
    default:
      return Var::Empty;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    _output.clear();
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    p = p.filename();
    if (_noExt && p.has_extension()) {
      p = p.replace_extension("");
    }
    _output.assign(p.string());
    return Var(_output);
  }
};

struct RelativeTo {
  std::string _output;
  fs::path _cachedBasePath;
  fs::path _cachedPath;
  bool _noExt = false;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_PARAMVAR(_basePath, "BasePath", "The base path to make the input path relative to", {CoreInfo::StringOrStringVar});
  PARAM_IMPL(PARAM_IMPL_FOR(_basePath));

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    _output.clear();

    _cachedPath = SHSTRING_PREFER_SHSTRVIEW(input);
    _cachedBasePath = SHSTRING_PREFER_SHSTRVIEW(_basePath.get());

    _output.assign(fs::relative(_cachedPath, _cachedBasePath).string());
    return Var(_output);
  }
};

struct Parent {
  std::string _output;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    _output.clear();
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    p = p.parent_path();
    _output.assign(p.string());
    return Var(_output);
  }
};

struct Read {
  std::vector<uint8_t> _buffer;
  bool _binary = false;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() {
    static Types _types{CoreInfo::BytesType, CoreInfo::StringType};
    return _types;
  }

  static inline ParamsInfo params = ParamsInfo(ParamsInfo::Param(
      "Bytes", SHCCSTR("If the output should be SHType::Bytes instead of SHType::String."), CoreInfo::BoolType));
  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _binary = bool(Var(value));
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_binary);
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) { return _binary ? CoreInfo::BytesType : CoreInfo::StringType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    _buffer.clear();
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    if (!fs::exists(p)) {
      SHLOG_ERROR("File is missing: {}", p);
      throw FileNotFoundException("FS.Read, file does not exist.");
    }

    if (_binary) {
      std::ifstream file(p.string(), std::ios::binary);
      _buffer.assign(std::istreambuf_iterator<char>(file), {});
      return Var(_buffer.data(), uint32_t(_buffer.size()));
    } else {
      std::ifstream file(p.string(), std::ios::binary);
      _buffer.assign(std::istreambuf_iterator<char>(file), {});
      _buffer.push_back(0);
      return Var((const char *)_buffer.data(), _buffer.size() - 1);
    }
  }
};

struct Write {
  ParamVar _contents{};
  SHExposedTypeInfo _requiring;
  bool _overwrite = false;
  bool _append = false;
  bool _sync = false;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  static inline Parameters params{
      {"Contents",
       SHCCSTR("The string or bytes to write as the file's contents."),
       {CoreInfo::StringType, CoreInfo::BytesType, CoreInfo::StringVarType, CoreInfo::BytesVarType, CoreInfo::NoneType}},
      {"Overwrite", SHCCSTR("Overwrite the file if it already exists."), {CoreInfo::BoolType}},
      {"Append", SHCCSTR("If we should append Contents to an existing file."), {CoreInfo::BoolType}},
      {"Sync", SHCCSTR("If we should sync the file to disk after writing."), {CoreInfo::BoolType}}};

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _contents = value;
      break;
    case 1:
      _overwrite = value.payload.boolValue;
      break;
    case 2:
      _append = value.payload.boolValue;
      break;
    case 3:
      _sync = value.payload.boolValue;
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _contents;
    case 1:
      return Var(_overwrite);
    case 2:
      return Var(_append);
    case 3:
      return Var(_sync);
    default:
      return Var::Empty;
    }
  }

  SHExposedTypesInfo requiredVariables() {
    if (_contents.isVariable()) {
      return {&_requiring, 1, 0};
    } else {
      return {};
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_contents.isVariable()) {
      auto type = findParamVarExposedType(data, _contents);
      if (!type)
        throw ComposeError("Content missing");
      _requiring = *type;
    }
    return outputTypes().elements[0];
  }

  void cleanup(SHContext *context) { _contents.cleanup(); }
  void warmup(SHContext *context) { _contents.warmup(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto contents = _contents.get();
    if (contents.valueType != SHType::None) {
      fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
      if (!_overwrite && !_append && fs::exists(p)) {
        throw ActivationError("FS.Write, file already exists and overwrite flag is not on!.");
      }

      // make sure to create directories
      auto parent_path = p.parent_path();
      if (!parent_path.empty() && !fs::exists(parent_path))
        fs::create_directories(p.parent_path());

      std::ios::openmode flags = std::ios::binary;
      if (_append) {
        flags |= std::ios::app;
      }
      std::ofstream file(p.string(), flags);
      if (contents.valueType == SHType::String) {
        auto len = contents.payload.stringLen > 0 || contents.payload.stringValue == nullptr
                       ? contents.payload.stringLen
                       : strlen(contents.payload.stringValue);
        file.write((const char *)contents.payload.stringValue, len);
      } else {
        file.write((const char *)contents.payload.bytesValue, contents.payload.bytesSize);
      }

      if (_sync) {
        portable_fsync(p);
      }
    }
    return input;
  }
};

struct Copy {
  enum class IfExists { Fail, Skip, Overwrite, Update };
  DECL_ENUM_INFO(IfExists, IfExists,
                 "Action to take when a destination file already exists during a copy operation. Determines whether to fail, "
                 "skip, overwrite, or update the file.",
                 'fsow');

  ParamVar _destination{};
  IfExists _overwrite{IfExists::Fail};

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  static inline ParamsInfo params = ParamsInfo(
      ParamsInfo::Param("Destination", SHCCSTR("The destination path, can be a file or a directory."),
                        CoreInfo::StringStringVarOrNone),
      ParamsInfo::Param("Behavior", SHCCSTR("What to do when the destination already exists."), IfExistsEnumInfo::Type));
  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _destination = value;
      break;
    case 1:
      _overwrite = IfExists(value.payload.enumValue);
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _destination;
    case 1:
      return Var::Enum(_overwrite, CoreCC, IfExistsEnumInfo::TypeId);
    default:
      return Var::Empty;
    }
  }

  void cleanup(SHContext *context) { _destination.cleanup(); }
  void warmup(SHContext *context) { _destination.warmup(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    const auto src = fs::path(SHSTRING_PREFER_SHSTRVIEW(input));
    if (!fs::exists(src))
      throw FileNotFoundException("Source path does not exist.");

    fs::copy_options options{};

    switch (_overwrite) {
    case IfExists::Fail:
      break;
    case IfExists::Skip:
      options |= fs::copy_options::skip_existing;
      break;
    case IfExists::Overwrite:
      options |= fs::copy_options::overwrite_existing;
      break;
    case IfExists::Update:
      options |= fs::copy_options::update_existing;
      break;
    }

    const auto dstVar = _destination.get();
    if (dstVar.valueType != SHType::String && dstVar.valueType != SHType::Path)
      throw ActivationError("Destination is not a valid");
    const auto dst = fs::path(SHSTRING_PREFER_SHSTRVIEW(dstVar));

    ErrorCode err;
    if (fs::is_regular_file(src) && (!fs::exists(dst) || fs::is_regular_file(dst))) {
      fs::copy_file(src, dst, options, err);
      if (err) {
        SHLOG_ERROR("copy error: {}", err.message());
        throw ActivationError("Copy failed.");
      }
    } else {
      options |= fs::copy_options::recursive;
      fs::copy(src, dst, options, err);
      if (err) {
        SHLOG_ERROR("copy error: {}", err.message());
        throw ActivationError("Copy failed.");
      }
    }

    return input;
  }
};

struct LastWriteTime {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }

  static SHParametersInfo parameters() { return SHParametersInfo{}; }

  void setParam(int index, const SHVar &value) {}
  SHVar getParam(int index) { return Var::Empty; }

  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    ErrorCode ec;
    auto lwt = fs::last_write_time(p, ec);
    if (ec.failed()) {
      throw FileNotFoundException(fmt::format("FS.LastWriteTime, file {} does not exist.", p));
    }
    return Var(static_cast<int64_t>(lwt));
  }
};

struct SetWriteTime {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_PARAMVAR(_time, "Time", "The new time to set as last write time", {CoreInfo::IntOrIntVar});
  PARAM_IMPL(PARAM_IMPL_FOR(_time));

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    auto time = _time.get();
    ErrorCode ec;
    fs::last_write_time(p, time.payload.intValue, ec);
    if (ec.failed()) {
      throw FileNotFoundException(fmt::format("FS.SetWriteTime, file {} does not exist.", p));
    }
    return input;
  }
};

struct CreateDirectories {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    if (!fs::exists(p)) {
      if (!fs::create_directories(p)) {
        throw ActivationError(fmt::format("FS.CreateDirectories, failed to create directories for {}.", p));
      }
    }
    return input;
  }
};

struct Absolute {
  std::string _output;

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_PARAMVAR(_canonical, "Canonical", "Whether to canonicalize the path, the file should exist for this to work", {CoreInfo::BoolType, CoreInfo::BoolVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_canonical));

  Absolute() { _canonical = Var(false); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    _output.clear();
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    p = fs::absolute(p);
    p = _canonical.get().payload.boolValue ? fs::canonical(p) : p;
    _output.assign(p.string());
    return Var(_output);
  }
};

struct Rename {
  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_PARAMVAR(_newName, "NewName", "The new name for the file", {CoreInfo::StringOrStringVar});
  PARAM_IMPL(PARAM_IMPL_FOR(_newName));

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    fs::path p(SHSTRING_PREFER_SHSTRVIEW(input));
    // check exists
    if (!fs::exists(p)) {
      throw ActivationError(fmt::format("FS.Rename, file {} does not exist.", p));
    }
    auto newName = SHSTRING_PREFER_SHSTRVIEW(_newName.get());
    // check if exists
    if (fs ::exists(newName)) {
      throw ActivationError(fmt::format("FS.Rename, file {} already exists.", newName));
    }
    fs::rename(p, newName);
    return input;
  }
};
}; // namespace FS

SHARDS_REGISTER_FN(fs) {
  REGISTER_ENUM(FS::Copy::IfExistsEnumInfo);

  REGISTER_SHARD("FS.Join", FS::Join);
  REGISTER_SHARD("FS.Iterate", FS::Iterate);
  REGISTER_SHARD("FS.Extension", FS::Extension);
  REGISTER_SHARD("FS.Filename", FS::Filename);
  REGISTER_SHARD("FS.RelativeTo", FS::RelativeTo);
  REGISTER_SHARD("FS.Parent", FS::Parent);
  REGISTER_SHARD("FS.Read", FS::Read);
  REGISTER_SHARD("FS.Write", FS::Write);
  REGISTER_SHARD("FS.IsFile", FS::IsFile);
  REGISTER_SHARD("FS.IsDirectory", FS::IsDirectory);
  REGISTER_SHARD("FS.Copy", FS::Copy);
  REGISTER_SHARD("FS.Remove", FS::Remove);
  REGISTER_SHARD("FS.RemoveAll", FS::RemoveAll);
  REGISTER_SHARD("FS.LastWriteTime", FS::LastWriteTime);
  REGISTER_SHARD("FS.SetWriteTime", FS::SetWriteTime);
  REGISTER_SHARD("FS.CreateDirectories", FS::CreateDirectories);
  REGISTER_SHARD("FS.Absolute", FS::Absolute);
  REGISTER_SHARD("FS.Rename", FS::Rename);
}
}; // namespace shards
