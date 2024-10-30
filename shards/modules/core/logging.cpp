/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include "logging.hpp"
#include <shards/shardwrapper.hpp>
#include <shards/utility.hpp>
#include <shards/core/params.hpp>
#include <spdlog/spdlog.h>
#include <atomic>
#include <numeric>
#include <string>
#include <cstdio>
#include <shards/log/log.hpp>
#include <spdlog/sinks/dist_sink.h>

namespace shards {
REGISTER_ENUM(Enums::LogLevelEnumInfo);

struct LoggingBase {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
};

#define SHLOG_LEVEL(_level_, ...)                                                                      \
  {                                                                                                    \
    SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), spdlog::level::level_enum(_level_), __VA_ARGS__); \
  }

struct Log : public LoggingBase {
  shards::logging::Logger _logger;
  std::string _name = "shards";

  static SHOptionalString inputHelp() { return SHCCSTR("The value to be logged to the console."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The same value that was inputted, unmodified."); }

  static SHOptionalString help() {
    return SHCCSTR("Logs the output of a shard or the value of a variable to the console along with an optional prefix string. "
                   "The logging level can be specified to control the verbosity of the log output.");
  }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &inValue) {
    switch (index) {
    case 0:
      _prefix = SHSTRVIEW(inValue);
      break;
    case 1:
      _level = Enums::LogLevel(inValue.payload.enumValue);
      break;
    case 2:
      _name = std::string(SHSTRVIEW(inValue));
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_prefix);
    case 1:
      return Var::Enum(_level, CoreCC, Enums::LogLevelEnumInfo::TypeId);
    case 2:
      return Var(_name);
    default:
      return Var::Empty;
    }
  }

  void warmup(SHContext *context) { _logger = shards::logging::getOrCreate(_name); }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto current = context->wireStack.back();
    auto id = findId(context);
    if (_prefix.size() > 0) {
      if (id != entt::null) {
        SPDLOG_LOGGER_CALL(_logger, spdlog::level::level_enum(_level), "[{} {}] {}: {}", current->name, id, _prefix, input);
      } else {
        SPDLOG_LOGGER_CALL(_logger, spdlog::level::level_enum(_level), "[{}] {}: {}", current->name, _prefix, input);
      }
    } else {
      if (id != entt::null) {
        SPDLOG_LOGGER_CALL(_logger, spdlog::level::level_enum(_level), "[{} {}] {}", current->name, id, input);
      } else {
        SPDLOG_LOGGER_CALL(_logger, spdlog::level::level_enum(_level), "[{}] {}", current->name, input);
      }
    }
    return input;
  }

  static inline Parameters _params = {
      {"Prefix",
       SHCCSTR("The message to prefix to the logged output. Note: the prefix will include a colon ':' before the value."),
       {CoreInfo::StringType}},
      {"Level", SHCCSTR("The level of logging."), {Enums::LogLevelEnumInfo::Type}},
      {"Name", SHCCSTR("The name of the logger to use, the default is 'shards'."), {CoreInfo::StringType}}};

  std::string _prefix;
  Enums::LogLevel _level{Enums::LogLevel::Info};
};

struct LogType : public Log {
  static SHOptionalString inputHelp() { return SHCCSTR("The value whose type will be logged to the console."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The same value that was inputted, unmodified."); }

  static SHOptionalString help() {
    return SHCCSTR("Logs the type of the value to the console along with an optional prefix string. The logging level can be "
                   "specified to control the verbosity of the log output.");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto current = context->wireStack.back();
    auto id = findId(context);
    if (_prefix.size() > 0) {
      if (id != entt::null) {
        SHLOG_LEVEL((int)_level, "[{} {}] {}: {}", current->name, id, _prefix, type2Name(input.valueType));
      } else {
        SHLOG_LEVEL((int)_level, "[{}] {}: {}", current->name, _prefix, type2Name(input.valueType));
      }
    } else {
      if (id != entt::null) {
        SHLOG_LEVEL((int)_level, "[{} {}] {}", current->name, id, type2Name(input.valueType));
      } else {
        SHLOG_LEVEL((int)_level, "[{}] {}", current->name, type2Name(input.valueType));
      }
    }
    return input;
  }
};

struct Msg : public LoggingBase {
  static SHOptionalString inputHelp() { return SHCCSTR("The input is ignored. This shard displays a static message."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The same variable that was inputted, unmodified."); }

  static SHOptionalString help() {
    return SHCCSTR("Displays the passed message string to the user via standard output. The input variable is ignored, and only "
                   "the static message is displayed.");
  }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &inValue) {
    switch (index) {
    case 0: {
      auto sv = SHSTRVIEW(inValue);
      _msg = sv;
    } break;
    case 1:
      _level = Enums::LogLevel(inValue.payload.enumValue);
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_msg);
    case 1:
      return Var::Enum(_level, CoreCC, Enums::LogLevelEnumInfo::TypeId);
    default:
      return Var::Empty;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto current = context->wireStack.back();
    auto id = findId(context);
    if (id != entt::null) {
      SHLOG_LEVEL((int)_level, "[{} {}] {}", current->name, id, _msg);
    } else {
      SHLOG_LEVEL((int)_level, "[{}] {}", current->name, _msg);
    }
    return input;
  }

private:
  static inline Parameters _params = {
      {"Message", SHCCSTR("The message to display on the user's screen or console."), {CoreInfo::StringType}},
      {"Level", SHCCSTR("The level of logging."), {Enums::LogLevelEnumInfo::Type}}};

  std::string _msg;
  Enums::LogLevel _level{Enums::LogLevel::Info};
};

struct Output {
  static SHOptionalString inputHelp() { return SHCCSTR("The string or bytes to output to stdout."); }
  static SHOptionalString outputHelp() { return SHCCSTR("The same variable that was inputted, unmodified."); }

  static SHOptionalString help() { return SHCCSTR("Outputs data directly to stdout"); }

  static SHTypesInfo inputTypes() {
    static Types types{CoreInfo::StringType, CoreInfo::BytesType};
    return types;
  }
  static SHTypesInfo outputTypes() { return inputTypes(); }

  PARAM_IMPL();

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (input.valueType == SHType::String) {
      fwrite(input.payload.stringValue, input.payload.stringLen, 1, stdout);
    } else if (input.valueType == SHType::Bytes) {
      fwrite(input.payload.bytesValue, input.payload.bytesSize, 1, stdout);
    }
    return input;
  }
};

/*
 * Custom ring buffer sink
 */
template <typename Mutex> class custom_ringbuffer_sink final : public spdlog::sinks::base_sink<Mutex> {
public:
  explicit custom_ringbuffer_sink(size_t n_items, spdlog::level::level_enum min_level = spdlog::level::debug)
      : q_{n_items}, min_level_(min_level) {}

  inline bool is_dirty() const noexcept { return _dirty.load(); }

  std::vector<std::string> get_last_formatted() {
    if (is_dirty()) {
      std::lock_guard<Mutex> lock(spdlog::sinks::base_sink<Mutex>::mutex_);
      _dirty = false;

      auto n_items = q_.size();
      _formatted_cache.clear();
      _formatted_cache.reserve(n_items);
      for (size_t i = 0; i < n_items; i++) {
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(q_.at(i), formatted);
        _formatted_cache.push_back(fmt::to_string(formatted));
      }
    }

    return _formatted_cache;
  }

protected:
  void sink_it_(const spdlog::details::log_msg &msg) override {
    if (msg.level >= min_level_) {
      _dirty = true;
      q_.push_back(spdlog::details::log_msg_buffer{msg});
    }
  }
  void flush_() override {}

private:
  spdlog::details::circular_q<spdlog::details::log_msg_buffer> q_;
  spdlog::level::level_enum min_level_;

  std::atomic<bool> _dirty{false};
  std::vector<std::string> _formatted_cache;
};

using custom_ringbuffer_sink_mt = custom_ringbuffer_sink<std::mutex>;
using custom_ringbuffer_sink_st = custom_ringbuffer_sink<spdlog::details::null_mutex>;

struct CaptureLog {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input is ignored. This shard captures log messages based on specified parameters.");
  }

  static SHTypesInfo outputTypes() { return CoreInfo::StringSeqType; }

  static SHOptionalString outputHelp() { return SHCCSTR("A sequence of captured log messages."); }

  static SHOptionalString help() {
    return SHCCSTR(
        "Captures log messages based on specified parameters, such as the number of messages to retain, the minimum log level, "
        "and the log format pattern. It can optionally suspend execution until new log messages are available.");
  }

  static SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _n_items = size_t(value.payload.intValue);
      break;
    case 1:
      _min_level = SHSTRVIEW(value);
      break;
    case 2:
      _pattern = SHSTRVIEW(value);
      break;
    case 3:
      _suspend = value.payload.boolValue;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(int64_t(_n_items));
    case 1:
      return Var(_min_level);
    case 2:
      return Var(_pattern);
    case 3:
      return Var(_suspend);
    default:
      return Var::Empty;
    }
  }

  void warmup(SHContext *context) {
    auto logger = spdlog::default_logger();
    assert(logger);
    auto sink = logger->sinks().at(0);
    assert(sink);
    auto dist_sink = std::dynamic_pointer_cast<spdlog::sinks::dist_sink_mt>(sink);
    if (dist_sink) {
      _ring = std::make_shared<custom_ringbuffer_sink_mt>(_n_items, spdlog::level::from_str(_min_level));
      _ring->set_formatter(
          std::unique_ptr<spdlog::formatter>(new spdlog::pattern_formatter(_pattern, spdlog::pattern_time_type::local)));
      dist_sink->add_sink(_ring);
    }
  }

  void cleanup(SHContext *context) {
    auto logger = spdlog::default_logger();
    assert(logger);
    auto sink = logger->sinks().at(0);
    assert(sink);
    auto dist_sink = std::dynamic_pointer_cast<spdlog::sinks::dist_sink_mt>(sink);
    if (dist_sink) {
      dist_sink->remove_sink(_ring);
      _ring.reset();
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_suspend) {
      OVERRIDE_ACTIVATE(data, activateWithSuspend);
    }

    return CoreInfo::StringSeqType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (likely((bool)_ring)) {
      if (_ring->is_dirty()) {
        updateSeq();
      }
    }
    return Var(SHSeq(_seq));
  }

  SHVar activateWithSuspend(SHContext *context, const SHVar &input) {
    if (likely((bool)_ring)) {
      while (!_ring->is_dirty()) {
        SH_SUSPEND(context, 0);
      }

      updateSeq();
    }
    return Var(SHSeq(_seq));
  }

private:
  void updateSeq() {
    auto msgs = _ring->get_last_formatted();
    auto size = msgs.size();
    _pool.resize(size);
    _seq.resize(size);
    for (size_t i = 0; i < size; i++) {
      _pool[i].assign(msgs[i]);
      _seq[i] = Var(_pool[i]);
    }
  }

  static inline Parameters _params{
      {"Size", SHCCSTR("The maximum number of logs to retain."), {CoreInfo::IntType}},
      {"MinLevel", SHCCSTR("The minimum level of logs to capture."), {CoreInfo::StringType}},
      {"Pattern", SHCCSTR("The pattern used to format the logs."), {CoreInfo::StringType}},
      {"Suspend", SHCCSTR("Suspend execution until new logs are available."), {CoreInfo::BoolType}},
  };

  std::vector<std::string> _pool;
  IterableSeq _seq;

  size_t _n_items{8};
  std::string _min_level{"debug"};
  std::string _pattern{"%^[%l]%$ [%Y-%m-%d %T.%e] [T-%t] [%s::%#] %v"};
  std::shared_ptr<custom_ringbuffer_sink_mt> _ring;
  bool _suspend{false};
};

SHVar logsFlushActivation(const SHVar &input) {
  spdlog::default_logger()->flush();
  return input;
}

SHVar logsChangeLevelActivation(const SHVar &input) {
  auto level = SHSTRING_PREFER_SHSTRVIEW(input);
  spdlog::set_level(spdlog::level::from_str(level));
  return input;
}

SHARDS_REGISTER_FN(logging) {
  REGISTER_SHARD("Log", Log);
  REGISTER_SHARD("LogType", LogType);
  REGISTER_SHARD("Msg", Msg);
  REGISTER_SHARD("Output", Output);
  REGISTER_SHARD("CaptureLog", CaptureLog);

  struct LogFlush : public LambdaShard<logsFlushActivation, CoreInfo::AnyType, CoreInfo::AnyType> {
    static SHOptionalString help() {
      return SHCCSTR("This shard flushes the log buffer to the console. This ensures that any pending log messages are "
                     "immediately written to the console.");
    }

    static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }

    static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }
  };

  struct LogChangeLevel : public LambdaShard<logsChangeLevelActivation, CoreInfo::StringType, CoreInfo::AnyType> {
    static SHOptionalString help() {
      return SHCCSTR("This shard changes the log level to the level specified by the string passed as input. ");
    }

    static SHOptionalString inputHelp() {
      return SHCCSTR("A string representing the new log level (e.g., 'debug', 'info', 'warn', 'error', 'critical').");
    }

    static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }
  };

  REGISTER_SHARD("FlushLog", LogFlush);
  REGISTER_SHARD("SetLogLevel", LogChangeLevel);
}
}; // namespace shards
