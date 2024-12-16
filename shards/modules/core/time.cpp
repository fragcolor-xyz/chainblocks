/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/gfx/moving_average.hpp>
#include "time.hpp"

namespace shards {
namespace Time {
struct Now {
  static inline ProcessClock _clock{};
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard outputs the amount of time that has elapsed since the shards application or script was launched in seconds.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the amount of time that has elapsed in seconds."); }
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto tnow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = tnow - _clock.Start;
    return Var(dt.count());
  }
};

struct NowMs : public Now {
  static SHOptionalString help() {
    return SHCCSTR("This shard outputs the amount of time that has elapsed since the shards application or script was launched "
                   "in milliseconds.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the amount of time that has elapsed in milliseconds."); }
  SHVar activate(SHContext *context, const SHVar &input) {
    auto tnow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = tnow - _clock.Start;
    return Var(dt.count());
  }
};

struct Delta {
  DeltaTimer _deltaTimer;

  static SHOptionalString help() {
    return SHCCSTR(R"(Outputs the time between the last call of this shard and the current call in seconds, capped to a limit)");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the amount of time that has elapsed in seconds."); }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }

  void warmup(SHContext *context) { _deltaTimer.reset(); }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    float dt = _deltaTimer.update();
    return Var{dt};
  }
};

struct DeltaMs : public Delta {
  static SHOptionalString help() {
    return SHCCSTR(
        "Outputs the time between the last call of this shard and the current call in milliseconds, capped to a limit");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the amount of time that has elapsed in milliseconds."); }

  SHVar activate(SHContext *context, const SHVar &input) {
    using Type = std::chrono::duration<double, std::milli>;
    return Var{_deltaTimer.update<Type>()};
  }
};

struct EpochMs {
  static SHOptionalString help() {
    return SHCCSTR("This shard outputs the the amount of time that has elapsed from the Unix epoch to the current system time in "
                   "milliseconds.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Amount of time since the Unix epoch in milliseconds."); }
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    using namespace std::chrono;
    milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    return Var(int64_t(ms.count()));
  }
};

struct Epoch {
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard outputs the the amount of time that has elapsed from the Unix epoch to the current system time in seconds.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Amount of time since the Unix epoch in seconds."); }
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    using namespace std::chrono;
    seconds ms = duration_cast<seconds>(system_clock::now().time_since_epoch());
    return Var(int64_t(ms.count()));
  }
};

struct EpochLocal {
  static SHOptionalString help() {
    return SHCCSTR("This shard outputs the amount of time that has elapsed from the Unix epoch to the current local system time "
                   "in seconds.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Amount of time since the Unix epoch in local time seconds."); }
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    using namespace std::chrono;
    auto now = system_clock::now();
    time_t tt = system_clock::to_time_t(now);

    // Get local time
    tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &tt);
    // Get UTC time
    tm utc_tm;
    gmtime_s(&utc_tm, &tt);
    // Calculate offset
    time_t local = mktime(&local_tm);
    time_t utc = mktime(&utc_tm);
    long offset = local - utc;
#else
    localtime_r(&tt, &local_tm);
    long offset = local_tm.tm_gmtoff;
#endif

    return Var(int64_t(tt + offset));
  }
};

struct EpochLocalMs {
  static SHOptionalString help() {
    return SHCCSTR("This shard outputs the amount of time that has elapsed from the Unix epoch to the current local system time "
                   "in milliseconds.");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Amount of time since the Unix epoch in local time milliseconds."); }
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::IntType; }

  SHVar activate(SHContext *context, const SHVar &input) {
    using namespace std::chrono;
    auto now = system_clock::now();
    time_t tt = system_clock::to_time_t(now);

    // Get local time
    tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &tt);
    // Get UTC time
    tm utc_tm;
    gmtime_s(&utc_tm, &tt);
    // Calculate offset
    time_t local = mktime(&local_tm);
    time_t utc = mktime(&utc_tm);
    long offset = local - utc;
#else
    localtime_r(&tt, &local_tm);
    long offset = local_tm.tm_gmtoff;
#endif

    auto ms = duration_cast<milliseconds>(now.time_since_epoch());
    return Var(int64_t(ms.count() + (offset * 1000)));
  }
};

struct ToString {
  static SHOptionalString help() { return SHCCSTR("This shard converts time into a human readable string."); }
  static SHTypesInfo inputTypes() { return _inputTypes; }
  static SHOptionalString inputHelp() { return SHCCSTR("The time to convert."); }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  static SHOptionalString outputHelp() { return SHCCSTR("A string representation of the time."); }

  SHParametersInfo parameters() { return _params; }

  void setParam(int index, const SHVar &value) { _isMillis = value.payload.boolValue; }

  SHVar getParam(int index) { return Var(_isMillis); }

  SHVar activate(SHContext *context, const SHVar &input) {

    switch (input.valueType) {
    case SHType::Int: {
      _output = _isMillis ? format(std::chrono::duration<int64_t, std::milli>(input.payload.intValue))
                          : format(std::chrono::duration<int64_t>(input.payload.intValue));
    } break;
    case SHType::Float: {
      _output = _isMillis ? format(std::chrono::duration<double, std::milli>(input.payload.floatValue))
                          : format(std::chrono::duration<double>(input.payload.floatValue));
    } break;
    default: {
      SHLOG_ERROR("Unexpected type for pure Time conversion: {}", type2Name(input.valueType));
      throw ActivationError("Type not supported for timeN conversion");
    }
    }

    return Var(_output);
  }

private:
  static inline Types _inputTypes = {{CoreInfo::IntType, CoreInfo::FloatType}};
  static inline Parameters _params{
      {"Millis", SHCCSTR("True if the input is given in milliseconds, False if given in seconds."), {CoreInfo::BoolType}}};

  bool _isMillis{false};
  std::string _output;

  template <typename T> inline std::string format(T timeunit) {
    using namespace std::chrono;

    milliseconds ms = duration_cast<milliseconds>(timeunit);
    std::ostringstream os;
    bool foundNonZero = false;
    os.fill('0');
    typedef duration<int, std::ratio<86400 * 365>> years;
    const auto y = duration_cast<years>(ms);
    if (y.count()) {
      foundNonZero = true;
      os << y.count() << "y:";
      ms -= y;
    }
    typedef duration<int, std::ratio<86400>> days;
    const auto d = duration_cast<days>(ms);
    if (d.count()) {
      foundNonZero = true;
      os << d.count() << "d:";
      ms -= d;
    }
    const auto h = duration_cast<hours>(ms);
    if (h.count() || foundNonZero) {
      foundNonZero = true;
      os << h.count() << "h:";
      ms -= h;
    }
    const auto m = duration_cast<minutes>(ms);
    if (m.count() || foundNonZero) {
      foundNonZero = true;
      os << m.count() << "m:";
      ms -= m;
    }
    const auto s = duration_cast<seconds>(ms);
#if !TIME_TOSTRING_PRINT_MILLIS
    os << s.count() << "s";
#else
    if (s.count() || foundNonZero) {
      foundNonZero = true;
      os << s.count() << "s:";
      ms -= s;
    }
    os << std::setw(3) << ms.count() << "ms";
#endif
    return os.str();
  }
};

struct MovingAverage {
  std::optional<gfx::MovingAverage<double>> _ma{};

  static SHOptionalString help() {
    return SHCCSTR("This shard computes the average of a floating point number over a specified number of frames.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("The floating point number to compute the average of."); }
  static SHOptionalString outputHelp() {
    return SHCCSTR("The average of the floating point number over the specified number of frames.");
  }
  static SHTypesInfo inputTypes() { return CoreInfo::FloatType; }
  static SHTypesInfo outputTypes() { return CoreInfo::FloatType; }

  PARAM_VAR(_windowSize, "Window", "The sample size in frames", {CoreInfo::IntType});
  PARAM_PARAMVAR(_clear, "Clear", "Set to true to clear the moving average", {CoreInfo::NoneType, CoreInfo::BoolVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_windowSize), PARAM_IMPL_FOR(_clear))

  MovingAverage() { _windowSize = Var(16); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return CoreInfo::FloatType;
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _ma.emplace(_windowSize.payload.intValue);
  }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!_clear.isNone() && _clear.get().payload.boolValue) {
      _ma->reset();
    }
    _ma->add(input.payload.floatValue);
    return Var(_ma->getAverage());
  }
};

} // namespace Time
SHARDS_REGISTER_FN(time) {
  REGISTER_SHARD("Time.Now", Time::Now);
  REGISTER_SHARD("Time.NowMs", Time::NowMs);
  REGISTER_SHARD("Time.Delta", Time::Delta);
  REGISTER_SHARD("Time.DeltaMs", Time::DeltaMs);
  REGISTER_SHARD("Time.EpochMs", Time::EpochMs);
  REGISTER_SHARD("Time.Epoch", Time::Epoch);
  REGISTER_SHARD("Time.EpochLocal", Time::EpochLocal);
  REGISTER_SHARD("Time.EpochLocalMs", Time::EpochLocalMs);
  REGISTER_SHARD("Time.ToString", Time::ToString);
  REGISTER_SHARD("Time.MovingAverage", Time::MovingAverage);
}
} // namespace shards
