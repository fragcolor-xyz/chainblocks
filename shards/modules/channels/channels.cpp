/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#include "channels.hpp"
#include <shards/core/runtime.hpp>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace shards {
namespace channels {

template <typename T> void verifyChannelType(T &channel, SHTypeInfo type, const char *name) {
  if (!matchTypes(type, channel.type, false, true, true)) {
    throw SHException(fmt::format("Attempted to change channel type: {}, desired type: {}, actual channel type: {}", name, type,
                                  (SHTypeInfo &)channel.type));
  }
}

template <typename T> T &getAndInitChannel(std::shared_ptr<Channel> &channel, SHTypeInfo type, bool noCopy, const char *name) {
  if (channel->index() == 0) {
    T &impl = channel->emplace<T>(noCopy);
    // no cloning here, this is potentially dangerous if the type is dynamic
    impl.type = type;
    return impl;
  } else if (T *impl = std::get_if<T>(channel.get())) {
    if (impl->type->basicType == SHType::None) {
      impl->type = type;
    } else {
      verifyChannelType(*impl, type, name);
    }
    return *impl;
  } else {
    throw SHException(fmt::format("MPMC Channel {} already initialized as another type of channel.", name));
  }
}

struct Base {
  std::string _name;
  bool _noCopy = false;
  OwnedVar _inType{};

  static inline Parameters producerParams{{"Name", SHCCSTR("The name of the channel."), {CoreInfo::StringType}},
                                          {"Type",
                                           SHCCSTR("The optional explicit (and unsafe because of that) we produce."),
                                           {CoreInfo::NoneType, CoreInfo::TypeType}},
                                          {"NoCopy!!",
                                           SHCCSTR("Unsafe flag that will improve performance by not copying "
                                                   "values when sending them thru the channel."),
                                           {CoreInfo::BoolType}}};

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0: {
      auto sv = SHSTRVIEW(value);
      _name = sv;
    } break;
    case 1:
      _inType = value;
      break;
    case 2: {
      _noCopy = value.payload.boolValue;
    } break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_name);
    case 1:
      return _inType;
    case 2:
      return Var(_noCopy);
    }
    return SHVar();
  }
};

struct Produce : public Base {
  std::shared_ptr<Channel> _channel;
  MPMCChannel *_mpChannel;

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() { return producerParams; }

  SHTypeInfo compose(const SHInstanceData &data) {
    _channel = get(_name);
    auto &receiverType = _inType.valueType == SHType::Type ? *_inType.payload.typeValue : data.inputType;
    _mpChannel = &getAndInitChannel<MPMCChannel>(_channel, receiverType, _noCopy, _name.c_str());
    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_mpChannel);

    SHVar tmp;
    // try to get from recycle bin
    // this might fail but we don't care
    // if it fails will just initialize a new variable
    if (!_mpChannel->recycle.pop(tmp)) // might modify tmp if fails
      tmp = SHVar();

    if (_noCopy) {
      tmp = input;
    } else {
      // this internally will reuse memory
      // yet it can be still slow for big vars
      cloneVar(tmp, input);
    }

    // enqueue for the stealing
    _mpChannel->data.push(tmp);

    return input;
  }
};

struct Broadcast : public Base {
  std::shared_ptr<Channel> _channel;
  BroadcastChannel *_bChannel;

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() { return producerParams; }

  SHTypeInfo compose(const SHInstanceData &data) {
    _channel = get(_name);
    auto &receiverType = _inType.valueType == SHType::Type ? *_inType.payload.typeValue : data.inputType;
    _bChannel = &getAndInitChannel<BroadcastChannel>(_channel, receiverType, _noCopy, _name.c_str());
    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_bChannel);

    // we need to support subscriptions during run-time
    // so we need to lock this operation
    // furthermore we allow multiple broadcasters so the erase needs this
    // but we use suspend instead of kernel locking!

    std::unique_lock<std::mutex> lock(_bChannel->subMutex, std::defer_lock);

    // try to lock, if we can't we suspend
    while (!lock.try_lock()) {
      SH_SUSPEND(context, 0);
    }

    // we locked it!
    for (auto it = _bChannel->subscribers.begin(); it != _bChannel->subscribers.end();) {
      if (it->closed) {
        it = _bChannel->subscribers.erase(it);
      } else {
        SHVar tmp;
        // try to get from recycle bin
        // this might fail but we don't care//
        // if it fails will just allocate a brand new
        if (!it->recycle.pop(tmp)) // might modify tmp if fails
          tmp = SHVar();

        if (_noCopy) {
          tmp = input;
        } else {
          // this internally will reuse memory
          cloneVar(tmp, input);
        }

        // enqueue for the stealing
        it->data.push(tmp);

        ++it;
      }
    }

    // we are done, exiting and going out of scope will unlock

    return input;
  }
};

struct BufferedConsumer {
  // utility to recycle memory and buffer
  // recycling is only for non blittable types basically
  std::vector<SHVar> buffer;

  void recycle(MPMCChannel *channel) {
    // send previous values to recycle
    for (auto &var : buffer) {
      channel->recycle.push(var);
    }
    buffer.clear();
  }

  void add(SHVar &var) { buffer.push_back(var); }

  bool empty() { return buffer.size() == 0; }

  operator SHVar() {
    auto len = buffer.size();
    assert(len > 0);
    if (len > 1) {
      SHVar res{};
      res.valueType = SHType::Seq;
      res.payload.seqValue.elements = &buffer[0];
      res.payload.seqValue.len = uint32_t(buffer.size());
      return res;
    } else {
      return buffer[0];
    }
  }
};

struct Consumers : public Base {
  std::shared_ptr<Channel> _channel;
  BufferedConsumer _storage;
  int64_t _bufferSize = 1;
  int64_t _current = 1;
  OwnedVar _outType{};
  SHTypeInfo _seqType{};

  static inline Parameters consumerParams{
      {"Name", SHCCSTR("The name of the channel."), {CoreInfo::StringType}},
      {"Type", SHCCSTR("The expected type to receive."), {CoreInfo::TypeType}},
      {"Buffer", SHCCSTR("The amount of values to buffer before outputting them."), {CoreInfo::IntType}},
  };

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() { return consumerParams; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _name = SHSTRVIEW(value);
      break;
    case 1:
      _outType = value;
      break;
    case 2:
      _bufferSize = value.payload.intValue;
      break;
    default:
      throw std::out_of_range("Invalid parameter index.");
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_name);
    case 1:
      return _outType;
    case 2:
      return Var(_noCopy);
    default:
      throw std::out_of_range("Invalid parameter index.");
    }
  }

  void cleanup(SHContext *context) {
    // reset buffer counter
    _current = _bufferSize;
  }
};

struct Consume : public Consumers {
  MPMCChannel *_mpChannel{};

  SHTypeInfo compose(const SHInstanceData &data) {
    auto outTypePtr = _outType.payload.typeValue;
    if (_outType->isNone() || outTypePtr->basicType == SHType::None) {
      throw std::logic_error("Consume: Type parameter is required.");
    }

    _channel = get(_name);
    _mpChannel = &getAndInitChannel<MPMCChannel>(_channel, *outTypePtr, _noCopy, _name.c_str());

    if (_bufferSize == 1) {
      return *outTypePtr;
    } else {
      _seqType.basicType = SHType::Seq;
      _seqType.seqTypes.elements = _outType.payload.typeValue;
      _seqType.seqTypes.len = 1;
      return _seqType;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_mpChannel);

    // send previous values to recycle
    _storage.recycle(_mpChannel);

    // reset buffer
    _current = _bufferSize;

    // blocking; and
    // everytime we are resumed we try to pop a value
    while (_current--) {
      SHVar output{};
      while (!_mpChannel->data.pop(output)) {
        // check also for channel completion
        if (_mpChannel->closed) {
          if (!_storage.empty()) {
            return _storage;
          } else {
            context->stopFlow(Var::Empty);
            return Var::Empty;
          }
        }
        SH_SUSPEND(context, 0);
      }

      // keep for recycling
      _storage.add(output);
    }

    return _storage;
  }

  void cleanup(SHContext *context) {
    Consumers::cleanup(context);

    // cleanup storage
    if (_mpChannel)
      _storage.recycle(_mpChannel);
  }
};

struct Listen : public Consumers {
  BroadcastChannel *_bChannel;
  MPMCChannel *_subscriptionChannel;

  void cleanup(SHContext *context) {
    Consumers::cleanup(context);

    // cleanup storage
    if (_subscriptionChannel) {
      _subscriptionChannel->closed = true;
      _storage.recycle(_subscriptionChannel);
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    auto outTypePtr = _outType.payload.typeValue;
    if (_outType->isNone() || outTypePtr->basicType == SHType::None) {
      throw std::logic_error("Listen: Type parameter is required.");
    }

    _channel = get(_name);
    _bChannel = &getAndInitChannel<BroadcastChannel>(_channel, *outTypePtr, _noCopy, _name.c_str());
    _subscriptionChannel = &_bChannel->subscribe();

    if (_bufferSize == 1) {
      return *outTypePtr;
    } else {
      _seqType.basicType = SHType::Seq;
      _seqType.seqTypes.elements = _outType.payload.typeValue;
      _seqType.seqTypes.len = 1;
      return _seqType;
    }
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_bChannel);
    assert(_subscriptionChannel);

    // send previous values to recycle
    _storage.recycle(_subscriptionChannel);

    // reset buffer
    _current = _bufferSize;

    // suspending; and
    // everytime we are resumed we try to pop a value
    while (_current--) {
      SHVar output{};
      while (!_subscriptionChannel->data.pop(output)) {
        // check also for channel completion
        if (_bChannel->closed) {
          if (!_storage.empty()) {
            return _storage;
          } else {
            context->stopFlow(Var::Empty);
            return Var::Empty;
          }
        }
        SH_SUSPEND(context, 0);
      }

      // keep for recycling
      _storage.add(output);
    }

    return _storage;
  }
};

struct Complete : public Base {
  std::shared_ptr<Channel> _channel;
  ChannelShared *_mpChannel{};

  static inline Parameters completeParams{
      {"Name", SHCCSTR("The name of the channel."), {CoreInfo::StringType}},
  };

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() { return completeParams; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _name = SHSTRVIEW(value);
      break;
    default:
      throw std::out_of_range("Invalid parameter index.");
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_name);
    default:
      throw std::out_of_range("Invalid parameter index.");
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _channel = get(_name);
    _mpChannel = std::visit(
        [&](auto &arg) {
          using T = std::decay_t<decltype(arg)>;
          if (std::is_same_v<T, DummyChannel>) {
            throw SHException("Expected a valid channel.");
          } else {
            return (ChannelShared *)&arg;
          }
        },
        *_channel.get());

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_mpChannel);

    if (_mpChannel->closed.exchange(true)) {
      SHLOG_INFO("Complete called on an already closed channel: {}", _name);
    }

    return input;
  }
};

std::shared_ptr<Channel> get(const std::string &name) {
  static std::unordered_map<std::string, std::weak_ptr<Channel>> channels;
  static std::shared_mutex mutex;

  std::shared_lock<decltype(mutex)> _l(mutex);
  auto it = channels.find(name);
  if (it == channels.end()) {
    _l.unlock();
    std::scoped_lock<decltype(mutex)> _l1(mutex);
    auto sp = std::make_shared<Channel>();
    channels[name] = sp;
    return sp;
  } else {
    std::shared_ptr<Channel> sp = it->second.lock();
    if (!sp) {
      _l.unlock();
      std::scoped_lock<decltype(mutex)> _l1(mutex);
      sp = std::make_shared<Channel>();
      channels[name] = sp;
    }
    return sp;
  }
}

// flush/cleanup a channel
struct Flush : public Base {
  std::shared_ptr<Channel> _channel;
  ChannelShared *_mpChannel{};

  static inline Parameters flushParams{
      {"Name", SHCCSTR("The name of the channel."), {CoreInfo::StringType}},
  };

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  static SHParametersInfo parameters() { return flushParams; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _name = SHSTRVIEW(value);
      break;
    default:
      throw std::out_of_range("Invalid parameter index.");
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(_name);
    default:
      throw std::out_of_range("Invalid parameter index.");
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    _channel = get(_name);
    _mpChannel = std::visit(
        [&](auto &arg) {
          using T = std::decay_t<decltype(arg)>;
          if (std::is_same_v<T, DummyChannel>) {
            throw SHException("Expected a valid channel.");
          } else {
            return (ChannelShared *)&arg;
          }
        },
        *_channel.get());

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_mpChannel);

    _mpChannel->clear();

    return input;
  }
};
} // namespace channels
} // namespace shards
SHARDS_REGISTER_FN(channels) {
  using namespace shards::channels;
  REGISTER_SHARD("Produce", Produce);
  REGISTER_SHARD("Broadcast", Broadcast);
  REGISTER_SHARD("Consume", Consume);
  REGISTER_SHARD("Listen", Listen);
  REGISTER_SHARD("Complete", Complete);
  REGISTER_SHARD("Flush", Flush);
}
