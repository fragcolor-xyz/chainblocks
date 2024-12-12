/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2023 Fragcolor Pte. Ltd. */

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/utility.hpp>

namespace shards {
namespace Events {
struct Base {
  std::optional<std::reference_wrapper<EventDispatcher>> _dispatcher;

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  PARAM(OwnedVar, _eventName, "Name", "The name of the event dispatcher to use.",
        {CoreInfo::StringType, CoreInfo::StringVarType});
  PARAM_PARAMVAR(_id, "ID", "The optional ID to use to differentiate events with the same name.",
                 {CoreInfo::IntType, CoreInfo::IntVarType, CoreInfo::NoneType});
  PARAM_IMPL(PARAM_IMPL_FOR(_eventName), PARAM_IMPL_FOR(_id));

  PARAM_REQUIRED_VARIABLES();

  SHTypeInfo compose(const SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    auto name = SHSTRVIEW(_eventName);
    _dispatcher = shards::getEventDispatcher(std::string(name));
    return data.inputType;
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }
};

struct Send : Base {
  SHTypeInfo compose(const SHInstanceData &data) {
    Base::compose(data);

    // when we send we store the type of the event
    // we store the type of the event
    (*_dispatcher).get().assignType(data.inputType);

    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_dispatcher);

    auto &dispatcher = _dispatcher->get();

    entt::id_type id;
    auto &idVar = _id.get();
    if (idVar.valueType == SHType::Int) {
      id = static_cast<entt::id_type>(idVar.payload.intValue);
    } else {
      id = findId(context);
    }

    OwnedVar ownedInput = input;
    if (id == entt::null)
      dispatcher->enqueue(std::move(ownedInput));
    else
      dispatcher->enqueue_hint(id, std::move(ownedInput));

    return input;
  }
};

struct Emit : Send {
  SHTypeInfo compose(const SHInstanceData &data) {
    auto dataCopy = data;
    dataCopy.inputType = CoreInfo::BoolType;
    Send::compose(dataCopy);
    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) { return Send::activate(context, Var(true)); }
};

struct Receive : Base {
  SeqVar _eventsIn;
  SeqVar _eventsOut;

  entt::connection _connection;

  Type singleType;
  Type outputType;

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnySeqType; }

  void onEvent(OwnedVar &event) { _eventsIn.push_back(event); }

  SHTypeInfo compose(const SHInstanceData &data) {
    Base::compose(data);

    // prevent none
    auto currentType = (*_dispatcher).get().getType();
    if (currentType.basicType == SHType::None) {
      SHLOG_ERROR("Event type not set for event: {}, use Events.Send first", _eventName);
      throw ComposeError("Event type not set");
    }

    // fixup type
    singleType = currentType;
    outputType = Type::SeqOf(singleType);

    return outputType;
  }

  void warmup(SHContext *context) { Base::warmup(context); }

  void cleanup(SHContext *context) {
    Base::cleanup(context);

    if (_connection)
      _connection.release();

    _prevId = Var::Empty;
  }

  SHVar _prevId{};

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &idVar = _id.get();
    if (!_connection || _prevId != idVar) {
      assert(_dispatcher);

      entt::id_type id;
      if (idVar.valueType == SHType::Int) {
        id = static_cast<entt::id_type>(idVar.payload.intValue);
      } else {
        id = findId(context);
      }

      if (_connection)
        _connection.release();

      if (id == entt::null)
        _connection = _dispatcher->get()->sink<OwnedVar>().connect<&Receive::onEvent>(this);
      else
        _connection = _dispatcher->get()->sink<OwnedVar>(id).connect<&Receive::onEvent>(this);
    }

    std::swap<SHVar>(_eventsOut, _eventsIn); // shallow swap is ok, since both are seqs of owned vars
    _eventsIn.clear();
    return Var(_eventsOut);
  }
};

struct Check : Receive {
  bool _triggered = false;

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return CoreInfo::BoolType; }

  SHTypeInfo compose(const SHInstanceData &data) {
    Base::compose(data);
    return CoreInfo::BoolType;
  }

  void onEvent(OwnedVar &event) { _triggered = true; }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &idVar = _id.get();
    if (!_connection || _prevId != idVar) {
      assert(_dispatcher);

      entt::id_type id;
      if (idVar.valueType == SHType::Int) {
        id = static_cast<entt::id_type>(idVar.payload.intValue);
      } else {
        id = findId(context);
      }

      if (_connection)
        _connection.release();

      _triggered = false;

      if (id == entt::null)
        _connection = _dispatcher->get()->sink<OwnedVar>().connect<&Check::onEvent>(this);
      else
        _connection = _dispatcher->get()->sink<OwnedVar>(id).connect<&Check::onEvent>(this);
    }

    if (_triggered) {
      _triggered = false;
      return Var(true);
    } else {
      return Var(false);
    }
  }
};

struct Update : Base {
  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_dispatcher);

    auto &dispatcher = _dispatcher->get();

    dispatcher->update();

    return input;
  }
};

} // namespace Events
} // namespace shards
SHARDS_REGISTER_FN(events) {
  using namespace shards::Events;
  REGISTER_SHARD("Events.Send", Send);
  REGISTER_SHARD("Events.Emit", Emit);
  REGISTER_SHARD("Events.Receive", Receive);
  REGISTER_SHARD("Events.Check", Check);
  REGISTER_SHARD("Events.Update", Update);
}
