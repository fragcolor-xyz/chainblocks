/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

#include "network.hpp"

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>
#include <shards/core/wire_doppelganger_pool.hpp>
#include <shards/core/serialization.hpp>
#include <shards/utility.hpp>
#include <optional>
#include <boost/lockfree/queue.hpp>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <stdint.h>
#include <thread>
#include <unordered_map>
#include <utility>

namespace shards {
namespace Network {

Writer &getSendWriter() {
  static thread_local Writer inst;
  return inst;
}

void Reader::deserializeInto(OwnedVar &output) {
  static thread_local Serialization serializer;
  serializer.reset();
  serializer.deserialize(*this, output);
}

boost::span<const uint8_t> Writer::varToSendBuffer(const SHVar &input) {
  static thread_local Serialization serializer;

  reset();
  serializer.reset();
  serializer.serialize(input, *this);
  finalize();

  return boost::span(data(), size());
}

Peer &getConnectedPeer(ParamVar &peerParam) {
  Peer &peer = varAsObjectChecked<Peer>(peerParam.get(), Types::Peer);
  if (peer.disconnected()) {
    throw ActivationError("Peer is disconnected");
  }
  return peer;
}

void setDefaultPeerParam(ParamVar &peerParam) {
  peerParam = Var("Network.Peer");
  peerParam->valueType = SHType::ContextVar;
}

Peer &getServer(ParamVar &serverParam) {
  Peer &peer = varAsObjectChecked<Peer>(serverParam.get(), Types::Server);
  return peer;
}

void setDefaultServerParam(ParamVar &peerParam) {
  peerParam = Var("Network.Server");
  peerParam->valueType = SHType::ContextVar;
}

struct Broadcast {

  static SHOptionalString help() {
    return SHCCSTR("This shard sends the input to all peers connected to the server (created by Network.Server) specified in the Server parameter.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input to broadcast to all connected peers.");
  }

  static SHOptionalString outputHelp() {
    return DefaultHelpText::OutputHelpPass;
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  PARAM_PARAMVAR(_server, "Server", "The server to send the input to.", {Types::ServerVar});
  PARAM_PARAMVAR(_exclude, "Exclude", "The list of Peer IDs to exclude from the broadcast.", {CoreInfo::IntVarSeqType, CoreInfo::IntSeqType, CoreInfo::NoneType});
  PARAM_PARAMVAR(_include, "Include", "The list of Peer IDs to include in the broadcast. If set, only these peers will receive the broadcast.", {CoreInfo::IntVarSeqType, CoreInfo::IntSeqType, CoreInfo::NoneType});
  PARAM_IMPL(PARAM_IMPL_FOR(_server), PARAM_IMPL_FOR(_exclude), PARAM_IMPL_FOR(_include));

  Broadcast() {
    _server = Var("Network.Server");
    _server->valueType = SHType::ContextVar;
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  Server &getServer() { return varAsObjectChecked<Server>(_server.get(), Types::Server); }

  SHVar activate(SHContext *context, const SHVar &input) {
    auto &server = getServer();
    if (_include.get().valueType != SHType::None) {
      server.broadcastVar(input, _include.get(), true);
    } else {
      server.broadcastVar(input, _exclude.get(), false);
    }
    return input;
  }
};

struct Send {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends the input to the peer specified in the Peer parameter.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input to send to the peer.");
  }

  static SHOptionalString outputHelp() {
    return DefaultHelpText::OutputHelpPass;
  }

  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }

  PARAM_EXT(ParamVar, _peer, Types::PeerParameterInfo);
  PARAM_IMPL(PARAM_IMPL_FOR(_peer));

  Send() { setDefaultPeerParam(_peer); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &peer = getConnectedPeer(_peer);
    peer.sendVar(input);
    return input;
  }
};

struct SendRaw {
  static SHTypesInfo inputTypes() {
    static shards::Types types{{CoreInfo::StringType, CoreInfo::BytesType}};
    return types;
  }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("This shard sends the input byte array or string to the peer specified in the Peer parameter."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input to send to the peer.");
  }

  static SHOptionalString outputHelp() {
    return DefaultHelpText::OutputHelpPass;
  }

  PARAM_EXT(ParamVar, _peer, Types::PeerParameterInfo);
  PARAM_IMPL(PARAM_IMPL_FOR(_peer));

  SendRaw() { setDefaultPeerParam(_peer); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &peer = getConnectedPeer(_peer);
    // if (input.valueType == SHType::String) {
    //   peer.send(boost::span(input.payload.stringValue, input.payload.stringLen));
    // } else {
    peer.send(boost::span(input.payload.bytesValue, input.payload.bytesSize));
    // }
    return input;
  }
};

struct PeerID {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::IntType; }
  static SHOptionalString help() { return SHCCSTR("This shard outputs the Peer ID of the peer specified in the Peer parameter as an integer."); }

  static SHOptionalString inputHelp() {
    return DefaultHelpText::InputHelpIgnored;
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The Peer ID of the peer specified in the Peer parameter.");
  }

  static inline ParameterInfo PeerIDParameterInfo{"Peer", SHCCSTR("The Peer object to get the ID of."), {Types::PeerVar}};

  PARAM_EXT(ParamVar, _peer, PeerIDParameterInfo);
  PARAM_IMPL(PARAM_IMPL_FOR(_peer));

  PeerID() { setDefaultPeerParam(_peer); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &peer = getConnectedPeer(_peer);
    return Var(peer.getId());
  }
};

struct PeerShard {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return Types::Peer; }
  static SHOptionalString help() { return SHCCSTR("This shard outputs the peer object of the peer with the Peer ID specified in the Peer parameter."); }

  static SHOptionalString inputHelp() {
    return DefaultHelpText::InputHelpIgnored;
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("Outputs the Peer object specified.");
  }

  static inline ParameterInfo PeerParameInfo{"Peer", SHCCSTR("The Peer ID of the Peer object to get."), {Types::PeerVar}};

  PARAM_EXT(ParamVar, _peer, PeerParameInfo);
  PARAM_IMPL(PARAM_IMPL_FOR(_peer));

  PeerShard() { setDefaultPeerParam(_peer); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) { return _peer.get(); }
};
}; // namespace Network
}; // namespace shards

SHARDS_REGISTER_FN(network_common) {
  using namespace shards::Network;
  REGISTER_SHARD("Network.Broadcast", Broadcast);
  REGISTER_SHARD("Network.SendRaw", SendRaw);
  REGISTER_SHARD("Network.Send", Send);
  REGISTER_SHARD("Network.PeerID", PeerID);
  REGISTER_SHARD("Network.Peer", PeerShard);
}
