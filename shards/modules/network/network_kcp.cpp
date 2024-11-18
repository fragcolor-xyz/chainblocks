/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2019 Fragcolor Pte. Ltd. */

// ASIO must go first!!
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang attribute push(__attribute__((no_sanitize("undefined"))), apply_to = function)
#endif
#include <boost/asio.hpp>
#include <ikcp.h>
#ifdef __clang__
#pragma clang attribute pop
#pragma clang diagnostic pop
#endif

#include <tracy/Wrapper.hpp>
#include "network.hpp"
#include <shards/core/shared.hpp>
#include <shards/core/foundation.hpp>
#include <shards/shards.hpp>
#include <shards/shardwrapper.hpp>
#include <shards/core/shared.hpp>
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

#include "shards/core/async.hpp"
#include "log.hpp"

#define IKCP_MAX_PKT_SIZE 10000

namespace shards {
namespace Network {

static inline auto logger = getLogger();

using boost::asio::ip::udp;

struct NetworkContext {
  boost::asio::io_context _io_context;
  std::thread _io_context_thread;

  NetworkContext() {
    _io_context_thread = std::thread([this] {
      // Force run to run even without work
      boost::asio::executor_work_guard<boost::asio::io_context::executor_type> _guard = boost::asio::make_work_guard(_io_context);
      try {
        SPDLOG_LOGGER_DEBUG(logger, "Boost asio context running...");
        _io_context.run();
      } catch (...) {
        SPDLOG_LOGGER_ERROR(logger, "Boost asio context run failed.");
      }
      SPDLOG_LOGGER_DEBUG(logger, "Boost asio context exiting...");
    });
  }

  ~NetworkContext() {
    SPDLOG_LOGGER_TRACE(logger, "NetworkContext dtor");
    _io_context.stop(); // internally it will lock and send stop to all threads
    if (_io_context_thread.joinable())
      _io_context_thread.join();
  }
};

struct KCPPeer;

struct NetworkBase {
  SHVar *_peerVar = nullptr;

  ParamVar _addr{Var("localhost")};
  ParamVar _port{Var(9191)};

  AnyStorage<NetworkContext> _sharedNetworkContext;

  std::optional<udp::socket> _socket;
  std::mutex _socketMutex;

  ExposedInfo _required;
  SHTypeInfo compose(const SHInstanceData &data) {
    _required.clear();
    collectRequiredVariables(data.shared, _required, (SHVar &)_addr);
    collectRequiredVariables(data.shared, _required, (SHVar &)_port);
    return data.inputType;
  }

  SHExposedTypesInfo requiredVariables() { return SHExposedTypesInfo(_required); }

  void cleanup(SHContext *context) {
    if (_sharedNetworkContext) {
      auto &io_context = _sharedNetworkContext->_io_context;

      // defer all in the context or we will crash!
      if (_socket) {
        std::scoped_lock<std::mutex> l(_socketMutex);

        struct Container {
          udp::socket socket;
          Container(udp::socket &&s) : socket(std::move(s)) {}
        };

        boost::asio::post(io_context, [socket_ = std::make_unique<Container>(std::move(_socket.value()))]() {
          SPDLOG_LOGGER_TRACE(logger, "Closing socket");
          socket_->socket.close();
        });

        _socket.reset();
      }
    }

    // clean context vars
    if (_peerVar) {
      releaseVariable(_peerVar);
      _peerVar = nullptr;
    }

    _addr.cleanup();
    _port.cleanup();
  }

  void warmup(SHContext *context) {
    auto mesh = context->main->mesh.lock();
    _sharedNetworkContext = getOrCreateAnyStorage<NetworkContext>(mesh.get(), "Network.Context");

    _addr.warmup(context);
    _port.warmup(context);
  }

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }

  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  void setPeer(SHContext *context, KCPPeer &peer) {
    if (!_peerVar) {
      _peerVar = referenceVariable(context, "Network.Peer");
    }
    assignVariableValue(*_peerVar, Var::Object(&peer, CoreCC, PeerCC));
  }
};

struct KCPPeer final : public Peer {
  KCPPeer() {}

  ~KCPPeer() {
    if (kcp) {
      ikcp_release(kcp);
      kcp = nullptr;
    }
  }

  bool tryReceive(SHContext *context) {
    std::scoped_lock peerLock(mutex);

    if (networkError) {
      decltype(networkError) e;
      std::swap(e, networkError);
      throw std::runtime_error(fmt::format("Failed to receive: {}", e->message()));
    }

    auto now = SHClock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - _start).count();
    ikcp_update(kcp, ms);

    // Initialize variables for buffer management.
    size_t offset = recvBuffer.size();
    auto nextChunkSize = ikcp_peeksize(kcp);
    // if (nextChunkSize != -1)
    //   SPDLOG_LOGGER_TRACE(logger, "nextChunkSize: {}, endpoint: {} port: {}, offset: {}", nextChunkSize,
    //   endpoint->address().to_string(),
    //               endpoint->port(), offset);

    // Loop to receive all available chunks.
    while (nextChunkSize > 0) {
      // Resize the buffer to hold the incoming chunk.
      recvBuffer.resize(offset + nextChunkSize);

      // Receive the chunk.
      auto receivedSize = ikcp_recv(kcp, (char *)recvBuffer.data() + offset, nextChunkSize);
      shassert(receivedSize == nextChunkSize && "ikcp_recv returned unexpected size");

      // If this is the first chunk, read the expected total size from its prefix.
      if (offset == 0) {
        expectedSize = *(uint32_t *)recvBuffer.data();
        // reserve in advance as well or we will have to resize MANY TIMES later, which is slow
        recvBuffer.reserve(expectedSize);
      }

      // Check if the current buffer matches the expected size.
      if (recvBuffer.size() == expectedSize) {
        // SPDLOG_LOGGER_TRACE(logger, "Received full packet, endpoint: {} port: {}, size: {}", endpoint->address().to_string(),
        // endpoint->port(),
        //             expectedSize);
        return true;
      } else {
        // We expect another chunk; update the offset.
        offset = recvBuffer.size();
        nextChunkSize = ikcp_peeksize(kcp);
        // SPDLOG_LOGGER_TRACE(logger, "nextChunkSize (2): {}, endpoint: {} port: {}, offset: {}, expected: {}", nextChunkSize,
        //             endpoint->address().to_string(), endpoint->port(), offset, expectedSize);
      }
    }

    return false;
  }

  void endReceive() {
    // Reset the buffer.
    recvBuffer.clear();
    expectedSize = 0;
  }

  bool disconnected() const override { return disconnected_; }

  void send(boost::span<const uint8_t> data) override {
    std::scoped_lock lock(mutex); // prevent concurrent sends
    auto size = data.size();

    ZoneScopedN("Network::Send");
    TracyMessageL("Network::Send");

    if (size > IKCP_MAX_PKT_SIZE) {
      // send in chunks
      size_t chunks = size / IKCP_MAX_PKT_SIZE;
      size_t remaining = size % IKCP_MAX_PKT_SIZE;
      auto ptr = (char *)data.data();
      for (size_t i = 0; i < chunks; ++i) {
        auto err = ikcp_send(kcp, ptr, IKCP_MAX_PKT_SIZE);
        if (err < 0) {
          SPDLOG_LOGGER_ERROR(logger, "ikcp_send error: {}", err);
          throw ActivationError("ikcp_send error");
        }
        ptr += IKCP_MAX_PKT_SIZE;
      }
      if (remaining > 0) {
        auto err = ikcp_send(kcp, ptr, remaining);
        if (err < 0) {
          SPDLOG_LOGGER_ERROR(logger, "ikcp_send error: {}", err);
          throw ActivationError("ikcp_send error");
        }
      }
    } else {
      // directly send as it's small enough
      auto err = ikcp_send(kcp, (char *)data.data(), data.size());
      if (err < 0) {
        SPDLOG_LOGGER_ERROR(logger, "ikcp_send error: {}", err);
        throw ActivationError("ikcp_send error");
      }
    }
  }

  void reset() {
    if (kcp)
      ikcp_release(kcp);

    kcp = ikcp_create('shrd', this);

    // set "turbo" mode
    ikcp_nodelay(kcp, 1, 10, 2, 1);

    _start = SHClock::now();
    _lastContact = SHClock::now();
    disconnected_ = false;
    networkError.reset();
    recvBuffer.clear();
    expectedSize = 0;
  }

  std::optional<udp::endpoint> endpoint{};
  ikcpcb *kcp = nullptr;
  Serialization des{};
  OwnedVar payload{};

  SHTime _start = SHClock::now();
  std::atomic<SHTime> _lastContact = SHClock::now();

  void *user = nullptr;

  // used only when Server
  std::shared_ptr<SHWire> wire;
  std::optional<entt::connection> onStopConnection;

  std::mutex mutex;

  std::vector<uint8_t> recvBuffer;
  uint32_t expectedSize = 0;

  std::optional<boost::system::error_code> networkError;

  std::atomic_bool disconnected_ = false;
};

struct KCPServer : public Server {
  std::unordered_map<udp::endpoint, KCPPeer *> _end2Peer;
  std::unordered_map<const SHWire *, KCPPeer *> _wire2Peer;

  std::unordered_set<int64_t> _list;

  void broadcast(boost::span<const uint8_t> data, const SHVar &nodes, bool include) {
    if (nodes.valueType == SHType::Seq) {
      _list.clear();

      for (auto &included : nodes) {
        _list.insert(included.payload.intValue);
      }

      if (!include) {
        // excluding
        for (auto &[end, peer] : _end2Peer) {
          if (_list.find(peer->getId()) == _list.end()) {
            peer->send(data);
          }
        }
      } else {
        // including
        for (auto &[end, peer] : _end2Peer) {
          if (_list.find(peer->getId()) != _list.end()) {
            peer->send(data);
          }
        }
      }
    } else {
      // fast path
      for (auto &[end, peer] : _end2Peer) {
        peer->send(data);
      }
    }
  }
};

struct ServerShard : public NetworkBase {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return Types::Server; }

  static SHOptionalString help() {
    return SHCCSTR("This shard sets up a UDP network server using the KCP protocol, on the address and port specified in the "
                   "Address and Port parameters. This server then handles client connections and disconnections, messages "
                   "received from clients and broadcasting messages to these clients.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHOptionalString outputHelp() { return SHCCSTR("The server object created."); }

  struct Composer {
    ServerShard &server;

    Composer(ServerShard &server) : server(server) {}

    void compose(SHWire *wire, void *nothing, bool recycling) {
      if (recycling)
        return;

      SHInstanceData data{};
      data.inputType = CoreInfo::AnyType;
      data.shared = SHExposedTypesInfo(server._sharedCopy);
      data.wire = wire;
      wire->mesh = server._mesh;
      auto res = composeWire(wire, data);
      arrayFree(res.exposedInfo);
      arrayFree(res.requiredInfo);
    }
  } _composer{*this};

  std::shared_mutex peersMutex;
  udp::endpoint _sender;

  KCPServer _server;
  SHVar *_serverVar{nullptr};

  std::unique_ptr<WireDoppelgangerPool<KCPPeer>> _pool;
  OwnedVar _handlerMaster{};

  std::atomic<bool> _running{false};

  float _timeoutSecs = 30.0f;

  ShardsVar _disconnectionHandler{};

  boost::lockfree::queue<const SHWire *> _stopWireQueue{16};

  ExposedInfo _sharedCopy;

  std::shared_ptr<SHMesh> _mesh;

  static inline Parameters params{
      {"Address", SHCCSTR("The local bind address or the remote address."), {CoreInfo::StringOrStringVar}},
      {"Port", SHCCSTR("The port to bind if server or to connect to if client."), {CoreInfo::IntOrIntVar}},
      {"Handler",
       SHCCSTR("The wire to spawn for each new peer that connects, stopping that wire will break the connection."),
       {CoreInfo::WireOrNone}},
      {"Timeout",
       SHCCSTR("The timeout in seconds after which a peer will be disconnected if there is no network activity."),
       {CoreInfo::FloatType}},
      {"OnDisconnect",
       SHCCSTR("The shards to execute when a peer disconnects, The Peer ID will be the input."),
       {CoreInfo::ShardsOrNone}}};

  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _addr = value;
      break;
    case 1:
      _port = value;
      break;
    case 2:
      _handlerMaster = value;
      break;
    case 3:
      _timeoutSecs = value.payload.floatValue;
      break;
    case 4:
      _disconnectionHandler = value;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _addr;
    case 1:
      return _port;
    case 2:
      return _handlerMaster;
    case 3:
      return Var(_timeoutSecs);
    case 4:
      return _disconnectionHandler;
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_handlerMaster.valueType == SHType::Wire)
      _pool.reset(new WireDoppelgangerPool<KCPPeer>(_handlerMaster.payload.wireValue));

    // compose first with data the _disconnectionHandler if any
    if (_disconnectionHandler) {
      SHInstanceData dataCopy = data;
      dataCopy.inputType = CoreInfo::IntType;
      _disconnectionHandler.compose(dataCopy);
    }

    // inject our special context vars
    _sharedCopy = ExposedInfo(data.shared);
    auto endpointInfo = ExposedInfo::Variable("Network.Peer", SHCCSTR("The active peer."), Types::Peer);
    _sharedCopy.push_back(endpointInfo);

    NetworkBase::compose(data);

    return Types::Server;
  }

  void gcWires(SHContext *context) {
    if (!_stopWireQueue.empty()) {
      SHWire *toStop{};
      while (_stopWireQueue.pop(toStop)) {
        // read lock this
        std::shared_lock<std::shared_mutex> lock(peersMutex);
        auto it = _server._wire2Peer.find(toStop);
        if (it == _server._wire2Peer.end())
          continue; // Wire is not managed by this server
        KCPPeer *container = it->second;
        lock.unlock();

        SPDLOG_LOGGER_TRACE(logger, "GC-ing wire {}", toStop->name);
        DEFER({ SPDLOG_LOGGER_TRACE(logger, "GC-ed wire {}", toStop->name); });

        SPDLOG_LOGGER_TRACE(logger, "Clearing endpoint {}", container->endpoint->address().to_string());

        container->disconnected_ = true;

        // call this before cleanup
        if (_disconnectionHandler) {
          Var peerId(*reinterpret_cast<int64_t *>(&toStop->id));
          SHVar output{};
          _disconnectionHandler.activate(context, peerId, output);
        }

        // ensure cleanup is called early here
        auto wireState = toStop->state.load();
        if (wireState != SHWire::State::Failed && wireState != SHWire::State::Stopped) {
          // in the above cases cleanup already happened
          const_cast<SHWire *>(toStop)->cleanup();
        } else {
          SPDLOG_LOGGER_TRACE(logger, "Wire {} already stopped, state: {}", toStop->name, wireState);
        }

        // now fire events
        if (_mesh) {
          OnPeerDisconnected event{
              // .endpoint = *container->endpoint,
              .wire = container->wire,
          };
          _mesh->dispatcher.trigger(std::move(event));
        }

        // write lock it now
        std::scoped_lock<std::shared_mutex> lock2(peersMutex);
        _server._end2Peer.erase(*container->endpoint);
        _pool->release(container);
      }
    }
  }

  void warmup(SHContext *context) {
    if (!_pool) {
      throw ComposeError("Peer wires pool not valid!");
    }

    if (_disconnectionHandler)
      _disconnectionHandler.warmup(context);

    _mesh = context->main->mesh.lock();

    NetworkBase::warmup(context);

    setServer(context, &_server);

    _running.store(true, std::memory_order_release);
  }

  void cleanup(SHContext *context) {
    _running.store(false, std::memory_order_release);

    // Stop handlers
    if (_pool) {
      std::scoped_lock<std::shared_mutex> lock(peersMutex);
      SPDLOG_LOGGER_TRACE(logger, "Stopping all wires");
      _pool->stopAll();
      _server._end2Peer.clear();
      _server._wire2Peer.clear();
      _pool.reset();
    } else {
      SPDLOG_LOGGER_TRACE(logger, "No pool to stop");
    }

    // Shut down connection
    NetworkBase::cleanup(context);

    if (_serverVar) {
      releaseVariable(_serverVar);
      _serverVar = nullptr;
    }

    if (_pool) {
      SPDLOG_LOGGER_TRACE(logger, "Stopping all wires");
      _pool->stopAll();
    } else {
      SPDLOG_LOGGER_TRACE(logger, "No pool to stop");
    }

    _mesh.reset();

    NetworkBase::cleanup(context);

    gcWires(context);

    _disconnectionHandler.cleanup(context);
  }

  static int udp_output(const char *buf, int len, ikcpcb *kcp, void *user) {
    ZoneScopedN("Network::UDPOutput");

    KCPPeer *p = (KCPPeer *)user;
    ServerShard *s = (ServerShard *)p->user;

    std::scoped_lock<std::mutex> l(s->_socketMutex); // not ideal but for now we gotta do it

    s->_socket->async_send_to(boost::asio::buffer(buf, len), *p->endpoint,
                              [](boost::system::error_code ec, std::size_t bytes_sent) {
                                TracyMessageL("Network::async_send_to (udp)");
                                if (ec) {
                                  // ignore flow-control No buffer space available
                                  if (ec != boost::asio::error::no_buffer_space && ec != boost::asio::error::would_block &&
                                      ec != boost::asio::error::try_again) {
                                    SPDLOG_LOGGER_ERROR(logger, "Error sending: {}", ec.message());
                                  } else {
                                    SPDLOG_LOGGER_DEBUG(logger, "Error sending (ignored): {}", ec.message());
                                  }
                                }
                              });
    return 0;
  }

  void wireOnStop(const SHWire::OnStopEvent &e) {
    SPDLOG_LOGGER_TRACE(logger, "Wire {} stopped", e.wire->name);

    _stopWireQueue.push(e.wire);
  }

  void do_receive() {
    thread_local std::vector<uint8_t> recv_buffer(0xFFFF);

    std::scoped_lock<std::mutex> l(_socketMutex); // not ideal but for now we gotta do it

    _socket->async_receive_from(boost::asio::buffer(recv_buffer.data(), recv_buffer.size()), _sender,
                                [this](boost::system::error_code ec, std::size_t bytes_recvd) {
                                  TracyMessageL("Network::async_receive_from (udp)");
                                  if (!ec && bytes_recvd > 0) {
                                    KCPPeer *currentPeer = nullptr;
                                    std::shared_lock<std::shared_mutex> lock(peersMutex);
                                    auto it = _server._end2Peer.find(_sender);
                                    if (it == _server._end2Peer.end()) {
                                      SPDLOG_LOGGER_TRACE(logger, "Received packet from unknown peer: {} port: {}",
                                                          _sender.address().to_string(), _sender.port());

                                      // new peer
                                      lock.unlock();

                                      // we write so hard lock this
                                      std::scoped_lock<std::shared_mutex> lock(peersMutex);

                                      // new peer
                                      try {
                                        auto peer = _pool->acquire(_composer, (void *)0);
                                        peer->reset();
                                        _server._end2Peer[_sender] = peer;
                                        peer->endpoint = _sender;
                                        peer->user = this;
                                        peer->kcp->user = peer;
                                        peer->kcp->output = &ServerShard::udp_output;
                                        SPDLOG_LOGGER_DEBUG(logger, "Added new peer: {} port: {}",
                                                            peer->endpoint->address().to_string(), peer->endpoint->port());

                                        // Assume that we recycle containers so the connection might already exist!
                                        _server._wire2Peer[peer->wire.get()] = peer;

                                        // set wire ID, in order for Events to be properly routed
                                        // for now we just use ptr as ID, until it causes problems
                                        peer->wire->id = static_cast<entt::id_type>(peer->getId());

                                        currentPeer = peer;
                                      } catch (std::exception &e) {
                                        SPDLOG_LOGGER_ERROR(logger, "Error acquiring peer: {}", e.what());

                                        // keep receiving
                                        if (_socket && _running.load(std::memory_order_acquire))
                                          return do_receive();
                                      }
                                    } else {
                                      // SPDLOG_LOGGER_TRACE(logger, "Received packet from known peer: {} port: {}",
                                      // _sender.address().to_string(), _sender.port());

                                      // existing peer
                                      currentPeer = it->second;

                                      lock.unlock();
                                    }

                                    {
                                      std::scoped_lock pLock(currentPeer->mutex);

                                      auto err = ikcp_input(currentPeer->kcp, (char *)recv_buffer.data(), bytes_recvd);
                                      // SPDLOG_LOGGER_TRACE(logger, "ikcp_input: {}, peer: {} port: {}, size: {}", err,
                                      // _sender.address().to_string(), _sender.port(),
                                      //             bytes_recvd);
                                      if (err < 0) {
                                        SPDLOG_LOGGER_ERROR(logger, "Error ikcp_input: {}, peer: {} port: {}", err,
                                                            _sender.address().to_string(), _sender.port());
                                        _stopWireQueue.push(currentPeer->wire.get());
                                      }

                                      currentPeer->_lastContact = SHClock::now();
                                    }

                                    // keep receiving
                                    if (_socket && _running.load(std::memory_order_acquire)) {
                                      return do_receive();
                                    } else {
                                      SPDLOG_LOGGER_DEBUG(logger, "Socket closed, stopping receive loop");
                                    }
                                  } else {
                                    if (ec == boost::asio::error::operation_aborted) {
                                      // we likely have invalid data under the hood, let's just ignore it
                                      SPDLOG_LOGGER_DEBUG(logger, "Operation aborted");
                                      return;
                                    } else if (ec == boost::asio::error::no_buffer_space ||
                                               ec == boost::asio::error::would_block || ec == boost::asio::error::try_again) {
                                      SPDLOG_LOGGER_DEBUG(logger, "Ignored error while receiving: {}", ec.message());
                                      return do_receive();
                                    }

                                    SPDLOG_LOGGER_DEBUG(logger, "Error receiving: {}, peer: {} port: {}", ec.message(),
                                                        _sender.address().to_string(), _sender.port());

                                    std::shared_lock<std::shared_mutex> lock(peersMutex);
                                    auto it = _server._end2Peer.find(_sender);
                                    if (it != _server._end2Peer.end()) {
                                      SPDLOG_LOGGER_TRACE(logger, "Removing peer: {} port: {}", _sender.address().to_string(),
                                                          _sender.port());
                                      _stopWireQueue.push(it->second->wire.get());
                                    }

                                    // keep receiving
                                    if (_socket && _running.load(std::memory_order_acquire)) {
                                      return do_receive();
                                    } else {
                                      SPDLOG_LOGGER_DEBUG(logger, "Socket closed, stopping receive loop");
                                    }
                                  }
                                });
  }

  void setServer(SHContext *context, KCPServer *server) {
    if (!_serverVar) {
      _serverVar = referenceVariable(context, "Network.Server");
    }
    assignVariableValue(*_serverVar, Var::Object(server, CoreCC, ServerCC));
  }

  SHExposedTypesInfo exposedVariables() {
    static std::array<SHExposedTypeInfo, 1> exposing;
    exposing[0].name = "Network.Server";
    exposing[0].help = SHCCSTR("The exposed server.");
    exposing[0].exposedType = Types::Server;
    exposing[0].isProtected = true;
    return {exposing.data(), 1, 0};
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_sharedNetworkContext);
    auto &io_context = _sharedNetworkContext->_io_context;

    if (!_socket) {
      // first activation, let's init
      _socket.emplace(io_context);
      _socket->open(udp::v4());
      _socket->set_option(boost::asio::ip::udp::socket::reuse_address(true));
      _socket->set_option(boost::asio::socket_base::send_buffer_size(65536));
      _socket->set_option(boost::asio::socket_base::receive_buffer_size(65536));
      _socket->bind(udp::endpoint(udp::v4(), _port.get().payload.intValue));

      // start receiving
      boost::asio::post(io_context, [this]() { do_receive(); });

      SPDLOG_LOGGER_TRACE(logger, "Network.Server listening on port {}", _port.get().payload.intValue);
    }

    gcWires(context);

    {
      std::shared_lock<std::shared_mutex> lock(peersMutex);

      auto now = SHClock::now();

      for (auto &[end, peer] : _server._end2Peer) {
        if (now > (peer->_lastContact.load() + SHDuration(_timeoutSecs))) {
          SPDLOG_LOGGER_DEBUG(logger, "Peer {}:{} timed out", peer->endpoint->address().to_string(), peer->endpoint->port());
          _stopWireQueue.push(peer->wire.get());
          continue;
        }

        setPeer(context, *peer);

        if (!peer->onStopConnection) {
          peer->onStopConnection =
              peer->wire->mesh.lock()->dispatcher.sink<SHWire::OnStopEvent>().connect<&ServerShard::wireOnStop>(this);
        }

        if (!peer->wire->warmedUp) {
          OnPeerConnected event{
              // .endpoint = *peer->endpoint,
              .wire = peer->wire,
          };
          context->main->mesh.lock()->dispatcher.trigger(std::move(event));

          // warmup the wire after the event in order to give it a chance to set up
          peer->wire->warmup(context);
        }

        if (peer->tryReceive(context)) {
          ZoneScopedN("Network::Receive");
          TracyMessageL("Network::Receive");
          auto &peer_ = peer; // avoid c++20 ext warning
          DEFER({
            // we need to ensure this gets called at all times
            peer_->endReceive();
          });

          if (!context->shouldContinue())
            return input;

          try {
            if (peer_->recvBuffer.size() > IKCP_MAX_PKT_SIZE) {
              // do this async as it's a big buffer
              await(
                  context,
                  [peer_]() {
                    // deserialize from buffer on top of the vector of payloads, wires might consume them out of band
                    Reader r((char *)peer_->recvBuffer.data() + 4, peer_->recvBuffer.size() - 4);
                    peer_->des.reset();
                    peer_->des.deserialize(r, peer_->payload);
                  },
                  [] {});
            } else {
              // deserialize from buffer on top of the vector of payloads, wires might consume them out of band
              Reader r((char *)peer_->recvBuffer.data() + 4, peer_->recvBuffer.size() - 4);
              peer_->des.reset();
              peer_->des.deserialize(r, peer_->payload);
            }

            // Run within the root flow
            auto runRes = runSubWire(peer->wire.get(), context, peer->payload);
            if (unlikely(runRes.state == SHRunWireOutputState::Failed || runRes.state == SHRunWireOutputState::Stopped ||
                         runRes.state == SHRunWireOutputState::Returned)) {
              stop(peer->wire.get());
            }
          } catch (std::exception &e) {
            SPDLOG_LOGGER_ERROR(logger, "Critical errors processing peer {}: {}, disconnecting it",
                                peer->endpoint->address().to_string(), e.what());
            stop(peer->wire.get());
          }
          // Always adjust the context back to continue, peer wire might have changed it
          context->resetErrorStack();
          context->continueFlow();
        }
      }
    }

    return *_serverVar;
  }
};

struct ClientShard : public NetworkBase {
  std::array<SHExposedTypeInfo, 1> _exposing;

  static SHOptionalString help() {
    return SHCCSTR("This shard creates a UDP client connection using the KCP protocol, on the address and port specified in the "
                   "Address and Port parameters.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHOptionalString outputHelp() { return SHCCSTR("The peer object created."); }

  SHExposedTypesInfo exposedVariables() {
    _exposing[0].name = "Network.Peer";
    _exposing[0].help = SHCCSTR("The exposed peer.");
    _exposing[0].exposedType = Types::Peer;
    _exposing[0].isProtected = true;
    return {_exposing.data(), 1, 0};
  }

  static inline Type PeerType{{SHType::Object, {.object = {.vendorId = CoreCC, .typeId = PeerCC}}}};
  static inline Type PeerObjectType = Type::VariableOf(PeerType);

  static SHTypesInfo outputTypes() { return PeerType; }

  KCPPeer _peer;
  ShardsVar _blks{};
  udp::endpoint _server;

  SHVar *_peerVarRef{};

  static inline Parameters params{
      {"Address", SHCCSTR("The local bind address or the remote address."), {CoreInfo::StringOrStringVar}},
      {"Port", SHCCSTR("The port to bind if server or to connect to if client."), {CoreInfo::IntOrIntVar}},
      {"Handler", SHCCSTR("The shards to execute when a packet is received."), {CoreInfo::ShardsOrNone}}};

  static SHParametersInfo parameters() { return SHParametersInfo(params); }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _addr = value;
      break;
    case 1:
      _port = value;
      break;
    case 2:
      _blks = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return _addr;
    case 1:
      return _port;
    case 2:
      return _blks;
    default:
      return Var::Empty;
    }
  }

  static int udp_output(const char *buf, int len, ikcpcb *kcp, void *user) {
    ClientShard *c = (ClientShard *)user;

    std::scoped_lock<std::mutex> l(c->_socketMutex); // not ideal but for now we gotta do it

    SPDLOG_LOGGER_TRACE(logger, "asio> queueing sending {} bytes", len);
    c->_socket->async_send_to(boost::asio::buffer(buf, len), c->_server,
                              [c](boost::system::error_code ec, std::size_t bytes_sent) {
                                if (ec) {
                                  // ignore flow-control No buffer space available
                                  if (ec != boost::asio::error::no_buffer_space && ec != boost::asio::error::would_block &&
                                      ec != boost::asio::error::try_again) {
                                    SPDLOG_LOGGER_ERROR(logger, "Error sending: {}", ec.message());
                                    c->_peer.networkError = ec;
                                  } else {
                                    SPDLOG_LOGGER_DEBUG(logger, "Error sending (ignored): {}", ec.message());
                                  }
                                } else {
                                  SPDLOG_LOGGER_TRACE(logger, "asio> sent {} bytes", bytes_sent);
                                }
                              });

    return 0;
  }

  void setup() {
    // new peer
    _peer.reset();
    _peer.kcp->user = this;
    _peer.kcp->output = &ClientShard::udp_output;
  }

  void do_receive() {
    thread_local std::vector<uint8_t> recv_buffer(0xFFFF);

    std::scoped_lock<std::mutex> l(_socketMutex); // not ideal but for now we gotta do it

    _socket->async_receive_from(boost::asio::buffer(recv_buffer.data(), recv_buffer.size()), _server,
                                [this](boost::system::error_code ec, std::size_t bytes_recvd) {
                                  if (ec) {
                                    // certain errors are expected, ignore them
                                    if (ec == boost::asio::error::no_buffer_space || ec == boost::asio::error::would_block ||
                                        ec == boost::asio::error::try_again) {
                                      SPDLOG_LOGGER_DEBUG(logger, "Ignored error while receiving: {}", ec.message());
                                      return do_receive(); // continue receiving
                                    } else if (ec == boost::asio::error::operation_aborted) {
                                      SPDLOG_LOGGER_ERROR(logger, "Socket aborted receiving: {}", ec.message());
                                      // _peer might be invalid at this point!
                                    } else {
                                      SPDLOG_LOGGER_ERROR(logger, "Error receiving: {}", ec.message());
                                      _peer.networkError = ec;
                                    }
                                  } else {
                                    if (bytes_recvd > 0) {
                                      std::scoped_lock lock(_peer.mutex);

                                      auto err = ikcp_input(_peer.kcp, (char *)recv_buffer.data(), bytes_recvd);
                                      if (err < 0) {
                                        SPDLOG_LOGGER_ERROR(logger, "Error ikcp_input: {}");
                                      }
                                    }
                                    // keep receiving
                                    do_receive();
                                  }
                                });
  }

  SHTypeInfo compose(SHInstanceData &data) {
    NetworkBase::compose(data);
    // inject our special context vars
    auto endpointInfo = ExposedInfo::Variable("Network.Peer", SHCCSTR("The active peer."), Types::Peer);
    ExposedInfo shared{data.shared};
    shared.push_back(endpointInfo);
    SHInstanceData innerData{data};
    innerData.shared = (SHExposedTypesInfo)shared;
    _blks.compose(innerData);
    return PeerType;
  }

  void cleanup(SHContext *context) {
    NetworkBase::cleanup(context);
    _blks.cleanup(context);

    if (_peerVarRef) {
      releaseVariable(_peerVarRef);
      _peerVarRef = nullptr;
    }
  }

  void warmup(SHContext *context) {
    NetworkBase::warmup(context);
    _blks.warmup(context);
    _peerVarRef = referenceVariable(context, "Network.Peer");
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_sharedNetworkContext);
    auto &io_context = _sharedNetworkContext->_io_context;

    if (!_socket) {
      setup(); // reset the peer

      // first activation, let's init
      _socket.emplace(io_context, udp::endpoint(udp::v4(), 0));
      boost::asio::socket_base::send_buffer_size option_send(65536);
      boost::asio::socket_base::receive_buffer_size option_recv(65536);
      _socket->set_option(option_send);
      _socket->set_option(option_recv);

      boost::asio::io_service io_service;
      udp::resolver resolver(io_service);
      auto sport = std::to_string(_port.get().payload.intValue);
      udp::resolver::query query(udp::v4(), SHSTRING_PREFER_SHSTRVIEW(_addr.get()), sport);
      _server = *resolver.resolve(query);
      _peer.endpoint = _server;

      // start receiving
      boost::asio::post(io_context, [this]() { do_receive(); });
    }

    assignVariableValue(*_peerVarRef, Var::Object(&_peer, CoreCC, PeerCC));

    if (_peer.tryReceive(context)) {
      if (!context->shouldContinue())
        return Var::Empty;

      if (_peer.recvBuffer.size() > IKCP_MAX_PKT_SIZE) {
        // do this async as it's a big buffer
        await(
            context,
            [&]() {
              // deserialize from buffer on top of the vector of payloads, wires might consume them out of band
              Reader r((char *)_peer.recvBuffer.data() + 4, _peer.recvBuffer.size() - 4);
              _peer.des.reset();
              _peer.des.deserialize(r, _peer.payload);
            },
            [] {});
      } else {
        // deserialize from buffer on top of the vector of payloads, wires might consume them out of band
        Reader r((char *)_peer.recvBuffer.data() + 4, _peer.recvBuffer.size() - 4);
        _peer.des.reset();
        _peer.des.deserialize(r, _peer.payload);
      }

      // at this point we can already cleanup the buffer
      _peer.endReceive();

      SHVar output{};
      activateShards(SHVar(_blks).payload.seqValue, context, _peer.payload, output);
      // no need to handle errors as context will eventually deal with it after activation
    }

    return Var::Object(&_peer, CoreCC, PeerCC);
  }
};

}; // namespace Network
}; // namespace shards

SHARDS_REGISTER_FN(network_kcp) {
  using namespace shards::Network;
  REGISTER_SHARD("Network.Server", ServerShard);
  REGISTER_SHARD("Network.Client", ClientShard);
}
