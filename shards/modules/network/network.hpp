#ifndef DB94A8B0_9E08_4B91_B2D6_6F71AF52C15C
#define DB94A8B0_9E08_4B91_B2D6_6F71AF52C15C

#include <shards/core/platform.hpp>
#include <boost/core/span.hpp>
#include <shards/core/foundation.hpp>
#include <shards/shards.h>
#include <shards/shards.hpp>
#include <vector>
#include <memory>
#include "log.hpp"

namespace shards {
namespace Network {

constexpr uint32_t PeerCC = 'netP';
constexpr uint32_t ServerCC = 'netS';

struct Types {
  static inline Type Server{{SHType::Object, {.object = {.vendorId = CoreCC, .typeId = ServerCC}}}};
  static inline Type ServerVar = Type::VariableOf(Server);
  static inline Type Peer{{SHType::Object, {.object = {.vendorId = CoreCC, .typeId = PeerCC}}}};
  static inline Type PeerVar = Type::VariableOf(Peer);
  static inline Type PeerSeq = Type::SeqOf(Peer);
  static inline Type PeerSeqVar = Type::VariableOf(PeerSeq);
  static inline ParameterInfo PeerParameterInfo{"Peer", SHCCSTR("The optional explicit peer to send packets to."), {PeerVar}};
};

struct Reader {
  char *buffer;
  size_t offset;
  size_t max;

  Reader(char *buf, size_t size) : buffer(buf), offset(0), max(size) {}

  void operator()(uint8_t *buf, size_t size) {
    auto newOffset = offset + size;
    if (newOffset > max) {
      throw std::runtime_error("Overflow requested");
    }
    memcpy(buf, buffer + offset, size);
    offset = newOffset;
  }

  void deserializeInto(OwnedVar &output);
};

struct Writer {
  // let's make this growable on demand
  std::vector<uint8_t> buffer;
  size_t offset;

  Writer() {}

  void operator()(const uint8_t *buf, size_t size) {
    auto newOffset = offset + size;
    // resize buffer if needed but it (*2) so to avoid too many resizes
    if (newOffset > buffer.size()) {
      buffer.resize(newOffset * 2);
    }
    memcpy(buffer.data() + offset, buf, size);
    offset = newOffset;
  }

  void reset() {
    offset = 4; // we reserve 4 bytes for size
    buffer.resize(offset);
  }

  void finalize() {
    // write size
    *(uint32_t *)buffer.data() = offset;
  }

  // expose data and size
  const uint8_t *data() { return (uint8_t *)buffer.data(); }
  size_t size() { return offset; }

  boost::span<const uint8_t> varToSendBuffer(const SHVar &input);
};

Writer &getSendWriter();

struct Peer {
  Peer() { id = nextId.fetch_add(1, std::memory_order_relaxed); }
  virtual ~Peer() = default;

  virtual void send(boost::span<const uint8_t> data) = 0;
  virtual bool disconnected() const = 0;
  int64_t getId() { return id; }
  void sendVar(const SHVar &input) { send(getSendWriter().varToSendBuffer(input)); }

private:
  static inline std::atomic_int64_t nextId{1};
  int64_t id;
};

struct Server {
  virtual void broadcast(boost::span<const uint8_t> data, const SHVar &nodes, bool include) = 0;
  void broadcastVar(const SHVar &input, const SHVar &nodes, bool include) {
    broadcast(getSendWriter().varToSendBuffer(input), nodes, include);
  }
};

struct OnPeerConnected {
  std::weak_ptr<SHWire> wire;
};

struct OnPeerDisconnected {
  std::weak_ptr<SHWire> wire;
};

Peer &getConnectedPeer(ParamVar &peerParam);
void setDefaultPeerParam(ParamVar &peerParam);

Peer &getServer(ParamVar &serverParam);
void setDefaultServerParam(ParamVar &peerParam);

} // namespace Network
} // namespace shards

#endif /* DB94A8B0_9E08_4B91_B2D6_6F71AF52C15C */
