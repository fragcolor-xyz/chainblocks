/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#include <boost/core/detail/string_view.hpp>
#include <shards/core/params.hpp>
#include <shards/core/platform.hpp>
#include <shards/log/log.hpp>

#if !SH_EMSCRIPTEN
#define BOOST_ERROR_CODE_HEADER_ONLY
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
namespace beast = boost::beast; // from <boost/beast.hpp>
namespace http = beast::http;   // from <boost/beast/http.hpp>
namespace net = boost::asio;    // from <boost/asio.hpp>
using tcp = net::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

#include <cctype>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>
#else
#include <boost/algorithm/string.hpp>
#include <emscripten/fetch.h>
#endif

#include <shards/core/shared.hpp>
#include <shards/core/wire_doppelganger_pool.hpp>

using namespace std;

static auto logger = shards::logging::getOrCreate("http");

inline std::string url_encode(const std::string_view &value) {
  std::ostringstream escaped;
  escaped.fill('0');
  escaped << hex;

  for (auto i = value.begin(), n = value.end(); i != n; ++i) {
    std::string::value_type c = (*i);

    // Keep alphanumeric and other accepted characters intact
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      escaped << c;
      continue;
    }

    // Any other characters are percent-encoded
    escaped << std::uppercase;
    escaped << '%' << std::setw(2) << int((unsigned char)c);
    escaped << std::nouppercase;
  }

  return escaped.str();
}

inline std::string url_decode(const std::string_view &value) {
  std::string decoded;
  decoded.reserve(value.size());

  for (std::size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '%') {
      if (i + 3 <= value.size()) {
        int v = 0;
        std::istringstream hex_stream(std::string(value.substr(i + 1, 2)));
        hex_stream >> std::hex >> v;
        decoded += static_cast<char>(v);
        i += 2;
      } else {
        throw std::runtime_error("Invalid URL encoding");
      }
    } else if (value[i] == '+') {
      decoded += ' ';
    } else {
      decoded += value[i];
    }
  }

  return decoded;
}

namespace shards {
namespace Http {
#if SH_EMSCRIPTEN
// those shards for non emscripten platforms are implemented in Rust

struct Base {
  static inline Types FullStrOutputTypes{{CoreInfo::IntType, CoreInfo::StringTableType, CoreInfo::StringType}};
  static inline Types FullBytesOutputTypes{{CoreInfo::IntType, CoreInfo::StringTableType, CoreInfo::BytesType}};
  static inline std::array<SHVar, 3> FullOutputKeys{Var("status"), Var("headers"), Var("body")};
  static inline Type FullStrOutputType = Type::TableOf(FullStrOutputTypes, FullOutputKeys);
  static inline Type FullBytesOutputType = Type::TableOf(FullBytesOutputTypes, FullOutputKeys);
  static inline Types AllOutputTypes{{FullBytesOutputType, CoreInfo::BytesType, FullStrOutputType, CoreInfo::StringType}};

  static inline Parameters params{
      {"URL", SHCCSTR("The url to request to"), {CoreInfo::StringType, CoreInfo::StringVarType}},
      {"Headers",
       SHCCSTR("The headers to use for the request."),
       {CoreInfo::NoneType, CoreInfo::StringTableType, CoreInfo::StringVarTableType}},
      {"Timeout", SHCCSTR("How many seconds to wait for the request to complete."), {CoreInfo::IntType}},
      {"Bytes", SHCCSTR("If instead of a string the shard should output bytes."), {CoreInfo::BoolType}},
      {"FullResponse",
       SHCCSTR("If the output should be a table with the full response, "
               "including headers and status."),
       {CoreInfo::BoolType}},
      {"AcceptInvalidCerts", SHCCSTR("Not implemented."), {CoreInfo::NoneType, CoreInfo::BoolType}},
      {"Retry", SHCCSTR("How many times to retry the request if it fails."), {CoreInfo::NoneType, CoreInfo::IntType}},
      {"KeepAlive",
       SHCCSTR("If the client instance should be kept alive, allowing connection reuse for multiple requests. The client won't "
               "be closed until this shard cleans up (including its worker thread)."),
       {CoreInfo::BoolType}},
      {"Streaming", SHCCSTR("If the output should be a stream of bytes."), {CoreInfo::BoolType}}};
  static SHParametersInfo parameters() { return params; }

  SHTypesInfo outputTypes() { return AllOutputTypes; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      url = value;
      break;
    case 1:
      headers = value;
      break;
    case 2:
      timeout = int(value.payload.intValue);
      break;
    case 3:
      asBytes = value.payload.boolValue;
      break;
    case 4:
      fullResponse = value.payload.boolValue;
      if (fullResponse) {
        outMap["headers"] = TableVar{};
      }
      break;
    case 5:
      break;
    case 6:
      numRetries = value.payload.intValue;
    case 7:
      break; // unused
    case 8:
      break; // unused
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return url;
    case 1:
      return headers;
    case 2:
      return Var(timeout);
    case 3:
      return Var(asBytes);
    case 4:
      return Var(fullResponse);
    case 5:
      return Var(false);
    case 6:
      return Var(numRetries);
    case 7:
      return Var(false);
    case 8:
      return Var(false);
    default:
      throw InvalidParameterIndex();
    }
  }

  void warmup(SHContext *context) {
    url.warmup(context);
    headers.warmup(context);
  }

  void cleanup(SHContext *context) {
    url.cleanup();
    headers.cleanup();
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (fullResponse) {
      if (asBytes) {
        return FullBytesOutputType;
      } else {
        return FullStrOutputType;
      }
    } else {
      if (asBytes) {
        return CoreInfo::BytesType;
      } else {
        return CoreInfo::StringType;
      }
    }
  }

  static void fetchSucceeded(emscripten_fetch_t *fetch) {
    SPDLOG_LOGGER_DEBUG(logger, "Fetch succeeded {} (s: {})", (void *)fetch, fetch->status);
    auto self = reinterpret_cast<Base *>(fetch->userData);
    if (self->fullResponse) {
      const auto len = emscripten_fetch_get_response_headers_length(fetch);
      self->hbuffer.resize(len + 1); // well I think the + 1 is not needed
      emscripten_fetch_get_response_headers(fetch, self->hbuffer.data(), len + 1);
      auto &mvar = self->outMap.get<TableVar>("headers");
      std::istringstream resp(self->hbuffer);
      std::string header;
      std::string::size_type index;
      while (std::getline(resp, header) && header != "\r") {
        index = header.find(':', 0);
        if (index != std::string::npos) {
          auto key = boost::algorithm::trim_copy(header.substr(0, index));
          boost::algorithm::to_lower(key);
          auto val = boost::algorithm::trim_copy(header.substr(index + 1));
          mvar[key] = Var(val);
        }
      }
      self->status = fetch->status;
    }
    self->buffer.assign(fetch->data, fetch->numBytes);
    emscripten_fetch_close(fetch);
    self->state = 1;
  }

  static void fetchFailed(emscripten_fetch_t *fetch) {
    SPDLOG_LOGGER_DEBUG(logger, "Fetch failed {} (s: {})", (void *)fetch, fetch->status);
    auto self = reinterpret_cast<Base *>(fetch->userData);
    self->buffer.assign(fetch->statusText);
    emscripten_fetch_close(fetch); // Also free data on failure.
    self->state = -1;
  }

  bool asBytes{false};
  bool fullResponse{false};
  uint16_t status;
  std::atomic_int state{0};
  int timeout{10};
  int numRetries{0};
  std::string buffer;
  std::string hbuffer;
  std::string urlBuffer;
  std::vector<const char *> headersCArray;
  TableVar outMap;
  ParamVar url{Var("")};
  ParamVar headers{};
};

template <const string_view &METHOD> struct GetLike : public Base {
  static inline Types InputTypes{{CoreInfo::NoneType, CoreInfo::StringTableType}};
  static SHTypesInfo inputTypes() { return InputTypes; }

  SHVar activate(SHContext *context, const SHVar &input) {
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, METHOD.data());
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = fetchSucceeded;
    attr.onerror = fetchFailed;
    attr.userData = this;
    attr.timeoutMSecs = timeout * 1000;

    urlBuffer.clear();
    urlBuffer.append(url.get().payload.stringValue);
    if (input.valueType == SHType::Table) {
      urlBuffer.append("?");
      ForEach(input.payload.tableValue, [&](auto key, auto &value) {
        urlBuffer.append(url_encode(SHSTRVIEW(key)));
        urlBuffer.append("=");
        urlBuffer.append(url_encode(SHSTRVIEW(value)));
        urlBuffer.append("&");
      });
      urlBuffer.resize(urlBuffer.size() - 1);
    }

    // add custom headers
    headersCArray.clear();
    if (headers.get().valueType == SHType::Table) {
      auto htab = headers.get().payload.tableValue;
      ForEach(htab, [&](auto key, auto &value) {
        headersCArray.emplace_back(SHSTRVIEW(key).data());
        headersCArray.emplace_back(SHSTRVIEW(value).data());
      });
    }
    headersCArray.emplace_back(nullptr);
    attr.requestHeaders = headersCArray.data();

    for (int a = 0; a <= numRetries; a++) {
      state = 0;
      buffer.clear();

      emscripten_fetch_t *fetch = emscripten_fetch(&attr, urlBuffer.c_str());
      SPDLOG_LOGGER_DEBUG(logger, "Sending {} request \"{}\" {}", METHOD, urlBuffer, (void *)fetch);

      while (state == 0) {
#if SH_EMSCRIPTEN
        // This is to avoid requests getting stuck if the called does not return control to the os thread
        emscripten_sleep(0);
#endif
        SH_SUSPEND(context, 0.0);
      }

      if (state == 1) {
        break;
      }
    }

    if (unlikely(fullResponse)) {
      outMap["status"] = Var(status);
      if (asBytes) {
        outMap["body"] = Var((uint8_t *)buffer.data(), buffer.size());
      } else {
        outMap["body"] = Var(buffer);
      }
      return outMap;
    } else {
      if (asBytes) {
        return Var((uint8_t *)buffer.data(), buffer.size());
      } else {
        return Var(buffer);
      }
    }
  }
};

constexpr string_view GET = "GET";
struct Get : public GetLike<GET> {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends a GET request to the specified URL and outputs the response.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR(
        "The input for this shard should either be none or an optional string table of query parameters to append to the URL.");
  }

  static SHOptionalString outputHelp() { return SHCCSTR("The output is the response from the URL through the GET request."); }
};

constexpr string_view HEAD = "HEAD";
struct Head : public GetLike<HEAD> {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends a HEAD request to the specified URL and outputs the response.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR(
        "The input for this shard should either be none or an optional string table of query parameters to append to the URL.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output is the headers of the response from the URL through the HEAD request.");
  }
};

template <const string_view &METHOD> struct PostLike : public Base {
  static inline Types InputTypes{{CoreInfo::NoneType, CoreInfo::StringTableType, CoreInfo::BytesType, CoreInfo::StringType}};
  static SHTypesInfo inputTypes() { return InputTypes; }

  SHVar activate(SHContext *context, const SHVar &input) {
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, METHOD.data());
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = fetchSucceeded;
    attr.onerror = fetchFailed;
    attr.userData = this;
    attr.timeoutMSecs = timeout * 1000;

    // add custom headers
    bool hasContentType = false;
    headersCArray.clear();
    if (headers.get().valueType == SHType::Table) {
      auto htab = headers.get().payload.tableValue;
      ForEach(htab, [&](auto key, auto &value) {
        std::string data(SHSTRVIEW(key));
        std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) { return std::tolower(c); });
        if (data == "content-type") {
          hasContentType = true;
        }
        headersCArray.emplace_back(SHSTRVIEW(key).data());
        headersCArray.emplace_back(value.payload.stringValue);
      });
    }

    if (input.valueType == SHType::String) {
      if (!hasContentType) {
        headersCArray.emplace_back("content-type");
        headersCArray.emplace_back("application/json");
      }
      attr.overriddenMimeType = "application/json";
      attr.requestData = input.payload.stringValue;
      attr.requestDataSize = SHSTRLEN(input);
    } else if (input.valueType == SHType::Bytes) {
      if (!hasContentType) {
        headersCArray.emplace_back("content-type");
        headersCArray.emplace_back("application/octet-stream");
      }
      attr.requestData = (const char *)input.payload.bytesValue;
      attr.requestDataSize = input.payload.bytesSize;
    } else if (input.valueType == SHType::Table) {
      if (!hasContentType) {
        headersCArray.emplace_back("content-type");
        headersCArray.emplace_back("application/x-www-form-urlencoded");
      }
      urlBuffer.clear();
      ForEach(input.payload.tableValue, [&](auto key, auto &value) {
        auto sv_key = SHSTRVIEW(key);
        auto sv_value = SHSTRVIEW(value);
        urlBuffer.append(url_encode(sv_key));
        urlBuffer.append("=");
        urlBuffer.append(url_encode(sv_value));
        urlBuffer.append("&");
      });
      urlBuffer.resize(urlBuffer.size() - 1);
      attr.requestData = urlBuffer.c_str();
      attr.requestDataSize = urlBuffer.size();
    }

    headersCArray.emplace_back(nullptr);
    attr.requestHeaders = headersCArray.data();

    std::string urlStr = std::string(SHSTRVIEW(url.get()));

    for (int a = 0; a <= numRetries; a++) {
      state = 0;
      buffer.clear();

      emscripten_fetch_t *fetch = emscripten_fetch(&attr, urlStr.c_str());
      SPDLOG_LOGGER_DEBUG(logger, "Sending {} request \"{}\" {}", METHOD, urlStr, (void *)fetch);

      while (state == 0) {
#if SH_EMSCRIPTEN
        // This is to avoid requests getting stuck if the called does not return control to the os thread
        emscripten_sleep(0);
#endif
        SH_SUSPEND(context, 0.0);
      }

      if (state == 1)
        break;
    }

    if (state == 1) {
      if (unlikely(fullResponse)) {
        outMap["status"] = Var(status);
        if (asBytes) {
          outMap["body"] = Var((uint8_t *)buffer.data(), buffer.size());
        } else {
          outMap["body"] = Var(buffer);
        }
        return outMap;
      } else {
        if (asBytes) {
          return Var((uint8_t *)buffer.data(), buffer.size());
        } else {
          return Var(buffer);
        }
      }
    } else {
      SHLOG_ERROR("Http request failed with status: {}", buffer);
      throw ActivationError("Http request failed");
    }
  }
};

constexpr string_view POST = "POST";
struct Post : public PostLike<POST> {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends a HTTP POST request to the specified URL and outputs the response.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input for this shard should either be none, string, bytes, or string table to send in the body of the "
                   "POST request.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output is the response from the server through the POST request as a string, byte array, or table (if "
                   "the FullResponse parameter is set to true).");
  }
};

constexpr string_view PUT = "PUT";
struct Put : public PostLike<PUT> {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends a HTTP PUT request to the specified URL and outputs the response.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR(
        "The input for this shard should either be none, string, bytes, or string table to send in the body of the PUT request.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output is the response from the server through the PUT request as a string, byte array, or a table "
                   "(if the FullResponse parameter is set to true).");
  }
};

constexpr string_view PATCH = "PATCH";
struct Patch : public PostLike<PATCH> {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends a HTTP PATCH request to the specified URL and outputs the response.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input for this shard should either be none, string, bytes, or string table to send in the body of the "
                   "PATCH request.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output is the response from the server through the PATCH request as a string, byte array, or a table "
                   "(if the FullResponse parameter is set to true).");
  }
};

constexpr string_view DELETE = "DELETE";
struct Delete : public PostLike<DELETE> {
  static SHOptionalString help() {
    return SHCCSTR("This shard sends a HTTP DELETE request to the specified URL and outputs the response.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input for this shard should either be none, string, bytes, or string table to send in the body of the "
                   "DELETE request.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output is the response from the server through the DELETE request as a string, byte array, or a table "
                   "(if the FullResponse parameter is set to true).");
  }
};

#else

struct Peer : public std::enable_shared_from_this<Peer> {
  static constexpr uint32_t PeerCC = 'httP';
  static inline Type Info{{SHType::Object, {.object = {.vendorId = CoreCC, .typeId = PeerCC}}}};

  std::shared_ptr<SHWire> wire;
  std::shared_ptr<tcp::socket> socket;
  std::optional<entt::connection> onStopConnection;

  ~Peer() {
    if (onStopConnection)
      onStopConnection->release();
  }
};

struct PeerError {
  std::string_view source;
  beast::error_code ec;
  Peer *peer;
};

struct Server {
  Server() : _port(Var(7070)) {}

  static inline Parameters params{
      {"Handler", SHCCSTR("The wire that will be spawned and handle a remote request."), {CoreInfo::WireOrNone}},
      {"Endpoint", SHCCSTR("The URL from where your service can be accessed by a client."), {CoreInfo::StringType}},
      {"Port", SHCCSTR("The port this service will use."), {CoreInfo::IntType, CoreInfo::IntVarType}}};

  static SHParametersInfo parameters() { return params; }

  static SHOptionalString help() {
    return SHCCSTR("This shard sets up an HTTP server that listens for incoming connections, creates new peers for each "
                   "connection, and delegates request handling to the specified handler wire. It manages the lifecycle of "
                   "connections and ensures proper cleanup when the server is stopped.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  // bypass
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }

  ParamVar _port;

  void setParam(int idx, const SHVar &val) {
    switch (idx) {
    case 0:
      _handlerMaster = val;
      break;
    case 1:
      _endpoint = SHSTRVIEW(val);
      break;
    case 2:
      _port = val;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int idx) {
    switch (idx) {
    case 0:
      return _handlerMaster;
    case 1:
      return Var(_endpoint);
    case 2:
      return _port;
    default:
      return Var::Empty;
    }
  }

  SHTypeInfo compose(const SHInstanceData &data) {
    if (_handlerMaster.valueType == SHType::Wire)
      _pool.reset(new WireDoppelgangerPool<Peer>(_handlerMaster.payload.wireValue));

    const IterableExposedInfo shared(data.shared);
    // copy shared
    _sharedCopy = shared;
    return data.inputType;
  }

  std::unordered_map<const SHWire *, Peer *> _wireContainers;

  void wireOnStop(const SHWire::OnStopEvent &e) {
    auto it = _wireContainers.find(e.wire);
    if (it != _wireContainers.end()) {
      SHLOG_DEBUG("Releasing peer for wire {}", e.wire->name);
      _pool->release(it->second);
      _wireContainers.erase(it);
    }
  }

  // "Loop" forever accepting new connections.
  void accept_once(SHContext *context) {
    auto peer = _pool->acquire(_composer, context);
    _wireContainers[peer->wire.get()] = peer;

    // Assume that we recycle containers so the connection might already exist!
    if (!peer->onStopConnection) {
      auto mesh = context->main->mesh.lock();
      if (mesh) {
        peer->onStopConnection = mesh->dispatcher.sink<SHWire::OnStopEvent>().connect<&Server::wireOnStop>(this);
      }
    }

    peer->socket.reset(new tcp::socket(*_ioc));
    _acceptor->async_accept(*peer->socket, [context, peer, this](beast::error_code ec) {
      if (!ec) {
        auto mesh = context->main->mesh.lock();
        if (mesh) {
          peer->wire->getVariable("Http.Server.Socket"_swl) = Var::Object(peer, CoreCC, Peer::PeerCC);
          mesh->schedule(peer->wire, Var::Empty, false);
        } else {
          _pool->release(peer);
        }
      } else {
        _pool->release(peer);
      }
      // continue accepting the next
      accept_once(context);
    });
  }

  bool _is_running = false;

  void warmup(SHContext *context) {
    if (!_pool) {
      throw ComposeError("Peer wires pool not valid!");
    }

    _port.warmup(context);
  }

  void cleanup(SHContext *context) {
    if (_pool)
      _pool->stopAll();

    _port.cleanup(context);
    _is_running = false;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    if (!_is_running) {
      _ioc.reset(new net::io_context());
      auto addr = net::ip::make_address(_endpoint);
      auto port = _port.get().payload.intValue;
      // ensure port is in range of 1-65535
      if (port < 1 || port > 65535) {
        throw ActivationError("Port must be in range 1-65535");
      }
      _acceptor.reset(new tcp::acceptor(*_ioc, {addr, uint16_t(port)}));
      _composer.context = context;
      // start accepting
      accept_once(context);

      _is_running = true;
    }

    try {
      _ioc->poll();
    } catch (PeerError pe) {
      SHLOG_DEBUG("Http request error: {} from {} - closing connection.", pe.ec.message(), pe.source);
      stop(pe.peer->wire.get());
    }
    return input;
  }

  struct Composer {
    Server &server;
    SHContext *context;

    void compose(SHWire *wire, SHContext *context, bool recycling) {
      if (recycling)
        return;

      SHInstanceData data{};
      data.inputType = CoreInfo::StringType;
      data.shared = server._sharedCopy;
      data.wire = wire;
      wire->mesh = context->main->mesh;
      auto res = composeWire(wire, data);
      arrayFree(res.exposedInfo);
      arrayFree(res.requiredInfo);
    }
  };

  std::string _endpoint{"0.0.0.0"};
  OwnedVar _handlerMaster{};
  std::unique_ptr<WireDoppelgangerPool<Peer>> _pool;
  IterableExposedInfo _sharedCopy;
  Composer _composer{*this};

  // The io_context is required for all I/O
  std::unique_ptr<net::io_context> _ioc;
  std::deque<Peer> _peers;
  std::unique_ptr<tcp::acceptor> _acceptor;
};

struct Read {
  static inline Types OutTypes{{CoreInfo::StringType, CoreInfo::StringTableType, CoreInfo::StringType, CoreInfo::StringType}};
  static inline std::array<SHVar, 4> OutKeys{Var("method"), Var("headers"), Var("target"), Var("body")};
  static inline Type OutputType = Type::TableOf(OutTypes, OutKeys);

  static SHOptionalString help() {
    return SHCCSTR(
        "This shard reads incoming HTTP requests from a client connection, parses its components, and outputs them as a table. "
        "This shard should be used in conjunction with the Http.Server shard to handle incoming requests. "
        "It properly handles both regular and chunked transfer encoding.");
  }

  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The output is a table containing the HTTP request method, headers, target, and body.");
  }

  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return OutputType; }

  TableVar _headers;
  beast::error_code _last_error;
  std::string _error_source;

  void warmup(SHContext *context) {
    _peerVar = referenceVariable(context, "Http.Server.Socket");
    if (_peerVar->valueType == SHType::None) {
      throw WarmupError("Socket variable not found in wire");
    }
  }

  void cleanup(SHContext *context) {
    releaseVariable(_peerVar);
    _peerVar = nullptr;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_peerVar->valueType == SHType::Object);
    assert(_peerVar->payload.objectValue);
    auto peer = reinterpret_cast<Peer *>(_peerVar->payload.objectValue);

    bool done = false;
    bool has_error = false;
    request.body().clear();
    request.clear();
    buffer.clear();

    // First read just the header
    http::request_parser<http::string_body> parser;
    parser.header_limit(32 * 1024);                               // 32KB header limit
    parser.body_limit(std::numeric_limits<std::uint64_t>::max()); // No body limit

    http::async_read_header(*peer->socket, buffer, parser, [&](beast::error_code ec, std::size_t nbytes) {
      if (ec) {
        if (ec == beast::error::timeout) {
          _last_error = ec;
          _error_source = "Read header timeout";
          has_error = true;
        } else if (ec == beast::http::error::end_of_stream || ec == net::error::eof || ec == net::error::connection_reset) {
          // Client disconnected - trigger normal peer cleanup
          throw PeerError{"Client disconnected", ec, peer};
        } else {
          _last_error = ec;
          _error_source = "Read header";
          has_error = true;
        }
      }
      done = true;
    });

    while (!done) {
      SH_SUSPEND(context, 0.0);
    }

    if (has_error) {
      throw ActivationError(fmt::format("Error reading header: {}", _last_error.message()));
    }

    // Now read the body
    done = false;
    has_error = false;

    // Continue with the same parser for the body
    http::async_read(*peer->socket, buffer, parser, [&](beast::error_code ec, std::size_t nbytes) {
      if (ec) {
        if (ec == beast::error::timeout) {
          _last_error = ec;
          _error_source = "Read body timeout";
          has_error = true;
        } else if (ec == beast::http::error::end_of_stream || ec == net::error::eof || ec == net::error::connection_reset) {
          // Client disconnected - trigger normal peer cleanup
          throw PeerError{"Client disconnected", ec, peer};
        } else {
          _last_error = ec;
          _error_source = "Read body";
          has_error = true;
        }
      } else {
        request = parser.release();
      }
      done = true;
    });

    while (!done) {
      SH_SUSPEND(context, 0.0);
    }

    if (has_error) {
      throw ActivationError(fmt::format("Error reading body: {}", _last_error.message()));
    }

    switch (request.method()) {
    case http::verb::get:
      _output[Var("method")] = Var("GET");
      break;
    case http::verb::post:
      _output[Var("method")] = Var("POST");
      break;
    case http::verb::put:
      _output[Var("method")] = Var("PUT");
      break;
    case http::verb::delete_:
      _output[Var("method")] = Var("DELETE");
      break;
    case http::verb::head:
      _output[Var("method")] = Var("HEAD");
      break;
    case http::verb::options:
      _output[Var("method")] = Var("OPTIONS");
      break;
    default:
      throw ActivationError(fmt::format("Unsupported HTTP method {}.", request.method()));
    }

    _headers.clear();
    for (auto &header : request) {
      auto k = header.name_string();
      auto v = header.value();
      _headers.insert(shards::Var(k.data(), k.size()), shards::Var(v.data(), v.size()));
    }
    _output[Var("headers")] = Var(_headers);

    auto target = request.target();
    _output[Var("target")] = Var(target.data(), target.size());

    _output[Var("body")] = Var(request.body());

    return _output;
  }

  SHVar *_peerVar{nullptr};
  TableVar _output;
  beast::flat_buffer buffer{8192}; // Changed to flat_buffer for better performance
  http::request<http::string_body> request;
};

struct Response {
  static inline Types PostInTypes{CoreInfo::StringType, CoreInfo::BytesType};

  static SHOptionalString help() {
    return SHCCSTR("This shard sends an HTTP response to the client after receiving an HTTP request.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input string or byte array will be used directly as the body of the response.");
  }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  static SHTypesInfo inputTypes() { return PostInTypes; }
  static SHTypesInfo outputTypes() { return PostInTypes; }

  PARAM_PARAMVAR(_status, "Status", "The HTTP status code to return.", {CoreInfo::IntType, CoreInfo::IntVarType})
  PARAM_PARAMVAR(_headers, "Headers", "The headers to attach to this response.",
                 {CoreInfo::StringTableType, CoreInfo::StringVarTableType, CoreInfo::NoneType})
  PARAM_PARAMVAR(_contentType, "ContentType", "The content type of the response.",
                 {CoreInfo::StringType, CoreInfo::StringVarType})
  PARAM_IMPL(PARAM_IMPL_FOR(_status), PARAM_IMPL_FOR(_headers), PARAM_IMPL_FOR(_contentType))

  Response() {
    _status = Var(200);
    _contentType = Var("application/json");
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _peerVar = referenceVariable(context, "Http.Server.Socket");
    if (_peerVar->valueType == SHType::None) {
      throw WarmupError("Socket variable not found in wire");
    }
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    releaseVariable(_peerVar);
    _peerVar = nullptr;
  }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return data.inputType;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_peerVar->valueType == SHType::Object);
    assert(_peerVar->payload.objectValue);
    auto peer = reinterpret_cast<Peer *>(_peerVar->payload.objectValue);
    _response.clear();

    _response.result(http::status(_status.get().payload.intValue));
    _response.set(http::field::content_type, SHSTRVIEW(_contentType.get()));
    auto input_view = SHSTRVIEW(input); // this also supports bytes cos POD layout
    _response.body() = input_view;

    // add custom headers (or replace current ones!)
    if (_headers.get().valueType == SHType::Table) {
      auto htab = _headers.get().payload.tableValue;
      ForEach(htab, [&](auto &key, auto &value) {
        if (key.valueType != SHType::String || value.valueType != SHType::String)
          throw ActivationError("Headers must be a table of strings.");
        boost::core::string_view s(key.payload.stringValue, key.payload.stringLen);
        boost::core::string_view v(value.payload.stringValue, value.payload.stringLen);
        _response.set(s, v);
        return true;
      });
    }

    _response.prepare_payload();

    bool done = false;
    http::async_write(*peer->socket, _response, [&, peer](beast::error_code ec, std::size_t nbytes) {
      if (ec) {
        throw PeerError{"Response", ec, peer};
      } else {
        SHLOG_TRACE("Response: async_write bytes: {}", nbytes);
        done = true;
      }
    });

    // we suspend here, that's why we captured & above!!
    while (!done) {
      SH_SUSPEND(context, 0.0);
    }

    return input;
  }

  SHVar *_peerVar{nullptr};
  http::response<http::string_body> _response;
};

struct Chunk {
  static inline Types PostInTypes{CoreInfo::StringType, CoreInfo::BytesType};

  static SHOptionalString help() {
    return SHCCSTR("This shard processes and packages outgoing Http response date into smaller manageable pieces and "
                   "subsequently writes them to the socket while managing the chunked transfer encoding process.");
  }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input to the chunk shard is the data (String or Bytes) to be sent in the current chunk. This data is "
                   "part of a larger response that will be sent in multiple chunks if necessary.");
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR(
        "The output is the same as the input. The chunked transfer encoding is handled internally when writing to the socket.");
  }

  static SHTypesInfo inputTypes() { return PostInTypes; }
  static SHTypesInfo outputTypes() { return PostInTypes; }

  static inline Parameters params{
      {"Status", SHCCSTR("The HTTP status code to return."), {CoreInfo::IntType}},
      {"Headers",
       SHCCSTR("The headers to attach to this response."),
       {CoreInfo::StringTableType, CoreInfo::StringVarTableType, CoreInfo::NoneType}},
  };

  static SHParametersInfo parameters() { return params; }

  void setParam(int index, const SHVar &value) {
    switch (index) {
    case 0:
      _status = http::status(value.payload.intValue);
      break;
    case 1:
      _headers = value;
      break;
    default:
      break;
    }
  }

  SHVar getParam(int index) {
    switch (index) {
    case 0:
      return Var(int64_t(_status));
    case 1:
      return _headers;
    default:
      return Var::Empty;
    }
  }

  // compose to fixup output type purely based on input type
  SHTypeInfo compose(const SHInstanceData &data) { return data.inputType; }

  void warmup(SHContext *context) {
    _headers.warmup(context);
    _peerVar = referenceVariable(context, "Http.Server.Socket");
    if (_peerVar->valueType == SHType::None) {
      throw WarmupError("Socket variable not found in wire");
    }
  }

  void cleanup(SHContext *context) {
    _headers.cleanup();
    releaseVariable(_peerVar);
    _peerVar = nullptr;
    _firstChunk = true;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_peerVar->valueType == SHType::Object);
    assert(_peerVar->payload.objectValue);
    auto peer = reinterpret_cast<Peer *>(_peerVar->payload.objectValue);

    bool done;

    if (_firstChunk) {
      _firstChunk = false;
      _response.clear();
      _response.result(_status);
      _response.set(http::field::version, "1.1");
      _response.set(http::field::content_type, "text/plain");
      // add custom headers (or replace current ones!)
      if (_headers.get().valueType == SHType::Table) {
        auto htab = _headers.get().payload.tableValue;
        ForEach(htab, [&](auto &key, auto &value) {
          if (key.valueType != SHType::String || value.valueType != SHType::String)
            throw ActivationError("Headers must be a table of strings.");
          boost::core::string_view s(key.payload.stringValue, key.payload.stringLen);
          boost::core::string_view v(value.payload.stringValue, value.payload.stringLen);
          _response.set(s, v);
          return true;
        });
      }
      _response.chunked(true);

      done = false;
      http::response_serializer<http::empty_body> _serializer{_response};
      http::async_write_header(*peer->socket, _serializer, [&, peer](beast::error_code ec, std::size_t nbytes) {
        if (ec) {
          throw PeerError{"Chunk", ec, peer};
        } else {
          SHLOG_TRACE("Chunk: async_write bytes (chunk headers): {}", nbytes);
          done = true;
        }
      });

      // we suspend here, that's why we captured & above!!
      while (!done) {
        SH_SUSPEND(context, 0.0);
      }
    }

    auto input_view = SHSTRVIEW(input); // this also supports bytes cos POD layout
    if (input_view.empty()) {
      // meaning we are done with the response after this chunk!
      _firstChunk = true;
    }

    done = false;
    auto chunkStr = fmt::format("{:X}\r\n{}\r\n", input_view.size(), input_view);
    net::async_write(*peer->socket, net::buffer(chunkStr), [&, peer](beast::error_code ec, std::size_t nbytes) {
      if (ec) {
        throw PeerError{"Chunk", ec, peer};
      } else {
        SHLOG_TRACE("Chunk: async_write (sent) bytes (chunk): {}", nbytes);
        done = true;
      }
    });

    // we suspend here, that's why we captured & above!!
    while (!done) {
      SH_SUSPEND(context, 0.0);
    }

    return input;
  }

  http::status _status{200};
  SHVar *_peerVar{nullptr};
  ParamVar _headers{};
  http::response<http::empty_body> _response{http::status::ok, 11};
  bool _firstChunk{true};
};

struct SendFile {
  static SHOptionalString help() { return SHCCSTR("This shard sends a static file to the client over HTTP."); }

  static SHOptionalString inputHelp() {
    return SHCCSTR("The input for this shard should be a string representing the path to the file to be sent.");
  }

  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }

  PARAM_PARAMVAR(_headers, "Headers", "The headers to attach to this response.",
                 {CoreInfo::StringTableType, CoreInfo::StringVarTableType, CoreInfo::NoneType})
  PARAM_IMPL(PARAM_IMPL_FOR(_headers))

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _peerVar = referenceVariable(context, "Http.Server.Socket");
    if (_peerVar->valueType == SHType::None) {
      throw WarmupError("Socket variable not found in wire");
    }
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    releaseVariable(_peerVar);
    _peerVar = nullptr;
    _response.clear();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  static boost::beast::string_view mime_type(boost::beast::string_view path) {
    using boost::beast::iequals;
    auto const ext = [&path] {
      auto const pos = path.rfind(".");
      if (pos == boost::beast::string_view::npos)
        return boost::beast::string_view{};
      return path.substr(pos);
    }();
    if (iequals(ext, ".htm"))
      return "text/html";
    if (iequals(ext, ".html"))
      return "text/html";
    if (iequals(ext, ".php"))
      return "text/html";
    if (iequals(ext, ".css"))
      return "text/css";
    if (iequals(ext, ".txt"))
      return "text/plain";
    if (iequals(ext, ".js"))
      return "application/javascript";
    if (iequals(ext, ".json"))
      return "application/json";
    if (iequals(ext, ".xml"))
      return "application/xml";
    if (iequals(ext, ".swf"))
      return "application/x-shockwave-flash";
    if (iequals(ext, ".flv"))
      return "video/x-flv";
    if (iequals(ext, ".png"))
      return "image/png";
    if (iequals(ext, ".jpe"))
      return "image/jpeg";
    if (iequals(ext, ".jpeg"))
      return "image/jpeg";
    if (iequals(ext, ".jpg"))
      return "image/jpeg";
    if (iequals(ext, ".gif"))
      return "image/gif";
    if (iequals(ext, ".bmp"))
      return "image/bmp";
    if (iequals(ext, ".ico"))
      return "image/vnd.microsoft.icon";
    if (iequals(ext, ".tiff"))
      return "image/tiff";
    if (iequals(ext, ".tif"))
      return "image/tiff";
    if (iequals(ext, ".svg"))
      return "image/svg+xml";
    if (iequals(ext, ".svgz"))
      return "image/svg+xml";
    if (iequals(ext, ".wasm"))
      return "application/wasm";
    return "application/text";
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    assert(_peerVar->valueType == SHType::Object);
    assert(_peerVar->payload.objectValue);
    auto peer = reinterpret_cast<Peer *>(_peerVar->payload.objectValue);

    fs::path p{GetGlobals().RootPath.c_str()};
    p /= SHSTRING_PREFER_SHSTRVIEW(input);

    http::file_body::value_type file;
    boost::beast::error_code ec;
    auto pstr = p.generic_string();
    bool done = false;
    file.open(pstr.c_str(), boost::beast::file_mode::read, ec);
    if (unlikely(bool(ec))) {
      _404_response.clear();
      _404_response.result(http::status::not_found);
      _404_response.body() = "File not found.";
      _404_response.prepare_payload();

      http::async_write(*peer->socket, _404_response, [&, peer](beast::error_code ec, std::size_t nbytes) {
        if (ec) {
          throw PeerError{"SendFile:1", ec, peer};
        } else {
          SHLOG_TRACE("SendFile:1: async_write bytes: {}", nbytes);
          done = true;
        }
      });
    } else {
      _response.clear();
      _response.result(http::status::ok);
      _response.set(http::field::content_type, mime_type(SHSTRVIEW(input)));
      _response.body() = std::move(file);

      // add custom headers
      if (_headers.get().valueType == SHType::Table) {
        auto htab = _headers.get().payload.tableValue;
        ForEach(htab, [&](auto &key, auto &value) {
          if (key.valueType != SHType::String || value.valueType != SHType::String) {
            throw std::runtime_error("Headers must be a table of strings");
          }
          boost::core::string_view k{key.payload.stringValue};
          boost::core::string_view v{value.payload.stringValue};
          _response.set(k, v);
          return true;
        });
      }

      _response.prepare_payload();

      http::async_write(*peer->socket, _response, [&, peer](beast::error_code ec, std::size_t nbytes) {
        if (ec) {
          throw PeerError{"SendFile:2", ec, peer};
        } else {
          SHLOG_TRACE("SendFile:2: async_write bytes: {}", nbytes);
          done = true;
        }
      });
    }

    // we suspend here, that's why we captured & above!!
    while (!done) {
      SH_SUSPEND(context, 0.0);
    }

    return input;
  }

  SHVar *_peerVar{nullptr};
  http::response<http::file_body> _response;
  http::response<http::string_body> _404_response;
};
#endif

struct EncodeURI {
  std::string _output;
  static SHOptionalString help() {
    return SHCCSTR("This shard encodes a string into a URI-encoded format making it safe to use in URLs.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("The string to be encoded."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The resulting URI-encoded string."); }

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    auto str = SHSTRVIEW(input);
    _output = url_encode(str);
    return Var(_output);
  }
};

struct DecodeURI {
  std::string _output;

  static SHOptionalString help() { return SHCCSTR("This shard decodes a URI-encoded string back into its original format."); }

  static SHOptionalString inputHelp() { return SHCCSTR("The URI-encoded string to be decoded."); }

  static SHOptionalString outputHelp() { return SHCCSTR("The resulting decoded string."); }

  static SHTypesInfo inputTypes() { return CoreInfo::StringType; }
  static SHTypesInfo outputTypes() { return CoreInfo::StringType; }
  SHVar activate(SHContext *context, const SHVar &input) {
    auto str = SHSTRVIEW(input);
    _output = url_decode(str);
    return Var(_output);
  }
};
} // namespace Http
} // namespace shards
SHARDS_REGISTER_FN(http) {
  using namespace shards::Http;
#if SH_EMSCRIPTEN
  REGISTER_SHARD("Http.Get", Get);
  REGISTER_SHARD("Http.Head", Head);
  REGISTER_SHARD("Http.Post", Post);
  REGISTER_SHARD("Http.Put", Put);
  REGISTER_SHARD("Http.Patch", Patch);
  REGISTER_SHARD("Http.Delete", Delete);
#else
  REGISTER_SHARD("Http.Server", Server);
  REGISTER_SHARD("Http.Read", Read);
  REGISTER_SHARD("Http.Response", Response);
  REGISTER_SHARD("Http.Chunk", Chunk);
  REGISTER_SHARD("Http.SendFile", SendFile);
#endif
  REGISTER_SHARD("String.EncodeURI", EncodeURI);
  REGISTER_SHARD("String.DecodeURI", DecodeURI);
}
