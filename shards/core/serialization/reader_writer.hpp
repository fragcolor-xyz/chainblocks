#ifndef FC879B5E_F5A6_4396_AF8E_37A459C8A0B9
#define FC879B5E_F5A6_4396_AF8E_37A459C8A0B9

#include <stdexcept>
#include <cassert>
#include <string_view>
#include <vector>

namespace shards {

enum class IOMode { Read, Write };

template <typename TAlloc = std::allocator<uint8_t>> struct BufferRefWriter {
  static constexpr IOMode Mode = IOMode::Write;

  std::vector<uint8_t, TAlloc> &_buffer;
  BufferRefWriter(std::vector<uint8_t, TAlloc> &stream, bool clear = true) : _buffer(stream) {
    if (clear) {
      _buffer.clear();
    }
  }
  void operator()(const uint8_t *buf, size_t size) { _buffer.insert(_buffer.end(), buf, buf + size); }
};

template <typename TAlloc = std::allocator<uint8_t>> struct BufferWriter {
  static constexpr IOMode Mode = IOMode::Write;

  std::vector<uint8_t, TAlloc> _buffer;
  BufferRefWriter<TAlloc> _inner;
  
  BufferWriter(TAlloc allocator = TAlloc()) : _buffer(allocator), _inner(_buffer) {}
  void operator()(const uint8_t *buf, size_t size) { _inner(buf, size); }
  std::vector<uint8_t, TAlloc> takeBuffer() { return std::move(_buffer); }
};

struct BytesReader {
  static constexpr IOMode Mode = IOMode::Read;

  const uint8_t *buffer;
  size_t offset;
  size_t max;

  BytesReader(std::string_view sv) : buffer((uint8_t *)sv.data()), offset(0), max(sv.size()) {}
  BytesReader(const uint8_t *buf, size_t size) : buffer(buf), offset(0), max(size) {}
  BytesReader(boost::span<uint8_t> buf) : buffer(buf.data()), offset(0), max(buf.size()) {}
  void operator()(uint8_t *buf, size_t size) {
    auto newOffset = offset + size;
    if (newOffset > max) {
      throw std::runtime_error("Overflow requested");
    }
    memcpy(buf, buffer + offset, size);
    offset = newOffset;
  }
};

struct BufferRefReader {
  static constexpr IOMode Mode = IOMode::Read;

  BytesReader _inner;
  BufferRefReader(const std::vector<uint8_t> &stream) : _inner(stream.data(), stream.size()) {}
  void operator()(uint8_t *buf, size_t size) { _inner(buf, size); }
};
} // namespace shards

#endif /* FC879B5E_F5A6_4396_AF8E_37A459C8A0B9 */
