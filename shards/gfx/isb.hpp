#ifndef GFX_ISB
#define GFX_ISB

#include <compare>
#include <functional>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <boost/core/span.hpp>

namespace gfx {

/// <div rustbindgen opaque></div>
struct ImmutableSharedBuffer {
public:
  /// <div rustbindgen hide></div>
  struct IContainer {
    virtual ~IContainer() = default;
    virtual size_t getLength() = 0;
    virtual const uint8_t *getData() = 0;
  };

private:
  /// <div rustbindgen hide></div>
  struct OwnedContainer : public IContainer {
    std::vector<uint8_t> data;

    OwnedContainer() = default;
    OwnedContainer(std::vector<uint8_t> &&data) : data(std::move(data)) {}

    virtual size_t getLength() { return data.size(); }
    virtual const uint8_t *getData() { return data.data(); }
  };

  /// <div rustbindgen hide></div>
  struct ForeignContainer : public IContainer {
    uint8_t *data{};
    size_t dataLength{};
    void *userData{};
    std::function<void(uint8_t *data, void *userData)> freeCallback;

    ForeignContainer() = default;
    ~ForeignContainer() { freeCallback(data, userData); }
    virtual size_t getLength() { return dataLength; }
    virtual const uint8_t *getData() { return data; }
  };

  std::shared_ptr<IContainer> container;

public:
  ImmutableSharedBuffer() = default;

  ImmutableSharedBuffer(std::shared_ptr<IContainer> container) : container(container) {}

  ImmutableSharedBuffer(const void *inData, size_t inDataLength) {
    container = std::make_shared<OwnedContainer>();
    auto &owned = *(OwnedContainer *)container.get();
    owned.data.resize(inDataLength);
    memcpy(owned.data.data(), inData, inDataLength);
  }

  ImmutableSharedBuffer(const void *inData, size_t inDataLength, std::function<void(void *, void *)> freeCallback,
                        void *userData = nullptr) {
    container = std::make_shared<ForeignContainer>();
    auto &foreign = *(ForeignContainer *)container.get();
    foreign.data = (uint8_t *)inData;
    foreign.dataLength = inDataLength;
    foreign.freeCallback = freeCallback;
    foreign.userData = userData;
  }

  ImmutableSharedBuffer(std::vector<uint8_t> &&data) { container = std::make_shared<OwnedContainer>(std::move(data)); }

  boost::span<const uint8_t> asSpan() const { return boost::span<const uint8_t>(getData(), getLength()); }

  std::strong_ordering operator<=>(const ImmutableSharedBuffer &other) const = default;

  size_t getLength() const { return container ? container->getLength() : 0; }
  const uint8_t *getData() const { return container ? container->getData() : nullptr; }
  size_t size() const { return getLength(); }
  const uint8_t *data() const { return getData(); }
  operator bool() const { return (bool)container; }
};

} // namespace gfx

#endif // GFX_ISB
