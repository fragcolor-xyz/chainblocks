#ifndef B58E65B7_CB61_45CD_9BEB_F5ECC4E9C1C7
#define B58E65B7_CB61_45CD_9BEB_F5ECC4E9C1C7

#include <map>
#include <memory>
#include <concepts>
#include "data_cache.hpp"

namespace gfx::data {

#define GFX_DATA_ID64(a, b) (uint64_t(a) << 32 | uint64_t(b))

struct IDerivedDataGenerator {
  // Unique ID of this generator
  virtual uint64_t getID() = 0;
  virtual LoadedAssetDataPtr generate(const LoadedAssetDataPtr &input) = 0;
  virtual ~IDerivedDataGenerator() = default;
};

struct DerivedDataGeneratorRegistry {
private:
  std::map<uint64_t, std::shared_ptr<IDerivedDataGenerator>> generators;

public:
  const std::shared_ptr<IDerivedDataGenerator> &find(uint64_t id) {
    static std::shared_ptr<IDerivedDataGenerator> nullGenerator;
    auto it = generators.find(id);
    if (it == generators.end()) {
      return nullGenerator;
    }
    return it->second;
  }

  void register_(std::shared_ptr<IDerivedDataGenerator> generator) { generators[generator->getID()] = generator; }
};

template <typename T>
concept DerivedDataGenerator =
    requires { T::StaticID->std::convertible_to<uint64_t>; } && std::derived_from<T, IDerivedDataGenerator>;
} // namespace gfx::data

#endif /* B58E65B7_CB61_45CD_9BEB_F5ECC4E9C1C7 */
