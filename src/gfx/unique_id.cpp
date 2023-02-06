#include "unique_id.hpp"
#include <vector>

namespace gfx {

static std::vector<std::weak_ptr<UniqueIdGenerator *>> &getRegister() {
  static std::vector<std::weak_ptr<UniqueIdGenerator *>> inst;
  return inst;
}

void UniqueIdGenerator::_resetAll() {
  for (auto it = getRegister().begin(); it != getRegister().end();) {
    if (it->expired()) {
      it = getRegister().erase(it);
    } else {
      auto &ptr = *it;
      UniqueIdGenerator &generator = **ptr.lock().get();
      generator.counter = 0;
      ++it;
    }
  }
}

void UniqueIdGenerator::_register(std::weak_ptr<UniqueIdGenerator *> &&tracker) {
  getRegister().emplace_back(std::move(tracker));
}
} // namespace gfx
