#ifndef B01B265C_2D67_4D6C_8DD1_D697ED304EC6
#define B01B265C_2D67_4D6C_8DD1_D697ED304EC6

#include "events.hpp"
#include "event_buffer.hpp"
#include "detached.hpp"
#include "input/events.hpp"
#include "messages.hpp"
#include <shared_mutex>
#include <mutex>
#include <shards/core/function.hpp>
#include <boost/lockfree/spsc_queue.hpp>

namespace gfx {
struct Window;
}

namespace shards::input {
struct FocusTracker {
  void *previous{};
  void *current{};

  void swap() {
    previous = current;
    current = nullptr;
  }

  // Checks current or previous acquisition of focus
  bool hasFocus(void *token) const { return token == current || (!current && token == previous); }

  // Check is the target can receive input based on focus rules
  // If no element has requested any focus, any element can receive input
  bool canReceiveInput(void *token) const { return token == current || (!current && (!previous || token == previous)); }

  // Request exclusive focus on the given element
  bool requestFocus(void *token) {
    // Same frame early out
    if (current == token)
      return true;

    // Obtain new focus or continue focus from last frame
    if (!current && (!previous || previous == token)) {
      current = token;
      return true;
    } else {
      return false;
    }
  }
};

struct InputMaster;

struct IInputHandler {
  virtual ~IInputHandler() = default;
  virtual int getPriority() const { return 0; }
  virtual void handle(InputMaster &master) = 0;
};

struct EventConsumer {
private:
  std::weak_ptr<IInputHandler> handler;

public:
  EventConsumer(std::weak_ptr<IInputHandler> handler) : handler(handler) {}
  inline EventConsumer &consume(ConsumableEvent &evt) {
    evt.consume(handler);
    return *this;
  }
  inline EventConsumer &operator()(ConsumableEvent &evt) { return consume(evt); }
  inline bool operator==(const std::weak_ptr<IInputHandler> &other) const { return handler.lock() == other.lock(); }
};

struct InputMaster {
private:
  std::vector<std::weak_ptr<IInputHandler>> handlers;
  std::vector<std::shared_ptr<IInputHandler>> handlersLocked;
  std::shared_mutex mutex;
  std::vector<shards::Function<void(InputMaster &)>> postInputCallbacks;

  std::vector<ConsumableEvent> events;

  boost::lockfree::spsc_queue<Message> messageQueue;

  FocusTracker focusTracker;

  DetachedInput input;

  bool terminateRequested{};

  bool foreground{true};

public:
  InputMaster();
  ~InputMaster();

  bool isTerminateRequested() const { return terminateRequested; }

  void postMessage(const Message &message);

  void addHandler(std::shared_ptr<IInputHandler> ptr) {
    std::unique_lock<decltype(mutex)> l(mutex);
    handlers.emplace_back(ptr);
  }

  void removeHandler(std::shared_ptr<IInputHandler> ptr) {
    std::unique_lock<decltype(mutex)> l(mutex);
    auto pred = [&](const auto &weak) { return weak.expired() || weak.lock() == ptr; };
    handlers.erase(std::remove_if(handlers.begin(), handlers.end(), pred), handlers.end());
  }

  void update(gfx::Window &window);
  void reset() {}

  // Only used for debug UI
  void addPostInputCallback(shards::Function<void(InputMaster &)> callback) {
    postInputCallbacks.emplace_back(std::move(callback));
  }

  // Only use from main thread
  void getHandlers(std::vector<std::shared_ptr<IInputHandler>> &outVec) {
    std::unique_lock<decltype(mutex)> l(mutex);
    updateAndSortHandlersLocked(outVec);
  }

  const InputState &getState() const { return input.state; }
  std::vector<ConsumableEvent> &getEvents() { return events; }
  FocusTracker &getFocusTracker() { return focusTracker; }

private:
  void handleMessage(const Message &message, gfx::Window& window);
  void updateAndSortHandlers();

  // Assumes already locked
  void updateAndSortHandlersLocked(std::vector<std::shared_ptr<IInputHandler>> &outVec);
};

} // namespace shards::input

#endif /* B01B265C_2D67_4D6C_8DD1_D697ED304EC6 */
