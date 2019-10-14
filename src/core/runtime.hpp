#pragma once

// ONLY CLANG AND GCC SUPPORTED FOR NOW

#include <string.h> // memset

#include <regex>

#include "blocks_macros.hpp"
#include "chainblocks.hpp"
// C++ Mandatory from now!

// Since we build the runtime we are free to use any std and lib
#include <atomic>
#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

// Required external dependencies
// For coroutines/context switches
#include <boost/context/continuation.hpp>
typedef boost::context::continuation CBCoro;
// For sleep
#if _WIN32
#include <Windows.h>
#else
#include <time.h>
#endif

#include <tuple>
// Tuple hashing
namespace std {
namespace {

// Code from boost
// Reciprocal of the golden ratio helps spread entropy
//     and handles duplicates.
// See Mike Seymour in magic-numbers-in-boosthash-combine:
//     http://stackoverflow.com/questions/4948780

template <class T> inline void hash_combine(std::size_t &seed, T const &v) {
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Recursive template code derived from Matthieu M.
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
  static void apply(size_t &seed, Tuple const &tuple) {
    HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
    hash_combine(seed, std::get<Index>(tuple));
  }
};

template <class Tuple> struct HashValueImpl<Tuple, 0> {
  static void apply(size_t &seed, Tuple const &tuple) {
    hash_combine(seed, std::get<0>(tuple));
  }
};
} // namespace

template <typename... TT> struct hash<std::tuple<TT...>> {
  size_t operator()(std::tuple<TT...> const &tt) const {
    size_t seed = 0;
    HashValueImpl<std::tuple<TT...>>::apply(seed, tt);
    return seed;
  }
};
} // namespace std

namespace chainblocks {
extern phmap::node_hash_map<std::string, CBBlockConstructor> BlocksRegister;
extern phmap::node_hash_map<std::tuple<int32_t, int32_t>, CBObjectInfo>
    ObjectTypesRegister;
extern phmap::node_hash_map<std::tuple<int32_t, int32_t>, CBEnumInfo>
    EnumTypesRegister;
extern std::map<std::string, CBCallback> RunLoopHooks;
extern phmap::node_hash_map<std::string, CBCallback> ExitHooks;
extern phmap::node_hash_map<std::string, CBChain *> GlobalChains;

CBlock *createBlock(const char *name);
void registerCoreBlocks();
void registerBlock(const char *fullName, CBBlockConstructor constructor);
void registerObjectType(int32_t vendorId, int32_t typeId, CBObjectInfo info);
void registerEnumType(int32_t vendorId, int32_t typeId, CBEnumInfo info);
void registerRunLoopCallback(const char *eventName, CBCallback callback);
void unregisterRunLoopCallback(const char *eventName);
void registerExitCallback(const char *eventName, CBCallback callback);
void unregisterExitCallback(const char *eventName);
void callExitCallbacks();
void registerChain(CBChain *chain);
void unregisterChain(CBChain *chain);
}; // namespace chainblocks

void freeDerivedInfo(CBTypeInfo info);
CBTypeInfo deriveTypeInfo(CBVar &value);

[[nodiscard]] CBValidationResult
validateConnections(const std::vector<CBlock *> &chain,
                    CBValidationCallback callback, void *userData,
                    CBTypeInfo inputType = CBTypeInfo(),
                    CBExposedTypesInfo consumables = nullptr);
[[nodiscard]] CBValidationResult
validateConnections(const CBlocks chain, CBValidationCallback callback,
                    void *userData, CBTypeInfo inputType = CBTypeInfo(),
                    CBExposedTypesInfo consumables = nullptr);
[[nodiscard]] CBValidationResult
validateConnections(const CBChain *chain, CBValidationCallback callback,
                    void *userData, CBTypeInfo inputType = CBTypeInfo(),
                    CBExposedTypesInfo consumables = nullptr);

bool validateSetParam(CBlock *block, int index, CBVar &value,
                      CBValidationCallback callback, void *userData);

struct CBChain {
  CBChain(const char *chain_name)
      : looped(false), unsafe(false), name(chain_name), coro(nullptr),
        started(false), finished(false), returned(false), failed(false),
        rootTickInput(CBVar()), finishedOutput(CBVar()), ownedOutput(false),
        context(nullptr), node(nullptr) {
    chainblocks::registerChain(this);
  }

  ~CBChain() {
    cleanup();
    chainblocks::unregisterChain(this);
    chainblocks::destroyVar(rootTickInput);
  }

  void cleanup();

  // Also the chain takes ownership of the block!
  void addBlock(CBlock *blk) { blocks.push_back(blk); }

  // Also removes ownership of the block
  void removeBlock(CBlock *blk) {
    auto findIt = std::find(blocks.begin(), blocks.end(), blk);
    if (findIt != blocks.end()) {
      blocks.erase(findIt);
    }
  }

  // Attributes
  bool looped;
  bool unsafe;

  std::string name;

  CBCoro *coro;

  // we could simply null check coro but actually some chains (sub chains), will
  // run without a coro within the root coro so we need this too
  bool started;

  // this gets cleared before every runChain and set after every runChain
  std::atomic_bool finished;

  // when running as coro if actually the coro lambda exited
  bool returned;
  bool failed;

  CBVar rootTickInput{};
  CBVar previousOutput{};
  CBVar finishedOutput{};
  bool ownedOutput;

  CBContext *context;
  CBNode *node;
  std::vector<CBlock *> blocks;
};

struct CBContext {
  CBContext(CBCoro &&sink, CBChain *running_chain)
      : chain(running_chain), restarted(false), aborted(false),
        shouldPause(false), paused(false), continuation(std::move(sink)),
        iterationCount(0) {
    static std::regex re(
        R"([^abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\-\._]+)");
    loggerName = std::regex_replace(chain->name, re, "_");
    loggerName = "chain." + loggerName;
    el::Loggers::getLogger(loggerName.c_str());
  }

  ~CBContext() { el::Loggers::unregisterLogger(loggerName.c_str()); }

  CBChain *chain;

  phmap::node_hash_map<std::string, CBVar> variables;
  std::string error;

  // Those 2 go together with CBVar chainstates restart and stop
  bool restarted;
  // Also used to cancel a chain
  bool aborted;
  // Used internally to pause a chain execution
  std::atomic_bool shouldPause;
  std::atomic_bool paused;

  // Used within the coro& stack! (suspend, etc)
  CBCoro &&continuation;
  Duration next;

  // Have a logger per context
  std::string loggerName;

  // Iteration counter
  uint64_t iterationCount;

  void setError(const char *errorMsg) { error = errorMsg; }
};

#include "blocks/core.hpp"
#include "blocks/math.hpp"

using json = nlohmann::json;
// The following procedures implement json.hpp protocol in order to allow easy
// integration! they must stay outside the namespace!
void to_json(json &j, const CBVar &var);
void from_json(const json &j, CBVar &var);
void to_json(json &j, const CBChainPtr &chain);
void from_json(const json &j, CBChainPtr &chain);

namespace chainblocks {

void installSignalHandlers();

ALWAYS_INLINE inline void activateBlock(CBlock *blk, CBContext *context,
                                        const CBVar &input,
                                        CBVar &previousOutput) {
  switch (blk->inlineBlockId) {
  case CoreConst: {
    auto cblock = reinterpret_cast<chainblocks::ConstRuntime *>(blk);
    previousOutput = cblock->core._value;
    return;
  }
  case CoreIs: {
    auto cblock = reinterpret_cast<chainblocks::IsRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreIsNot: {
    auto cblock = reinterpret_cast<chainblocks::IsNotRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreAnd: {
    auto cblock = reinterpret_cast<chainblocks::AndRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreOr: {
    auto cblock = reinterpret_cast<chainblocks::OrRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreNot: {
    auto cblock = reinterpret_cast<chainblocks::NotRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreIsMore: {
    auto cblock = reinterpret_cast<chainblocks::IsMoreRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreIsLess: {
    auto cblock = reinterpret_cast<chainblocks::IsLessRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreIsMoreEqual: {
    auto cblock = reinterpret_cast<chainblocks::IsMoreEqualRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreIsLessEqual: {
    auto cblock = reinterpret_cast<chainblocks::IsLessEqualRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreSleep: {
    auto cblock = reinterpret_cast<chainblocks::SleepRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreInput: {
    auto cblock = reinterpret_cast<chainblocks::InputRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreStop: {
    auto cblock = reinterpret_cast<chainblocks::StopRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreRestart: {
    auto cblock = reinterpret_cast<chainblocks::RestartRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreTake: {
    auto cblock = reinterpret_cast<chainblocks::TakeRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CorePush: {
    auto cblock = reinterpret_cast<chainblocks::PushRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreRepeat: {
    auto cblock = reinterpret_cast<chainblocks::RepeatRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreGet: {
    auto cblock = reinterpret_cast<chainblocks::GetRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreSet: {
    auto cblock = reinterpret_cast<chainblocks::SetRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreUpdate: {
    auto cblock = reinterpret_cast<chainblocks::UpdateRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case CoreSwap: {
    auto cblock = reinterpret_cast<chainblocks::SwapRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAdd: {
    auto cblock = reinterpret_cast<chainblocks::Math::AddRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathSubtract: {
    auto cblock = reinterpret_cast<chainblocks::Math::SubtractRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathMultiply: {
    auto cblock = reinterpret_cast<chainblocks::Math::MultiplyRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathDivide: {
    auto cblock = reinterpret_cast<chainblocks::Math::DivideRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathXor: {
    auto cblock = reinterpret_cast<chainblocks::Math::XorRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAnd: {
    auto cblock = reinterpret_cast<chainblocks::Math::AndRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathOr: {
    auto cblock = reinterpret_cast<chainblocks::Math::OrRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathMod: {
    auto cblock = reinterpret_cast<chainblocks::Math::ModRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathLShift: {
    auto cblock = reinterpret_cast<chainblocks::Math::LShiftRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathRShift: {
    auto cblock = reinterpret_cast<chainblocks::Math::RShiftRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAbs: {
    auto cblock = reinterpret_cast<chainblocks::Math::AbsRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathExp: {
    auto cblock = reinterpret_cast<chainblocks::Math::ExpRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathExp2: {
    auto cblock = reinterpret_cast<chainblocks::Math::Exp2Runtime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathExpm1: {
    auto cblock = reinterpret_cast<chainblocks::Math::Expm1Runtime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathLog: {
    auto cblock = reinterpret_cast<chainblocks::Math::LogRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathLog10: {
    auto cblock = reinterpret_cast<chainblocks::Math::Log10Runtime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathLog2: {
    auto cblock = reinterpret_cast<chainblocks::Math::Log2Runtime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathLog1p: {
    auto cblock = reinterpret_cast<chainblocks::Math::Log1pRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathSqrt: {
    auto cblock = reinterpret_cast<chainblocks::Math::SqrtRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathCbrt: {
    auto cblock = reinterpret_cast<chainblocks::Math::CbrtRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathSin: {
    auto cblock = reinterpret_cast<chainblocks::Math::SinRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathCos: {
    auto cblock = reinterpret_cast<chainblocks::Math::CosRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathTan: {
    auto cblock = reinterpret_cast<chainblocks::Math::TanRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAsin: {
    auto cblock = reinterpret_cast<chainblocks::Math::AsinRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAcos: {
    auto cblock = reinterpret_cast<chainblocks::Math::AcosRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAtan: {
    auto cblock = reinterpret_cast<chainblocks::Math::AtanRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathSinh: {
    auto cblock = reinterpret_cast<chainblocks::Math::SinhRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathCosh: {
    auto cblock = reinterpret_cast<chainblocks::Math::CoshRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathTanh: {
    auto cblock = reinterpret_cast<chainblocks::Math::TanhRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAsinh: {
    auto cblock = reinterpret_cast<chainblocks::Math::AsinhRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAcosh: {
    auto cblock = reinterpret_cast<chainblocks::Math::AcoshRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathAtanh: {
    auto cblock = reinterpret_cast<chainblocks::Math::AtanhRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathErf: {
    auto cblock = reinterpret_cast<chainblocks::Math::ErfRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathErfc: {
    auto cblock = reinterpret_cast<chainblocks::Math::ErfcRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathTGamma: {
    auto cblock = reinterpret_cast<chainblocks::Math::TGammaRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathLGamma: {
    auto cblock = reinterpret_cast<chainblocks::Math::LGammaRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathCeil: {
    auto cblock = reinterpret_cast<chainblocks::Math::CeilRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathFloor: {
    auto cblock = reinterpret_cast<chainblocks::Math::FloorRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathTrunc: {
    auto cblock = reinterpret_cast<chainblocks::Math::TruncRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  case MathRound: {
    auto cblock = reinterpret_cast<chainblocks::Math::RoundRuntime *>(blk);
    previousOutput = cblock->core.activate(context, input);
    return;
  }
  default: {
    // NotInline
    previousOutput = blk->activate(blk, context, &input);
    return;
  }
  }
}

static CBRunChainOutput runChain(CBChain *chain, CBContext *context,
                                 const CBVar &chainInput) {
  chain->previousOutput = CBVar();

  // Detect and pause if we need to here
  // avoid pausing in the middle or so, that is for a proper debug mode runner,
  // here we care about performance
  while (context->shouldPause) {
    context->paused = true;

    auto suspendRes = suspend(context, 0.0);
    // Since we suspended we need to make sure we should continue when resuming
    switch (suspendRes.payload.chainState) {
    case CBChainState::Restart: {
      return {chain->previousOutput, Restarted};
    }
    case CBChainState::Stop: {
      return {chain->previousOutput, Stopped};
    }
    default:
      continue;
    }
  }

  chain->started = true;
  chain->context = context;
  context->paused = false;

  auto input = chainInput;
  for (auto blk : chain->blocks) {
    try {
      activateBlock(blk, context, input, chain->previousOutput);
      input = chain->previousOutput;

      if (chain->previousOutput.valueType == None) {
        switch (chain->previousOutput.payload.chainState) {
        case CBChainState::Restart: {
          return {chain->previousOutput, Restarted};
        }
        case CBChainState::Stop: {
          // Print errors if any, we might have stopped because of some error!
          if (unlikely(context->error.length() > 0)) {
            LOG(ERROR) << "Block activation error, failed block: "
                       << std::string(blk->name(blk));
            LOG(ERROR) << "Last error: " << std::string(context->error);
            return {chain->previousOutput, Failed};
          } else {
            return {chain->previousOutput, Stopped};
          }
        }
        case CBChainState::Return: {
          // Use input as output, return previous block result
          return {input, Restarted};
        }
        case CBChainState::Rebase:
          // Rebase means we need to put back main input
          input = chainInput;
          break;
        case CBChainState::Continue:
          break;
        }
      }
    } catch (boost::context::detail::forced_unwind const &e) {
      throw; // required for Boost Coroutine!
    } catch (const std::exception &e) {
      LOG(ERROR) << "Block activation error, failed block: "
                 << std::string(blk->name(blk));
      if (context->error.length() > 0)
        LOG(ERROR) << "Last error: " << std::string(context->error);
      LOG(ERROR) << e.what();
      return {chain->previousOutput, Failed};
    } catch (...) {
      LOG(ERROR) << "Block activation error, failed block: "
                 << std::string(blk->name(blk));
      if (context->error.length() > 0)
        LOG(ERROR) << "Last error: " << std::string(context->error);
      return {chain->previousOutput, Failed};
    }
  }

  return {chain->previousOutput, Running};
}

inline CBRunChainOutput runSubChain(CBChain *chain, CBContext *context,
                                    const CBVar &input) {
  chain->finished = false; // Reset finished flag (atomic)
  auto runRes = chainblocks::runChain(chain, context, input);
  chain->finishedOutput = runRes.output; // Write result before setting flag
  chain->finished = true;                // Set finished flag (atomic)
  return runRes;
}

inline void cleanup(CBChain *chain) {
  // Run cleanup on all blocks, prepare them for a new start if necessary
  // Do this in reverse to allow a safer cleanup
  for (auto it = chain->blocks.rbegin(); it != chain->blocks.rend(); ++it) {
    auto blk = *it;
    try {
      blk->cleanup(blk);
    } catch (boost::context::detail::forced_unwind const &e) {
      throw; // required for Boost Coroutine!
    } catch (const std::exception &e) {
      LOG(ERROR) << "Block cleanup error, failed block: "
                 << std::string(blk->name(blk));
      LOG(ERROR) << e.what() << '\n';
    } catch (...) {
      LOG(ERROR) << "Block cleanup error, failed block: "
                 << std::string(blk->name(blk));
    }
  }
}

static boost::context::continuation run(CBChain *chain,
                                        boost::context::continuation &&sink) {
  auto running = true;
  // Reset return state
  chain->returned = false;
  // Clean previous output if we had one
  if (chain->ownedOutput) {
    destroyVar(chain->finishedOutput);
    chain->ownedOutput = false;
  }
  // Reset error
  chain->failed = false;
  // Create a new context and copy the sink in
  CBContext context(std::move(sink), chain);

  // We prerolled our coro, suspend here before actually starting.
  // This allows us to allocate the stack ahead of time.
  context.continuation = context.continuation.resume();
  if (context.aborted) // We might have stopped before even starting!
    goto endOfChain;

  while (running) {
    running = chain->looped;
    context.restarted = false; // Remove restarted flag

    chain->finished = false; // Reset finished flag (atomic)
    auto runRes = runChain(chain, &context, chain->rootTickInput);
    chain->finishedOutput = runRes.output; // Write result before setting flag
    chain->finished = true;                // Set finished flag (atomic)
    context.iterationCount++;              // increatse iteration counter
    if (unlikely(runRes.state == Failed)) {
      chain->failed = true;
      context.aborted = true;
      break;
    } else if (unlikely(runRes.state == Stopped)) {
      context.aborted = true;
      break;
    }

    if (!chain->unsafe && chain->looped) {
      // Ensure no while(true), yield anyway every run
      context.next = Duration(0);
      context.continuation = context.continuation.resume();
      // This is delayed upon continuation!!
      if (context.aborted)
        break;
    }
  }

endOfChain:
  // Copy the output variable since the next call might wipe it
  auto tmp = chain->finishedOutput;
  chain->finishedOutput =
      CBVar(); // Reset it we are not sure on the internal state
  chain->ownedOutput = true;
  cloneVar(chain->finishedOutput, tmp);

  // run cleanup on all the blocks
  cleanup(chain);

  // Need to take care that we might have stopped the chain very early due to
  // errors and the next eventual stop() should avoid resuming
  chain->returned = true;
  return std::move(context.continuation);
}

inline void prepare(CBChain *chain) {
  if (chain->coro)
    return;

  chain->coro = new CBCoro(
      boost::context::callcc([&chain](boost::context::continuation &&sink) {
        return run(chain, std::move(sink));
      }));
}

inline void start(CBChain *chain, CBVar input = {}) {
  if (!chain->coro || !(*chain->coro) || chain->started)
    return; // check if not null and bool operator also to see if alive!

  chainblocks::cloneVar(chain->rootTickInput, input);
  *chain->coro = chain->coro->resume();
}

inline bool stop(CBChain *chain, CBVar *result = nullptr) {
  // Clone the results if we need them
  if (result)
    cloneVar(*result, chain->finishedOutput);

  if (chain->coro) {
    // Run until exit if alive, need to propagate to all suspended blocks!
    if ((*chain->coro) && !chain->returned) {
      // set abortion flag, we always have a context in this case
      chain->context->aborted = true;

      // BIG Warning: chain->context existed in the coro stack!!!
      // after this resume chain->context is trash!
      chain->coro->resume();
    }

    // delete also the coro ptr
    delete chain->coro;
    chain->coro = nullptr;
  } else {
    // if we had a coro this will run inside it!
    cleanup(chain);
  }

  chain->started = false;

  if (chain->failed) {
    return false;
  }

  return true;
}

inline bool tick(CBChain *chain, CBVar rootInput = chainblocks::Empty) {
  if (!chain->context || !chain->coro || !(*chain->coro) || chain->returned ||
      !chain->started)
    return false; // check if not null and bool operator also to see if alive!

  Duration now = Clock::now().time_since_epoch();
  if (now >= chain->context->next) {
    if (rootInput != chainblocks::Empty) {
      chain->rootTickInput = rootInput;
    }
    *chain->coro = chain->coro->resume();
  }
  return true;
}

inline bool isRunning(CBChain *chain) {
  return chain->started && !chain->returned;
}

inline bool hasEnded(CBChain *chain) {
  return chain->started && chain->returned;
}

inline bool isCanceled(CBContext *context) { return context->aborted; }

inline void sleep(double seconds = -1.0) {
  // negative = no sleep, just run callbacks
  if (seconds >= 0) {
#ifdef _WIN32
    HANDLE timer;
    LARGE_INTEGER ft;
    ft.QuadPart = -(int64_t(seconds * 10000000));
    timer = CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
    WaitForSingleObject(timer, INFINITE);
    CloseHandle(timer);
#else
    struct timespec delay = {0, int64_t(seconds * 1000000000)};
    while (nanosleep(&delay, &delay))
      ;
#endif
  }

  // Run loop callbacks after sleeping
  for (auto &cbinfo : RunLoopHooks) {
    if (cbinfo.second) {
      cbinfo.second();
    }
  }
}
}; // namespace chainblocks

struct CBNode {
  ~CBNode() { terminate(); }

  void schedule(CBChain *chain, CBVar input = {}, bool validate = true) {
    if (chain->node)
      throw chainblocks::CBException(
          "schedule failed, chain was already scheduled!");

    // Validate the chain
    if (validate) {
      auto validation = validateConnections(
          chain->blocks,
          [](const CBlock *errorBlock, const char *errorTxt,
             bool nonfatalWarning, void *userData) {
            auto node = reinterpret_cast<CBNode *>(userData);
            auto blk = const_cast<CBlock *>(errorBlock);
            if (!nonfatalWarning) {
              node->errorMsg.assign(errorTxt);
              node->errorMsg += ", input block: " + std::string(blk->name(blk));
              throw chainblocks::CBException(node->errorMsg.c_str());
            } else {
              LOG(INFO) << "Validation warning: " << errorTxt
                        << " input block: " << blk->name(blk);
            }
          },
          this);
      stbds_arrfree(validation.exposedInfo);
    }

    chains.push_back(chain);
    chain->node = this;
    chainblocks::prepare(chain);
    chainblocks::start(chain, input);
  }

  bool tick(CBVar input = chainblocks::Empty) {
    auto noErrors = true;
    chainsTicking = chains;
    for (auto chain : chainsTicking) {
      chainblocks::tick(chain, input);
      if (!chainblocks::isRunning(chain)) {
        if (!chainblocks::stop(chain)) {
          noErrors = false;
        }
        chains.remove(chain);
        chain->node = nullptr;
      }
    }
    return noErrors;
  }

  void terminate() {
    for (auto chain : chains) {
      chainblocks::stop(chain);
      chain->node = nullptr;
    }
    chains.clear();
  }

  void remove(CBChain *chain) {
    chainblocks::stop(chain);
    chains.remove(chain);
    chain->node = nullptr;
  }

  bool empty() { return chains.empty(); }

  phmap::node_hash_map<std::string, CBVar> variables;

private:
  std::list<CBChain *> chains;
  std::list<CBChain *> chainsTicking;
  std::string errorMsg;
};
