#ifndef SH_EXTRA_GFX_SHADER_TRANSLATOR
#define SH_EXTRA_GFX_SHADER_TRANSLATOR
#include "wgsl.hpp"
#include <gfx/shader/blocks.hpp>
#include <gfx/shader/temp_variable.hpp>
#include <log/log.hpp>
#include <map>
#include <shards.hpp>
#include <spdlog/spdlog.h>
#include <string>

namespace gfx {
namespace shader {
template <typename T> using UniquePtr = std::unique_ptr<T>;

struct ShaderComposeError : public std::runtime_error {
  Shard *shard;

  ShaderComposeError(const char *what, Shard *shard = nullptr) : std::runtime_error(what), shard(shard){};
  ShaderComposeError(std::string &&what, Shard *shard = nullptr) : std::runtime_error(std::move(what)), shard(shard){};
};

struct IAppender {
  virtual ~IAppender() = default;
  virtual void append(blocks::Block *block, BlockPtr &&blockToAppend) = 0;
};

template <typename T> struct Appender : public IAppender {
  static inline IAppender *getInstance() { return nullptr; }
};

template <> struct Appender<blocks::Compound> : public IAppender {
  void append(blocks::Block *blockPtr, BlockPtr &&blockToAppend) {
    blocks::Compound *block = static_cast<blocks::Compound *>(blockPtr);
    block->children.emplace_back(std::move(blockToAppend));
  }

  static inline IAppender *getInstance() {
    static Appender instance;
    return &instance;
  }
};

struct VariableInfo {
  FieldType type;
  bool isReadOnly = false;

  VariableInfo() = default;
  VariableInfo(FieldType type, bool isReadOnly = false) : type(type), isReadOnly(isReadOnly) {}
};

typedef std::map<std::string, VariableInfo> VariableInfoMap;

// Filters invalid characters out of variable names
inline std::string filterVariableName(std::string_view varName) {
  std::string result;
  result.reserve(varName.size());
  for (size_t i = 0; i < varName.size(); i++) {
    char c = varName[i];
    if (c == '-') {
      c = '_';
    }
    result.push_back(c);
  }
  return result;
}

struct VariableStorage {
  // Stores type assignment
  VariableInfoMap variables;

  // Maps variable names to globally unique variable names
  std::map<std::string, std::string> uniqueVariableNames;

  // Maps a variable name to a unique variable name
  const std::string &mapUniqueVariableName(const std::string &varName, const std::string &uniqueName) {
    return uniqueVariableNames.insert_or_assign(varName, uniqueName).first->second;
  }

  // Retrieves globally unique variable name from a regular variable name
  const std::string &resolveUniqueVariableName(const std::string &varName) const {
    auto it = uniqueVariableNames.find(varName);
    if (it == uniqueVariableNames.end())
      throw std::logic_error("Variable does not exist in this scope"); // Should not happen here, catch externally
    return it->second;
  }
};

// Keeps track of pushes into sequences
// used to resolve them as matrix types
struct VirtualSeq {
  NumFieldType elementType;
  std::vector<std::unique_ptr<IWGSLGenerated>> elements;

  VirtualSeq() = default;
  VirtualSeq(VirtualSeq &&other) = default;
};

struct VirtualTable {
  std::map<std::string, std::unique_ptr<IWGSLGenerated>> elements;

  VirtualTable() = default;
  VirtualTable(VirtualTable &&other) = default;
};

// Virtual table that can be passed around as stack value
struct VirtualTableOnStack : public IWGSLGenerated {
  VirtualTable table;

  VirtualTableOnStack(VirtualTable &&table) : table(std::move(table)) {}
  const FieldType &getType() const { throw std::logic_error("Can not generate table in wgsl"); }
  blocks::BlockPtr toBlock() const { throw std::logic_error("Can not generate table in wgsl"); }
};

// References a shader block together with a strategy for appending children into it
struct TranslationBlockRef {
  blocks::Block *block{};
  IAppender *appender{};
  VariableStorage variables;

  std::map<std::string, VirtualSeq> virtualSequences;

  TranslationBlockRef(blocks::Block *block, IAppender *appender) : block(block), appender(appender) {}
  TranslationBlockRef(TranslationBlockRef &&other) = default;

  template <typename T> static TranslationBlockRef make(blocks::Block *block) {
    return TranslationBlockRef(block, Appender<T>::getInstance());
  }

  template <typename T> static TranslationBlockRef make(const UniquePtr<T> &block) {
    return TranslationBlockRef(block.get(), Appender<T>::getInstance());
  }
};

struct TranslationRegistry;

struct TranslatedFunctionArgument {
  FieldType type;
  std::string wgslName;
  std::string shardsName;
};

struct TranslatedFunction {
  std::string functionName;
  std::optional<FieldType> outputType;
  std::optional<FieldType> inputType;
  std::vector<TranslatedFunctionArgument> arguments;

  TranslatedFunction() = default;
  TranslatedFunction(TranslatedFunction &&) = default;
};

// Context used during shader translations
// inputs C++ shards blocks
// outputs a shader block hierarchy defining a shader function
struct TranslationContext {
  TranslationRegistry &translationRegistry;

  shards::logging::Logger logger;

  VariableStorage globals;

  // Keeps track of which wires have been translated info functions
  std::map<SHWire *, TranslatedFunction> translatedWires;

  UniquePtr<blocks::Compound> root;
  std::vector<TranslationBlockRef> stack;

  // The value wgsl source generated from the last shards block
  std::unique_ptr<IWGSLGenerated> wgslTop;

private:
  TempVariableAllocator tempVariableAllocator{"_tsl"};

public:
  TranslationContext();
  TranslationContext(const TranslationContext &) = delete;
  TranslationContext &operator=(const TranslationContext &) = delete;

  TranslationBlockRef &getTop() {
    assert(stack.size() >= 0);
    return stack.back();
  }

  // Add a new generated shader blocks without entering it
  template <typename T> void addNew(std::unique_ptr<T> &&ptr) {
    enterNew<T>(std::move(ptr));
    leave();
  }

  // Add a new generated shader block and push it on the stack
  template <typename T> void enterNew(std::unique_ptr<T> &&ptr) {
    TranslationBlockRef &top = getTop();
    if (!top.appender) {
      throw std::logic_error("Current shader block can not have children");
    }

    blocks::Block *newBlock = ptr.get();
    top.appender->append(getTop().block, std::move(ptr));
    stack.push_back(TranslationBlockRef::make<T>(newBlock));
  }

  // Remove a generated shader block from the stack
  void leave() { stack.pop_back(); }

  // Enter a shard and translate it recursively
  void processShard(ShardPtr shard);

  // Translates a wire into a function
  const TranslatedFunction &processWire(const std::shared_ptr<SHWire> &wire, const std::optional<FieldType> &inputType);

  // Translates shards into a function
  TranslatedFunction processShards(const std::vector<ShardPtr> &shards, const SHComposeResult &composeResult,
                                   const std::optional<FieldType> &inputType, const std::string &name = "anonymous");
  TranslatedFunction processShards(const Shards &shardsSeq, const SHComposeResult &composeResult,
                                   const std::optional<FieldType> &inputType, const std::string &name = "anonymous") {
    std::vector<ShardPtr> shards;
    for (size_t i = 0; i < shardsSeq.len; i++)
      shards.push_back(shardsSeq.elements[i]);
    return processShards(shards, composeResult, inputType, name);
  }

  // Assign a block to a temporary variable and return it's name
  template <typename T> const std::string &assignTempVar(std::unique_ptr<T> &&ptr) {
    const std::string &varName = getUniqueVariableName();
    addNew(blocks::makeCompoundBlock(fmt::format("let {} = ", varName), std::move(ptr), ";\n"));
    return varName;
  }

  // Set the intermediate wgsl source generated from the last shard that was translated
  template <typename T, typename... TArgs> void setWGSLTop(TArgs... args) {
    wgslTop = std::make_unique<T>(std::forward<TArgs>(args)...);
  }

  // Set the intermediate wgsl source but reference it as a single variable
  // use to avoide duplicating function calls when setting the result as a stack value
  template <typename T> const std::string &setWGSLTopVar(FieldType type, std::unique_ptr<T> &&ptr) {
    const std::string &varName = assignTempVar(std::move(ptr));
    setWGSLTop<WGSLSource>(type, varName);
    return varName;
  }

  void clearWGSLTop() { wgslTop.reset(); }

  std::unique_ptr<IWGSLGenerated> takeWGSLTop() {
    std::unique_ptr<IWGSLGenerated> result;
    wgslTop.swap(result);
    return result;
  }

  const std::string &getUniqueVariableName(const std::string_view &hint = std::string_view()) {
    if (!hint.empty()) {
      return tempVariableAllocator.get(filterVariableName(hint));
    }
    return tempVariableAllocator.get();
  }

  // Tries to find a global variable
  bool findVariableGlobal(const std::string &varName, const FieldType *&outFieldType) const;

  // Tries to find a variable in all scopes & globally
  // outParent will be null in case of global variables
  bool findVariable(const std::string &varName, const FieldType *&outFieldType, const TranslationBlockRef *&outParent) const;

  // Tries to expand sequences in the current scope into their respective matrix types
  // returns true if it did so, the variable can be acessed using findVariable
  bool tryExpandIntoVariable(const std::string &varName);

  // Assigns or updates a variable
  // returns a reference to the variable in the same format as `reference` does
  WGSLBlock assignVariable(const std::string &varName, bool global, bool allowUpdate, std::unique_ptr<IWGSLGenerated> &&value);

  // Returns a shader block that references a variable by name
  // the variable can either be defined in the current, parent or global scope
  WGSLBlock reference(const std::string &varName);
};

// Handles translating shards C++ blocks to shader blocks
struct ITranslationHandler {
  virtual ~ITranslationHandler() = default;
  virtual void translate(ShardPtr shard, TranslationContext &context) = 0;
};

// Register that maps shard name to translation handlers
struct TranslationRegistry {
private:
  std::map<std::string, ITranslationHandler *> handlers;

public:
  void registerHandler(const char *shardName, ITranslationHandler *translateable);
  ITranslationHandler *resolve(Shard *shard);
};

TranslationRegistry &getTranslationRegistry();

void applyShaderEntryPoint(SHContext *context, shader::EntryPoint &entryPoint, const SHVar &input);

} // namespace shader
} // namespace gfx

#endif // SH_EXTRA_GFX_SHADER_TRANSLATOR
