#ifndef GFX_SHADER_BLOCK
#define GFX_SHADER_BLOCK

#include <memory>
#include <string>

namespace gfx {
namespace shader {
struct IGeneratorContext;
namespace blocks {
using String = std::string;
template <typename T> using UniquePtr = std::unique_ptr<T>;

struct Block {
  virtual ~Block() = default;
  virtual void apply(IGeneratorContext &context) const = 0;
  virtual std::unique_ptr<Block> clone() = 0;
};
using BlockPtr = UniquePtr<Block>;

// Source block injected directly into the generated shader
struct Direct : public Block {
  String code;

  Direct(const char *code) : code(code) {}
  Direct(const String &code) : code(code) {}
  Direct(String &&code) : code(code) {}
  void apply(IGeneratorContext &context) const;
  BlockPtr clone() { return std::make_unique<Direct>(code); }
};

struct ConvertibleToBlock {
  UniquePtr<Block> block;
  ConvertibleToBlock(const char *str) { block = std::make_unique<Direct>(str); }
  ConvertibleToBlock(const std::string &str) { block = std::make_unique<Direct>(str); }
  ConvertibleToBlock(std::string &&str) { block = std::make_unique<Direct>(std::move(str)); }
  ConvertibleToBlock(BlockPtr &&block) : block(std::move(block)) {}
  UniquePtr<Block> operator()() { return std::move(block); }
};

template <typename T, typename T1 = void> struct ConvertToBlock {};
template <typename T> struct ConvertToBlock<T, typename std::enable_if<std::is_base_of_v<Block, T>>::type> {
  UniquePtr<Block> operator()(T &&block) { return std::make_unique<T>(std::move(block)); }
};
template <typename T> struct ConvertToBlock<T, typename std::enable_if<!std::is_base_of_v<Block, T>>::type> {
  UniquePtr<Block> operator()(T &&arg) { return ConvertibleToBlock(std::move(arg))(); }
};

// Block applied outside of the function body
struct Header : public Block {
  BlockPtr inner;

  template <typename T>
  Header(T &&inner) : inner(ConvertToBlock<T>{}(std::forward<T>(inner))) {}

  void apply(IGeneratorContext &context) const;
  BlockPtr clone() { return std::make_unique<Header>(inner->clone()); }
};

} // namespace blocks
} // namespace shader
} // namespace gfx

#endif // GFX_SHADER_BLOCK
