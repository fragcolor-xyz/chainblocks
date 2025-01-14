#ifndef D6E6F079_B676_4C2F_8B5C_032302417557
#define D6E6F079_B676_4C2F_8B5C_032302417557

#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>
#include <gfx/shader/buffer_serializer.hpp>
#include <gfx/shader/layout_traverser2.hpp>
#include <gfx/buffer.hpp>
#include <tracy/Wrapper.hpp>
#include "buffer.hpp"
#include "gfx.hpp"
#include "shards_types.hpp"

namespace gfx {
 
template <typename T> inline void visitHostSharableShardsType(SHVar &var, T arg) {
  switch (var.valueType) {
  case SHType::Int:
    arg.template visit<SHType::Int>(var.payload.intValue);
    break;
  case SHType::Int2:
    arg.template visit<SHType::Int2>((linalg::vec<int64_t, 2> &)var.payload.int2Value);
    break;
  case SHType::Int3:
    arg.template visit<SHType::Int3>((gfx::int3 &)var.payload.int3Value);
    break;
  case SHType::Int4:
    arg.template visit<SHType::Int4>((gfx::int3 &)var.payload.int4Value);
    break;
  case SHType::Float:
    arg.template visit<SHType::Float>(var.payload.floatValue);
    break;
  case SHType::Float2:
    arg.template visit<SHType::Float2>((linalg::vec<double, 2> &)var.payload);
    break;
  case SHType::Float3:
    arg.template visit<SHType::Float3>((gfx::float3 &)var.payload);
    break;
  case SHType::Float4:
    arg.template visit<SHType::Float4>((gfx::float4 &)var.payload);
    break;
  default:
    throw std::out_of_range(fmt::format("Invalid shader type {}", var));
  }
}

struct CallSerialize {
  shader::BufferSerializer &serializer;
  shader::LayoutRef2 &lr;
  template <SHType BaseType, typename Q> void visit(Q &var) { serializer.write(lr, var); }
};

struct CallDeserialize {
  shader::BufferSerializer &serializer;
  shader::LayoutRef2 &lr;
  template <SHType BaseType, typename Q> void visit(Q &var) { serializer.read(lr, var); }
};

template <typename Operation> struct SHBufferVisitor {
  shader::BufferSerializer &serializer;
  shader::LayoutTraverser2 &lt;
  boost::container::small_vector<shader::LayoutRef2, 8> stack;

  SHBufferVisitor(shader::BufferSerializer &ser, shader::LayoutTraverser2 &lt) : serializer(ser), lt(lt) {
    stack.push_back(lt.root());
  }

  shader::LayoutRef2 &top() { return stack.back(); }

  void visit(shards::TableVar &tbl) {
    ZoneScopedN("BufferWriteVisitor::table");
    for (auto &[k, v] : tbl) {
      if (k.valueType != SHType::String)
        throw std::runtime_error("Expected a string key");

      auto innerNode = top().inner(k.payload.stringValue);
      if (!innerNode)
        throw std::runtime_error(fmt::format("Invalid path: {}", k.payload.stringValue));

      stack.push_back(*innerNode);
      visit(tbl.get<SHVar>(k));
      stack.pop_back();
    }
  }

  void visit(shards::SeqVar &seq) {
    ZoneScopedN("BufferWriteVisitor::seq");
    auto atPtr = std::get_if<shader::ArrayType>(&top().type());
    if (!atPtr)
      throw std::runtime_error("Expected an array type");
    auto &at = *atPtr;

    auto first = top().inner();
    size_t arrayStride = first.arrayElementStride.value();
    if (at->fixedLength) {
      size_t fixedLength = *at->fixedLength;
      for (size_t i = 0; i < std::min(fixedLength, seq.size()); i++) {
        auto ref = first;
        ref.offset += arrayStride * i;
        stack.push_back(ref);
        visit(seq[i]);
        stack.pop_back();
      }
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  void visitVal(SHVar &var) {
    ZoneScopedN("BufferWriteVisitor::SHVar/value");
    try {
      auto nt = std::get_if<shader::NumType>(&top().type());
      if (!nt)
        throw std::runtime_error("Expected a number type");

      visitHostSharableShardsType(var, Operation{.serializer = serializer, .lr = top()});
    } catch (gfx::shader::SerializeTypeMismatch &e) {
      throw std::runtime_error(fmt::format("Type mismatch: expected {}, got {} ({})", e.expected, var, var.valueType));
    }
  }

  void visit(SHVar &var) {
    if (var.valueType == SHType::Table) {
      visit((shards::TableVar &)var);
    } else if (var.valueType == SHType::Seq) {
      visit((shards::SeqVar &)var);
    } else {
      visitVal(var);
    }
  }
};

struct BufferShard {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::Buffer; }
  static SHOptionalString help() { return SHCCSTR("Creates a new graphics buffer "); }

  PARAM_VAR(_type, "Type", "The type descriptor of the buffer", {shards::CoreInfo::AnyTableType});
  PARAM_VAR(_addressSpace, "AddressSpace", "The address space to use the buffer with",
            {shards::CoreInfo::NoneType, ShardsTypes::BufferAddressSpaceEnumInfo::Type});
  PARAM_IMPL(PARAM_IMPL_FOR(_type), PARAM_IMPL_FOR(_addressSpace));

  shader::Type _parsedType;
  DerivedLayoutInfo _dli;
  SHBuffer *_output{};

  void resetOutput() {
    if (_output) {
      ShardsTypes::BufferObjectVar.Release(_output);
      _output = nullptr;
    }
  }

  BufferShard() {}

  shader::AddressSpace getAddressSpace() {
    return _addressSpace->isNone() ? shader::AddressSpace::Uniform : (shader::AddressSpace)_addressSpace->payload.enumValue;
  }

  void parseDataType() { _dli = parseStructType(_parsedType, *_type, getAddressSpace()); }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    parseDataType();
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    resetOutput();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    if (!_output) {
      _output = ShardsTypes::BufferObjectVar.New();
      _output->buffer = std::make_shared<gfx::Buffer>();
      _output->buffer->setAddressSpaceUsage(getAddressSpace());
      _output->designatedAddressSpace = getAddressSpace();
      _output->type = std::get<shader::StructType>(_parsedType);
      if (!_dli.runtimeSizedField) {
        std::vector<uint8_t> empty(_dli.layout.size, 0);
        _output->data = gfx::ImmutableSharedBuffer(std::move(empty));
        _output->buffer->update(_output->data);
      }
    }

    return ShardsTypes::BufferObjectVar.Get(_output);
  }
};

struct ReadBuffer {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::BoolType; }
  static SHOptionalString help() { return SHCCSTR("Creates a new graphics buffer "); }

  PARAM_PARAMVAR(_buffer, "Buffer", "The buffer to read", {shards::Type::VariableOf(ShardsTypes::Buffer)});
  PARAM_PARAMVAR(_var, "Var", "The variable to read the data into", {shards::CoreInfo::AnyVarTableType});
  PARAM_VAR(_wait, "Wait", "Wait for read to complete", {shards::CoreInfo::BoolType});
  PARAM_IMPL(PARAM_IMPL_FOR(_buffer), PARAM_IMPL_FOR(_var), PARAM_IMPL_FOR(_wait));

  std::optional<DerivedLayoutInfo> _dli;
  UniqueId _lastBufferId;

  RequiredGraphicsContext _requiredGraphicsContext;
  GpuReadBufferPtr _readBuffer = makeGpuReadBuffer();

  ReadBuffer() { _wait = shards::Var(false); }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredGraphicsContext.warmup(context);
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _requiredGraphicsContext.cleanup(context);
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredVariables.push_back(RequiredGraphicsContext::getExposedTypeInfo());

    if (_var.isNone())
      throw std::runtime_error("ReadBuffer: output Var is required");

    if (_buffer.isNone())
      throw std::runtime_error("ReadBuffer: Buffer is required");

    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &buffer = shards::varAsObjectChecked<SHBuffer>(_buffer.get(), ShardsTypes::Buffer);
    if (!_dli || _lastBufferId != buffer.buffer->getId()) {
      _dli = DerivedLayoutInfo::make(buffer.designatedAddressSpace, buffer.type);
      _lastBufferId = buffer.buffer->getId();
    }

    _requiredGraphicsContext->renderer->copyBuffer(buffer.buffer, _readBuffer, (bool)*_wait);

    // Poll for previously queued operation completion
    if (!*_wait)
      _requiredGraphicsContext->renderer->pollBufferCopies();

    if (_readBuffer->data.empty()) {
      return shards::Var(false);
    }

    shader::BufferSerializer ser(_readBuffer->data.data(), _readBuffer->data.size());
    shader::LayoutTraverser2 lt(_dli->layout, _dli->structLayoutLookup, buffer.designatedAddressSpace);
    SHBufferVisitor<CallDeserialize> visitor(ser, lt);

    {
      ZoneScopedN("GFX.ReadBuffer/visit");
      visitor.visit(_var.get());
    }

    return shards::Var(true);
  }
};

struct WriteBuffer {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return shards::CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("Creates a new graphics buffer "); }

  PARAM_PARAMVAR(_buffer, "Buffer", "The buffer", {shards::Type::VariableOf(ShardsTypes::Buffer)});
  PARAM_PARAMVAR(_runtimeLength, "RuntimeLength", "The length of the runtime sized array of the buffer, if any",
                 {shards::CoreInfo::NoneType, shards::CoreInfo::IntType, shards::CoreInfo::IntVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_buffer), PARAM_IMPL_FOR(_runtimeLength));

  UniqueId _lastBufferId;
  std::optional<DerivedLayoutInfo> _dli;

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    if (_buffer.isNone())
      throw std::runtime_error("ReadBuffer: Buffer is required");
    return data.inputType;
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto &buffer = shards::varAsObjectChecked<SHBuffer>(_buffer.get(), ShardsTypes::Buffer);
    if (!_dli || _lastBufferId != buffer.buffer->getId()) {
      _dli = DerivedLayoutInfo::make(buffer.designatedAddressSpace, buffer.type);
      _lastBufferId = buffer.buffer->getId();
    }

    if (_dli->runtimeSizedField) {
      size_t rtLength = _runtimeLength.isNone() ? 0 : _runtimeLength.get().payload.intValue;
      size_t newBufferSize = shader::runtimeBufferSizeHelper(_dli->layout, *_dli->runtimeSizedField, rtLength);
      if (newBufferSize != buffer.data.getLength()) {
        ZoneScopedN("GFX.ResizeCPUBuffer");
        std::vector<uint8_t> newBuffer(buffer.data.getData(), buffer.data.getData() + buffer.data.getLength());
        newBuffer.resize(newBufferSize);
        buffer.data = gfx::ImmutableSharedBuffer(std::move(newBuffer));
      }
    }

    shader::BufferSerializer ser(buffer.getDataMut(), buffer.data.getLength());
    shader::LayoutTraverser2 lt(_dli->layout, _dli->structLayoutLookup, buffer.designatedAddressSpace);
    SHBufferVisitor<CallSerialize> visitor(ser, lt);

    {
      ZoneScopedN("GFX.WriteBuffer/visit");
      visitor.visit(const_cast<SHVar &>(input));
    }

    buffer.buffer->update(buffer.data);

    return input;
  }
};
} // namespace gfx

namespace shards {
SHARDS_REGISTER_FN(buffer) {
  using namespace gfx;
  REGISTER_SHARD("GFX.Buffer", BufferShard);
  REGISTER_SHARD("GFX.WriteBuffer", WriteBuffer);
  REGISTER_SHARD("GFX.ReadBuffer", ReadBuffer);
}
} // namespace shards

#endif /* D6E6F079_B676_4C2F_8B5C_032302417557 */
