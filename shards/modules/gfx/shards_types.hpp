#ifndef E452293C_6700_4675_8B6E_5293674E0A33
#define E452293C_6700_4675_8B6E_5293674E0A33

#include "shards/shards.h"
#include <shards/common_types.hpp>
#include <shards/core/foundation.hpp>
#include <shards/core/object_type.hpp>
#include <shards/shards.hpp>
#include <gfx/enums.hpp>
#include <gfx/fwd.hpp>
#include <gfx/pipeline_step.hpp>
#include <gfx/shader/entry_point.hpp>
#include <gfx/shader/types.hpp>
#include <gfx/shader/struct_layout.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/drawables/mesh_tree_drawable.hpp>
#include <gfx/gltf/animation.hpp>
#include <gfx/texture.hpp>
#include <gfx/feature.hpp>
#include <gfx/view.hpp>
#include <memory>
#include <variant>
#include <vector>

namespace gfx {

struct SHDrawable {
  using Variant = std::variant<MeshDrawable::Ptr, MeshTreeDrawable::Ptr>;
  Variant drawable;
  std::unordered_map<std::string, Animation> animations;
  std::unordered_map<std::string, MaterialPtr> materials;
  linalg::vec<float, 3> boundsMin = {0, 0, 0};
  linalg::vec<float, 3> boundsMax = {0, 0, 0};

  bool rootNodeWrapped{};

  void assign(const std::shared_ptr<IDrawable> &generic) {
    if (auto mesh = std::dynamic_pointer_cast<MeshDrawable>(generic)) {
      this->drawable = mesh;
    } else if (auto meshTree = std::dynamic_pointer_cast<MeshTreeDrawable>(generic)) {
      this->drawable = meshTree;
    } else {
      throw std::logic_error("unsupported");
    }
  }
};

struct SHBuffer {
  shader::AddressSpace designatedAddressSpace;
  shader::StructType type;
  size_t runtimeSize{};
  ImmutableSharedBuffer data;
  BufferPtr buffer;

  uint8_t *getDataMut() { return const_cast<uint8_t *>(data.getData()); }
};

struct SHView {
  ViewPtr view;

  static std::vector<uint8_t> serialize(const SHView &);
  static SHView deserialize(const std::string_view &);
};

struct SHMaterial {
  MaterialPtr material;
};

struct SHRenderTarget {
  RenderTargetPtr renderTarget;
};

struct SHDrawQueue {
  DrawQueuePtr queue;
};

struct SHSampler {
  // Unused for now
};

constexpr uint32_t VendorId = shards::CoreCC;

namespace detail {
using namespace shards;
// NOTE: This needs to be a struct ensure correct initialization order under clang
struct Container {
#define OBJECT(_id, _displayName, _definedAs, ...)                                                                              \
  static constexpr uint32_t SH_CONCAT(_definedAs, TypeId) = uint32_t(_id);                                                      \
  static inline Type _definedAs{{SHType::Object, {.object = {.vendorId = VendorId, .typeId = SH_CONCAT(_definedAs, TypeId)}}}}; \
  static inline ObjectVar<__VA_ARGS__> SH_CONCAT(_definedAs, ObjectVar){_displayName, VendorId, SH_CONCAT(_definedAs, TypeId)};

  OBJECT('draw', "GFX.Drawable", Drawable, SHDrawable, nullptr, nullptr, nullptr, /*ThreadSafe*/ true)
  OBJECT('mesh', "GFX.Mesh", Mesh, MeshPtr)
  OBJECT('dque', "GFX.DrawQueue", DrawQueue, SHDrawQueue, nullptr, nullptr, nullptr, /*ThreadSafe*/ true)
  OBJECT('tex_', "GFX.Texture2D", Texture, TexturePtr, nullptr, nullptr, nullptr, /*ThreadSafe*/ true)
  OBJECT('texc', "GFX.TextureCube", TextureCube, TexturePtr, nullptr, nullptr, nullptr, /*ThreadSafe*/ true)
  OBJECT('smpl', "GFX.Sampler", Sampler, SHSampler)
  OBJECT('gbuf', "GFX.Buffer", Buffer, SHBuffer, nullptr, nullptr, nullptr, /*ThreadSafe*/ true)
  OBJECT('__RT', "GFX.RenderTarget", RenderTarget, SHRenderTarget)

  DECL_ENUM_INFO(gfx::WindingOrder, WindingOrder, "Specifies the order in which vertices are defined to determine the front face of a polygon. Affects how faces are culled in rendering.", '_e0');
  DECL_ENUM_INFO(gfx::ShaderFieldBaseType, ShaderFieldBaseType, "Defines the fundamental data types used in shader fields. Determines how shader variables are interpreted and stored.", '_e1');
  DECL_ENUM_INFO(gfx::ProgrammableGraphicsStage, ProgrammableGraphicsStage, "Identifies the programmable stages in the graphics pipeline. Used to specify which part of the rendering process a shader program targets.", '_e2');
  DECL_ENUM_INFO(shader::DependencyType, DependencyType, "Specifies the type of dependency in shader compilation. Determines how different shader components relate to and depend on each other.", '_e3');

  enum class BlendFactor_ {
    Zero = WGPUBlendFactor_Zero,
    One = WGPUBlendFactor_One,
    Src = WGPUBlendFactor_Src,
    OneMinusSrc = WGPUBlendFactor_OneMinusSrc,
    SrcAlpha = WGPUBlendFactor_SrcAlpha,
    OneMinusSrcAlpha = WGPUBlendFactor_OneMinusSrcAlpha,
    Dst = WGPUBlendFactor_Dst,
    OneMinusDst = WGPUBlendFactor_OneMinusDst,
    DstAlpha = WGPUBlendFactor_DstAlpha,
    OneMinusDstAlpha = WGPUBlendFactor_OneMinusDstAlpha,
    SrcAlphaSaturated = WGPUBlendFactor_SrcAlphaSaturated,
    Constant = WGPUBlendFactor_Constant,
    OneMinusConstant = WGPUBlendFactor_OneMinusConstant,
  };
  DECL_ENUM_INFO(BlendFactor_, BlendFactor, "Defines how color components are combined during blending operations. Used to control transparency and color mixing in rendering.", '_e4');

  enum class BlendOperation_ {
    Add = WGPUBlendOperation_Add,
    Subtract = WGPUBlendOperation_Subtract,
    ReverseSubtract = WGPUBlendOperation_ReverseSubtract,
    Min = WGPUBlendOperation_Min,
    Max = WGPUBlendOperation_Max,
  };
  DECL_ENUM_INFO(BlendOperation_, BlendOperation, "Specifies the mathematical operation used to combine source and destination colors during blending. Affects how overlapping objects are rendered.", '_e5');

  enum class FilterMode_ {
    Nearest = WGPUFilterMode_Nearest,
    Linear = WGPUFilterMode_Linear,
  };
  DECL_ENUM_INFO(FilterMode_, FilterMode, "Determines the method used for sampling textures. Affects the appearance of textures when scaled or viewed at different distances.", '_e6');

  enum class CompareFunction_ {
    Undefined = WGPUCompareFunction_Undefined,
    Never = WGPUCompareFunction_Never,
    Less = WGPUCompareFunction_Less,
    LessEqual = WGPUCompareFunction_LessEqual,
    Greater = WGPUCompareFunction_Greater,
    GreaterEqual = WGPUCompareFunction_GreaterEqual,
    Equal = WGPUCompareFunction_Equal,
    NotEqual = WGPUCompareFunction_NotEqual,
    Always = WGPUCompareFunction_Always,
  };
  DECL_ENUM_INFO(CompareFunction_, CompareFunction, "Defines the comparison operation used in depth and stencil tests. Controls how fragments are accepted or rejected during rendering.", '_e7');

  enum class ColorMask_ {
    None = WGPUColorWriteMask_None,
    Red = WGPUColorWriteMask_Red,
    Green = WGPUColorWriteMask_Green,
    Blue = WGPUColorWriteMask_Blue,
    Alpha = WGPUColorWriteMask_Alpha,
    All = WGPUColorWriteMask_All,
  };
  DECL_ENUM_INFO(ColorMask_, ColorMask, "Specifies which color channels are affected by rendering operations. Used to control which components of a color buffer can be modified.", '_e8');

  enum class TextureType_ {
    Default = 0,
    Int,
    UInt,
    UNorm,
    UNormSRGB,
    SNorm,
    Float,
  };
  DECL_ENUM_INFO(TextureType_, TextureType, "Defines the data format and interpretation of texture data. Affects how texture values are stored and processed in shaders.", '_e9');

  enum class SortMode_ : uint8_t {
    Batch = uint8_t(SortMode::Batch),
    Queue = uint8_t(SortMode::Queue),
    BackToFront = uint8_t(SortMode::BackToFront),
  };
  DECL_ENUM_INFO(SortMode_, SortMode, "Determines how rendering commands are sorted before execution. Affects the order in which objects are drawn, impacting performance and visual output.", '_e10');

  enum class TextureFormat_ : uint32_t {
    // Undefined = WGPUTextureFormat_Undefined,
    R8Unorm = WGPUTextureFormat_R8Unorm,
    R8Snorm = WGPUTextureFormat_R8Snorm,
    R8Uint = WGPUTextureFormat_R8Uint,
    R8Sint = WGPUTextureFormat_R8Sint,
    R16Uint = WGPUTextureFormat_R16Uint,
    R16Sint = WGPUTextureFormat_R16Sint,
    R16Float = WGPUTextureFormat_R16Float,
    RG8Unorm = WGPUTextureFormat_RG8Unorm,
    RG8Snorm = WGPUTextureFormat_RG8Snorm,
    RG8Uint = WGPUTextureFormat_RG8Uint,
    RG8Sint = WGPUTextureFormat_RG8Sint,
    R32Float = WGPUTextureFormat_R32Float,
    R32Uint = WGPUTextureFormat_R32Uint,
    R32Sint = WGPUTextureFormat_R32Sint,
    RG16Uint = WGPUTextureFormat_RG16Uint,
    RG16Sint = WGPUTextureFormat_RG16Sint,
    RG16Float = WGPUTextureFormat_RG16Float,
    RGBA8Unorm = WGPUTextureFormat_RGBA8Unorm,
    RGBA8UnormSrgb = WGPUTextureFormat_RGBA8UnormSrgb,
    RGBA8Snorm = WGPUTextureFormat_RGBA8Snorm,
    RGBA8Uint = WGPUTextureFormat_RGBA8Uint,
    RGBA8Sint = WGPUTextureFormat_RGBA8Sint,
    BGRA8Unorm = WGPUTextureFormat_BGRA8Unorm,
    BGRA8UnormSrgb = WGPUTextureFormat_BGRA8UnormSrgb,
    RGB10A2Unorm = WGPUTextureFormat_RGB10A2Unorm,
    RG11B10Ufloat = WGPUTextureFormat_RG11B10Ufloat,
    RGB9E5Ufloat = WGPUTextureFormat_RGB9E5Ufloat,
    RG32Float = WGPUTextureFormat_RG32Float,
    RG32Uint = WGPUTextureFormat_RG32Uint,
    RG32Sint = WGPUTextureFormat_RG32Sint,
    RGBA16Uint = WGPUTextureFormat_RGBA16Uint,
    RGBA16Sint = WGPUTextureFormat_RGBA16Sint,
    RGBA16Float = WGPUTextureFormat_RGBA16Float,
    RGBA32Float = WGPUTextureFormat_RGBA32Float,
    RGBA32Uint = WGPUTextureFormat_RGBA32Uint,
    RGBA32Sint = WGPUTextureFormat_RGBA32Sint,
    Stencil8 = WGPUTextureFormat_Stencil8,
    Depth16Unorm = WGPUTextureFormat_Depth16Unorm,
    Depth24Plus = WGPUTextureFormat_Depth24Plus,
    Depth24PlusStencil8 = WGPUTextureFormat_Depth24PlusStencil8,
    // Not supported on WGPU
    // Depth24UnormStencil8 = WGPUTextureFormat_Depth24UnormStencil8,
    Depth32Float = WGPUTextureFormat_Depth32Float,
    Depth32FloatStencil8 = WGPUTextureFormat_Depth32FloatStencil8,
    // BC1RGBAUnorm = WGPUTextureFormat_BC1RGBAUnorm,
    // BC1RGBAUnormSrgb = WGPUTextureFormat_BC1RGBAUnormSrgb,
    // BC2RGBAUnorm = WGPUTextureFormat_BC2RGBAUnorm,
    // BC2RGBAUnormSrgb = WGPUTextureFormat_BC2RGBAUnormSrgb,
    // BC3RGBAUnorm = WGPUTextureFormat_BC3RGBAUnorm,
    // BC3RGBAUnormSrgb = WGPUTextureFormat_BC3RGBAUnormSrgb,
    // BC4RUnorm = WGPUTextureFormat_BC4RUnorm,
    // BC4RSnorm = WGPUTextureFormat_BC4RSnorm,
    // BC5RGUnorm = WGPUTextureFormat_BC5RGUnorm,
    // BC5RGSnorm = WGPUTextureFormat_BC5RGSnorm,
    // BC6HRGBUfloat = WGPUTextureFormat_BC6HRGBUfloat,
    // BC6HRGBFloat = WGPUTextureFormat_BC6HRGBFloat,
    // BC7RGBAUnorm = WGPUTextureFormat_BC7RGBAUnorm,
    // BC7RGBAUnormSrgb = WGPUTextureFormat_BC7RGBAUnormSrgb,
    // ETC2RGB8Unorm = WGPUTextureFormat_ETC2RGB8Unorm,
    // ETC2RGB8UnormSrgb = WGPUTextureFormat_ETC2RGB8UnormSrgb,
    // ETC2RGB8A1Unorm = WGPUTextureFormat_ETC2RGB8A1Unorm,
    // ETC2RGB8A1UnormSrgb = WGPUTextureFormat_ETC2RGB8A1UnormSrgb,
    // ETC2RGBA8Unorm = WGPUTextureFormat_ETC2RGBA8Unorm,
    // ETC2RGBA8UnormSrgb = WGPUTextureFormat_ETC2RGBA8UnormSrgb,
    // EACR11Unorm = WGPUTextureFormat_EACR11Unorm,
    // EACR11Snorm = WGPUTextureFormat_EACR11Snorm,
    // EACRG11Unorm = WGPUTextureFormat_EACRG11Unorm,
    // EACRG11Snorm = WGPUTextureFormat_EACRG11Snorm,
    // ASTC4x4Unorm = WGPUTextureFormat_ASTC4x4Unorm,
    // ASTC4x4UnormSrgb = WGPUTextureFormat_ASTC4x4UnormSrgb,
    // ASTC5x4Unorm = WGPUTextureFormat_ASTC5x4Unorm,
    // ASTC5x4UnormSrgb = WGPUTextureFormat_ASTC5x4UnormSrgb,
    // ASTC5x5Unorm = WGPUTextureFormat_ASTC5x5Unorm,
    // ASTC5x5UnormSrgb = WGPUTextureFormat_ASTC5x5UnormSrgb,
    // ASTC6x5Unorm = WGPUTextureFormat_ASTC6x5Unorm,
    // ASTC6x5UnormSrgb = WGPUTextureFormat_ASTC6x5UnormSrgb,
    // ASTC6x6Unorm = WGPUTextureFormat_ASTC6x6Unorm,
    // ASTC6x6UnormSrgb = WGPUTextureFormat_ASTC6x6UnormSrgb,
    // ASTC8x5Unorm = WGPUTextureFormat_ASTC8x5Unorm,
    // ASTC8x5UnormSrgb = WGPUTextureFormat_ASTC8x5UnormSrgb,
    // ASTC8x6Unorm = WGPUTextureFormat_ASTC8x6Unorm,
    // ASTC8x6UnormSrgb = WGPUTextureFormat_ASTC8x6UnormSrgb,
    // ASTC8x8Unorm = WGPUTextureFormat_ASTC8x8Unorm,
    // ASTC8x8UnormSrgb = WGPUTextureFormat_ASTC8x8UnormSrgb,
    // ASTC10x5Unorm = WGPUTextureFormat_ASTC10x5Unorm,
    // ASTC10x5UnormSrgb = WGPUTextureFormat_ASTC10x5UnormSrgb,
    // ASTC10x6Unorm = WGPUTextureFormat_ASTC10x6Unorm,
    // ASTC10x6UnormSrgb = WGPUTextureFormat_ASTC10x6UnormSrgb,
    // ASTC10x8Unorm = WGPUTextureFormat_ASTC10x8Unorm,
    // ASTC10x8UnormSrgb = WGPUTextureFormat_ASTC10x8UnormSrgb,
    // ASTC10x10Unorm = WGPUTextureFormat_ASTC10x10Unorm,
    // ASTC10x10UnormSrgb = WGPUTextureFormat_ASTC10x10UnormSrgb,
    // ASTC12x10Unorm = WGPUTextureFormat_ASTC12x10Unorm,
    // ASTC12x10UnormSrgb = WGPUTextureFormat_ASTC12x10UnormSrgb,
    // ASTC12x12Unorm = WGPUTextureFormat_ASTC12x12Unorm,
    // ASTC12x12UnormSrgb = WGPUTextureFormat_ASTC12x12UnormSrgb,
    // Force32 = WGPUTextureFormat_Force32,
  };
  DECL_ENUM_INFO(TextureFormat_, TextureFormat, "Specifies the format of texture data, including color depth and component layout. Determines how texture data is stored and interpreted by the GPU.", '_e11');

  enum class TextureDimension_ {
    D1,
    D2,
    Cube,
  };
  DECL_ENUM_INFO(TextureDimension_, TextureDimension, "Defines the dimensionality of a texture. Affects how textures are sampled and used in shaders.", '_e12');

  enum class TextureAddressing_ {
    Repeat = WGPUAddressMode_Repeat,
    MirrorRepeat = WGPUAddressMode_MirrorRepeat,
    ClampToEdge = WGPUAddressMode_ClampToEdge,
  };
  DECL_ENUM_INFO(TextureAddressing_, TextureAddressing, "Specifies how texture coordinates outside the [0, 1] range are handled. Affects texture wrapping and tiling behavior.", '_e13');

  enum class TextureFiltering_ {
    Nearest = WGPUFilterMode_Nearest,
    Linear = WGPUFilterMode_Linear,
  };
  DECL_ENUM_INFO(TextureFiltering_, TextureFiltering, "Determines how textures are sampled when scaled. Affects the visual quality and performance of texture rendering.", '_e14');

  DECL_ENUM_INFO(::gfx::OrthographicSizeType, OrthographicSizeType, "Specifies how the size of an orthographic projection is defined. Affects how 2D scenes are rendered and scaled.", '_e15');

  DECL_ENUM_INFO(::gfx::BindGroupId, BindGroupId, "Identifies different bind groups in a shader. Used to organize and access shader resources efficiently.", '_e16');

  DECL_ENUM_INFO(::gfx::TextureSampleType, TextureSampleType, "Defines how a texture is sampled in a shader. Affects how texture data is read and interpreted in shader programs.", '_e17');

  DECL_ENUM_INFO(::gfx::shader::AddressSpace, BufferAddressSpace, "Specifies the memory space for buffer resources in shaders. Determines how buffer data is accessed and optimized by the GPU.", '_e18');

  OBJECT('feat', "GFX.Feature", Feature, FeaturePtr)
  static inline Type FeatureSeq = Type::SeqOf(Feature);
  static inline Type FeatureVarType = Type::VariableOf(Feature);
  static inline Type FeatureVarSeq = Type::SeqOf(FeatureVarType);

  OBJECT('pips', "GFX.PipelineStep", PipelineStep, PipelineStepPtr)
  static inline Type PipelineStepSeq = Type::SeqOf(PipelineStep);

  OBJECT('view', "GFX.View", View, SHView, &SHView::serialize, &SHView::deserialize, nullptr);
  static inline Type ViewSeq = Type::SeqOf(View);

  OBJECT('mat_', "GFX.Material", Material, SHMaterial)

  static inline Types TextureTypes = {{
      Texture,
      TextureCube,
  }};

  static inline Type TextureVarType = Type::VariableOf(TextureTypes);

  static inline Types ShaderParamTypes{TextureTypes,
                                       {
                                           Type::VariableOf(Buffer),
                                           CoreInfo::Float4x4Type,
                                           CoreInfo::Float4Type,
                                           CoreInfo::Float3Type,
                                           CoreInfo::Float2Type,
                                           CoreInfo::FloatType,
                                           CoreInfo::IntType,
                                           CoreInfo::Int2Type,
                                           CoreInfo::Int3Type,
                                           CoreInfo::Int4Type,
                                       }};

  static inline Types ShaderParamOrVarTypes{ShaderParamTypes, {Type::VariableOf(ShaderParamTypes)}};

  // Shared drawable parameters
  static inline Type TransformVarType = Type::VariableOf(CoreInfo::Float4x4Type);

  // Valid types for shader :Params
  static inline Type ShaderParamTable = Type::TableOf(ShaderParamOrVarTypes);

  static inline ParameterInfo TransformParameterInfo{
      "Transform", SHCCSTR("The transform to use"), {CoreInfo::NoneType, TransformVarType}};

  static inline ParameterInfo MaterialParameterInfo{
      "Material", SHCCSTR("The material"), {CoreInfo::NoneType, Type::VariableOf(Material)}};

  static inline ParameterInfo ParamsParameterInfo{"Params",
                                                  SHCCSTR("Shader parameters for this drawable"),
                                                  {CoreInfo::NoneType, ShaderParamTable, Type::VariableOf(ShaderParamTable)}};

  static inline ParameterInfo FeaturesParameterInfo{
      "Features", SHCCSTR("Features to attach to this drawable"), {CoreInfo::NoneType, FeatureSeq, Type::VariableOf(FeatureSeq)}};

  static inline Type OutputsType = Type::SeqOf(CoreInfo::AnyTableType);

  static inline ParameterInfo OutputsParameterInfo{
      "Outputs", SHCCSTR("The outputs to render into"), {CoreInfo::NoneType, OutputsType, Type::VariableOf(OutputsType)}};

  static inline ParameterInfo OutputScaleParameterInfo{
      "OutputScale", SHCCSTR("The scale that the output should be rendered as"), {CoreInfo::NoneType, CoreInfo::AnyType}};

  static inline ParameterInfo NameParameterInfo{
      "Name", SHCCSTR("A name for this pass, to aid in debugging"), {CoreInfo::NoneType, CoreInfo::StringType}};

#undef ENUM
#undef OBJECT
};
} // namespace detail
using ShardsTypes = detail::Container;

} // namespace gfx

#endif /* E452293C_6700_4675_8B6E_5293674E0A33 */
