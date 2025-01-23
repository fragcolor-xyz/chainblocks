#ifndef SH_EXTRA_GFX
#define SH_EXTRA_GFX

#include <shards/common_types.hpp>
#include <shards/core/foundation.hpp>
#include <gfx/drawable.hpp>
#include <gfx/fwd.hpp>
#include <gfx/pipeline_step.hpp>
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
#include <input/input_stack.hpp>
#include <shards/core/exposed_type_utils.hpp>
#include <vector>
#include <shards/shards.hpp>

namespace gfx {

constexpr uint32_t VendorId = shards::CoreCC;

struct Window;
struct Renderer;



struct SHDrawable {
  using Variant = std::variant<MeshDrawable::Ptr, MeshTreeDrawable::Ptr>;
  Variant drawable;
  std::unordered_map<std::string, Animation> animations;
  std::unordered_map<std::string, MaterialPtr> materials;
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


struct GraphicsContext {
  static constexpr uint32_t TypeId = 'mwnd';
  static inline SHTypeInfo Type{SHType::Object, {.object = {.vendorId = VendorId, .typeId = TypeId}}};
  static inline const char VariableName[] = "GFX.Context";
  static inline const SHOptionalString VariableDescription = SHCCSTR("The graphics context.");
  static inline SHExposedTypeInfo VariableInfo = shards::ExposedInfo::ProtectedVariable(VariableName, VariableDescription, Type);

  std::shared_ptr<Context> context;
  std::shared_ptr<Window> window;
  std::shared_ptr<Renderer> renderer;

  double time{};
  float deltaTime{};
  bool frameInProgress{};

  ::gfx::Context &getContext();
  ::gfx::Window &getWindow();
};

using RequiredGraphicsContext =
    shards::RequiredContextVariable<GraphicsContext, GraphicsContext::Type, GraphicsContext::VariableName>;
using OptionalGraphicsContext =
    shards::RequiredContextVariable<GraphicsContext, GraphicsContext::Type, GraphicsContext::VariableName, false>;

struct GraphicsRendererContext {
  static constexpr uint32_t TypeId = 'grcx';
  static inline SHTypeInfo Type{SHType::Object, {.object = {.vendorId = VendorId, .typeId = TypeId}}};
  static inline const char VariableName[] = "GFX.RendererContext";
  static inline const SHOptionalString VariableDescription = SHCCSTR("The graphics renderer context.");
  static inline SHExposedTypeInfo VariableInfo = shards::ExposedInfo::ProtectedVariable(VariableName, VariableDescription, Type);

  Renderer *renderer{};

  // Overridable callback for rendering
  std::function<void(ViewPtr view, const PipelineSteps &pipelineSteps)> render;
};

typedef shards::RequiredContextVariable<GraphicsRendererContext, GraphicsRendererContext::Type,
                                        GraphicsRendererContext::VariableName>
    RequiredGraphicsRendererContext;
typedef shards::RequiredContextVariable<GraphicsRendererContext, GraphicsRendererContext::Type,
                                        GraphicsRendererContext::VariableName, false>
    OptionalGraphicsRendererContext;

} // namespace gfx

#endif // SH_EXTRA_GFX
