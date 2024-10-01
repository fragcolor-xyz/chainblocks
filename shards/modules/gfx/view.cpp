#include "gfx.hpp"
#include <gfx/view.hpp>
#include <gfx/renderer.hpp>
#include <gfx/window.hpp>
#include <gfx/fmt.hpp>
#include <input/input_stack.hpp>
#include <shards/common_types.hpp>
#include "shards_utils.hpp"
#include <shards/core/foundation.hpp>
#include "gfx/error_utils.hpp"
#include "gfx/fwd.hpp"
#include "gfx/render_target.hpp"
#include <shards/core/params.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/shards.h>
#include "shards_utils.hpp"
#include "drawable_utils.hpp"
#include <shards/modules/inputs/inputs.hpp>
#include <shards/core/serialization/linalg.hpp>
#include "window.hpp"

using namespace shards;

namespace shards {
template <SerializerStream T> void serde(T &stream, gfx::FovDirection &v) { serdeAs<int8_t>(stream, v); }
template <SerializerStream T> void serde(T &stream, gfx::OrthographicSizeType &v) { serdeAs<int8_t>(stream, v); }
template <SerializerStream T> void serde(T &stream, gfx::ViewPerspectiveProjection &v) {
  serde(stream, v.far);
  serde(stream, v.near);
  serde(stream, v.fov);
  serde(stream, v.fovType);
}
template <SerializerStream T> void serde(T &stream, gfx::ViewOrthographicProjection &v) {
  serde(stream, v.far);
  serde(stream, v.near);
  serde(stream, v.size);
  serde(stream, v.sizeType);
}
} // namespace shards

namespace gfx {

std::vector<uint8_t> SHView::serialize(const SHView &view_) {
  BufferWriter writer;

  auto &view = view_.view;
  serde(writer, view->view);
  uint8_t index = view->proj.index();
  serde(writer, index);
  std::visit([&](auto &arg) { serde(writer, arg); }, view->proj);

  return std::move(writer._buffer);
}

SHView SHView::deserialize(const std::string_view &data) {
  BytesReader reader(data);

  ViewPtr newView = std::make_shared<gfx::View>();
  serde(reader, newView->view);
  uint8_t index{};
  serde(reader, index);
  switch (index) {
  case 1:
    serde(reader, newView->proj.emplace<ViewPerspectiveProjection>());
    break;
  case 2:
    serde(reader, newView->proj.emplace<ViewOrthographicProjection>());
    break;
  case 3:
    serde(reader, newView->proj.emplace<float4x4>());
    break;
  default:
    break;
  }

  return SHView{
      .view = newView,
  };
}

struct ViewShard {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return ShardsTypes::View; }

  static SHOptionalString help() {
    return SHCCSTR("This shard creates and manages a view object, which defines and controls the view or camera settings for a "
                   "rendering pipeline.");
  }

  static SHOptionalString inputHelp() {
    return DefaultHelpText::InputHelpIgnored;
  }

  static SHOptionalString outputHelp() {
    return SHCCSTR("The view object created.");
  }

  PARAM_PARAMVAR(_viewTransform, "View", "The view matrix.", {CoreInfo::NoneType, Type::VariableOf(CoreInfo::Float4x4Type)});
  PARAM_PARAMVAR(_ortho, "OrthographicSize", "The orthographic size. (Implies orthographic projection)",
                 {CoreInfo::NoneType, CoreInfo::FloatType, Type::VariableOf(CoreInfo::FloatType)});
  PARAM_PARAMVAR(_orthoType, "OrthographicSizeType", "The type of orthographic size. (Implies orthographic projection)",
                 {CoreInfo::NoneType, ShardsTypes::OrthographicSizeTypeEnumInfo::Type});
  PARAM_PARAMVAR(_fov, "Fov", "The vertical field of view. (In radians. Implies perspective projection)",
                 {CoreInfo::NoneType, CoreInfo::FloatType, Type::VariableOf(CoreInfo::FloatType)});
  PARAM_PARAMVAR(_near, "Near", "Near clipping distance",
                 {CoreInfo::NoneType, CoreInfo::FloatType, Type::VariableOf(CoreInfo::FloatType)});
  PARAM_PARAMVAR(_far, "Far", "Far clipping distance",
                 {CoreInfo::NoneType, CoreInfo::FloatType, Type::VariableOf(CoreInfo::FloatType)});
  PARAM_IMPL(PARAM_IMPL_FOR(_viewTransform), PARAM_IMPL_FOR(_fov), PARAM_IMPL_FOR(_orthoType), PARAM_IMPL_FOR(_ortho),
             PARAM_IMPL_FOR(_near), PARAM_IMPL_FOR(_far));

  SHView *_view;
  bool _isPerspective = false;

  void warmup(SHContext *context) {
    _view = ShardsTypes::ViewObjectVar.New();

    PARAM_WARMUP(context);
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);

    if (_view) {
      ShardsTypes::ViewObjectVar.Release(_view);
      _view = nullptr;
    }
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    bool isOrtho = !_ortho.isNone() || !_orthoType.isNone();
    _isPerspective = !_fov.isNone();

    // Default to perspecitive projection if nothing is set
    if (!isOrtho && !_isPerspective) {
      _isPerspective = true;
    }

    if (isOrtho && _isPerspective) {
      throw formatException("View can not have both orthographic and perspective projection");
    }
    if (isOrtho && _ortho.isNone()) {
      throw formatException("View is missing OrthographicSize");
    }

    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    ViewPtr &view = _view->view;
    if (!view)
      view = std::make_shared<View>();

    if (_viewTransform->valueType != SHType::None) {
      view->view = shards::Mat4(_viewTransform.get());
    }

    float near = 0.1f;
    float far = 10000.0f;

    Var &nearVar = (Var &)_near.get();
    if (!nearVar.isNone())
      near = float(nearVar);

    Var &farVar = (Var &)_far.get();
    if (!farVar.isNone())
      far = float(farVar);

    if (_isPerspective) {
      auto proj = ViewPerspectiveProjection{
          .near = near,
          .far = far,
      };

      Var fovVar = (Var &)_fov.get();
      if (!fovVar.isNone())
        proj.fov = float(fovVar);

      view->proj = proj;
    } else {
      auto proj = ViewOrthographicProjection{
          .size = float((Var &)_ortho.get()),
          .sizeType = OrthographicSizeType::Horizontal,
          .near = near,
          .far = far,

      };

      Var typeVar = (Var &)_orthoType.get();
      if (!typeVar.isNone()) {
        proj.sizeType = OrthographicSizeType(int(typeVar));
      }

      view->proj = proj;
    }

    return ShardsTypes::ViewObjectVar.Get(_view);
  }
};

// :Textures {:outputName .textureVar}
// Face indices are ordered   X+ X- Y+ Y- Z+ Z-
// :Textures {:outputName {:Texture .textureVar :Face 0}}
// Mip map indices starting at 0, can be combined with cubemap faces
// :Textures {:outputName {:Texture .textureVar :Mip 1 :Face 0}}
struct RenderIntoShard {
  // static inline std::array<SHString, 4> TextureSubresourceTableKeys{"Texture", "Face", "Mip", ""};
  // static inline shards::Types TextureSubresourceTableTypes{Type::VariableOf(ShardsTypes::Texture), CoreInfo::IntType,
  //                                                          Type::VariableOf(CoreInfo::IntType)};
  // static inline shards::Type TextureSubresourceTable = Type::TableOf(TextureSubresourceTableTypes,
  // TextureSubresourceTableKeys);
  // Can not check, support any table for now
  static inline shards::Type TextureSubresourceTable = CoreInfo::AnyTableType;

  static inline shards::Types AttachmentTableTypes{TextureSubresourceTable, Type::VariableOf(ShardsTypes::Texture)};
  static inline shards::Type AttachmentTable = Type::TableOf(AttachmentTableTypes);

  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static SHOptionalString help() { return SHCCSTR("Renders within a region of the screen and/or to a render target"); }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpPass; }
  static SHOptionalString outputHelp() { return DefaultHelpText::OutputHelpPass; }

  PARAM(OwnedVar, _textures, "Textures", "The textures to render into to create.", {CoreInfo::NoneType, AttachmentTable});
  PARAM(ShardsVar, _contents, "Contents", "The shards that will render into the given textures.", {CoreInfo::Shards});
  PARAM_PARAMVAR(_referenceSize, "Size", "The reference size. This will control the size of the render targets.",
                 {CoreInfo::NoneType, CoreInfo::Int2Type, CoreInfo::Int2VarType});
  PARAM_VAR(_matchOutputSize, "MatchOutputSize",
            "When true, the texture rendered into is automatically resized to match the output size.",
            {CoreInfo::NoneType, CoreInfo::BoolType});
  PARAM_PARAMVAR(_viewport, "Viewport", "The viewport.", {CoreInfo::NoneType, CoreInfo::Int4Type, CoreInfo::Int4VarType});
  PARAM_PARAMVAR(_windowRegion, "WindowRegion", "Sets the window region for input handling.",
                 {CoreInfo::NoneType, CoreInfo::Int4Type, CoreInfo::Int4VarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_textures), PARAM_IMPL_FOR(_contents), PARAM_IMPL_FOR(_referenceSize),
             PARAM_IMPL_FOR(_matchOutputSize), PARAM_IMPL_FOR(_viewport), PARAM_IMPL_FOR(_windowRegion));

  RequiredGraphicsRendererContext _graphicsRendererContext;
  shards::input::OptionalInputContext _inputContext;
  RenderTargetPtr _renderTarget;

  void warmup(SHContext *context) {
    _graphicsRendererContext.warmup(context);
    _inputContext.warmup(context);
    _renderTarget = std::make_shared<RenderTarget>();
    PARAM_WARMUP(context);
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _graphicsRendererContext.cleanup(context);
    _inputContext.cleanup(context);
    _renderTarget.reset();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);

    _requiredVariables.push_back(decltype(_graphicsRendererContext)::getExposedTypeInfo());

    if (findExposedVariable(data.shared, decltype(_inputContext)::getExposedTypeInfo().name)) {
      _requiredVariables.push_back(decltype(_inputContext)::getExposedTypeInfo());
    }

    return _contents.compose(data).outputType;
  }

  TextureSubResource applyAttachment(SHContext *shContext, const SHVar &input) {
    if (input.valueType == SHType::ContextVar) {
      ParamVar var{input};
      var.warmup(shContext);
      return varAsObjectChecked<TexturePtr>(var.get(), ShardsTypes::Texture);
    } else {
      checkType(input.valueType, SHType::Table, "Attachment");
      auto &table = input.payload.tableValue;

      Var textureVar;
      if (!getFromTable(shContext, table, Var("Texture"), textureVar)) {
        throw formatException("Texture is required");
      }

      TexturePtr texture = varToTexture(textureVar);

      uint8_t faceIndex{};
      Var faceIndexVar;
      if (getFromTable(shContext, table, Var("Face"), faceIndexVar)) {
        checkType(faceIndexVar.valueType, SHType::Int, "Face should be an integer(variable)");
        faceIndex = uint8_t(int(faceIndexVar));
      }

      uint8_t mipIndex{};
      Var mipIndexVar;
      if (getFromTable(shContext, table, Var("Mip"), mipIndexVar)) {
        checkType(mipIndexVar.valueType, SHType::Int, "Mip should be an integer(variable)");
        mipIndex = uint8_t(int(mipIndexVar));
      }

      return TextureSubResource(texture, faceIndex, mipIndex);
    }
  }

  void applyAttachments(SHContext *shContext, std::map<FastString, TextureSubResource> &outAttachments) {
    outAttachments.clear();
    if (!_textures->isNone()) {
      auto &table = _textures.payload.tableValue;
      ForEach(table, [&](SHVar &k, SHVar &v) {
        if (k.valueType != SHType::String)
          throw formatException("RenderInto attachment key should be a string");
        outAttachments.emplace(SHSTRVIEW(k), applyAttachment(shContext, v));
      });
    }
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    // Set the render target textures
    auto &rt = _renderTarget;
    auto &attachments = rt->attachments;

    applyAttachments(shContext, attachments);

    ViewStack::Item viewItem{};
    input::InputStack::Item inputItem;

    if (!_textures->isNone())
      viewItem.renderTarget = _renderTarget;

    GraphicsRendererContext &ctx = _graphicsRendererContext;
    const ViewStack &viewStack = ctx.renderer->getViewStack();

    Var referenceSizeVar = (Var &)_referenceSize.get();
    bool matchOutputSize = !_matchOutputSize->isNone() ? (bool)*_matchOutputSize : false;

    int2 referenceSize{};
    if (!referenceSizeVar.isNone()) {
      referenceSize = (int2)toInt2(referenceSizeVar);
    } else if (matchOutputSize) {
      referenceSize = viewStack.getOutput().referenceSize;
    } else {
      for (auto &attachment : attachments) {
        referenceSize = attachment.second.getResolution();
        if (referenceSize.x != 0 || referenceSize.y != 0)
          break;
      }
    }

    if (_inputContext) {
      // (optional) Push window region for input
      Var windowRegionVar{_windowRegion.get()};
      if (!windowRegionVar.isNone()) {
        // Read SHType::Float4 rect as (X0, Y0, X1, Y1)
        int4 region = toVec<int4>(windowRegionVar);

        // This is used for translating cursor inputs from window to view space
        // TODO: Adjust relative to parent region, for now always assume this is in global window coordinates
        inputItem.windowMapping = input::WindowSubRegion{
            .region = region,
        };

        // Set reference size to actual window size
        float2 regionSize = float2(region.z - region.x, region.w - region.y);
        referenceSize = int2(linalg::floor(float2(regionSize)));
      }
    }

    // (optional) set viewport rectangle
    Var viewportVar = (Var &)_viewport.get();
    if (!viewportVar.isNone()) {
      auto &v = viewportVar.payload.int4Value;
      // Read SHType::Float4 rect as (X0, Y0, X1, Y1)
      viewItem.viewport = Rect::fromCorners(v[0], v[1], v[2], v[3]);

      if (referenceSize.x == 0 || referenceSize.y == 0) {
        referenceSize = int2(viewItem.viewport->width, viewItem.viewport->height);
      }
    }

    if (referenceSize.x == 0 || referenceSize.y == 0)
      throw formatException("Invalid texture size for RenderInto: {}", referenceSize);

    viewItem.referenceSize = referenceSize;

    // Make render target fixed size
    rt->size = RenderTargetSize::Fixed{.size = referenceSize};

    ctx.renderer->pushView(std::move(viewItem));

    input::InputStack *inputStack = _inputContext ? &_inputContext->getInputStack() : nullptr;
    if (inputStack) {
      inputStack->push(std::move(inputItem));
    }

    SHVar contentOutput;
    _contents.activate(shContext, input, contentOutput);

    ctx.renderer->popView();

    if (inputStack) {
      inputStack->pop();
    }

    return contentOutput;
  }
};

struct ViewProjectionMatrixShard {
  static SHTypesInfo inputTypes() { return ShardsTypes::View; }
  static SHTypesInfo outputTypes() { return CoreInfo::Float4x4Type; }
  static SHOptionalString help() {
    return SHCCSTR("This shard takes the input view object and outputs its combined view-projection matrix, which encapsulates "
                   "both the view position and orientation(view matrix) and its orthographic projection(projection matrix).");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("A view object created by GFX.View."); }
  static SHOptionalString outputHelp() {
    return SHCCSTR("A 4x4 matrix representing the view object's combined view-projection matrix.");
  }

  PARAM_PARAMVAR(_viewSize, "ViewSize", "The size of the screen this view is being used with",
                 {CoreInfo::Float2Type, CoreInfo::Float2VarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_viewSize));

  Mat4 _result;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return CoreInfo::Float4x4Type;
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHView &shView = varAsObjectChecked<SHView>(input, ShardsTypes::View);
    auto &view = shView.view;
    auto projMatrix = view->getProjectionMatrix(toVec<float2>(_viewSize.get()));
    _result = linalg::mul(projMatrix, view->view);
    return _result;
  }
};

struct ViewMatrixShard {
  static SHTypesInfo inputTypes() { return ShardsTypes::View; }
  static SHTypesInfo outputTypes() { return CoreInfo::Float4x4Type; }
  static SHOptionalString help() {
    return SHCCSTR("This shard takes the input view object and outputs its view matrix, which represents the camera's position "
                   "and orientation in the world space.");
  }
  static SHOptionalString inputHelp() { return SHCCSTR("A view object created by GFX.View."); }
  static SHOptionalString outputHelp() { return SHCCSTR("A 4x4 matrix representing the view object's view matrix."); }

  PARAM_IMPL();

  Mat4 _result;

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return CoreInfo::Float4x4Type;
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHView &shView = varAsObjectChecked<SHView>(input, ShardsTypes::View);
    auto &view = shView.view;
    _result = view->view;
    return _result;
  }
};

struct ViewRangeShard {
  static SHTypesInfo inputTypes() { return ShardsTypes::View; }
  static SHTypesInfo outputTypes() { return CoreInfo::Float2Type; }
  static SHOptionalString help() {
    return SHCCSTR(
        "This shard takes the input view object and outputs a float2 representing the view's near and far clipping range. The "
        "first value in the float2 represents the near clipping range while the second value represents the far clipping range.");
  }

  static SHOptionalString inputHelp() { return SHCCSTR("A view object created by GFX.View."); }
  static SHOptionalString outputHelp() {
    return SHCCSTR("A float2 representing the view's near and far clipping range.");
  }

  PARAM_IMPL();

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  SHVar activate(SHContext *context, const SHVar &input) {
    SHView &shView = varAsObjectChecked<SHView>(input, ShardsTypes::View);
    auto &view = shView.view;
    float4x4 mat = view->getProjectionMatrix(float2(1.0f, 1.0f));
    float a = mat[2][2];
    float b = mat[3][2];
    float near = b / a;
    float far = b / (a + 1.0f);
    return toVar(float2(near, far));
  }
};

void registerViewShards() {
  REGISTER_SHARD("GFX.View", ViewShard);
  REGISTER_SHARD("GFX.RenderInto", RenderIntoShard);
  REGISTER_SHARD("GFX.ViewProjectionMatrix", ViewProjectionMatrixShard);
  REGISTER_SHARD("GFX.ViewMatrix", ViewMatrixShard);
  REGISTER_SHARD("GFX.ViewRange", ViewRangeShard);
}
} // namespace gfx
