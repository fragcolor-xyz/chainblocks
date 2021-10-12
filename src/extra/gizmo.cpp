/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

#include "gizmo.hpp"

using namespace chainblocks;

namespace chainblocks {
namespace Gizmo {

namespace Helper {
static const float identity[16] = {
    1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f,
};

// FIXME: copied from linal.cpp, should be shared
constexpr linalg::aliases::float3 AxisX{1.0, 0.0, 0.0};
constexpr linalg::aliases::float3 AxisY{0.0, 1.0, 0.0};
constexpr linalg::aliases::float3 AxisZ{0.0, 0.0, 1.0};
const float PI = 3.141592653589793238463f;
const float PI_2 = PI * 0.5f;
}; // namespace Helper

struct CubeView {
  static CBOptionalString help() {
    return CBCCSTR("Display a cube gizmo that can be used to control the "
                   "orientation of the current camera.");
  }

  static CBTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static CBOptionalString inputhelp() {
    return CBCCSTR("The input value is not used and will pass through.");
  }

  static CBTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static CBOptionalString outputhelp() {
    return CBCCSTR("The output of this block will be its input.");
  }

  static CBParametersInfo parameters() { return _params; }

  void setParam(int index, const CBVar &value) {
    switch (index) {
    case 0:
      _view = value;
      break;

    default:
      throw CBException("Parameter out of range.");
    }
  }

  CBVar getParam(int index) {
    switch (index) {
    case 0:
      return _view;

    default:
      throw CBException("Parameter out of range.");
    }
  }

  void cleanup() { _view.cleanup(); }

  void warmup(CBContext *context) { _view.warmup(context); }

  CBVar activate(CBContext *context, const CBVar &input) {
    auto &view = _view.get();
    // HACKish, I use Mat4 because the API hides some complexity, but it feels
    // unnecessary to do that many copies should manipulate the seq of float4
    // directly
    Mat4 mat{};
    mat = view;
    float arrView[16];
    Mat4::ToArrayUnsafe(mat, arrView);

    ImGuiIO &io = ::ImGui::GetIO();
    float viewManipulateRight =
        io.DisplaySize.x;        // TODO: make it a param (% of viewport?)
    float viewManipulateTop = 0; // TODO: make it a param
    ImGuizmo::ViewManipulate(
        arrView, 5, ImVec2(viewManipulateRight - 128, viewManipulateTop),
        ImVec2(128, 128), 0x10101010); // TODO: make offset and size a param

    chainblocks::cloneVar(view, Mat4::FromArrayUnsafe(arrView));

    return input;
  }

private:
  static inline Parameters _params = {
      {"ViewMatrix",
       CBCCSTR(
           "A variable that points to the vuew matrix that will be modified."),
       {Type::VariableOf(CoreInfo::Float4x4Type)}}};

  ParamVar _view{};
};

struct Grid : public BGFX::BaseConsumer {
  static CBOptionalString help() { return CBCCSTR("Displays a grid."); }

  static CBTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static CBOptionalString inputhelp() {
    return CBCCSTR("The input value is not used and will pass through.");
  }

  static CBTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static CBOptionalString outputhelp() {
    return CBCCSTR("The output of this block will be its input.");
  }

  static CBParametersInfo parameters() { return _params; }

  void setParam(int index, const CBVar &value) {
    switch (index) {
    case 0:
      _axis = value;
      break;
    case 1:
      _size = value;
      break;

    default:
      throw CBException("Parameter out of range.");
    }
  }

  CBVar getParam(int index) {
    switch (index) {
    case 0:
      return _axis;
    case 1:
      return _size;

    default:
      throw CBException("Parameter out of range.");
    }
  }

  void cleanup() {
    _axis.cleanup();
    _size.cleanup();

    BGFX::BaseConsumer::cleanup();
  }

  void warmup(CBContext *context) {
    BGFX::BaseConsumer::warmup(context);

    _axis.warmup(context);
    _size.warmup(context);
  }

  CBVar activate(CBContext *context, const CBVar &input) {

    linalg::aliases::float3 axis;
    switch (Enums::GridAxis(_axis.get().payload.enumValue)) {
    case Enums::GridAxis::X:
      axis = Helper::AxisX;
      break;
    case Enums::GridAxis::Y:
      axis = Helper::AxisY;
      break;
    case Enums::GridAxis::Z:
      axis = Helper::AxisZ;
      break;
    }

    auto *ctx =
        reinterpret_cast<BGFX::Context *>(_bgfxCtx->payload.objectValue);
    const auto &currentView = ctx->currentView();

    auto rot = linalg::rotation_quat(axis, Helper::PI_2);
    _matView = linalg::mul(currentView.view, linalg::rotation_matrix(rot));

    float arrView[16];
    float arrProj[16];
    Mat4::ToArrayUnsafe(_matView, arrView);
    Mat4::ToArrayUnsafe(currentView.proj, arrProj);

    ImGuiIO &io = ::ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    ImGuizmo::DrawGrid(arrView, arrProj, Helper::identity,
                       _size.get().payload.floatValue);

    return input;
  }

private:
  static inline Parameters _params = {
      // TODO: allow to pass a user (float3) as axis?)
      {"Axis",
       CBCCSTR("The normal axis of the grid plane"),
       {Enums::GridAxisType, Type::VariableOf(Enums::GridAxisType)}},
      {"Size",
       CBCCSTR("The size of the extents of the grid in all directions."),
       {CoreInfo::FloatType, CoreInfo::FloatVarType}},
  };

  // params
  ParamVar _axis{Var((CBEnum)Enums::GridAxis::Y)};
  ParamVar _size{Var(100.0f)};

  Mat4 _matView{};
};

struct Transform : public BGFX::BaseConsumer {
  static CBOptionalString help() {
    return CBCCSTR("Displays a gizmo that can be used to move, rotate or scale "
                   "an object.");
  }

  static CBTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static CBOptionalString inputhelp() {
    return CBCCSTR("The input value is not used and will pass through.");
  }

  static CBTypesInfo outputTypes() { return CoreInfo::AnyType; }
  static CBOptionalString outputhelp() {
    return CBCCSTR("The output of this block will be its input.");
  }

  static CBParametersInfo parameters() { return _params; }

  void setParam(int index, const CBVar &value) {
    switch (index) {
    case 0:
      _matrix = value;
      break;
    case 1:
      _mode = value;
      break;
    case 2:
      _operation = value;
      break;
    case 3:
      _snap = value;
      break;

    default:
      throw CBException("Parameter out of range.");
    }
  }

  CBVar getParam(int index) {
    switch (index) {
    case 0:
      return _matrix;
    case 1:
      return _mode;
    case 2:
      return _operation;
    case 3:
      return _snap;

    default:
      throw CBException("Parameter out of range.");
    }
  }

  void cleanup() {
    _matrix.cleanup();
    _mode.cleanup();
    _operation.cleanup();
    _snap.cleanup();

    BGFX::BaseConsumer::cleanup();
  }

  void warmup(CBContext *context) {
    BGFX::BaseConsumer::warmup(context);

    _matrix.warmup(context);
    _mode.warmup(context);
    _operation.warmup(context);
    _snap.warmup(context);
  }

  CBVar activate(CBContext *context, const CBVar &input) {
    auto *ctx =
        reinterpret_cast<BGFX::Context *>(_bgfxCtx->payload.objectValue);
    const auto &currentView = ctx->currentView();

    float arrView[16];
    float arrProj[16];
    Mat4::ToArrayUnsafe(currentView.view, arrView);
    Mat4::ToArrayUnsafe(currentView.proj, arrProj);

    auto &matrix = _matrix.get();
    // HACKish, I use Mat4 because the API hides some complexity, but it feels
    // unnecessary to do that many copies should manipulate the seq of float4
    // directly
    Mat4 mat{};
    mat = matrix;
    float arrMatrix[16];
    Mat4::ToArrayUnsafe(mat, arrMatrix);

    auto &snap = _snap.get();
    auto useSnap = snap.valueType == CBType::Float3;
    auto snapPtr = useSnap
                       ? reinterpret_cast<float *>(&snap.payload.float3Value)
                       : nullptr;

    ImGuizmo::Manipulate(
        arrView, arrProj,
        Enums::OperationToGuizmo(_operation.get().payload.enumValue),
        Enums::ModeToGuizmo(_mode.get().payload.enumValue), arrMatrix, nullptr,
        snapPtr);

    chainblocks::cloneVar(matrix, Mat4::FromArrayUnsafe(arrMatrix));

    return input;
  }

private:
  static inline Parameters _params = {
      {"Matrix",
       CBCCSTR("Object matrix modified by this transform gizmo."),
       {Type::VariableOf(CoreInfo::Float4x4Type)}},
      // TODO add checks: matrix is required and must be non-null
      {"Mode",
       CBCCSTR("Type of coordinates used. Either local to the object or global "
               "(world)."),
       {Enums::ModeType, Type::VariableOf(Enums::ModeType)}},
      {"Operation",
       CBCCSTR("Operation to apply (translation, rotation or scaling)."),
       {Enums::OperationType, Type::VariableOf(Enums::OperationType)}},
      {"Snap",
       CBCCSTR("Snapping values on all 3 axes. World unit for translation. "
               "Angle in degrees for rotation. Multiplier for scaling. Use "
               "`None` or `(float3 0 0 0)` to disable."),
       {CoreInfo::NoneType, CoreInfo::Float3Type, CoreInfo::Float3VarType}}};

  ParamVar _matrix{};
  ParamVar _mode{};
  ParamVar _operation{};
  ParamVar _snap{};
};

void registerGizmoBlocks() {
  REGISTER_CBLOCK("Gizmo.CubeView", CubeView);
  REGISTER_CBLOCK("Gizmo.Grid", Grid);
  REGISTER_CBLOCK("Gizmo.Transform", Transform);
}
}; // namespace Gizmo
}; // namespace chainblocks
