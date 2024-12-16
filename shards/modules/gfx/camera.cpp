#include "camera.hpp"
#include <shards/linalg_shim.hpp>
#include <gfx/math.hpp>
#include <gfx/window.hpp>
#include <gfx/fmt.hpp>
#include <shards/modules/inputs/inputs.hpp>
#include "input/gestures.hpp"
#include "window.hpp"
#include "gfx.hpp"

using namespace shards;
using namespace shards::input;
using namespace shards::input::gestures;

namespace gfx {

// Mouse/Right stick rotation input and mouse X/Y movement (drag)
struct PointerInputState {
  float2 position{};
  float2 prevPosition{};
};

// Input state for a button axis
struct AxisInputState {
  bool pos{};
  bool neg{};

  float getValue() const {
    if (pos)
      return 1.0f;
    if (neg)
      return -1.0f;
    return 0.0f;
  }
};

// Tracked cursor & button states
struct InputState {
  input::GestureRecognizer gestures;
  std::shared_ptr<Pinch> pinchGesture = std::make_shared<Pinch>();
  std::shared_ptr<MultiSlide> slideGesture = std::make_shared<MultiSlide>();
  std::shared_ptr<MultiSlide> rotateGesture = std::make_shared<MultiSlide>();

  InputState() {
    pinchGesture->threshold = 0.03f;
    slideGesture->threshold = 0.02f;
    rotateGesture->threshold = 0.02f;
    gestures.gestures.push_back(pinchGesture);
    gestures.gestures.push_back(slideGesture);
    gestures.gestures.push_back(rotateGesture);
    rotateGesture->numFingers = 1;
  }

  PointerInputState pointer;
  AxisInputState keyboardXAxis;
  AxisInputState keyboardYAxis;
  AxisInputState keyboardZAxis;
  bool lookInteraction{};
  bool panInteraction{};
  float zoom{};
  float deltaTime;
};

// Inputs for camera movement
struct CameraInputs {
  // XYZ camera translation (time scaled)
  float3 velocity;
  // XYZ camera translation (absolute)
  float3 translation;
  // Roll-Pitch-Yaw camera rotation
  float3 lookRotation;
};

// Update tracked buttons and pointers
inline void updateInputState(InputState &inputState, IInputContext &inputContext) {
  inputState.pointer.prevPosition = inputState.pointer.position;
  inputState.pointer.position = inputContext.getState().cursorPosition;
  inputState.zoom = 0;
  bool canReceiveInput = inputContext.canReceiveInput();

  for (auto &ce : inputContext.getEvents()) {
    std::visit(
        [&](auto &&event) {
          using T = std::decay_t<decltype(event)>;
          if constexpr (std::is_same_v<T, KeyEvent>) {
            if ((!canReceiveInput || ce.isConsumed()) && event.pressed)
              return;

            bool passthrough = false;
            switch (event.key) {
            case SDLK_w:
              inputState.keyboardZAxis.pos = event.pressed;
              break;
            case SDLK_s:
              inputState.keyboardZAxis.neg = event.pressed;
              break;
            case SDLK_a:
              inputState.keyboardXAxis.neg = event.pressed;
              break;
            case SDLK_d:
              inputState.keyboardXAxis.pos = event.pressed;
              break;
            case SDLK_e:
              inputState.keyboardYAxis.pos = event.pressed;
              break;
            case SDLK_q:
              inputState.keyboardYAxis.neg = event.pressed;
              break;
            default:
              passthrough = true;
              break;
            }
            if (!passthrough && (inputState.lookInteraction || inputState.panInteraction))
              ce.consume(inputContext.getHandler());
          } else if constexpr (std::is_same_v<T, PointerButtonEvent>) {
            if ((!canReceiveInput || ce.isConsumed()) && event.pressed)
              return;

            inputState.pointer.position = float2(event.pos.x, event.pos.y);
            switch (event.index) {
            case SDL_BUTTON_RIGHT:
              inputState.lookInteraction = event.pressed;
              ce.consume(inputContext.getHandler());
              break;
            case SDL_BUTTON_MIDDLE:
              inputState.panInteraction = event.pressed;
              ce.consume(inputContext.getHandler());
              break;
            }
          } else if constexpr (std::is_same_v<T, ScrollEvent>) {
            if (!canReceiveInput || ce.isConsumed())
              return;
            inputState.zoom += float(event.delta);
          }
        },
        ce.event);
  }

  auto &ctxState = inputContext.getState();

  if (!Pointer::HasPersistentPointer) {
    inputState.lookInteraction = false;

    auto consume = inputContext.getEventConsumer();
    inputState.gestures.update(inputContext.canReceiveInput(), consume, ctxState, inputContext.getEvents());
    if (inputState.gestures.activeGesture == inputState.pinchGesture.get()) {
      inputState.zoom += inputState.pinchGesture->delta / inputContext.getState().region.size.y * 50.0f;
      SPDLOG_INFO("Pinching {}", inputState.zoom);
    }
  } else {
    if (inputState.lookInteraction | inputState.panInteraction) {
      inputContext.requestFocus();
    }
  }

  inputState.deltaTime = inputContext.getDeltaTime();
}

template <typename T> inline T getParamVarOrDefault(ParamVar &paramVar, const T &_default) {
  Var var(paramVar.get());
  return var.isNone() ? _default : (T)var;
}

// Generates CameraInputs from an input state
inline CameraInputs getCameraInputs(const InputState &inputState, ParamVar &lookSpeedParam, ParamVar &panSpeedParam,
                                    ParamVar &scrollSpeedParam) {
  CameraInputs inputs;

  const float mouseBaseFactor = 2000.0f;

  float lookSpeed = getParamVarOrDefault(lookSpeedParam, 1.0f) / mouseBaseFactor * 0.7f * gfx::pi2;
  float panSpeed = getParamVarOrDefault(panSpeedParam, 1.0f) / mouseBaseFactor * 2.0f;
  float scrollSpeed = getParamVarOrDefault(scrollSpeedParam, 1.0f) * 0.4f;

  float2 pointerDelta = inputState.pointer.position - inputState.pointer.prevPosition;

  if (inputState.lookInteraction) {
    // Apply look rotation
    inputs.lookRotation.y = -pointerDelta.x * lookSpeed;
    inputs.lookRotation.x = -pointerDelta.y * lookSpeed;

    // Fly keys
    inputs.velocity.x += inputState.keyboardXAxis.getValue();
    inputs.velocity.y += inputState.keyboardYAxis.getValue();
    inputs.velocity.z += inputState.keyboardZAxis.getValue();
  }

  if (inputState.panInteraction) {
    inputs.translation.x += pointerDelta.x * panSpeed;
    inputs.translation.y += -pointerDelta.y * panSpeed;
  }

  if (inputState.gestures.activeGesture == inputState.slideGesture.get()) {
    float2 slideDelta = inputState.slideGesture->delta;
    inputs.translation.x += slideDelta.x * panSpeed * 2.0f;
    inputs.translation.y += -slideDelta.y * panSpeed * 2.0f;
    SPDLOG_TRACE("Sliding {} ({})", slideDelta, inputState.slideGesture->slideOffset);
  } else if (inputState.gestures.activeGesture == inputState.rotateGesture.get()) {
    float2 rotateDelta = inputState.rotateGesture->delta;
    inputs.lookRotation.y -= rotateDelta.x * lookSpeed * 2.0f;
    inputs.lookRotation.x -= rotateDelta.y * lookSpeed * 2.0f;
    SPDLOG_TRACE("Rotating {} ({})", rotateDelta, inputState.rotateGesture->slideOffset);
  }

  if (inputState.zoom != 0.0f) {
    inputs.translation.z += -inputState.zoom * scrollSpeed;
  }

  return inputs;
}

struct FreeCameraShard {
  static SHTypesInfo inputTypes() { return CoreInfo::Float4x4Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::Float4x4Type; }
  static SHOptionalString help() { return SHCCSTR("Provides editor free camera controls"); }

  PARAM_PARAMVAR(_flySpeed, "FlySpeed", "Controls fly speed with the keyboard",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_scrollSpeed, "ScrollSpeed", "Controls middle mouse movement speed",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_panSpeed, "PanSpeed", "Controls middle mouse pan speed",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_lookSpeed, "LookSpeed", "Controls right mouse look speed",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_flySpeed), PARAM_IMPL_FOR(_scrollSpeed), PARAM_IMPL_FOR(_panSpeed), PARAM_IMPL_FOR(_lookSpeed));

  RequiredInputContext _inputContext;

  Mat4 _result;
  InputState _inputState;

  // Applies camera inputs to a view matrix
  float4x4 applyCameraInputsToView(const CameraInputs &inputs, const float4x4 &inputMatrix, float deltaTime) {
    float flySpeed = getParamVarOrDefault(_flySpeed, 1.0f) * 1.0f;

    // Output either a view matrix or camera matrix (inverse view matrix)
    const bool asViewMatrix = false;
    float4x4 invViewMatrix = asViewMatrix ? linalg::inverse(inputMatrix) : inputMatrix;

    float3 cameraTranslation;
    float3 cameraScale;
    float3x3 cameraRotationMatrix;
    gfx::decomposeTRS(invViewMatrix, cameraTranslation, cameraScale, cameraRotationMatrix);
    float4 cameraRotation = linalg::rotation_quat(cameraRotationMatrix);

    float3 cameraX = linalg::qxdir(cameraRotation);
    float3 cameraY = linalg::qydir(cameraRotation);
    float3 cameraZ = linalg::qzdir(cameraRotation);

    // Apply look rotation
    float4 rotX = linalg::rotation_quat(cameraX, inputs.lookRotation.x);
    float4 rotY = linalg::rotation_quat(cameraY, inputs.lookRotation.y);
    float4 rotZ = linalg::rotation_quat(cameraZ, inputs.lookRotation.z);
    cameraRotation = linalg::qmul(rotZ, cameraRotation);
    cameraRotation = linalg::qmul(rotX, cameraRotation);
    cameraRotation = linalg::qmul(rotY, cameraRotation);

    cameraTranslation +=
        (cameraX * inputs.velocity.x + cameraY * inputs.velocity.y + cameraZ * inputs.velocity.z) * flySpeed * deltaTime;
    cameraTranslation += (cameraX * inputs.translation.x + cameraY * inputs.translation.y + cameraZ * inputs.translation.z);

    // Reconstruct view matrix using the inverse rotation/translation of the camera
    float4x4 outputMatrix = linalg::rotation_matrix(asViewMatrix ? linalg::qconj(cameraRotation) : cameraRotation);
    outputMatrix = linalg::mul(linalg::translation_matrix(asViewMatrix ? -cameraTranslation : cameraTranslation), outputMatrix);

    return outputMatrix;
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    updateInputState(_inputState, _inputContext);

    CameraInputs cameraInputs = getCameraInputs(_inputState, _lookSpeed, _panSpeed, _scrollSpeed);

    float4x4 viewMatrix = (Mat4)input;
    _result = applyCameraInputsToView(cameraInputs, viewMatrix, _inputState.deltaTime);

    return _result;
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _inputContext.cleanup();
  }
  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _inputContext.warmup(context);
  }

  SHTypeInfo compose(SHInstanceData &data) { return outputTypes().elements[0]; }

  SHExposedTypesInfo requiredVariables() {
    static auto e = exposedTypesOf(RequiredInputContext::getExposedTypeInfo());
    return e;
  }
};

struct TargetCameraUpdate {
  static SHTypesInfo inputTypes() { return TargetCameraStateTable::Type; }
  static SHTypesInfo outputTypes() { return TargetCameraStateTable::Type; }
  static SHOptionalString help() { return SHCCSTR("Provides editor free camera controls"); }

  PARAM_PARAMVAR(_flySpeed, "FlySpeed", "Controls fly speed with the keyboard",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_scrollSpeed, "ScrollSpeed", "Controls middle mouse movement speed",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_panSpeed, "PanSpeed", "Controls middle mouse pan speed",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_lookSpeed, "LookSpeed", "Controls right mouse look speed",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_pivotDistance, "PivotDistance", "Controls distance to the point being looked at",
                 {CoreInfo::NoneType, CoreInfo::FloatType, CoreInfo::FloatVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_flySpeed), PARAM_IMPL_FOR(_scrollSpeed), PARAM_IMPL_FOR(_panSpeed), PARAM_IMPL_FOR(_lookSpeed),
             PARAM_IMPL_FOR(_pivotDistance));

  RequiredInputContext _inputContext;

  InputState _inputState;

  TargetCameraStateTable _output;

  // Applies camera inputs to a view matrix
  void updateState(TargetCameraState &state, const CameraInputs &inputs, float deltaTime) {
    float flySpeed = getParamVarOrDefault(_flySpeed, 1.0f) * 0.8f;

    float4 rotPitch = linalg::rotation_quat(float3(1.0f, 0.0, 0.0f), state.rotation.x);
    float4 rotYaw = linalg::rotation_quat(float3(0.0f, 1.0f, 0.0f), state.rotation.y);
    float4 cameraRotation = linalg::qmul(rotYaw, rotPitch);

    float3 cameraX = linalg::qxdir(cameraRotation);
    float3 cameraY = linalg::qydir(cameraRotation);
    float3 cameraZ = -linalg::qzdir(cameraRotation);

    float distanceSpeedScale = (1.0f + (linalg::clamp(state.distance, 1.0f, 100.0f) - 1.0f) * 0.4f);

    state.rotation.x += inputs.lookRotation.x;
    state.rotation.y += inputs.lookRotation.y;

    state.pivot += (inputs.velocity.x * cameraX +  //
                    inputs.velocity.y * cameraY +  //
                    inputs.velocity.z * cameraZ) * //
                   (deltaTime * flySpeed * distanceSpeedScale);

    state.pivot -= (inputs.translation.x * cameraX + //
                    inputs.translation.y * cameraY) *
                   distanceSpeedScale; //

    if ((inputs.translation.z < 0 && state.distance <= MinDistance) ||
        (inputs.translation.z > 0 && state.distance >= MaxDistance)) {
      state.pivot -= inputs.translation.z * cameraZ;
    } else {
      state.distance = linalg::clamp(state.distance + inputs.translation.z, MinDistance, MaxDistance);
    }
  }

  const float MinDistance = 0.2f;
  const float MaxDistance = 100.0f;

  SHVar activate(SHContext *context, const SHVar &input) {
    TargetCameraState state = TargetCameraState::fromTable((TableVar &)input);

    updateInputState(_inputState, _inputContext);
    CameraInputs cameraInputs = getCameraInputs(_inputState, _lookSpeed, _panSpeed, _scrollSpeed);
    updateState(state, cameraInputs, _inputState.deltaTime);

    return (_output = state);
  }

  void cleanup(SHContext *context) {
    PARAM_CLEANUP(context);
    _inputContext.cleanup();
  }
  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _inputContext.warmup(context);
  }

  SHTypeInfo compose(SHInstanceData &data) { return outputTypes().elements[0]; }

  SHExposedTypesInfo requiredVariables() {
    static auto e = exposedTypesOf(decltype(_inputContext)::getExposedTypeInfo());
    return e;
  }
};

struct TargetCameraFromLookAt {
  static SHTypesInfo inputTypes() { return CoreInfo::NoneType; }
  static SHTypesInfo outputTypes() { return TargetCameraStateTable::Type; }
  static SHOptionalString help() { return SHCCSTR("Provides editor free camera controls"); }

  PARAM_PARAMVAR(_target, "Target", "", {CoreInfo::NoneType, CoreInfo::Float3Type, CoreInfo::Float3VarType});
  PARAM_PARAMVAR(_position, "Position", "", {CoreInfo::NoneType, CoreInfo::Float3Type, CoreInfo::Float3VarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_target), PARAM_IMPL_FOR(_position));

  TargetCameraStateTable _result;

  TargetCameraFromLookAt() {
    _target = toVar(float3());
    _position = toVar(float3(2.5f, 2.5f, 5.0f));
  }

  SHVar activate(SHContext *context, const SHVar &input) {
    float3 pos = toFloat3(_position.get());
    float3 target = toFloat3(_target.get());

    auto state = TargetCameraState::fromLookAt(pos, target);
    _result.distance = Var(state.distance);
    _result.pivot = toVar(state.pivot);
    _result.rotation = toVar(state.rotation);
    return _result;
  }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }
};

struct TargetCameraMatrix {
  static SHTypesInfo inputTypes() { return TargetCameraStateTable::Type; }
  static SHTypesInfo outputTypes() { return CoreInfo::Float4x4Type; }
  static SHOptionalString help() { return SHCCSTR("Turns the target camera state into a view matrix"); }

  PARAM_IMPL();

  Mat4 _result;

  SHVar activate(SHContext *context, const SHVar &input) {
    TableVar &inputTable = (TableVar &)input;
    TargetCameraState state{
        .pivot = toFloat3(inputTable.get<Var>("pivot")),
        .distance = (float)(inputTable.get<Var>("distance")),
        .rotation = (float2)toFloat2(inputTable.get<Var>("rotation")),
    };

    float4 rotPitch = linalg::rotation_quat(float3(1.0f, 0.0, 0.0f), state.rotation.x);
    float4 rotYaw = linalg::rotation_quat(float3(0.0f, 1.0f, 0.0f), state.rotation.y);
    float4 cameraRotation = linalg::qmul(rotYaw, rotPitch);

    float3 lookDirection = -linalg::qzdir(cameraRotation);

    // Reconstruct view matrix using the inverse rotation/translation of the camera
    float4x4 newViewMatrix = linalg::identity;
    newViewMatrix = linalg::mul(linalg::translation_matrix(-(state.pivot - lookDirection * state.distance)), newViewMatrix);
    newViewMatrix = linalg::mul(linalg::rotation_matrix(linalg::qconj(cameraRotation)), newViewMatrix);
    return (_result = newViewMatrix);
  }

  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }
  void warmup(SHContext *context) { PARAM_WARMUP(context); }

  PARAM_REQUIRED_VARIABLES()
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }
};

void registerCameraShards() {
  REGISTER_SHARD("FreeCamera", FreeCameraShard);
  REGISTER_SHARD("TargetCamera", TargetCameraUpdate);
  REGISTER_SHARD("TargetCamera.FromLookAt", TargetCameraFromLookAt);
  REGISTER_SHARD("TargetCamera.Matrix", TargetCameraMatrix);
}

} // namespace gfx
