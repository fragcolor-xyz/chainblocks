#ifndef FA3CD16D_2B62_442D_AEC9_50A3AE9EFD8C
#define FA3CD16D_2B62_442D_AEC9_50A3AE9EFD8C

#include "../fwd.hpp"
#include "../linalg.hpp"
#include "../math.hpp"
#include "../view.hpp"
#include "gizmo_math.hpp"
#include <memory>
#include <optional>

namespace gfx {
namespace gizmos {

struct InputContext;
struct Handle;

// Callback handler for gizmo handles
struct IGizmoCallbacks {
  virtual ~IGizmoCallbacks() = default;

  // Called when handle grabbed
  virtual void grabbed(InputContext &context, Handle &handle) = 0;
  // Called when handle released
  virtual void released(InputContext &context, Handle &handle) = 0;
  // Called every update while handle is being held
  virtual void move(InputContext &context, Handle &handle) = 0;
};

struct Handle {
  IGizmoCallbacks *callbacks{};
  void *userData{};
};

struct InputState {
  float2 cursorPosition{};
  // Size of the input surface
  float2 inputSize{};
  // Size of the output surface in pixels
  float2 viewportSize{};
  // When the button was pressed this frame, implies held
  bool pressed{};
  // When the button is being held
  bool held{};
};

// Information about handle that was hit by cursor ray
struct GizmoHit {
  float distance;
  int layer;

  GizmoHit() = default;
  // Either distance or layer less than will be a more important hit
  GizmoHit(float distance, int layer = 0) : distance(distance), layer(layer) {}

  // Represents no hit, everything will be less
  static GizmoHit none() {
    return GizmoHit{
        std::numeric_limits<float>::max(),
        std::numeric_limits<int>::max(),
    };
  }

  auto compareTie() const { return std::tie(layer, distance); }
  bool operator<(const GizmoHit &other) const { return compareTie() < other.compareTie(); }
};

// Input tracking for gizmo handles
// use withing a loop where you call
// - begin
// - updateHandle (for each handle in the view)
// - end (this will run callbacks)
// user is responsible for keeping handle data alive between begin/end
struct InputContext {
public:
  // Currently held handle
  // it will keep being held until the interation button is released
  Handle *held{};
  bool heldHandleUpdated = false;

  float3 eyeLocation;
  float3 rayDirection;

  // Currently hovered handle
  // will be locked to the held handle while the interaction button is pressed
  Handle *hovering{};

  GizmoHit hit;
  float3 hitLocation;

  // Last input state
  InputState inputState;
  InputState prevInputState;

private:
  float4x4 cachedViewProj;
  float4x4 cachedViewProjInv;
  float4x4 cachedViewInv;

public:
  void begin(const InputState &inputState, ViewPtr view);
  // Call this within the update for each handle in view
  //  when a handle is grabbed, this is allowed to run the move callback directly
  //  since no raycast needs to be performed on the entire set of handles
  void updateHandle(Handle &handle, GizmoHit hit);
  // Call to end input update and run input callbacks
  void end();

  // Returns normalized world direction vectors of the 3 axes relative to the screen
  float3x3 getScreenSpacePlaneAxes() const;
  // Returns the normalized up vector of the camera in world space
  float3 getUpVector() const;
  // Returns the normalized forward vector of the camera in world space
  float3 getForwardVector() const;

private:
  // Computes eye location and cursor ray direction
  void updateView(ViewPtr view);
  // Updates movement for the currently held handle
  void updateHeldHandle();
  // Updates `hitLocation` variable based on hovered/held handle
  void updateHitLocation();

  // Set to private to prevent using the cached inverse view projection matrix for the 3 axes
  // Forces the user to use the `getScreenSpacePlaneAxes` function
  float3 getRightVector() const;
};
} // namespace gizmos

} // namespace gfx

#endif /* FA3CD16D_2B62_442D_AEC9_50A3AE9EFD8C */
