#ifndef GFX_TESTS_RENDERER_UTILS
#define GFX_TESTS_RENDERER_UTILS

#include <gfx/feature.hpp>
#include <gfx/features/base_color.hpp>
#include <gfx/features/debug_color.hpp>
#include <gfx/features/transform.hpp>
#include <gfx/geom.hpp>
#include <gfx/mesh.hpp>
#include <gfx/pipeline_step.hpp>
#include <gfx/view.hpp>
#include <gfx/mesh_utils.hpp>

namespace gfx {
struct VertexP {
  float position[3];

  VertexP() = default;
  VertexP(const geom::VertexPNT &other) { setPosition(*(float3 *)other.position); }

  void setPosition(const float3 &v) { memcpy(position, &v.x, sizeof(float) * 3); }

  static inline std::vector<MeshVertexAttribute> getAttributes() {
    std::vector<MeshVertexAttribute> attribs;
    attribs.emplace_back("position", 3, StorageType::Float32);
    return attribs;
  }
};

struct VertexPC {
  float position[3];
  float color[4];

  VertexPC() = default;
  VertexPC(const geom::VertexPNT &other) {
    setPosition(*(float3 *)other.position);
    setColor(float4(1, 1, 1, 1));
  }

  void setPosition(const float3 &v) { memcpy(position, &v.x, sizeof(float) * 3); }
  void setColor(const float4 &v) { memcpy(color, &v.x, sizeof(float) * 4); }

  static inline std::vector<MeshVertexAttribute> getAttributes() {
    std::vector<MeshVertexAttribute> attribs;
    attribs.emplace_back("position", 3, StorageType::Float32);
    attribs.emplace_back("color", 4, StorageType::Float32);
    return attribs;
  }
};

template <typename T> std::vector<T> convertVertices(const std::vector<geom::VertexPNT> &input) {
  std::vector<T> result;
  for (auto &vert : input)
    result.emplace_back(vert);
  return result;
}

inline MeshPtr createSphereMesh() {
  geom::SphereGenerator gen;
  gen.radius = 1.0f;
  gen.generate();
  return createMesh(gen.vertices, gen.indices);
}

inline MeshPtr createPlaneMesh() {
  geom::PlaneGenerator gen{.flipTextureVertically = true};
  gen.generate();
  return createMesh(gen.vertices, gen.indices);
}

inline MeshPtr createCubeMesh() {
  geom::CubeGenerator gen;
  gen.generate();
  return createMesh(gen.vertices, gen.indices);
}

inline ViewPtr createTestProjectionView() {
  ViewPtr view = std::make_shared<View>();
  view->view = linalg::lookat_matrix(float3(0, 10.0f, 10.0f), float3(0, 0, 0), float3(0, 1, 0));
  view->proj = ViewPerspectiveProjection{
      degToRad(45.0f),
      FovDirection::Horizontal,
  };
  return view;
}

inline PipelineSteps createTestPipelineSteps(DrawQueuePtr queue) {
  return PipelineSteps{
      makePipelineStep(RenderDrawablesStep{
          .drawQueue = queue,
          .features =
              {
                  features::Transform::create(),
                  features::BaseColor::create(),
                  features::DebugColor::create("normal", ProgrammableGraphicsStage::Vertex),
              },
      }),
  };
}

} // namespace gfx

#endif // GFX_TESTS_RENDERER_UTILS
