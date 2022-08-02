#ifndef GFX_GEOM
#define GFX_GEOM

#include "linalg.hpp"
#include "math.hpp"
#include "mesh.hpp"
#include <vector>

namespace gfx {
namespace geom {
struct VertexPNT {
  float position[3];
  float normal[3];
  float texCoord[2];

  void setPosition(const float3 &v) { memcpy(position, &v.x, sizeof(float) * 3); }
  void setNormal(const float3 &v) { memcpy(normal, &v.x, sizeof(float) * 3); }
  void setTexCoord(const float2 &v) { memcpy(texCoord, &v.x, sizeof(float) * 2); }

  static std::vector<MeshVertexAttribute> getAttributes();
};

struct GeneratorBase {
  typedef uint16_t index_t;
  std::vector<VertexPNT> vertices;
  std::vector<uint16_t> indices;

  void reset() {
    vertices.clear();
    indices.clear();
  }
};

struct SphereGenerator : public GeneratorBase {
  float radius = 1.0f;

  size_t widthSegments = 32;
  size_t heightSegments = 20;

  float phiStart = 0;
  float phiLength = pi2;

  float thetaStart = 0;
  float thetaLength = pi;

  void generate();
};

struct PlaneGenerator : public GeneratorBase {
  float width = 1;
  float height = 1;
  size_t widthSegments = 1;
  size_t heightSegments = 1;

  void generate();
};

struct CubeGenerator : public GeneratorBase {
  float width = 1;
  float height = 1;
  float depth = 1;

  size_t widthSegments = 1;
  size_t heightSegments = 1;
  size_t depthSegments = 1;

  void generate();
};

struct CylinderGenerator : public GeneratorBase {
  float radiusTop = 1;
  float radiusBottom = 1;
  float height = 1;

  size_t radialSegments = 8;
  size_t heightSegments = 1;

  bool openEnded = false;
  float thetaStart = 0.0f;
  float thetaLength = pi2;

  void generate();
};

} // namespace geom
} // namespace gfx

#endif // GFX_GEOM
