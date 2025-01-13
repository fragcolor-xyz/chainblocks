// The MIT License
//
// Copyright © 2010-2021 three.js authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "geom.hpp"
#include <algorithm>
#include <cmath>

namespace gfx {
namespace geom {

std::vector<MeshVertexAttribute> VertexPNT::getAttributes() {
  std::vector<MeshVertexAttribute> attribs;
  attribs.emplace_back("position", 3, StorageType::Float32);
  attribs.emplace_back("normal", 3, StorageType::Float32);
  attribs.emplace_back("texCoord0", 2, StorageType::Float32);
  return attribs;
}

// https://github.com/mrdoob/three.js/blob/master/src/geometries/SphereGeometry.js#L87
void SphereGenerator::generate() {
  reset();

  size_t widthSegments = std::max<size_t>(3, this->widthSegments);
  size_t heightSegments = std::max<size_t>(2, this->heightSegments);

  auto thetaEnd = std::min(thetaStart + thetaLength, pi);

  auto index = 0;
  std::vector<std::vector<index_t>> grid;

  for (size_t iy = 0; iy <= heightSegments; iy++) {
    std::vector<index_t> verticesRow;

    auto v = float(iy) / heightSegments;

    // special case for the poles

    auto uOffset = 0;

    if (iy == 0 && thetaStart == 0) {
      uOffset = 0.5 / widthSegments;
    } else if (iy == heightSegments && thetaEnd == pi) {
      uOffset = -0.5 / widthSegments;
    }

    for (size_t ix = 0; ix <= widthSegments; ix++) {
      auto u = float(ix) / widthSegments;

      VertexPNT &vertex = vertices.emplace_back();

      float3 position;
      position.x = -radius * std::cos(phiStart + u * phiLength) * std::sin(thetaStart + v * thetaLength);
      position.y = radius * std::cos(thetaStart + v * thetaLength);
      position.z = radius * std::sin(phiStart + u * phiLength) * std::sin(thetaStart + v * thetaLength);
      vertex.setPosition(position);

      // normal
      vertex.setNormal(linalg::normalize(position));

      vertex.setTexCoord(float2(u + uOffset, v));

      verticesRow.push_back(index++);
    }

    grid.push_back(verticesRow);
  }

  for (size_t iy = 0; iy < heightSegments; iy++) {
    for (size_t ix = 0; ix < widthSegments; ix++) {
      auto a = grid[iy][ix + 1];
      auto b = grid[iy][ix];
      auto c = grid[iy + 1][ix];
      auto d = grid[iy + 1][ix + 1];

      if (iy != 0 || thetaStart > 0) {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(d);
      }

      if (iy != heightSegments - 1 || thetaEnd < pi) {
        indices.push_back(b);
        indices.push_back(c);
        indices.push_back(d);
      }
    }
  }
}

// https://github.com/mrdoob/three.js/blob/master/src/geometries/PlaneGeometry.js
void PlaneGenerator::generate() {
  reset();

  auto width_half = width / 2.0f;
  auto height_half = height / 2.0f;

  auto gridX = widthSegments;
  auto gridY = heightSegments;

  auto gridX1 = gridX + 1;
  auto gridY1 = gridY + 1;

  auto segment_width = width / gridX;
  auto segment_height = height / gridY;

  for (size_t iy = 0; iy < gridY1; iy++) {
    auto y = float(iy) * segment_height - height_half;

    for (size_t ix = 0; ix < gridX1; ix++) {
      auto x = float(ix) * segment_width - width_half;

      VertexPNT &vertex = vertices.emplace_back();
      vertex.setPosition(float3(x, -y, 0));
      vertex.setNormal(float3(0, 0, 1));
      vertex.setTexCoord(float2(float(ix) / gridX, float(iy) / gridY));
    }
  }

  for (size_t iy = 0; iy < gridY; iy++) {
    for (size_t ix = 0; ix < gridX; ix++) {
      auto a = ix + gridX1 * iy;
      auto b = ix + gridX1 * (iy + 1);
      auto c = (ix + 1) + gridX1 * (iy + 1);
      auto d = (ix + 1) + gridX1 * iy;

      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(d);
      indices.push_back(b);
      indices.push_back(c);
      indices.push_back(d);
    }
  }
}

// https://github.com/mrdoob/three.js/blob/master/src/geometries/PlaneGeometry.js
void CubeGenerator::generate() {
  reset();

  size_t vertexOffset = 0;
  auto buildPlane = [&](float u, float v, float w, float udir, float vdir, float width, float height, float depth, size_t gridX,
                        size_t gridY) {
    const float segmentWidth = width / gridX;
    const float segmentHeight = height / gridY;

    const float widthHalf = width / 2;
    const float heightHalf = height / 2;
    const float depthHalf = depth / 2;

    const size_t gridX1 = gridX + 1;
    const size_t gridY1 = gridY + 1;

    // generate vertices, normals and uvs
    size_t vertexCounter = 0;
    for (size_t iy = 0; iy < gridY1; iy++) {
      const float y = iy * segmentHeight - heightHalf;

      for (size_t ix = 0; ix < gridX1; ix++) {
        const float x = ix * segmentWidth - widthHalf;

        VertexPNT &vertex = vertices.emplace_back();
        ++vertexCounter;

        float3 pos;
        pos[u] = x * udir;
        pos[v] = y * vdir;
        pos[w] = depthHalf;
        vertex.setPosition(pos);

        float3 normal;
        normal[u] = 0;
        normal[v] = 0;
        normal[w] = depth > 0 ? 1 : -1;
        vertex.setNormal(normal);

        float2 uv;
        uv.x = (ix / gridX);
        uv.y = (iy / gridY);
        vertex.setTexCoord(uv);
      }
    }

    // indices
    for (size_t iy = 0; iy < gridY; iy++) {
      for (size_t ix = 0; ix < gridX; ix++) {
        const size_t a = vertexOffset + ix + gridX1 * iy;
        const size_t b = vertexOffset + ix + gridX1 * (iy + 1);
        const size_t c = vertexOffset + (ix + 1) + gridX1 * (iy + 1);
        const size_t d = vertexOffset + (ix + 1) + gridX1 * iy;

        // faces
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(d);

        indices.push_back(b);
        indices.push_back(c);
        indices.push_back(d);
      }
    }

    vertexOffset += vertexCounter;
  };

  buildPlane(2, 1, 0, -1, -1, depth, height, width, depthSegments, heightSegments);  // px
  buildPlane(2, 1, 0, 1, -1, depth, height, -width, depthSegments, heightSegments);  // nx
  buildPlane(0, 2, 1, 1, 1, width, depth, height, widthSegments, depthSegments);     // py
  buildPlane(0, 2, 1, 1, -1, width, depth, -height, widthSegments, depthSegments);   // ny
  buildPlane(0, 1, 2, 1, -1, width, height, depth, widthSegments, heightSegments);   // pz
  buildPlane(0, 1, 2, -1, -1, width, height, -depth, widthSegments, heightSegments); // nz
}

void CylinderGenerator::generate() {
  reset();

  size_t index = 0;
  size_t groupStart = 0;
  auto halfHeight = height / 2;

  std::vector<std::vector<uint16_t>> indexRows;

  auto generateCap = [&](bool top) {
    // save the index of the first center vertex
    auto centerIndexStart = index;

    auto radius = (top) ? radiusTop : radiusBottom;
    auto sign = (top) ? 1.0f : -1.0f;

    // first we generate the center vertex data of the cap.
    // because the geometry needs one set of uvs per face,
    // we must generate a center vertex per face/segment
    for (size_t x = 1; x <= radialSegments; x++) {
      auto &vertex = vertices.emplace_back();

      float3 pos(0, halfHeight * sign, 0);
      vertex.setPosition(pos);

      float3 normal(0, sign, 0);
      vertex.setNormal(normal);

      vertex.setTexCoord(float2(0.5, 0.5));

      index++;
    }

    // save the index of the last center vertex
    auto centerIndexEnd = index;

    // now we generate the surrounding vertices, normals and uvs
    for (size_t x = 0; x <= radialSegments; x++) {
      auto u = float(x) / radialSegments;
      auto theta = u * thetaLength + thetaStart;

      auto cosTheta = std::cos(theta);
      auto sinTheta = std::sin(theta);

      auto &vertex = vertices.emplace_back();

      float3 pos;
      pos.x = radius * sinTheta;
      pos.y = halfHeight * sign;
      pos.z = radius * cosTheta;
      vertex.setPosition(pos);

      float3 normal(0, sign, 0);
      vertex.setNormal(normal);

      float2 uv;
      uv.x = (cosTheta * 0.5) + 0.5;
      uv.y = (sinTheta * 0.5 * sign) + 0.5;
      vertex.setTexCoord(uv);

      index++;
    }

    // generate indices
    for (size_t x = 0; x < radialSegments; x++) {
      auto c = centerIndexStart + x;
      auto i = centerIndexEnd + x;

      if (top) {
        indices.push_back(i);
        indices.push_back(i + 1);
        indices.push_back(c);
      } else {
        indices.push_back(i + 1);
        indices.push_back(i);
        indices.push_back(c);
      }
    }
  };

  auto generateTorso = [&]() {
    auto groupCount = 0;

    // this will be used to calculate the normal
    auto slope = (radiusBottom - radiusTop) / height;

    // generate vertices, normals and uvs
    for (size_t y = 0; y <= heightSegments; y++) {
      std::vector<uint16_t> indexRow;

      auto v = float(y) / float(heightSegments);
      auto radius = v * (radiusBottom - radiusTop) + radiusTop;

      for (size_t x = 0; x <= radialSegments; x++) {
        auto u = float(x) / float(radialSegments);

        auto theta = u * thetaLength + thetaStart;

        auto sinTheta = std::sin(theta);
        auto cosTheta = std::cos(theta);

        auto &vertex = vertices.emplace_back();

        float3 pos;
        pos.x = radius * sinTheta;
        pos.y = -v * height + halfHeight;
        pos.z = radius * cosTheta;
        vertex.setPosition(pos);

        float3 normal = linalg::normalize(float3(sinTheta, slope, cosTheta));
        vertex.setNormal(normal);

        vertex.setTexCoord(float2(u, v));

        indexRow.emplace_back(index++);
      }

      indexRows.emplace_back(std::move(indexRow));
    }

    for (size_t x = 0; x < radialSegments; x++) {
      for (size_t y = 0; y < heightSegments; y++) {
        auto a = indexRows[y][x];
        auto b = indexRows[y + 1][x];
        auto c = indexRows[y + 1][x + 1];
        auto d = indexRows[y][x + 1];

        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(d);

        indices.push_back(b);
        indices.push_back(c);
        indices.push_back(d);

        groupCount += 6;
      }
    }

    groupStart += groupCount;
  };

  generateTorso();

  if (!openEnded) {
    if (radiusTop > 0)
      generateCap(true);
    if (radiusBottom > 0)
      generateCap(false);
  }
}

void QuadGenerator::generateFullscreenTriangle() {
  float3 size = float3(max - min, 0);
  float2 uvDelta = uvMax - uvMin;

  float3 f0 = float3(min.x, min.y, 0);

  vertices.resize(3);
  auto &v0 = vertices[0];
  v0.setPosition(f0);
  v0.setTexCoord(uvMin);

  auto &v1 = vertices[1];
  v1.setPosition(f0 + float3(size.x, 0, 0) * 2);
  v1.setTexCoord(uvMin + float2(uvDelta.x * 2, 0));

  auto &v2 = vertices[2];
  v2.setPosition(f0 + float3(0, size.y, 0) * 2);
  v2.setTexCoord(uvMin + float2(0, uvDelta.y * 2));

  indices = {0, 1, 2};
}

void QuadGenerator::generateQuad() {
  float2 size = max - min;
  float2 uvDelta = uvMax - uvMin;

  auto gridX = 1;
  auto gridY = 1;

  auto gridX1 = gridX + 1;
  auto gridY1 = gridY + 1;

  auto segmentSize = size / float2(gridX, gridY);

  for (size_t iy = 0; iy < gridY1; iy++) {
    auto y = float(iy) * segmentSize.y + min.y;
    auto uvY = float(iy) * uvDelta.y + uvMin.y;

    for (size_t ix = 0; ix < gridX1; ix++) {
      auto x = float(ix) * segmentSize.x + min.x;
      auto uvX = float(ix) * uvDelta.x + uvMin.x;

      VertexPNT &vertex = vertices.emplace_back();
      vertex.setPosition(float3(x, -y, 0));
      vertex.setNormal(float3(0, 0, 1));
      vertex.setTexCoord(float2(uvX, uvY));
    }
  }

  for (size_t iy = 0; iy < gridY; iy++) {
    for (size_t ix = 0; ix < gridX; ix++) {
      auto a = ix + gridX1 * iy;
      auto b = ix + gridX1 * (iy + 1);
      auto c = (ix + 1) + gridX1 * (iy + 1);
      auto d = (ix + 1) + gridX1 * iy;

      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(d);
      indices.push_back(b);
      indices.push_back(c);
      indices.push_back(d);
    }
  }
}

void QuadGenerator::generate() {
  reset();

  if (optimizeForFullscreen)
    generateFullscreenTriangle();
  else
    generateQuad();
}
} // namespace geom
} // namespace gfx
