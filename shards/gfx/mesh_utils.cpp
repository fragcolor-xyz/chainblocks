#include "mesh_utils.hpp"
#include "linalg.hpp"
#include "math.hpp"
#include <spdlog/spdlog.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#include "MikkTSpace/mikktspace.c"
#pragma clang diagnostic pop

using namespace boost::container;

namespace gfx {

// Generate surface orientation information in the form of a quaternion
// in case the mesh already contains at least a normal & tangent, these will be converted into quaternion form
// otherwise, the surface orientation quaternion will be generated from the normal and the texture coordinates (if available)
MeshPtr generateLocalBasisAttribute(const MeshDescCPUCopy& mesh) {
  auto &srcFormat = mesh.format;

  small_vector<size_t, 16> attributesToCopy;

  std::optional<size_t> positionIndex;
  std::optional<size_t> normalIndex;
  std::optional<size_t> texCoordIndex;

  small_vector<AttribBuffer, 16> srcAttributes;
  AttribBuffer indexBuffer;

  auto &indexData = mesh.indexData;
  auto &vertexData = mesh.vertexData;
  size_t numIndices = mesh.getNumIndices();

  if (srcFormat.primitiveType != PrimitiveType::TriangleList) {
    throw std::runtime_error("Only triangle lists are supported");
  }

  if (numIndices > 0) {
    StorageType indexType = indexFormatToStorageType(srcFormat.indexFormat);
    indexBuffer.initFrom(indexType, 1, numIndices, getStorageTypeSize(indexType), indexData.data());
  } else {
    indexBuffer = AttribBuffer::generateIndexBuffer(numIndices / 3);
  }

  auto filterAttrib = [&](const MeshVertexAttribute &attrib, size_t index) {
    if (attrib.name == "position") {
      positionIndex = index;
    } else if (attrib.name == "normal") {
      normalIndex = index;
    } else if (attrib.name == "texCoord0") {
      texCoordIndex = index;
    } else if (attrib.name == "tangent" || attrib.name == "bitangent") {
      return false;
    }
    return true;
  };

  size_t srcStride = srcFormat.getVertexSize();
  size_t offset{};
  for (size_t i = 0; i < srcFormat.vertexAttributes.size(); i++) {
    auto &attrib = srcFormat.vertexAttributes[i];

    auto &buffer = srcAttributes.emplace_back();
    buffer.initFromIndexedTriangleList(attrib.type, attrib.numComponents, storageTypeToIndexFormat(indexBuffer.type),
                                       indexBuffer.data(), indexBuffer.length, srcStride, vertexData.data() + offset);

    offset += attrib.numComponents * getStorageTypeSize(attrib.type);
    if (filterAttrib(attrib, i)) {
      attributesToCopy.push_back(i);
    }
  }

  // Validate required attributes
  {
    if (!positionIndex.has_value()) {
      throw std::runtime_error("Mesh does not have a position attribute");
    }
    auto positionAttrib = srcFormat.vertexAttributes[positionIndex.value()];
    if (positionAttrib.type != StorageType::Float32 && positionAttrib.numComponents != 3) {
      throw std::runtime_error("position attribute must be float3");
    }

    if (!normalIndex) {
      throw std::runtime_error("Mesh does not have a normal attribute");
    }
    auto normalAttrib = srcFormat.vertexAttributes[normalIndex.value()];
    if (normalAttrib.type != StorageType::Float32 && normalAttrib.numComponents != 3) {
      throw std::runtime_error("normal attribute must be float3");
    }

    if (!texCoordIndex) {
      throw std::runtime_error("Mesh does not have a texCoord0 attribute");
    }
    auto texCoordAttrib = srcFormat.vertexAttributes[texCoordIndex.value()];
    if (texCoordAttrib.type != StorageType::Float32 && texCoordAttrib.numComponents != 2) {
      throw std::runtime_error("texCoord0 attribute must be float2");
    }
  }

  if (indexData.size() == 0) {
    throw std::runtime_error("Mesh does not have an index buffer");
  }

  AttribBuffer tangentBuffer;
  tangentBuffer.initZero(StorageType::Float32, 4, numIndices);

  struct Context {
    SMikkTSpaceContext ctx;
    const AttribBuffer &indexBuffer;
    const AttribBuffer &positionBuffer;
    const AttribBuffer &normalBuffer;
    const AttribBuffer &texCoordBuffer;
    AttribBuffer &tangentBuffer;
  } context{
      .indexBuffer = indexBuffer,
      .positionBuffer = srcAttributes[positionIndex.value()],
      .normalBuffer = srcAttributes[normalIndex.value()],
      .texCoordBuffer = srcAttributes[texCoordIndex.value()],
      .tangentBuffer = tangentBuffer,
  };

  SMikkTSpaceInterface iface{
      // Returns the number of faces (triangles/quads) on the mesh to be processed.
      .m_getNumFaces = [](const SMikkTSpaceContext *ctx) -> int { return ((Context *)ctx)->indexBuffer.length / 3; },
      .m_getNumVerticesOfFace = [](const SMikkTSpaceContext *ctx, const int iFace) -> int { return 3; },
  };
  iface.m_getPosition = [](const SMikkTSpaceContext *ctx, float fvPosOut[], const int iFace, const int iVert) {
    auto vec = ((Context *)ctx)->positionBuffer.read<float, 3, false>(iFace * 3 + iVert);
    fvPosOut[0] = float(vec.x);
    fvPosOut[1] = float(vec.y);
    fvPosOut[2] = float(vec.z);
  };
  iface.m_getNormal = [](const SMikkTSpaceContext *ctx, float fvPosOut[], const int iFace, const int iVert) {
    auto vec = ((Context *)ctx)->normalBuffer.read<float, 3, false>(iFace * 3 + iVert);
    fvPosOut[0] = float(vec.x);
    fvPosOut[1] = float(vec.y);
    fvPosOut[2] = float(vec.z);
  };
  iface.m_getTexCoord = [](const SMikkTSpaceContext *ctx, float fvPosOut[], const int iFace, const int iVert) {
    auto vec = ((Context *)ctx)->texCoordBuffer.read<float, 2, false>(iFace * 3 + iVert);
    // NOTE: Texcoord is flipped to match glTF space
    //   reference: https://github.com/mrdoob/three.js/issues/25263
    fvPosOut[0] = fmod(float(vec.x), 1.0f);
    fvPosOut[1] = 1.0f - fmod(float(vec.y), 1.0f);
  };

  iface.m_setTSpaceBasic = [](const SMikkTSpaceContext *ctx, const float fvTangent[], const float fSign, const int iFace,
                              const int iVert) {
    float3 x = float3(fvTangent);
    auto &tangentBuffer = ((Context *)ctx)->tangentBuffer;
    tangentBuffer.write(iFace * 3 + iVert, float4(x, fSign));
  };

  context.ctx.m_pInterface = &iface;
  genTangSpaceDefault(&context.ctx);

  small_vector<std::tuple<AttribBuffer *, FastString>, 16> outAttribs;
  for (size_t i : attributesToCopy) {
    outAttribs.push_back({&srcAttributes[i], srcFormat.vertexAttributes[i].name});
  }
  outAttribs.push_back({&tangentBuffer, "tangent"});
  return generateMesh(std::nullopt, outAttribs);
}
} // namespace gfx