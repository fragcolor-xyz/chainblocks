#include "gltf2.hpp"
#include "gfx/drawables/mesh_tree_drawable.hpp"
#include "gfx/linalg.hpp"
#include "linalg.h"
#include "spdlog/spdlog.h"
#include "../filesystem.hpp"
#include <algorithm>
#include <cmath>
#include <gfx/data_cache/data_cache.hpp>
#include <gfx/error_utils.hpp>
#include <gfx/filesystem.hpp>
#include <gfx/material.hpp>
#include <gfx/mesh.hpp>
#include <gfx/texture.hpp>
#include <gfx/texture_file/texture_file.hpp>
#include <gfx/drawables/mesh_drawable.hpp>
#include <gfx/features/alpha_cutoff.hpp>
#include <shards/defer.hpp>
#include <shards/core/serialization/generic.hpp>
#include <boost/container/flat_map.hpp>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <filesystem>
#include <regex>
#include <fstream>
#include <iostream>
#include "impl.hpp"

#include <stb_image.h>

namespace gfx {

static WGPUAddressMode convertAddressMode(int mode) {
  if (mode < 0)
    return WGPUAddressMode_Repeat;

  switch (mode) {
  case TINYGLTF_TEXTURE_WRAP_REPEAT:
    return WGPUAddressMode_Repeat;
  case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
    return WGPUAddressMode_ClampToEdge;
  case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
    return WGPUAddressMode_MirrorRepeat;
  }
  shassert(false);
  return WGPUAddressMode(~0);
}

static WGPUFilterMode convertFilterMode(int mode) {
  if (mode < 0)
    return WGPUFilterMode_Linear;

  switch (mode) {
  case TINYGLTF_TEXTURE_FILTER_NEAREST:
    return WGPUFilterMode_Nearest;
  case TINYGLTF_TEXTURE_FILTER_LINEAR:
    return WGPUFilterMode_Linear;
    // NOTE: Mipmap filtering mode is ignored
  case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
    return WGPUFilterMode_Nearest;
  case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
    return WGPUFilterMode_Linear;
  case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
    return WGPUFilterMode_Nearest;
  case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
    return WGPUFilterMode_Linear;
  }
  shassert(false);
  return WGPUFilterMode(~0);
}

static SamplerState convertSampler(const tinygltf::Sampler &sampler) {
  SamplerState samplerState{};
  samplerState.addressModeU = convertAddressMode(sampler.wrapS);
  samplerState.addressModeV = convertAddressMode(sampler.wrapT);
  samplerState.filterMode = convertFilterMode(sampler.minFilter);
  return samplerState;
}

static float4 convertFloat4Vec(const std::vector<double> &v) {
  shassert(v.size() == 4);
  return float4(v[0], v[1], v[2], v[3]);
}

static TRS convertNodeTransform(const tinygltf::Node &node) {
  if (node.matrix.size() != 0) {
    return TRS(float4x4(double4x4(node.matrix.data())));
  } else {
    TRS result;
    result.translation = float3(node.translation.size() != 0 ? double3(node.translation.data()) : double3());
    result.rotation = float4(node.rotation.size() != 0 ? double4(node.rotation.data()) : double4(0, 0, 0, 1));
    result.scale = float3(node.scale.size() != 0 ? double3(node.scale.data()) : double3(1, 1, 1));
    return result;
  }
}

struct Loader2 {
  struct Mesh {
    std::vector<MeshDrawable::Ptr> primitives;
  };

  struct TextureKey {
    int imageIndex;
    int samplerIndex;
    std::strong_ordering operator<=>(const TextureKey &rhs) const = default;
  };
  boost::container::flat_map<TextureKey, TexturePtr> textureLookup;

  std::shared_ptr<data::DataCache> dataCache = data::getInstance();

  fs::Path rootPath;

  tinygltf::Model &model;
  std::vector<MeshPtr> primitiveMap;
  std::vector<Mesh> meshMap;
  std::vector<MaterialPtr> materialMap;
  std::vector<TexturePtr> textureMap;
  std::vector<MeshTreeDrawable::Ptr> nodeMap;
  std::vector<MeshTreeDrawable::Ptr> sceneMap;
  std::unordered_map<std::string, gfx::Animation> animations;
  std::unordered_map<std::string, MaterialPtr> materials;
  std::vector<std::shared_ptr<gfx::Skin>> skins;

  Loader2(tinygltf::Model &model, fs::Path rootPath) : rootPath(rootPath), model(model) {}

  std::string translateAttributeName(const std::string &in) {
    std::string nameLower;
    nameLower.resize(in.size());
    std::transform(in.begin(), in.end(), nameLower.begin(), [](unsigned char c) { return std::tolower(c); });

    const char texcoordPrefix[] = "texcoord_";
    const char colorPrefix[] = "color_";
    const char jointsPrefix[] = "joints_";
    const char weightsPrefix[] = "weights_";
    if (nameLower.starts_with(texcoordPrefix)) {
      std::string numberSuffix = nameLower.substr(sizeof(texcoordPrefix) - 1);
      return std::string("texCoord") + (numberSuffix.empty() ? "0" : numberSuffix.c_str());
    } else if (nameLower.starts_with(colorPrefix)) {
      return std::string("color") + nameLower.substr(sizeof(colorPrefix) - 1);
    } else if (nameLower.starts_with(jointsPrefix)) {
      return "joints";
    } else if (nameLower.starts_with(weightsPrefix)) {
      return "weights";
    } else {
      return nameLower;
    }
  }

  void reorderVertexAttributes(std::vector<const tinygltf::Accessor *> &accessors, std::vector<size_t> &offsets,
                               MeshFormat &format) {
    int positionIndex = -1;
    for (size_t i = 0; i < format.vertexAttributes.size(); i++) {
      if (format.vertexAttributes[i].name == "position") {
        positionIndex = i;
        break;
      }
    }

    // Make position the first attribute to aid renderdoc debugging
    if (positionIndex > 0) {
      std::swap(accessors[0], accessors[positionIndex]);
      std::swap(format.vertexAttributes[0], format.vertexAttributes[positionIndex]);

      // Recompute offsets
      size_t offset = 0;
      for (size_t i = 0; i < format.vertexAttributes.size(); i++) {
        auto &attrib = format.vertexAttributes[i];
        offsets[i] = offset;
        offset += getStorageTypeSize(attrib.type) * attrib.numComponents;
      }
    }
  }

  MeshDrawable::Ptr loadPrimitive(tinygltf::Primitive &primitive) {
    std::vector<const tinygltf::Accessor *> accessors;
    std::vector<size_t> offsets;
    size_t vertexStride{};
    size_t numVertices{};
    MeshFormat format;

    for (auto &gltfAttrib : primitive.attributes) {
      auto &gltfAccessor = model.accessors[gltfAttrib.second];

      MeshVertexAttribute &attrib = format.vertexAttributes.emplace_back();
      attrib.name = translateAttributeName(gltfAttrib.first);
      attrib.numComponents = tinygltf::GetNumComponentsInType(gltfAccessor.type);
      switch (gltfAccessor.componentType) {
      case TINYGLTF_COMPONENT_TYPE_BYTE:
        attrib.type = StorageType::Int8;
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        attrib.type = StorageType::UInt8;
        break;
      case TINYGLTF_COMPONENT_TYPE_SHORT:
        attrib.type = StorageType::Int16;
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        attrib.type = StorageType::UInt16;
        break;
      case TINYGLTF_COMPONENT_TYPE_INT:
        attrib.type = StorageType::Int32;
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        attrib.type = StorageType::UInt32;
        break;
      case TINYGLTF_COMPONENT_TYPE_FLOAT:
        attrib.type = StorageType::Float32;
        break;
      case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        throw formatException("Accessor {}: double component type is not supported", gltfAttrib.second);
        break;
      }
      accessors.push_back(&gltfAccessor);
      offsets.push_back(vertexStride);
      vertexStride += getStorageTypeSize(attrib.type) * attrib.numComponents;

      if (numVertices == 0) {
        numVertices = gltfAccessor.count;
      } else {
        shassert(gltfAccessor.count == numVertices);
      }
    }

    reorderVertexAttributes(accessors, offsets, format);

    std::vector<uint8_t> vertexBuffer;
    vertexBuffer.resize(vertexStride * numVertices);
    for (size_t attribIndex{}; attribIndex < accessors.size(); attribIndex++) {
      size_t dstOffset = offsets[attribIndex];
      const tinygltf::Accessor &accessor = *accessors[attribIndex];
      const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
      const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
      const MeshVertexAttribute &attrib = format.vertexAttributes[attribIndex];

      size_t elementSize = getStorageTypeSize(attrib.type) * attrib.numComponents;
      int srcStride = accessor.ByteStride(bufferView);

      const uint8_t *srcPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
      uint8_t *dstPtr = vertexBuffer.data() + dstOffset;
      for (size_t i = 0; i < numVertices; i++) {
        memcpy(dstPtr, srcPtr, elementSize);
        srcPtr += srcStride;
        dstPtr += vertexStride;
      }
    }

    std::vector<uint8_t> indexBuffer;
    bool requireConversion{};
    if (primitive.indices >= 0) {
      const tinygltf::Accessor &accessor = model.accessors[primitive.indices];
      const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
      const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
      switch (accessor.componentType) {
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        format.indexFormat = IndexFormat::UInt16;
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        format.indexFormat = IndexFormat::UInt32;
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        format.indexFormat = IndexFormat::UInt16;
        requireConversion = true;
        break;
      default:
        throw formatException("Accessor {}: Unsupported index format", primitive.indices);
      }

      size_t dstElementSize = getIndexFormatSize(format.indexFormat);
      int srcStride = accessor.ByteStride(bufferView);

      indexBuffer.resize(dstElementSize * accessor.count);
      uint8_t *dstPtr = indexBuffer.data();
      const uint8_t *srcPtr = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
      if (requireConversion) {
        for (size_t i = 0; i < accessor.count; i++) {
          switch (accessor.componentType) {
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            *(uint16_t *)dstPtr = uint16_t(*(uint8_t *)srcPtr);
            break;
          }
          dstPtr += dstElementSize;
          srcPtr += srcStride;
        }
      } else {
        for (size_t i = 0; i < accessor.count; i++) {
          memcpy(dstPtr, srcPtr, dstElementSize);
          dstPtr += dstElementSize;
          srcPtr += srcStride;
        }
      }
    }

    switch (primitive.mode) {
    case TINYGLTF_MODE_TRIANGLES:
      format.primitiveType = PrimitiveType::TriangleList;
      break;
    case TINYGLTF_MODE_TRIANGLE_STRIP:
      format.primitiveType = PrimitiveType::TriangleStrip;
      break;
    default:
      SPDLOG_WARN("Unsupported primitive mode ({})", primitive.mode);
      return nullptr;
    }

    MeshPtr mesh = std::make_shared<gfx::Mesh>();
    mesh->update(format, std::move(vertexBuffer), std::move(indexBuffer));

    MeshDrawable::Ptr result = std::make_shared<MeshDrawable>(mesh);
    if (primitive.material >= 0) {
      result->material = materialMap[primitive.material];
    }

    return result;
  }

  void loadMeshes() {
    size_t numMeshes = model.meshes.size();
    meshMap.resize(numMeshes);

    for (size_t i = 0; i < numMeshes; i++) {
      auto &mesh = meshMap[i];

      auto &gltfMesh = model.meshes[i];
      for (auto &gltfPrim : gltfMesh.primitives) {
        MeshDrawable::Ptr primitive = loadPrimitive(gltfPrim);
        if (!primitive)
          continue;
        mesh.primitives.push_back(primitive);
      }
    }
  }

  void initNode(size_t nodeIndex) {
    MeshTreeDrawable::Ptr &node = nodeMap[nodeIndex];
    node = std::make_shared<MeshTreeDrawable>();

    const Node &gltfNode = model.nodes[nodeIndex];
    node->name = gltfNode.name;

    if (gltfNode.mesh >= 0) {
      for (auto &prim : meshMap[gltfNode.mesh].primitives) {
        if (!prim)
          continue;
        node->drawables.push_back(prim);
      }
    }

    node->trs = convertNodeTransform(gltfNode);

    for (size_t i = 0; i < gltfNode.children.size(); i++) {
      int childNodeIndex = gltfNode.children[i];
      MeshTreeDrawable::Ptr &childNode = nodeMap[childNodeIndex];
      if (!childNode) {
        childNode = std::make_shared<MeshTreeDrawable>();
        initNode(childNodeIndex);
      }
      node->addChild(childNode);
    }
  }

  void allocateNodes() {
    nodeMap.resize(model.nodes.size());
    for (auto &node : nodeMap) {
      node = nullptr;
    }
  }

  void loadNodes() {
    shassert(nodeMap.size() == model.nodes.size());
    for (size_t i = 0; i < nodeMap.size(); i++) {
      if (!nodeMap[i])
        initNode(i);
    }
  }

  void insertImageIntoCache(data::AssetKey assetKey, const fs::Path &filePath) {
    auto metaAssetKey = assetKey;
    metaAssetKey.categoryFlags = data::AssetCategoryFlags::MetaData;

    std::string filePathStr = filePath.string();
    FILE *file = fopen(filePathStr.c_str(), "rb");
    if (!file) {
      throw std::runtime_error("Failed to open image file: " + filePathStr);
    }
    DEFER({ fclose(file); });

    int2 imageSize;
    int numChannels{};
    StorageType storageType;
    size_t desired{};
    stbi_info_from_file(file, &imageSize.x, &imageSize.y, &numChannels);
    if (stbi_is_16_bit_from_file(file)) {
      storageType = StorageType::UInt16;
    } else {
      storageType = StorageType::UInt8;
    }

    if (numChannels == 3) {
      // Convert image to RGBA
    }

    TextureFormat format;
    // format.pixelFormat = TextureFormat::deriveFrom(numChannels, StorageType storageType, bool preferSrgb)

        // dataCache->store(metaAssetKey);

    if (storageType == StorageType::UInt16) {
      stbi_load_from_file_16(file, &imageSize.x, &imageSize.y, &numChannels, 0);
    } else {
    else {
      // stbi_load_from_file(file, &imageSize.x, &imageSize.y, &numChannels, 0);
    }
  }

  void loadTextures() {
    size_t numTextures = model.textures.size();
    textureMap.resize(numTextures);
    for (size_t i = 0; i < numTextures; i++) {
      const tinygltf::Texture &gltfTexture = model.textures[i];
      const tinygltf::Image &gltfImage = model.images[gltfTexture.source];

      TextureKey cacheKey{.imageIndex = gltfTexture.source, .samplerIndex = gltfTexture.sampler};
      auto itCached = textureLookup.find(cacheKey);
      if (itCached != textureLookup.end()) {
        textureMap[i] = itCached->second;
        continue;
      } else {
        if (!gltfImage.uri.empty()) {
          // gltfImage.
          fs::Path relativePath = (rootPath / gltfImage.uri).lexically_normal();
          auto assetKey = dataCache->generateSourceKey(relativePath.string(), data::AssetCategory::Image);

          if (!dataCache->hasAsset(assetKey)) {
            insertImageIntoCache(assetKey, relativePath);
          }
        } else {
          throw std::runtime_error("Embedded texture not supported");
        }

        TexturePtr &texture = textureMap[i];
        texture = std::make_shared<Texture>();

        SamplerState samplerState{};
        if (gltfTexture.sampler >= 0) {
          samplerState = convertSampler(model.samplers[gltfTexture.sampler]);
        }

        int2 resolution(gltfImage.width, gltfImage.height);
        // texture
        //     ->init(TextureDesc{
        //         .format = convertTextureFormat(gltfImage, false),
        //         .resolution = resolution,
        //         .source =
        //             TextureSource{
        //                 .data = ImmutableSharedBuffer(gltfImage.image.data(), gltfImage.image.size()),
        //             },
        //     })
        //     .initWithSamplerState(samplerState);

        textureLookup[cacheKey] = texture;
      }
    }
  }

  void loadMaterials() {
    size_t numMaterials = model.materials.size();
    materialMap.resize(numMaterials);
    for (size_t i = 0; i < model.materials.size(); i++) {
      MaterialPtr &material = materialMap[i];
      material = std::make_shared<Material>();

      const tinygltf::Material &gltfMaterial = model.materials[i];
      materials[gltfMaterial.name] = material;

      auto convertTextureParam = [&](const char *name, const auto &textureInfo, bool asSrgb) {
        if (textureInfo.index >= 0) {
          const tinygltf::Texture &gltfTexture = model.textures[textureInfo.index];
          const tinygltf::Image &gltfImage = model.images[gltfTexture.source];
          TexturePtr texture = textureMap[textureInfo.index];
          material->parameters.set(name, TextureParameter(texture, textureInfo.texCoord));

          // Update texture format to apply srgb/gamma hint from usage
          // WGPUTextureFormat targetFormat = convertTextureFormat(gltfImage, asSrgb).pixelFormat;
          // texture->initWithPixelFormat(targetFormat);
        }
      };

      auto convertOptionalFloat4Param = [&](const char *name, const std::vector<double> &param, const float4 &defaultValue) {
        float4 f4val = convertFloat4Vec(param);
        if (f4val != defaultValue) {
          material->parameters.set(name, f4val);
        }
      };
      auto convertOptionalFloatParam = [&](const char *name, const float &param, const float &defaultValue) {
        if (param != defaultValue) {
          material->parameters.set(name, param);
        }
      };

      if (gltfMaterial.alphaMode == "MASK") {
        material->features.push_back(features::AlphaCutoff::create());
        material->parameters.set("alphaCutoff", float(gltfMaterial.alphaCutoff));
      } else if (gltfMaterial.alphaMode == "BLEND") {
        auto &feature = material->features.emplace_back(std::make_shared<Feature>());
        feature->state.set_blend(BlendState{
            .color = BlendComponent::Alpha,
            .alpha = BlendComponent::Opaque,
        });
      }

      convertOptionalFloat4Param("baseColor", gltfMaterial.pbrMetallicRoughness.baseColorFactor, float4(1, 1, 1, 1));
      convertOptionalFloatParam("roughness", gltfMaterial.pbrMetallicRoughness.roughnessFactor, 1.0f);
      convertOptionalFloatParam("metallic", gltfMaterial.pbrMetallicRoughness.metallicFactor, 0.0f);
      convertOptionalFloatParam("normalScale", gltfMaterial.normalTexture.scale, 1.0f);
      convertTextureParam("baseColorTexture", gltfMaterial.pbrMetallicRoughness.baseColorTexture, true);
      convertTextureParam("emissiveTexture", gltfMaterial.emissiveTexture, true);
      convertTextureParam("metallicRoughnessTexture", gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture, false);
      convertTextureParam("normalTexture", gltfMaterial.normalTexture, false);
    }
  }

  animation::Track loadAnimationTrack(const tinygltf::Animation &gltfAnimation, const tinygltf::AnimationChannel &gltfChannel) {
    auto &gltfAnimSampler = gltfAnimation.samplers[gltfChannel.sampler];
    auto &input = model.accessors[gltfAnimSampler.input];
    auto &output = model.accessors[gltfAnimSampler.output];

    animation::Track result;
    if (gltfAnimSampler.interpolation == "LINEAR") {
      result.interpolation = animation::Interpolation::Linear;
    } else if (gltfAnimSampler.interpolation == "STEP") {
      result.interpolation = animation::Interpolation::Step;
    } else if (gltfAnimSampler.interpolation == "CUBICSPLINE") {
      result.interpolation = animation::Interpolation::Cubic;
    } else {
      throw formatException("Interpolation mode not supported: {}", gltfAnimSampler.interpolation);
    }

    const tinygltf::BufferView &inputBufferView = model.bufferViews[input.bufferView];
    const tinygltf::BufferView &outputBufferView = model.bufferViews[output.bufferView];
    const tinygltf::Buffer &inputBuffer = model.buffers[inputBufferView.buffer];
    const tinygltf::Buffer &outputBuffer = model.buffers[outputBufferView.buffer];
    const uint8_t *inputData = inputBuffer.data.data() + inputBufferView.byteOffset + input.byteOffset;
    const uint8_t *outputData = outputBuffer.data.data() + outputBufferView.byteOffset + output.byteOffset;

    size_t numFrames = input.count;
    size_t numComponents = GetNumComponentsInType(output.type);
    result.elementSize = numComponents;
    for (size_t i = 0; i < numFrames; i++) {
      const float &time = *(float *)(inputData + input.ByteStride(inputBufferView) * i);
      result.times.emplace_back(time);
    }

    // Convert animation values to float
    size_t numValues = numFrames;
    if (result.interpolation == animation::Interpolation::Cubic)
      numValues = numFrames * 3;
    shassert(numValues <= output.count);
    if (output.count > numValues)
      SPDLOG_WARN("Animation {}: output count {} is larger than input count {}", gltfAnimation.name, output.count, numValues);
    result.data.resize(numComponents * numValues);

    for (size_t i = 0; i < numValues; i++) {
      float *outData = result.data.data() + numComponents * i;
      const uint8_t *inputData = outputData + output.ByteStride(outputBufferView) * i;
      switch (output.componentType) {
      case TINYGLTF_COMPONENT_TYPE_FLOAT:
        for (size_t c = 0; c < numComponents; c++)
          outData[c] = ((float *)inputData)[c];
        break;
      case TINYGLTF_COMPONENT_TYPE_BYTE:
        for (size_t c = 0; c < numComponents; c++)
          outData[c] = std::max((float)((int8_t *)inputData)[c] / float(0x7f), -1.0f);
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        for (size_t c = 0; c < numComponents; c++)
          outData[c] = (float)((uint8_t *)inputData)[c] / float(0xff);
        break;
      case TINYGLTF_COMPONENT_TYPE_SHORT:
        for (size_t c = 0; c < numComponents; c++)
          outData[c] = std::max((float)((int16_t *)inputData)[c] / float(0x7fff), -1.0f);
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        for (size_t c = 0; c < numComponents; c++)
          outData[c] = (float)((uint16_t *)inputData)[c] / float(0x7fff);
        break;
      default:
        throw formatException("Invalid component type for animation {}/{}: {}", gltfAnimation.name, output.name,
                              output.componentType);
      }
    }

    if (gltfChannel.target_node >= 0) {
      result.targetNode = this->nodeMap[gltfChannel.target_node];
      if (gltfChannel.target_path == "rotation") {
        result.target = animation::BuiltinTarget::Rotation;
        if (numComponents != 4)
          throw formatException("Invalid animation {}: rotation requires 4 components", gltfAnimation.name);
      } else if (gltfChannel.target_path == "translation") {
        result.target = animation::BuiltinTarget::Translation;
        if (numComponents != 3)
          throw formatException("Invalid animation {}: translation requires 3 components", gltfAnimation.name);
      } else if (gltfChannel.target_path == "scale") {
        result.target = animation::BuiltinTarget::Scale;
        if (numComponents != 3)
          throw formatException("Invalid animation {}: scale requires 3 components", gltfAnimation.name);
      } else if (gltfChannel.target_path == "weights") {
        SPDLOG_WARN("morph target animations are not implemented");
      } else {
        throw formatException("Invalid animation target {}: {}", gltfAnimation.name, gltfChannel.target_path);
      }
    }

    return result;
  }

  void loadAnimations() {
    size_t numAnimations = model.animations.size();
    for (size_t i = 0; i < numAnimations; i++) {
      tinygltf::Animation &gltfAnimation = model.animations[i];
      auto &animation = animations.emplace(gltfAnimation.name, Animation()).first->second;

      for (auto &gltfChannel : gltfAnimation.channels) {
        auto track = loadAnimationTrack(gltfAnimation, gltfChannel);
        if (track.target != animation::BuiltinTarget::None)
          animation.tracks.emplace_back(std::move(track));
      }
    }
  }

  void loadSkins() {
    size_t numSkins = model.skins.size();
    for (size_t i = 0; i < numSkins; i++) {
      tinygltf::Skin &gltfSkin = model.skins[i];
      auto &skin = skins.emplace_back();
      skin = std::make_shared<Skin>();

      // Read inverse bind matrix data
      auto loadInverseBindMatrix = [&](int jointIndex) {
        auto &ac = model.accessors[gltfSkin.inverseBindMatrices];
        auto &bv = model.bufferViews[ac.bufferView];
        auto &buf = model.buffers[bv.buffer];
        float *data = (float *)(buf.data.data() + bv.byteOffset + ac.byteOffset + jointIndex * ac.ByteStride(bv));
        return float4x4(data);
      };

      size_t jointIndex = 0;
      for (auto &jointNodeIndex : gltfSkin.joints) {
        auto &jointNode = nodeMap[jointNodeIndex];

        skin->joints.push_back(jointNode->getPath());
        skin->inverseBindMatrices.push_back(loadInverseBindMatrix(jointIndex));
        ++jointIndex;
      }
    }

    for (size_t i = 0; i < nodeMap.size(); i++) {
      int skinIndex = model.nodes[i].skin;
      if (skinIndex >= 0) {
        for (auto &drawable : nodeMap[i]->drawables) {
          drawable->skin = skins[skinIndex];
        }
      }
    }
  }

  void load() {
    allocateNodes();
    loadTextures();
    // loadMaterials();
    loadMeshes();
    loadNodes();

    // Unify everyhing into a single node per scene
    // needs to be done before computing animation paths in skins
    size_t numScenes = model.scenes.size();
    sceneMap.resize(numScenes);
    for (size_t i = 0; i < numScenes; i++) {
      const Scene &gltfScene = model.scenes[i];

      MeshTreeDrawable::Ptr &scene = sceneMap[i];

      if (gltfScene.nodes.size() == 1) {
        // Extract single model node
        scene = nodeMap[gltfScene.nodes[0]];
      } else {
        // Group into parent node
        scene = std::make_shared<MeshTreeDrawable>();
        for (size_t i = 0; i < gltfScene.nodes.size(); i++) {
          int nodeIndex = gltfScene.nodes[i];
          MeshTreeDrawable::Ptr node = nodeMap[nodeIndex];
          scene->addChild(node);
        }
      }
    }

    loadSkins();
    loadAnimations();
  }
};

glTF2 processGltfModel2(tinygltf::Model &model, const fs::Path &rootPath) {
  if (model.defaultScene == -1) {
    if (model.scenes.empty())
      throw formatException("glTF2 model has no scenes");
    if (model.scenes.size() > 1)
      SPDLOG_WARN("Model has no default scene out of {} scenes, using first scene", model.scenes.size());
    model.defaultScene = 0;
  }

  Loader2 gfxLoader(model, rootPath);
  gfxLoader.load();

  glTF2 result;
  result.root = std::move(gfxLoader.sceneMap[model.defaultScene]);
  result.animations = std::move(gfxLoader.animations);
  result.materials = std::move(gfxLoader.materials);
  return result;
}

static inline bool isPossiblyText(const fs::path &inFilepath) {
  const auto &ext = inFilepath.extension();
  return ext == ".gltf2" || ext == ".json" || ext == ".txt";
}

static inline bool isBinaryFormat(boost::span<uint8_t> data) {
  if (data.size() >= 12) {
    if (((uint32_t *)data.data())[0] == 0x46546C67)
      return true;
  }
  return false;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
struct GLTFIO {
  std::shared_ptr<data::DataCache> dataCache;
  std::unordered_map<int, data::AssetInfo> imageAssetMap;

  GLTFIO(const std::shared_ptr<data::DataCache> &dataCache) : dataCache(dataCache) {}

  tinygltf::FsCallbacks getFSCallbacks() {
    tinygltf::FsCallbacks ret;
    ret.user_data = this;
    ret.FileExists = [](const std::string &abs_filename, void *user_data) -> bool {
      GLTFIO &self = *static_cast<GLTFIO *>(user_data);
      return true;
    };
    ret.ExpandFilePath = [](const std::string &abs_filename, void *user_data) -> std::string {
      GLTFIO &self = *static_cast<GLTFIO *>(user_data);
      return abs_filename;
    };
    ret.ReadWholeFile = [](std::vector<unsigned char> *out, std::string *err, const std::string &abs_filename,
                           void *user_data) -> bool {
      GLTFIO &self = *static_cast<GLTFIO *>(user_data);
      std::fstream file(abs_filename, std::ios::in | std::ios::binary);
      if (!file.is_open()) {
        return false;
      }
      file.seekg(0, std::ios::end);
      size_t size = file.tellg();
      file.seekg(0, std::ios::beg);
      out->resize(size);
      file.read((char *)out->data(), size);
      return true;
      // return false;
    };
    ret.WriteWholeFile = [](std::string *out, const std::string &abs_filename, const std::vector<unsigned char> &data,
                            void *user_data) -> bool {
      GLTFIO &self = *static_cast<GLTFIO *>(user_data);
      return false;
    };
    return ret;
  }
  tinygltf::URICallbacks getUriCallbacks() {
    tinygltf::URICallbacks ret;
    ret.user_data = this;
    ret.encode = [](const std::string &in_uri, const std::string &object_type, std::string *out_uri, void *user_data) -> bool {
      GLTFIO &self = *static_cast<GLTFIO *>(user_data);
      return false;
    };
    ret.decode = [](const std::string &in_uri, std::string *out_uri, void *user_data) -> bool {
      GLTFIO &self = *static_cast<GLTFIO *>(user_data);
      return false;
    };
    return ret;
  }

  tinygltf::LoadImageDataFunction getLoadImageDataFunction() {
    return false;
    // return [](tinygltf::Image *image, int image_idx, std::string *err, std::string *warn, int req_width, int req_height,
    //           const unsigned char *bytes, int size, void *user_data) -> bool {
    //   GLTFIO &self = *static_cast<GLTFIO *>(user_data);

    //   // Resolve or store into the asset cache
    //   auto key = self.dataCache->generateSourceKey(boost::span<uint8_t>(image->image), data::AssetCategory::Image);
    //   if (!self.dataCache->hasAsset(key)) {
    //     self.dataCache->store(key, image->image);
    //   }

    //   // Encode the asset UUID into the image data
    //   image->image.resize(sizeof(boost::uuids::uuid));
    //   ((boost::uuids::uuid *)image->image.data())[0] = key.key;

    //   return true;
    // };
  }
};
#pragma clang diagnostic pop

tinygltf::Model loadGltfModelFromFile2(std::string_view inFilepath) {
  tinygltf::TinyGLTF context;
  GLTFIO io(gfx::data::getInstance());
  context.SetFsCallbacks(io.getFSCallbacks());
  context.SetURICallbacks(io.getUriCallbacks());
  context.SetImageLoader(io.getLoadImageDataFunction(), &io);

  tinygltf::Model model;
  fs::path filepath((std::string(inFilepath)));

  if (!fs::exists(filepath)) {
    throw formatException("glTF2 model file \"{}\" does not exist", filepath.string());
  }

  std::string err;
  std::string warn;
  bool success{};
  if (isPossiblyText(filepath)) {
    success = context.LoadASCIIFromFile(&model, &err, &warn, filepath.string());
  } else {
    success = context.LoadBinaryFromFile(&model, &err, &warn, filepath.string());
  }

  if (!success) {
    throw formatException("Failed to load glTF2 model \"{}\"\nErrors: {}\nWarnings: {}", filepath.string(), err, warn);
  }
  return model;
}

tinygltf::Model loadGltfModelFromMemory2(boost::span<uint8_t> data, const fs::Path &rootPath) {
  tinygltf::Model model;
  tinygltf::TinyGLTF context;

  std::string err;
  std::string warn;
  bool success;
  if (isBinaryFormat(data)) {
    success = context.LoadBinaryFromMemory(&model, &err, &warn, data.data(), data.size());
  } else {
    std::string rootPath1 = rootPath.string();
    success = context.LoadASCIIFromString(&model, &err, &warn, (char *)data.data(), data.size(), rootPath1);
  }

  if (!success) {
    throw std::runtime_error(std::string() + "Failed to load glTF2 model (" + err + " " + warn + ")");
  }
  return model;
}

// glTF2 loadGltfFromMemory2(const uint8_t *data, size_t dataLength) {
//   tinygltf::TinyGLTF context;
//   auto loader = [&](tinygltf::Model &model) {
//     std::string err;
//     std::string warn;
//     bool success{};

//     success = context.LoadBinaryFromMemory(&model, &err, &warn, data, dataLength);

//     if (!success) {
//       throw formatException("Failed to load glTF2 model \nErrors: {}\nWarnings: {}", err, warn);
//     }
//   };

//   return load(loader);
// }
} // namespace gfx
