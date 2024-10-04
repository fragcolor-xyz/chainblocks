#ifndef FE9A8471_61AF_483B_8D2F_84E570864FB2
#define FE9A8471_61AF_483B_8D2F_84E570864FB2

#include "../gfx_wgpu.hpp"
#include "../unique_id.hpp"
#include "../drawables/mesh_drawable.hpp"
#include "../drawables/mesh_tree_drawable.hpp"
#include "../drawable_processor.hpp"
#include "../renderer_storage.hpp"
#include "../shader/types.hpp"
#include "../texture_placeholder.hpp"
#include "../sampler_cache.hpp"
#include "../mesh_utils.hpp"
#include "../drawable_processor_helpers.hpp"
#include "../pipeline_builder.hpp"
#include "../renderer_types.hpp"
#include "../shader/struct_layout.hpp"
#include <shards/core/pmr/unordered_map.hpp>
#include <shards/core/pmr/vector.hpp>
#include <shards/core/pmr/list.hpp>
#include <shards/defer.hpp>
#include <boost/container/flat_map.hpp>
#include <spdlog/spdlog.h>
#include <functional>
#include <tracy/Wrapper.hpp>

namespace gfx::detail {

struct SharedBufferPool {
private:
  WGPUBufferPool bufferPool;
  std::mutex mutex;

public:
  SharedBufferPool(std::function<WGPUBuffer(size_t)> &&init) : bufferPool(std::move(init)) {}

  WGPUBuffer allocate(size_t size) {
    std::lock_guard<std::mutex> _lock(mutex);
    return bufferPool.allocateBuffer(size);
  }

  void reset() {
    std::lock_guard<std::mutex> _lock(mutex);
    bufferPool.recycle();
  }
};

inline void allocateBuffer(SharedBufferPool &pool, PreparedBuffer &preparedBuffer, size_t bindGroup, const BufferBinding &binding,
                           size_t numElements) {
  std::visit(
      [&](auto &&arg) {
        using T1 = std::decay_t<decltype(arg)>;

        preparedBuffer.binding = binding.index;
        preparedBuffer.bindGroup = bindGroup;
        preparedBuffer.stride = binding.layout.getArrayStride();
        preparedBuffer.bindDimension = bind::PerInstance{};
        if constexpr (std::is_same_v<T1, shader::dim::One>) {
          preparedBuffer.length = preparedBuffer.stride;
          preparedBuffer.bindDimension = bind::One{};
        } else if constexpr (std::is_same_v<T1, shader::dim::PerInstance>) {
          preparedBuffer.length = preparedBuffer.stride * numElements;
          preparedBuffer.bindDimension = bind::PerInstance{};
        } else if constexpr (std::is_same_v<T1, shader::dim::Fixed> || std::is_same_v<T1, shader::dim::Dynamic>) {
          preparedBuffer.length = preparedBuffer.stride * numElements;
          preparedBuffer.bindDimension = bind::Sized{numElements};
        }
        preparedBuffer.buffer = pool.allocate(preparedBuffer.length);
      },
      binding.dimension);
}

struct MeshDrawableProcessor final : public IDrawableProcessor {
  struct DrawableData {
    using allocator_type = shards::pmr::PolymorphicAllocator<>;

    Hash128 groupHash;
    const IDrawable *drawable{};
    size_t queueIndex{}; // Original index in the queue
    MeshContextData *mesh{};
    std::optional<int4> clipRect{};
    shards::pmr::vector<TextureData> textures;
    shards::pmr::vector<BufferData> buffers;
    ParameterStorage parameters; // TODO: Load values directly into buffer
    float projectedDepth{};      // Projected view depth, only calculated when sorting by depth

    DrawableData() = default;
    DrawableData(allocator_type allocator) : textures(allocator), buffers(allocator), parameters(allocator) {}
    DrawableData(DrawableData &&other, allocator_type allocator)
        : textures(allocator), buffers(allocator), parameters(allocator) {
      (*this) = std::move(other);
    }
    DrawableData &operator=(DrawableData &&other) = default;
  };

  struct GroupData {
    WGPUBindGroup drawBindGroup{};
    WgpuHandle<WGPUBindGroup> ownedDrawBindGroup;
    shards::pmr::vector<PreparedBuffer> buffers;
  };

  struct PreparedGroupData {
    using allocator_type = shards::pmr::PolymorphicAllocator<>;

    shards::pmr::vector<DrawGroup> groups;    // the group ranges
    shards::pmr::vector<GroupData> groupData; // Data associated with groups

    PreparedGroupData(allocator_type allocator) : groups(allocator), groupData(allocator) {}
  };

  struct PrepareData {
    using allocator_type = shards::pmr::PolymorphicAllocator<>;

    std::optional<PreparedGroupData> groups;
    shards::pmr::vector<DrawableData *> drawableData{};

    shards::pmr::vector<PreparedBuffer> globalBuffers;
    shards::pmr::vector<BufferData> viewBufferBindings;

    WgpuHandle<WGPUBindGroup> viewBindGroup;

    PrepareData(allocator_type allocator) : drawableData(allocator), globalBuffers(allocator), viewBufferBindings(allocator) {}
  };

  SharedBufferPool uniformBufferPool;
  SharedBufferPool storageBufferPool;
  WGPUSupportedLimits limits{};

  TextureViewCache textureViewCache;
  SamplerCache samplerCache;
  size_t frameCounter{};

  static constexpr size_t NumPlaceholderTextures = 3;
  TexturePtr placeholderTextures[NumPlaceholderTextures]{
      PlaceholderTexture::create(TextureDimension::D1, int2(2, 1), float4(1, 1, 1, 1)),
      PlaceholderTexture::create(TextureDimension::D2, int2(2, 2), float4(1, 1, 1, 1)),
      PlaceholderTexture::create(TextureDimension::Cube, int2(2, 2), float4(1, 1, 1, 1)),
  };

  struct CachedDrawable {
    MeshPtr mesh;
    bool flipped{};
    size_t lastTouched{};
  };
  boost::container::flat_map<UniqueId, CachedDrawable> drawableCache;

  struct MeshCacheKey {
    UniqueId id;
    RequiredAttributes flags;

    MeshCacheKey(UniqueId id, RequiredAttributes flags) : id(id), flags(flags) {}
    std::strong_ordering operator<=>(const MeshCacheKey &other) const = default;
  };
  struct CachedMesh {
    MeshPtr mesh;
    bool hasLocalBasisVector{};
    size_t version{};
    size_t lastTouched{};
  };
  boost::container::flat_map<MeshCacheKey, CachedMesh> meshCache;

  ContextDataCollector contextDataCollector;

  MeshDrawableProcessor(Context &context)
      : uniformBufferPool(getUniformBufferInitializer(context)), storageBufferPool(getStorageBufferInitializer(context)),
        samplerCache(context.wgpuDevice) {
    wgpuDeviceGetLimits(context.wgpuDevice, &limits);
  }

  static std::function<WGPUBuffer(size_t)> getStorageBufferInitializer(Context &context) {
    return [device = context.wgpuDevice](size_t bufferSize) {
      WGPUBufferDescriptor desc{
          .label = "<storage buffer>",
          .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
          .size = bufferSize,
      };
      return wgpuDeviceCreateBuffer(device, &desc);
    };
  }

  static std::function<WGPUBuffer(size_t)> getUniformBufferInitializer(Context &context) {
    return [device = context.wgpuDevice](size_t bufferSize) {
      WGPUBufferDescriptor desc{
          .label = "<uniform buffer>",
          .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform,
          .size = bufferSize,
      };
      return wgpuDeviceCreateBuffer(device, &desc);
    };
  }

  void reset(size_t frameCounter) override {
    ZoneScoped;
    this->frameCounter = frameCounter;
    uniformBufferPool.reset();
    storageBufferPool.reset();

    textureViewCache.clearOldCacheItems(frameCounter, 32);
    clearOldCacheItemsIn(drawableCache, frameCounter, 32);
    clearOldCacheItemsIn(meshCache, frameCounter, 32);
  }

  void buildPipeline(PipelineBuilder &builder, const BuildPipelineOptions &options) override {
    using namespace shader;

    auto &viewBinding = builder.getOrCreateBufferBinding("view");
    viewBinding.bindGroupId = BindGroupId::View;
    auto &viewStructType = viewBinding.structType;
    viewStructType->addField("view", Types::Float4x4);
    viewStructType->addField("proj", Types::Float4x4);
    viewStructType->addField("invView", Types::Float4x4);
    viewStructType->addField("invProj", Types::Float4x4);
    viewStructType->addField("viewport", Types::Float4);

    auto &objectBinding = builder.getOrCreateBufferBinding("object");
    objectBinding.bindGroupId = BindGroupId::Draw;
    objectBinding.addressSpace = shader::AddressSpace::Storage;
    objectBinding.dimension = dim::PerInstance{};
    objectBinding.hasDynamicOffset = true;
    auto &objectStructType = objectBinding.structType;
    objectStructType->addField("world", Types::Float4x4);
    objectStructType->addField("invWorld", Types::Float4x4);
    objectStructType->addField("invTransWorld", Types::Float4x4);

    const MeshDrawable &meshDrawable = static_cast<const MeshDrawable &>(builder.firstDrawable);
    auto &cached = drawableCache[meshDrawable.getId()];

    if (meshDrawable.skin) {
      auto &jointsBufferBinding = builder.getOrCreateBufferBinding("joints");
      jointsBufferBinding.addressSpace = shader::AddressSpace::Storage;
      jointsBufferBinding.bindGroupId = BindGroupId::Draw;
      jointsBufferBinding.dimension = dim::Dynamic{};
      auto &jointsStructType = jointsBufferBinding.structType;
      jointsStructType->addField("transform", Types::Float4x4);
    }

    builder.meshFormat = cached.mesh->getFormat();

    if (isFlippedCoordinateSpace(meshDrawable.transform)) {
      builder.isRenderingFlipped = !builder.isRenderingFlipped;
    }

    if (!options.ignoreDrawableFeatures) {
      if (meshDrawable.material) {
        for (auto &feature : meshDrawable.material->features) {
          builder.features.push_back(feature.get());
        }
      }
      for (auto &feature : meshDrawable.features) {
        builder.features.push_back(feature.get());
      }
    }
  }

  void generateDrawableData(PrepareData *prepareData, DrawableData &data, Context &context, const CachedPipeline &cachedPipeline,
                            const IDrawable *drawable, const ViewData &viewData, const ParameterStorage *baseDrawData,
                            const ParameterStorage *baseViewData, size_t frameCounter, bool needProjectedDepth = false) {
    ZoneScoped;

    const MeshDrawable &meshDrawable = static_cast<const MeshDrawable &>(*drawable);
    const auto &cachedData = drawableCache[meshDrawable.getId()];
    const auto &mesh = cachedData.mesh;

    // Optionally compute projected depth based on transform center
    if (needProjectedDepth) {
      float4 projected = mul(viewData.cachedView.viewProjectionTransform, mul(meshDrawable.transform, float4(0, 0, 0, 1)));
      data.projectedDepth = projected.z;
    }

    data.drawable = drawable;

    ParameterStorage &parameters = data.parameters;

    static FastString fs_world = "world";
    parameters.setParam(fs_world, meshDrawable.transform);

    float4x4 worldInverse = linalg::inverse(meshDrawable.transform);
    static FastString fs_invWorld = "invWorld";
    static FastString fs_invTransWorld = "invTransWorld";
    parameters.setParam(fs_invWorld, worldInverse);
    parameters.setParam(fs_invTransWorld, linalg::transpose(worldInverse));

    auto &textureBindings = cachedPipeline.textureBindingLayout.bindings;
    data.textures.resize(textureBindings.size());

    auto &drawBufferBindings = cachedPipeline.drawBufferBindings;
    for (auto &b : drawBufferBindings) {
      size_t s = b.index + 1;
      if (s > data.buffers.size())
        data.buffers.resize(s);
    }

    auto setTextureParameter = [&](FastString name, const TexturePtr &texture) {
      if (!texture)
        return;
      int32_t targetSlot = mapTextureBinding(textureBindings, name);
      if (targetSlot >= 0) {
        bool isFilterable = textureBindings[targetSlot].type.format == TextureSampleType::Float;
        contextDataCollector.fixup(data.textures[targetSlot].data, texture);
        data.textures[targetSlot].sampler = samplerCache.getDefaultSampler(frameCounter, isFilterable, texture);
        data.textures[targetSlot].id = texture->getId();
      }
    };

    auto setBufferParameter = [&](FastString name, const BufferPtr &buffer) {
      if (!buffer)
        return;
      auto binding = cachedPipeline.findDrawBufferBinding(name);
      if (binding) {
        contextDataCollector.fixup(data.buffers[binding->index].data, buffer);
        data.buffers[binding->index].id = buffer->getId();
      }
    };

    auto applyParameters = [&](const auto &srcParams) {
      for (auto &param : srcParams.basic)
        parameters.setParam(param.first, param.second);

      for (auto &param : srcParams.textures)
        setTextureParameter(param.first, param.second.texture);

      using T = std::decay_t<decltype(srcParams)>;
      if constexpr (std::is_same_v<T, MaterialParameters>) {
        for (auto &param : srcParams.blocks)
          setBufferParameter(param.first, param.second);
      }
    };

    // NOTE : The parameters below are ordered by priority, so later entries overwrite previous entries

    // Grab default parameters
    applyParameters(cachedPipeline.baseDrawParameters);

    // Grab parameters from material
    if (Material *material = meshDrawable.material.get()) {
      applyParameters(material->parameters);
    }

    // Grab parameters from drawable
    applyParameters(meshDrawable.parameters);

    // Grab dynamic parameters from view
    // TODO(guusw): Currently store per-view textures in the object buffer
    //    Need to split the binding logic so that they can be on the view buffer as well
    if (baseViewData) {
      for (auto &pair : baseViewData->textures) {
        setTextureParameter(pair.first, pair.second.texture);
      }
    }

    // Grab dynamic parameters from drawable
    if (baseDrawData) {
      applyParameters(*baseDrawData);
    }

    contextDataCollector.fixup(data.mesh, mesh);
    data.clipRect = meshDrawable.clipRect;

    // Validate texture bindings
    for (size_t i = 0; i < textureBindings.size(); i++) {
      auto textureData = data.textures[i].data;
      if (textureData) {
        auto &binding = textureBindings[i];
        auto expectedFormat = binding.type.format;
        auto expectedSampleTypes = getTextureFormatDescription(textureData->format.pixelFormat).compatibleSampleTypes;
        if ((uint8_t(expectedSampleTypes) & uint8_t(expectedFormat)) == 0) {
          throw std::runtime_error(
              fmt::format("Texture format mismatch on {}, expected a {} sample, but format {} does not support it", binding.name,
                          magic_enum::enum_name(expectedSampleTypes), magic_enum::enum_name(textureData->format.pixelFormat)));
        }
      }
    }

    // Generate grouping hash
    HasherXXH3<HasherDefaultVisitor> hasher;
    hasher(size_t(mesh->getId()));
    hasher(data.clipRect);
    for (auto &texture : data.textures)
      hasher(size_t(texture.id));
    for (auto &buffer : data.buffers)
      hasher(size_t(buffer.id));
    if (meshDrawable.skin) {
      hasher(size_t(meshDrawable.skin.get())); // WARNING: by pointer
    } else {
      hasher(size_t(~0));
    }
    data.groupHash = hasher.getDigest();
  }

  TransientPtr prepare(DrawablePrepareContext &context) override {
    ZoneScoped;

    auto &storage = context.storage;
    auto &allocator = context.storage.workerMemory;
    const CachedPipeline &cachedPipeline = context.cachedPipeline;

    // NOTE: Memory is thrown away at end of frame, deconstructed by renderer
    PrepareData *prepareData = allocator->new_object<PrepareData>();
    prepareData->drawableData.resize(context.drawables.size());

    // Allocate slots for external view buffer bindings
    for (auto &b : cachedPipeline.viewBuffersBindings) {
      size_t s = b.index + 1;
      if (s > prepareData->viewBufferBindings.size())
        prepareData->viewBufferBindings.resize(s);
    }

    // Allocate internal buffers ("object" & "view")
    size_t numDrawables{};
    {
      ZoneScopedN("allocateBuffers");
      numDrawables = context.drawables.size();
      for (auto &binding : cachedPipeline.drawBufferBindings) {
        if (binding.isExternal)
          continue;
        if (std::get_if<shader::dim::PerInstance>(&binding.dimension)) {
          auto &preparedBuffer = prepareData->globalBuffers.emplace_back();
          allocateBuffer(storageBufferPool, preparedBuffer, PipelineBuilder::getDrawBindGroupIndex(), binding, numDrawables);
        }
      }

      // Allocate global(view) buffers
      for (auto &binding : cachedPipeline.viewBuffersBindings) {
        if (binding.isExternal)
          continue;
        auto &preparedBuffer = prepareData->globalBuffers.emplace_back();
        allocateBuffer(uniformBufferPool, preparedBuffer, PipelineBuilder::getViewBindGroupIndex(), binding, 1);
      }
    }

    using DrawableDataVec = boost::container::stable_vector<DrawableData, shards::pmr::polymorphic_allocator<DrawableData>>;
    auto *drawableDatas = allocator->new_object<DrawableDataVec>();
    DEFER({ allocator->destroy(drawableDatas); });

    // Does the following:
    // - Build mesh data
    // - Collect parameters & texture bindings
    // - Generate batch keys
    bool needProjectedDepth = context.sortMode == SortMode::BackToFront;
    {
      ZoneScopedN("generateDrawableData");
      for (size_t index = 0; index < context.drawables.size(); ++index) {
        const IDrawable *drawable = context.drawables[index];
        auto &drawableData = drawableDatas->emplace_back();

        ParameterStorage *baseDrawData{};
        if (context.generatorData.drawParameters) {
          baseDrawData = &(*context.generatorData.drawParameters)[index];
        }

        // TODO: Temporary to store per-view textures in per-object bind group
        ParameterStorage *baseDrawData1 = context.generatorData.viewParameters;

        generateDrawableData(prepareData, drawableData, context.context, cachedPipeline, drawable, context.viewData, baseDrawData,
                             baseDrawData1, context.storage.frameCounter, needProjectedDepth);
      }
    }

    // Sort and group drawables
    {
      ZoneScopedN("sortAndGroup");

      // Collect results from generate task
      size_t insertIndex{};
      for (auto &data : *drawableDatas) {
        prepareData->drawableData[insertIndex++] = &data;
      }
      shassert(insertIndex == context.drawables.size());

      if (context.sortMode == SortMode::Batch) {
        auto comparison = [](const DrawableData *left, const DrawableData *right) { return left->groupHash < right->groupHash; };
        std::stable_sort(prepareData->drawableData.begin(), prepareData->drawableData.end(), comparison);
      } else if (context.sortMode == SortMode::BackToFront) {
        auto compareBackToFront = [](const DrawableData *left, const DrawableData *right) {
          return left->projectedDepth > right->projectedDepth;
        };
        std::stable_sort(prepareData->drawableData.begin(), prepareData->drawableData.end(), compareBackToFront);
      }

      prepareData->groups.emplace(allocator);
      PreparedGroupData &preparedGroupData = prepareData->groups.value();
      {
        ZoneScopedN("groupDrawables");
        groupDrawables(prepareData->drawableData.size(), preparedGroupData.groups,
                       [&](size_t index) { return prepareData->drawableData[index]->groupHash; });
        preparedGroupData.groupData.resize(preparedGroupData.groups.size());
      }
    }

    // Allocate additional data (skinning matrices)
    static FastString fs_joints = "joints";
    auto jointsBinding = cachedPipeline.findDrawBufferBinding(fs_joints);
    if (jointsBinding) {
      ZoneScopedN("packJointsBuffer");

      PreparedGroupData &preparedGroupData = prepareData->groups.value();
      for (size_t index = 0; index < preparedGroupData.groups.size(); ++index) {
        auto &group = preparedGroupData.groups[index];
        auto &groupData = preparedGroupData.groupData[index];

        auto &firstDrawableData = *prepareData->drawableData[group.startIndex];
        const MeshDrawable &meshDrawable = static_cast<const MeshDrawable &>(*firstDrawableData.drawable);

        uint8_t *stagingBuffer{};
        size_t bufferLen{};
        size_t numJoints{};
        if (meshDrawable.skin) {
          auto &skin = *meshDrawable.skin.get();
          numJoints = meshDrawable.skin->joints.size();
          bufferLen = jointsBinding->layout.getArrayStride() * numJoints;
          stagingBuffer = (uint8_t *)allocator->allocate(bufferLen);
          for (size_t k = 0; k < numJoints; k++) {
            float *dst = (float *)(stagingBuffer + jointsBinding->layout.getArrayStride() * k);
            packMatrix(skin.jointMatrices[k], dst);
          }
        } else {
          // This should not happen
          SPDLOG_LOGGER_WARN(getLogger(), "Pipeline built with skinning but mesh doesn't have a skin");
        }

        PreparedBuffer &pb = groupData.buffers.emplace_back();
        allocateBuffer(storageBufferPool, pb, PipelineBuilder::getDrawBindGroupIndex(), *jointsBinding,
                       std::max<size_t>(1, numJoints));
        if (numJoints > 0) {
          wgpuQueueWriteBuffer(context.context.wgpuQueue, pb.buffer, 0, stagingBuffer, bufferLen);
        }
      }
    }

    // Setup view buffer data
    {
      ZoneScopedN("fillViewBuffer");

      // Set hard-coded view parameters (view/projection matrix)
      ParameterStorage viewParameters(allocator);

      // Set default parameters
      for (auto &param : cachedPipeline.baseViewParameters.basic)
        viewParameters.setParam(param.first, param.second);

      // Set builtin view paramters (transforms)
      setViewParameters(viewParameters, context.viewData, context.viewport);

      // Merge generator parameters
      auto baseViewData = context.generatorData.viewParameters;
      if (baseViewData) {
        for (auto &pair : baseViewData->basic) {
          viewParameters.setParam(pair.first, pair.second);
        }
        // TODO: Bind per-view textures here too once bindings are split

        for (auto &buffer : baseViewData->blocks) {
          auto binding = cachedPipeline.findViewBufferBinding(buffer.first);
          if (binding) {
            contextDataCollector.fixup(prepareData->viewBufferBindings[binding->index].data, buffer.second);
            prepareData->viewBufferBindings[binding->index].id = buffer.second->getId();
          }
        }
      }

      for (auto &buffer : prepareData->globalBuffers) {
        if (buffer.bindGroup == PipelineBuilder::getViewBindGroupIndex()) {
          auto &binding = cachedPipeline.viewBuffersBindings[buffer.binding];
          uint8_t *stagingBuffer = (uint8_t *)allocator->allocate(buffer.length);
          packDrawData(stagingBuffer, buffer.stride, binding.layout, viewParameters);
          wgpuQueueWriteBuffer(context.context.wgpuQueue, buffer.buffer, 0, stagingBuffer, buffer.length);
        }
      }
    }

    TextureContextData *placeholderTextureContextData[NumPlaceholderTextures]{};
    {
      ZoneScopedN("fixupContextData");
      for (size_t i = 0; i < std::size(placeholderTextures); i++) {
        contextDataCollector.fixup(&placeholderTextureContextData[i], placeholderTextures[i]);
      }
      contextDataCollector.doFixup(context.storage.contextDataStorage, context.context, context.storage.frameCounter);
    }

    // Setup draw buffer data
    {
      ZoneScopedN("fillDrawBuffer");

      PreparedGroupData &preparedGroupData = prepareData->groups.value();

      for (auto &buffer : prepareData->globalBuffers) {
        if (buffer.bindGroup == PipelineBuilder::getDrawBindGroupIndex() &&
            std::get_if<bind::PerInstance>(&buffer.bindDimension)) {
          auto &binding = cachedPipeline.drawBufferBindings[buffer.binding];
          uint8_t *stagingBuffer = (uint8_t *)allocator->allocate(buffer.length);
          for (auto &group : preparedGroupData.groups) {
            size_t offset = buffer.stride * group.startIndex;
            for (size_t i = 0; i < group.numInstances; i++) {
              packDrawData(stagingBuffer + offset, buffer.stride, binding.layout,
                           prepareData->drawableData[group.startIndex + i]->parameters);
              offset += buffer.stride;
            }
          }
          wgpuQueueWriteBuffer(context.context.wgpuQueue, buffer.buffer, 0, stagingBuffer, buffer.length);
        }
      }
    }

    // Generate bind groups
    WGPUBindGroupLayout viewBindGroupLayout = cachedPipeline.bindGroupLayouts[PipelineBuilder::getViewBindGroupIndex()];

    {
      ZoneScopedN("createViewBindGroup");
      BindGroupBuilder viewBindGroupBuilder(allocator);
      for (size_t i = 0; i < prepareData->globalBuffers.size(); i++) {
        auto &buffer = prepareData->globalBuffers[i];
        if (buffer.bindGroup == PipelineBuilder::getViewBindGroupIndex())
          viewBindGroupBuilder.addBinding(cachedPipeline.viewBuffersBindings[buffer.binding], buffer, numDrawables);
      }

      // Check required buffers
      bool missingBuffers{};
      for (size_t i = 0; i < cachedPipeline.viewBuffersBindings.size(); i++) {
        auto &b = cachedPipeline.viewBuffersBindings[i];
        static FastString fs_view = "view";
        if (b.name != fs_view) {
          if (prepareData->viewBufferBindings[b.index]) {
            viewBindGroupBuilder.addBinding(b, prepareData->viewBufferBindings[b.index].data->buffer);
          } else {
            missingBuffers = true;
            SPDLOG_LOGGER_WARN(getLogger(), fmt::format("missing view buffer binding \"{}\"", b.name));
          }
        }
      }

      if (missingBuffers) {
        // Mark entire pipeline as noop and return
        prepareData->groups.reset();
        return prepareData;
      }

      prepareData->viewBindGroup.reset(viewBindGroupBuilder.finalize(context.context.wgpuDevice, viewBindGroupLayout));
    }

    {
      ZoneScopedN("createDrawBindGroups");

      PreparedGroupData &preparedGroupData = prepareData->groups.value();
      WGPUBindGroupLayout drawBindGroupLayout = cachedPipeline.bindGroupLayouts[PipelineBuilder::getDrawBindGroupIndex()];

      auto cachedBindGroups = allocator->new_object<shards::pmr::unordered_map<Hash128, WGPUBindGroup>>();
      for (size_t index = 0; index < preparedGroupData.groups.size(); ++index) {
        shards::pmr::unordered_map<Hash128, WGPUBindGroup> &cache = *cachedBindGroups;

        auto &group = preparedGroupData.groups[index];
        auto &groupData = preparedGroupData.groupData[index];
        auto &firstDrawableData = *prepareData->drawableData[group.startIndex];

        // Generate texture binding hash
        HasherXXH3<HasherDefaultVisitor> hasher;
        for (auto &v : firstDrawableData.textures)
          hasher(v ? size_t(v.id) : size_t(0));
        for (auto &v : firstDrawableData.buffers)
          hasher(v ? size_t(v.id) : size_t(0));
        for (auto &buffer : groupData.buffers)
          hasher(size_t(buffer.buffer));
        // NOTE: This is hashed since per-index bindings are bound using the required number of instance
        hasher(size_t(group.numInstances));
        Hash128 bindGroupHash = hasher.getDigest();

        auto it = cache.find(bindGroupHash);
        if (it == cache.end()) {
          BindGroupBuilder drawBindGroupBuilder(allocator);
          for (size_t i = 0; i < prepareData->globalBuffers.size(); i++) {
            auto &buffer = prepareData->globalBuffers[i];
            if (buffer.bindGroup == PipelineBuilder::getDrawBindGroupIndex())
              // NOTE: the buffer contains all drawables
              //  however only the number of instances for this group is bound, and later offset using dynamic bindings
              drawBindGroupBuilder.addBinding(cachedPipeline.drawBufferBindings[buffer.binding], buffer, group.numInstances);
          }

          for (auto &buffer : groupData.buffers) {
            // NOTE: the buffer contains all drawables
            //  however only the number of instances for this group is bound, and later offset using dynamic bindings
            drawBindGroupBuilder.addBinding(cachedPipeline.drawBufferBindings[buffer.binding], buffer, group.numInstances);
          }

          // Check required buffers
          bool missingBuffers{};
          for (size_t i = 0; i < cachedPipeline.drawBufferBindings.size(); i++) {
            auto &b = cachedPipeline.drawBufferBindings[i];
            static FastString fs_object = "object";
            static FastString fs_joints = "joints";
            if (b.name != fs_object && b.name != fs_joints) {
              if (firstDrawableData.buffers[b.index]) {
                drawBindGroupBuilder.addBinding(b, firstDrawableData.buffers[b.index].data->buffer, group.numInstances);
              } else {
                missingBuffers = true;
                SPDLOG_LOGGER_WARN(getLogger(), fmt::format("missing object buffer binding \"{}\"", b.name));
              }
            }
          }

          if (missingBuffers) {
            // Mark as noop and continue
            group.numInstances = 0;
            continue;
          }

          // TODO(guusw): Need to split this so that we can have per-view texture bindings
          for (size_t i = 0; i < cachedPipeline.textureBindingLayout.bindings.size(); i++) {
            auto &binding = cachedPipeline.textureBindingLayout.bindings[i];
            auto texture = firstDrawableData.textures[i];
            if (texture) {
              drawBindGroupBuilder.addTextureBinding(binding, textureViewCache.getDefaultTextureView(frameCounter, *texture.data),
                                                     texture.sampler);
            } else {
              auto &placeholder = placeholderTextures[size_t(binding.type.dimension)];
              auto &placeholderCd = *placeholderTextureContextData[size_t(binding.type.dimension)];
              WGPUSampler sampler = samplerCache.getDefaultSampler(frameCounter, true, placeholder);
              drawBindGroupBuilder.addTextureBinding(binding, textureViewCache.getDefaultTextureView(frameCounter, placeholderCd),
                                                     sampler);
            }
          }

          groupData.ownedDrawBindGroup.reset(drawBindGroupBuilder.finalize(context.context.wgpuDevice, drawBindGroupLayout));
          groupData.drawBindGroup = groupData.ownedDrawBindGroup;
          cache.emplace(bindGroupHash, groupData.drawBindGroup);
        } else {
          groupData.drawBindGroup = it->second;
        }
      }
    }

    return prepareData;
  }

  void encode(DrawableEncodeContext &context) override {
    const PrepareData &prepareData = context.preparedData.get<PrepareData>();
    const CachedPipeline &cachedPipeline = context.cachedPipeline;
    auto encoder = context.encoder;
    auto &allocator = context.storage.workerMemory;

    if (!prepareData.groups)
      return;

    wgpuRenderPassEncoderSetPipeline(encoder, cachedPipeline.pipeline);
    wgpuRenderPassEncoderSetBindGroup(encoder, PipelineBuilder::getViewBindGroupIndex(), prepareData.viewBindGroup, 0, nullptr);

    auto &groups = prepareData.groups.value();
    for (size_t i = 0; i < groups.groups.size(); i++) {
      auto &group = groups.groups[i];
      auto &groupData = groups.groupData[i];
      auto &firstDrawable = *prepareData.drawableData[group.startIndex];

      if (group.numInstances == 0)
        continue;

      if (firstDrawable.clipRect) {
        auto clipRect = firstDrawable.clipRect.value();

        int2 viewportMin = int2(context.viewport.x, context.viewport.y);
        int2 viewportMax = int2(context.viewport.getX1(), context.viewport.getY1());

        // Clamp and make relative to viewport
        clipRect.x = linalg::clamp(clipRect.x + viewportMin.x, viewportMin.x, viewportMax.x);
        clipRect.y = linalg::clamp(clipRect.y + viewportMin.y, viewportMin.y, viewportMax.y);
        clipRect.z = linalg::clamp(clipRect.z + viewportMin.x, viewportMin.x, viewportMax.x);
        clipRect.w = linalg::clamp(clipRect.w + viewportMin.y, viewportMin.y, viewportMax.y);
        int w = clipRect.z - clipRect.x;
        int h = clipRect.w - clipRect.y;
        if (w == 0 || h == 0)
          continue; // Discard draw call instead, wgpu doesn't allow w/h == 0
        wgpuRenderPassEncoderSetScissorRect(encoder, clipRect.x, clipRect.y, w, h);
      } else {
        wgpuRenderPassEncoderSetScissorRect(encoder, context.viewport.x, context.viewport.y, context.viewport.width,
                                            context.viewport.height);
      }

      uint32_t *dynamicOffsets = allocator->new_objects<uint32_t>(cachedPipeline.dynamicBufferRefs.size());
      for (size_t i = 0; i < cachedPipeline.dynamicBufferRefs.size(); i++) {
        auto &binding = cachedPipeline.resolveBufferBindingRef(cachedPipeline.dynamicBufferRefs[i]);
        if (std::get_if<shader::dim::PerInstance>(&binding.dimension)) {
          dynamicOffsets[i] = uint32_t(group.startIndex * binding.layout.getArrayStride());
          shassert(dynamicOffsets[i] < binding.layout.getArrayStride() * prepareData.drawableData.size());
        } else {
          dynamicOffsets[i] = 0;
        }
      }
      wgpuRenderPassEncoderSetBindGroup(encoder, PipelineBuilder::getDrawBindGroupIndex(), groupData.drawBindGroup,
                                        cachedPipeline.dynamicBufferRefs.size(), dynamicOffsets);

      MeshContextData *meshContextData = firstDrawable.mesh;
      shassert(meshContextData->vertexBuffer);
      wgpuRenderPassEncoderSetVertexBuffer(encoder, 0, meshContextData->vertexBuffer, 0, meshContextData->vertexBufferLength);

      if (meshContextData->indexBuffer) {
        WGPUIndexFormat indexFormat = getWGPUIndexFormat(meshContextData->format.indexFormat);
        wgpuRenderPassEncoderSetIndexBuffer(encoder, meshContextData->indexBuffer, indexFormat, 0,
                                            meshContextData->indexBufferLength);

        wgpuRenderPassEncoderDrawIndexed(encoder, (uint32_t)meshContextData->numIndices, group.numInstances, 0, 0, 0);
      } else {
        wgpuRenderPassEncoderDraw(encoder, (uint32_t)meshContextData->numVertices, group.numInstances, 0, 0);
      }
    }
  }

  void preprocess(const DrawablePreprocessContext &context) override {
    auto &workerMemory = context.storage.workerMemory;

    auto &hashCache = *workerMemory->new_object<PipelineHashCache>();
    PipelineHashCollector pipelineHashCollector{.storage = &hashCache};

    // Generate another shared hash including base features
    pipelineHashCollector(context.sharedHash);
    for (auto stepFeature : context.features)
      pipelineHashCollector.hashObject(*stepFeature);
    Hash128 sharedHash = pipelineHashCollector.getDigest();

    RequiredAttributes sharedRequiredAttributes;
    for (auto &feature : context.features) {
      sharedRequiredAttributes = sharedRequiredAttributes | feature->requiredAttributes;
    }

    for (size_t i = 0; i < context.drawables.size(); i++) {
      auto &drawable = reinterpret_cast<const MeshDrawable &>(*context.drawables[i]);
      auto &mesh = drawable.mesh;

      pipelineHashCollector.reset();
      pipelineHashCollector(sharedHash);

      RequiredAttributes requiredAttributes = sharedRequiredAttributes;
      if (!context.buildPipelineOptions.ignoreDrawableFeatures) {
        for (auto &feature : drawable.features) {
          pipelineHashCollector(feature);
          requiredAttributes = requiredAttributes | feature->requiredAttributes;
        }
      }

      // Skinned (yes/no)
      pipelineHashCollector((bool)drawable.skin);

      auto &meshCacheEntry = detail::getCacheEntry(meshCache, MeshCacheKey(mesh->getId(), requiredAttributes));
      if (!meshCacheEntry.mesh || meshCacheEntry.version != mesh->getVersion()) {
        meshCacheEntry.mesh = mesh;
        meshCacheEntry.hasLocalBasisVector = false;
        meshCacheEntry.version = mesh->getVersion();

        if (requiredAttributes.requirePerVertexLocalBasis) {
          // Optionally generate tangent mesh
          try {
            const MeshDescCPUCopy *meshDescPtr = std::get_if<MeshDescCPUCopy>(&meshCacheEntry.mesh->getDesc());
            if(!meshDescPtr)
              throw std::runtime_error("MeshDesc is not a MeshDescCPUCopy");
            // TODO
            const MeshDescCPUCopy &meshDesc = *meshDescPtr;
            meshCacheEntry.mesh = generateLocalBasisAttribute(meshDesc);
            meshCacheEntry.hasLocalBasisVector = true;
          } catch (...) {
            SPDLOG_LOGGER_WARN(getLogger(), "Failed to generate tangents for mesh {}", mesh->getId().value);
          }
        }
      }
      touchCacheItem(meshCacheEntry, frameCounter);

      auto &entry = detail::getCacheEntry(drawableCache, drawable.getId());
      entry.mesh = meshCacheEntry.mesh;
      entry.flipped = isFlippedCoordinateSpace(drawable.transform);
      touchCacheItem(entry, frameCounter);

      pipelineHashCollector((bool)entry.flipped);

      pipelineHashCollector(mesh);
      if (drawable.material) {
        pipelineHashCollector(drawable.material);
      }
      drawable.parameters.pipelineHashCollect(pipelineHashCollector);

      context.outHashes[i] = pipelineHashCollector.getDigest();
    }
  }
};
} // namespace gfx::detail

#endif /* FE9A8471_61AF_483B_8D2F_84E570864FB2 */
