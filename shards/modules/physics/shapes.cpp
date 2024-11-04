#include "../gfx/shards_types.hpp"
#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/common_types.hpp>
#include "physics.hpp"

#include <gfx/worker_memory.hpp>
#include <gfx/transform_updater.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>

#include <shards/core/pmr/shared_temp_allocator.hpp>

namespace shards::Physics {

static auto logger = getLogger();

struct BoxShapeShard {
  static SHTypesInfo inputTypes() { return CoreInfo::Float3Type; }
  static SHTypesInfo outputTypes() { return SHShape::Type; }
  static SHOptionalString help() { return SHCCSTR("This shard creates a box collision shape from the input half extents provided."); }
  static SHOptionalString inputHelp() { return SHCCSTR("The x,y and z half extents of the box collision shape to create."); }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the created box collision shape."); }

  PARAM_IMPL();

  OwnedVar _output;

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto [shape, var] = SHShape::ObjectVar.NewOwnedVar();

    JPH::Shape::ShapeResult result;
    JPH::Vec3 v3 = toJPHVec3(input.payload.float3Value);
    JPH::BoxShapeSettings settings(v3);
    JPH::Ref<JPH::BoxShape> hullShape = new JPH::BoxShape(settings, result);

    if (result.HasError()) {
      throw SHException(fmt::format("Failed to create convex hull shape: {}", result.GetError().c_str()));
    }
    shape.shape = hullShape;

    return (_output = std::move(var));
  }
};

struct SphereShapeShard {
  static SHTypesInfo inputTypes() { return CoreInfo::FloatType; }
  static SHTypesInfo outputTypes() { return SHShape::Type; }
  static SHOptionalString help() { return SHCCSTR("Create a sphere collision shape, from the input radius provided."); }
  static SHOptionalString inputHelp() { return SHCCSTR("The radius of the sphere collision shape to create."); }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the created sphere collision shape."); }

  PARAM_IMPL();

  OwnedVar _output;

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto [shape, var] = SHShape::ObjectVar.NewOwnedVar();

    JPH::Shape::ShapeResult result;
    JPH::SphereShapeSettings settings(input.payload.floatValue);
    JPH::Ref<JPH::SphereShape> hullShape = new JPH::SphereShape(settings, result);

    if (result.HasError()) {
      throw SHException(fmt::format("Failed to create convex hull shape: {}", result.GetError().c_str()));
    }
    shape.shape = hullShape;

    return (_output = std::move(var));
  }
};

struct CapsuleShapeShard {
  static SHTypesInfo inputTypes() { return CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return SHShape::Type; }
  static SHOptionalString help() {
    return SHCCSTR("This shard creates a capsule physics collision shape, using the height and radius provided in the HalfHeight and Radius parameters respectively. The capsule will be centered around the origin with one sphere cap at (0, -HalfHeight, 0) and the "
                   "other at (0, HalfHeight, 0).");
  }
  static SHOptionalString inputHelp() { return DefaultHelpText::InputHelpIgnored; }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the created capsule collision shape."); }

  PARAM_PARAMVAR(_halfHeight, "HalfHeight", "Half the height of the capsule.",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_radius, "Radius", "Radius of the capsule.", {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_halfHeight), PARAM_IMPL_FOR(_radius));

  OwnedVar _output;

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    auto [shape, var] = SHShape::ObjectVar.NewOwnedVar();

    JPH::Shape::ShapeResult result;
    JPH::CapsuleShapeSettings settings(_halfHeight.get().payload.floatValue, _radius.get().payload.floatValue);

    result = settings.Create();
    if (result.HasError()) {
      throw SHException(fmt::format("Failed to create convex hull shape: {}", result.GetError().c_str()));
    }
    shape.shape = result.Get();

    return (_output = std::move(var));
  }
};

struct MeshSrcTypes {
  static inline shards::Types Types{gfx::ShardsTypes::Mesh, gfx::ShardsTypes::Drawable};
};

struct PointsBuilder {
  JPH::Array<JPH::Vec3> points;
  void addPoint(const JPH::Vec3 &point) { points.push_back(point); }

  void add(const SHVar &var) {
    Type objType = Type::Object(var.payload.objectVendorId, var.payload.objectTypeId);
    if (objType == gfx::ShardsTypes::Mesh) {
      auto &mesh = *(gfx::MeshPtr *)var.payload.objectValue;
      addMesh(mesh, linalg::identity);
    } else if (objType == gfx::ShardsTypes::Drawable) {
      auto &drawable = *(gfx::SHDrawable *)var.payload.objectValue;
      addDrawable(drawable);
    } else {
      throw SHException("Invalid input type");
    }
  }

  void addDrawable(const gfx::SHDrawable &drawable) {
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, gfx::MeshDrawable::Ptr>) {
            this->addDrawable(arg, linalg::identity);
          } else if constexpr (std::is_same_v<T, gfx::MeshTreeDrawable::Ptr>) {
            this->addDrawable(arg);
          } else {
            throw SHException("Invalid input type");
          }
        },
        drawable.drawable);
  }

  void addDrawable(const gfx::MeshTreeDrawable::Ptr &drawable) {
    pmr::SharedTempAllocator tempAllocator;
    gfx::TransformUpdaterCollector collector(tempAllocator.getAllocator());
    collector.ignoreRootTransform = true;
    collector.collector = [&](const gfx::DrawablePtr &drawable, const float4x4 &worldTransform) {
      if (gfx::MeshDrawable::Ptr ptr = std::dynamic_pointer_cast<gfx::MeshDrawable>(drawable)) {
        this->addDrawable(ptr, worldTransform);
      }
    };
    collector.updateNoModify(*drawable);
  }

  void addDrawable(const gfx::MeshDrawable::Ptr &drawable, const float4x4 &worldTransform) {
    auto &mesh = drawable->mesh;
    if (!mesh) {
      SPDLOG_LOGGER_WARN(logger, "Ignoring drawable with no mesh");
      return;
    }
    addMesh(mesh, worldTransform);
  }

  void addMesh(const gfx::MeshPtr &mesh, const float4x4 &worldTransform) {

    auto desc = std::get<gfx::MeshDescCPUCopy>(mesh->getDesc());
    auto &srcFormat = desc.format;
    auto &vertexData = desc.vertexData;

    std::optional<size_t> positionIndex;
    std::optional<size_t> positionOffset;

    size_t srcStride = srcFormat.getVertexSize();
    size_t offset{};
    for (size_t i = 0; i < srcFormat.vertexAttributes.size(); i++) {
      auto &attrib = srcFormat.vertexAttributes[i];
      if (attrib.name == "position") {
        positionIndex = i;
        positionOffset = offset;
        break;
      }\
      offset += attrib.numComponents * getStorageTypeSize(attrib.type);
    }

    // Validate required attributes
    {
      if (!positionIndex.has_value()) {
        throw std::runtime_error("Mesh does not have a position attribute");
      }
      auto positionAttrib = srcFormat.vertexAttributes[positionIndex.value()];
      if (positionAttrib.type != gfx::StorageType::Float32 && positionAttrib.numComponents != 3) {
        throw std::runtime_error("position attribute must be float3");
      }
    }

    {
      const uint8_t *basePtr = vertexData.data() + *positionOffset;
      for (size_t i = 0; i < desc.getNumVertices(); i++) {
        auto &pt = *(float3 *)(basePtr + srcStride * i);
        float3 transformedP = linalg::mul(worldTransform, float4(pt, 1.0)).xyz();
        points.push_back(toJPHVec3(transformedP));
      }
    }
  }
};

struct MeshHullShapeShard {
  static SHTypesInfo inputTypes() { return MeshSrcTypes::Types; }
  static SHTypesInfo outputTypes() { return SHShape::Type; }
  static SHOptionalString help() { return SHCCSTR("Creates a physics shape from a GFX mesh or drawable input"); }
  static SHOptionalString inputHelp() { return SHCCSTR("A GFX mesh or drawable object."); }
  static SHOptionalString outputHelp() { return SHCCSTR("Outputs the created physics collisionshape."); }

  // See "Convex Radius" in https://jrouwe.github.io/JoltPhysics/index.html
  PARAM_PARAMVAR(_maxConvexRadius, "MaxConvexRadius",
                 "The convex radius given to the collision shape. A larger convex radius results in better performance but a "
                 "less accurate simulation. A convex radius of 0 is allowed",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_maxConvexRadius));

  OwnedVar _output;
  PointsBuilder _builder;

  MeshHullShapeShard() { _maxConvexRadius = Var(JPH::cDefaultConvexRadius); }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    _builder.points.clear();
    _builder.add(input);

    auto [shape, var] = SHShape::ObjectVar.NewOwnedVar();

    JPH::Shape::ShapeResult result;
    JPH::ConvexHullShapeSettings settings(_builder.points, _maxConvexRadius.get().payload.floatValue);
    JPH::Ref<JPH::ConvexHullShape> hullShape = new JPH::ConvexHullShape(settings, result);

    if (result.HasError()) {
      throw SHException(fmt::format("Failed to create convex hull shape: {}", result.GetError().c_str()));
    }
    shape.shape = hullShape;

    return (_output = std::move(var));
  }
};

} // namespace shards::Physics

SHARDS_REGISTER_FN(shapes) {
  using namespace shards::Physics;

  REGISTER_SHARD("Physics.BoxShape", BoxShapeShard);
  REGISTER_SHARD("Physics.SphereShape", SphereShapeShard);
  REGISTER_SHARD("Physics.CapsuleShape", CapsuleShapeShard);
  REGISTER_SHARD("Physics.HullShape", MeshHullShapeShard);
}
