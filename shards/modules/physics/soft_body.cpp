#include <shards/core/shared.hpp>
#include <shards/core/params.hpp>
#include <shards/linalg_shim.hpp>
#include <shards/common_types.hpp>
#include <gfx/worker_memory.hpp>
#include <gfx/transform_updater.hpp>
#include "../gfx/shards_types.hpp"
#include "physics.hpp"
#include "core.hpp"

namespace shards::Physics {

enum SoftBodyParamTypeMask : uint32_t {
  SBPTM_None = 0,
  SBPTM_Friction = 1 << 0,
  SBPTM_Restitution = 1 << 1,
  SBPTM_Pressure = 1 << 2,
  SBPTM_LinearDamping = 1 << 3,

  SBPTM_MaxLinearVelocity = 1 << 4,
  SBPTM_GravityFactor = 1 << 5,
  SBPTM_MotionType = 1 << 6,
  SBPTM_CollisionGroup = 1 << 7,
};

static SoftBodyParamTypeMask &operator|=(SoftBodyParamTypeMask &lhs, SoftBodyParamTypeMask rhs) {
  return lhs = static_cast<SoftBodyParamTypeMask>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

struct SoftBodyShard {
  static SHTypesInfo inputTypes() { return shards::CoreInfo::AnyType; }
  static SHTypesInfo outputTypes() { return SHBody::Type; }
  static SHOptionalString help() { return SHCCSTR("Defines a new node"); }

  static inline Type PhysicsDOFSeqType = Type::SeqOf(PhysicsDOFEnumInfo::Type);
  static inline Types PhysicsDOFTypes{PhysicsDOFEnumInfo::Type, PhysicsDOFSeqType, Type::VariableOf(PhysicsDOFSeqType)};

  PARAM_PARAMVAR(_location, "Location", "The initial location, updated by physics simulation", {shards::CoreInfo::Float3VarType});
  PARAM_PARAMVAR(_rotation, "Rotation", "The initial location, updated by physics simulation", {shards::CoreInfo::Float4VarType});
  PARAM_VAR(_static, "Static", "Static node, persist when not activated", {shards::CoreInfo::BoolType});
  PARAM_PARAMVAR(_enabled, "Enabled", "Can be used to toggle this node when it has static persistence",
                 {shards::CoreInfo::BoolType, shards::CoreInfo::BoolVarType});
  PARAM_PARAMVAR(_shape, "Shape", "The shape of the body", {SHSoftBodyShape::VarType});
  PARAM_PARAMVAR(_friction, "Friction", "", {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_restitution, "Restitution", "Restitution coefficient",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_linearDamping, "LinearDamping", "Linear damping coefficient",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_maxLinearVelocity, "MaxLinearVelocity", "Max linear velocity",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_gravityFactor, "GravityFactor", "Gravity factor",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_pressure, "Pressure", "Pressure", {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(
      _collisionGroup, "CollisionGroup",
      "Collision filtering type (the first component contains group membership mask, the second part contains a filter mask)"
      "If any bits match the filter of the other, the two objects will collide",
      {CoreInfo::Int2Type, CoreInfo::Int2VarType});
  PARAM_PARAMVAR(_tag, "Tag", "Tag for the body used in collision events", {CoreInfo::AnyType});
  PARAM_PARAMVAR(_context, "Context", "The physics context", {ShardsContext::VarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_location), PARAM_IMPL_FOR(_rotation), PARAM_IMPL_FOR(_static), PARAM_IMPL_FOR(_enabled),
             PARAM_IMPL_FOR(_shape), PARAM_IMPL_FOR(_friction), PARAM_IMPL_FOR(_restitution), PARAM_IMPL_FOR(_linearDamping),
             PARAM_IMPL_FOR(_maxLinearVelocity), PARAM_IMPL_FOR(_gravityFactor), PARAM_IMPL_FOR(_pressure),
             PARAM_IMPL_FOR(_collisionGroup), PARAM_IMPL_FOR(_tag), PARAM_IMPL_FOR(_context));

  SoftBodyParamTypeMask _dynamicParameterMask = SBPTM_None;

  struct Instance {
    OwnedVar var;
    SHBody *node{};

    Instance() {}
    Instance(Instance &&other) = default;
    Instance &operator=(Instance &&other) = default;
    void warmup() {
      auto [node, var] = SHBody::ObjectVar.NewOwnedVar();
      this->var = std::move(var);
      this->node = &node;
    }
    void cleanup() {
      var.reset();
      node = nullptr;
    }
    SHBody *operator->() { return node; }
  };
  Instance _instance;
  RequiredContext _requiredContext;

  SoftBodyShard() {
    _static = Var(false);
    _enabled = Var(true);
    _context = Var::ContextVar(ShardsContext::VariableName);
    _friction = Var(0.2f);
    _restitution = Var(0.0f);
    _linearDamping = Var(0.05f);
    _maxLinearVelocity = Var(500.0f);
    _gravityFactor = Var(1.0f);
    _pressure = Var(0.0f);
    _collisionGroup = toVar(int2(0xFFFFFFFF, 0xFFFFFFFF));
  }

  void warmup(SHContext *context) {
    PARAM_WARMUP(context);
    _requiredContext.warmup(context, &_context);
    _instance.warmup();
  }

  void cleanup(SHContext *context) {
    _requiredContext.cleanup(context);
    PARAM_CLEANUP(context);
    _instance.cleanup();
  }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    _requiredContext.compose(data, _requiredVariables);

    if (!_location.isVariable())
      throw std::runtime_error("Location must be a variable");
    auto lv = findExposedVariable(data.shared, _location);
    if (!lv || !lv->isMutable)
      throw std::runtime_error("Location must be mutable");

    if (!_rotation.isVariable())
      throw std::runtime_error("Rotation must be a variable");
    auto rv = findExposedVariable(data.shared, _rotation);
    if (!rv || !rv->isMutable)
      throw std::runtime_error("Rotation must be mutable");

    if (!_shape.isVariable())
      throw std::runtime_error("Shape must be set");

    // Determine dynamic parameters
    if (_friction.isVariable())
      _dynamicParameterMask |= SBPTM_Friction;
    if (_restitution.isVariable())
      _dynamicParameterMask |= SBPTM_Restitution;
    if (_pressure.isVariable())
      _dynamicParameterMask |= SBPTM_Pressure;
    if (_linearDamping.isVariable())
      _dynamicParameterMask |= SBPTM_LinearDamping;
    if (_maxLinearVelocity.isVariable())
      _dynamicParameterMask |= SBPTM_MaxLinearVelocity;
    if (_gravityFactor.isVariable())
      _dynamicParameterMask |= SBPTM_GravityFactor;
    if (_collisionGroup.isVariable())
      _dynamicParameterMask |= SBPTM_CollisionGroup;

    return outputTypes().elements[0];
  }

  JPH::EAllowedDOFs convertAllowedDOFs(const SHVar &input) {
    if (input.valueType != SHType::Seq) {
      return (JPH::EAllowedDOFs)input.payload.enumValue;
    } else {
      JPH::EAllowedDOFs dofs = JPH::EAllowedDOFs::None;
      for (auto &dof : input) {
        dofs |= (JPH::EAllowedDOFs)dof.payload.enumValue;
      }
      return dofs;
    }
  }

  void initNewNode(std::shared_ptr<BodyNode> node) {
    if ((bool)*_static)
      node->persistence = true;
    node->enabled = _enabled.get().payload.boolValue;
    node->tag = _tag.get();

    auto &outParams = node->params.soft;
    outParams.friction = _friction.get().payload.floatValue;
    outParams.restitution = _restitution.get().payload.floatValue;
    outParams.pressure = _pressure.get().payload.floatValue;
    outParams.linearDamping = _linearDamping.get().payload.floatValue;

    outParams.maxLinearVelocity = _maxLinearVelocity.get().payload.floatValue;
    outParams.gravityFactor = _gravityFactor.get().payload.floatValue;

    auto cm = toUInt2(_collisionGroup.get());
    outParams.groupMembership = cm.x;
    outParams.collisionMask = cm.y;

    auto &shardsShape = varAsObjectChecked<SHSoftBodyShape>(_shape.get(), SHSoftBodyShape::Type);
    shassert(shardsShape.settings && "Invalid shape");
    node->shape = shardsShape.settings;
    node->shapeUid = shardsShape.uid;

    memcpy(&node->location.payload, &_location.get().payload, sizeof(SHVarPayload));
    memcpy(&node->rotation.payload, &_rotation.get().payload, sizeof(SHVarPayload));
    node->prevLocation = node->location;
    node->prevRotation = node->rotation;

    node->updateParamHash0();
    node->updateParamHash1();
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {
    ZoneScopedN("Physics::entity");

    if (!_instance->node) {
      _instance->node = _requiredContext->core->createNewBodyNode();
      _instance->core = _requiredContext->core;
      initNewNode(_instance->node);
    }

    auto &instNode = _instance->node;

    instNode->enabled = _enabled.get().payload.boolValue;
    _requiredContext->core->touchNode(instNode);

    // Filtering to avoid unnecessary checks for static parameters
    auto &outParams = instNode->params.soft;
    if ((_dynamicParameterMask & 0b1111) != 0) {
      if (_dynamicParameterMask & SBPTM_Friction) {
        outParams.friction = _friction.get().payload.floatValue;
      }
      if (_dynamicParameterMask & SBPTM_Restitution) {
        outParams.restitution = _restitution.get().payload.floatValue;
      }
      if (_dynamicParameterMask & SBPTM_Pressure) {
        outParams.pressure = _pressure.get().payload.floatValue;
      }
      if (_dynamicParameterMask & SBPTM_LinearDamping) {
        outParams.linearDamping = _linearDamping.get().payload.floatValue;
      }
      instNode->updateParamHash0();
    }
    if ((_dynamicParameterMask & 0b11110000) != 0) {
      if (_dynamicParameterMask & SBPTM_MaxLinearVelocity) {
        outParams.maxLinearVelocity = _maxLinearVelocity.get().payload.floatValue;
      }
      if (_dynamicParameterMask & SBPTM_GravityFactor) {
        outParams.gravityFactor = _gravityFactor.get().payload.floatValue;
      }
      if (_dynamicParameterMask & SBPTM_CollisionGroup) {
        auto cm = toUInt2(_collisionGroup.get());
        outParams.groupMembership = cm.x;
        outParams.collisionMask = cm.y;
      }
      instNode->updateParamHash1();
    }

    auto &shardsShape = varAsObjectChecked<SHSoftBodyShape>(_shape.get(), SHSoftBodyShape::Type);
    shassert(shardsShape.settings && "Invalid shape");
    instNode->shape = shardsShape.settings;
    instNode->shapeUid = shardsShape.uid;

    bool updatePose{};
    if (memcmp(&instNode->prevLocation.payload, &_location.get().payload.float3Value, sizeof(SHFloat3)) != 0) {
      memcpy(&instNode->location.payload, &_location.get().payload, sizeof(SHVarPayload));
      updatePose = true;
    }
    if (memcmp(&instNode->prevRotation.payload, &_rotation.get().payload.float4Value, sizeof(SHFloat4)) != 0) {
      memcpy(&instNode->rotation.payload, &_rotation.get().payload, sizeof(SHVarPayload));
      updatePose = true;
    }

    if (updatePose) {
      instNode->data->getPhysicsObject()->SetPositionAndRotationInternal(toJPHVec3(instNode->location),
                                                                         toJPHQuat(instNode->rotation));
    } else {
      assignVariableValue(_location.get(), toVar(instNode->location));
      assignVariableValue(_rotation.get(), toVar(instNode->rotation));
    }

    return _instance.var;
  }
};

struct SoftBodyBuilder {
  JPH::Ref<JPH::SoftBodySharedSettings> settings;

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
    auto &srcFormat = mesh->getFormat();

auto desc = std::get<gfx::MeshDescCPUCopy>(mesh->getDesc());

    auto &vertexData = desc.vertexData;
    auto &indexData = desc.indexData;

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
      }
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

    const uint8_t *basePtr = vertexData.data() + *positionOffset;
    for (size_t i = 0; i < desc.getNumVertices(); i++) {
      auto &pt = *(float3 *)(basePtr + srcStride * i);
      float3 transformedP = linalg::mul(worldTransform, float4(pt, 1.0)).xyz();
      auto &vertex = settings->mVertices.emplace_back();
      vertex.mPosition = toJPHFloat3(transformedP);
    }
    if (desc.getNumIndices() > 0) {
      size_t numFaces = desc.getNumIndices() / 3;
      if (srcFormat.indexFormat == gfx::IndexFormat::UInt16) {
        const uint16_t *basePtr = (const uint16_t *)indexData.data();
        for (size_t i = 0; i < numFaces; i++) {
          settings->AddFace(JPH::SoftBodySharedSettings::Face(basePtr[i * 3 + 0], basePtr[i * 3 + 1], basePtr[i * 3 + 2]));
        }
      } else {
        const uint32_t *basePtr = (const uint32_t *)indexData.data();
        for (size_t i = 0; i < numFaces; i++) {
          settings->AddFace(JPH::SoftBodySharedSettings::Face(basePtr[i * 3 + 0], basePtr[i * 3 + 1], basePtr[i * 3 + 2]));
        }
      }
    } else {
      size_t numFaces = desc.getNumIndices() / 3;
      for (size_t i = 0; i < numFaces; i++) {
        settings->AddFace(JPH::SoftBodySharedSettings::Face(i * 3 + 0, i * 3 + 1, i * 3 + 2));
      }
    }
  }
};

struct MeshSrcTypes {
  static inline shards::Types Types{gfx::ShardsTypes::Mesh, gfx::ShardsTypes::Drawable};
};

// Sphere shape
struct SoftBodyShape {
  static SHTypesInfo inputTypes() { return MeshSrcTypes::Types; }
  static SHTypesInfo outputTypes() { return SHSoftBodyShape::Type; }
  static SHOptionalString help() { return SHCCSTR(""); }

  PARAM_PARAMVAR(_compliance, "Compliance", "Compliance", {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_shearCompliance, "ShearCompliance", "Shear compliance",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_PARAMVAR(_bendCompliance, "BendCompliance", "Bend compliance",
                 {shards::CoreInfo::FloatType, shards::CoreInfo::FloatVarType});
  PARAM_IMPL(PARAM_IMPL_FOR(_compliance), PARAM_IMPL_FOR(_shearCompliance), PARAM_IMPL_FOR(_bendCompliance));

  OwnedVar _output;

  SoftBodyShape() {
    _compliance = Var(1.0e-4f);
    _shearCompliance = Var(1.0e-4f);
    _bendCompliance = Var(1.0e-3f);
  }

  void warmup(SHContext *context) { PARAM_WARMUP(context); }
  void cleanup(SHContext *context) { PARAM_CLEANUP(context); }

  PARAM_REQUIRED_VARIABLES();
  SHTypeInfo compose(SHInstanceData &data) {
    PARAM_COMPOSE_REQUIRED_VARIABLES(data);
    return outputTypes().elements[0];
  }

  SHVar activate(SHContext *shContext, const SHVar &input) {

    auto [shape, var] = SHSoftBodyShape::ObjectVar.NewOwnedVar();

    JPH::Ref<JPH::SoftBodySharedSettings> settings = new JPH::SoftBodySharedSettings();
    SoftBodyBuilder _builder{settings};
    _builder.add(input);

    JPH::SoftBodySharedSettings::VertexAttributes attrib;
    attrib.mCompliance = _compliance.get().payload.floatValue;
    attrib.mShearCompliance = _shearCompliance.get().payload.floatValue;
    attrib.mBendCompliance = _bendCompliance.get().payload.floatValue;

    settings->CreateConstraints(&attrib, 1, JPH::SoftBodySharedSettings::EBendType::None, JPH::DegreesToRadians(8.0f));
    settings->Optimize();
    shape.settings = settings;

    return (_output = std::move(var));
  }
};
} // namespace shards::Physics

SHARDS_REGISTER_FN(soft_body) {
  using namespace shards::Physics;
  REGISTER_SHARD("Physics.SoftBody", SoftBodyShard);
  REGISTER_SHARD("Physics.SoftBodyShape", SoftBodyShape);
}
