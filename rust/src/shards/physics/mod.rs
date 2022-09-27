/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

extern crate crossbeam;
extern crate rapier3d;

use crate::four_character_code;
use crate::shardsc::SHTypeInfo_Details_Object;
use crate::shardsc::SHType_Float4;

use crate::types::common_type;

use crate::types::ExposedInfo;

use crate::types::ParamVar;
use crate::types::Seq;
use crate::types::Type;
use crate::types::FRAG_CC;

use rapier3d::dynamics::{
  CCDSolver, IntegrationParameters, JointSet, RigidBodyHandle, RigidBodySet,
};
use rapier3d::geometry::SharedShape;
use rapier3d::geometry::{
  BroadPhase, ColliderHandle, ColliderSet, ContactEvent, IntersectionEvent, NarrowPhase,
};
use rapier3d::na::{Isometry3, Matrix3, Matrix4, Vector3};
use rapier3d::pipeline::{ChannelEventCollector, PhysicsPipeline, QueryPipeline};
use rapier3d::prelude::IslandManager;

lazy_static! {
  static ref SIMULATION_TYPE: Type = {
    let mut t = common_type::OBJECT;
    t.details.object = SHTypeInfo_Details_Object {
      vendorId: FRAG_CC,
      typeId: four_character_code(*b"phys"),
    };
    t
  };
  static ref EXPOSED_SIMULATION: Vec<ExposedInfo> = vec![ExposedInfo::new_static_with_help(
    cstr!("Physics.Simulation"),
    shccstr!("The physics simulation subsystem."),
    *SIMULATION_TYPE
  )];
  static ref SHAPE_TYPE: Type = {
    let mut t = common_type::OBJECT;
    t.details.object = SHTypeInfo_Details_Object {
      vendorId: FRAG_CC,
      typeId: four_character_code(*b"phyS"),
    };
    t
  };
  static ref SHAPE_TYPE_VEC: Vec<Type> = vec![*SHAPE_TYPE];
  static ref SHAPES_TYPE: Type = Type::seq(&SHAPE_TYPE_VEC);
  static ref SHAPE_VAR_TYPE: Type = Type::context_variable(&SHAPE_TYPE_VEC);
  static ref SHAPES_VAR_TYPE: Type = Type::context_variable(&SHAPE_TYPE_VEC);
  static ref SHAPE_TYPES: Vec<Type> = vec![*SHAPE_TYPE];
  static ref RIGIDBODY_TYPE: Type = {
    let mut t = common_type::OBJECT;
    t.details.object = SHTypeInfo_Details_Object {
      vendorId: FRAG_CC,
      typeId: four_character_code(*b"phyR"),
    };
    t
  };
  static ref RIGIDBODY_TYPE_VEC: Vec<Type> = vec![*RIGIDBODY_TYPE];
  static ref RIGIDBODIES_TYPE: Type = Type::seq(&RIGIDBODY_TYPE_VEC);
  static ref RIGIDBODY_VAR_TYPE: Type = Type::context_variable(&RIGIDBODY_TYPE_VEC);
  static ref RIGIDBODIES_TYPES_SLICE: Vec<Type> = vec![*RIGIDBODY_VAR_TYPE, common_type::NONE];
  static ref SHAPES_TYPES_SLICE: Vec<Type> =
    vec![*SHAPE_VAR_TYPE, *SHAPES_VAR_TYPE, common_type::NONE];
}

static POSITIONS_TYPES_SLICE: &[Type] = &[
  common_type::FLOAT3,
  common_type::FLOAT3_VAR,
  common_type::FLOAT3S,
  common_type::FLOAT3S_VAR,
];
static ROTATIONS_TYPES_SLICE: &[Type] = &[
  common_type::FLOAT4,
  common_type::FLOAT4_VAR,
  common_type::FLOAT4S,
  common_type::FLOAT4S_VAR,
];
static POSITION_TYPES_SLICE: &[Type] = &[common_type::FLOAT3, common_type::FLOAT3_VAR];
static ROTATION_TYPES_SLICE: &[Type] = &[common_type::FLOAT4, common_type::FLOAT4_VAR];

struct Simulation {
  pipeline: PhysicsPipeline,
  islands_manager: IslandManager,
  query_pipeline: QueryPipeline,
  gravity: Vector3<f32>,
  integration_parameters: IntegrationParameters,
  broad_phase: BroadPhase,
  narrow_phase: NarrowPhase,
  bodies: RigidBodySet,
  colliders: ColliderSet,
  joints: JointSet,
  ccd_solver: CCDSolver,
  contacts_channel: crossbeam::channel::Receiver<ContactEvent>,
  intersections_channel: crossbeam::channel::Receiver<IntersectionEvent>,
  event_handler: ChannelEventCollector,
  self_obj: ParamVar,
}

struct RigidBody {
  simulation_var: ParamVar,
  shape_var: ParamVar,
  rigid_bodies: Vec<RigidBodyHandle>,
  colliders: Vec<ColliderHandle>,
  position: ParamVar,
  rotation: ParamVar,
  user_data: u128,
}

struct BaseShape {
  shape: Option<SharedShape>,
  position: Option<Isometry3<f32>>,
}

fn fill_seq_from_mat4(var: &mut Seq, mat: &Matrix4<f32>) {
  unsafe {
    let mat = mat.as_slice();
    var[0].valueType = SHType_Float4;
    var[0].payload.__bindgen_anon_1.float4Value[0] = mat[0];
    var[0].payload.__bindgen_anon_1.float4Value[1] = mat[1];
    var[0].payload.__bindgen_anon_1.float4Value[2] = mat[2];
    var[0].payload.__bindgen_anon_1.float4Value[3] = mat[3];
    var[1].valueType = SHType_Float4;
    var[1].payload.__bindgen_anon_1.float4Value[0] = mat[4];
    var[1].payload.__bindgen_anon_1.float4Value[1] = mat[5];
    var[1].payload.__bindgen_anon_1.float4Value[2] = mat[6];
    var[1].payload.__bindgen_anon_1.float4Value[3] = mat[7];
    var[2].valueType = SHType_Float4;
    var[2].payload.__bindgen_anon_1.float4Value[0] = mat[8];
    var[2].payload.__bindgen_anon_1.float4Value[1] = mat[9];
    var[2].payload.__bindgen_anon_1.float4Value[2] = mat[10];
    var[2].payload.__bindgen_anon_1.float4Value[3] = mat[11];
    var[3].valueType = SHType_Float4;
    var[3].payload.__bindgen_anon_1.float4Value[0] = mat[12];
    var[3].payload.__bindgen_anon_1.float4Value[1] = mat[13];
    var[3].payload.__bindgen_anon_1.float4Value[2] = mat[14];
    var[3].payload.__bindgen_anon_1.float4Value[3] = mat[15];
  }
}

fn mat4_from_seq(var: &Seq) -> Matrix4<f32> {
  unsafe {
    Matrix4::new(
      var[0].payload.__bindgen_anon_1.float4Value[0],
      var[0].payload.__bindgen_anon_1.float4Value[1],
      var[0].payload.__bindgen_anon_1.float4Value[2],
      var[0].payload.__bindgen_anon_1.float4Value[3],
      var[1].payload.__bindgen_anon_1.float4Value[0],
      var[1].payload.__bindgen_anon_1.float4Value[1],
      var[1].payload.__bindgen_anon_1.float4Value[2],
      var[1].payload.__bindgen_anon_1.float4Value[3],
      var[2].payload.__bindgen_anon_1.float4Value[0],
      var[2].payload.__bindgen_anon_1.float4Value[1],
      var[2].payload.__bindgen_anon_1.float4Value[2],
      var[2].payload.__bindgen_anon_1.float4Value[3],
      var[3].payload.__bindgen_anon_1.float4Value[0],
      var[3].payload.__bindgen_anon_1.float4Value[1],
      var[3].payload.__bindgen_anon_1.float4Value[2],
      var[3].payload.__bindgen_anon_1.float4Value[3],
    )
  }
}

fn mat3_from_seq(var: &Seq) -> Matrix3<f32> {
  unsafe {
    Matrix3::new(
      var[0].payload.__bindgen_anon_1.float3Value[0],
      var[0].payload.__bindgen_anon_1.float3Value[1],
      var[0].payload.__bindgen_anon_1.float3Value[2],
      var[1].payload.__bindgen_anon_1.float3Value[0],
      var[1].payload.__bindgen_anon_1.float3Value[1],
      var[1].payload.__bindgen_anon_1.float3Value[2],
      var[2].payload.__bindgen_anon_1.float3Value[0],
      var[2].payload.__bindgen_anon_1.float3Value[1],
      var[2].payload.__bindgen_anon_1.float3Value[2],
    )
  }
}

pub mod forces;
pub mod queries;
pub mod rigidbody;
pub mod shapes;
pub mod simulation;
