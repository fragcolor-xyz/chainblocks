#[macro_use]
extern crate lazy_static;

use shards::core::{register_enum, register_shard};
use shards::types::{Type, FRAG_CC};
use shards::{fourCharacterCode, ref_counted_object_type_impl};

use candle_core::{DType, Tensor};

mod tensor;
mod tokenizer;

struct TensorWrapper(Tensor);
ref_counted_object_type_impl!(TensorWrapper);

lazy_static! {
  pub static ref TENSOR_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"cTEN")); // last letter used as version
  pub static ref TENSOR_TYPE_VEC: Vec<Type> = vec![*TENSOR_TYPE];
  pub static ref TENSOR_VAR_TYPE: Type = Type::context_variable(&TENSOR_TYPE_VEC);
}

#[derive(shards::shards_enum, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[enum_info(b"TENt", "TensorType", "The type of the tensor.")]
pub enum TensorType {
  #[enum_value("An unsigned 8-bit integer tensor.")]
  U8 = 0x0,
  #[enum_value("An unsigned 32-bit integer tensor.")]
  U32 = 0x1,
  #[enum_value("A signed 64-bit integer tensor.")]
  I64 = 0x2,
  #[enum_value("A brain floating-point 16-bit tensor.")]
  BF16 = 0x3,
  #[enum_value("A floating-point 16-bit tensor.")]
  F16 = 0x4,
  #[enum_value("A floating-point 32-bit tensor.")]
  F32 = 0x5,
  #[enum_value("A floating-point 64-bit tensor.")]
  F64 = 0x6,
}

impl From<DType> for TensorType {
  fn from(dtype: DType) -> Self {
    match dtype {
      DType::U8 => TensorType::U8,
      DType::U32 => TensorType::U32,
      DType::I64 => TensorType::I64,
      DType::BF16 => TensorType::BF16,
      DType::F16 => TensorType::F16,
      DType::F32 => TensorType::F32,
      DType::F64 => TensorType::F64,
    }
  }
}

impl From<TensorType> for DType {
  fn from(tensor_type: TensorType) -> Self {
    match tensor_type {
      TensorType::U8 => DType::U8,
      TensorType::U32 => DType::U32,
      TensorType::I64 => DType::I64,
      TensorType::BF16 => DType::BF16,
      TensorType::F16 => DType::F16,
      TensorType::F32 => DType::F32,
      TensorType::F64 => DType::F64,
    }
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_ml_rust(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }

  register_shard::<tokenizer::MLTokenizer>();
  register_shard::<tokenizer::TokensShard>();
  register_enum::<TensorType>();
  register_shard::<tensor::MLTensorToStringShard>();
  register_shard::<tensor::TensorShard>();
}
