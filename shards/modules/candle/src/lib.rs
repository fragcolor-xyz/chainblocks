#[macro_use]
extern crate lazy_static;

use shards::core::{register_enum, register_object_type, register_shard};
use shards::types::{Type, FRAG_CC};
use shards::{fourCharacterCode, ref_counted_object_type_impl};

use candle_core::{DType, Device, Tensor as CandleTensor};

mod audio;
mod model;
mod tensor;
mod tokenizer;

use once_cell::sync::OnceCell;

static GLOBAL_DEVICE: OnceCell<Device> = OnceCell::new();

pub fn get_global_device() -> &'static Device {
  GLOBAL_DEVICE.get_or_init(|| {
    let mut device = Device::Cpu;

    #[cfg(any(
      target_os = "macos",
      target_os = "ios",
      target_os = "visionos",
      target_os = "tvos"
    ))]
    {
      device = Device::new_metal(0).unwrap_or(Device::Cpu);
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
      device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    }

    device
  })
}

struct Tensor(CandleTensor);
ref_counted_object_type_impl!(Tensor);

lazy_static! {
  pub static ref TENSOR_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"cTEN")); // last letter used as version
  pub static ref TENSOR_TYPE_VEC: Vec<Type> = vec![*TENSOR_TYPE];
  pub static ref TENSOR_VAR_TYPE: Type = Type::context_variable(&TENSOR_TYPE_VEC);
  pub static ref TENSORS_TYPE: Type = Type::seq(&TENSOR_TYPE_VEC);
  pub static ref TENSORS_TYPE_VEC: Vec<Type> = vec![*TENSORS_TYPE];
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

  register_enum::<TensorType>();

  register_shard::<tokenizer::MLTokenizer>();
  register_shard::<tokenizer::MLDetokenizer>();
  register_shard::<tokenizer::TokensShard>();
  register_shard::<tokenizer::MLTimestampDetokenizer>();
  register_shard::<tokenizer::PrepareAudioTokensShard>();

  register_shard::<tensor::MLTensorToStringShard>();
  register_shard::<tensor::TensorShard>();
  register_shard::<model::ModelShard>();
  register_enum::<model::ModelType>();
  register_enum::<model::Formats>();
  register_shard::<tensor::TensorZerosLikeShard>();
  register_shard::<model::ForwardShard>();
  register_shard::<model::SpeechToTextShard>();

  register_shard::<tensor::TensorMulShard>();
  register_shard::<tensor::TensorSubShard>();
  register_shard::<tensor::TensorAddShard>();
  register_shard::<tensor::TensorPowShard>();
  register_shard::<tensor::TensorDivShard>();
  register_shard::<tensor::TensorMatMulShard>();
  register_shard::<tensor::TensorSumShard>();
  register_shard::<tensor::TensorReshapeShard>();
  register_shard::<tensor::TensorTransposeShard>();
  register_shard::<tensor::TensorToFloatShard>();
  register_shard::<tensor::ShapeShard>();
  register_shard::<tensor::TensorSplitShard>();
  register_shard::<tensor::TensorStackShard>();
  register_shard::<tensor::TensorSliceShard>();
  register_shard::<tensor::TensorToIntsShard>();
  register_shard::<tensor::TensorToFloatsShard>();

  register_shard::<audio::MLAudioToMel>();

  register_object_type::<Tensor>(FRAG_CC, fourCharacterCode(*b"cTEN"));
  register_object_type::<tokenizer::Tokenizer>(FRAG_CC, fourCharacterCode(*b"TOKn"));
  register_object_type::<model::Model>(FRAG_CC, fourCharacterCode(*b"cMOD"));
}
