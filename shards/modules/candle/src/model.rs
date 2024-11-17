use std::collections::HashMap;

use candle_transformers::models::bert::BertModel;
use candle_transformers::models::bert::DTYPE;
use shards::fourCharacterCode;
use shards::ref_counted_object_type_impl;
use shards::shard::Shard;
use shards::shlog_error;
use shards::types::common_type;
use shards::types::AutoSeqVar;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::ParamVar;
use shards::types::SeqVar;
use shards::types::TableVar;
use shards::types::BYTES_TYPES;
use shards::types::FRAG_CC;
use shards::types::{ClonedVar, Context, Type, Types, Var};

use candle_core::Device;

use crate::get_global_device;
use crate::Tensor;
use crate::TENSORS_TYPE_VEC;
use crate::TENSOR_TYPE;

pub enum Model {
  Bert(BertModel),
}

ref_counted_object_type_impl!(Model);

lazy_static! {
  pub static ref MODEL_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"cMOD")); // last letter used as version
  pub static ref MODEL_TYPE_VEC: Vec<Type> = vec![*MODEL_TYPE];
  pub static ref MODEL_VAR_TYPE: Type = Type::context_variable(&MODEL_TYPE_VEC);
}

#[derive(shards::shards_enum)]
#[enum_info(b"mMDL", "MLModels", "A machine learning model type and architecture.")]
pub enum ModelType {
  #[enum_value("A BERT model.")]
  Bert = 0x1,
}

#[derive(shards::shards_enum)]
#[enum_info(b"mFMT", "MLFormats", "The format of the machine learning model.")]
pub enum Formats {
  #[enum_value("GGUF")]
  GGUF = 0x1,
  #[enum_value("SafeTensor")]
  SafeTensor = 0x2,
}

#[derive(shards::shard)]
#[shard_info("ML.Model", "This shard allows you to load a machine learning model and specify its format and configuration.")]
pub(crate) struct ModelShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Model", "The model to use.", MODELTYPE_TYPES)]
  model: ClonedVar,

  #[shard_param("Format", "The format of the model.", FORMATS_TYPES)]
  format: ClonedVar,

  #[shard_param(
    "Configuration",
    "The configuration of the model.",
    [common_type::any_table, common_type::any_table_var]
  )]
  configuration: ParamVar,

  #[shard_param("GPU", "Whether to use the GPU (if available).", [common_type::bool])]
  gpu: ClonedVar,

  output: ClonedVar,
}

impl Default for ModelShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      model: ClonedVar::default(),
      format: ClonedVar::default(),
      configuration: ParamVar::default(),
      gpu: false.into(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for ModelShard {
  fn input_types(&mut self) -> &Types {
    &BYTES_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &MODEL_TYPE_VEC
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    self.output = ClonedVar::default();
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    if self.model.0.is_none() {
      return Err("Model is required");
    }

    if self.format.0.is_none() {
      return Err("Format is required");
    }

    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let model: ModelType = self.model.0.as_ref().try_into().unwrap();
    let format: Formats = self.format.0.as_ref().try_into().unwrap();

    let data: &[u8] = input.try_into()?;

    let model = match model {
      ModelType::Bert => match format {
        Formats::SafeTensor => {
          if self.configuration.is_none() {
            return Err("Configuration is required");
          }

          let device = if self.gpu.as_ref().try_into()? {
            get_global_device()
          } else {
            &Device::Cpu
          };

          let vb =
            candle_nn::VarBuilder::from_slice_safetensors(data, DTYPE, device).map_err(|e| {
              shlog_error!("Failed to load model: {}", e);
              "Failed to load model"
            })?;
          let config: TableVar = self.configuration.get().as_ref().try_into()?;
          let config = BertConfig::try_from(&config)?;
          let model = BertModel::load(vb, &config.0).map_err(|e| {
            shlog_error!("Failed to load model: {}", e);
            "Failed to load model"
          })?;
          Model::Bert(model)
        }
        _ => {
          return Err("Unsupported format");
        }
      },
    };

    self.output = Var::new_ref_counted(model, &*MODEL_TYPE).into();

    Ok(Some(self.output.0))
  }
}

struct BertConfig(candle_transformers::models::bert::Config);
impl TryFrom<&TableVar> for BertConfig {
  type Error = &'static str;

  fn try_from(value: &TableVar) -> Result<Self, Self::Error> {
    let mut config_map = HashMap::new();
    for (ref key, ref value) in value.iter() {
      let key: &str = key.try_into()?;
      match key {
        "vocab_size" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("vocab_size", value);
        }
        "hidden_size" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("hidden_size", value);
        }
        "num_hidden_layers" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("num_hidden_layers", value);
        }
        "num_attention_heads" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("num_attention_heads", value);
        }
        "intermediate_size" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("intermediate_size", value);
        }
        "hidden_act" => {
          let value: &str = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("hidden_act", value);
        }
        "hidden_dropout_prob" => {
          let value: f64 = value.try_into()?;
          let value = serde_json::to_value(value).map_err(|_| "Failed to convert value to f64")?;
          config_map.insert("hidden_dropout_prob", value);
        }
        "max_position_embeddings" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("max_position_embeddings", value);
        }
        "type_vocab_size" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("type_vocab_size", value);
        }
        "initializer_range" => {
          let value: f64 = value.try_into()?;
          let value = serde_json::to_value(value).map_err(|_| "Failed to convert value to f64")?;
          config_map.insert("initializer_range", value);
        }
        "layer_norm_eps" => {
          let value: f64 = value.try_into()?;
          let value = serde_json::to_value(value).map_err(|_| "Failed to convert value to f64")?;
          config_map.insert("layer_norm_eps", value);
        }
        "pad_token_id" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("pad_token_id", value);
        }
        "position_embedding_type" => {
          let value: &str = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("position_embedding_type", value);
        }
        "use_cache" => {
          let value: bool = value.try_into()?;
          let value = serde_json::to_value(value).map_err(|_| "Failed to convert value to bool")?;
          config_map.insert("use_cache", value);
        }
        "classifier_dropout" => {
          if value.is_none() {
            config_map.insert("classifier_dropout", serde_json::Value::Null);
          } else {
            let value: f64 = value.try_into()?;
            let value =
              serde_json::to_value(value).map_err(|_| "Failed to convert value to f64")?;
            config_map.insert("classifier_dropout", value);
          }
        }
        "model_type" => {
          let value: String = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to String")?;
          config_map.insert("model_type", value);
        }
        _ => {} // just ignore it
      }
    }
    let json =
      serde_json::to_string(&config_map).map_err(|_| "Failed to convert config to JSON")?;
    let config: candle_transformers::models::bert::Config =
      serde_json::from_str(&json).map_err(|_| "Failed to convert JSON to config")?;
    Ok(BertConfig(config))
  }
}

#[derive(shards::shard)]
#[shard_info("ML.Forward", "Forward a tensor through a model.")]
pub(crate) struct ForwardShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Model", "The model to use.", [*MODEL_VAR_TYPE])]
  model: ParamVar,

  outputs: AutoSeqVar,
}

impl Default for ForwardShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      model: ParamVar::default(),
      outputs: AutoSeqVar::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for ForwardShard {
  fn input_types(&mut self) -> &Types {
    &TENSORS_TYPE_VEC
  }

  fn output_types(&mut self) -> &Types {
    &TENSORS_TYPE_VEC
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    self.outputs = AutoSeqVar::new();
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    if self.model.is_none() {
      return Err("Model is required");
    }

    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let tensors: SeqVar = input.try_into()?;

    let model =
      unsafe { &mut *Var::from_ref_counted_object::<Model>(&self.model.get(), &*MODEL_TYPE)? };

    self.outputs.0.clear();

    match model {
      Model::Bert(model) => {
        if tensors.len() == 2 {
          let input_ids =
            unsafe { &mut *Var::from_ref_counted_object::<Tensor>(&tensors[0], &*TENSOR_TYPE)? };
          let input_type_ids =
            unsafe { &mut *Var::from_ref_counted_object::<Tensor>(&tensors[1], &*TENSOR_TYPE)? };
          let output = model
            .forward(&input_ids.0, &input_type_ids.0, None)
            .map_err(|e| {
              shlog_error!("Failed to forward: {}", e);
              "Failed to forward"
            })?;
          let output = Var::new_ref_counted(Tensor(output), &*TENSOR_TYPE);
          self.outputs.0.push(&output);
        } else {
          return Err("Invalid number of tensors");
        }
      }
    }

    Ok(Some(self.outputs.0 .0))
  }
}
