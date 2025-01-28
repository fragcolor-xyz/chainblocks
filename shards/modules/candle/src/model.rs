use std::collections::HashMap;
use std::io::Write;

use candle_transformers::models::bert::BertModel;
use candle_transformers::models::bert::DTYPE;
use candle_transformers::models::whisper::{self as Whisper, Config as WhisperConfigType};
use candle_transformers::quantized_var_builder;
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
use shards::types::STRING_TYPES;
use shards::types::{ClonedVar, Context, Type, Types, Var};

use candle_core::{Device, Tensor as CandleTensor};

use crate::get_global_device;
use crate::tokenizer::Tokenizer;
use crate::tokenizer::TOKENIZER_TYPE;
use crate::tokenizer::TOKENIZER_VAR_TYPE;
use crate::Tensor;
use crate::TENSORS_TYPE_VEC;
use crate::TENSOR_TYPE;
use crate::TENSOR_TYPE_VEC;

pub enum Model {
  Bert(BertModel),
  Whisper(Whisper::model::Whisper),
  WhisperQuantized(Whisper::quantized_model::Whisper),
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
  #[enum_value("A Whisper speech recognition model.")]
  Whisper = 0x2,
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

    let model = match (model, format) {
      (ModelType::Bert, Formats::SafeTensor) => {
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
      (ModelType::Whisper, Formats::SafeTensor) => {
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
        let config = WhisperConfig::try_from(&config)?;
        let model = Whisper::model::Whisper::load(&vb, config.0).map_err(|e| {
          shlog_error!("Failed to load model: {}", e);
          "Failed to load model"
        })?;
        Model::Whisper(model)
      }
      (ModelType::Whisper, Formats::GGUF) => {
        if self.configuration.is_none() {
          return Err("Configuration is required");
        }

        let device = if self.gpu.as_ref().try_into()? {
          get_global_device()
        } else {
          &Device::Cpu
        };

        // Create a temporary file with unique name in the current directory
        let temp_path = std::path::PathBuf::from(format!("temp_model_{}.gguf", std::process::id()));
        let mut temp_file = std::fs::File::create(&temp_path).map_err(|e| {
          shlog_error!("Failed to create temporary file: {}", e);
          "Failed to create temporary file"
        })?;

        // Write the model data to the temporary file
        let write_result = temp_file.write_all(data).map_err(|e| {
          // Try to clean up the file even if write failed
          let _ = std::fs::remove_file(&temp_path);
          shlog_error!("Failed to write model data: {}", e);
          "Failed to write model data"
        })?;

        // Drop the file handle to ensure it's properly written
        drop(temp_file);

        // Use the temporary file to load the model
        let load_result = quantized_var_builder::VarBuilder::from_gguf(&temp_path, device).map_err(|e| {
          // Clean up the file if loading fails
          let _ = std::fs::remove_file(&temp_path);
          shlog_error!("Failed to load model: {}", e);
          "Failed to load model"
        });

        // Clean up the temporary file now that we're done with it
        if let Err(e) = std::fs::remove_file(&temp_path) {
          shlog_error!("Failed to remove temporary file: {}", e);
          // Continue anyway since the model is loaded
        }

        // Now handle the result of loading
        let vb = load_result?;
        let config: TableVar = self.configuration.get().as_ref().try_into()?;
        let config = WhisperConfig::try_from(&config)?;
        let model = Whisper::quantized_model::Whisper::load(&vb, config.0).map_err(|e| {
          shlog_error!("Failed to load model: {}", e);
          "Failed to load model"
        })?;
        Model::WhisperQuantized(model)
      }
      _ => return Err("Unsupported model/format combination"),
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

struct WhisperConfig(WhisperConfigType);

impl TryFrom<&TableVar> for WhisperConfig {
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
        "num_mel_bins" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("num_mel_bins", value);
        }
        "max_source_positions" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("max_source_positions", value);
        }
        "max_target_positions" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("max_target_positions", value);
        }
        "encoder_attention_heads" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("encoder_attention_heads", value);
        }
        "encoder_layers" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("encoder_layers", value);
        }
        "decoder_attention_heads" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("decoder_attention_heads", value);
        }
        "decoder_layers" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("decoder_layers", value);
        }
        "encoder_ffn_dim" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("encoder_ffn_dim", value);
        }
        "decoder_ffn_dim" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("decoder_ffn_dim", value);
        }
        "d_model" => {
          let value: usize = value.try_into()?;
          let value =
            serde_json::to_value(value).map_err(|_| "Failed to convert value to usize")?;
          config_map.insert("d_model", value);
        }
        _ => {} // ignore unknown keys
      }
    }
    let json =
      serde_json::to_string(&config_map).map_err(|_| "Failed to convert config to JSON")?;
    let config: WhisperConfigType =
      serde_json::from_str(&json).map_err(|_| "Failed to convert JSON to config")?;
    Ok(WhisperConfig(config))
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
      Model::Whisper(model) => {
        if tensors.len() != 1 {
          return Err("Whisper expects a single mel spectrogram tensor");
        }

        let mel =
          unsafe { &mut *Var::from_ref_counted_object::<Tensor>(&tensors[0], &*TENSOR_TYPE)? };

        let encoder_output = model.encoder.forward(&mel.0, true).map_err(|e| {
          shlog_error!("Failed to encode: {}", e);
          "Failed to encode audio"
        })?;

        let output = Var::new_ref_counted(Tensor(encoder_output), &*TENSOR_TYPE);
        self.outputs.0.push(&output);
      }
      Model::WhisperQuantized(model) => {
        if tensors.len() != 1 {
          return Err("Whisper expects a single mel spectrogram tensor");
        }

        let mel =
          unsafe { &mut *Var::from_ref_counted_object::<Tensor>(&tensors[0], &*TENSOR_TYPE)? };

        let encoder_output = model.encoder.forward(&mel.0, true).map_err(|e| {
          shlog_error!("Failed to encode: {}", e);
          "Failed to encode audio"
        })?;

        let output = Var::new_ref_counted(Tensor(encoder_output), &*TENSOR_TYPE);
        self.outputs.0.push(&output);
      }
    }

    Ok(Some(self.outputs.0 .0))
  }
}

trait WhisperExt {
  fn detect_language(
    &mut self,
    mel: &CandleTensor,
    tokenizer: &mut Tokenizer,
  ) -> Result<u32, candle_core::Error>;
  fn decode(
    &mut self,
    mel: &CandleTensor,
    tokenizer: &mut Tokenizer,
    initial_tokens: &[u32],
    _beam_size: usize,
  ) -> Result<WhisperDecodeOutput, candle_core::Error>;
}

struct WhisperDecodeOutput {
  text: String,
  tokens: Vec<u32>,
}

impl WhisperExt for Whisper::model::Whisper {
  fn detect_language(
    &mut self,
    mel: &CandleTensor,
    tokenizer: &mut Tokenizer,
  ) -> Result<u32, candle_core::Error> {
    // Ensure mel is f32
    let mel = mel.to_dtype(candle_core::DType::F32)?;
    
    // Get encoder output first
    let audio_features = self.encoder.forward(&mel, true)?;
    
    // Create token tensor directly as i64
    let sot_token = tokenizer.get_sot_token().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let tokens_i64 = vec![sot_token as i64];
    let tokens_t = CandleTensor::from_slice(&tokens_i64, (1, 1), mel.device())?;
    
    // Run decoder with first iteration flag
    let ys = self.decoder.forward(&tokens_t, &audio_features, true)?;
    let logits = self.decoder.final_linear(&ys)?.squeeze(0)?;
    
    logits.argmax(0)?.to_scalar::<u32>()
  }

  fn decode(
    &mut self,
    mel: &CandleTensor,
    tokenizer: &mut Tokenizer,
    initial_tokens: &[u32],
    _beam_size: usize,
  ) -> Result<WhisperDecodeOutput, candle_core::Error> {
    // Ensure mel is f32
    let mel = mel.to_dtype(candle_core::DType::F32)?;
    
    // Get encoder output first
    let audio_features = self.encoder.forward(&mel, true)?;
    
    let mut tokens = initial_tokens.to_vec();
    let sample_len = self.config.max_target_positions / 2;
    
    for i in 0..sample_len {
      // Create tokens tensor directly
      let tokens_i64: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
      let tokens_t = CandleTensor::from_slice(&tokens_i64, (1, tokens_i64.len()), mel.device())?;
      
      // Forward pass with first iteration flag
      let ys = self.decoder.forward(&tokens_t, &audio_features, i == 0)?;
      
      // Get logits for next token prediction
      let (_, seq_len, _) = ys.dims3()?;
      let logits = self.decoder.final_linear(&ys.narrow(1, seq_len - 1, 1)?)?.squeeze(1)?;
      
      // Get next token using argmax
      let next_token = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;
      
      if next_token == tokenizer.get_token("<|endoftext|>").map_err(|e| candle_core::Error::Msg(e.to_string()))? {
        break;
      }
      tokens.push(next_token);
      if tokens.len() > 448 { // Max length
        break;
      }
    }
    
    let text = tokenizer.decode(&tokens, true).unwrap_or_default();
    Ok(WhisperDecodeOutput { text, tokens })
  }
}

impl WhisperExt for Whisper::quantized_model::Whisper {
  fn detect_language(
    &mut self,
    mel: &CandleTensor,
    tokenizer: &mut Tokenizer,
  ) -> Result<u32, candle_core::Error> {
    // Ensure mel is f32
    let mel = mel.to_dtype(candle_core::DType::F32)?;
    
    // Get encoder output first
    let audio_features = self.encoder.forward(&mel, true)?;
    
    // Create token tensor directly as i64
    let sot_token = tokenizer.get_sot_token().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let tokens_i64 = vec![sot_token as i64];
    let tokens_t = CandleTensor::from_slice(&tokens_i64, (1, 1), mel.device())?;
    
    // Run decoder with first iteration flag
    let ys = self.decoder.forward(&tokens_t, &audio_features, true)?;
    let logits = self.decoder.final_linear(&ys)?.squeeze(0)?;
    
    logits.argmax(0)?.to_scalar::<u32>()
  }

  fn decode(
    &mut self,
    mel: &CandleTensor,
    tokenizer: &mut Tokenizer,
    initial_tokens: &[u32],
    _beam_size: usize,
  ) -> Result<WhisperDecodeOutput, candle_core::Error> {
    // Ensure mel is f32
    let mel = mel.to_dtype(candle_core::DType::F32)?;
    
    // Get encoder output first
    let audio_features = self.encoder.forward(&mel, true)?;
    
    let mut tokens = initial_tokens.to_vec();
    let sample_len = self.config.max_target_positions / 2;
    
    for i in 0..sample_len {
      // Create tokens tensor directly
      let tokens_i64: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
      let tokens_t = CandleTensor::from_slice(&tokens_i64, (1, tokens_i64.len()), mel.device())?;
      
      // Forward pass with first iteration flag
      let ys = self.decoder.forward(&tokens_t, &audio_features, i == 0)?;
      
      // Get logits for next token prediction
      let (_, seq_len, _) = ys.dims3()?;
      let logits = self.decoder.final_linear(&ys.narrow(1, seq_len - 1, 1)?)?.squeeze(1)?;
      
      // Get next token using argmax
      let next_token = logits.argmax(1)?.squeeze(0)?.to_scalar::<u32>()?;
      
      if next_token == tokenizer.get_token("<|endoftext|>").map_err(|e| candle_core::Error::Msg(e.to_string()))? {
        break;
      }
      tokens.push(next_token);
      if tokens.len() > 448 { // Max length
        break;
      }
    }
    
    let text = tokenizer.decode(&tokens, true).unwrap_or_default();
    Ok(WhisperDecodeOutput { text, tokens })
  }
}

#[derive(shards::shard)]
#[shard_info(
    "ML.SpeechToText",
    "Complete speech-to-text pipeline using Whisper model. Takes a MEL spectrogram tensor as input and outputs transcribed text."
)]
pub(crate) struct SpeechToTextShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Model", "The Whisper model to use.", [*MODEL_VAR_TYPE])]
  model: ParamVar,

  #[shard_param("Tokenizer", "The tokenizer to use.", [*TOKENIZER_VAR_TYPE])]
  tokenizer: ParamVar,

  #[shard_param("Language", "Optional language code (e.g. 'en', 'fr'). If not specified, language will be auto-detected.", [common_type::string])]
  language: ParamVar,

  #[shard_param("Task", "The task type ('transcribe' or 'translate').", [common_type::string])]
  task: ParamVar,

  #[shard_param("Timestamps", "Whether to include timestamps in output.", [common_type::bool])]
  timestamps: ClonedVar,

  #[shard_param("BeamSize", "Beam size for decoding (default: 5).", [common_type::int])]
  beam_size: ClonedVar,

  output: ClonedVar,
}

impl Default for SpeechToTextShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      model: ParamVar::default(),
      tokenizer: ParamVar::default(),
      language: ParamVar::default(),
      task: ParamVar::default(),
      timestamps: false.into(),
      beam_size: 5i64.into(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for SpeechToTextShard {
  fn input_types(&mut self) -> &Types {
    &TENSOR_TYPE_VEC
  }

  fn output_types(&mut self) -> &Types {
    &STRING_TYPES
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

    if self.model.is_none() {
      return Err("Model is required");
    }
    if self.tokenizer.is_none() {
      return Err("Tokenizer is required");
    }

    Ok(STRING_TYPES[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let model_var = self.model.get();
    let model = unsafe { &mut *Var::from_ref_counted_object::<Model>(&model_var, &*MODEL_TYPE)? };

    let tokenizer_var = self.tokenizer.get();
    let tokenizer = unsafe { &mut *Var::from_ref_counted_object::<crate::tokenizer::Tokenizer>(&tokenizer_var, &*TOKENIZER_TYPE)? };

    let mel_tensor = unsafe { &*Var::from_ref_counted_object::<Tensor>(input, &*TENSOR_TYPE)? };

    // Convert to f32 before processing
    let mel_tensor = mel_tensor.0.to_dtype(candle_core::DType::F32).map_err(|e| {
      shlog_error!("Failed to convert tensor to f32: {}", e);
      "Failed to convert tensor to f32"
    })?;

    // Get language token if specified
    let language_token = if !self.language.get().is_none() {
      let lang: &str = self.language.get().as_ref().try_into()?;
      Some(tokenizer.get_language_token(lang)?)
    } else {
      None
    };

    // Get task type if specified
    let task = if !self.task.get().is_none() {
      let task_str: &str = self.task.get().as_ref().try_into()?;
      match task_str {
        "translate" => Some(crate::whisper::Task::Translate),
        "transcribe" => Some(crate::whisper::Task::Transcribe),
        _ => return Err("Invalid task type"),
      }
    } else {
      None
    };

    let timestamps: bool = self.timestamps.as_ref().try_into()?;

    // Extract the underlying TokenizerPure from our Tokenizer enum
    let tokenizer_pure = match tokenizer {
      crate::tokenizer::Tokenizer::Normal(t) | crate::tokenizer::Tokenizer::Quantized(t) => t,
    };

    // Create decoder
    let mut decoder = crate::whisper::Decoder::new(
      match model {
        Model::Whisper(m) => crate::whisper::Model::Normal(m.clone()),
        Model::WhisperQuantized(m) => crate::whisper::Model::Quantized(m.clone()),
        _ => return Err("Model must be a Whisper model"),
      },
      tokenizer_pure.clone(),
      299792458, // Fixed seed
      mel_tensor.device(),
      language_token,
      task,
      timestamps,
      false, // verbose
    ).map_err(|e| {
      shlog_error!("Failed to create decoder: {}", e);
      "Failed to create decoder"
    })?;

    // Run the decoder
    let segments = decoder.run(&mel_tensor).map_err(|e| {
      shlog_error!("Failed to run decoder: {}", e);
      "Failed to run decoder"
    })?;

    // Collect all text segments
    let mut final_text = String::new();
    for segment in segments {
      if !final_text.is_empty() {
        final_text.push(' ');
      }
      final_text.push_str(&segment.dr.text);
    }

    self.output = final_text.into();
    Ok(Some(self.output.0))
  }
}
