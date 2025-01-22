use candle_core::{Device, Tensor as CandleTensor};
use shards::shard::Shard;
use shards::types::{common_type, ClonedVar, Context, Type, Types, Var, FRAG_CC};
use shards::types::{AutoSeqVar, ExposedTypes, ParamVar, SEQ_OF_INT_TYPES, STRING_TYPES};
use shards::types::{InstanceData, SeqVar};
use shards::{fourCharacterCode, ref_counted_object_type_impl, shlog_error};

use std::str::FromStr;
use tokenizers::Tokenizer as TokenizerPure;

use crate::{get_global_device, Tensor, TensorType, TENSORTYPE_TYPE, TENSOR_TYPE};

pub(crate) enum Tokenizer {
    Normal(TokenizerPure),
    Quantized(TokenizerPure),
}

impl Tokenizer {
    pub fn encode(&mut self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, &'static str> {
        match self {
            Tokenizer::Normal(tokenizer) | Tokenizer::Quantized(tokenizer) => {
                tokenizer
                    .with_padding(None)
                    .with_truncation(None)
                    .map_err(|_| "Failed to set tokenizer options")?
                    .encode(text, add_special_tokens)
                    .map_err(|_| "Failed to encode text")
                    .map(|encoded| encoded.get_ids().to_vec())
            }
        }
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String, &'static str> {
        match self {
            Tokenizer::Normal(tokenizer) | Tokenizer::Quantized(tokenizer) => {
                tokenizer.decode(tokens, skip_special_tokens)
                    .map_err(|_| "Failed to decode tokens")
            }
        }
    }

    pub fn get_token(&self, token: &str) -> Result<u32, &'static str> {
        match self {
            Tokenizer::Normal(tokenizer) | Tokenizer::Quantized(tokenizer) => {
                tokenizer.token_to_id(token)
                    .ok_or("Token not found in vocabulary")
            }
        }
    }

    pub fn get_language_token(&self, language: &str) -> Result<u32, &'static str> {
        self.get_token(&format!("<|{}|>", language))
    }

    pub fn get_sot_token(&self) -> Result<u32, &'static str> {
        self.get_token("<|startoftranscript|>")
    }

    pub fn get_translate_token(&self) -> Result<u32, &'static str> {
        self.get_token("<|translate|>")
    }

    pub fn get_transcribe_token(&self) -> Result<u32, &'static str> {
        self.get_token("<|transcribe|>")
    }

    pub fn get_notimestamps_token(&self) -> Result<u32, &'static str> {
        self.get_token("<|notimestamps|>")
    }

    #[allow(dead_code)]  // These methods may be needed for future timestamp generation
    pub fn get_timestamp_token(&self, timestamp: f32) -> Result<u32, &'static str> {
        let token = format!("<|{:.2}|>", timestamp);
        self.get_token(&token)
    }

    #[allow(dead_code)]
    pub fn get_timestamp_begin(&self) -> Result<u32, &'static str> {
        self.get_token("<|0.00|>")
    }

    #[allow(dead_code)]
    pub fn get_timestamp_end(&self) -> Result<u32, &'static str> {
        self.get_token("<|30.00|>")  // Default 30 second window
    }

    pub fn is_timestamp_token(&self, token: u32) -> bool {
        match self {
            Tokenizer::Normal(tokenizer) | Tokenizer::Quantized(tokenizer) => {
                if let Some(token_str) = tokenizer.id_to_token(token) {
                    token_str.starts_with("<|") && token_str.ends_with("|>") && 
                    token_str[2..token_str.len()-2].parse::<f32>().is_ok()
                } else {
                    false
                }
            }
        }
    }

    pub fn timestamp_from_token(&self, token: u32) -> Option<f32> {
        match self {
            Tokenizer::Normal(tokenizer) | Tokenizer::Quantized(tokenizer) => {
                if let Some(token_str) = tokenizer.id_to_token(token) {
                    if token_str.starts_with("<|") && token_str.ends_with("|>") {
                        token_str[2..token_str.len()-2].parse::<f32>().ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    pub fn decode_with_timestamps(&self, tokens: &[u32]) -> Result<Vec<(f32, String)>, &'static str> {
        let mut segments = Vec::new();
        let mut current_text = String::new();
        let mut start_time = 0.0;
        
        for &token in tokens {
            if self.is_timestamp_token(token) {
                // If we have accumulated text and find a timestamp, save the segment
                if !current_text.is_empty() {
                    if let Some(time) = self.timestamp_from_token(token) {
                        segments.push((start_time, current_text));
                        current_text = String::new();
                        start_time = time;
                    }
                } else if let Some(time) = self.timestamp_from_token(token) {
                    start_time = time;
                }
            } else {
                // Decode single token and append to current text
                if let Ok(text) = self.decode(&[token], true) {
                    current_text.push_str(&text);
                }
            }
        }

        // Add final segment if there's remaining text
        if !current_text.is_empty() {
            segments.push((start_time, current_text));
        }

        Ok(segments)
    }

    pub fn prepare_audio_tokens(&mut self, language: Option<&str>, task: Option<&str>, timestamps: bool) -> Result<Vec<u32>, &'static str> {
        let mut tokens = vec![self.get_sot_token()?];
        
        // Add language token if specified
        if let Some(lang) = language {
            tokens.push(self.get_language_token(lang)?);
        }

        // Add task token if specified
        if let Some(task) = task {
            match task {
                "translate" => tokens.push(self.get_translate_token()?),
                "transcribe" => tokens.push(self.get_transcribe_token()?),
                _ => return Err("Invalid task type"),
            }
        }

        // Add notimestamps token if timestamps are disabled
        if !timestamps {
            tokens.push(self.get_notimestamps_token()?);
        }

        Ok(tokens)
    }
}

ref_counted_object_type_impl!(Tokenizer);

lazy_static! {
  pub static ref TOKENIZER_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"TOKn")); // last letter used as version
  pub static ref TOKENIZER_TYPE_VEC: Vec<Type> = vec![*TOKENIZER_TYPE];
  pub static ref TOKENIZER_VAR_TYPE: Type = Type::context_variable(&TOKENIZER_TYPE_VEC);
  pub static ref INTS_OR_TENSOR_TYPES: Vec<Type> = vec![SEQ_OF_INT_TYPES[0], *TENSOR_TYPE];
}

#[derive(shards::shard)]
#[shard_info(
  "ML.Tokenizer",
  "Loads a tokenizer from an input json string, ready to be used for tokenizing text."
)]
pub(crate) struct MLTokenizer {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Quantized", "Whether to use quantized tokenizer.", [common_type::bool])]
  quantized: ClonedVar,

  output: ClonedVar,
}

impl Default for MLTokenizer {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      quantized: false.into(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for MLTokenizer {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &TOKENIZER_TYPE_VEC
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
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let input_str: &str = input.try_into()?;
    let tokenizer = TokenizerPure::from_str(input_str).map_err(|e| {
      shlog_error!("Failed to create tokenizer: {:?}", e);
      "Failed to create tokenizer"
    })?;

    let quantized: bool = self.quantized.as_ref().try_into()?;
    let tokenizer = if quantized {
        Tokenizer::Quantized(tokenizer)
    } else {
        Tokenizer::Normal(tokenizer)
    };

    self.output = Var::new_ref_counted(tokenizer, &*TOKENIZER_TYPE).into();
    Ok(Some(self.output.0))
  }
}

#[derive(shards::shard)]
#[shard_info("ML.Tokens", "Tokenizes text using a tokenizer.")]
pub(crate) struct TokensShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Tokenizer", "The tokenizer to use.", [*TOKENIZER_VAR_TYPE])]
  tokenizer: ParamVar,

  #[shard_param("AddSpecialTokens", "If true, add special tokens.", [common_type::bool])]
  add_special_tokens: ClonedVar,

  #[shard_param("AsTensor", "Outputs a tensor object instead of an int sequence.", [common_type::bool])]
  as_tensor: ClonedVar,

  #[shard_param("Format", "The format of the output tensor. If As Tensor is true.", [*TENSORTYPE_TYPE])]
  format: ClonedVar,

  #[shard_param("GPU", "If true, the output tensor will be on the GPU (if ).", [common_type::bool])]
  gpu: ClonedVar,

  output_seq: AutoSeqVar,
  output_tensor: ClonedVar,
}

impl Default for TokensShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      tokenizer: ParamVar::new(Var::default()),
      add_special_tokens: true.into(),
      as_tensor: false.into(),
      output_seq: AutoSeqVar::new(),
      output_tensor: ClonedVar::default(),
      format: TensorType::U32.into(),
      gpu: false.into(),
    }
  }
}

#[shards::shard_impl]
impl Shard for TokensShard {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &SEQ_OF_INT_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    self.output_seq = AutoSeqVar::new();
    self.output_tensor = ClonedVar::default();
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    if self.tokenizer.is_none() {
      return Err("Tokenizer is not set");
    }

    let as_tensor = self.as_tensor.as_ref().try_into()?;
    if as_tensor {
      Ok(*TENSOR_TYPE)
    } else {
      Ok(SEQ_OF_INT_TYPES[0])
    }
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let input_str: &str = input.try_into()?;

    let tokenizer = unsafe {
      &mut *Var::from_ref_counted_object::<Tokenizer>(&self.tokenizer.get(), &*TOKENIZER_TYPE)?
    };

    let add_special_tokens: bool = self.add_special_tokens.as_ref().try_into()?;
    let tokens = tokenizer.encode(input_str, add_special_tokens).map_err(|e| {
      shlog_error!("Failed to tokenize text: {}", e);
      "Failed to tokenize text"
    })?;

    let as_tensor: bool = self.as_tensor.as_ref().try_into()?;
    if as_tensor {
      let format: TensorType = self.format.0.as_ref().try_into()?;
      let device = if self.gpu.as_ref().try_into()? {
        get_global_device()
      } else {
        &Device::Cpu
      };
      let token_ids = match format {
        TensorType::U32 => CandleTensor::new(&tokens[..], device)
          .map_err(|e| {
            shlog_error!("Failed to create tensor: {:?}", e);
            "Failed to create tensor"
          })?
          .unsqueeze(0)
          .map_err(|e| {
            shlog_error!("Failed to unsqueeze tensor: {:?}", e);
            "Failed to unsqueeze tensor"
          }),
        TensorType::I64 => {
          let tokens: Vec<i64> = tokens.into_iter().map(|token| token as i64).collect();
          CandleTensor::new(&tokens[..], device)
            .map_err(|e| {
              shlog_error!("Failed to create tensor: {:?}", e);
              "Failed to create tensor"
            })?
            .unsqueeze(0)
            .map_err(|e| {
              shlog_error!("Failed to unsqueeze tensor: {:?}", e);
              "Failed to unsqueeze tensor"
            })
        }
        _ => Err("Invalid format"),
      }?;
      self.output_tensor = Var::new_ref_counted(Tensor(token_ids), &*TENSOR_TYPE).into();
      Ok(Some(self.output_tensor.0))
    } else {
      self.output_seq.0.clear();
      for token in tokens {
        let token: Var = token.into();
        self.output_seq.0.push(&token);
      }
      Ok(Some(self.output_seq.0 .0))
    }
  }
}

#[derive(shards::shard)]
#[shard_info(
  "ML.Detokenize",
  "Converts token IDs or tensors back into text using a tokenizer."
)]
pub(crate) struct MLDetokenizer {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Tokenizer", "The tokenizer to use for detokenization.", [*TOKENIZER_VAR_TYPE])]
  tokenizer: ParamVar,

  #[shard_param("SkipSpecialTokens", "If true, skip special tokens during detokenization.", [common_type::bool])]
  skip_special_tokens: ClonedVar,

  output: ClonedVar,
}

impl Default for MLDetokenizer {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      tokenizer: ParamVar::new(Var::default()),
      skip_special_tokens: true.into(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for MLDetokenizer {
  fn input_types(&mut self) -> &Types {
    &INTS_OR_TENSOR_TYPES
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

    if self.tokenizer.is_none() {
      return Err("Tokenizer is not set");
    }

    Ok(STRING_TYPES[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let tokenizer = unsafe {
      &mut *Var::from_ref_counted_object::<Tokenizer>(&self.tokenizer.get(), &*TOKENIZER_TYPE)?
    };

    let skip_special_tokens: bool = self.skip_special_tokens.as_ref().try_into()?;

    let token_ids: Vec<u32> = if input.is_seq() {
      let input_seq: SeqVar = input.try_into()?;
      input_seq
        .iter()
        .map(|token| u32::try_from(&token).unwrap())
        .collect()
    } else {
      let tensor = unsafe { &*Var::from_ref_counted_object::<Tensor>(input, &*TENSOR_TYPE)? };
      tensor
        .0
        .to_dtype(candle_core::DType::U32)
        .and_then(|tensor| tensor.flatten_all()?.to_vec1())
        .map_err(|e| {
          shlog_error!("Failed to convert tensor to vector: {:?}", e);
          "Failed to convert tensor to vector"
        })?
    };

    let decoded_text = tokenizer.decode(&token_ids, skip_special_tokens).map_err(|e| {
      shlog_error!("Failed to detokenize text: {:?}", e);
      "Failed to detokenize text"
    })?;

    self.output = decoded_text.into();
    Ok(Some(self.output.0))
  }
}

#[derive(shards::shard)]
#[shard_info(
  "ML.PrepareAudioTokens",
  "Prepares the initial tokens for Whisper audio processing with language and task settings."
)]
pub(crate) struct PrepareAudioTokensShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Tokenizer", "The tokenizer to use.", [*TOKENIZER_VAR_TYPE])]
  tokenizer: ParamVar,

  #[shard_param("Language", "The language code (e.g. 'en', 'fr', etc).", [common_type::string])]
  language: ParamVar,

  #[shard_param("Task", "The task type ('transcribe' or 'translate').", [common_type::string])]
  task: ParamVar,

  #[shard_param("Timestamps", "Whether to include timestamp tokens.", [common_type::bool])]
  timestamps: ClonedVar,

  #[shard_param("AsTensor", "Output as tensor instead of sequence.", [common_type::bool])]
  as_tensor: ClonedVar,

  #[shard_param("GPU", "If true, the output tensor will be on GPU.", [common_type::bool])]
  gpu: ClonedVar,

  output_seq: AutoSeqVar,
  output_tensor: ClonedVar,
}

impl Default for PrepareAudioTokensShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      tokenizer: ParamVar::default(),
      language: ParamVar::default(),
      task: ParamVar::default(),
      timestamps: true.into(),
      as_tensor: false.into(),
      gpu: false.into(),
      output_seq: AutoSeqVar::new(),
      output_tensor: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for PrepareAudioTokensShard {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &INTS_OR_TENSOR_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    self.output_seq = AutoSeqVar::new();
    self.output_tensor = ClonedVar::default();
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    if self.tokenizer.is_none() {
      return Err("Tokenizer is required");
    }

    let as_tensor = self.as_tensor.as_ref().try_into()?;
    if as_tensor {
      Ok(*TENSOR_TYPE)
    } else {
      Ok(SEQ_OF_INT_TYPES[0])
    }
  }

  fn activate(&mut self, _context: &Context, _input: &Var) -> Result<Option<Var>, &str> {
    let tokenizer = unsafe {
      &mut *Var::from_ref_counted_object::<Tokenizer>(&self.tokenizer.get(), &*TOKENIZER_TYPE)?
    };

    let language = if self.language.is_none() {
      None
    } else {
      Some(self.language.get().as_ref().try_into()?)
    };

    let task = if self.task.is_none() {
      None
    } else {
      Some(self.task.get().as_ref().try_into()?)
    };

    let timestamps: bool = self.timestamps.as_ref().try_into()?;

    let tokens = tokenizer.prepare_audio_tokens(language, task, timestamps)?;

    let as_tensor: bool = self.as_tensor.as_ref().try_into()?;
    if as_tensor {
      let device = if self.gpu.as_ref().try_into()? {
        get_global_device()
      } else {
        &Device::Cpu
      };

      let token_tensor = CandleTensor::new(&tokens[..], device)
        .map_err(|e| {
          shlog_error!("Failed to create tensor: {:?}", e);
          "Failed to create tensor"
        })?
        .unsqueeze(0)
        .map_err(|e| {
          shlog_error!("Failed to unsqueeze tensor: {:?}", e);
          "Failed to unsqueeze tensor"
        })?;

      self.output_tensor = Var::new_ref_counted(Tensor(token_tensor), &*TENSOR_TYPE).into();
      Ok(Some(self.output_tensor.0))
    } else {
      self.output_seq.0.clear();
      for token in tokens {
        let token: Var = token.into();
        self.output_seq.0.push(&token);
      }
      Ok(Some(self.output_seq.0.0))
    }
  }
}

#[derive(shards::shard)]
#[shard_info(
  "ML.DetokenizeWithTimestamps",
  "Converts token IDs back into text segments with timestamps using a tokenizer."
)]
pub(crate) struct MLTimestampDetokenizer {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Tokenizer", "The tokenizer to use for detokenization.", [*TOKENIZER_VAR_TYPE])]
  tokenizer: ParamVar,

  output: ClonedVar,
}

impl Default for MLTimestampDetokenizer {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      tokenizer: ParamVar::default(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for MLTimestampDetokenizer {
  fn input_types(&mut self) -> &Types {
    &INTS_OR_TENSOR_TYPES
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

    if self.tokenizer.is_none() {
      return Err("Tokenizer is required");
    }

    Ok(STRING_TYPES[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let tokenizer = unsafe {
      &mut *Var::from_ref_counted_object::<Tokenizer>(&self.tokenizer.get(), &*TOKENIZER_TYPE)?
    };

    let token_ids: Vec<u32> = if input.is_seq() {
      let input_seq: SeqVar = input.try_into()?;
      input_seq
        .iter()
        .map(|token| u32::try_from(&token).unwrap())
        .collect()
    } else {
      let tensor = unsafe { &*Var::from_ref_counted_object::<Tensor>(input, &*TENSOR_TYPE)? };
      tensor
        .0
        .to_dtype(candle_core::DType::U32)
        .and_then(|tensor| tensor.flatten_all()?.to_vec1())
        .map_err(|e| {
          shlog_error!("Failed to convert tensor to vector: {:?}", e);
          "Failed to convert tensor to vector"
        })?
    };

    let segments = tokenizer.decode_with_timestamps(&token_ids).map_err(|e| {
      shlog_error!("Failed to decode with timestamps: {}", e);
      "Failed to decode with timestamps"
    })?;

    // Format segments as JSON string
    let segments_json = serde_json::to_string(&segments).map_err(|e| {
      shlog_error!("Failed to serialize segments: {}", e);
      "Failed to serialize segments"
    })?;

    self.output = segments_json.into();
    Ok(Some(self.output.0))
  }
}
