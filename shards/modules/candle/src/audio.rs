use candle_core::{Device, Tensor as CandleTensor};
use candle_transformers::models::whisper;
use shards::shard::Shard;
use shards::shardsc::SHAudio;
use shards::shlog_error;
use shards::types::{common_type, ClonedVar, Context, Type, Types, Var};
use shards::types::{ExposedTypes, InstanceData, BYTES_TYPES};

use crate::{get_global_device, Tensor, TENSOR_TYPE};

const SAMPLE_RATE: u32 = 16000;
const CHUNK_LENGTH: usize = 30;
const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE as usize;

// Include the mel filters data
const MEL_FILTERS_80: &[u8] = include_bytes!("melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("melfilters128.bytes");

#[derive(shards::shard)]
#[shard_info(
  "ML.AudioToMel",
  "Converts raw audio bytes to mel spectrograms using Whisper's mel filters."
)]
pub(crate) struct MLAudioToMel {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("GPU", "Whether to use GPU for processing.", [common_type::bool])]
  gpu: ClonedVar,

  #[shard_param("NumMelBins", "Number of mel bins (80 or 128).", [common_type::int])]
  num_mel_bins: ClonedVar,

  output: ClonedVar,
  output_types: Types,
}

impl Default for MLAudioToMel {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      gpu: false.into(),
      num_mel_bins: 80i64.into(),
      output: ClonedVar::default(),
      output_types: vec![*TENSOR_TYPE],
    }
  }
}

#[shards::shard_impl]
impl Shard for MLAudioToMel {
  fn input_types(&mut self) -> &Types {
    &BYTES_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &self.output_types
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
    let audio: SHAudio = input.try_into().map_err(|_| "Expected SHAudio input")?;

    // Pad or truncate to N_SAMPLES
    let mut pcm = vec![0f32; N_SAMPLES];
    let n = audio.nsamples as usize;
    if n > 0 {
      let samples_slice = unsafe { std::slice::from_raw_parts(audio.samples, n) };
      let copy_len = n.min(N_SAMPLES);
      pcm[..copy_len].copy_from_slice(&samples_slice[..copy_len]);
    }

    // Check sample rate
    if audio.sampleRate != SAMPLE_RATE {
      return Err("input audio must have a 16kHz sampling rate");
    }

    let device = if self.gpu.as_ref().try_into()? {
      get_global_device()
    } else {
      &Device::Cpu
    };

    // Load mel filters based on num_mel_bins
    let num_mel_bins: i64 = self.num_mel_bins.as_ref().try_into()?;
    let mel_bytes = match num_mel_bins {
      80 => MEL_FILTERS_80,
      128 => MEL_FILTERS_128,
      _ => return Err("num_mel_bins must be either 80 or 128"),
    };

    // Convert bytes to f32 mel filters
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    for i in 0..mel_filters.len() {
      let bytes = &mel_bytes[i * 4..(i + 1) * 4];
      mel_filters[i] = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }

    // Convert to mel spectrogram
    let config = whisper::Config {
      d_model: 384,
      encoder_attention_heads: 6,
      encoder_layers: 6,
      decoder_attention_heads: 6,
      decoder_layers: 6,
      suppress_tokens: vec![],
      num_mel_bins: num_mel_bins as usize,
      max_source_positions: 1500,
      max_target_positions: 448,
      vocab_size: 51865,
    };

    let mel = whisper::audio::pcm_to_mel(&config, &pcm, &mel_filters);
    let mel_len = mel.len();

    let mel = CandleTensor::from_vec(
      mel,
      (1, num_mel_bins as usize, mel_len / num_mel_bins as usize),
      device,
    )
    .map_err(|e| {
      shlog_error!("Failed to create mel spectrogram tensor: {}", e);
      "Failed to create mel spectrogram tensor"
    })?;

    self.output = Var::new_ref_counted(Tensor(mel), &*TENSOR_TYPE).into();
    Ok(Some(self.output.0))
  }
}
