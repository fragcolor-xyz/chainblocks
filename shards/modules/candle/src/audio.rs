use candle_core::{Device, Tensor as CandleTensor};
use candle_transformers::models::whisper;
use shards::shard::Shard;
use shards::shardsc::SHAudio;
use shards::shlog_error;
use shards::types::{common_type, ClonedVar, Context, Type, Types, Var};
use shards::types::{ExposedTypes, InstanceData, BYTES_TYPES};

use crate::{get_global_device, Tensor, TENSOR_TYPE, TENSOR_TYPE_VEC};

const SAMPLE_RATE: usize = 16000;
const N_SAMPLES: usize = SAMPLE_RATE * 30; // 30 seconds of audio

// Include mel filter bytes
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
}

impl Default for MLAudioToMel {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      gpu: false.into(),
      num_mel_bins: 80i64.into(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for MLAudioToMel {
  fn input_types(&mut self) -> &Types {
    &BYTES_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &TENSOR_TYPE_VEC
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
    let audio: &[u8] = input
      .try_into()
      .map_err(|_| "Expected raw pcm bytes (16kHz, 1 channel) input")?;

    // Get audio samples
    let n = audio.len() / 4;
    let samples_slice = unsafe { std::slice::from_raw_parts(audio.as_ptr() as *const f32, n) };

    // Load mel filters based on num_mel_bins
    let num_mel_bins = i64::try_from(self.num_mel_bins.as_ref())?;
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

    // Create config for this chunk
    let mut config = whisper::Config {
      num_mel_bins: num_mel_bins as usize,
      max_source_positions: 1500,
      max_target_positions: 448,
      d_model: 384,
      encoder_attention_heads: 6,
      encoder_layers: 6,
      decoder_attention_heads: 6,
      decoder_layers: 6,
      suppress_tokens: vec![],
      vocab_size: 51865,
    };

    config.num_mel_bins = num_mel_bins as usize; // that's all we need here
                                                 // Convert chunk to mel spectrogram
    let mel = whisper::audio::pcm_to_mel(&config, &samples_slice, &mel_filters);

    // Calculate final dimensions
    let n_frames = mel.len() / i64::try_from(self.num_mel_bins.as_ref())? as usize;

    // Create device
    let device = if bool::try_from(self.gpu.as_ref())? {
      get_global_device()
    } else {
      &Device::Cpu
    };

    // Create final tensor
    let mel_tensor = CandleTensor::from_vec(
      mel,
      (
        1,
        i64::try_from(self.num_mel_bins.as_ref())? as usize,
        n_frames,
      ),
      device,
    )
    .map_err(|e| {
      shlog_error!("Failed to create mel spectrogram tensor: {}", e);
      "Failed to create mel spectrogram tensor"
    })?;

    // Store output
    self.output = Var::new_ref_counted(Tensor(mel_tensor), &*TENSOR_TYPE).into();
    Ok(Some(self.output.0))
  }
}
