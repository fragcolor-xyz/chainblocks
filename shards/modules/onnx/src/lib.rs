#[macro_use]
extern crate shards;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate compile_time_crc32;

use ort::{
  execution_providers::{CoreMLExecutionProvider, ExecutionProviderDispatch},
  session::{
    builder::{GraphOptimizationLevel, SessionBuilder},
    Session,
  },
  value::Value,
};
use ndarray::Array;
use shards::shardsc::SHObjectTypeInfo;
use shards::types::OptionalString;
use shards::types::{
  ExposedInfo, ExposedTypes, ParamVar, Seq, NONE_TYPES, SEQ_OF_FLOAT_TYPES, SEQ_OF_INT_TYPES,
  STRING_TYPES,
};
use shards::{
  core::register_legacy_shard,
  shard::LegacyShard,
  types::{common_type, ClonedVar, Context, Parameters, Type, Types, Var, FRAG_CC},
};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

lazy_static! {
    static ref MODEL_TYPE: Type = Type::object(
        FRAG_CC,
        0x6f6e6e78 // 'onnx'
    );
    static ref MODEL_TYPE_VEC: Vec<Type> = vec![*MODEL_TYPE];
    static ref MODEL_VAR_TYPE: Type = Type::context_variable(&MODEL_TYPE_VEC);
    static ref MODEL_TYPE_VEC_VAR: Vec<Type> = vec![*MODEL_VAR_TYPE];
    static ref LOAD_PARAMETERS: Parameters = vec![
        (
            cstr!("Path"),
            shccstr!("The path to the onnx model to load."),
            &STRING_TYPES[..]
        )
            .into(),
        (
            cstr!("InputShape"),
            shccstr!("The shape of the input tensor."),
            &SEQ_OF_INT_TYPES[..]
        )
            .into(),
    ];
    static ref ACTIVATE_PARAMETERS: Parameters = vec![
        (
            cstr!("Model"),
            shccstr!("The ONNX model to use to perform the activation."),
            &MODEL_TYPE_VEC_VAR[..]
        )
            .into(),
    ];
}
struct ModelWrapper {
  session: Session,
  shape: Vec<i64>,
}

#[derive(Default)]
struct Load {
  model: Option<Rc<ModelWrapper>>,
  path: ClonedVar,
  shape: ClonedVar,
}

impl LegacyShard for Load {
  fn registerName() -> &'static str {
    cstr!("ONNX.Load")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("ONNX.Load-rust-0x20230101")
  }

  fn name(&mut self) -> &str {
    "ONNX.Load"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!("This shard creates a loaded ONNX model from the file specified in the Path parameter. The model will also expect the input shape specified in the InputShape parameter. Uses CUDA for GPU acceleration when available."))
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!("The input of this shard is ignored."))
  }

  fn outputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!("Returns the ONNX model object."))
  }

  fn inputTypes(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &MODEL_TYPE_VEC
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&LOAD_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.path = value.into()),
      1 => Ok(self.shape = value.into()),
      _ => unreachable!(),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.path.0,
      1 => self.shape.0,
      _ => unreachable!(),
    }
  }

  fn activate(&mut self, _context: &Context, _input: &Var) -> Result<Option<Var>, &str> {
    let path: &str = self.path.0.as_ref().try_into()?;

    // Convert shape from sequence to Vec<i64>
    let shape = self.shape.0.as_seq()?;
    let shape: Vec<i64> = shape
      .iter()
      .map(|v| {
        v.as_ref()
          .try_into()
          .expect("Shards validation should prevent this")
      })
      .collect();

    // Create session with platform-specific acceleration
    let mut execution_providers = Vec::new();

    // Try to add available execution providers in order of preference
    #[cfg(target_os = "macos")]
    execution_providers.push(CoreMLExecutionProvider::default().with_subgraphs().build());

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    execution_providers.push(DirectMLExecutionProvider::default().build());

    // Create optimized session
    let session = SessionBuilder::new()
      .map_err(|e| {
        shlog!("Error: {}", e);
        "Failed to create session builder"
      })?
      .with_optimization_level(GraphOptimizationLevel::Level3)
      .map_err(|e| {
        shlog!("Error: {}", e);
        "Failed to set optimization level"
      })?
      .with_execution_providers(execution_providers)
      .map_err(|e| {
        shlog!("Error: {}", e);
        "Failed to set execution providers"
      })?
      .commit_from_file(path)
      .map_err(|e| {
        shlog!("Error: {}", e);
        "Failed to load model from given input path"
      })?;

    self.model = Some(Rc::new(ModelWrapper { session, shape }));
    let model_ref = self.model.as_ref().unwrap();
    Ok(Some(Var::new_object(model_ref, &MODEL_VAR_TYPE)))
  }
}

#[derive(Default)]
struct Activate {
  model_var: ParamVar,
  previous_model: Option<Var>,
  model: Option<Rc<ModelWrapper>>,
  output: Seq,
  reqs: ExposedTypes,
}

impl LegacyShard for Activate {
  fn registerName() -> &'static str {
    cstr!("ONNX.Activate")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("ONNX.Activate-rust-0x20230101")
  }

  fn name(&mut self) -> &str {
    "ONNX.Activate"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!("This shard runs the loaded ONNX model (created using the ONNX.Load shard) specified in the Model parameter. It takes the float sequence input, creates a tensor with a matching shape expected by the model, runs the model on the tensor using CUDA acceleration and returns the output tensor as a sequence of floats."))
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!("The float sequence to run inference on."))
  }

  fn outputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!("The output tensor as a sequence of floats."))
  }

  fn inputTypes(&mut self) -> &Types {
    &SEQ_OF_FLOAT_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &SEQ_OF_FLOAT_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&ACTIVATE_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => self.model_var.set_param(value),
      _ => unreachable!(),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.model_var.get_param(),
      _ => unreachable!(),
    }
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.reqs.clear();
    self.reqs.push(ExposedInfo::new_with_help_from_ptr(
      self.model_var.get_name(),
      shccstr!("The required ONNX model."),
      *MODEL_TYPE,
    ));
    Some(&self.reqs)
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.model_var.warmup(context);
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.model_var.cleanup(ctx);
    Ok(())
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let current_model = self.model_var.get();
    let model = match self.previous_model {
      None => {
        self.model = Some(Var::from_object_as_clone(current_model, &MODEL_VAR_TYPE)?);
        unsafe {
          let model_ptr = Rc::as_ptr(self.model.as_ref().unwrap()) as *mut ModelWrapper;
          &*model_ptr
        }
      }
      Some(ref previous_model) => {
        if previous_model != current_model {
          self.model = Some(Var::from_object_as_clone(current_model, &MODEL_VAR_TYPE)?);
          unsafe {
            let model_ptr = Rc::as_ptr(self.model.as_ref().unwrap()) as *mut ModelWrapper;
            &*model_ptr
          }
        } else {
          unsafe {
            let model_ptr = Rc::as_ptr(self.model.as_ref().unwrap()) as *mut ModelWrapper;
            &*model_ptr
          }
        }
      }
    };

    // Convert input sequence to tensor
    let seq: Seq = input.try_into()?;
    if seq.len() as i64 != model.shape.iter().product() {
      return Err("Input sequence length does not match model input shape");
    }

    let values: Vec<f32> = seq
      .iter()
      .map(|v| v.as_ref().try_into())
      .collect::<Result<Vec<f32>, _>>()?;

    // Convert shape from i64 to usize
    let shape: Vec<usize> = model.shape.iter().map(|&x| x as usize).collect();

    // Create tensor from values and shape
    let input_array = Array::from_shape_vec(shape, values)
      .map_err(|_| "Failed to create input array")?;
    let input_tensor = Value::from_array(input_array).map_err(|e| {
      shlog!("Error: {}", e);
      "Failed to create input tensor"
    })?;

    // Run inference
    let input_name = &model.session.inputs[0].name;
    let outputs = model.session.run(vec![(input_name, input_tensor)]).map_err(|e| {
      shlog!("Error: {}", e);
      "Failed to run model inference"
    })?;

    // Convert output tensor to sequence
    let output_tensor = &outputs[0];
    let tensor = output_tensor.try_extract_tensor::<f32>().map_err(|e| {
      shlog!("Error: {}", e);
      "Failed to extract output tensor"
    })?;
    let result = tensor.view().as_slice().unwrap().to_vec();

    self.output.set_len(result.len());
    for (i, v) in result.iter().enumerate() {
      self.output[i] = Var::from(*v);
    }

    Ok(Some(self.output.as_ref().into()))
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_onnx_rust(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }
  register_legacy_shard::<Load>();
  register_legacy_shard::<Activate>();
}
