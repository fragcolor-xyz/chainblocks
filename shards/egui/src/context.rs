/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use crate::bindings;
use crate::bindings::make_native_io_output;
use crate::bindings::make_texture_updates;
use crate::egui_host::EguiHost;
use crate::util;

use crate::HELP_OUTPUT_EQUAL_INPUT;
use crate::INPUT_CONTEXT_TYPE;

use shards::core::register_shard;
use shards::fourCharacterCode;
use shards::shard::Shard;
use shards::shardsc;
use shards::types::common_type;
use shards::types::Context;
use shards::types::ExposedInfo;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::OptionalString;
use shards::types::ParamVar;
use shards::types::ShardsVar;
use shards::types::Type;
use shards::types::Types;
use shards::types::Var;
use shards::types::ANY_TYPES;
use shards::types::FRAG_CC;
use shards::types::NONE_TYPES;
use shards::types::SHARDS_OR_NONE_TYPES;
use std::f32::consts::E;
use std::ffi::CStr;
use std::rc::Rc;

struct UIOutput {
  full_output: egui::FullOutput,
  ctx: egui::Context,
}

static UI_OUTPUT_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"uiui"));

lazy_static! {
  static ref GFX_QUEUE_TYPE: Type =
    unsafe { *(bindings::gfx_getQueueType() as *mut shardsc::SHTypeInfo) };
  static ref GFX_QUEUE_TYPES: Vec<Type> = vec![*GFX_QUEUE_TYPE];
  static ref GFX_QUEUE_VAR: Type = Type::context_variable(&GFX_QUEUE_TYPES);
  static ref GFX_QUEUE_VAR_TYPES: Vec<Type> = vec![*GFX_QUEUE_VAR];
  static ref GFX_QUEUE_VAR_OR_NONE_TYPES: Vec<Type> = vec![common_type::none, *GFX_QUEUE_VAR];
  static ref UI_OUTPUT_TYPES: Vec<Type> = vec![UI_OUTPUT_TYPE];
  static ref UI_OUTPUT_SEQ_TYPE: Type = Type::seq(&UI_OUTPUT_TYPES);
  static ref UI_OUTPUT_SEQ_TYPES: Vec<Type> = vec![*UI_OUTPUT_SEQ_TYPE];
}

#[derive(shards::shard)]
#[shard_info("UI", "Initializes a UI context")]
struct ContextShard {
  #[shard_param("Contents", "The UI contents.", SHARDS_OR_NONE_TYPES)]
  contents: ShardsVar,
  #[shard_param("Scale", "The UI scale", [common_type::none, common_type::float, common_type::float_var])]
  scale: ParamVar,
  #[shard_param("Queue", "The draw queue.", GFX_QUEUE_VAR_OR_NONE_TYPES)]
  queue: ParamVar,
  host: EguiHost,
  #[shard_required]
  requiring: ExposedTypes,
  exposing: ExposedTypes,
  #[shard_warmup]
  input_context: ParamVar,
  input_translator: bindings::InputTranslator,
}

impl Default for ContextShard {
  fn default() -> Self {
    Self {
      host: EguiHost::default(),
      requiring: Vec::new(),
      queue: ParamVar::default(),
      contents: ShardsVar::default(),
      scale: ParamVar::default(),
      exposing: Vec::new(),
      input_context: unsafe {
        ParamVar::new_named(
          CStr::from_ptr(bindings::gfx_getInputContextVarName())
            .to_str()
            .unwrap(),
        )
      },
      input_translator: bindings::InputTranslator::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for ContextShard {
  fn input_types(&mut self) -> &std::vec::Vec<Type> {
    &ANY_TYPES
  }

  fn input_help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "The value that will be passed to the Contents shards of the UI."
    ))
  }

  fn output_types(&mut self) -> &std::vec::Vec<Type> {
    &UI_OUTPUT_TYPES
  }

  fn output_help(&mut self) -> OptionalString {
    *HELP_OUTPUT_EQUAL_INPUT
  }

  fn exposed_variables(&mut self) -> Option<&ExposedTypes> {
    Some(&self.exposing)
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    // Add Input context to the list of required variables
    let exp_info = ExposedInfo {
      exposedType: *INPUT_CONTEXT_TYPE,
      name: self.input_context.get_name(),
      help: cstr!("The input context.").into(),
      ..ExposedInfo::default()
    };
    self.requiring.push(exp_info);

    // we need to inject the UI context to the inner shards
    let mut data = *data;
    // clone shared into a new vector we can append things to
    let mut shared: ExposedTypes = data.shared.into();

    // expose UI context
    for exposed in self.host.get_exposed_info() {
      shared.push(*exposed);
    }

    // update shared
    data.shared = (&shared).into();

    let _result = self.contents.compose(&data)?;

    self.exposing.clear();
    util::expose_contents_variables(&mut self.exposing, &self.contents);

    Ok(UI_OUTPUT_TYPE)
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.host.warmup(ctx)?;
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.cleanup_helper()?;
    self.host.cleanup()?;
    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    if self.contents.is_empty() {
      return Ok(*input);
    }

    let egui_input = unsafe {
      let scale: Option<f32> = self.scale.get().try_into().ok();

      &*(bindings::gfx_getEguiWindowInputs(
        self.input_translator.as_mut_ptr() as *mut bindings::gfx_EguiInputTranslator,
        self.input_context.get() as *const _ as *const bindings::SHVar,
        scale.unwrap_or(1.0),
      ) as *const bindings::egui_Input)
    };

    let full_output = if egui_input.pixelsPerPoint > 0.0 {
      self
        .host
        .activate(&egui_input, &(&self.contents).into(), context, input)?;
      self.host.take_egui_output()
    } else {
      egui::FullOutput::default()
    };

    let ctx = &self.host.get_context().egui_ctx;
    unsafe {
      let io_output = make_native_io_output(ctx, &full_output)?;
      let c_output = io_output.get_c_output();
      bindings::gfx_applyEguiIOOutput(
        self.input_translator.as_mut_ptr() as *mut bindings::gfx_EguiInputTranslator,
        &c_output,
        self.input_context.get() as *const _ as *const bindings::SHVar,
      );
    }

    Ok(Var::new_ref_counted(
      UIOutput {
        full_output,
        ctx: self.host.get_context().egui_ctx.clone(),
      },
      &UI_OUTPUT_TYPE,
    ))
  }
}

#[derive(shards::shard)]
#[shard_info("UI.Render", "Render given UI")]
struct RenderShard {
  #[shard_required]
  required: ExposedTypes,
  #[shard_param("Queue", "The draw queue.", GFX_QUEUE_VAR_TYPES)]
  queue: ParamVar,
  renderer: bindings::Renderer,
}

impl Default for RenderShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      queue: ParamVar::default(),
      renderer: bindings::Renderer::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for RenderShard {
  fn input_types(&mut self) -> &Types {
    &UI_OUTPUT_SEQ_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &UI_OUTPUT_SEQ_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.cleanup_helper()?;

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Var, &str> {
    let queue_var = self.queue.get();
    let seq = input.as_seq()?;
    let num_ui_outputs = seq.len();
    for (idx, var) in seq.iter().enumerate() {
      let ui_output = unsafe { &*Var::from_ref_counted_object::<UIOutput>(&var, &UI_OUTPUT_TYPE)? };

      // Only render the most recent output
      if idx == (num_ui_outputs - 1) {
        let draw_scale = ui_output.ctx.pixels_per_point();
        let queue = unsafe {
          bindings::gfx_getDrawQueueFromVar(queue_var as *const _ as *const bindings::SHVar)
            as *const bindings::gfx_DrawQueuePtr
        };
        self
          .renderer
          .render(&ui_output.ctx, &ui_output.full_output, queue, draw_scale)?;
      } else {
        // Apply texture updates only, skip rendering
        self
          .renderer
          .apply_texture_updates_only(&ui_output.full_output)?;
      }
    }

    Ok(*input)
  }
}

pub fn register_shards() {
  register_shard::<ContextShard>();
  register_shard::<RenderShard>();
}
