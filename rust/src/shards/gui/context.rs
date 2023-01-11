/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use super::util;
use super::EguiContext;
use super::CONTEXTS_NAME;
use super::EGUI_CTX_TYPE;
use super::EGUI_UI_SEQ_TYPE;
use super::GFX_GLOBALS_TYPE;
use super::GFX_QUEUE_VAR_TYPES;
use super::HELP_OUTPUT_EQUAL_INPUT;
use super::PARENTS_UI_NAME;
use crate::shard::Shard;
use crate::shardsc;
use crate::types::Context;
use crate::types::ExposedInfo;
use crate::types::ExposedTypes;
use crate::types::InstanceData;
use crate::types::OptionalString;
use crate::types::ParamVar;
use crate::types::Parameters;
use crate::types::RawString;
use crate::types::Seq;
use crate::types::ShardsVar;
use crate::types::Type;
use crate::types::Var;
use crate::types::WireState;
use crate::types::ANY_TYPES;
use crate::types::SHARDS_OR_NONE_TYPES;
use std::ffi::CStr;

lazy_static! {
  static ref CONTEXT_PARAMETERS: Parameters = vec![
    (
      cstr!("Queue"),
      shccstr!("The draw queue."),
      &GFX_QUEUE_VAR_TYPES[..]
    )
      .into(),
    (
      cstr!("Contents"),
      shccstr!("The UI contents."),
      &SHARDS_OR_NONE_TYPES[..],
    )
      .into(),
  ];
}

impl Default for EguiContext {
  fn default() -> Self {
    let mut ctx = ParamVar::default();
    ctx.set_name(CONTEXTS_NAME);

    let mut mw_globals = ParamVar::default();
    unsafe {
      let gfx_globals_var_name = shardsc::gfx_getGraphicsContextVarName() as shardsc::SHString;
      mw_globals.set_name(CStr::from_ptr(gfx_globals_var_name).to_str().unwrap());
    }

    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);

    Self {
      context: None,
      instance: ctx,
      requiring: Vec::new(),
      queue: ParamVar::default(),
      contents: ShardsVar::default(),
      main_window_globals: mw_globals,
      parents,
      renderer: egui_gfx::Renderer::new(),
      input_translator: egui_gfx::InputTranslator::new(),
    }
  }
}

impl Shard for EguiContext {
  fn registerName() -> &'static str {
    cstr!("UI")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("UI-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!("Initializes a UI context."))
  }

  fn inputTypes(&mut self) -> &std::vec::Vec<Type> {
    &ANY_TYPES
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "The value that will be passed to the Contents shards of the UI."
    ))
  }

  fn outputTypes(&mut self) -> &std::vec::Vec<Type> {
    &ANY_TYPES
  }

  fn outputHelp(&mut self) -> OptionalString {
    *HELP_OUTPUT_EQUAL_INPUT
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&CONTEXT_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.queue.set_param(value)),
      1 => self.contents.set_param(value),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.queue.get_param(),
      1 => self.contents.get_param(),
      _ => Var::default(),
    }
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    // Add GFX.MainWindow to the list of required variables
    let exp_info = ExposedInfo {
      exposedType: *GFX_GLOBALS_TYPE,
      name: self.main_window_globals.get_name(),
      help: cstr!("The exposed main window.").into(),
      ..ExposedInfo::default()
    };
    self.requiring.push(exp_info);

    Some(&self.requiring)
  }

  fn hasCompose() -> bool {
    true
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    // we need to inject the UI context to the inner shards
    let mut data = *data;
    // clone shared into a new vector we can append things to
    let mut shared: ExposedTypes = data.shared.into();
    // expose UI context
    let ctx_info = ExposedInfo {
      exposedType: EGUI_CTX_TYPE,
      name: self.instance.get_name(),
      help: cstr!("The UI context.").into(),
      isMutable: false,
      isProtected: true, // don't allow to be used in code/wires
      isTableEntry: false,
      global: false,
      isPushTable: false,
    };
    shared.push(ctx_info);
    // expose UI parents seq
    let ui_info = ExposedInfo {
      exposedType: EGUI_UI_SEQ_TYPE,
      name: self.parents.get_name(),
      help: cstr!("The parent UI objects.").into(),
      isMutable: false,
      isProtected: true, // don't allow to be used in code/wires
      isTableEntry: false,
      global: false,
      isPushTable: false,
    };
    shared.push(ui_info);
    // update shared
    data.shared = (&shared).into();

    if !self.contents.is_empty() {
      let outputType = self.contents.compose(&data)?;
      return Ok(outputType);
    }

    // Always passthrough the input
    Ok(data.inputType)
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.context = Some(egui::Context::default());
    self.instance.warmup(ctx);
    self.queue.warmup(ctx);
    self.contents.warmup(ctx)?;
    self.main_window_globals.warmup(ctx);
    self.parents.warmup(ctx);

    // Initialize the parents stack in the root UI.
    // Every other UI elements will reference it and push or pop UIs to it.
    if !self.parents.get().is_seq() {
      self.parents.set(Seq::new().as_ref().into());
    }

    // Context works the same
    if !self.instance.get().is_seq() {
      self.instance.set(Seq::new().as_ref().into());
    }

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.parents.cleanup();
    self.main_window_globals.cleanup();
    self.contents.cleanup();
    self.queue.cleanup();
    self.instance.cleanup();
    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    let gui_ctx = if let Some(gui_ctx) = &self.context {
      gui_ctx
    } else {
      return Err("No UI context");
    };

    if self.contents.is_empty() {
      return Ok(*input);
    }

    let raw_input = unsafe {
      let inputs = shardsc::gfx_getEguiWindowInputs(
        self.input_translator.as_mut_ptr() as *mut shardsc::gfx_EguiInputTranslator,
        self.main_window_globals.get(),
        1.0,
      ) as *const egui_gfx::egui_Input;
      egui_gfx::translate_raw_input(&*inputs)
    };

    match raw_input {
      Err(_error) => {
        shlog_debug!("Input translation error: {:?}", _error);
        Err("Input translation error")
      }
      Ok(raw_input) => {
        let draw_scale = raw_input.pixels_per_point.unwrap_or(1.0);

        let mut error: Option<&str> = None;
        let egui_output = gui_ctx.run(raw_input, |ctx| {
          error = (|| -> Result<(), &str> {
            // Push empty parent UI in case this context is nested inside another UI
            util::update_seq(&mut self.parents, |seq| {
              seq.push(Var::default());
            })?;

            let mut _output = Var::default();
            let wire_state =
              util::with_object_stack_var(&mut self.instance, ctx, &EGUI_CTX_TYPE, || {
                Ok(self.contents.activate(context, input, &mut _output))
              })?;

            if wire_state == WireState::Error {
              return Err("Failed to activate UI contents");
            }

            // Pop empty parent UI
            util::update_seq(&mut self.parents, |seq| {
              seq.pop();
            })?;

            Ok(())
          })()
          .err();
        });

        if let Some(e) = error {
          return Err(e);
        }

        #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
        if let Some(url) = &egui_output.platform_output.open_url {
          webbrowser::open(&url.url).map_err(|e| {
            shlog_error!("{}", e);
            "Failed to open URL."
          })?;
        }

        let queue_var = self.queue.get();
        unsafe {
          let queue = shardsc::gfx_getDrawQueueFromVar(queue_var);
          self.renderer.render(
            gui_ctx,
            egui_output,
            queue as *const egui_gfx::gfx_DrawQueuePtr,
            draw_scale,
          )?;
        }

        Ok(*input)
      }
    }
  }
}
