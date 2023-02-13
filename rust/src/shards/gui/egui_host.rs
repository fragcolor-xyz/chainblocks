use super::util;
use super::EguiContext;
use super::CONTEXTS_NAME;
use super::EGUI_CTX_TYPE;
use super::EGUI_UI_SEQ_TYPE;
use super::GFX_CONTEXT_TYPE;
use super::GFX_QUEUE_VAR_TYPES;
use super::HELP_OUTPUT_EQUAL_INPUT;
use super::INPUT_CONTEXT_TYPE;
use super::PARENTS_UI_NAME;
use crate::core::Core;
use crate::shard::Shard;
use crate::shardsc;
use crate::shardsc::Shards;
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
use egui_gfx::egui_FullOutput;
use egui_gfx::egui_Input;
use egui_gfx::make_native_full_output;
use egui_gfx::NativeFullOutput;
use std::ffi::CStr;

pub struct EguiHost {
  context: Option<egui::Context>,
  full_output: Option<NativeFullOutput>,
  instance: ParamVar,
  parents: ParamVar,
  exposed: Vec<ExposedInfo>,
}

impl Default for EguiHost {
  fn default() -> Self {
    let mut instance = ParamVar::default();
    instance.set_name(CONTEXTS_NAME);

    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);

    let exposed = vec![
      ExposedInfo {
        exposedType: EGUI_CTX_TYPE,
        name: instance.get_name(),
        help: cstr!("The UI context.").into(),
        isMutable: false,
        isProtected: true, // don't allow to be used in code/wires
        isTableEntry: false,
        global: false,
        isPushTable: false,
      },
      ExposedInfo {
        exposedType: EGUI_UI_SEQ_TYPE,
        name: parents.get_name(),
        help: cstr!("The parent UI objects.").into(),
        isMutable: false,
        isProtected: true, // don't allow to be used in code/wires
        isTableEntry: false,
        global: false,
        isPushTable: false,
      },
    ];

    Self {
      context: None,
      full_output: None,
      instance: instance,
      parents: parents,
      exposed: exposed,
    }
  }
}

impl EguiHost {
  pub fn get_context(&self) -> &egui::Context {
    self.context.as_ref().expect("No UI context")
  }

  pub fn get_exposed_info(&self) -> &Vec<ExposedInfo> {
    &self.exposed
  }

  pub fn warmup(&mut self, ctx: &Context) -> Result<(), &'static str> {
    self.context = Some(egui::Context::default());
    self.instance.warmup(ctx);
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

  pub fn cleanup(&mut self) -> Result<(), &'static str> {
    self.parents.cleanup();
    self.instance.cleanup();
    Ok(())
  }

  pub fn activate_base<F, R>(
    &mut self,
    egui_input: &egui_Input,
    run_ui: F,
  ) -> Result<R, &'static str>
  where
    F: FnOnce(&mut Self, &egui::Context) -> Result<R, &'static str>,
  {
    // note: we have to clone the egui context here, otherwise we can't borrow it and at the same
    // time pass self to the closure. Fortunately, cloning the context is both cheap and a
    // supported pattern.
    let gui_ctx = if let Some(gui_ctx) = &self.context {
      gui_ctx.clone()
    } else {
      return Err("No UI context");
    };

    let raw_input = egui_gfx::translate_raw_input(egui_input);
    match raw_input {
      Err(_error) => {
        shlog_debug!("Input translation error: {:?}", _error);
        return Err("Input translation error");
      }
      Ok(raw_input) => {
        let draw_scale = raw_input.pixels_per_point.unwrap_or(1.0);

        let mut result = Err("Failed to run the UI");
        let egui_output = gui_ctx.run(raw_input, |ctx| {
          result = run_ui(self, ctx);
        });

        return match result {
          Ok(r) => {
            self.full_output = Some(make_native_full_output(&gui_ctx, egui_output, draw_scale)?);
            Ok(r)
          }
          Err(e) => Err(e),
        };
      }
    }
  }

  pub fn activate_shards(
    &mut self,
    egui_input: &egui_Input,
    contents: &Shards,
    context: &Context,
    input: &Var,
  ) -> Result<Var, &'static str> {
    self.activate_base(egui_input, |this, ctx| {
      // Push empty parent UI in case this context is nested inside another UI
      util::update_seq(&mut this.parents, |seq| {
        seq.push(&Var::default());
      })?;

      let mut _output = Var::default();
      let wire_state: WireState =
        util::with_object_stack_var(&mut this.instance, ctx, &EGUI_CTX_TYPE, || {
          Ok(unsafe {
            (*Core).runShards.unwrap()(
              *contents,
              context as *const _ as *mut _,
              input,
              &mut _output,
            )
            .into()
          })
        })?;

      if wire_state == WireState::Error {
        return Err("Failed to activate UI contents");
      }

      // Pop empty parent UI
      util::update_seq(&mut this.parents, |seq| {
        seq.pop();
      })?;

      Ok(())
    })?;

    Ok(*input)
  }

  pub fn get_egui_output(&self) -> *const egui_FullOutput {
    return &self.full_output.as_ref().unwrap().full_output;
  }
}

mod native {
  use egui_gfx::{egui_FullOutput, egui_Input};

  use super::EguiHost;
  use crate::{
    shardsc::{self, SHVar, Shards},
    types::{Context, Var},
  };

  #[no_mangle]
  unsafe extern "C" fn egui_createHost() -> *mut EguiHost {
    Box::into_raw(Box::new(EguiHost::default()))
  }

  #[no_mangle]
  unsafe extern "C" fn egui_destroyHost(ptr: *mut EguiHost) {
    drop(Box::from_raw(ptr))
  }

  #[no_mangle]
  unsafe extern "C" fn egui_hostGetExposedTypeInfo(
    ptr: *mut EguiHost,
    out_info: *mut shardsc::SHExposedTypesInfo,
  ) {
    (*out_info) = (&(*ptr).exposed).into();
  }

  #[no_mangle]
  unsafe extern "C" fn egui_hostWarmup(ptr: *mut EguiHost, ctx: &Context) {
    (*ptr).warmup(&ctx).unwrap();
  }

  #[no_mangle]
  unsafe extern "C" fn egui_hostCleanup(ptr: *mut EguiHost) {
    (*ptr).cleanup().unwrap();
  }

  #[no_mangle]
  unsafe extern "C" fn egui_hostActivate(
    ptr: *mut EguiHost,
    egui_input: *const egui_Input,
    contents: &Shards,
    context: &Context,
    input: &SHVar,
    output: *mut SHVar,
  ) -> *const u8 {
    match (*ptr).activate_shards(&*egui_input, &contents, &context, &input) {
      Ok(result) => {
        *output = result;
        std::ptr::null()
      }
      Err(str) => str.as_ptr(),
    }
  }

  #[no_mangle]
  unsafe extern "C" fn egui_hostGetOutput(ptr: *const EguiHost) -> *const egui_FullOutput {
    &(*ptr)
      .full_output
      .as_ref()
      .expect("Expected egui output")
      .full_output
  }
}
