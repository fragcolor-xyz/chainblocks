/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use crate::containers::WindowFlags;
use crate::containers::SEQ_OF_WINDOW_FLAGS;
use crate::containers::WINDOW_FLAGS_TYPE;
use crate::util;
use crate::Anchor;
use crate::ANCHOR_TYPES;
use crate::BOOL_VAR_SLICE;
use crate::HELP_OUTPUT_EQUAL_INPUT;
use shards::cstr;
use shards::shard;
use shards::shard::Shard;
use shards::shard_impl;
use shards::types::OptionalString;
use shards::types::ANY_TYPES;
use shards::types::{
  common_type, Context, ExposedInfo, ExposedTypes, InstanceData, ParamVar, Seq, ShardsVar, Type,
  Types, Var, ANYS_TYPES, SHARDS_OR_NONE_TYPES, STRING_TYPES,
};

use crate::{CONTEXTS_NAME, FLOAT2_VAR_SLICE, PARENTS_UI_NAME};
use shards::{
  core::register_shard,
  types::{BOOL_OR_NONE_SLICE, INT_OR_NONE_TYPES_SLICE, STRING_VAR_OR_NONE_SLICE},
};

lazy_static! {
  static ref WINDOW_FLAGS_OR_SEQ_TYPES: Vec<Type> = vec![*WINDOW_FLAGS_TYPE, *SEQ_OF_WINDOW_FLAGS];
}

#[derive(shard)]
#[shard_info(
  "UI.Window",
  "Creates a floating window which can be dragged, closed, collapsed, and resized."
)]
struct FormabbleWindowShard {
  #[shard_param("Title", "The window title displayed on the titlebar.", STRING_TYPES)]
  pub title: ParamVar,
  #[shard_param(
    "Position",
    "Absolute position; or when anchor is set, relative offset.",
    FLOAT2_VAR_SLICE
  )]
  pub position: ParamVar,
  #[shard_param("Anchor", "Corner or center of the screen.", ANCHOR_TYPES)]
  pub anchor: ParamVar,
  #[shard_param("Width", "The width of the rendered window.", INT_OR_NONE_TYPES_SLICE)]
  pub width: ParamVar,
  #[shard_param(
    "Height",
    "The height of the rendered window.",
    INT_OR_NONE_TYPES_SLICE
  )]
  pub height: ParamVar,
  #[shard_param(
    "Closable",
    "Whether the window will have a close button in its window title bar.",
    BOOL_OR_NONE_SLICE
  )]
  pub closable: ParamVar,
  #[shard_param(
    "Open",
    "Whether the window is open and being shown, or hidden.",
    BOOL_VAR_SLICE
  )]
  pub open: ParamVar,
  #[shard_param("Flags", "Window flags.", WINDOW_FLAGS_OR_SEQ_TYPES)]
  pub flags: ParamVar,
  #[shard_param(
    "ID",
    "An optional ID value to make the window unique if the title name collides.",
    STRING_VAR_OR_NONE_SLICE
  )]
  pub id: ParamVar,
  pub cached_id: Option<egui::Id>,
  #[shard_param("Contents", "The UI contents.", SHARDS_OR_NONE_TYPES)]
  pub contents: ShardsVar,
  #[shard_warmup]
  contexts: ParamVar,
  #[shard_warmup]
  parents: ParamVar,
  #[shard_required]
  required: ExposedTypes,
}

impl Default for FormabbleWindowShard {
  fn default() -> Self {
    Self {
      contexts: ParamVar::new_named(CONTEXTS_NAME),
      parents: ParamVar::new_named(PARENTS_UI_NAME),
      title: ParamVar::default(),
      position: ParamVar::default(),
      anchor: ParamVar::default(),
      width: ParamVar::default(),
      height: ParamVar::default(),
      closable: ParamVar::default(),
      open: ParamVar::default(),
      flags: ParamVar::default(),
      id: ParamVar::default(),
      cached_id: None,
      contents: ShardsVar::default(),
      required: Vec::new(),
    }
  }
}

#[shard_impl]
impl Shard for FormabbleWindowShard {
  fn input_types(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn input_help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "The value that will be passed to the Contents shards of the rendered window."
    ))
  }

  fn output_types(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn output_help(&mut self) -> OptionalString {
    *HELP_OUTPUT_EQUAL_INPUT
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.warmup_helper(context)?;

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.cleanup_helper()?;

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    util::require_context(&mut self.required);
    util::require_parents(&mut self.required);

    if self.id.is_variable() {
      let id_info = ExposedInfo {
        exposedType: common_type::string,
        name: self.id.get_name(),
        help: cstr!("The ID variable.").into(),
        ..ExposedInfo::default()
      };
      self.required.push(id_info);
    }

    if !self.contents.is_empty() {
      self.contents.compose(data)?;
    }

    // Always passthrough the input
    Ok(data.inputType)
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    // Try to retrieve a state variable determining whether the window should be shown
    let open: bool = if !self.open.is_none() {
      self.open.get().try_into()?
    } else {
      true
    };

    // If the window should be drawn this frame
    if open {
      let gui_ctx = &util::get_current_context(&self.contexts)?.egui_ctx;

      let mut failed = false;
      if !self.contents.is_empty() {
        let title: &str = self.title.get().try_into()?;
        let mut window = egui::Window::new(title);

        if let Ok(id) = <&str>::try_from(self.id.get()) {
          let id = self.cached_id.get_or_insert_with(|| egui::Id::new(id));
          window = window.id(*id);
        }

        window = if self.anchor.get().is_none() {
          // note: in egui the position is relative to the top-left corner of the whole UI
          // but a window is constrained by the available rect of the central panel.
          // Thus, we offset it to make it more intuitive for users.
          // i.e. the position is now relative to the top-left corner of the central panel.
          let position = self.position.get();
          if !position.is_none() {
            let pos: (f32, f32) = self.position.get().try_into()?;
            let rect = gui_ctx.available_rect();
            window.default_pos(egui::Pos2 {
              x: pos.0 + rect.min.x,
              y: pos.1 + rect.min.y,
            })
          } else {
            window
          }
        } else {
          let offset: (f32, f32) = self.position.get().try_into().unwrap_or_default();
          let anchor: Anchor = self.anchor.get().try_into()?;
          window.anchor(anchor.into(), offset)
        };

        let width = self.width.get();
        if !width.is_none() {
          let width: i64 = width.try_into()?;
          window = window.min_width(width as f32).default_width(width as f32);
        }

        let height = self.height.get();
        if !height.is_none() {
          let height: i64 = height.try_into()?;
          window = window
            .min_height(height as f32)
            .default_height(height as f32)
        }

        let closable = self.closable.get();
        if !closable.is_none() {
          let closable: bool = closable.try_into()?;

          if closable {
            // if close button is enabled, it must be provided with a state variable for tracking whether window is open
            // this open variable is to provide a way for other Shards to be notified of the window being closed
            if !self.open.is_variable() {
              return Err("Window: open variable required when closable is true");
            } else {
              window = window.open(self.open.get_mut().try_into()?);
            }
          }
        } // else, if no closable provided, then default to false, which does nothing

        for bits in FormabbleWindowShard::try_get_flags(self.flags.get())? {
          match (WindowFlags { bits }) {
            WindowFlags::NoTitleBar => {
              window = window.title_bar(false);
            }
            WindowFlags::NoResize => {
              window = window.resizable(false);
            }
            WindowFlags::Scrollbars => {
              window = window.scroll2([true, true]);
            }
            WindowFlags::NoCollapse => {
              window = window.collapsible(false);
            }
            WindowFlags::Immovable => {
              window = window.movable(false);
            }
            _ => (),
          }
        }

        window.show(gui_ctx, |ui| {
          if !width.is_none() {
            ui.set_width(ui.available_width());
          }
          if !height.is_none() {
            ui.set_height(ui.available_height());
          }
          if util::activate_ui_contents(context, input, ui, &mut self.parents, &mut self.contents)
            .is_err()
          {
            failed = true;
          }
        });

        if failed {
          return Err("Failed to activate window contents");
        }
      }
    }

    // Always passthrough the input
    Ok(*input)
  }
}

impl FormabbleWindowShard {
  fn try_get_flags(var: &Var) -> Result<Vec<i32>, &str> {
    match var.valueType {
      shards::SHType_Enum => Ok(vec![unsafe {
        var.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue
      }]),
      shards::SHType_Seq => {
        let seq: Seq = var.try_into()?;
        seq
          .iter()
          .map(|v| Ok(unsafe { v.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue }))
          .collect()
      }
      shards::SHType_None => Ok(Vec::new()),
      _ => Err("Invalid type"),
    }
  }
}

pub fn register_shards() {
  register_shard::<FormabbleWindowShard>();
}
