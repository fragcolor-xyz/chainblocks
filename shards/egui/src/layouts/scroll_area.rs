/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use super::ScrollArea;
use crate::util;
use crate::util::with_possible_panic;
use crate::EguiId;
use crate::HELP_OUTPUT_EQUAL_INPUT;
use crate::PARENTS_UI_NAME;
use shards::shard::LegacyShard;
use shards::types::Context;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::OptionalString;
use shards::types::ParamVar;
use shards::types::Parameters;
use shards::types::ShardsVar;
use shards::types::Type;
use shards::types::Types;
use shards::types::Var;
use shards::types::ANY_TYPES;
use shards::types::BOOL_TYPES;
use shards::types::SHARDS_OR_NONE_TYPES;

lazy_static! {
  static ref AREA_PARAMETERS: Parameters = vec![
    (
      cstr!("Contents"),
      shccstr!("The UI contents."),
      &SHARDS_OR_NONE_TYPES[..],
    )
      .into(),
    (
      cstr!("Horizontal"),
      shccstr!("Enable horizontal scrolling."),
      &BOOL_TYPES[..],
    )
      .into(),
    (
      cstr!("Vertical"),
      shccstr!("Enable vertical scrolling."),
      &BOOL_TYPES[..],
    )
      .into(),
    (
      cstr!("AlwaysShow"),
      shccstr!("Always show the enabled scroll bars even if not needed."),
      &BOOL_TYPES[..],
    )
      .into(),
  ];
}

impl Default for ScrollArea {
  fn default() -> Self {
    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);
    Self {
      parents,
      requiring: Vec::new(),
      contents: ShardsVar::default(),
      horizontal: ParamVar::new(false.into()),
      vertical: ParamVar::new(true.into()),
      alwaysShow: ParamVar::new(false.into()),
      exposing: Vec::new(),
    }
  }
}

impl LegacyShard for ScrollArea {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("UI.ScrollArea")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("UI.ScrollArea-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.ScrollArea"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "Add vertical and/or horizontal scrolling to a contained UI."
    ))
  }

  fn inputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "The value that will be passed to the Contents shards of the scroll area."
    ))
  }

  fn outputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn outputHelp(&mut self) -> OptionalString {
    *HELP_OUTPUT_EQUAL_INPUT
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&AREA_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => self.contents.set_param(value),
      1 => self.horizontal.set_param(value),
      2 => self.vertical.set_param(value),
      3 => self.alwaysShow.set_param(value),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.contents.get_param(),
      1 => self.horizontal.get_param(),
      2 => self.vertical.get_param(),
      3 => self.alwaysShow.get_param(),
      _ => Var::default(),
    }
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    // Add UI.Parents to the list of required variables
    util::require_parents(&mut self.requiring);

    Some(&self.requiring)
  }

  fn exposedVariables(&mut self) -> Option<&ExposedTypes> {
    self.exposing.clear();

    if util::expose_contents_variables(&mut self.exposing, &self.contents) {
      Some(&self.exposing)
    } else {
      None
    }
  }

  fn hasCompose() -> bool {
    true
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    if !self.contents.is_empty() {
      self.contents.compose(data)?;
    }

    // Always passthrough the input
    Ok(data.inputType)
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.parents.warmup(ctx);
    if !self.contents.is_empty() {
      self.contents.warmup(ctx)?;
    }
    self.horizontal.warmup(ctx);
    self.vertical.warmup(ctx);
    self.alwaysShow.warmup(ctx);

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.alwaysShow.cleanup(ctx);
    self.vertical.cleanup(ctx);
    self.horizontal.cleanup(ctx);
    if !self.contents.is_empty() {
      self.contents.cleanup(ctx);
    }
    self.parents.cleanup(ctx);

    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    if self.contents.is_empty() {
      return Ok(*input);
    }

    if let Some(ui) = util::get_current_parent_opt(self.parents.get())? {
      with_possible_panic(|| {
        let visibility = if self.alwaysShow.get().try_into()? {
          egui::scroll_area::ScrollBarVisibility::AlwaysVisible
        } else {
          egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded
        };
        egui::ScrollArea::new([
          self.horizontal.get().try_into()?,
          self.vertical.get().try_into()?,
        ])
        .id_source(EguiId::new(self, 0))
        .scroll_bar_visibility(visibility)
        .show(ui, |ui| {
          util::activate_ui_contents(context, input, ui, &mut self.parents, &mut self.contents)
        })
        .inner
      })??;

      // Always passthrough the input
      Ok(*input)
    } else {
      Err("No UI parent")
    }
  }
}
