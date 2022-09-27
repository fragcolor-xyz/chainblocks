/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use super::Combo;
use crate::core::cloneVar;
use crate::shard::Shard;
use crate::shards::gui::util;
use crate::shards::gui::FLOAT_VAR_SLICE;
use crate::shards::gui::INT_VAR_OR_NONE_SLICE;
use crate::shards::gui::PARENTS_UI_NAME;
use crate::shardsc;
use crate::types::common_type;
use crate::types::Context;
use crate::types::ExposedInfo;
use crate::types::ExposedTypes;
use crate::types::InstanceData;
use crate::types::OptionalString;
use crate::types::ParamVar;
use crate::types::Parameters;
use crate::types::Seq;
use crate::types::Type;
use crate::types::Types;
use crate::types::Var;
use crate::types::ANYS_TYPES;
use crate::types::ANY_TYPES;
use crate::types::STRING_OR_NONE_SLICE;
use std::cmp::Ordering;
use std::ffi::CStr;

lazy_static! {
  static ref COMBO_PARAMETERS: Parameters = vec![
    (
      cstr!("Label"),
      cstr!("The text label of this checkbox."),
      STRING_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("Index"),
      cstr!("The index of the selected item."),
      INT_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("Width"),
      cstr!("The width of the button and menu."),
      FLOAT_VAR_SLICE,
    )
      .into(),
  ];
}

impl Default for Combo {
  fn default() -> Self {
    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);
    Self {
      parents,
      requiring: Vec::new(),
      label: ParamVar::default(),
      index: ParamVar::default(),
      width: ParamVar::default(),
      exposing: Vec::new(),
      should_expose: false,
      tmp: 0,
    }
  }
}

impl Shard for Combo {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("UI.Combo")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("UI.Combo-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.Combo"
  }

  fn inputTypes(&mut self) -> &Types {
    &ANYS_TYPES
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!("A sequence of values."))
  }

  fn outputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn outputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!("The selected value."))
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&COMBO_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.label.set_param(value)),
      1 => Ok(self.index.set_param(value)),
      2 => Ok(self.width.set_param(value)),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.label.get_param(),
      1 => self.index.get_param(),
      2 => self.width.get_param(),
      _ => Var::default(),
    }
  }

  fn hasCompose() -> bool {
    true
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    if self.index.is_variable() {
      self.should_expose = true; // assume we expose a new variable

      let shared: ExposedTypes = data.shared.into();
      for var in shared {
        let (a, b) = unsafe {
          (
            CStr::from_ptr(var.name),
            CStr::from_ptr(self.index.get_name()),
          )
        };
        if CStr::cmp(a, b) == Ordering::Equal {
          self.should_expose = false;
          if var.exposedType.basicType != shardsc::SHType_Int {
            return Err("Combo: int variable required.");
          }
          break;
        }
      }
    }

    Ok(common_type::ANY)
  }

  fn exposedVariables(&mut self) -> Option<&ExposedTypes> {
    if self.index.is_variable() && self.should_expose {
      self.exposing.clear();

      let exp_info = ExposedInfo {
        exposedType: common_type::INT,
        name: self.index.get_name(),
        help: cstr!("The exposed int variable").into(),
        ..ExposedInfo::default()
      };

      self.exposing.push(exp_info);
      Some(&self.exposing)
    } else {
      None
    }
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    // Add UI.Parents to the list of required variables
    util::require_parents(&mut self.requiring, &self.parents);

    Some(&self.requiring)
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.parents.warmup(ctx);

    self.label.warmup(ctx);
    self.index.warmup(ctx);
    self.width.warmup(ctx);

    if self.should_expose {
      self.index.get_mut().valueType = common_type::INT.basicType;
    }

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.width.cleanup();
    self.index.cleanup();
    self.label.cleanup();

    self.parents.cleanup();

    Ok(())
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Var, &str> {
    if let Some(ui) = util::get_current_parent(*self.parents.get())? {
      let label: &str = self.label.get().try_into()?;

      let index = &mut if self.index.is_variable() {
        let index: i64 = self.index.get().try_into()?;
        index as usize
      } else {
        self.tmp
      };

      let mut combo = egui::ComboBox::from_label(label);
      let width = self.width.get();
      if !width.is_none() {
        combo = combo.width(width.try_into()?);
      }

      let seq: Seq = input.try_into()?;
      let response = combo.show_index(ui, index, seq.len(), |i| {
        // FIXME type might not be string so we need a way to convert in all cases
        let str: &str = (&seq[i]).try_into().unwrap();
        str.to_owned()
      });

      if response.changed {
        if self.index.is_variable() {
          self.index.set((*index as i64).into());
        } else {
          self.tmp = *index;
        }
      }

      let mut ret = Var::default();
      cloneVar(&mut ret, &seq[*index]);
      Ok(ret)
    } else {
      Err("No UI parent")
    }
  }
}
