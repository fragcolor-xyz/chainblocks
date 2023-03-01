/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use super::ListBox;
use crate::shard::Shard;
use crate::shards::gui::util;
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
use crate::types::ShardsVar;
use crate::types::Type;
use crate::types::Types;
use crate::types::Var;
use crate::types::ANYS_TYPES;
use crate::types::ANY_TYPES;
use crate::types::SHARDS_OR_NONE_TYPES;
use std::cmp::Ordering;
use std::ffi::CStr;

lazy_static! {
  static ref LISTBOX_PARAMETERS: Parameters = vec![
    (
      cstr!("Index"),
      shccstr!("The index of the selected item."),
      INT_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("Template"),
      shccstr!("Custom rendering"),
      &SHARDS_OR_NONE_TYPES[..],
    )
      .into(),
  ];
}

impl Default for ListBox {
  fn default() -> Self {
    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);
    Self {
      parents,
      requiring: Vec::new(),
      index: ParamVar::default(),
      template: ShardsVar::default(),
      exposing: Vec::new(),
      should_expose: false,
      tmp: 0,
    }
  }
}

impl Shard for ListBox {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("UI.ListBox")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("UI.ListBox-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.ListBox"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!("A list selection."))
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
    Some(&LISTBOX_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.index.set_param(value)),
      1 => self.template.set_param(value),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.index.get_param(),
      1 => self.template.get_param(),
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

    let input_type = data.inputType;
    let slice = unsafe {
      let ptr = input_type.details.seqTypes.elements;
      std::slice::from_raw_parts(ptr, input_type.details.seqTypes.len as usize)
    };

    let element_type = match slice.len() {
      0 => common_type::none,
      1 => slice[0],
      _ => {
        if slice.iter().skip(1).all(|t| *t == slice[0]) {
          slice[0]
        } else {
          common_type::any
        }
      }
    };

    if !self.template.is_empty() {
      let mut data = *data;
      data.inputType = element_type;
      self.template.compose(&data)?;
    } else if element_type.basicType != shardsc::SHType_String {
      return Err("Input is not a sequence of strings, a template must be provided.");
    }

    Ok(element_type)
  }

  fn exposedVariables(&mut self) -> Option<&ExposedTypes> {
    if self.index.is_variable() && self.should_expose {
      self.exposing.clear();

      let exp_info = ExposedInfo {
        exposedType: common_type::int,
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

    self.index.warmup(ctx);
    if self.should_expose {
      self.index.get_mut().valueType = common_type::int.basicType;
    }

    if !self.template.is_empty() {
      self.template.warmup(ctx)?;
    }

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.template.cleanup();
    self.index.cleanup();
    self.parents.cleanup();

    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    if let Some(ui) = util::get_current_parent(self.parents.get())? {
      let current_index = if self.index.is_variable() {
        self.index.get().try_into()?
      } else {
        self.tmp
      };

      let mut new_index = None;

      let seq: Seq = input.try_into()?;

      ui.group(|ui| {
        for i in 0..seq.len() {
          if self.template.is_empty() {
            let str: &str = (&seq[i]).try_into()?;
            if ui
              .selectable_label(i as i64 == current_index, str.to_owned())
              .clicked()
            {
              new_index = Some(i as i64);
            }
          } else {
            let inner_margin = egui::style::Margin::same(3.0);
            let background_id = ui.painter().add(egui::Shape::Noop);
            let outer_rect = ui.available_rect_before_wrap();
            let mut inner_rect = outer_rect;
            inner_rect.min += inner_margin.left_top();
            inner_rect.max -= inner_margin.right_bottom();
            // Make sure we don't shrink to the negative:
            inner_rect.max.x = inner_rect.max.x.max(inner_rect.min.x);
            inner_rect.max.y = inner_rect.max.y.max(inner_rect.min.y);

            let mut content_ui = ui.child_ui_with_id_source(inner_rect, *ui.layout(), i);
            util::activate_ui_contents(
              context,
              &seq[i],
              &mut content_ui,
              &mut self.parents,
              &mut self.template,
            )?;

            let mut paint_rect = content_ui.min_rect();
            paint_rect.min -= inner_margin.left_top();
            paint_rect.max += inner_margin.right_bottom();

            let selected = i as i64 == current_index;
            let response = ui.allocate_rect(paint_rect, egui::Sense::click());
            let visuals = ui.style().interact_selectable(&response, selected);
            if selected || response.hovered() || response.has_focus() {
              let rect = paint_rect.expand(visuals.expansion);
              let shape = egui::Shape::Rect(egui::epaint::RectShape {
                rect,
                rounding: visuals.rounding,
                fill: visuals.bg_fill,
                stroke: visuals.bg_stroke,
              });
              ui.painter().set(background_id, shape);
            }

            if response.clicked() {
              new_index = Some(i as i64);
            }
          }
        }
        Ok::<(), &str>(())
      })
      .inner?;

      let current_index = if let Some(new_index) = new_index {
        new_index
      } else {
        current_index
      };

      if current_index >= seq.len() as i64 || current_index < 0 {
        // also fixup the index if it's out of bounds
        let fixup = -1i64;
        if self.index.is_variable() {
          self.index.set_fast_unsafe(&fixup.into());
        } else {
          self.tmp = fixup;
        }

        Ok(Var::default())
      } else {
        if self.index.is_variable() {
          self.index.set_fast_unsafe(&current_index.into());
        } else {
          self.tmp = current_index;
        }

        // this is fine because we don't own input and seq is just a view of it in this case
        Ok(seq[current_index as usize])
      }
    } else {
      Err("No UI parent")
    }
  }
}
