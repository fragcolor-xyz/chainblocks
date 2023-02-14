/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2023 Fragcolor Pte. Ltd. */

use super::WireViewer;
use crate::core::ShardInstance;
use crate::shard::Shard;
use crate::shards::editor::util::ColorUtils;
use crate::shards::editor::ShardData;
use crate::shards::editor::UIRenderer;
use crate::shards::gui::util;
use crate::shards::gui::HELP_OUTPUT_EQUAL_INPUT;
use crate::shards::gui::HELP_VALUE_IGNORED;
use crate::shards::gui::PARENTS_UI_NAME;
use crate::types::common_type;
use crate::types::Context;
use crate::types::ExposedTypes;
use crate::types::OptionalString;
use crate::types::ParamVar;
use crate::types::Parameters;
use crate::types::Type;
use crate::types::Types;
use crate::types::Var;
use crate::types::WireRef;
use crate::types::ANY_TYPES;

lazy_static! {
  static ref WIRE_OR_VAR_TYPES: Vec<Type> = vec![common_type::wire, common_type::wire_var];
  static ref VIEWER_PARAMETERS: Parameters =
    vec![(cstr!("Wire"), shccstr!("TODO"), &WIRE_OR_VAR_TYPES[..]).into(),];
}

impl<'a> Default for WireViewer<'a> {
  fn default() -> Self {
    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);
    Self {
      wire: ParamVar::default(),
      parents,
      requiring: Vec::new(),
      graph: Default::default(),
      node_order: Vec::new(),
    }
  }
}

impl<'a> Shard for WireViewer<'a> {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("WireViewer")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("WireViewer-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "WireViewer"
  }

  fn inputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn inputHelp(&mut self) -> OptionalString {
    *HELP_VALUE_IGNORED
  }

  fn outputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn outputHelp(&mut self) -> OptionalString {
    *HELP_OUTPUT_EQUAL_INPUT
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&VIEWER_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.wire.set_param(value)),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.wire.get_param(),
      _ => Var::default(),
    }
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    // Add UI.Parents to the list of required variables
    util::require_parents(&mut self.requiring, &self.parents);

    Some(&self.requiring)
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.wire.warmup(ctx);
    self.parents.warmup(ctx);

    // FIXME we should recompute the data if the wire changed (or not)
    self.compute_data(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.graph.clear();

    self.wire.cleanup();
    self.parents.cleanup();

    Ok(())
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Var, &str> {
    if let Some(ui) = util::get_current_parent(self.parents.get())? {
      for node_id in self.node_order.iter().copied() {
        ui.push_id(node_id, |ui| {
          use egui::epaint::*;
          use egui::style::*;
          use egui::*;

          let margin = Margin::symmetric(15.0, 5.0);
          let (background_color, text_color) = if ui.visuals().dark_mode {
            (
              Color32::from_hex("#3f3f3f").unwrap(),
              Color32::from_hex("#fefefe").unwrap(),
            )
          } else {
            (
              Color32::from_hex("#ffffff").unwrap(),
              Color32::from_hex("#505050").unwrap(),
            )
          };
          ui.visuals_mut().widgets.noninteractive.fg_stroke = Stroke::new(2.0, text_color);

          // Preallocate shapes to paint below contents
          let background_shape = ui.painter().add(Shape::Noop);

          let mut title_height = 0.0;

          let node = &mut self.graph[node_id];
          Frame::none().inner_margin(margin).show(ui, |ui| {
            ui.vertical(|ui| {
              ui.horizontal(|ui| {
                ui.add(Label::new(
                  RichText::new(&node.label)
                    .text_style(TextStyle::Button)
                    .color(text_color),
                ));
                ui.add_space(8.0); // The size of the little cross icon
              });
              title_height = ui.min_size().y;
              ui.add_space(margin.top);

              // First pass: Draw the inner fields. Compute port heights
              node.data.ui(ui);
            })
          });

          // Second pass, iterate again to draw the ports. This happens outside
          // the child_ui because we want ports to overflow the node background.
          let outer_rect = ui.min_rect();

          // Draw the background shape.
          // NOTE: This code is a bit more involved than it needs to be because egui
          // does not support drawing rectangles with asymmetrical round corners.
          let shape = {
            let rounding_radius = 4.0;

            let titlebar_height = title_height + margin.top * 2.0;
            let titlebar_rect =
              Rect::from_min_size(outer_rect.min, vec2(outer_rect.width(), titlebar_height));
            let titlebar = Shape::Rect(RectShape {
              rect: titlebar_rect,
              rounding: Rounding {
                nw: rounding_radius,
                ne: rounding_radius,
                sw: 0.0,
                se: 0.0,
              },
              fill: background_color.lighten(0.8),
              stroke: Stroke::NONE,
            });

            let body_rect = Rect::from_min_size(
              outer_rect.min + vec2(0.0, titlebar_height),
              vec2(outer_rect.width(), outer_rect.height() - titlebar_height),
            );
            let body = Shape::Rect(RectShape {
              rect: body_rect,
              rounding: Rounding {
                nw: 0.0,
                ne: 0.0,
                sw: rounding_radius,
                se: rounding_radius,
              },
              fill: background_color,
              stroke: Stroke::NONE,
            });

            Shape::Vec(vec![titlebar, body])
          };

          ui.painter().set(background_shape, shape);
        });
      }

      // Always passthrough the input
      Ok(*input)
    } else {
      Err("No UI parent")
    }
  }
}

impl<'a> WireViewer<'a> {
  fn compute_data(&mut self, _ctx: &Context) -> Result<(), &str> {
    let wire: WireRef = self.wire.get().try_into()?;
    let info = wire.get_wire_info();

    let shards = info.shards;
    let shards: Vec<ShardInstance> =
      unsafe { core::slice::from_raw_parts(shards.elements, shards.len as usize) }
        .iter()
        .map(|ptr| (*ptr).into())
        .collect();

    for s in shards.into_iter() {
      let data: ShardData = s.into();
      let node_id = self.graph.add_node(data.name.to_string(), data);
      self.node_order.push(node_id);
    }

    Ok(())
  }
}
