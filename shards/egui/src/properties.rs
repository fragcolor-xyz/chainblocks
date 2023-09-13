/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use crate::util;
use crate::CONTEXTS_NAME;
use crate::EGUI_CTX_TYPE;
use crate::HELP_VALUE_IGNORED;
use crate::PARENTS_UI_NAME;
use shards::core::register_enum;
use shards::core::register_legacy_enum;
use shards::core::register_legacy_shard;
use shards::core::register_shard;
use shards::fourCharacterCode;
use shards::shard::Shard;
use shards::types::common_type;
use shards::types::ClonedVar;
use shards::types::Context;
use shards::types::ExposedInfo;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::OptionalString;
use shards::types::ParamVar;
use shards::types::Parameters;
use shards::types::Type;
use shards::types::Types;
use shards::types::Var;
use shards::types::FRAG_CC;
use shards::types::NONE_TYPES;
use shards::SHType_Enum;

#[derive(shards::shards_enum)]
#[enum_info(
  b"uipt",
  "UIProperty",
  "Identifies UI properties to retrieve from the UI context"
)]
pub enum Property {
  #[enum_value("Return the remaining space within an UI widget.")]
  RemainingSpace = 0x0,
  #[enum_value("The screen size of the UI.")]
  ScreenSize = 0x1,
  #[enum_value("The amounts of pixels that correspond to 1 UI point.")]
  PixelsPerPoint = 0x2,
}

lazy_static! {
  static ref OUTPUT_TYPES: Types =
    vec![common_type::float4, common_type::float2, common_type::float];
}

#[derive(shards::shard)]
#[shard_info("UI.Property", "Retrieves values from the current state of the UI.")]
struct PropertyShard {
  #[shard_param("Property", "The property to retrieve from the UI context", [*PROPERTY_TYPE])]
  property: ClonedVar,
  #[shard_warmup]
  contexts: ParamVar,
  #[shard_warmup]
  parents: ParamVar,
  #[shard_required]
  required: ExposedTypes,
}

impl Default for PropertyShard {
  fn default() -> Self {
    Self {
      property: ClonedVar::default(),
      contexts: ParamVar::new_named(CONTEXTS_NAME),
      parents: ParamVar::new_named(PARENTS_UI_NAME),
      required: ExposedTypes::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for PropertyShard {
  fn input_types(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &OUTPUT_TYPES
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

    let (require_ui_parent) = match (&self.property.0).try_into()? {
      Property::RemainingSpace => true,
      _ => false,
    };

    if require_ui_parent {
      util::require_parents(&mut self.required);
    }

    util::require_context(&mut self.required);

    match (&self.property.0).try_into()? {
      Property::RemainingSpace => Ok(common_type::float4),
      Property::ScreenSize => Ok(common_type::float2),
      Property::PixelsPerPoint => Ok(common_type::float),
    }
  }

  fn activate(&mut self, _context: &Context, _input: &Var) -> Result<Var, &str> {
    let ui = util::get_current_parent_opt(self.parents.get()).map_err(|_| "No parent UI");

    match (&self.property.0).try_into()? {
      Property::PixelsPerPoint => {
        let ctx = util::get_current_context(&self.contexts)?;
        Ok(ctx.pixels_per_point().into())
      }
      Property::RemainingSpace => {
        let ui = ui?.ok_or("No parent UI")?;
        let target_size = ui.available_size();
        let target_pos = ui.next_widget_position().to_vec2();

        let draw_scale = ui.ctx().pixels_per_point();

        let min = target_pos * draw_scale;
        let max = (target_pos + target_size) * draw_scale;

        // Float4 rect as (X0, Y0, X1, Y1)
        let result_rect: Var = (min.x, min.y, max.x, max.y).into();
        Ok(result_rect)
      }
      Property::ScreenSize => {
        let ctx = util::get_current_context(&self.contexts)?;
        let size = ctx.screen_rect().size();
        Ok((size.x, size.y).into())
      }
    }
  }
}

pub fn register_shards() {
  register_shard::<PropertyShard>();
  register_enum::<Property>();
}
