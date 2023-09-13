/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use shards::core::register_legacy_shard;
use shards::types::ExposedTypes;
use shards::types::ParamVar;

struct Reset {
  parents: ParamVar,
  requiring: ExposedTypes,
}

struct AddFont {
  instance: ParamVar,
  requiring: ExposedTypes,
}

struct Style {
  instance: ParamVar,
  parents: ParamVar,
  requiring: ExposedTypes,
}

mod add_font;
mod reset;
mod style;
mod painter;
pub(crate) mod style_util;

pub fn register_shards() {
  register_legacy_shard::<Reset>();
  register_legacy_shard::<Style>();
  register_legacy_shard::<AddFont>();
  painter::register_shards();
}
