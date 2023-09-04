/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use shards::core::register_legacy_enum;
use shards::core::register_legacy_shard;
use shards::fourCharacterCode;
use shards::types::ExposedTypes;
use shards::types::ParamVar;
use shards::types::ShardsVar;
use shards::types::Type;

use shards::types::FRAG_CC;

struct Area {
  instance: ParamVar,
  requiring: ExposedTypes,
  position: ParamVar,
  anchor: ParamVar,
  contents: ShardsVar,
  parents: ParamVar,
  exposing: ExposedTypes,
}

shenum! {
  struct Anchor {
    [description("Top left corner.")]
    const TopLeft = 0x00;
    [description("Middle left.")]
    const Left = 0x10;
    [description("Bottom left corner.")]
    const BottomLeft = 0x20;
    [description("Top middle.")]
    const Top = 0x01;
    [description("Center.")]
    const Center = 0x11;
    [description("Bottom middle.")]
    const Bottom = 0x21;
    [description("Top right corner.")]
    const TopRight = 0x02;
    [description("Middle right.")]
    const Right = 0x12;
    [description("Bottom right corner.")]
    const BottomRight = 0x22;
  }

  struct AnchorInfo {}
}

shenum_types! {
  AnchorInfo,
  const AnchorCC = fourCharacterCode(*b"egAn");
  static ref AnchorEnumInfo;
  static ref ANCHOR_TYPE: Type;
  static ref ANCHOR_TYPES: Vec<Type>;
  static ref SEQ_OF_ANCHOR: Type;
  static ref SEQ_OF_ANCHOR_TYPES: Vec<Type>;
}

struct DockArea {
  instance: ParamVar,
  requiring: ExposedTypes,
  contents: ParamVar,
  parents: ParamVar,
  exposing: ExposedTypes,
  headers: Vec<ParamVar>,
  shards: Vec<ShardsVar>,
  tabs: egui_dock::Tree<(ParamVar, ShardsVar)>,
}

struct Scope {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  exposing: ExposedTypes,
}

struct Tab {
  parents: ParamVar,
  requiring: ExposedTypes,
  title: ParamVar,
  contents: ShardsVar,
  exposing: ExposedTypes,
}

/// Standalone window.
struct Window {
  instance: ParamVar,
  requiring: ExposedTypes,
  title: ParamVar,
  position: ParamVar,
  anchor: ParamVar,
  width: ParamVar,
  height: ParamVar,
  flags: ParamVar,
  contents: ShardsVar,
  parents: ParamVar,
  id: ParamVar,
  cached_id: Option<egui::Id>,
}

shenum! {
  struct WindowFlags {
    [description("Do not display the title bar.")]
    const NoTitleBar = 1 << 0;
    [description("Do not allow resizing the window.")]
    const NoResize = 1 << 1;
    [description("Display scrollbars.")]
    const Scrollbars = 1 << 2;
    [description("Do not display the collapse button.")]
    const NoCollapse = 1 << 3;
    [description("Do not allow window movement.")]
    const Immovable = 1 << 4;
  }
  struct WindowFlagsInfo {}
}

shenum_types! {
  WindowFlagsInfo,
  const WindowFlagsCC = fourCharacterCode(*b"egWF");
  static ref WindowFlagsEnumInfo;
  static ref WINDOW_FLAGS_TYPE: Type;
  static ref WINDOW_FLAGS_TYPES: Vec<Type>;
  static ref SEQ_OF_WINDOW_FLAGS: Type;
  static ref SEQ_OF_WINDOW_FLAGS_TYPES: Vec<Type>;
}

macro_rules! decl_panel {
  ($name:ident, $default_size:ident, $min_size:ident, $max_size:ident) => {
    struct $name {
      instance: ParamVar,
      requiring: ExposedTypes,
      resizable: ParamVar,
      $default_size: ParamVar,
      $min_size: ParamVar,
      $max_size: ParamVar,
      contents: ShardsVar,
      parents: ParamVar,
      exposing: ExposedTypes,
    }
  };
}

decl_panel!(BottomPanel, default_height, min_height, max_height);
decl_panel!(LeftPanel, default_width, min_width, max_width);
decl_panel!(RightPanel, default_width, min_width, max_width);
decl_panel!(TopPanel, default_height, min_height, max_height);

struct CentralPanel {
  instance: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  parents: ParamVar,
  exposing: ExposedTypes,
}

mod area;
mod docking;
mod panels;
mod scope;
mod window;

pub fn registerShards() {
  register_legacy_shard::<Area>();
  register_legacy_enum(FRAG_CC, AnchorCC, AnchorEnumInfo.as_ref().into());
  docking::registerShards();
  register_legacy_shard::<Scope>();
  register_legacy_shard::<Window>();
  register_legacy_enum(FRAG_CC, WindowFlagsCC, WindowFlagsEnumInfo.as_ref().into());
  register_legacy_shard::<BottomPanel>();
  register_legacy_shard::<CentralPanel>();
  register_legacy_shard::<LeftPanel>();
  register_legacy_shard::<RightPanel>();
  register_legacy_shard::<TopPanel>();

  assert_eq!(AnchorCC, 1701265774);
  assert_eq!(WindowFlagsCC, 1701271366);
}
