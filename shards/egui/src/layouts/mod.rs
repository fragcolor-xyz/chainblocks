/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use shards::core::registerEnumType;
use shards::core::registerShard;
use shards::fourCharacterCode;
use shards::types::ClonedVar;
use shards::types::ExposedTypes;
use shards::types::ParamVar;
use shards::types::ShardsVar;
use shards::types::Type;
use shards::types::FRAG_CC;
use std::rc::Rc;

struct CollapsingHeader {
  parents: ParamVar,
  requiring: ExposedTypes,
  text: ParamVar,
  header: ShardsVar,
  contents: ShardsVar,
  defaultOpen: ParamVar,
  exposing: ExposedTypes,
}

struct Columns {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ParamVar,
  shards: Vec<ShardsVar>,
  exposing: ExposedTypes,
}

struct Disable {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  disable: ParamVar,
  exposing: ExposedTypes,
}

struct Frame {
  parents: ParamVar,
  requiring: ExposedTypes,
  innerMargin: ParamVar,
  outerMargin: ParamVar,
  rounding: ParamVar,
  fillColor: ParamVar,
  strokeColor: ParamVar,
  strokeWidth: ParamVar,
  contents: ShardsVar,
  exposing: ExposedTypes,
}

struct Grid {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  striped: ParamVar,
  min_width: ParamVar,
  max_width: ParamVar,
  spacing: ParamVar,
  exposing: ExposedTypes,
}

struct Group {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  exposing: ExposedTypes,
}

struct Horizontal {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  wrap: bool,
  centered: bool,
  exposing: ExposedTypes,
}

#[derive(Clone)]
struct EguiScrollAreaSettings {
  horizontal_scroll_enabled: bool,
  vertical_scroll_enabled: bool,
  max_width: f32,
  max_height: f32,
  scroll_visibility: ScrollVisibility,
}

impl EguiScrollAreaSettings {
  pub fn to_egui_scrollarea(&self) -> egui::ScrollArea {
    egui::ScrollArea::new([self.horizontal_scroll_enabled, self.vertical_scroll_enabled]).scroll_bar_visibility(self.scroll_visibility.into()).max_width(self.max_width).max_height(self.max_height)
  }
}

struct LayoutClass {
  parent: *const LayoutClass,
  layout: Option<egui::Layout>,
  min_size: Option<(f32, f32)>,
  fill_width: Option<bool>,
  fill_height: Option<bool>,
  disabled: Option<bool>,
  frame: Option<LayoutFrame>,
  scroll_area: Option<EguiScrollAreaSettings>,
}

struct LayoutConstructor {
  parent: ParamVar,
  layout_class: Option<Rc<LayoutClass>>,
  layout: ParamVar,
  min_size: ParamVar,
  fill_width: ParamVar,
  fill_height: ParamVar,
  disabled: ParamVar,
  frame: ParamVar,
  scroll_area: ParamVar,
}

struct Layout {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  layout_class: ParamVar,
  min_size: ParamVar,
  fill_width: ParamVar,
  fill_height: ParamVar,
  exposing: ExposedTypes,
}

shenum! {
  pub struct ScrollVisibility {
    [description("The scroll bars will always be visible.")]
    const AlwaysVisible = 1 << 0;
    [description("The scroll bars will only be visible when needed")]
    const VisibleWhenNeeded = 1 << 1;
    [description("The scroll bars will always be hidden.")]
    const AlwaysHidden = 1 << 2;
  }
  pub struct ScrollVisibilityInfo {}
}

impl From<ScrollVisibility> for egui::scroll_area::ScrollBarVisibility {
  fn from(value: ScrollVisibility) -> Self {
    match value {
      ScrollVisibility::AlwaysVisible => egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
      ScrollVisibility::VisibleWhenNeeded => egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
      ScrollVisibility::AlwaysHidden => egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
      _ => unreachable!(),
    }
  }
}

shenum_types! {
  ScrollVisibilityInfo,
  const ScrollVisibilityCC = fourCharacterCode(*b"egSV");
  pub static ref ScrollVisibilityEnumInfo;
  pub static ref SCROLL_VISIBILITY_TYPE: Type;
  pub static ref SCROLL_VISIBILITY_TYPES: Vec<Type>;
  pub static ref SEQ_OF_SCROLL_VISIBILITY: Type;
  pub static ref SEQ_OF_SCROLL_VISIBILITY_TYPES: Vec<Type>;
}

shenum! {
  pub struct LayoutFrame {
  [description("Use the frame styling for grouping widgets together.")]
  const Widgets = 1 << 0;
  [description("Use the frame styling for a side top panel.")]
  const SideTopPanel = 1 << 1;
  [description("Use the frame styling for the central panel.")]
  const CentralPanel = 1 << 2;
  [description("Use the frame styling for a window.")]
  const Window = 1 << 3;
  [description("Use the frame styling for a menu.")]
  const Menu = 1 << 4;
  [description("Use the frame styling for a popup.")]
  const Popup = 1 << 5;
  [description("Use the frame styling for a canvas to draw on.")]
  const Canvas = 1 << 6;
  [description("Use the frame styling for a dark canvas to draw on.")]
  const DarkCanvas = 1 << 7;
  }
  pub struct LayoutFrameInfo {}
}

shenum_types! {
  LayoutFrameInfo,
  const LayoutFrameCC = fourCharacterCode(*b"egLF");
  pub static ref LayoutFrameEnumInfo;
  pub static ref LAYOUT_FRAME_TYPE: Type;
  pub static ref LAYOUT_FRAME_TYPES: Vec<Type>;
  pub static ref SEQ_OF_LAYOUT_FRAME: Type;
  pub static ref SEQ_OF_LAYOUT_FRAME_TYPES: Vec<Type>;
}

shenum! {
  pub struct LayoutDirection {
    [description("Describes a horizontal layout where its contents are arranged from the left to the right.")]
    const LeftToRight = 1 << 0;
    [description("Describes a horizontal layout where its contents are arranged from the right to the left.")]
    const RightToLeft = 1 << 1;
    [description("Describes a vertical layout where its contents are arranged from the top to the bottom.")]
    const TopDown = 1 << 2;
    [description("Describes a vertical layout where its contents are arranged from the bottom to the top.")]
    const BottomUp = 1 << 3;
  }
  pub struct LayoutDirectionInfo {}
}

impl From<LayoutDirection> for egui::Direction {
  fn from(value: LayoutDirection) -> Self {
    match value {
      LayoutDirection::LeftToRight => egui::Direction::LeftToRight,
      LayoutDirection::RightToLeft => egui::Direction::RightToLeft,
      LayoutDirection::TopDown => egui::Direction::TopDown,
      LayoutDirection::BottomUp => egui::Direction::BottomUp,
      _ => unreachable!(),
    }
  }
}

shenum_types! {
  LayoutDirectionInfo,
  const LayoutDirectionCC = fourCharacterCode(*b"egLD");
  pub static ref LayoutDirectionEnumInfo;
  pub static ref LAYOUT_DIRECTION_TYPE: Type;
  pub static ref LAYOUT_DIRECTION_TYPES: Vec<Type>;
  pub static ref SEQ_OF_LAYOUT_DIRECTION: Type;
  pub static ref SEQ_OF_LAYOUT_DIRECTION_TYPES: Vec<Type>;
}

shenum! {
  pub struct LayoutAlign {
  [description("Left or top alignment for e.g. anchors and layouts.")]
  const Min = 1 << 0;
  [description("Left alignment for e.g. anchors and layouts.")]
  const Left = 1 << 1;
  [description("Top alignment for e.g. anchors and layouts.")]
  const Top = 1 << 2;
  [description("Horizontal or vertical center alignment for e.g. anchors and layouts.")]
  const Center = 1 << 3;
  [description("Right or bottom center alignment for e.g. anchors and layouts.")]
  const Max = 1 << 4;
  [description("Right alignment for e.g. anchors and layouts.")]
  const Right = 1 << 5;
  [description("Bottom center alignment for e.g. anchors and layouts.")]
  const Bottom = 1 << 6;
  }
  pub struct LayoutAlignInfo {}
}

impl From<LayoutAlign> for egui::Align {
  fn from(value: LayoutAlign) -> Self {
    match value {
      LayoutAlign::Min => egui::Align::Min,
      LayoutAlign::Left => egui::Align::Min,
      LayoutAlign::Top => egui::Align::Min,
      LayoutAlign::Center => egui::Align::Center,
      LayoutAlign::Max => egui::Align::Max,
      LayoutAlign::Right => egui::Align::Max,
      LayoutAlign::Bottom => egui::Align::Max,
      _ => unreachable!(),
    }
  }
}

shenum_types! {
  LayoutAlignInfo,
  const LayoutAlignCC = fourCharacterCode(*b"egLA");
  pub static ref LayoutAlignEnumInfo;
  pub static ref LAYOUT_ALIGN_TYPE: Type;
  pub static ref LAYOUT_ALIGN_TYPES: Vec<Type>;
  pub static ref SEQ_OF_LAYOUT_ALIGN: Type;
  pub static ref SEQ_OF_LAYOUT_ALIGN_TYPES: Vec<Type>;
}

struct Indent {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  exposing: ExposedTypes,
}

struct NextRow {
  parents: ParamVar,
  requiring: ExposedTypes,
}

struct ScrollArea {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  horizontal: ParamVar,
  vertical: ParamVar,
  alwaysShow: ParamVar,
  exposing: ExposedTypes,
}

struct Separator {
  parents: ParamVar,
  requiring: ExposedTypes,
}

struct Space {
  parents: ParamVar,
  requiring: ExposedTypes,
  amount: ParamVar,
}

struct Table {
  parents: ParamVar,
  requiring: ExposedTypes,
  rows: ClonedVar,
  columns: ClonedVar,
  striped: ParamVar,
  resizable: ParamVar,
  row_index: ParamVar,
  shards: Vec<ShardsVar>,
  header_shards: Vec<Option<ShardsVar>>,
  exposing: ExposedTypes,
}

struct Vertical {
  parents: ParamVar,
  requiring: ExposedTypes,
  contents: ShardsVar,
  centered: bool,
  exposing: ExposedTypes,
}

struct Sized {
  parents: ParamVar,
  contents: ShardsVar,
  requiring: ExposedTypes,
  exposing: ExposedTypes,
  width: ClonedVar,
  height: ClonedVar,
  fill_width: ClonedVar,
  fill_height: ClonedVar,
}

mod collapsing_header;
mod columns;
mod disable;
mod frame;
mod grid;
mod group;
mod horizontal;
mod indent;
mod layout;
mod scroll_area;
mod separator;
mod sized;
mod space;
mod table;
mod vertical;

pub fn registerShards() {
  registerShard::<CollapsingHeader>();
  registerShard::<Columns>();
  registerShard::<Disable>();
  registerShard::<Frame>();
  registerShard::<Grid>();
  registerShard::<Group>();
  registerShard::<Horizontal>();
  registerShard::<LayoutConstructor>();
  registerShard::<Layout>();
  registerEnumType(
    FRAG_CC,
    ScrollVisibilityCC,
    ScrollVisibilityEnumInfo.as_ref().into(),
  );
  registerEnumType(
    FRAG_CC,
    LayoutDirectionCC,
    LayoutDirectionEnumInfo.as_ref().into(),
  );
  registerEnumType(
    FRAG_CC,
    LayoutFrameCC,
    LayoutFrameEnumInfo.as_ref().into(),
  );
  registerEnumType(FRAG_CC, LayoutAlignCC, LayoutAlignEnumInfo.as_ref().into());
  registerShard::<Indent>();
  registerShard::<NextRow>();
  registerShard::<ScrollArea>();
  registerShard::<Separator>();
  registerShard::<Space>();
  registerShard::<Table>();
  registerShard::<Vertical>();
  registerShard::<Sized>();
}
