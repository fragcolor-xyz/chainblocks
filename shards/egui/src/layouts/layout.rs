/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use super::EguiScrollAreaSettings;
use super::Layout;
use super::LayoutAlign;
use super::LayoutAlignCC;
use super::LayoutClass;
use super::LayoutConstructor;
use super::LayoutDirection;
use super::LayoutDirectionCC;
use super::LayoutFrame;
use super::LayoutFrameCC;
use super::ScrollVisibility;
use crate::layouts::ScrollVisibilityCC;
use crate::layouts::LAYOUT_ALIGN_TYPES;
use crate::layouts::LAYOUT_DIRECTION_TYPES;
use crate::layouts::SCROLL_VISIBILITY_TYPES;
use crate::util;
use crate::EguiId;
use crate::EGUI_UI_TYPE;
use crate::FLOAT2_VAR_SLICE;
use crate::FLOAT_VAR_SLICE;
use crate::HELP_OUTPUT_EQUAL_INPUT;
use crate::LAYOUTCLASS_TYPE;
use crate::LAYOUTCLASS_TYPE_VEC;
use crate::LAYOUTCLASS_TYPE_VEC_VAR;
use crate::LAYOUT_FRAME_TYPE_VEC;
use crate::PARENTS_UI_NAME;
use shards::shard::LegacyShard;
use shards::shard::Shard;
use shards::types::Context;
use shards::types::ExposedInfo;
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
use shards::types::BOOL_VAR_OR_NONE_SLICE;
use shards::types::SHARDS_OR_NONE_TYPES;
use std::rc::Rc;

macro_rules! retrieve_parameter {
  ($param:ident, $name:literal, $type:ty) => {
    if !$param.is_none() {
      let value: $type = $param.try_into().map_err(|e| {
        shlog!("{}: {}", $name, e);
        "Invalid attribute value received"
      })?;
      Some(value)
    } else {
      None
    }
  };
}

macro_rules! retrieve_enum_parameter {
  ($param:ident, $name:literal, $type:ident, $typeId:expr) => {
    // Check if parameter has been passed in, and if so retrieve it
    if !$param.is_none() {
      let value = $param;
      // Check for correct enum type
      if value.valueType == crate::shardsc::SHType_Enum
        && unsafe { value.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId == $typeId }
      {
        // Retrieve value
        let bits = unsafe { value.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue };
        let value = $type { bits };
        Some(value)
      } else {
        Err("Expected Enum value of same Enum type.").map_err(|e| {
          shlog!("{}: {}", $name, e);
          "Invalid attribute value received"
        })?
      }
    } else {
      // No parameter to retrieve from, caller should retrieve from parent or use default value
      None
    }
  };
}

macro_rules! retrieve_layout_class_attribute {
  ($layout_class:ident, $attribute:ident) => {{
    let mut parent = $layout_class as *const LayoutClass;
    let mut result = None;
    while !parent.is_null() {
      unsafe {
        if let Some(attribute) = &(*parent).$attribute {
          // found the attribute, can return now
          result = Some(attribute.clone());
          break;
        } else {
          // attribute not found in parent, continue looking through parents
          parent = (*parent).parent;
        }
      }
    }
    result
  }};
  ($layout_class:ident, $attribute:ident, $member:ident) => {{
    let mut parent = $layout_class as *const LayoutClass;
    let mut result = None;
    while !parent.is_null() {
      unsafe {
        if let Some(attribute) = &(*parent).$attribute {
          // found the member attribute, can return now
          result = Some(attribute.$member.clone());
          break;
        } else {
          // attribute not found in parent, continue looking through parents
          parent = (*parent).parent;
        }
      }
    }
    result
  }};
}

lazy_static! {
  static ref LAYOUT_CONSTRUCTOR_PARAMETERS: Parameters = vec![
    (
      cstr!("Parent"),
      shccstr!("The parent Layout class to inherit parameters from."),
      &LAYOUTCLASS_TYPE_VEC_VAR[..],
    )
      .into(),
    (
      cstr!("MainDirection"),
      shccstr!("The main direction of the UI element layout."),
      &LAYOUT_DIRECTION_TYPES[..],
    )
      .into(),
    (
      cstr!("MainWrap"),
      shccstr!("Whether the UI elements in the layout should wrap around when reaching the end of the direction of the cursor."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("MainAlign"),
      shccstr!("How the UI elements in the layout should be aligned on the main axis."),
      &LAYOUT_ALIGN_TYPES[..],
    )
      .into(),
    (
      cstr!("MainJustify"),
      shccstr!("Whether the UI elements in the layout should be justified along the main axis."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("CrossAlign"),
      shccstr!("How the UI elements in the layout should be aligned on the cross axis."),
      &LAYOUT_ALIGN_TYPES[..],
    )
      .into(),
    (
      cstr!("CrossJustify"),
      shccstr!("Whether the UI elements in the layout should be justified along the across axis."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("MinSize"),
      shccstr!("The minimum size of the space to be reserved by this UI for its contents. This allows the UI to take up more space than required for its widget contents. Can be overidden by FillWidth and FillHeight."),
      FLOAT2_VAR_SLICE,
    )
      .into(),
    (
      cstr!("MaxSize"),
      shccstr!("The maximum size of the space to be reserved by this UI for its contents. Prevents UI from taking as much space as possible. Can be overidden by FillWidth and FillHeight."),
      FLOAT2_VAR_SLICE,
    )
      .into(),
    (cstr!("FillWidth"), shccstr!("Whether the Layout should take up the full width of the available space."), &BOOL_TYPES[..],).into(),
    (cstr!("FillHeight"), shccstr!("Whether the Layout should take up the full height of the available space."), &BOOL_TYPES[..],).into(),
    (
      cstr!("Disabled"),
      shccstr!("Whether the drawn layout should be disabled or not."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("Frame"),
      shccstr!("The frame to be drawn around the layout."),
      &LAYOUT_FRAME_TYPE_VEC[..],
    )
      .into(),
    (
      cstr!("EnableHorizontalScrollBar"),
      shccstr!("Enable the horizontal scroll bar. If either this or EnableVerticalScrollBar is true, a ScrollArea will be created within the layout."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("EnableVerticalScrollBar"),
      shccstr!("Enable the vertical scroll bar. If either this or EnableHorizontalScrollBar is true, a ScrollArea will be created within the layout."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("ScrollBarVisibility"),
      shccstr!("Whether the scroll bars of the scroll area should be AlwaysVisible, VisibleWhenNeeded, or Always Hidden. Default: AlwaysVisible"),
      &SCROLL_VISIBILITY_TYPES[..],
    )
      .into(),
    (
      cstr!("ScrollAreaMinWidth"),
      shccstr!("The minimum width of the scroll area to be drawn. Note: This is not the minimum width of the contents of the scroll area."),
      FLOAT_VAR_SLICE,
    )
      .into(),
    (
      cstr!("ScrollAreaMinHeight"),
      shccstr!("The minimum height of the scroll area to be drawn. Note: This is not the minimum height of the contents of the scroll area."),
      FLOAT_VAR_SLICE,
    )
      .into(),
    (
      cstr!("ScrollAreaMaxWidth"),
      shccstr!("The maximum width of the scroll area to be drawn. Note: This is not the maximum width of the contents of the scroll area."),
      FLOAT_VAR_SLICE,
    )
      .into(),
    (
      cstr!("ScrollAreaMaxHeight"),
      shccstr!("The maximum height of the scroll area to be drawn. Note: This is not the maximum height of the contents of the scroll area."),
      FLOAT_VAR_SLICE,
    )
      .into(),
    (
      cstr!("ScrollAreaAutoShrinkWidth"),
      shccstr!("Whether the scroll area's width should automatically shrink to fit the size of its contents."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("ScrollAreaAutoShrinkHeight"),
      shccstr!("Whether the scroll area's height should automatically shrink to fit the size of its contents."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
    (
      cstr!("ScrollAreaEnableScrolling"),
      shccstr!("Whether the scroll area's scrolling should be enabled. This is akin to the disable setting for UI elements."),
      BOOL_VAR_OR_NONE_SLICE,
    )
      .into(),
  ];

  static ref LAYOUT_PARAMETERS: Parameters = vec![
    (
      cstr!("Contents"),
      shccstr!("The UI contents."),
      &SHARDS_OR_NONE_TYPES[..],
    )
      .into(),
    (
      cstr!("Class"),
      shccstr!("The Layout class describing all of the options relating to the layout of this UI."),
      &LAYOUTCLASS_TYPE_VEC_VAR[..],
    )
    .into(),
    (
      cstr!("MinSize"),
      shccstr!("The minimum size of the space to be reserved by this UI. This allows the UI to take up more space than required for its widget contents. Can be overidden by FillWidth and FillHeight."),
      FLOAT2_VAR_SLICE,
    )
      .into(),
    (
      cstr!("MaxSize"),
      shccstr!("The maximum size of the space to be reserved by this UI. Prevents UI from taking as much space as possible. Can be overidden by FillWidth and FillHeight."),
      FLOAT2_VAR_SLICE,
    )
      .into(),
    (cstr!("FillWidth"), shccstr!("Whether the Layout should take up the full width of the available space."), &BOOL_TYPES[..],).into(),
    (cstr!("FillHeight"), shccstr!("Whether the Layout should take up the full height of the available space."), &BOOL_TYPES[..],).into(),
  ];
}

impl Default for LayoutConstructor {
  fn default() -> Self {
    Self {
      parent: ParamVar::default(),
      layout_class: None,
      main_direction: ParamVar::default(),
      main_wrap: ParamVar::default(),
      main_align: ParamVar::default(),
      main_justify: ParamVar::default(),
      cross_align: ParamVar::default(),
      cross_justify: ParamVar::default(),
      min_size: ParamVar::default(),
      max_size: ParamVar::default(),
      fill_width: ParamVar::default(),
      fill_height: ParamVar::default(),
      disabled: ParamVar::default(),
      frame: ParamVar::default(),
      enable_horizontal_scroll_bar: ParamVar::default(),
      enable_vertical_scroll_bar: ParamVar::default(),
      scroll_visibility: ParamVar::default(),
      scroll_area_min_width: ParamVar::default(),
      scroll_area_min_height: ParamVar::default(),
      scroll_area_max_width: ParamVar::default(),
      scroll_area_max_height: ParamVar::default(),
      scroll_area_auto_shrink_width: ParamVar::default(),
      scroll_area_auto_shrink_height: ParamVar::default(),
      scroll_area_enable_scrolling: ParamVar::default(),
    }
  }
}

impl LegacyShard for LayoutConstructor {
  fn registerName() -> &'static str {
    cstr!("UI.LayoutClass")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("UI.LayoutClass-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.LayoutClass"
  }

  fn inputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &LAYOUTCLASS_TYPE_VEC
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&LAYOUT_CONSTRUCTOR_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &'static str> {
    match index {
      0 => self.parent.set_param(value),
      1 => self.main_direction.set_param(value),
      2 => self.main_wrap.set_param(value),
      3 => self.main_align.set_param(value),
      4 => self.main_justify.set_param(value),
      5 => self.cross_align.set_param(value),
      6 => self.cross_justify.set_param(value),
      7 => self.min_size.set_param(value),
      8 => self.max_size.set_param(value),
      9 => self.fill_width.set_param(value),
      10 => self.fill_height.set_param(value),
      11 => self.disabled.set_param(value),
      12 => self.frame.set_param(value),
      13 => self.enable_horizontal_scroll_bar.set_param(value),
      14 => self.enable_vertical_scroll_bar.set_param(value),
      15 => self.scroll_visibility.set_param(value),
      16 => self.scroll_area_min_width.set_param(value),
      17 => self.scroll_area_min_height.set_param(value),
      18 => self.scroll_area_max_width.set_param(value),
      19 => self.scroll_area_max_height.set_param(value),
      20 => self.scroll_area_auto_shrink_width.set_param(value),
      21 => self.scroll_area_auto_shrink_height.set_param(value),
      22 => self.scroll_area_enable_scrolling.set_param(value),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.parent.get_param(),
      1 => self.main_direction.get_param(),
      2 => self.main_wrap.get_param(),
      3 => self.main_align.get_param(),
      4 => self.main_justify.get_param(),
      5 => self.cross_align.get_param(),
      6 => self.cross_justify.get_param(),
      7 => self.min_size.get_param(),
      8 => self.max_size.get_param(),
      9 => self.fill_width.get_param(),
      10 => self.fill_height.get_param(),
      11 => self.disabled.get_param(),
      12 => self.frame.get_param(),
      13 => self.enable_horizontal_scroll_bar.get_param(),
      14 => self.enable_vertical_scroll_bar.get_param(),
      15 => self.scroll_visibility.get_param(),
      16 => self.scroll_area_min_width.get_param(),
      17 => self.scroll_area_min_height.get_param(),
      18 => self.scroll_area_max_width.get_param(),
      19 => self.scroll_area_max_height.get_param(),
      20 => self.scroll_area_auto_shrink_width.get_param(),
      21 => self.scroll_area_auto_shrink_height.get_param(),
      22 => self.scroll_area_enable_scrolling.get_param(),
      _ => Var::default(),
    }
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.parent.warmup(ctx);
    self.main_direction.warmup(ctx);
    self.main_wrap.warmup(ctx);
    self.main_align.warmup(ctx);
    self.main_justify.warmup(ctx);
    self.cross_align.warmup(ctx);
    self.cross_justify.warmup(ctx);
    self.min_size.warmup(ctx);
    self.max_size.warmup(ctx);
    self.fill_width.warmup(ctx);
    self.fill_height.warmup(ctx);
    self.disabled.warmup(ctx);
    self.frame.warmup(ctx);
    self.enable_horizontal_scroll_bar.warmup(ctx);
    self.enable_vertical_scroll_bar.warmup(ctx);
    self.scroll_visibility.warmup(ctx);
    self.scroll_area_min_width.warmup(ctx);
    self.scroll_area_min_height.warmup(ctx);
    self.scroll_area_max_width.warmup(ctx);
    self.scroll_area_max_height.warmup(ctx);
    self.scroll_area_auto_shrink_width.warmup(ctx);
    self.scroll_area_auto_shrink_height.warmup(ctx);
    self.scroll_area_enable_scrolling.warmup(ctx);

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.scroll_area_enable_scrolling.cleanup();
    self.scroll_area_auto_shrink_height.cleanup();
    self.scroll_area_auto_shrink_width.cleanup();
    self.scroll_area_max_height.cleanup();
    self.scroll_area_max_width.cleanup();
    self.scroll_area_min_height.cleanup();
    self.scroll_area_min_width.cleanup();
    self.scroll_visibility.cleanup();
    self.enable_vertical_scroll_bar.cleanup();
    self.enable_horizontal_scroll_bar.cleanup();
    self.frame.cleanup();
    self.disabled.cleanup();
    self.fill_height.cleanup();
    self.fill_width.cleanup();
    self.max_size.cleanup();
    self.min_size.cleanup();
    self.cross_justify.cleanup();
    self.cross_align.cleanup();
    self.main_justify.cleanup();
    self.main_align.cleanup();
    self.main_wrap.cleanup();
    self.main_direction.cleanup();
    self.parent.cleanup();

    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    let mut has_parent = false;

    let parent_layout_class = if !self.parent.get().is_none() {
      let parent_layout_class: Option<Rc<LayoutClass>> = Some(Var::from_object_as_clone(
        self.parent.get(),
        &LAYOUTCLASS_TYPE,
      )?);
      let parent_layout_class = unsafe {
        let parent_layout_ptr =
          Rc::as_ptr(parent_layout_class.as_ref().unwrap()) as *mut LayoutClass;
        &*parent_layout_ptr
      };

      has_parent = true;
      Some(parent_layout_class)
    } else {
      None
    };

    // Track if a new layout has to be created. This is in the case that the layout class is new or there has been an overwrite in its values
    let mut create_new_layout = false;

    let cross_align = self.cross_align.get();
    let cross_align = if let Some(cross_align) =
      retrieve_enum_parameter!(cross_align, "cross_align", LayoutAlign, LayoutAlignCC)
    {
      create_new_layout = true;
      cross_align.into()
    } else {
      // if there is a parent, retrieve the parent's value over the default
      if let Some(parent_layout_class) = parent_layout_class {
        // this is guaranteed to be Some because the parent has already been constructed
        retrieve_layout_class_attribute!(parent_layout_class, layout, cross_align).unwrap()
      } else {
        create_new_layout = true;
        egui::Align::Min // default cross aligns
      }
    };

    let main_direction = self.main_direction.get();
    let main_direction = if let Some(main_direction) = retrieve_enum_parameter!(
      main_direction,
      "main_direction",
      LayoutDirection,
      LayoutDirectionCC
    ) {
      create_new_layout = true;
      main_direction.into()
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, layout, main_dir).unwrap()
      } else {
        return Err("Invalid main direction provided. Main direction is a required parameter");
      }
    };

    let main_align = self.main_align.get();
    let main_align = if let Some(main_align) =
      retrieve_enum_parameter!(main_align, "main_align", LayoutAlign, LayoutAlignCC)
    {
      create_new_layout = true;
      main_align.into()
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, layout, main_align).unwrap()
      } else {
        create_new_layout = true;
        egui::Align::Center // default main align
      }
    };

    let main_wrap = self.main_wrap.get();
    let main_wrap = if let Some(main_wrap) = retrieve_parameter!(main_wrap, "main_wrap", bool) {
      create_new_layout = true;
      main_wrap
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, layout, main_wrap).unwrap()
      } else {
        create_new_layout = true;
        false // default main wrap
      }
    };

    let main_justify = self.main_justify.get();
    let main_justify =
      if let Some(main_justify) = retrieve_parameter!(main_justify, "main_justify", bool) {
        create_new_layout = true;
        main_justify
      } else {
        if let Some(parent_layout_class) = parent_layout_class {
          retrieve_layout_class_attribute!(parent_layout_class, layout, main_justify).unwrap()
        } else {
          create_new_layout = true;
          false // default main justify
        }
      };

    let cross_justify = self.cross_justify.get();
    let cross_justify =
      if let Some(cross_justify) = retrieve_parameter!(cross_justify, "cross_justify", bool) {
        create_new_layout = true;
        cross_justify
      } else {
        if let Some(parent_layout_class) = parent_layout_class {
          retrieve_layout_class_attribute!(parent_layout_class, layout, cross_justify).unwrap()
        } else {
          create_new_layout = true;
          false // default cross justify
        }
      };

    let mut layout = if create_new_layout {
      Some(
        egui::Layout::from_main_dir_and_cross_align(main_direction, cross_align)
          .with_main_align(main_align)
          .with_main_wrap(main_wrap)
          .with_main_justify(main_justify)
          .with_cross_justify(cross_justify),
      )
    } else {
      if has_parent {
        None
      } else {
        return Err("Invalid Layout provided. Layout is a required parameter when there is no parent LayoutClass provided");
      }
    };

    let min_size = if !self.min_size.get().is_none() {
      Some(self.min_size.get().try_into()?)
    } else {
      None
    };

    let max_size = if !self.max_size.get().is_none() {
      Some(self.max_size.get().try_into()?)
    } else {
      None
    };

    let fill_width: Option<bool> = if !self.fill_width.get().is_none() {
      Some(self.fill_width.get().try_into()?)
    } else {
      None // default fill width
    };
    let fill_height: Option<bool> = if !self.fill_height.get().is_none() {
      Some(self.fill_height.get().try_into()?)
    } else {
      None // default fill height
    };

    let disabled = if !self.disabled.get().is_none() {
      Some(self.disabled.get().try_into()?)
    } else {
      None // default value should be interpreted as false later on
    };

    let frame = if !self.frame.get().is_none() {
      let frame = self.frame.get();
      if frame.valueType == crate::shardsc::SHType_Enum
        && unsafe { frame.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId == LayoutFrameCC }
      {
        let bits = unsafe { frame.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue };
        Some(LayoutFrame { bits })
      } else {
        // should be unreachable due to parameter type checking
        return Err("Invalid frame type provided. Expected LayoutFrame for Frame");
      }
    } else {
      None
    };

    let mut create_new_scroll_area = false;

    let enable_horizontal_scroll_bar = self.enable_horizontal_scroll_bar.get();
    let enable_horizontal_scroll_bar = if let Some(enable_horizontal_scroll_bar) = retrieve_parameter!(
      enable_horizontal_scroll_bar,
      "enable_horizontal_scroll_bar",
      bool
    ) {
      create_new_scroll_area = true;
      enable_horizontal_scroll_bar
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(
          parent_layout_class,
          scroll_area,
          enable_horizontal_scroll_bar
        )
        .unwrap()
      } else {
        create_new_scroll_area = true;
        false // default enable_horizontal_scroll_bar
      }
    };

    let enable_vertical_scroll_bar = self.enable_vertical_scroll_bar.get();
    let enable_vertical_scroll_bar = if let Some(enable_vertical_scroll_bar) = retrieve_parameter!(
      enable_vertical_scroll_bar,
      "enable_vertical_scroll_bar",
      bool
    ) {
      create_new_scroll_area = true;
      enable_vertical_scroll_bar
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(
          parent_layout_class,
          scroll_area,
          enable_vertical_scroll_bar
        )
        .unwrap()
      } else {
        create_new_scroll_area = true;
        false // default enable_vertical_scroll_bar
      }
    };

    let scroll_visibility = self.scroll_visibility.get();
    let scroll_visibility = if let Some(scroll_visibility) = retrieve_enum_parameter!(
      scroll_visibility,
      "scroll_visibility",
      ScrollVisibility,
      ScrollVisibilityCC
    ) {
      create_new_scroll_area = true;
      scroll_visibility
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, scroll_area, scroll_visibility)
          .unwrap()
      } else {
        create_new_scroll_area = true;
        ScrollVisibility::AlwaysVisible // Default should normally be VisibleWhenNeeded, but it is buggy at the moment
      }
    };

    const MIN_SCROLLING_SIZE: f32 = 64.0; // default value for min_scrolling_width and min_scrolling_height as of egui 0.22.0

    let min_width = self.scroll_area_min_width.get();
    let min_width =
      if let Some(min_width) = retrieve_parameter!(min_width, "scroll_area_min_width", f32) {
        create_new_scroll_area = true;
        min_width
      } else {
        if let Some(parent_layout_class) = parent_layout_class {
          retrieve_layout_class_attribute!(parent_layout_class, scroll_area, min_width).unwrap()
        } else {
          create_new_scroll_area = true;
          MIN_SCROLLING_SIZE // default min_width
        }
      };

    let min_height = self.scroll_area_min_height.get();
    let min_height =
      if let Some(min_height) = retrieve_parameter!(min_height, "scroll_area_min_height", f32) {
        create_new_scroll_area = true;
        min_height
      } else {
        if let Some(parent_layout_class) = parent_layout_class {
          retrieve_layout_class_attribute!(parent_layout_class, scroll_area, min_height).unwrap()
        } else {
          create_new_scroll_area = true;
          MIN_SCROLLING_SIZE // default min_height
        }
      };

    let max_width = self.scroll_area_max_width.get();
    let max_width =
      if let Some(max_width) = retrieve_parameter!(max_width, "scroll_area_max_width", f32) {
        create_new_scroll_area = true;
        max_width
      } else {
        if let Some(parent_layout_class) = parent_layout_class {
          retrieve_layout_class_attribute!(parent_layout_class, scroll_area, max_width).unwrap()
        } else {
          create_new_scroll_area = true;
          f32::INFINITY // default max_width
        }
      };

    let max_height = self.scroll_area_max_height.get();
    let max_height =
      if let Some(max_height) = retrieve_parameter!(max_height, "scroll_area_max_height", f32) {
        create_new_scroll_area = true;
        max_height
      } else {
        if let Some(parent_layout_class) = parent_layout_class {
          retrieve_layout_class_attribute!(parent_layout_class, scroll_area, max_height).unwrap()
        } else {
          create_new_scroll_area = true;
          f32::INFINITY // default max_height
        }
      };

    let auto_shrink_width = self.scroll_area_auto_shrink_width.get();
    let auto_shrink_width = if let Some(auto_shrink_width) =
      retrieve_parameter!(auto_shrink_width, "scroll_area_auto_shrink_width", bool)
    {
      create_new_scroll_area = true;
      auto_shrink_width
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, scroll_area, auto_shrink_width)
          .unwrap()
      } else {
        create_new_scroll_area = true;
        true // default auto_shrink_width
      }
    };

    let auto_shrink_height = self.scroll_area_auto_shrink_height.get();
    let auto_shrink_height = if let Some(auto_shrink_height) =
      retrieve_parameter!(auto_shrink_height, "scroll_area_auto_shrink_height", bool)
    {
      create_new_scroll_area = true;
      auto_shrink_height
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, scroll_area, auto_shrink_height)
          .unwrap()
      } else {
        create_new_scroll_area = true;
        true // default auto_shrink_height
      }
    };

    let enable_scrolling = self.scroll_area_enable_scrolling.get();
    let enable_scrolling = if let Some(enable_scrolling) =
      retrieve_parameter!(enable_scrolling, "scroll_area_enable_scrolling", bool)
    {
      create_new_scroll_area = true;
      enable_scrolling
    } else {
      if let Some(parent_layout_class) = parent_layout_class {
        retrieve_layout_class_attribute!(parent_layout_class, scroll_area, enable_scrolling)
          .unwrap()
      } else {
        create_new_scroll_area = true;
        true // default enable_scrolling
      }
    };

    let mut scroll_area = if create_new_scroll_area {
      Some(EguiScrollAreaSettings {
        enable_horizontal_scroll_bar,
        enable_vertical_scroll_bar,
        min_width,
        min_height,
        max_width,
        max_height,
        auto_shrink_width,
        auto_shrink_height,
        scroll_visibility,
        enable_scrolling,
      })
    } else {
      // Whether there is a parent or not, if there is no scroll area override, then there is no need to create a new scroll area object
      None
    };

    if let Some(parent_layout_class) = parent_layout_class {
      self.layout_class = Some(Rc::new(LayoutClass {
        parent: parent_layout_class,
        layout,
        min_size,
        max_size,
        fill_width,
        fill_height,
        disabled,
        frame,
        scroll_area,
      }));
    } else {
      self.layout_class = Some(Rc::new(LayoutClass {
        parent: std::ptr::null(),
        layout,
        min_size,
        max_size,
        fill_width,
        fill_height,
        disabled,
        frame,
        scroll_area,
      }));
    }

    let layout_class_ref = self.layout_class.as_ref().unwrap();
    Ok(Var::new_object(layout_class_ref, &LAYOUTCLASS_TYPE))
  }
}

impl Default for Layout {
  fn default() -> Self {
    let mut parents = ParamVar::default();
    parents.set_name(PARENTS_UI_NAME);
    Self {
      parents,
      requiring: Vec::new(),
      contents: ShardsVar::default(),
      layout_class: ParamVar::default(),
      min_size: ParamVar::default(),
      max_size: ParamVar::default(),
      fill_width: ParamVar::default(),
      fill_height: ParamVar::default(),
      exposing: Vec::new(),
    }
  }
}

impl LegacyShard for Layout {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("UI.Layout")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("UI.Layout-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.Layout"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "Versatile layout with many options for customizing the desired UI."
    ))
  }

  fn inputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "The value that will be passed to the Contents shards of the layout."
    ))
  }

  fn outputTypes(&mut self) -> &Types {
    &ANY_TYPES
  }

  fn outputHelp(&mut self) -> OptionalString {
    *HELP_OUTPUT_EQUAL_INPUT
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&LAYOUT_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => self.contents.set_param(value),
      1 => self.layout_class.set_param(value),
      2 => self.min_size.set_param(value),
      3 => self.max_size.set_param(value),
      4 => self.fill_width.set_param(value),
      5 => self.fill_height.set_param(value),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.contents.get_param(),
      1 => self.layout_class.get_param(),
      2 => self.min_size.get_param(),
      3 => self.max_size.get_param(),
      4 => self.fill_width.get_param(),
      5 => self.fill_height.get_param(),
      _ => Var::default(),
    }
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    self.requiring.push(ExposedInfo::new_with_help_from_ptr(
      self.layout_class.get_name(),
      shccstr!("The required layout class."),
      *LAYOUTCLASS_TYPE,
    ));

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
    self.layout_class.warmup(ctx);
    self.min_size.warmup(ctx);
    self.max_size.warmup(ctx);
    self.fill_width.warmup(ctx);
    self.fill_height.warmup(ctx);

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.fill_height.cleanup();
    self.fill_width.cleanup();
    self.max_size.cleanup();
    self.min_size.cleanup();
    self.layout_class.cleanup();
    if !self.contents.is_empty() {
      self.contents.cleanup();
    }
    self.parents.cleanup();

    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    if let Some(ui) = util::get_current_parent(self.parents.get())? {
      let layout_class: Option<Rc<LayoutClass>> = Some(Var::from_object_as_clone(
        self.layout_class.get(),
        &LAYOUTCLASS_TYPE,
      )?);
      let layout_class = unsafe {
        let layout_ptr = Rc::as_ptr(layout_class.as_ref().unwrap()) as *mut LayoutClass;
        &*layout_ptr
      };

      let layout = if let Some(layout) = retrieve_layout_class_attribute!(layout_class, layout) {
        layout
      } else {
        return Err("No layout found in LayoutClass. LayoutClass is invalid/corrupted");
      };

      let mut min_size = if !self.min_size.get().is_none() {
        self.min_size.get().try_into()?
      } else {
        if let Some(min_size) = retrieve_layout_class_attribute!(layout_class, min_size) {
          min_size
        } else {
          (0.0, 0.0) // default value for min_size
        }
      };

      let mut max_size = if !self.max_size.get().is_none() {
        Some(self.max_size.get().try_into()?)
      } else {
        if let Some(max_size) = retrieve_layout_class_attribute!(layout_class, max_size) {
          Some(max_size)
        } else {
          None // default value for max_size (no max size)
        }
      };

      // shard parameters have higher priority and override layout class
      let fill_width = if !self.fill_width.get().is_none() {
        self.fill_width.get().try_into()?
      } else {
        if let Some(fill_width) = retrieve_layout_class_attribute!(layout_class, fill_width) {
          fill_width
        } else {
          false // default value for fill_width
        }
      };

      let fill_height = if !self.fill_height.get().is_none() {
        self.fill_height.get().try_into()?
      } else {
        if let Some(fill_height) = retrieve_layout_class_attribute!(layout_class, fill_height) {
          fill_height
        } else {
          false // default value for fill_height
        }
      };

      let mut min_size: egui::Vec2 = min_size.into();

      let mut max_size = if let Some(max_size) = max_size {
        egui::Vec2::new(max_size.0, max_size.1)
      } else {
        ui.available_size_before_wrap() // try to take up all available space (no max)
      };

      if fill_width {
        min_size.x = ui.available_size_before_wrap().x;
        max_size.x = ui.available_size_before_wrap().x;
      }
      if fill_height {
        min_size.y = ui.available_size_before_wrap().y;
        max_size.y = ui.available_size_before_wrap().y;
      }

      // If the size is still 0, use only minimum size for an interactive widget
      if min_size.x == 0.0 {
        min_size.x = ui.spacing().interact_size.x;
      }
      if min_size.y == 0.0 {
        min_size.y = ui.spacing().interact_size.y;
      }

      let disabled =
        if let Some(disabled) = retrieve_layout_class_attribute!(layout_class, disabled) {
          disabled
        } else {
          false // default value for disabled
        };

      let frame = if let Some(frame) = retrieve_layout_class_attribute!(layout_class, frame) {
        let style = ui.style();
        match frame {
          LayoutFrame::Widgets => Some(egui::Frame::group(style)),
          LayoutFrame::SideTopPanel => Some(egui::Frame::side_top_panel(style)),
          LayoutFrame::CentralPanel => Some(egui::Frame::central_panel(style)),
          LayoutFrame::Window => Some(egui::Frame::window(style)),
          LayoutFrame::Menu => Some(egui::Frame::menu(style)),
          LayoutFrame::Popup => Some(egui::Frame::popup(style)),
          LayoutFrame::Canvas => Some(egui::Frame::canvas(style)),
          LayoutFrame::DarkCanvas => Some(egui::Frame::dark_canvas(style)),
          _ => unreachable!(),
        }
      } else {
        None // default value for frame
      };

      let scroll_area =
        if let Some(scroll_area) = retrieve_layout_class_attribute!(layout_class, scroll_area) {
          Some(scroll_area.to_egui_scrollarea())
        } else {
          None // default value for scroll_area
        };

      let scroll_area_id = EguiId::new(self, 0); // TODO: Check if have scroll area first

      // if there is a frame, draw it as the outermost ui element
      if let Some(frame) = frame {
        frame
          .show(ui, |ui| {
            // set whether all widgets in the contents are enabled or disabled
            ui.set_enabled(!disabled);
            // add the new child_ui created by frame onto the stack of parents
            util::with_object_stack_var_pass_stack_var(
              &mut self.parents,
              ui,
              &EGUI_UI_TYPE,
              |parent_stack_var| {
                // inside of frame
                let ui = util::get_current_parent(parent_stack_var.get())?.unwrap();
                // render scroll area and inner layout if there is a scroll area
                if let Some(scroll_area) = scroll_area {
                  scroll_area
                    .id_source(scroll_area_id)
                    .show(ui, |ui| {
                      util::with_object_stack_var_pass_stack_var(
                        parent_stack_var,
                        ui,
                        &EGUI_UI_TYPE,
                        |parent_stack_var| {
                          // inside of scroll area
                          let ui = util::get_current_parent(parent_stack_var.get())?.unwrap();
                          ui.allocate_ui_with_layout(max_size, layout, |ui| {
                            ui.set_min_size(min_size); // set minimum size of entire layout

    if self.contents.is_empty() {
      return Ok(*input);
    }
                            util::activate_ui_contents(
                              context,
                              input,
                              ui,
                              parent_stack_var,
                              &mut self.contents,
                            )
                          })
                          .inner
                        },
                      )
                    })
                    .inner
                } else {
                  // inside of frame, no scroll area to render, render inner layout
                  ui.allocate_ui_with_layout(max_size, layout, |ui| {
                    ui.set_min_size(min_size); // set minimum size of entire layout

                    if self.contents.is_empty() {
                      return Ok(*input);
                    }

                    util::activate_ui_contents(
                      context,
                      input,
                      ui,
                      parent_stack_var,
                      &mut self.contents,
                    )
                  })
                  .inner
                }
              },
            )
          })
          .inner?;
      } else {
        // no frame to render, render only the scroll area (if applicable) and inner layout
        if let Some(scroll_area) = scroll_area {
          scroll_area
            .id_source(scroll_area_id)
            .show(ui, |ui| {
              util::with_object_stack_var_pass_stack_var(
                &mut self.parents,
                ui,
                &EGUI_UI_TYPE,
                |parent_stack_var| {
                  // inside of scroll area
                  let ui = util::get_current_parent(parent_stack_var.get())?.unwrap();
                  ui.allocate_ui_with_layout(max_size, layout, |ui| {
                    ui.set_min_size(min_size); // set minimum size of entire layout

    if self.contents.is_empty() {
      return Ok(*input);
    }

                    util::activate_ui_contents(
                      context,
                      input,
                      ui,
                      parent_stack_var,
                      &mut self.contents,
                    )
                  })
                  .inner
                },
              )
            })
            .inner?;
        } else {
          // inside of frame, no scroll area to render, render inner layout
          ui.allocate_ui_with_layout(max_size, layout, |ui| {
            ui.set_min_size(min_size); // set minimum size of entire layout

            if self.contents.is_empty() {
              return Ok(*input);
            }

            util::activate_ui_contents(context, input, ui, &mut self.parents, &mut self.contents)
          })
          .inner?;
        }
      }

      // Always passthrough the input
      Ok(*input)
    } else {
      Err("No UI parent")
    }
  }
}
