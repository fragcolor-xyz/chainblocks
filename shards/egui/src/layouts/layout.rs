/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2022 Fragcolor Pte. Ltd. */

use super::Layout;
use super::LayoutAlign;
use super::LayoutAlignCC;
use super::LayoutClass;
use super::LayoutConstructor;
use super::LayoutDirection;
use super::LayoutDirectionCC;
use super::LayoutFrame;
use super::LayoutFrameCC;
use super::EguiScrollAreaSettings;
use super::ScrollVisibility;
use crate::layouts::ScrollVisibilityCC;
use crate::util;
use crate::EguiId;
use crate::ANY_TABLE_SLICE;
use crate::EGUI_UI_TYPE;
use crate::FLOAT2_VAR_SLICE;
use crate::HELP_OUTPUT_EQUAL_INPUT;
use crate::LAYOUTCLASS_TYPE;
use crate::LAYOUTCLASS_TYPE_VEC;
use crate::LAYOUTCLASS_TYPE_VEC_VAR;
use crate::LAYOUT_FRAME_TYPE_VEC;
use crate::PARENTS_UI_NAME;
use egui::ScrollArea;
use shards::shard::Shard;
use shards::types::Context;
use shards::types::ExposedInfo;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::OptionalString;
use shards::types::ParamVar;
use shards::types::Parameters;
use shards::types::ShardsVar;
use shards::types::Table;
use shards::types::Type;
use shards::types::Types;
use shards::types::Var;
use shards::types::ANY_TYPES;
use shards::types::BOOL_TYPES;
use shards::types::BOOL_VAR_OR_NONE_SLICE;
use shards::types::SHARDS_OR_NONE_TYPES;
use std::rc::Rc;

macro_rules! retrieve_parameter {
  ($table:ident, $name:literal, $type:ty) => {
    if let Some(value) = $table.get_static($name) {
      let value: $type = value.try_into().map_err(|e| {
        shlog!("{}: {}", $name, e);
        "Invalid attribute value received"
      })?;
      Some(value)
    } else {
      None
    }
  };
  ($table:ident, $name:literal, $type:ty, $convert:expr) => {
    if let Some(value) = $table.get_static($name) {
      let value: $type = value.try_into().map_err(|e| {
        shlog!("{}: {}", $name, e);
        "Invalid attribute value received"
      })?;
      Some($convert(value)?)
    } else {
      None
    }
  };
}

macro_rules! retrieve_enum_parameter {
  ($table:ident, $name:literal, $type:ident, $typeId:expr) => {
    if let Some(value) = $table.get_static($name) {
      if value.valueType == crate::shardsc::SHType_Enum
        && unsafe { value.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId == $typeId }
      {
        // TODO: can double check that this is correct
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
      cstr!("Layout"),
      shccstr!("The parameters relating to the layout of the UI element."),
      ANY_TABLE_SLICE,
    )
      .into(),
    (
      cstr!("MinSize"),
      shccstr!("The minimum size of the space to be reserved by this UI. Can be overidden by FillWidth and FillHeight."),
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
      cstr!("ScrollArea"),
      shccstr!("The scroll area to be drawn around the layout to provide scroll bars."),
      ANY_TABLE_SLICE,
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
      shccstr!("The minimum size of the space to be reserved by this UI. Can be overidden by FillWidth and FillHeight."),
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
      layout: ParamVar::default(),
      layout_class: None,
      min_size: ParamVar::default(),
      max_size: ParamVar::default(),
      fill_width: ParamVar::default(),
      fill_height: ParamVar::default(),
      disabled: ParamVar::default(),
      frame: ParamVar::default(),
      scroll_area: ParamVar::default(),
    }
  }
}

impl Shard for LayoutConstructor {
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

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.parent.set_param(value)),
      1 => Ok(self.layout.set_param(value)),
      2 => Ok(self.min_size.set_param(value)),
      3 => Ok(self.max_size.set_param(value)),
      4 => Ok(self.fill_width.set_param(value)),
      5 => Ok(self.fill_height.set_param(value)),
      6 => Ok(self.disabled.set_param(value)),
      7 => Ok(self.frame.set_param(value)),
      8 => Ok(self.scroll_area.set_param(value)),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.parent.get_param(),
      1 => self.layout.get_param(),
      2 => self.min_size.get_param(),
      3 => self.max_size.get_param(),
      4 => self.fill_width.get_param(),
      5 => self.fill_height.get_param(),
      6 => self.disabled.get_param(),
      7 => self.frame.get_param(),
      8 => self.scroll_area.get_param(),
      _ => Var::default(),
    }
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.parent.warmup(ctx);
    self.layout.warmup(ctx);
    self.min_size.warmup(ctx);
    self.max_size.warmup(ctx);
    self.fill_width.warmup(ctx);
    self.fill_height.warmup(ctx);
    self.disabled.warmup(ctx);
    self.frame.warmup(ctx);
    self.scroll_area.warmup(ctx);

    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.scroll_area.cleanup();
    self.frame.cleanup();
    self.disabled.cleanup();
    self.fill_height.cleanup();
    self.fill_width.cleanup();
    self.max_size.cleanup();
    self.min_size.cleanup();
    self.layout.cleanup();
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

    let layout = if !self.layout.get().is_none() {
      let layout_table = self.layout.get();
      if layout_table.valueType == crate::shardsc::SHType_Table {
        let layout_table: Table = layout_table.try_into()?;

        let cross_align = if let Some(cross_align) =
          retrieve_enum_parameter!(layout_table, "cross_align", LayoutAlign, LayoutAlignCC)
        {
          cross_align.into()
        } else {
          // if there is a parent, retrieve the parent's value over the default
          if let Some(parent_layout_class) = parent_layout_class {
            // this is guaranteed to be Some because the parent has already been constructed
            retrieve_layout_class_attribute!(parent_layout_class, layout, cross_align).unwrap()
          } else {
            egui::Align::Min // default cross align
          }
        };

        let main_direction = if let Some(main_direction) = retrieve_enum_parameter!(
          layout_table,
          "main_direction",
          LayoutDirection,
          LayoutDirectionCC
        ) {
          main_direction.into()
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, layout, main_dir).unwrap()
          } else {
            return Err("Invalid main direction provided. Main direction is a required parameter");
          }
        };

        let mut layout = egui::Layout::from_main_dir_and_cross_align(main_direction, cross_align);

        let main_align = if let Some(main_align) =
          retrieve_enum_parameter!(layout_table, "main_align", LayoutAlign, LayoutAlignCC)
        {
          main_align.into()
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, layout, main_align).unwrap()
          } else {
            egui::Align::Center // default main align
          }
        };
        layout = layout.with_main_align(main_align);

        let main_wrap =
          if let Some(main_wrap) = retrieve_parameter!(layout_table, "main_wrap", bool) {
            main_wrap
          } else {
            if let Some(parent_layout_class) = parent_layout_class {
              retrieve_layout_class_attribute!(parent_layout_class, layout, main_wrap).unwrap()
            } else {
              false // default main wrap
            }
          };
        layout = layout.with_main_wrap(main_wrap);

        let main_justify =
          if let Some(main_justify) = retrieve_parameter!(layout_table, "main_justify", bool) {
            main_justify
          } else {
            if let Some(parent_layout_class) = parent_layout_class {
              retrieve_layout_class_attribute!(parent_layout_class, layout, main_justify).unwrap()
            } else {
              false // default main justify
            }
          };
        layout = layout.with_main_justify(main_justify);

        let cross_justify =
          if let Some(cross_justify) = retrieve_parameter!(layout_table, "cross_justify", bool) {
            cross_justify
          } else {
            if let Some(parent_layout_class) = parent_layout_class {
              retrieve_layout_class_attribute!(parent_layout_class, layout, cross_justify).unwrap()
            } else {
              false // default cross justify
            }
          };
        layout = layout.with_cross_justify(cross_justify);

        Some(layout)
      } else {
        return Err("Invalid attribute value received. Expected Table for Layout");
      }
    } else {
      if has_parent {
        // if has parent, then leave it empty and use the reference to the parent to grab the value
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

    let scroll_area = if !self.scroll_area.get().is_none() {
      let scroll_area_table = self.scroll_area.get();
      if scroll_area_table.valueType == crate::shardsc::SHType_Table {
        let scroll_area_table: Table = scroll_area_table.try_into()?;

        let horizontal_scroll_enabled = if let Some(enabled) =
          retrieve_parameter!(scroll_area_table, "horizontal_scroll_enabled", bool)
        {
          enabled
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            // unlike layout, it is possible for this to be none because none of the parents may have had a ScrollArea
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, horizontal_scroll_enabled).unwrap_or(false)
          } else {
            false // default horizontal_scroll_enabled
          }
        };

        let vertical_scroll_enabled = if let Some(enabled) =
          retrieve_parameter!(scroll_area_table, "vertical_scroll_enabled", bool)
        {
          enabled
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, vertical_scroll_enabled).unwrap_or(false)
          } else {
            false // default vertical_scroll_enabled
          }
        };

        let scroll_visibility = if let Some(visibility) = retrieve_enum_parameter!(
          scroll_area_table,
          "scroll_visibility",
          ScrollVisibility,
          ScrollVisibilityCC
        ) {
          visibility
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, scroll_visibility).unwrap_or(ScrollVisibility::AlwaysVisible)
          } else {
            ScrollVisibility::AlwaysVisible
          }
        };

        const MIN_SCROLLING_SIZE: f32 = 64.0; // default value for min_scrolling_width and min_scrolling_height as of egui 0.22.0

        let min_width = if let Some(min_width) =
          retrieve_parameter!(scroll_area_table, "min_width", f32)
        {
          min_width
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, min_width).unwrap_or(MIN_SCROLLING_SIZE)
          } else {
            MIN_SCROLLING_SIZE // default min_width
          }
        };

        let min_height = if let Some(min_height) =
          retrieve_parameter!(scroll_area_table, "min_height", f32)
        {
          min_height
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, min_height).unwrap_or(MIN_SCROLLING_SIZE)
          } else {
            MIN_SCROLLING_SIZE // default min_height
          }
        };

        let enable_scrolling = if let Some(enable_scrolling) =
          retrieve_parameter!(scroll_area_table, "enable_scrolling", bool)
        {
          enable_scrolling
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, enable_scrolling).unwrap_or(true)
          } else {
            true // default enable_scrolling
          }
        };

        let max_width = if let Some(max_width) =
          retrieve_parameter!(scroll_area_table, "max_width", f32)
        {
          max_width
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, max_width).unwrap_or(f32::INFINITY)
          } else {
            f32::INFINITY // default max_width
          }
        };

        let max_height = if let Some(max_height) =
          retrieve_parameter!(scroll_area_table, "max_height", f32)
        {
          max_height
        } else {
          if let Some(parent_layout_class) = parent_layout_class {
            retrieve_layout_class_attribute!(parent_layout_class, scroll_area, max_height).unwrap_or(f32::INFINITY)
          } else {
            f32::INFINITY // default max_height
          }
        };

        Some(
          EguiScrollAreaSettings {
            horizontal_scroll_enabled,
            vertical_scroll_enabled,
            min_width,
            min_height,
            enable_scrolling,
            max_width,
            max_height,
            scroll_visibility,
          }
        )
      } else {
          return Err("Invalid scroll bar type provided. Expected Table for ScrollArea");
      }
    } else {
      // if has parent, put None and let shard retrieve from parent. else, default is also None
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

impl Shard for Layout {
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
      1 => Ok(self.layout_class.set_param(value)),
      2 => Ok(self.min_size.set_param(value)),
      3 => Ok(self.max_size.set_param(value)),
      4 => Ok(self.fill_width.set_param(value)),
      5 => Ok(self.fill_height.set_param(value)),
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
    util::require_parents(&mut self.requiring, &self.parents);

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
    if self.contents.is_empty() {
      return Ok(*input);
    }

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

      if fill_width {
        min_size.0 = ui.available_size_before_wrap().x;
      }
      if fill_height {
        min_size.1 = ui.available_size_before_wrap().y;
      }

      // If the size is still 0, use only minimum size for an interactive widget
      if min_size.0 == 0.0 {
        min_size.0 = ui.spacing().interact_size.x;
      }
      if min_size.1 == 0.0 {
        min_size.1 = ui.spacing().interact_size.y;
      }

      let max_size = if let Some(max_size) = max_size {
        egui::Vec2::new(max_size.0, max_size.1)
      } else {
        ui.available_size_before_wrap() // try to take up all available space (no max)
      };

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
        frame.show(ui, |ui| {
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
                          ui.set_min_size(min_size.into()); // set minimum size of entire layout

                          util::activate_ui_contents(
                            context,
                            input,
                            ui,
                            parent_stack_var,
                            &mut self.contents,
                          )
                        })
                        .inner
                      })
                    
                  })
                  .inner
              } else {
                // inside of frame, no scroll area to render, render inner layout
                ui.allocate_ui_with_layout(max_size, layout, |ui| {
                  ui.set_min_size(min_size.into()); // set minimum size of entire layout
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
            }
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
                    ui.set_min_size(min_size.into()); // set minimum size of entire layout

                    util::activate_ui_contents(
                      context,
                      input,
                      ui,
                      parent_stack_var,
                      &mut self.contents,
                    )
                  })
                  .inner
                })
              
            })
            .inner?;
        } else {
          // inside of frame, no scroll area to render, render inner layout
          ui.allocate_ui_with_layout(max_size, layout, |ui| {
            ui.set_min_size(min_size.into()); // set minimum size of entire layout
            util::activate_ui_contents(
              context,
              input,
              ui,
              &mut self.parents,
              &mut self.contents,
            )
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
