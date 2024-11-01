use egui::{Id, Layout, Pos2, Rect, Ui, Vec2};
use shards::{
  core::register_shard,
  types::{
    common_type, Context, ExposedTypes, InstanceData, ParamVar, Seq, ShardsVar, Type, Types, Var,
    ANYS_TYPES, BOOL_VAR_OR_NONE_SLICE, SHARDS_OR_NONE_TYPES,
  },
};

use shards::util::from_raw_parts_allow_null;

use crate::shards::shard::Shard;
use crate::util::{self};
use crate::{shards::shard, util::with_possible_panic};

use crate::{CONTEXTS_NAME, FLOAT_VAR_SLICE, PARENTS_UI_NAME};

pub struct AutoGrid {
  pub max_grid_width: Option<f32>,
  pub item_width: f32,
  pub column_spacing: Option<f32>,
  pub row_spacing: Option<f32>,
  pub center: bool,
  pub max_tile_height: Option<f32>,
}

impl Default for AutoGrid {
  fn default() -> Self {
    Self {
      max_grid_width: None,
      item_width: 100.0,
      column_spacing: None,
      row_spacing: None,
      center: true,
      max_tile_height: None,
    }
  }
}

impl AutoGrid {
  pub fn new() -> Self {
    Self::default()
  }
  pub fn show<F: FnMut(&mut Ui, usize)>(
    self,
    ui: &mut egui::Ui,
    num_children: usize,
    add_child: F,
  ) -> egui::Response {
    let mut add_child = add_child;
    let column_spacing: f32 = if let Some(s) = self.column_spacing {
      s
    } else {
      ui.style().spacing.item_spacing.x
    };

    let row_spacing: f32 = if let Some(s) = self.row_spacing {
      s
    } else {
      ui.style().spacing.item_spacing.y
    };

    let item_width: f32 = self.item_width;

    let max_available_width = {
      if let Some(f) = self.max_grid_width {
        f
      } else {
        ui.available_size_before_wrap().x
      }
    };

    let f = max_available_width / (item_width + column_spacing);
    let max_num_cols = i32::max(0, f32::floor(f) as i32);
    let max_num_cols = if max_num_cols >= 1 {
      // Check if one more column fits since we need to account for spacing being uneven between columns
      let w2 = item_width * (max_num_cols + 1) as f32 + column_spacing * (max_num_cols) as f32;
      if w2 <= max_available_width {
        max_num_cols + 1
      } else {
        max_num_cols
      }
    } else {
      max_num_cols
    };

    let items_width =
      (item_width * max_num_cols as f32) + (column_spacing * (max_num_cols - 1) as f32);

    let avail_x = ui.available_size_before_wrap().x;
    let remaining_x_space = avail_x - items_width;

    let initial_cursor_0 = ui.cursor().min;

    let initial_cursor_1 = if self.center {
      (initial_cursor_0.to_vec2() + Vec2::new(remaining_x_space / 2.0, 0.0)).to_pos2()
    } else {
      initial_cursor_0
    };

    let used_rect = if let Some(tile_height) = self.max_tile_height {
      let viewport_min_y = ui.clip_rect().min.y;
      let viewport_max_y = ui.clip_rect().max.y;

      // First pass: calculate total height without rendering
      let num_rows = (num_children + max_num_cols as usize - 1) / max_num_cols as usize;

      // Calculate total height based on number of rows, ensuring full visibility of last row
      let total_height = if num_rows > 0 {
        let row_heights = num_rows as f32 * tile_height;
        let spacing_heights = (num_rows - 1) as f32 * row_spacing;
        // Add full tile height as padding to ensure last row is fully visible
        row_heights + spacing_heights + tile_height
      } else {
        0.0
      } + (initial_cursor_1.y - initial_cursor_0.y);

      // Second pass: actual rendering
      let mut cursor = initial_cursor_1;
      let mut col_idx = 0;
      let mut max_y = initial_cursor_1.y;
      let mut max_item_height = 0.0;

      let id_src = Id::new("AutoGrid").with(ui.id());

      for i in 0..num_children {
        if col_idx >= max_num_cols {
          cursor.x = initial_cursor_1.x;
          cursor.y += max_item_height + row_spacing;
          max_item_height = 0.0;
          col_idx = 0;
        }

        let is_visible = cursor.y <= viewport_max_y && cursor.y + tile_height >= viewport_min_y;

        if is_visible {
          let widget_pos = cursor;
          let item_max_rect = Rect::from_min_max(
            widget_pos,
            Pos2::new(widget_pos.x + item_width, ui.max_rect().bottom()),
          );

          let child_layout = Layout::top_down(egui::Align::Center);
          let mut child_ui =
            ui.child_ui_with_id_source(item_max_rect, child_layout, id_src.with(i), None);

          add_child(&mut child_ui, i);

          let child_ui_rect = child_ui.min_rect();
          let item_size = child_ui_rect.size();
          max_item_height = f32::max(max_item_height, item_size.y);
        } else {
          max_item_height = f32::max(max_item_height, tile_height);
        }

        max_y = f32::max(max_y, cursor.y + max_item_height);
        cursor.x += item_width + column_spacing;
        col_idx += 1;
      }

      // Use the calculated total height for the allocation
      Rect::from_min_max(
        initial_cursor_0,
        Pos2 {
          x: initial_cursor_0.x + avail_x,
          y: initial_cursor_0.y + total_height,
        },
      )
    } else {
      let mut cursor = initial_cursor_1;
      let mut col_idx: i32 = 0;
      let mut max_y = initial_cursor_1.y;
      let mut max_item_height = 0.0;

      let id_src = Id::new("AutoGrid").with(ui.id());

      // Pass each element in the given sequence to the shards in the contents and render
      for i in 0..num_children {
        // Check if contents will exceed max grid width if added onto current row
        if col_idx >= max_num_cols {
          cursor.x = initial_cursor_1.x;
          cursor.y += max_item_height + row_spacing;
          max_item_height = 0.0;
          col_idx = 0;
        }

        let widget_pos = cursor;
        let item_max_rect = Rect::from_min_max(
          widget_pos,
          Pos2::new(widget_pos.x + item_width, ui.max_rect().bottom()),
        );

        let child_layout = Layout::top_down(egui::Align::Center);
        let mut child_ui =
          ui.child_ui_with_id_source(item_max_rect, child_layout, id_src.with(i), None);

        add_child(&mut child_ui, i);

        let child_ui_rect = child_ui.min_rect();
        let item_size = child_ui_rect.size();

        max_y = f32::max(max_y, cursor.y + item_size.y);
        cursor.x += item_width + column_spacing;
        max_item_height = f32::max(max_item_height, item_size.y);

        col_idx += 1;
      }

      Rect::from_min_max(
        initial_cursor_0,
        Pos2 {
          x: initial_cursor_0.x + avail_x,
          y: max_y,
        },
      )
    };

    ui.allocate_rect(used_rect, egui::Sense::hover())
  }
}

#[derive(shard)]
#[shard_info("UI.AutoGrid", "Works like UI.Grid, but given a Sequence, it will, it each object in the Sequence, execute the Shard provided in its Contents and automatically wrap the generated contents when it exceeds the grid's width.")]
struct AutoGridShard {
  #[shard_param("Contents", "The UI contents to be generated and inserted in each column for each element in the given sequence.", SHARDS_OR_NONE_TYPES)]
  pub contents: ShardsVar,
  #[shard_param(
    "Striped",
    "Whether to alternate a subtle background color to every other row.",
    BOOL_VAR_OR_NONE_SLICE
  )]
  pub striped: ParamVar,
  #[shard_param("MaxGridWidth", "Maximum grid width.", FLOAT_VAR_SLICE)]
  pub max_grid_width: ParamVar,
  #[shard_param("ItemWidth", "The width of each item.", FLOAT_VAR_SLICE)]
  pub item_width: ParamVar,
  #[shard_param("ColumnSpacing", "Spacing between columns.", FLOAT_VAR_SLICE)]
  pub column_spacing: ParamVar,
  #[shard_param("RowSpacing", "Spacing between rows.", FLOAT_VAR_SLICE)]
  pub row_spacing: ParamVar,
  #[shard_warmup]
  contexts: ParamVar,
  #[shard_warmup]
  parents: ParamVar,
  #[shard_required]
  required: ExposedTypes,
  #[shard_param(
    "MaxTileHeight",
    "Maximum height of each tile for viewport culling.",
    FLOAT_VAR_SLICE
  )]
  pub max_tile_height: ParamVar,
}

impl Default for AutoGridShard {
  fn default() -> Self {
    Self {
      contexts: ParamVar::new_named(CONTEXTS_NAME),
      parents: ParamVar::new_named(PARENTS_UI_NAME),
      striped: ParamVar::new(false.into()),
      max_grid_width: ParamVar::default(),
      item_width: ParamVar::default(),
      // item_height: ParamVar::default(),
      column_spacing: ParamVar::default(),
      row_spacing: ParamVar::default(),
      required: Vec::new(),
      contents: ShardsVar::default(),
      max_tile_height: ParamVar::default(),
    }
  }
}

#[shard_impl]
impl Shard for AutoGridShard {
  fn input_types(&mut self) -> &Types {
    &ANYS_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &ANYS_TYPES
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.warmup_helper(context)?;

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;
    util::require_parents(&mut self.required);

    let input_type = data.inputType;
    let slice = unsafe {
      let ptr = input_type.details.seqTypes.elements;
      from_raw_parts_allow_null(ptr, input_type.details.seqTypes.len as usize)
    };

    let element_type = if unsafe { input_type.details.seqTypes.len == 1 } {
      slice[0]
    } else {
      common_type::any
    };

    let mut data = *data;
    data.inputType = element_type;
    self.contents.compose(&data)?;

    if self.item_width.is_none() {
      return Err("Item width is required");
    }

    shards::util::require_shards_contents(&mut self.required, &self.contents);

    // Always passthrough the input
    Ok(input_type)
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    if self.contents.is_empty() {
      return Ok(None);
    }

    if let Some(ui) = util::get_current_parent_opt(self.parents.get())? {
      let seq = Seq::try_from(input)?;

      let mut grid = AutoGrid::new();

      if let Ok(column_spacing) = TryInto::<f32>::try_into(self.column_spacing.get()) {
        grid.column_spacing = Some(column_spacing);
      }

      if let Ok(row_spacing) = TryInto::<f32>::try_into(self.row_spacing.get()) {
        grid.row_spacing = Some(row_spacing);
      }

      let item_width_var = self.item_width.get();
      grid.item_width = item_width_var.try_into().unwrap();

      if let Ok(max_tile_height) = TryInto::<f32>::try_into(self.max_tile_height.get()) {
        grid.max_tile_height = Some(max_tile_height);
      }

      let mut failure: Option<&'static str> = None;

      with_possible_panic(|| {
        grid.show(ui, seq.len(), |ui, idx| {
          if failure.is_some() {
            return;
          }
          let input_elem = &seq[idx];
          if let Err(e) = util::activate_ui_contents(
            context,
            input_elem,
            ui,
            &mut self.parents,
            &mut self.contents,
          ) {
            failure.replace(e);
          }
        });
      })?;

      match failure {
        Some(e) => Err(e),
        None => Ok(None),
      }
    } else {
      Err("No UI parent")
    }
  }
}

pub fn register_shards() {
  register_shard::<AutoGridShard>();
}
