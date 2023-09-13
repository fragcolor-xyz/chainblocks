use crate::{util, Anchor, EguiId, Order, CONTEXTS_NAME, PARENTS_UI_NAME};
use egui::{Pos2, Vec2};
use shards::{
  core::register_shard,
  shard::Shard,
  types::{
    common_type, Context, ExposedTypes, InstanceData, OptionalString, ParamVar, ShardsVar, Type,
    Types, Var, ANY_TYPES, SHARDS_OR_NONE_TYPES,
  },
};

lazy_static! {
  static ref ANCHOR_VAR_TYPE: Type = Type::context_variable(&crate::ANCHOR_TYPES);
}

#[derive(shard)]
#[shard_info("UI.Area", "Places UI element at a specific position.")]
struct AreaShard {
  #[shard_param("Contents", "The UI contents.", SHARDS_OR_NONE_TYPES)]
  pub contents: ShardsVar,
  #[shard_param("Position", "Absolute UI position; or when anchor is set, relative offset. (X/Y)", [common_type::float2, common_type::float2_var])]
  pub position: ParamVar,
  #[shard_param("Pivot", "The pivot for the inner UI", [*crate::ANCHOR_TYPE, *ANCHOR_VAR_TYPE])]
  pub pivot: ParamVar,
  #[shard_param("Anchor", "Side of the screen to anchor the UI to.", [*crate::ANCHOR_TYPE, *ANCHOR_VAR_TYPE])]
  pub anchor: ParamVar,
  #[shard_param("Order", "Paint layer to be used for this UI. Default is background", [*crate::ORDER_TYPE])]
  pub order: ParamVar,
  contexts: ParamVar,
  parents: ParamVar,
  inner_exposed: ExposedTypes,
  #[shard_required]
  required: ExposedTypes,
}

impl Default for AreaShard {
  fn default() -> Self {
    Self {
      contexts: ParamVar::new_named(CONTEXTS_NAME),
      parents: ParamVar::new_named(PARENTS_UI_NAME),
      position: ParamVar::default(),
      pivot: ParamVar::default(),
      anchor: ParamVar::default(),
      order: ParamVar::default(),
      required: Vec::new(),
      contents: ShardsVar::default(),
      inner_exposed: ExposedTypes::new(),
    }
  }
}

#[shard_impl]
impl Shard for AreaShard {
  fn input_types(&mut self) -> &Types {
    &ANY_TYPES
  }
  fn output_types(&mut self) -> &Types {
    &ANY_TYPES
  }
  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.warmup_helper(context)?;
    self.contexts.warmup(context);
    self.parents.warmup(context);
    Ok(())
  }
  fn cleanup(&mut self) -> Result<(), &str> {
    self.cleanup_helper()?;
    self.contexts.cleanup();
    self.parents.cleanup();
    Ok(())
  }
  fn exposed_variables(&mut self) -> Option<&ExposedTypes> {
    Some(&self.inner_exposed)
  }
  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    self.inner_exposed.clear();
    let output_type = self.contents.compose(data)?.outputType;
    shards::util::expose_shards_contents(&mut self.inner_exposed, &self.contents);
    shards::util::require_shards_contents(&mut self.required, &self.contents);

    Ok(output_type)
  }
  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    let (x, y): (f32, f32) = self.position.get().try_into().unwrap_or_default();

    let ui_ctx = util::get_current_context(&self.contexts)?;

    let mut frame = egui::Area::new(EguiId::new(self, 1));

    let order: Option<Order> = self.order.get().try_into().ok();
    frame = frame.order(order.unwrap_or(Order::Background).into());

    let pivot: Option<Anchor> = self.pivot.get().try_into().ok();
    frame = frame.pivot(pivot.unwrap_or(Anchor::Center).into());

    // Either anchor or fix size
    if let Some(anchor) = self.anchor.get().try_into().ok() as Option<Anchor> {
      frame = frame.anchor(anchor.into(), Vec2::new(x, y));
    } else {
      frame = frame.fixed_pos(Pos2::new(x, y));
    }

    let result = frame
      .show(ui_ctx, |ui| {
        util::activate_ui_contents(context, input, ui, &mut self.parents, &mut self.contents)
      })
      .inner?;

    Ok(result)
  }
}

pub fn register_shards() {
  register_shard::<AreaShard>();
}
