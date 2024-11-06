use lazy_static::__Deref;

use crate::util;
use crate::CONTEXTS_NAME;
use crate::EGUI_CTX_TYPE;
use crate::HELP_VALUE_IGNORED;
use crate::OUTPUT_PASSED;
use shards::core::register_legacy_shard;
use shards::shard::LegacyShard;
use shards::types::Context;
use shards::types::ExposedInfo;
use shards::types::ExposedTypes;
use shards::types::OptionalString;
use shards::types::ParamVar;
use shards::types::Var;
use shards::types::BYTES_TYPES;
use shards::types::NONE_TYPES;

pub struct Restore {
  instance: ParamVar,
  requiring: ExposedTypes,
}

pub struct Save {
  data: Option<Vec<u8>>,
  instance: ParamVar,
  requiring: ExposedTypes,
}

impl Default for Save {
  fn default() -> Self {
    let mut ctx = ParamVar::default();
    ctx.set_name(CONTEXTS_NAME);
    Self {
      data: None,
      instance: ctx,
      requiring: Vec::new(),
    }
  }
}

impl LegacyShard for Save {
  fn registerName() -> &'static str {
    cstr!("UI.SaveState")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("UI.SaveState-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.SaveState"
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    let exp_info = ExposedInfo {
      exposedType: EGUI_CTX_TYPE,
      name: self.instance.get_name(),
      ..ExposedInfo::default()
    };
    self.requiring.push(exp_info);

    Some(&self.requiring)
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!("This shard saves the current state of the UI as a byte array. It saves UI information such as the position of windows, the state of checkboxes, the state of sliders, etc."))
  }

  fn inputHelp(&mut self) -> OptionalString {
    *HELP_VALUE_IGNORED
  }

  fn inputTypes(&mut self) -> &shards::types::Types {
    &NONE_TYPES
  }

  fn outputTypes(&mut self) -> &shards::types::Types {
    &BYTES_TYPES
  }

  fn outputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "Returns the current state of the UI as a byte array."
    ))
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.instance.warmup(ctx);
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.instance.cleanup(ctx);
    Ok(())
  }

  fn activate(
    &mut self,
    _context: &shards::types::Context,
    _input: &shards::types::Var,
  ) -> Result<Option<Var>, &str> {
    let egui_ctx = &util::get_current_context(&self.instance)?.egui_ctx;
    self.data = egui_ctx.memory(|mem| {
      Ok::<_, &str>(Some(
        bitcode::serialize(mem).map_err(|_| "Failed to serialize UI state")?,
      ))
    })?;

    let data: Var = self.data.as_ref().unwrap().as_slice().into();
    Ok(Some(data))
  }
}

impl Default for Restore {
  fn default() -> Self {
    let mut ctx = ParamVar::default();
    ctx.set_name(CONTEXTS_NAME);
    Self {
      instance: ctx,
      requiring: Vec::new(),
    }
  }
}

impl LegacyShard for Restore {
  fn registerName() -> &'static str {
    cstr!("UI.RestoreState")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("UI.RestoreState-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "UI.RestoreState"
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    let exp_info = ExposedInfo {
      exposedType: EGUI_CTX_TYPE,
      name: self.instance.get_name(),
      ..ExposedInfo::default()
    };
    self.requiring.push(exp_info);

    Some(&self.requiring)
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!("This shard restores the UI to a previously saved state (provided as input as a byte array)."))
  }

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "The UI state to restore to represented as a byte array."
    ))
  }

  fn outputHelp(&mut self) -> OptionalString {
    *OUTPUT_PASSED
  }

  fn inputTypes(&mut self) -> &shards::types::Types {
    &BYTES_TYPES
  }

  fn outputTypes(&mut self) -> &shards::types::Types {
    &BYTES_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.instance.warmup(ctx);
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.instance.cleanup(ctx);
    Ok(())
  }

  fn activate(
    &mut self,
    _context: &shards::types::Context,
    input: &shards::types::Var,
  ) -> Result<Option<Var>, &str> {
    let egui_ctx = &util::get_current_context(&self.instance)?.egui_ctx;
    let bytes: &[u8] = input.try_into()?;
    egui_ctx.memory_mut(|mem| {
      Ok::<_, &str>(
        *mem = bitcode::deserialize(bytes).map_err(|_| "Failed to deserialize UI state")?,
      )
    })?;

    Ok(None)
  }
}

pub fn register_shards() {
  register_legacy_shard::<Save>();
  register_legacy_shard::<Restore>();
}
