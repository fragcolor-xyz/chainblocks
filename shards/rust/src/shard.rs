/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

use crate::core::abortWire;
use crate::core::Core;
use crate::shardsc::SHContext;
use crate::shardsc::SHError;
use crate::shardsc::SHExposedTypesInfo;
use crate::shardsc::SHInstanceData;
use crate::shardsc::SHOptionalString;
use crate::shardsc::SHParameterInfo;
use crate::shardsc::SHParametersInfo;
use crate::shardsc::SHSeq;
use crate::shardsc::SHShardComposeResult;
use crate::shardsc::SHString;
use crate::shardsc::SHTable;
use crate::shardsc::SHTypeInfo;
use crate::shardsc::SHTypesInfo;
use crate::shardsc::SHVar;
use crate::shardsc::SHWire;
use crate::shardsc::Shard as CShard;
use crate::shardsc::ShardPtr;
use crate::types::ComposeResult;
use crate::types::Context;
use crate::types::ExposedInfo;
use crate::types::ExposedTypes;
use crate::types::InstanceData;
use crate::types::OptionalString;
use crate::types::Parameters;
use crate::types::Table;
use crate::types::Type;
use crate::types::Types;
use crate::types::Var;
use crate::types::Wire;
use crate::SHStringWithLen;
use core::convert::TryInto;
use core::result::Result;
use core::slice;
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;

pub trait ParameterSet {
  fn parameters() -> &'static Parameters;
  fn num_params() -> usize;
  fn set_param(&mut self, index: i32, value: &Var) -> Result<(), &'static str>;
  fn get_param(&mut self, index: i32) -> Var;
  fn warmup_helper(&mut self, context: &Context) -> Result<(), &'static str>;
  fn cleanup_helper(&mut self, context: Option<&Context>) -> Result<(), &'static str>;
  fn compose_helper(
    &mut self,
    out_required: &mut ExposedTypes,
    data: &InstanceData,
  ) -> Result<(), &'static str>;
}

pub trait ShardGenerated {
  fn register_name() -> &'static str
  where
    Self: Sized;
  fn hash() -> u32
  where
    Self: Sized;
  fn name(&mut self) -> &str;
  fn help(&mut self) -> OptionalString;
  fn parameters(&mut self) -> Option<&Parameters>;
  fn set_param(&mut self, _index: i32, _value: &Var) -> Result<(), &str>;
  fn get_param(&mut self, _index: i32) -> Var;
  fn required_variables(&mut self) -> Option<&ExposedTypes>;
}

pub trait ShardGeneratedOverloads {
  fn has_compose() -> bool;
  fn has_warmup() -> bool;
  fn has_mutate() -> bool;
  fn has_crossover() -> bool;
  fn has_get_state() -> bool;
  fn has_set_state() -> bool;
  fn has_reset_state() -> bool;
}

pub trait Shard {
  fn setup(&mut self) {}

  fn destroy(&mut self) {}

  fn input_types(&mut self) -> &Types;

  fn input_help(&mut self) -> OptionalString {
    OptionalString::default()
  }

  fn output_types(&mut self) -> &Types;

  fn output_help(&mut self) -> OptionalString {
    OptionalString::default()
  }

  fn properties(&mut self) -> Option<&Table> {
    None
  }

  fn exposed_variables(&mut self) -> Option<&ExposedTypes> {
    None
  }

  fn compose(&mut self, _data: &InstanceData) -> Result<Type, &str> {
    Ok(Type::default())
  }

  fn warmup(&mut self, _context: &Context) -> Result<(), &str> {
    Ok(())
  }

  fn cleanup(&mut self, _ctx: Option<&Context>) -> Result<(), &str> {
    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &str>;

  fn mutate(&mut self, _options: Table) {}

  fn crossover(&mut self, _state0: &Var, _state1: &Var) {}

  fn get_state(&mut self) -> Var {
    Var::default()
  }

  fn set_state(&mut self, _state: &Var) {}

  fn reset_state(&mut self) {}
}

pub trait LegacyShard {
  fn registerName() -> &'static str
  where
    Self: Sized;

  fn hash() -> u32
  where
    Self: Sized;

  fn name(&mut self) -> &str;

  fn help(&mut self) -> OptionalString {
    OptionalString::default()
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    None
  }

  fn setParam(&mut self, _index: i32, _value: &Var) -> Result<(), &str> {
    Ok(())
  }

  fn getParam(&mut self, _index: i32) -> Var {
    Var::default()
  }

  fn setup(&mut self) {}

  fn destroy(&mut self) {}

  fn inputTypes(&mut self) -> &Types;

  fn inputHelp(&mut self) -> OptionalString {
    OptionalString::default()
  }

  fn outputTypes(&mut self) -> &Types;

  fn outputHelp(&mut self) -> OptionalString {
    OptionalString::default()
  }

  fn properties(&mut self) -> Option<&Table> {
    None
  }

  fn exposedVariables(&mut self) -> Option<&ExposedTypes> {
    None
  }
  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    None
  }

  fn hasCompose() -> bool
  where
    Self: Sized,
  {
    false
  }

  fn compose(&mut self, _data: &InstanceData) -> Result<Type, &str> {
    Ok(Type::default())
  }

  fn warmup(&mut self, _context: &Context) -> Result<(), &str> {
    Ok(())
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &str>;

  fn cleanup(&mut self, _ctx: Option<&Context>) -> Result<(), &str> {
    Ok(())
  }

  fn hasMutate() -> bool
  where
    Self: Sized,
  {
    false
  }

  fn mutate(&mut self, _options: Table) {}

  fn hasCrossover() -> bool
  where
    Self: Sized,
  {
    false
  }

  fn crossover(&mut self, _state0: &Var, _state1: &Var) {}

  fn hasState() -> bool
  where
    Self: Sized,
  {
    false
  }
  fn getState(&mut self) -> Var {
    Var::default()
  }

  fn setState(&mut self, _state: &Var) {}

  fn resetState(&mut self) {}
}

#[repr(C, align(16))] // ensure alignment is 16 bytes
pub struct LegacyShardWrapper<T: LegacyShard> {
  header: CShard,
  pub shard: T,
  name: Option<CString>,
  help: Option<CString>,
  error: Option<CString>,
  pub outputStorage: Var,
}

#[repr(C, align(16))] // ensure alignment is 16 bytes
pub struct ShardWrapper<T: Shard + ShardGenerated + ShardGeneratedOverloads> {
  header: CShard,
  pub shard: T,
  name: Option<CString>,
  help: Option<CString>,
  error: Option<CString>,
  pub outputStorage: Var,
}

/// # Safety
///
/// Used internally actually
pub unsafe extern "C" fn legacy_shard_construct<T: Default + LegacyShard>() -> *mut CShard {
  let wrapper: Box<LegacyShardWrapper<T>> = Box::new(create());
  let wptr = Box::into_raw(wrapper);
  wptr as *mut CShard
}

unsafe extern "C" fn legacy_shard_name<T: LegacyShard>(
  arg1: *mut CShard,
) -> *const ::std::os::raw::c_char {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  if (*blk).name.is_some() {
    return (*blk).name.as_ref().unwrap().as_ptr();
  } else {
    let name = (*blk).shard.name();
    (*blk).name = Some(CString::new(name).expect("CString::new failed"));
    (*blk).name.as_ref().unwrap().as_ptr()
  }
}

unsafe extern "C" fn legacy_shard_hash<T: LegacyShard>(_arg1: *mut CShard) -> u32 {
  T::hash()
}

unsafe extern "C" fn legacy_shard_help<T: LegacyShard>(arg1: *mut CShard) -> SHOptionalString {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.help().0
}

unsafe extern "C" fn legacy_shard_inputHelp<T: LegacyShard>(arg1: *mut CShard) -> SHOptionalString {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.inputHelp().0
}

unsafe extern "C" fn legacy_shard_outputHelp<T: LegacyShard>(
  arg1: *mut CShard,
) -> SHOptionalString {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.outputHelp().0
}

unsafe extern "C" fn legacy_shard_properties<T: LegacyShard>(arg1: *mut CShard) -> *const SHTable {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  if let Some(properties) = (*blk).shard.properties() {
    &properties.t as *const SHTable
  } else {
    core::ptr::null()
  }
}

unsafe extern "C" fn legacy_shard_inputTypes<T: LegacyShard>(arg1: *mut CShard) -> SHTypesInfo {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  let t = (*blk).shard.inputTypes();
  SHTypesInfo::from(t)
}

unsafe extern "C" fn legacy_shard_outputTypes<T: LegacyShard>(arg1: *mut CShard) -> SHTypesInfo {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  let t = (*blk).shard.outputTypes();
  SHTypesInfo::from(t)
}

unsafe extern "C" fn legacy_shard_setup<T: LegacyShard>(arg1: *mut CShard) {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.setup();
}

unsafe extern "C" fn legacy_shard_destroy<T: LegacyShard>(arg1: *mut CShard) {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.destroy();
  drop(Box::from_raw(blk)); // this will deallocate the Box
}

unsafe extern "C" fn legacy_shard_warmup<T: LegacyShard>(
  arg1: *mut CShard,
  arg2: *mut SHContext,
) -> SHError {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  match (*blk).shard.warmup(&(*arg2)) {
    Ok(_) => SHError::default(),
    Err(error) => SHError {
      message: SHStringWithLen {
        string: error.as_ptr() as *const c_char,
        len: error.len() as u64,
      },
      code: 1,
    },
  }
}

unsafe extern "C" fn legacy_shard_activate<T: LegacyShard>(
  arg1: *mut CShard,
  arg2: *mut SHContext,
  arg3: *const SHVar,
) -> *const SHVar {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  match (*blk).shard.activate(&(*arg2), &(*arg3)) {
    Ok(value) => {
      if let Some(value) = value {
        (*blk).outputStorage = value;
        &(*blk).outputStorage
      } else {
        arg3
      }
    }
    Err(error) => {
      abortWire(&(*arg2), error);
      &(*blk).outputStorage
    }
  }
}

unsafe extern "C" fn legacy_shard_mutate<T: LegacyShard>(arg1: *mut CShard, arg2: SHTable) {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.mutate(arg2.into());
}

unsafe extern "C" fn legacy_shard_cleanup<T: LegacyShard>(
  arg1: *mut CShard,
  arg2: *mut SHContext,
) -> SHError {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  match (*blk)
    .shard
    .cleanup(if arg2.is_null() { None } else { Some(&*arg2) })
  {
    Ok(_) => SHError::default(),
    Err(error) => SHError {
      message: SHStringWithLen {
        string: error.as_ptr() as *const c_char,
        len: error.len() as u64,
      },
      code: 1,
    },
  }
}

unsafe extern "C" fn legacy_shard_exposedVariables<T: LegacyShard>(
  arg1: *mut CShard,
) -> SHExposedTypesInfo {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  if let Some(exposed) = (*blk).shard.exposedVariables() {
    SHExposedTypesInfo::from(exposed)
  } else {
    SHExposedTypesInfo::default()
  }
}

unsafe extern "C" fn legacy_shard_requiredVariables<T: LegacyShard>(
  arg1: *mut CShard,
) -> SHExposedTypesInfo {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  if let Some(required) = (*blk).shard.requiredVariables() {
    SHExposedTypesInfo::from(required)
  } else {
    SHExposedTypesInfo::default()
  }
}

unsafe extern "C" fn legacy_shard_compose<T: LegacyShard>(
  arg1: *mut CShard,
  data: *mut SHInstanceData,
) -> SHShardComposeResult {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  match (*blk).shard.compose(&*data) {
    Ok(output) => SHShardComposeResult {
      error: SHError::default(),
      result: output,
    },
    Err(error) => SHShardComposeResult {
      error: SHError {
        message: SHStringWithLen {
          string: error.as_ptr() as *const c_char,
          len: error.len() as u64,
        },
        code: 1,
      },
      result: SHTypeInfo::default(),
    },
  }
}

unsafe extern "C" fn legacy_shard_parameters<T: LegacyShard>(
  arg1: *mut CShard,
) -> SHParametersInfo {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  if let Some(params) = (*blk).shard.parameters() {
    SHParametersInfo::from(params)
  } else {
    SHParametersInfo::default()
  }
}

unsafe extern "C" fn legacy_shard_getParam<T: LegacyShard>(
  arg1: *mut CShard,
  arg2: ::std::os::raw::c_int,
) -> SHVar {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.getParam(arg2)
}

unsafe extern "C" fn legacy_shard_setParam<T: LegacyShard>(
  arg1: *mut CShard,
  arg2: ::std::os::raw::c_int,
  arg3: *const SHVar,
) -> SHError {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  match (*blk).shard.setParam(arg2, &*arg3) {
    Ok(_) => SHError::default(),
    Err(error) => SHError {
      message: SHStringWithLen {
        string: error.as_ptr() as *const c_char,
        len: error.len() as u64,
      },
      code: 1,
    },
  }
}

unsafe extern "C" fn legacy_shard_crossover<T: LegacyShard>(
  arg1: *mut CShard,
  s0: *const Var,
  s1: *const Var,
) {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.crossover(&*s0, &*s1);
}

unsafe extern "C" fn legacy_shard_getState<T: LegacyShard>(arg1: *mut CShard) -> Var {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.getState()
}

unsafe extern "C" fn legacy_shard_setState<T: LegacyShard>(arg1: *mut CShard, state: *const Var) {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.setState(&*state);
}

unsafe extern "C" fn legacy_shard_resetState<T: LegacyShard>(arg1: *mut CShard) {
  let blk = arg1 as *mut LegacyShardWrapper<T>;
  (*blk).shard.resetState();
}

pub fn create<T: Default + LegacyShard>() -> LegacyShardWrapper<T> {
  let mut shard = LegacyShardWrapper::<T> {
    header: CShard {
      inlineShardId: 0,
      refCount: 0,
      owned: false,
      nameLength: 0,
      line: 0,
      column: 0,
      id: 0,
      name: Some(legacy_shard_name::<T>),
      hash: Some(legacy_shard_hash::<T>),
      help: Some(legacy_shard_help::<T>),
      inputHelp: Some(legacy_shard_inputHelp::<T>),
      outputHelp: Some(legacy_shard_outputHelp::<T>),
      properties: Some(legacy_shard_properties::<T>),
      inputTypes: Some(legacy_shard_inputTypes::<T>),
      outputTypes: Some(legacy_shard_outputTypes::<T>),
      setup: Some(legacy_shard_setup::<T>),
      destroy: Some(legacy_shard_destroy::<T>),
      exposedVariables: Some(legacy_shard_exposedVariables::<T>),
      requiredVariables: Some(legacy_shard_requiredVariables::<T>),
      compose: if T::hasCompose() {
        Some(legacy_shard_compose::<T>)
      } else {
        None
      },
      composeV2: None,
      parameters: Some(legacy_shard_parameters::<T>),
      setParam: Some(legacy_shard_setParam::<T>),
      getParam: Some(legacy_shard_getParam::<T>),
      warmup: Some(legacy_shard_warmup::<T>),
      activate: Some(legacy_shard_activate::<T>),
      cleanup: Some(legacy_shard_cleanup::<T>),
      mutate: if T::hasMutate() {
        Some(legacy_shard_mutate::<T>)
      } else {
        None
      },
      crossover: if T::hasCrossover() {
        Some(legacy_shard_crossover::<T>)
      } else {
        None
      },
      getState: if T::hasState() {
        Some(legacy_shard_getState::<T>)
      } else {
        None
      },
      setState: if T::hasState() {
        Some(legacy_shard_setState::<T>)
      } else {
        None
      },
      resetState: if T::hasState() {
        Some(legacy_shard_resetState::<T>)
      } else {
        None
      },
    },
    shard: T::default(),
    name: None,
    help: None,
    error: None,
    outputStorage: Var::default(),
  };
  shard.header.nameLength = shard.shard.name().len() as u32;
  return shard;
}

pub unsafe extern "C" fn shard_construct<
  T: Default + Shard + ShardGenerated + ShardGeneratedOverloads,
>() -> *mut CShard {
  let wrapper: Box<ShardWrapper<T>> = Box::new(create2());
  let wptr = Box::into_raw(wrapper);
  wptr as *mut CShard
}

unsafe extern "C" fn shard_name<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> *const ::std::os::raw::c_char {
  let blk = arg1 as *mut ShardWrapper<T>;
  if (*blk).name.is_some() {
    return (*blk).name.as_ref().unwrap().as_ptr();
  } else {
    let name = (*blk).shard.name();
    (*blk).name = Some(CString::new(name).expect("CString::new failed"));
    (*blk).name.as_ref().unwrap().as_ptr()
  }
}

unsafe extern "C" fn shard_hash<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  _arg1: *mut CShard,
) -> u32 {
  T::hash()
}

unsafe extern "C" fn shard_help<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHOptionalString {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.help().0
}

unsafe extern "C" fn shard_requiredVariables<
  T: Shard + ShardGenerated + ShardGeneratedOverloads,
>(
  arg1: *mut CShard,
) -> SHExposedTypesInfo {
  let blk = arg1 as *mut ShardWrapper<T>;
  if let Some(required) = (*blk).shard.required_variables() {
    SHExposedTypesInfo::from(required)
  } else {
    SHExposedTypesInfo::default()
  }
}

unsafe extern "C" fn shard_parameters<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHParametersInfo {
  let blk = arg1 as *mut ShardWrapper<T>;
  if let Some(params) = (*blk).shard.parameters() {
    SHParametersInfo::from(params)
  } else {
    SHParametersInfo::default()
  }
}

unsafe extern "C" fn shard_getParam<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  arg2: ::std::os::raw::c_int,
) -> SHVar {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.get_param(arg2)
}

unsafe extern "C" fn shard_setParam<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  arg2: ::std::os::raw::c_int,
  arg3: *const SHVar,
) -> SHError {
  let blk = arg1 as *mut ShardWrapper<T>;
  match (*blk).shard.set_param(arg2, &*arg3) {
    Ok(_) => SHError::default(),
    Err(error) => SHError {
      message: SHStringWithLen {
        string: error.as_ptr() as *const c_char,
        len: error.len() as u64,
      },
      code: 1,
    },
  }
}

unsafe extern "C" fn shard_inputHelp<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHOptionalString {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.input_help().0
}

unsafe extern "C" fn shard_outputHelp<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHOptionalString {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.output_help().0
}

unsafe extern "C" fn shard_properties<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> *const SHTable {
  let blk = arg1 as *mut ShardWrapper<T>;
  if let Some(properties) = (*blk).shard.properties() {
    &properties.t as *const SHTable
  } else {
    core::ptr::null()
  }
}

unsafe extern "C" fn shard_inputTypes<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHTypesInfo {
  let blk = arg1 as *mut ShardWrapper<T>;
  let t = (*blk).shard.input_types();
  SHTypesInfo::from(t)
}

unsafe extern "C" fn shard_outputTypes<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHTypesInfo {
  let blk = arg1 as *mut ShardWrapper<T>;
  let t = (*blk).shard.output_types();
  SHTypesInfo::from(t)
}

unsafe extern "C" fn shard_setup<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.setup();
}

unsafe extern "C" fn shard_destroy<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.destroy();
  drop(Box::from_raw(blk)); // this will deallocate the Box
}

unsafe extern "C" fn shard_warmup<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  arg2: *mut SHContext,
) -> SHError {
  let blk = arg1 as *mut ShardWrapper<T>;
  match (*blk).shard.warmup(&(*arg2)) {
    Ok(_) => SHError::default(),
    Err(error) => SHError {
      message: SHStringWithLen {
        string: error.as_ptr() as *const c_char,
        len: error.len() as u64,
      },
      code: 1,
    },
  }
}

unsafe extern "C" fn shard_activate<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  arg2: *mut SHContext,
  arg3: *const SHVar,
) -> *const SHVar {
  let blk = arg1 as *mut ShardWrapper<T>;
  match (*blk).shard.activate(&(*arg2), &(*arg3)) {
    Ok(value) => {
      if let Some(value) = value {
        (*blk).outputStorage = value;
        &(*blk).outputStorage
      } else {
        arg3 // pass through the input
      }
    }
    Err(error) => {
      abortWire(&(*arg2), error);
      &(*blk).outputStorage
    }
  }
}

unsafe extern "C" fn shard_mutate<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  arg2: SHTable,
) {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.mutate(arg2.into());
}

unsafe extern "C" fn shard_cleanup<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  arg2: *mut SHContext,
) -> SHError {
  let blk = arg1 as *mut ShardWrapper<T>;
  match (*blk)
    .shard
    .cleanup(if arg2.is_null() { None } else { Some(&*arg2) })
  {
    Ok(_) => SHError::default(),
    Err(error) => SHError {
      message: SHStringWithLen {
        string: error.as_ptr() as *const c_char,
        len: error.len() as u64,
      },
      code: 1,
    },
  }
}

unsafe extern "C" fn shard_exposedVariables<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> SHExposedTypesInfo {
  let blk = arg1 as *mut ShardWrapper<T>;
  if let Some(exposed) = (*blk).shard.exposed_variables() {
    SHExposedTypesInfo::from(exposed)
  } else {
    SHExposedTypesInfo::default()
  }
}

unsafe extern "C" fn shard_compose<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  data: *mut SHInstanceData,
) -> SHShardComposeResult {
  let blk = arg1 as *mut ShardWrapper<T>;
  match (*blk).shard.compose(&*data) {
    Ok(output) => SHShardComposeResult {
      error: SHError::default(),
      result: output,
    },
    Err(error) => SHShardComposeResult {
      error: SHError {
        message: SHStringWithLen {
          string: error.as_ptr() as *const c_char,
          len: error.len() as u64,
        },
        code: 1,
      },
      result: SHTypeInfo::default(),
    },
  }
}

unsafe extern "C" fn shard_crossover<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  s0: *const Var,
  s1: *const Var,
) {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.crossover(&*s0, &*s1);
}

unsafe extern "C" fn shard_getState<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) -> Var {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.get_state()
}

unsafe extern "C" fn shard_setState<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
  state: *const Var,
) {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.set_state(&*state);
}

unsafe extern "C" fn shard_resetState<T: Shard + ShardGenerated + ShardGeneratedOverloads>(
  arg1: *mut CShard,
) {
  let blk = arg1 as *mut ShardWrapper<T>;
  (*blk).shard.reset_state();
}

pub fn create2<T: Default + Shard + ShardGenerated + ShardGeneratedOverloads>() -> ShardWrapper<T> {
  let mut shard = ShardWrapper::<T> {
    header: CShard {
      inlineShardId: 0,
      refCount: 0,
      owned: false,
      nameLength: 0,
      line: 0,
      column: 0,
      id: 0,
      name: Some(shard_name::<T>),
      hash: Some(shard_hash::<T>),
      help: Some(shard_help::<T>),
      inputHelp: Some(shard_inputHelp::<T>),
      outputHelp: Some(shard_outputHelp::<T>),
      properties: Some(shard_properties::<T>),
      inputTypes: Some(shard_inputTypes::<T>),
      outputTypes: Some(shard_outputTypes::<T>),
      setup: Some(shard_setup::<T>),
      destroy: Some(shard_destroy::<T>),
      exposedVariables: Some(shard_exposedVariables::<T>),
      requiredVariables: Some(shard_requiredVariables::<T>),
      compose: if T::has_compose() {
        Some(shard_compose::<T>)
      } else {
        None
      },
      composeV2: None,
      parameters: Some(shard_parameters::<T>),
      setParam: Some(shard_setParam::<T>),
      getParam: Some(shard_getParam::<T>),
      warmup: if T::has_warmup() {
        Some(shard_warmup::<T>)
      } else {
        None
      },
      activate: Some(shard_activate::<T>),
      cleanup: Some(shard_cleanup::<T>),
      mutate: if T::has_mutate() {
        Some(shard_mutate::<T>)
      } else {
        None
      },
      crossover: if T::has_crossover() {
        Some(shard_crossover::<T>)
      } else {
        None
      },
      getState: if T::has_get_state() {
        Some(shard_getState::<T>)
      } else {
        None
      },
      setState: if T::has_set_state() {
        Some(shard_setState::<T>)
      } else {
        None
      },
      resetState: if T::has_reset_state() {
        Some(shard_resetState::<T>)
      } else {
        None
      },
    },
    shard: T::default(),
    name: None,
    help: None,
    error: None,
    outputStorage: Var::default(),
  };
  shard.header.nameLength = shard.shard.name().len() as u32;
  return shard;
}

/// Declares a function as an override to [`Shard::activate`].
///
/// This macro is meant to be invoked from [`Shard::compose`].
///
/// # Examples
///
/// ```ignore (only-for-syntax-highlight)
/// impl LegacyShard for MyShard {
///   fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
///     decl_override_activate! {
///       data.activate = MyShard::my_activate_override;
///     }
///   }
/// }
/// ```
///
/// See also: [`impl_legacy_override_activate!`]
#[macro_export]
macro_rules! decl_override_activate {
  (
    $data:ident.activate = $legacy_shard_name:ident::$override_name:ident;
  ) => {
    unsafe {
      (*$data.shard).activate = Some($legacy_shard_name::$override_name);
    }
  };
}

/// Implements an override of [`Shard::activate`] that can be called from C/C++.
///
/// This macro is meant to be invoked from the custom implementation of a [`Shard`].
///
/// # Examples
///
/// ```ignore (only-for-syntax-highlight)
/// impl MyShard {
///   fn activateOverride() {
///     todo!()
///   }
///
///   impl_legacy_override_activate! {
///     extern "C" fn my_activate_override() -> Var {
///       MyShard::activateOverride()
///     }
///   }
/// }
/// ```
///
/// See also: [`decl_override_activate!`]
#[macro_export]
macro_rules! impl_legacy_override_activate {
  (
    $(#[$meta:meta])*
    extern "C" fn $override_name:ident() -> *const Var {
      $legacy_shard_name:ident::$override_impl:ident()
    }
  ) => {
    $(#[$meta])*
    unsafe extern "C" fn $override_name(
      arg1: *mut $crate::shardsc::Shard,
      arg2: *mut Context,
      arg3: *const Var,
    ) -> *const Var {
      let blk = arg1 as *mut $crate::shard::LegacyShardWrapper<$legacy_shard_name>;
      match (*blk).shard.$override_impl(&(*arg2), &(*arg3)) {
        Ok(value) => {
          if let Some(value) = value {
            (*blk).outputStorage = value;
            &(*blk).outputStorage
          } else {
            arg3 // pass through the input
          }
        }
        Err(error) => {
          $crate::core::abortWire(&(*arg2), error);
          &(*blk).outputStorage
        }
      }
    }
  };
}

#[macro_export]
macro_rules! impl_override_activate {
  (
    $(#[$meta:meta])*
    extern "C" fn $override_name:ident() -> *const Var {
      $shard_name:ident::$override_impl:ident()
    }
  ) => {
    $(#[$meta])*
    unsafe extern "C" fn $override_name(
      arg1: *mut $crate::shardsc::Shard,
      arg2: *mut Context,
      arg3: *const Var,
    ) -> *const Var {
      let blk = arg1 as *mut $crate::shard::ShardWrapper<$shard_name>;
      match (*blk).shard.$override_impl(&(*arg2), &(*arg3)) {
        Ok(value) => {
          if let Some(value) = value {
            (*blk).outputStorage = value;
            &(*blk).outputStorage
          } else {
            arg3 // pass through the input
          }
        }
        Err(error) => {
          $crate::core::abortWire(&(*arg2), error);
          &(*blk).outputStorage
        }
      }
    }
  };
}
