/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

// Core Modules
use crate::core::{cloneVar, destroyVar, Core};
use crate::{
  fourCharacterCode, shlog, shlog_error, shlog_warn, SHAudio, SHExtendedObjectTypeInfo,
  SHType_Audio, SHVarPayload__bindgen_ty_1__bindgen_ty_3, SHVAR_FLAGS_WEAK_OBJECT,
};

// Shard Constants
use crate::shardsc::{
  SHBool, SHColor, SHComposeResult, SHContext, SHEnumInfo, SHEnumTypeInfo, SHExposedTypeInfo,
  SHExposedTypesInfo, SHImage, SHInstanceData, SHMeshRef, SHObjectTypeInfo, SHOptionalString,
  SHOptionalStrings, SHParameterInfo, SHParametersInfo, SHPointer, SHSeq, SHString, SHStrings,
  SHTable, SHTableIterator, SHTableTypeInfo, SHTraits, SHTypeInfo, SHTypeInfo_Details, SHType_Any,
  SHType_Bool, SHType_Bytes, SHType_Color, SHType_ContextVar, SHType_Enum, SHType_Float,
  SHType_Float2, SHType_Float3, SHType_Float4, SHType_Image, SHType_Int, SHType_Int16, SHType_Int2,
  SHType_Int3, SHType_Int4, SHType_Int8, SHType_None, SHType_Object, SHType_Path, SHType_Seq,
  SHType_ShardRef, SHType_String, SHType_Table, SHType_Wire, SHTypesInfo, SHVar, SHVarPayload,
  SHVarPayload__bindgen_ty_1, SHVarPayload__bindgen_ty_1__bindgen_ty_1,
  SHVarPayload__bindgen_ty_1__bindgen_ty_2, SHVarPayload__bindgen_ty_1__bindgen_ty_4, SHWire,
  SHWireInfo, SHWireRef, SHWireState, SHWireState_Continue, SHWireState_Rebase,
  SHWireState_Restart, SHWireState_Return, SHWireState_Stop, Shard, ShardPtr, Shards,
  SHIMAGE_FLAGS_16BITS_INT, SHIMAGE_FLAGS_32BITS_FLOAT, SHVAR_FLAGS_EXTERNAL,
  SHVAR_FLAGS_REF_COUNTED, SHVAR_FLAGS_USES_OBJINFO,
};

// Additional Shard Utilities
use crate::{SHObjectInfo, SHStringWithLen, SHType_Type, SHVar__bindgen_ty_1, SHWireState_Error};

// Core Conversions
use core::{
  convert::{TryFrom, TryInto},
  fmt::{Debug, Formatter},
  mem::transmute,
  ops::{Index, IndexMut},
  slice,
};
use std::ptr::NonNull;

// Serde for Serialization/Deserialization
use serde::{
  de::{MapAccess, SeqAccess, Visitor},
  ser::{SerializeMap, SerializeSeq},
  Deserialize, Deserializer, Serialize,
};

// Standard Libraries
use std::{
  borrow::Cow,
  ffi::{c_void, CStr, CString},
  hash::{Hash, Hasher},
  i32::MAX,
  os::raw::c_char,
  pin::Pin,
  rc::Rc,
  str::Utf8Error,
  sync::atomic::{AtomicU32, Ordering},
  sync::RwLock,
};

#[macro_export]
macro_rules! cstr {
  ($text:expr) => {
    concat!($text, "\0")
  };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SHType {
  None,
  Any,
  Enum,
  Bool,
  Int,
  Int2,
  Int3,
  Int4,
  Int8,
  Int16,
  Float,
  Float2,
  Float3,
  Float4,
  Color,
  Bytes,
  String,
  Path,
  ContextVar,
  Image,
  Seq,
  Table,
  Wire,
  ShardRef,
  Object,
  Audio,
  Type,
  Trait,
}

impl From<crate::shardsc::SHType> for SHType {
  fn from(t: crate::shardsc::SHType) -> Self {
    match t {
      crate::shardsc::SHType_None => SHType::None,
      crate::shardsc::SHType_Any => SHType::Any,
      crate::shardsc::SHType_Enum => SHType::Enum,
      crate::shardsc::SHType_Bool => SHType::Bool,
      crate::shardsc::SHType_Int => SHType::Int,
      crate::shardsc::SHType_Int2 => SHType::Int2,
      crate::shardsc::SHType_Int3 => SHType::Int3,
      crate::shardsc::SHType_Int4 => SHType::Int4,
      crate::shardsc::SHType_Int8 => SHType::Int8,
      crate::shardsc::SHType_Int16 => SHType::Int16,
      crate::shardsc::SHType_Float => SHType::Float,
      crate::shardsc::SHType_Float2 => SHType::Float2,
      crate::shardsc::SHType_Float3 => SHType::Float3,
      crate::shardsc::SHType_Float4 => SHType::Float4,
      crate::shardsc::SHType_Color => SHType::Color,
      crate::shardsc::SHType_Bytes => SHType::Bytes,
      crate::shardsc::SHType_String => SHType::String,
      crate::shardsc::SHType_Path => SHType::Path,
      crate::shardsc::SHType_ContextVar => SHType::ContextVar,
      crate::shardsc::SHType_Image => SHType::Image,
      crate::shardsc::SHType_Seq => SHType::Seq,
      crate::shardsc::SHType_Table => SHType::Table,
      crate::shardsc::SHType_Wire => SHType::Wire,
      crate::shardsc::SHType_ShardRef => SHType::ShardRef,
      crate::shardsc::SHType_Object => SHType::Object,
      crate::shardsc::SHType_Audio => SHType::Audio,
      crate::shardsc::SHType_Type => SHType::Type,
      crate::shardsc::SHType_Trait => SHType::Trait,
      _ => panic!("Unknown SHType: {:?}", t),
    }
  }
}

impl From<SHType> for crate::shardsc::SHType {
  fn from(t: SHType) -> Self {
    match t {
      SHType::None => crate::shardsc::SHType_None,
      SHType::Any => crate::shardsc::SHType_Any,
      SHType::Enum => crate::shardsc::SHType_Enum,
      SHType::Bool => crate::shardsc::SHType_Bool,
      SHType::Int => crate::shardsc::SHType_Int,
      SHType::Int2 => crate::shardsc::SHType_Int2,
      SHType::Int3 => crate::shardsc::SHType_Int3,
      SHType::Int4 => crate::shardsc::SHType_Int4,
      SHType::Int8 => crate::shardsc::SHType_Int8,
      SHType::Int16 => crate::shardsc::SHType_Int16,
      SHType::Float => crate::shardsc::SHType_Float,
      SHType::Float2 => crate::shardsc::SHType_Float2,
      SHType::Float3 => crate::shardsc::SHType_Float3,
      SHType::Float4 => crate::shardsc::SHType_Float4,
      SHType::Color => crate::shardsc::SHType_Color,
      SHType::Bytes => crate::shardsc::SHType_Bytes,
      SHType::String => crate::shardsc::SHType_String,
      SHType::Path => crate::shardsc::SHType_Path,
      SHType::ContextVar => crate::shardsc::SHType_ContextVar,
      SHType::Image => crate::shardsc::SHType_Image,
      SHType::Seq => crate::shardsc::SHType_Seq,
      SHType::Table => crate::shardsc::SHType_Table,
      SHType::Wire => crate::shardsc::SHType_Wire,
      SHType::ShardRef => crate::shardsc::SHType_ShardRef,
      SHType::Object => crate::shardsc::SHType_Object,
      SHType::Audio => crate::shardsc::SHType_Audio,
      SHType::Type => crate::shardsc::SHType_Type,
      SHType::Trait => crate::shardsc::SHType_Trait,
    }
  }
}

pub type Context = SHContext;
pub type Var = SHVar;
pub type Type = SHTypeInfo;
pub type InstanceData = SHInstanceData;
pub type ComposeResult = SHComposeResult;
pub type ExposedInfo = SHExposedTypeInfo;
pub type ParameterInfo = SHParameterInfo;
pub type RawString = SHString;

#[repr(transparent)] // force it same size of original
#[derive(Default, Serialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClonedVar(pub Var);

impl Ord for Var {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    unsafe {
      // Use partialOrder to determine the full ordering directly
      match (*Core).compareVar.unwrap_unchecked()(
        self as *const SHVar as *mut SHVar,
        other as *const SHVar as *mut SHVar,
      ) {
        -1 => std::cmp::Ordering::Less,
        0 => std::cmp::Ordering::Equal,
        1 => std::cmp::Ordering::Greater,
        _ => unreachable!("compareVar returned an invalid value"),
      }
    }
  }
}

impl PartialOrd for Var {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl ClonedVar {
  pub fn new_deserializing(bytes_buffer_var: &Var) -> Self {
    ClonedVar(unsafe { shards_deserialize_var(bytes_buffer_var) })
  }

  pub fn new_string(s: &str) -> Self {
    let string = Var::ephemeral_string(s);
    string.into()
  }

  pub fn new_path(s: &str) -> Self {
    let mut path = Var::ephemeral_string(s);
    path.valueType = SHType_Path;
    path.into()
  }

  pub fn new_context_var(s: &str) -> Self {
    let mut context_var = Var::ephemeral_string(s);
    context_var.valueType = SHType_ContextVar;
    context_var.into()
  }

  pub fn new_bytes(bytes: &[u8]) -> Self {
    let bytes = Var::ephemeral_slice(bytes);
    bytes.into()
  }

  pub fn new_image(
    width: u16,
    height: u16,
    channels: u8,
    flags: u8,
    data: &[u8],
  ) -> Result<Self, &'static str> {
    let dataLen = unsafe {
      let mut dummyImg = SHImage {
        width,
        height,
        channels,
        flags,
        ..Default::default()
      };
      (*Core).imageDeriveDataLength.unwrap_unchecked()(&mut dummyImg)
    };

    if data.len() != dataLen as usize {
      return Err("Invalid image data length");
    }

    return Ok(unsafe {
      let image = (*Core).imageNew.unwrap_unchecked()(dataLen);
      (*image).width = width;
      (*image).height = height;
      (*image).channels = channels;
      (*image).flags = flags;
      std::ptr::copy_nonoverlapping(data.as_ptr(), (*image).data, dataLen as usize);

      ClonedVar(Var {
        valueType: SHType_Image,
        payload: SHVarPayload {
          __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { imageValue: image },
        },
        ..Default::default()
      })
    });
  }

  pub fn assign(&mut self, other: &Var) {
    cloneVar(&mut self.0, other);
  }

  pub fn set_param(&mut self, other: &Var) -> Result<(), &'static str> {
    self.assign(other);
    Ok(())
  }

  pub fn assign_string(&mut self, s: &str) {
    let cstr = CString::new(s).unwrap();
    let tmp = Var::from(&cstr);
    self.assign(&tmp);
  }
}

impl Clone for ClonedVar {
  #[inline(always)]
  fn clone(&self) -> Self {
    let mut ret = ClonedVar::default();
    ret.assign(&self.0);
    ret
  }
}

impl Drop for ClonedVar {
  #[inline(always)]
  fn drop(&mut self) {
    unsafe {
      let rv = &self.0 as *const SHVar as *mut SHVar;
      (*Core).destroyVar.unwrap_unchecked()(rv);
    }
  }
}

impl Debug for ClonedVar {
  fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
    write!(f, "{:?}", self.0)
  }
}

#[repr(transparent)] // force it same size of original
pub struct ExternalVar(pub Var);

#[repr(transparent)] // force it same size of original
#[derive(Default)]
pub struct WrappedVar(pub Var); // used in DSL macro, ignore it

#[repr(transparent)] // force it same size of original
#[derive(Clone)]
pub struct Mesh(pub SHMeshRef);

#[repr(transparent)] // force it same size of original
#[derive(Copy, Clone)]
pub struct WireRef(pub SHWireRef);

#[repr(transparent)] // force it same size of original
#[derive(Copy, Clone)]
pub struct ShardRef(pub ShardPtr);

pub struct AutoShardRef(pub ShardRef);

impl Drop for AutoShardRef {
  fn drop(&mut self) {
    self.0.destroy();
  }
}

impl Default for Mesh {
  fn default() -> Self {
    Mesh(unsafe { (*Core).createMesh.unwrap_unchecked()() })
  }
}

impl Drop for Mesh {
  fn drop(&mut self) {
    unsafe { (*Core).destroyMesh.unwrap_unchecked()(self.0) }
  }
}

impl Mesh {
  pub fn compose(&self, wire: WireRef) -> bool {
    unsafe { (*Core).compose.unwrap_unchecked()(self.0, wire.0) }
  }

  pub fn schedule(&mut self, wire: WireRef, compose: bool) {
    unsafe { (*Core).schedule.unwrap_unchecked()(self.0, wire.0, compose) }
  }

  pub fn tick(&mut self) -> bool {
    unsafe { (*Core).tick.unwrap_unchecked()(self.0) }
  }

  pub fn is_empty(&mut self) -> bool {
    unsafe { (*Core).isEmpty.unwrap_unchecked()(self.0) }
  }

  pub fn terminate(&mut self) {
    unsafe { (*Core).terminate.unwrap_unchecked()(self.0) }
  }
}

#[repr(transparent)]
pub struct MeshVar(pub ClonedVar);

impl MeshVar {
  pub fn new() -> Self {
    MeshVar(ClonedVar(unsafe {
      (*Core).createMeshVar.unwrap_unchecked()()
    }))
  }

  pub fn mesh_ref(&self) -> SHMeshRef {
    unsafe {
      self
        .0
         .0
        .payload
        .__bindgen_anon_1
        .__bindgen_anon_1
        .objectValue as SHMeshRef
    }
  }

  pub fn compose(&self, wire: WireRef) -> bool {
    unsafe { (*Core).compose.unwrap_unchecked()(self.mesh_ref(), wire.0) }
  }

  pub fn schedule(&mut self, wire: WireRef, compose: bool) {
    unsafe { (*Core).schedule.unwrap_unchecked()(self.mesh_ref(), wire.0, compose) }
  }

  /// Returns false if we had any wire failure
  pub fn tick(&mut self) -> bool {
    unsafe { (*Core).tick.unwrap_unchecked()(self.mesh_ref()) }
  }

  pub fn is_empty(&mut self) -> bool {
    unsafe { (*Core).isEmpty.unwrap_unchecked()(self.mesh_ref()) }
  }

  pub fn terminate(&mut self) {
    unsafe { (*Core).terminate.unwrap_unchecked()(self.mesh_ref()) }
  }

  pub fn set_label(&mut self, str: &str) {
    unsafe {
      let name = SHStringWithLen {
        string: str.as_ptr() as *const c_char,
        len: str.len() as u64,
      };
      (*Core).setMeshLabel.unwrap_unchecked()(self.mesh_ref(), name)
    }
  }
}

pub struct Wire(pub WireRef);

impl Drop for Wire {
  fn drop(&mut self) {
    unsafe { (*Core).destroyWire.unwrap_unchecked()(self.0 .0) }
  }
}

extern "C" {
  fn shards_set_wire_debug_id(wire: SHWireRef, id: u64);
  fn shards_deserialize_var(bytes_buffer_var: *const SHVar) -> SHVar;
  fn shards_serialize_var(var: *const SHVar) -> SHVar;
}

impl Wire {
  pub fn new(name: &str) -> Self {
    let name = SHStringWithLen {
      string: name.as_ptr() as *const c_char,
      len: name.len() as u64,
    };
    Wire(WireRef(unsafe {
      (*Core).createWire.unwrap_unchecked()(name)
    }))
  }

  pub fn set_debug_id(self, id: u64) -> Self {
    unsafe { shards_set_wire_debug_id(self.0 .0, id) };
    self
  }

  pub fn add_shard(&self, shard: ShardRef) {
    unsafe { (*Core).addShard.unwrap_unchecked()(self.0 .0, shard.0) }
  }

  pub fn set_looped(&self, looped: bool) {
    unsafe { (*Core).setWireLooped.unwrap_unchecked()(self.0 .0, looped) }
  }

  pub fn set_traits(&self, traits: SHSeq) {
    unsafe { (*Core).setWireTraits.unwrap_unchecked()(self.0 .0, traits) }
  }

  pub fn set_unsafe(&self, unsafe_: bool) {
    unsafe { (*Core).setWireUnsafe.unwrap_unchecked()(self.0 .0, unsafe_) }
  }

  pub fn set_pure(&self, pure: bool) {
    unsafe { (*Core).setWirePure.unwrap_unchecked()(self.0 .0, pure) }
  }

  pub fn set_stack_size(&self, stack_size: usize) {
    unsafe { (*Core).setWireStackSize.unwrap_unchecked()(self.0 .0, stack_size as u64) }
  }

  pub fn stop(&self) -> ClonedVar {
    unsafe { ClonedVar((*Core).stopWire.unwrap_unchecked()(self.0 .0)) }
  }

  pub fn get_info(&self) -> SHWireInfo {
    unsafe { (*Core).getWireInfo.unwrap_unchecked()(self.0 .0) }
  }
}

unsafe extern "C" fn error_cb(
  errorShard: *const Shard,
  errorTxt: SHStringWithLen,
  nonfatalWarning: SHBool,
  userData: *mut c_void,
) {
  let shard_name = CStr::from_ptr((*errorShard).name.unwrap()(errorShard as *mut _));
  let msg = std::str::from_utf8(unsafe {
    if errorTxt.len == 0 {
      &[]
    } else {
      slice::from_raw_parts(errorTxt.string as *const u8, errorTxt.len as usize)
    }
  })
  .unwrap();
  if !nonfatalWarning {
    shlog_error!(
      "Fatal error: {} shard: {}",
      msg,
      shard_name.to_str().unwrap()
    );
    let failed = userData as *mut bool;
    *failed = true;
  } else {
    shlog_warn!("Warning: {} shard: {}", msg, shard_name.to_str().unwrap());
  }
}

impl ShardRef {
  pub fn output_types(&self) -> &[Type] {
    unsafe {
      let info = (*self.0).outputTypes.unwrap_unchecked()(self.0);
      if info.len == 0 {
        return &[];
      }
      core::slice::from_raw_parts(info.elements, info.len as usize)
    }
  }

  pub fn input_types(&self) -> &[Type] {
    unsafe {
      let info = (*self.0).inputTypes.unwrap_unchecked()(self.0);
      if info.len == 0 {
        return &[];
      }
      core::slice::from_raw_parts(info.elements, info.len as usize)
    }
  }

  pub fn create(name: &str, debug_info: Option<(u32, u32)>) -> Option<Self> {
    unsafe {
      let ptr = (*Core).createShard.unwrap_unchecked()(SHStringWithLen {
        string: name.as_ptr() as *const c_char,
        len: name.len() as u64,
      });
      if ptr.is_null() {
        None
      } else {
        if let Some(debug_info) = debug_info {
          (*ptr).line = debug_info.0;
          (*ptr).column = debug_info.1;
        }
        (*ptr).setup.unwrap_unchecked()(ptr);
        Some(ShardRef(ptr))
      }
    }
  }

  pub fn name(&self) -> &str {
    unsafe {
      let c_name = (*self.0).name.unwrap_unchecked()(self.0);
      CStr::from_ptr(c_name).to_str().unwrap()
    }
  }

  pub fn destroy(&self) {
    unsafe {
      (*Core).releaseShard.unwrap_unchecked()(self.0);
    }
  }

  pub fn cleanup(&self, context: Option<&Context>) -> Result<(), &'static str> {
    unsafe {
      let result = (*self.0).cleanup.unwrap_unchecked()(
        self.0,
        if let Some(_ref) = context {
          &_ref as *const _ as *mut _
        } else {
          std::ptr::null_mut()
        },
      );
      if result.code == 0 {
        Ok(())
      } else {
        if result.message.len == 0 {
          return Err("Unknown error");
        }
        let cstr = std::str::from_utf8(slice::from_raw_parts(
          result.message.string as *const u8,
          result.message.len as usize,
        ));
        Err(cstr.unwrap())
      }
    }
  }

  pub fn warmup(&self, context: &Context) -> Result<(), &'static str> {
    unsafe {
      if (*self.0).warmup.is_some() {
        let result = (*self.0).warmup.unwrap_unchecked()(self.0, context as *const _ as *mut _);
        if result.code == 0 {
          Ok(())
        } else {
          if result.message.len == 0 {
            return Err("Unknown error");
          }
          let cstr = std::str::from_utf8(slice::from_raw_parts(
            result.message.string as *const u8,
            result.message.len as usize,
          ));
          Err(cstr.unwrap())
        }
      } else {
        Ok(())
      }
    }
  }

  pub fn parameters(&self) -> &[ParameterInfo] {
    unsafe {
      let params = (*self.0).parameters.unwrap_unchecked()(self.0);
      if params.len == 0 {
        return &[];
      } else {
        std::slice::from_raw_parts(params.elements, params.len as usize)
      }
    }
  }

  pub fn set_parameter(&self, index: i32, value: Var) -> Result<(), &'static str> {
    unsafe {
      let success = (*Core).validateSetParam.unwrap_unchecked()(self.0, index, &value);
      if !success {
        Err("Set parameter validation failed")
      } else {
        let err = (*self.0).setParam.unwrap_unchecked()(self.0, index, &value);
        if err.code != 0 {
          Err(err.message.into())
        } else {
          Ok(())
        }
      }
    }
  }

  pub fn get_parameter(&self, index: i32) -> Var {
    unsafe { (*self.0).getParam.unwrap_unchecked()(self.0, index) }
  }

  pub fn get_line_info(&self) -> (u32, u32) {
    unsafe { ((*self.0).line, (*self.0).column) }
  }

  pub fn properties(&self) -> Option<Table> {
    if unsafe { (*self.0).properties.is_some() } {
      let sh_table_ptr = unsafe { (*self.0).properties.unwrap_unchecked()(self.0) };
      unsafe { Some(Table::from_sh_table(*sh_table_ptr)) }
    } else {
      None
    }
  }
}

#[derive(PartialEq)]
pub struct String(pub SHString);
#[derive(Default, Clone, Copy)]
pub struct OptionalString(pub SHOptionalString);
pub struct DerivedType(pub Type);

#[macro_export]
macro_rules! shstr {
  ($text:expr) => {{
    const shstr: RawString = concat!($text, "\0").as_ptr() as *const std::os::raw::c_char;
    shstr
  }};
}

impl Drop for DerivedType {
  fn drop(&mut self) {
    let ti = &mut self.0;
    unsafe {
      (*Core).freeDerivedTypeInfo.unwrap_unchecked()(ti as *mut _);
    }
  }
}

#[derive(PartialEq, Eq)]
pub enum WireState {
  Continue,
  Return,
  Rebase,
  Restart,
  Stop,
  Error,
}

impl From<SHWireState> for WireState {
  fn from(state: SHWireState) -> Self {
    match state {
      SHWireState_Continue => WireState::Continue,
      SHWireState_Return => WireState::Return,
      SHWireState_Rebase => WireState::Rebase,
      SHWireState_Restart => WireState::Restart,
      SHWireState_Stop => WireState::Stop,
      SHWireState_Error => WireState::Error,
      _ => unreachable!(),
    }
  }
}

unsafe impl Send for Var {}
unsafe impl Send for Context {}
unsafe impl Send for Shard {}
unsafe impl Send for Table {}
unsafe impl Sync for Var {}
unsafe impl Sync for Table {}
unsafe impl Sync for OptionalString {}
unsafe impl Sync for WireRef {}
unsafe impl Sync for Mesh {}
unsafe impl Sync for ClonedVar {}
unsafe impl Sync for ExternalVar {}

/*
SHTypeInfo & co
*/
unsafe impl Send for SHObjectInfo {}
unsafe impl Sync for SHTypeInfo {}
unsafe impl Sync for SHExposedTypeInfo {}
unsafe impl Sync for SHParameterInfo {}
unsafe impl Sync for SHStrings {}
unsafe impl Sync for SHObjectInfo {}

// Todo SHTypeInfo proper wrapper Type with helpers

pub type Types = Vec<Type>;

impl From<&Types> for SHTypesInfo {
  fn from(types: &Types) -> Self {
    SHTypesInfo {
      elements: types.as_ptr() as *mut SHTypeInfo,
      len: types.len() as u32,
      cap: 0,
    }
  }
}

impl From<&[Type]> for SHTypesInfo {
  fn from(types: &[Type]) -> Self {
    SHTypesInfo {
      elements: types.as_ptr() as *mut SHTypeInfo,
      len: types.len() as u32,
      cap: 0,
    }
  }
}

fn internal_from_types(types: &[Type]) -> SHTypesInfo {
  let len = types.len();
  SHTypesInfo {
    elements: types.as_ptr() as *mut SHTypeInfo,
    len: len as u32,
    cap: 0,
  }
}

/*
SHExposedTypeInfo & co
*/

impl ExposedInfo {
  pub fn new(name: *const c_char, ctype: SHTypeInfo) -> Self {
    let chelp = core::ptr::null();
    SHExposedTypeInfo {
      exposedType: ctype,
      name,
      help: SHOptionalString {
        string: chelp, // TODO: Pull from parameter into
        crc: 0,
      },
      isMutable: false,
      isProtected: false,
      global: false,
      tracked: false,
      declared: false,
    }
  }

  pub fn new_with_help(name: &CString, help: &CString, ctype: SHTypeInfo) -> Self {
    SHExposedTypeInfo {
      exposedType: ctype,
      name: name.as_ptr(),
      help: SHOptionalString {
        string: help.as_ptr(),
        crc: 0,
      },
      isMutable: false,
      isProtected: false,
      global: false,
      tracked: false,
      declared: false,
    }
  }

  pub fn new_with_help_from_ptr(name: SHString, help: SHOptionalString, ctype: SHTypeInfo) -> Self {
    SHExposedTypeInfo {
      exposedType: ctype,
      name,
      help,
      isMutable: false,
      isProtected: false,
      global: false,
      tracked: false,
      declared: false,
    }
  }

  pub const fn new_static(name: &'static str, ctype: SHTypeInfo) -> Self {
    let cname = name.as_ptr() as *const std::os::raw::c_char;
    let chelp = core::ptr::null();
    SHExposedTypeInfo {
      exposedType: ctype,
      name: cname,
      help: SHOptionalString {
        string: chelp,
        crc: 0,
      },
      isMutable: false,
      isProtected: false,
      global: false,
      tracked: false,
      declared: false,
    }
  }

  pub const fn new_static_with_help(
    name: &'static str,
    help: SHOptionalString,
    ctype: SHTypeInfo,
  ) -> Self {
    let cname = name.as_ptr() as *const std::os::raw::c_char;
    SHExposedTypeInfo {
      exposedType: ctype,
      name: cname,
      help,
      isMutable: false,
      isProtected: false,
      global: false,
      tracked: false,
      declared: false,
    }
  }
}

pub type ExposedTypes = Vec<ExposedInfo>;

impl From<SHExposedTypesInfo> for ExposedTypes {
  fn from(types: SHExposedTypesInfo) -> Self {
    let mut exposed_types = Vec::with_capacity(types.len as usize);
    for i in 0..types.len {
      let exposed_type = unsafe { &*types.elements.add(i as usize) };
      // copy is fine, we just care of the vector
      exposed_types.push(*exposed_type);
    }
    exposed_types
  }
}

impl From<&ExposedTypes> for SHExposedTypesInfo {
  fn from(vec: &ExposedTypes) -> Self {
    SHExposedTypesInfo {
      elements: vec.as_ptr() as *mut SHExposedTypeInfo,
      len: vec.len() as u32,
      cap: 0,
    }
  }
}

/*
SHParameterInfo & co
*/
impl ParameterInfo {
  fn new(name: &'static str, types: &[Type]) -> Self {
    SHParameterInfo {
      name: name.as_ptr() as *mut std::os::raw::c_char,
      help: SHOptionalString {
        string: core::ptr::null(),
        crc: 0,
      },
      valueTypes: internal_from_types(types),
      variableSetter: false,
    }
  }

  fn new1(name: &'static str, help: &'static str, types: &[Type]) -> Self {
    SHParameterInfo {
      name: name.as_ptr() as *mut std::os::raw::c_char,
      help: SHOptionalString {
        string: help.as_ptr() as *mut std::os::raw::c_char,
        crc: 0,
      },
      valueTypes: internal_from_types(types),
      variableSetter: false,
    }
  }

  fn new2(name: &'static str, help: SHOptionalString, types: &[Type]) -> Self {
    SHParameterInfo {
      name: name.as_ptr() as *mut std::os::raw::c_char,
      help,
      valueTypes: internal_from_types(types),
      variableSetter: false,
    }
  }
}

impl From<SHOptionalString> for OptionalString {
  fn from(s: SHOptionalString) -> OptionalString {
    OptionalString(s)
  }
}

// impl From<&str> for OptionalString {
//   fn from(s: &str) -> OptionalString {
//     let cos = SHOptionalString {
//       string: s.as_ptr() as *const std::os::raw::c_char,
//       crc: 0, // TODO
//     };
//     OptionalString(cos)
//   }
// }

// impl From<&str> for SHOptionalString {
//   fn from(s: &str) -> SHOptionalString {
//     SHOptionalString {
//       string: s.as_ptr() as *const std::os::raw::c_char,
//       crc: 0, // TODO
//     }
//   }
// }

impl From<(&'static str, &[Type])> for ParameterInfo {
  fn from(v: (&'static str, &[Type])) -> ParameterInfo {
    ParameterInfo::new(v.0, v.1)
  }
}

impl From<(&'static str, &'static str, &[Type])> for ParameterInfo {
  fn from(v: (&'static str, &'static str, &[Type])) -> ParameterInfo {
    ParameterInfo::new1(v.0, v.1, v.2)
  }
}

impl From<(&'static str, SHOptionalString, &[Type])> for ParameterInfo {
  fn from(v: (&'static str, SHOptionalString, &[Type])) -> ParameterInfo {
    ParameterInfo::new2(v.0, v.1, v.2)
  }
}

pub type Parameters = Vec<ParameterInfo>;

impl From<&Parameters> for SHParametersInfo {
  fn from(vec: &Parameters) -> Self {
    SHParametersInfo {
      elements: vec.as_ptr() as *mut SHParameterInfo,
      len: vec.len() as u32,
      cap: 0,
    }
  }
}

impl From<SHParametersInfo> for &[SHParameterInfo] {
  fn from(_: SHParametersInfo) -> Self {
    unimplemented!()
  }
}

impl From<SHTypeInfo> for ClonedVar {
  fn from(t: SHTypeInfo) -> Self {
    let mut var = Var::default();
    var.valueType = SHType_Type;
    var.payload.__bindgen_anon_1.typeValue = &t as *const SHTypeInfo as *mut SHTypeInfo; // Directly assign the pointer
    let mut cloned = ClonedVar::default();
    cloneVar(&mut cloned.0, &var);
    cloned
  }
}

impl From<SHStringWithLen> for &str {
  fn from(s: SHStringWithLen) -> Self {
    unsafe {
      if s.len == 0 {
        return "";
      }
      let slice = slice::from_raw_parts(s.string as *const u8, s.len as usize);
      let s = std::str::from_utf8(slice);
      match s {
        Ok(s) => s,
        Err(e) => {
          let valid = e.valid_up_to();
          std::str::from_utf8(&slice[..valid]).unwrap()
        }
      }
    }
  }
}

impl From<&str> for SHStringWithLen {
  fn from(s: &str) -> Self {
    SHStringWithLen {
      string: s.as_ptr() as *const c_char,
      len: s.len() as u64,
    }
  }
}

/*
Static common type infos utility
*/
pub mod common_type {
  use crate::shardsc::SHSeq;
  use crate::shardsc::SHStrings;
  use crate::shardsc::SHTableTypeInfo;
  use crate::shardsc::SHType;
  use crate::shardsc::SHTypeInfo;
  use crate::shardsc::SHTypeInfo_Details;
  use crate::shardsc::SHType_Any;
  use crate::shardsc::SHType_Audio;
  use crate::shardsc::SHType_Bool;
  use crate::shardsc::SHType_Bytes;
  use crate::shardsc::SHType_Color;
  use crate::shardsc::SHType_ContextVar;
  use crate::shardsc::SHType_Enum;
  use crate::shardsc::SHType_Float;
  use crate::shardsc::SHType_Float2;
  use crate::shardsc::SHType_Float3;
  use crate::shardsc::SHType_Float4;
  use crate::shardsc::SHType_Image;
  use crate::shardsc::SHType_Int;
  use crate::shardsc::SHType_Int16;
  use crate::shardsc::SHType_Int2;
  use crate::shardsc::SHType_Int3;
  use crate::shardsc::SHType_Int4;
  use crate::shardsc::SHType_Int8;
  use crate::shardsc::SHType_None;
  use crate::shardsc::SHType_Object;
  use crate::shardsc::SHType_Path;
  use crate::shardsc::SHType_Seq;
  use crate::shardsc::SHType_ShardRef;
  use crate::shardsc::SHType_String;
  use crate::shardsc::SHType_Table;
  use crate::shardsc::SHType_Wire;
  use crate::shardsc::SHTypesInfo;
  use std::ffi::CStr;

  const fn base_info() -> SHTypeInfo {
    SHTypeInfo {
      basicType: SHType_None,
      details: SHTypeInfo_Details {
        seqTypes: SHTypesInfo {
          elements: core::ptr::null_mut(),
          len: 0,
          cap: 0,
        },
      },
      fixedSize: 0,
      innerType: SHType_None,
      recursiveSelf: false,
    }
  }

  pub static none: SHTypeInfo = base_info();

  pub fn type2name(value_type: SHType) -> &'static str {
    unsafe {
      let ptr = (*crate::core::Core).type2Name.unwrap_unchecked()(value_type);
      let s = CStr::from_ptr(ptr);
      s.to_str().unwrap()
    }
  }

  macro_rules! shtype {
    ($fname:ident, $type:expr, $name:ident, $names:ident, $name_var:ident, $names_var:ident, $name_table:ident, $name_table_var:ident) => {
      const fn $fname() -> SHTypeInfo {
        let mut res = base_info();
        res.basicType = $type;
        res
      }

      pub static $name: SHTypeInfo = $fname();

      pub static $names: SHTypeInfo = SHTypeInfo {
        basicType: SHType_Seq,
        details: SHTypeInfo_Details {
          seqTypes: SHTypesInfo {
            elements: &$name as *const SHTypeInfo as *mut SHTypeInfo,
            len: 1,
            cap: 0,
          },
        },
        fixedSize: 0,
        innerType: SHType_None,
        recursiveSelf: false,
      };

      pub static $name_table: SHTypeInfo = SHTypeInfo {
        basicType: SHType_Table,
        details: SHTypeInfo_Details {
          table: SHTableTypeInfo {
            keys: SHSeq {
              elements: core::ptr::null_mut(),
              len: 0,
              cap: 0,
            },
            types: SHTypesInfo {
              elements: &$name as *const SHTypeInfo as *mut SHTypeInfo,
              len: 1,
              cap: 0,
            },
          },
        },
        fixedSize: 0,
        innerType: SHType_None,
        recursiveSelf: false,
      };

      pub static $name_var: SHTypeInfo = SHTypeInfo {
        basicType: SHType_ContextVar,
        details: SHTypeInfo_Details {
          contextVarTypes: SHTypesInfo {
            elements: &$name as *const SHTypeInfo as *mut SHTypeInfo,
            len: 1,
            cap: 0,
          },
        },
        fixedSize: 0,
        innerType: SHType_None,
        recursiveSelf: false,
      };

      pub static $names_var: SHTypeInfo = SHTypeInfo {
        basicType: SHType_ContextVar,
        details: SHTypeInfo_Details {
          contextVarTypes: SHTypesInfo {
            elements: &$names as *const SHTypeInfo as *mut SHTypeInfo,
            len: 1,
            cap: 0,
          },
        },
        fixedSize: 0,
        innerType: SHType_None,
        recursiveSelf: false,
      };

      pub static $name_table_var: SHTypeInfo = SHTypeInfo {
        basicType: SHType_ContextVar,
        details: SHTypeInfo_Details {
          contextVarTypes: SHTypesInfo {
            elements: &$name_table as *const SHTypeInfo as *mut SHTypeInfo,
            len: 1,
            cap: 0,
          },
        },
        fixedSize: 0,
        innerType: SHType_None,
        recursiveSelf: false,
      };
    };
  }

  shtype!(
    make_any,
    SHType_Any,
    any,
    anys,
    any_var,
    anys_var,
    any_table,
    any_table_var
  );
  shtype!(
    make_object,
    SHType_Object,
    object,
    objects,
    object_var,
    objects_var,
    object_table,
    object_table_var
  );
  shtype!(
    make_enum,
    SHType_Enum,
    enumeration,
    enums,
    enum_var,
    enums_var,
    enum_table,
    enum_table_var
  );
  shtype!(
    make_string,
    SHType_String,
    string,
    strings,
    string_var,
    strings_var,
    string_table,
    string_table_var
  );
  shtype!(
    make_bytes,
    SHType_Bytes,
    bytes,
    bytezs,
    bytes_var,
    bytess_var,
    bytes_table,
    bytes_table_var
  );
  shtype!(
    make_image,
    SHType_Image,
    image,
    images,
    image_var,
    images_var,
    image_table,
    images_table_var
  );
  shtype!(
    make_int,
    SHType_Int,
    int,
    ints,
    int_var,
    ints_var,
    int_table,
    int_table_var
  );
  shtype!(
    make_int2,
    SHType_Int2,
    int2,
    int2s,
    int2_var,
    int2s_var,
    int2_table,
    int2_table_var
  );
  shtype!(
    make_int3,
    SHType_Int3,
    int3,
    int3s,
    int3_var,
    int3s_var,
    int3_table,
    int3_table_var
  );
  shtype!(
    make_int4,
    SHType_Int4,
    int4,
    int4s,
    int4_var,
    int4s_var,
    int4_table,
    int4_table_var
  );
  shtype!(
    make_int8,
    SHType_Int8,
    int8,
    int8s,
    int8_var,
    int8s_var,
    int8_table,
    int8_table_var
  );
  shtype!(
    make_int16,
    SHType_Int16,
    int16,
    int16s,
    int16_var,
    int16s_var,
    int16_table,
    int16_table_var
  );
  shtype!(
    make_float,
    SHType_Float,
    float,
    floats,
    float_var,
    floats_var,
    float_table,
    float_table_var
  );
  shtype!(
    make_float2,
    SHType_Float2,
    float2,
    float2s,
    float2_var,
    float2s_var,
    float2_table,
    float2_table_var
  );
  shtype!(
    make_float3,
    SHType_Float3,
    float3,
    float3s,
    float3_var,
    float3s_var,
    float3_table,
    float3_table_var
  );
  shtype!(
    make_float4,
    SHType_Float4,
    float4,
    float4s,
    float4_var,
    float4s_var,
    float4_table,
    float4_table_var
  );
  shtype!(
    make_color,
    SHType_Color,
    color,
    colors,
    color_var,
    colors_var,
    color_table,
    color_table_var
  );
  shtype!(
    make_bool,
    SHType_Bool,
    bool,
    bools,
    bool_var,
    bools_var,
    bool_table,
    bool_table_var
  );
  shtype!(
    make_shard,
    SHType_ShardRef,
    shard,
    shards,
    shard_var,
    shards_var,
    shard_table,
    shard_table_var
  );
  shtype!(
    make_wire,
    SHType_Wire,
    wire,
    wires,
    wire_var,
    wires_var,
    wire_table,
    wire_table_var
  );
  shtype!(
    make_path,
    SHType_Path,
    path,
    paths,
    path_var,
    paths_var,
    path_table,
    path_table_var
  );
  shtype!(
    make_audio,
    SHType_Audio,
    audio,
    audios,
    audio_var,
    audios_var,
    audio_table,
    audio_table_var
  );
}

impl Type {
  pub const fn context_variable(types: &[Type]) -> Type {
    Type {
      basicType: SHType_ContextVar,
      details: SHTypeInfo_Details {
        contextVarTypes: SHTypesInfo {
          elements: types.as_ptr() as *mut SHTypeInfo,
          len: types.len() as u32,
          cap: 0,
        },
      },
      fixedSize: 0,
      innerType: SHType_None,
      recursiveSelf: false,
    }
  }

  pub const fn enumeration(vendorId: i32, typeId: i32) -> Type {
    Type {
      basicType: SHType_Enum,
      details: SHTypeInfo_Details {
        enumeration: SHEnumTypeInfo { vendorId, typeId },
      },
      fixedSize: 0,
      innerType: SHType_None,
      recursiveSelf: false,
    }
  }

  pub const fn object(vendorId: i32, typeId: i32) -> Type {
    Type {
      basicType: SHType_Object,
      details: SHTypeInfo_Details {
        object: SHObjectTypeInfo {
          vendorId,
          typeId,
          extInfo: core::ptr::null(),
          extInfoData: core::ptr::null_mut(),
        },
      },
      fixedSize: 0,
      innerType: SHType_None,
      recursiveSelf: false,
    }
  }

  pub const fn table(keys: &[Var], types: &[Type]) -> Type {
    Type {
      basicType: SHType_Table,
      details: SHTypeInfo_Details {
        table: SHTableTypeInfo {
          keys: SHSeq {
            elements: keys.as_ptr() as *mut _,
            len: keys.len() as u32,
            cap: 0,
          },
          types: SHTypesInfo {
            elements: types.as_ptr() as *mut SHTypeInfo,
            len: types.len() as u32,
            cap: 0,
          },
        },
      },
      fixedSize: 0,
      innerType: SHType_None,
      recursiveSelf: false,
    }
  }

  pub const fn seq(types: &[Type]) -> Type {
    Type {
      basicType: SHType_Seq,
      details: SHTypeInfo_Details {
        seqTypes: SHTypesInfo {
          elements: types.as_ptr() as *mut SHTypeInfo,
          len: types.len() as u32,
          cap: 0,
        },
      },
      fixedSize: 0,
      innerType: SHType_None,
      recursiveSelf: false,
    }
  }
}

/*
SHVar utility
 */

impl Serialize for SeqVar {
  fn serialize<S>(
    &self,
    se: S,
  ) -> Result<<S as serde::Serializer>::Ok, <S as serde::Serializer>::Error>
  where
    S: serde::Serializer,
  {
    let mut s = se.serialize_seq(Some(self.len()))?;
    for ref value in self.iter() {
      s.serialize_element(value)?;
    }
    s.end()
  }
}

impl Serialize for TableVar {
  fn serialize<S>(
    &self,
    se: S,
  ) -> Result<<S as serde::Serializer>::Ok, <S as serde::Serializer>::Error>
  where
    S: serde::Serializer,
  {
    let mut t = se.serialize_map(None)?;
    for (key, value) in self.iter() {
      let key: &str = key.as_ref().try_into().map_err(serde::ser::Error::custom)?;
      t.serialize_entry(&key, &value)?;
    }
    t.end()
  }
}

impl Serialize for Var {
  fn serialize<S>(
    &self,
    se: S,
  ) -> Result<<S as serde::Serializer>::Ok, <S as serde::Serializer>::Error>
  where
    S: serde::Serializer,
  {
    match self.valueType {
      SHType_None | SHType_Any => {
        let mut s = se.serialize_seq(Some(1))?;
        s.serialize_element(&self.valueType)?;
        s.end()
      }
      SHType_Enum => {
        let value: i32 = unsafe { self.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue };
        let vendor: i32 = unsafe { self.payload.__bindgen_anon_1.__bindgen_anon_3.enumVendorId };
        let type_: i32 = unsafe { self.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId };
        let mut s = se.serialize_seq(Some(4))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.serialize_element(&vendor)?;
        s.serialize_element(&type_)?;
        s.end()
      }
      SHType_Bool => {
        let value: bool = unsafe { self.payload.__bindgen_anon_1.boolValue };
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_Int => {
        let value: i64 = unsafe { self.payload.__bindgen_anon_1.intValue };
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_Int2 => {
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        unsafe {
          s.serialize_element(&self.payload.__bindgen_anon_1.int2Value)?;
        }
        s.end()
      }
      SHType_Int3 | SHType_Int4 => {
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        unsafe {
          s.serialize_element(&self.payload.__bindgen_anon_1.int4Value)?;
        }
        s.end()
      }
      SHType_Int8 => {
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        unsafe {
          s.serialize_element(&self.payload.__bindgen_anon_1.int8Value)?;
        }
        s.end()
      }
      SHType_Int16 => {
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        unsafe {
          s.serialize_element(&self.payload.__bindgen_anon_1.int16Value)?;
        }
        s.end()
      }
      SHType_Float => {
        let value: f64 = unsafe { self.payload.__bindgen_anon_1.floatValue };
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_Float2 => {
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        unsafe {
          s.serialize_element(&self.payload.__bindgen_anon_1.float2Value)?;
        }
        s.end()
      }
      SHType_Float3 | SHType_Float4 => {
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        unsafe {
          s.serialize_element(&self.payload.__bindgen_anon_1.float4Value)?;
        }
        s.end()
      }
      SHType_Color => {
        let value0: u8 = unsafe { self.payload.__bindgen_anon_1.colorValue.r };
        let value1: u8 = unsafe { self.payload.__bindgen_anon_1.colorValue.g };
        let value2: u8 = unsafe { self.payload.__bindgen_anon_1.colorValue.b };
        let value3: u8 = unsafe { self.payload.__bindgen_anon_1.colorValue.a };
        let arr = [value0, value1, value2, value3];
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&arr)?;
        s.end()
      }
      SHType_Bytes => {
        let value: &[u8] = self.try_into().unwrap();
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_String => {
        let value: &str = self.try_into().unwrap();
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_Path => {
        let value: &str = self.try_into().unwrap();
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_ContextVar => {
        let value: &str = self.try_into().unwrap();
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&value)?;
        s.end()
      }
      SHType_Image => {
        let image = unsafe {
          self
            .payload
            .__bindgen_anon_1
            .imageValue
            .as_ref()
            .unwrap_unchecked()
        };
        let width = image.width;
        let height = image.height;
        let channels = image.channels;
        let flags = image.flags;
        let data = image.data;
        let pixsize = if (flags as u32 & SHIMAGE_FLAGS_16BITS_INT) == SHIMAGE_FLAGS_16BITS_INT {
          2
        } else if (flags as u32 & SHIMAGE_FLAGS_32BITS_FLOAT) == SHIMAGE_FLAGS_32BITS_FLOAT {
          4
        } else {
          1
        };
        let data = unsafe {
          if data.is_null() {
            &[]
          } else {
            std::slice::from_raw_parts(
              data,
              (width as usize * height as usize * channels as usize * pixsize as usize) as usize,
            )
          }
        };
        let mut s = se.serialize_seq(Some(6))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&width)?;
        s.serialize_element(&height)?;
        s.serialize_element(&channels)?;
        s.serialize_element(&flags)?;
        s.serialize_element(&data)?;
        s.end()
      }
      SHType_Seq => {
        let seq: SeqVar = self.try_into().unwrap();
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&seq)?;
        s.end()
      }
      SHType_Table => {
        let table: TableVar = self.try_into().unwrap();
        let mut s = se.serialize_seq(Some(2))?;
        s.serialize_element(&self.valueType)?;
        s.serialize_element(&table)?;
        s.end()
      }
      _ => Err(serde::ser::Error::custom("Unsupported Var type")),
    }
  }
}

impl<'de> Deserialize<'de> for AutoSeqVar {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct SeqVisitor;

    impl<'de> Visitor<'de> for SeqVisitor {
      type Value = AutoSeqVar;

      fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a supported Seq value")
      }

      fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
      where
        A: SeqAccess<'de>,
      {
        let mut dst = AutoSeqVar::new();
        while let Some(var) = seq.next_element::<ClonedVar>()? {
          dst.0.push(&var.0);
        }
        Ok(dst)
      }
    }

    deserializer.deserialize_seq(SeqVisitor)
  }
}

impl<'de> Deserialize<'de> for AutoTableVar {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct TableVisitor;

    impl<'de> Visitor<'de> for TableVisitor {
      type Value = AutoTableVar;

      fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a supported Table value")
      }

      fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
      where
        A: MapAccess<'de>,
      {
        let mut table = AutoTableVar::new();
        while let Some((key, value)) = map.next_entry::<&str, ClonedVar>()? {
          let key = Var::ephemeral_string(key);
          table.0.insert_fast(key, &value.0);
        }
        Ok(table)
      }
    }

    deserializer.deserialize_map(TableVisitor)
  }
}

impl<'de> Deserialize<'de> for ClonedVar {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct VarVisitor;

    impl<'de> Visitor<'de> for VarVisitor {
      type Value = ClonedVar;

      fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a supported Var value")
      }

      fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
      where
        A: SeqAccess<'de>,
      {
        let type_: u8 = seq.next_element()?.unwrap();
        let mut v = Var::default();
        v.valueType = type_;
        match type_ {
          SHType_None | SHType_Any => {}
          SHType_Enum => {
            let value: i32 = seq.next_element()?.unwrap();
            let vendor: i32 = seq.next_element()?.unwrap();
            let enum_type: i32 = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue = value;
            v.payload.__bindgen_anon_1.__bindgen_anon_3.enumVendorId = vendor;
            v.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId = enum_type;
          }
          SHType_Bool => {
            let value: bool = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.boolValue = value;
          }
          SHType_Int => {
            let value: i64 = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.intValue = value;
          }
          SHType_Int2 => {
            let value: [i64; 2] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.int2Value = value;
          }
          SHType_Int3 | SHType_Int4 => {
            let value: [i32; 4] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.int4Value = value;
          }
          SHType_Int8 => {
            let value: [i16; 8] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.int8Value = value;
          }
          SHType_Int16 => {
            let value: [i8; 16] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.int16Value = value;
          }
          SHType_Float => {
            let value: f64 = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.floatValue = value;
          }
          SHType_Float2 => {
            let value: [f64; 2] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.float2Value = value;
          }
          SHType_Float3 | SHType_Float4 => {
            let value: [f32; 4] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.float4Value = value;
          }
          SHType_Color => {
            let value: [u8; 4] = seq.next_element()?.unwrap();
            v.payload.__bindgen_anon_1.colorValue.r = value[0];
            v.payload.__bindgen_anon_1.colorValue.g = value[1];
            v.payload.__bindgen_anon_1.colorValue.b = value[2];
            v.payload.__bindgen_anon_1.colorValue.a = value[3];
          }
          SHType_Bytes => {
            let value: &[u8] = seq.next_element()?.unwrap();
            let len = value.len();
            let ptr = value.as_ptr();
            v.payload.__bindgen_anon_1.__bindgen_anon_4.bytesValue = ptr as *mut u8;
            v.payload.__bindgen_anon_1.__bindgen_anon_4.bytesSize = len as u32;
          }
          SHType_String | SHType_Path | SHType_ContextVar => {
            let value: &str = seq.next_element()?.unwrap();
            let value = Var::ephemeral_string(value);
            v = value.into();
          }
          SHType_Image => {
            return ClonedVar::new_image(
              seq.next_element()?.unwrap(),
              seq.next_element()?.unwrap(),
              seq.next_element()?.unwrap(),
              seq.next_element()?.unwrap(),
              seq.next_element()?.unwrap(),
            )
            .map_err(|s| serde::de::Error::custom(s));
          }
          SHType_Seq => {
            let seq: AutoSeqVar = seq.next_element()?.unwrap();
            // just reinterpret the sequence as a ClonedVar! (this is safe)
            return Ok(unsafe { std::mem::transmute(seq) });
          }
          SHType_Table => {
            let table: AutoTableVar = seq.next_element()?.unwrap();
            // just reinterpret the sequence as a ClonedVar! (this is safe)
            return Ok(unsafe { std::mem::transmute(table) });
          }
          _ => return Err(serde::de::Error::custom("Unsupported Var type")),
        }
        Ok(v.into())
      }
    }

    deserializer.deserialize_seq(VarVisitor)
  }
}

impl<T> From<T> for ClonedVar
where
  T: Into<Var>,
{
  #[inline]
  fn from(v: T) -> Self {
    let vt: Var = v.into();
    let res = ClonedVar(Var::default());
    unsafe {
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = &vt as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    res
  }
}

impl<T> From<T> for ExternalVar
where
  T: Into<Var>,
{
  #[inline]
  fn from(v: T) -> Self {
    let vt: Var = v.into();
    let mut res = ExternalVar(Var::default());
    unsafe {
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = &vt as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    // ensure this flag is set
    res.0.flags |= SHVAR_FLAGS_EXTERNAL as u16;
    res
  }
}

impl From<&Var> for ClonedVar {
  fn from(v: &Var) -> Self {
    let res = ClonedVar(Var::default());
    unsafe {
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = v as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    res
  }
}

impl From<&Var> for ExternalVar {
  fn from(v: &Var) -> Self {
    let mut res = ExternalVar(Var::default());
    unsafe {
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = v as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    // ensure this flag is set
    res.0.flags |= SHVAR_FLAGS_EXTERNAL as u16;
    res
  }
}

impl From<Vec<Var>> for ClonedVar {
  fn from(v: Vec<Var>) -> Self {
    let tmp = Var::from(&v);
    let res = ClonedVar(Var::default());
    unsafe {
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = &tmp as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    res
  }
}

impl From<std::string::String> for ClonedVar {
  fn from(v: std::string::String) -> Self {
    let cstr = CString::new(v).unwrap();
    let tmp = Var::from(&cstr);
    let res = ClonedVar(Var::default());
    unsafe {
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = &tmp as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    res
  }
}

impl From<&[ClonedVar]> for ClonedVar {
  fn from(vec: &[ClonedVar]) -> Self {
    let res = ClonedVar(Var::default());
    unsafe {
      let src: &[Var] = &*(vec as *const [ClonedVar] as *const [SHVar]);
      let vsrc: Var = src.into();
      let rv = &res.0 as *const SHVar as *mut SHVar;
      let sv = &vsrc as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    res
  }
}

impl Default for ExternalVar {
  #[inline(always)]
  fn default() -> Self {
    let mut res = ExternalVar(Var::default());
    res.0.flags |= SHVAR_FLAGS_EXTERNAL as u16;
    res
  }
}

impl ExternalVar {
  #[inline(always)]
  pub fn update<T>(&mut self, value: T)
  where
    T: Into<Var>,
  {
    let vt: Var = value.into();
    unsafe {
      let rv = &self.0 as *const SHVar as *mut SHVar;
      let sv = &vt as *const SHVar;
      (*Core).cloneVar.unwrap_unchecked()(rv, sv);
    }
    // ensure this flag is set
    self.0.flags |= SHVAR_FLAGS_EXTERNAL as u16;
  }
}

impl Drop for ExternalVar {
  #[inline(always)]
  fn drop(&mut self) {
    unsafe {
      let rv = &self.0 as *const SHVar as *mut SHVar;
      (*Core).destroyVar.unwrap_unchecked()(rv);
    }
  }
}

macro_rules! var_from {
  ($type:ident, $varfield:ident, $shtype:expr) => {
    impl From<$type> for Var {
      #[inline(always)]
      fn from(v: $type) -> Self {
        SHVar {
          valueType: $shtype,
          payload: SHVarPayload {
            __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { $varfield: v },
          },
          ..Default::default()
        }
      }
    }
  };
}

macro_rules! var_from_as_i64 {
  ($type:ident, $varfield:ident, $shtype:expr) => {
    impl From<$type> for Var {
      #[inline(always)]
      fn from(v: $type) -> Self {
        SHVar {
          valueType: $shtype,
          payload: SHVarPayload {
            __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
              $varfield: v as i64,
            },
          },
          ..Default::default()
        }
      }
    }
  };
}

macro_rules! var_from_as_f64 {
  ($type:ident, $varfield:ident, $shtype:expr) => {
    impl From<$type> for Var {
      #[inline(always)]
      fn from(v: $type) -> Self {
        SHVar {
          valueType: $shtype,
          payload: SHVarPayload {
            __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
              $varfield: v as f64,
            },
          },
          ..Default::default()
        }
      }
    }
  };
}

macro_rules! var_try_from {
  ($type:ident, $varfield:ident, $shtype:expr) => {
    impl TryFrom<$type> for Var {
      type Error = &'static str;

      #[inline(always)]
      fn try_from(v: $type) -> Result<Self, Self::Error> {
        Ok(SHVar {
          valueType: $shtype,
          payload: SHVarPayload {
            __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
              $varfield: v
                .try_into()
                .map_err(|_| "Conversion failed, value out of range")?,
            },
          },
          ..Default::default()
        })
      }
    }
  };
}

var_from!(bool, boolValue, SHType_Bool);
var_from!(i64, intValue, SHType_Int);
var_from!(SHColor, colorValue, SHType_Color);
var_from_as_i64!(u8, intValue, SHType_Int);
var_from_as_i64!(u16, intValue, SHType_Int);
var_from_as_i64!(u32, intValue, SHType_Int);
var_from_as_i64!(i32, intValue, SHType_Int);
var_try_from!(u128, intValue, SHType_Int);
var_try_from!(i128, intValue, SHType_Int);
var_try_from!(usize, intValue, SHType_Int);
// var_try_from!(u64, intValue, SHType_Int); // Don't panic!
var_from!(f64, floatValue, SHType_Float);
var_from_as_f64!(f32, floatValue, SHType_Float);

impl From<ShardPtr> for Var {
  #[inline(always)]
  fn from(v: ShardPtr) -> Self {
    SHVar {
      valueType: SHType_ShardRef,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { shardValue: v },
      },
      ..Default::default()
    }
  }
}

impl From<SHString> for Var {
  #[inline(always)]
  fn from(v: SHString) -> Self {
    SHVar {
      valueType: SHType_String,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_2: SHVarPayload__bindgen_ty_1__bindgen_ty_2 {
            stringValue: v,
            stringLen: 0,
            stringCapacity: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

impl From<String> for &str {
  #[inline(always)]
  fn from(v: String) -> Self {
    unsafe {
      let cstr = CStr::from_ptr(v.0);
      cstr.to_str().unwrap()
    }
  }
}

impl From<&CStr> for Var {
  #[inline(always)]
  fn from(v: &CStr) -> Self {
    SHVar {
      valueType: SHType_String,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_2: SHVarPayload__bindgen_ty_1__bindgen_ty_2 {
            stringValue: v.as_ptr(),
            stringLen: v.to_bytes().len() as u32,
            stringCapacity: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

// 64-bit precision :[i64;2]
impl From<(i64, i64)> for Var {
  #[inline(always)]
  fn from(v: (i64, i64)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = v.0;
      res.payload.__bindgen_anon_1.int2Value[1] = v.1;
    }
    res
  }
}

impl From<&[i64; 2]> for Var {
  #[inline(always)]
  fn from(v: &[i64; 2]) -> Self {
    let mut res = Var {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = v[0];
      res.payload.__bindgen_anon_1.int2Value[1] = v[1];
    }
    res
  }
}

// 64-bit precision :u64
impl From<u64> for Var {
  #[inline(always)]
  fn from(v: u64) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int,
      ..Default::default()
    };
    res.payload.__bindgen_anon_1.intValue = i64::from_ne_bytes((v).to_ne_bytes());
    res
  }
}

// 64-bit precision :[u64;2]
impl From<(u64, u64)> for Var {
  #[inline(always)]
  fn from(v: (u64, u64)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = i64::from_ne_bytes((v.0).to_ne_bytes());
      res.payload.__bindgen_anon_1.int2Value[1] = i64::from_ne_bytes((v.1).to_ne_bytes());
    }
    res
  }
}

// 64-bit precision :[f64;2]
impl From<(f64, f64)> for Var {
  #[inline(always)]
  fn from(v: (f64, f64)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float2Value[0] = v.0;
      res.payload.__bindgen_anon_1.float2Value[1] = v.1;
    }
    res
  }
}

impl From<&[f64; 2]> for Var {
  #[inline(always)]
  fn from(v: &[f64; 2]) -> Self {
    let mut res = Var {
      valueType: crate::shardsc::SHType_Float2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float2Value[0] = v[0];
      res.payload.__bindgen_anon_1.float2Value[1] = v[1];
    }
    res
  }
}

// 32-bit precision :[i32;2]
impl From<(i32, i32)> for Var {
  #[inline(always)]
  fn from(v: (i32, i32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = v.0 as i64;
      res.payload.__bindgen_anon_1.int2Value[1] = v.1 as i64;
    }
    res
  }
}

// 32-bit precision :[i32;3]
impl From<(i32, i32, i32)> for Var {
  #[inline(always)]
  fn from(v: (i32, i32, i32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int3Value[0] = v.0;
      res.payload.__bindgen_anon_1.int3Value[1] = v.1;
      res.payload.__bindgen_anon_1.int3Value[2] = v.2;
    }
    res
  }
}

impl From<&[i32; 3]> for Var {
  #[inline(always)]
  fn from(v: &[i32; 3]) -> Self {
    let mut res = Var {
      valueType: SHType_Int3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int3Value[0] = v[0];
      res.payload.__bindgen_anon_1.int3Value[1] = v[1];
      res.payload.__bindgen_anon_1.int3Value[2] = v[2];
    }
    res
  }
}

// 32-bit precision :[i32;4]
impl From<(i32, i32, i32, i32)> for Var {
  #[inline(always)]
  fn from(v: (i32, i32, i32, i32)) -> Self {
    Var {
      valueType: SHType_Int4,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          int4Value: [v.0, v.1, v.2, v.3],
        },
      },
      ..Default::default()
    }
  }
}

impl From<&[i32; 4]> for Var {
  #[inline(always)]
  fn from(v: &[i32; 4]) -> Self {
    Var {
      valueType: SHType_Int4,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { int4Value: *v },
      },
      ..Default::default()
    }
  }
}

impl From<&[i16; 8]> for Var {
  #[inline(always)]
  fn from(v: &[i16; 8]) -> Self {
    Var {
      valueType: SHType_Int8,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { int8Value: *v },
      },
      ..Default::default()
    }
  }
}

impl From<&[i8; 16]> for Var {
  #[inline(always)]
  fn from(v: &[i8; 16]) -> Self {
    Var {
      valueType: SHType_Int16,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { int16Value: *v },
      },
      ..Default::default()
    }
  }
}

impl From<&[u8; 16]> for Var {
  #[inline(always)]
  fn from(v: &[u8; 16]) -> Self {
    let val = v as *const [u8; 16] as *const [i8; 16];
    let val = unsafe { *val };
    Var {
      valueType: SHType_Int16,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { int16Value: val },
      },
      ..Default::default()
    }
  }
}

impl TryFrom<&Var> for [i8; 16] {
  type Error = &'static str;

  // convert int16Value into
  #[inline(always)]
  fn try_from(v: &Var) -> Result<Self, Self::Error> {
    if v.valueType != SHType_Int16 {
      return Err("Invalid type");
    }
    let val = unsafe { v.payload.__bindgen_anon_1.int16Value };
    Ok(val)
  }
}

impl TryFrom<&Var> for [u8; 16] {
  type Error = &'static str;

  // convert int16Value into
  #[inline(always)]
  fn try_from(v: &Var) -> Result<Self, Self::Error> {
    if v.valueType != SHType_Int16 {
      return Err("Invalid type");
    }
    let val = unsafe { v.payload.__bindgen_anon_1.int16Value };
    let val = &val as *const [i8; 16] as *const [u8; 16];
    let val = unsafe { *val };
    Ok(val)
  }
}

// 32-bit precision :[u32;2]
impl From<(u32, u32)> for Var {
  #[inline(always)]
  fn from(v: (u32, u32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = i32::from_ne_bytes((v.0).to_ne_bytes()) as i64;
      res.payload.__bindgen_anon_1.int2Value[1] = i32::from_ne_bytes((v.1).to_ne_bytes()) as i64;
    }
    res
  }
}

// 32-bit precision :[u32;3]
impl From<(u32, u32, u32)> for Var {
  #[inline(always)]
  fn from(v: (u32, u32, u32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int3Value[0] = i32::from_ne_bytes((v.0).to_ne_bytes());
      res.payload.__bindgen_anon_1.int3Value[1] = i32::from_ne_bytes((v.1).to_ne_bytes());
      res.payload.__bindgen_anon_1.int3Value[2] = i32::from_ne_bytes((v.2).to_ne_bytes());
    }
    res
  }
}

// 32-bit precision :[u32;4]
impl From<(u32, u32, u32, u32)> for Var {
  #[inline(always)]
  fn from(v: (u32, u32, u32, u32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int4,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int4Value[0] = i32::from_ne_bytes((v.0).to_ne_bytes());
      res.payload.__bindgen_anon_1.int4Value[1] = i32::from_ne_bytes((v.1).to_ne_bytes());
      res.payload.__bindgen_anon_1.int4Value[2] = i32::from_ne_bytes((v.2).to_ne_bytes());
      res.payload.__bindgen_anon_1.int4Value[3] = i32::from_ne_bytes((v.3).to_ne_bytes());
    }
    res
  }
}

// 32-bit precision :[f32;2]
impl From<(f32, f32)> for Var {
  #[inline(always)]
  fn from(v: (f32, f32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float2Value[0] = v.0 as f64;
      res.payload.__bindgen_anon_1.float2Value[1] = v.1 as f64;
    }
    res
  }
}

// 32-bit precision :[f32;3]
impl From<(f32, f32, f32)> for Var {
  #[inline(always)]
  fn from(v: (f32, f32, f32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float3Value[0] = v.0;
      res.payload.__bindgen_anon_1.float3Value[1] = v.1;
      res.payload.__bindgen_anon_1.float3Value[2] = v.2;
    }
    res
  }
}

impl From<&[f32; 3]> for Var {
  #[inline(always)]
  fn from(v: &[f32; 3]) -> Self {
    let mut res = Var {
      valueType: crate::shardsc::SHType_Float3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float3Value[0] = v[0];
      res.payload.__bindgen_anon_1.float3Value[1] = v[1];
      res.payload.__bindgen_anon_1.float3Value[2] = v[2];
    }
    res
  }
}

// 32-bit precision :[f32;4]
impl From<(f32, f32, f32, f32)> for Var {
  #[inline(always)]
  fn from(v: (f32, f32, f32, f32)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float4,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float4Value[0] = v.0;
      res.payload.__bindgen_anon_1.float4Value[1] = v.1;
      res.payload.__bindgen_anon_1.float4Value[2] = v.2;
      res.payload.__bindgen_anon_1.float4Value[3] = v.3;
    }
    res
  }
}

impl From<&[f32; 4]> for Var {
  #[inline(always)]
  fn from(v: &[f32; 4]) -> Self {
    let mut res = Var {
      valueType: crate::shardsc::SHType_Float4,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float4Value[0] = v[0];
      res.payload.__bindgen_anon_1.float4Value[1] = v[1];
      res.payload.__bindgen_anon_1.float4Value[2] = v[2];
      res.payload.__bindgen_anon_1.float4Value[3] = v[3];
    }
    res
  }
}

// 16-bit precision :[i16;2]
impl From<(i16, i16)> for Var {
  #[inline(always)]
  fn from(v: (i16, i16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = v.0 as i64;
      res.payload.__bindgen_anon_1.int2Value[1] = v.1 as i64;
    }
    res
  }
}

// 16-bit precision :[i16;3]
impl From<(i16, i16, i16)> for Var {
  #[inline(always)]
  fn from(v: (i16, i16, i16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int3Value[0] = v.0 as i32;
      res.payload.__bindgen_anon_1.int3Value[1] = v.1 as i32;
      res.payload.__bindgen_anon_1.int3Value[2] = v.2 as i32;
    }
    res
  }
}

// 16-bit precision :[i16;4]
impl From<(i16, i16, i16, i16)> for Var {
  #[inline(always)]
  fn from(v: (i16, i16, i16, i16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int4,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int4Value[0] = v.0 as i32;
      res.payload.__bindgen_anon_1.int4Value[1] = v.1 as i32;
      res.payload.__bindgen_anon_1.int4Value[2] = v.2 as i32;
      res.payload.__bindgen_anon_1.int4Value[3] = v.3 as i32;
    }
    res
  }
}

// 16-bit precision :[u16;2]
impl From<(u16, u16)> for Var {
  #[inline(always)]
  fn from(v: (u16, u16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int2Value[0] = i16::from_ne_bytes((v.0).to_ne_bytes()) as i64;
      res.payload.__bindgen_anon_1.int2Value[1] = i16::from_ne_bytes((v.1).to_ne_bytes()) as i64;
    }
    res
  }
}

// 16-bit precision :[u16;3]
impl From<(u16, u16, u16)> for Var {
  #[inline(always)]
  fn from(v: (u16, u16, u16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int3Value[0] = i16::from_ne_bytes((v.0).to_ne_bytes()) as i32;
      res.payload.__bindgen_anon_1.int3Value[1] = i16::from_ne_bytes((v.1).to_ne_bytes()) as i32;
      res.payload.__bindgen_anon_1.int3Value[2] = i16::from_ne_bytes((v.2).to_ne_bytes()) as i32;
    }
    res
  }
}

// 16-bit precision :[u16;4]
impl From<(u16, u16, u16, u16)> for Var {
  #[inline(always)]
  fn from(v: (u16, u16, u16, u16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Int4,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.int4Value[0] = i16::from_ne_bytes((v.0).to_ne_bytes()) as i32;
      res.payload.__bindgen_anon_1.int4Value[1] = i16::from_ne_bytes((v.1).to_ne_bytes()) as i32;
      res.payload.__bindgen_anon_1.int4Value[2] = i16::from_ne_bytes((v.2).to_ne_bytes()) as i32;
      res.payload.__bindgen_anon_1.int4Value[3] = i16::from_ne_bytes((v.3).to_ne_bytes()) as i32;
    }
    res
  }
}

// 16-bit precision :[f16;2]
impl From<(half::f16, half::f16)> for Var {
  #[inline(always)]
  fn from(v: (half::f16, half::f16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float2,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float2Value[0] = f64::from(v.0);
      res.payload.__bindgen_anon_1.float2Value[1] = f64::from(v.1);
    }
    res
  }
}

// 16-bit precision :[f16;3]
impl From<(half::f16, half::f16, half::f16)> for Var {
  #[inline(always)]
  fn from(v: (half::f16, half::f16, half::f16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float3,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float3Value[0] = f32::from(v.0);
      res.payload.__bindgen_anon_1.float3Value[1] = f32::from(v.1);
      res.payload.__bindgen_anon_1.float3Value[2] = f32::from(v.2);
    }
    res
  }
}

// 16-bit precision :[f16;4]
impl From<(half::f16, half::f16, half::f16, half::f16)> for Var {
  #[inline(always)]
  fn from(v: (half::f16, half::f16, half::f16, half::f16)) -> Self {
    let mut res = SHVar {
      valueType: SHType_Float4,
      ..Default::default()
    };
    unsafe {
      res.payload.__bindgen_anon_1.float4Value[0] = f32::from(v.0);
      res.payload.__bindgen_anon_1.float4Value[1] = f32::from(v.1);
      res.payload.__bindgen_anon_1.float4Value[2] = f32::from(v.2);
      res.payload.__bindgen_anon_1.float4Value[3] = f32::from(v.3);
    }
    res
  }
}

impl From<&CString> for Var {
  #[inline(always)]
  fn from(v: &CString) -> Self {
    SHVar {
      valueType: SHType_String,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_2: SHVarPayload__bindgen_ty_1__bindgen_ty_2 {
            stringValue: v.as_ptr(),
            stringLen: v.as_bytes().len() as u32,
            stringCapacity: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

impl From<Option<&CString>> for Var {
  #[inline(always)]
  fn from(v: Option<&CString>) -> Self {
    if let Some(v) = v {
      Var::from(v)
    } else {
      Var::default()
    }
  }
}

impl From<()> for Var {
  #[inline(always)]
  fn from(_: ()) -> Self {
    SHVar {
      valueType: SHType_None,
      ..Default::default()
    }
  }
}

impl From<&Vec<Var>> for Var {
  #[inline(always)]
  fn from(vec: &Vec<Var>) -> Self {
    SHVar {
      valueType: SHType_Seq,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          seqValue: SHSeq {
            elements: vec.as_ptr() as *mut SHVar,
            len: vec.len() as u32,
            cap: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

impl From<&Vec<ClonedVar>> for Var {
  #[inline(always)]
  fn from(vec: &Vec<ClonedVar>) -> Self {
    SHVar {
      valueType: SHType_Seq,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          seqValue: SHSeq {
            elements: vec.as_ptr() as *mut SHVar,
            len: vec.len() as u32,
            cap: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

impl From<&[Var]> for Var {
  #[inline(always)]
  fn from(vec: &[Var]) -> Self {
    SHVar {
      valueType: SHType_Seq,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          seqValue: SHSeq {
            elements: vec.as_ptr() as *mut SHVar,
            len: vec.len() as u32,
            cap: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

impl From<&[u8]> for Var {
  #[inline(always)]
  fn from(vec: &[u8]) -> Self {
    SHVar {
      valueType: SHType_Bytes,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_4: SHVarPayload__bindgen_ty_1__bindgen_ty_4 {
            bytesValue: vec.as_ptr() as *mut u8,
            bytesSize: vec.len() as u32,
            bytesCapacity: 0,
          },
        },
      },
      ..Default::default()
    }
  }
}

pub trait RCObjectVar {
  fn get_info() -> *mut SHObjectInfo;
}

pub struct RefCounted<T> {
  rc: AtomicU32,
  weak_rc: AtomicU32,
  value: Option<T>,
}

impl<T> RefCounted<T> {
  pub fn new(value: T) -> Self {
    Self {
      rc: AtomicU32::new(1),
      weak_rc: AtomicU32::new(1), // 1 weak reference to keep the object alive until last weak ref is gone
      value: Some(value),
    }
  }

  pub fn inc_ref(&self) {
    let prev_count = self.rc.fetch_add(1, Ordering::SeqCst);
    assert!(prev_count < u32::MAX, "Reference count overflowed");
  }

  pub fn dec_ref(&mut self) -> u32 {
    let prev_count = self.rc.fetch_sub(1, Ordering::SeqCst);
    assert!(prev_count > 0, "Reference count underflowed");

    if prev_count == 1 {
      self.value = None; // Drop the value when the last strong reference is released
      self.dec_weak_ref(); // Decrease the weak reference count when the strong count goes to 0
    }

    prev_count - 1
  }

  pub fn inc_weak_ref(&self) {
    let prev_count = self.weak_rc.fetch_add(1, Ordering::SeqCst);
    assert!(prev_count < u32::MAX, "Weak reference count overflowed");
  }

  pub fn dec_weak_ref(&self) -> u32 {
    let prev_count = self.weak_rc.fetch_sub(1, Ordering::SeqCst);
    assert!(prev_count > 0, "Weak reference count underflowed");

    if prev_count == 1 && self.rc.load(Ordering::SeqCst) == 0 {
      // Free the object when the last weak reference is released and no strong references exist
      unsafe {
        drop(Box::from_raw(self as *const _ as *mut Self));
      }
    }

    prev_count - 1
  }

  pub fn upgrade_weak(&self) -> Option<NonNull<Self>> {
    loop {
      let strong_count = self.rc.load(Ordering::SeqCst);
      if strong_count == 0 {
        return None;
      }
      if self
        .rc
        .compare_exchange(
          strong_count,
          strong_count + 1,
          Ordering::SeqCst,
          Ordering::SeqCst,
        )
        .is_ok()
      {
        self.dec_weak_ref();
        return Some(NonNull::from(self));
      }
    }
  }
}

#[macro_export]
macro_rules! ref_counted_object_type_impl {
  ($type:ident) => {
    lazy_static! {
      static ref TYPE_OBJECT_NAME: &'static str = concat!(stringify!($type), "\0");
      static ref TYPE_OBJECT_INFO: shards::SHObjectInfo = {
        unsafe extern "C" fn reference(arg1: *mut std::os::raw::c_void) {
          let rc = arg1 as *mut shards::types::RefCounted<$type>;
          (*rc).inc_ref();
        }

        unsafe extern "C" fn release(arg1: *mut std::os::raw::c_void) {
          let rc = arg1 as *mut shards::types::RefCounted<$type>;
          (*rc).dec_ref();
        }

        unsafe extern "C" fn weak_reference(arg1: *mut std::os::raw::c_void) {
          let rc = arg1 as *mut shards::types::RefCounted<$type>;
          (*rc).inc_weak_ref();
        }

        unsafe extern "C" fn weak_release(arg1: *mut std::os::raw::c_void) {
          let rc = arg1 as *mut shards::types::RefCounted<$type>;
          (*rc).dec_weak_ref();
        }

        unsafe extern "C" fn upgrade_weak(
          arg1: *mut std::os::raw::c_void,
        ) -> *mut std::os::raw::c_void {
          let rc = arg1 as *mut shards::types::RefCounted<$type>;
          (*rc)
            .upgrade_weak()
            .map_or(std::ptr::null_mut(), |ptr| ptr.as_ptr() as *mut _)
        }

        shards::SHObjectInfo {
          name: TYPE_OBJECT_NAME.as_ptr() as *const ::core::ffi::c_char,
          serialize: None,
          free: None,
          deserialize: None,
          reference: Some(reference),
          release: Some(release),
          weakReference: Some(weak_reference),
          weakRelease: Some(weak_release),
          upgradeWeak: Some(upgrade_weak),
          hash: None,
          isThreadSafe: false,
        }
      };
    }

    impl shards::types::RCObjectVar for $type {
      fn get_info() -> *mut shards::SHObjectInfo {
        &*TYPE_OBJECT_INFO as *const _ as *mut _
      }
    }
  };
}

impl Var {
  pub fn new_none() -> Self {
    Var {
      valueType: SHType_None,
      ..Default::default()
    }
  }

  pub fn new_any() -> Self {
    Var {
      valueType: SHType_Any,
      ..Default::default()
    }
  }

  pub fn new_bool(b: bool) -> Self {
    Var {
      valueType: SHType_Bool,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { boolValue: b },
      },
      ..Default::default()
    }
  }

  pub fn new_float(f: f64) -> Self {
    Var {
      valueType: SHType_Float,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { floatValue: f },
      },
      ..Default::default()
    }
  }

  pub fn new_int(i: i64) -> Self {
    Var {
      valueType: SHType_Int,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { intValue: i },
      },
      ..Default::default()
    }
  }

  pub fn new_int2(i0: i64, i1: i64) -> Self {
    Var {
      valueType: SHType_Int2,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          int2Value: [i0, i1],
        },
      },
      ..Default::default()
    }
  }

  pub fn new_int3(i0: i32, i1: i32, i2: i32) -> Self {
    Var {
      valueType: SHType_Int3,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          int3Value: [i0, i1, i2, 0],
        },
      },
      ..Default::default()
    }
  }

  pub fn new_int4(i0: i32, i1: i32, i2: i32, i3: i32) -> Self {
    Var {
      valueType: SHType_Int4,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          int4Value: [i0, i1, i2, i3],
        },
      },
      ..Default::default()
    }
  }

  pub fn new_int8(values: [i16; 8]) -> Self {
    Var {
      valueType: SHType_Int8,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { int8Value: values },
      },
      ..Default::default()
    }
  }

  pub fn new_int16(values: [i8; 16]) -> Self {
    Var {
      valueType: SHType_Int16,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { int16Value: values },
      },
      ..Default::default()
    }
  }

  pub fn new_color(r: u8, g: u8, b: u8, a: u8) -> Self {
    Var {
      valueType: SHType_Color,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          colorValue: SHColor { r, g, b, a },
        },
      },
      ..Default::default()
    }
  }

  pub fn new_float2(f0: f64, f1: f64) -> Self {
    Var {
      valueType: SHType_Float2,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          float2Value: [f0, f1],
        },
      },
      ..Default::default()
    }
  }

  pub fn new_float3(f0: f32, f1: f32, f2: f32) -> Self {
    Var {
      valueType: SHType_Float3,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          float3Value: [f0, f1, f2, 0.0],
        },
      },
      ..Default::default()
    }
  }

  pub fn new_float4(f0: f32, f1: f32, f2: f32, f3: f32) -> Self {
    Var {
      valueType: SHType_Float4,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          float4Value: [f0, f1, f2, f3],
        },
      },
      ..Default::default()
    }
  }

  pub fn new_enum(value: i32, vendor_id: i32, type_id: i32) -> Self {
    Var {
      valueType: SHType_Enum,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_3: SHVarPayload__bindgen_ty_1__bindgen_ty_3 {
            enumValue: value,
            enumVendorId: vendor_id,
            enumTypeId: type_id,
          },
        },
      },
      ..Default::default()
    }
  }

  pub fn get_type(&self) -> SHType {
    self.valueType.into()
  }

  pub fn color_u8s(r: u8, g: u8, b: u8, a: u8) -> Var {
    SHVar {
      valueType: SHType_Color,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          colorValue: SHColor { r, g, b, a },
        },
      },
      ..Default::default()
    }
  }

  pub fn color_bytes(var: &Var) -> Result<(u8, u8, u8, u8), &'static str> {
    if var.valueType != SHType_Color {
      return Err("Invalid type");
    }
    unsafe {
      Ok((
        var.payload.__bindgen_anon_1.colorValue.r,
        var.payload.__bindgen_anon_1.colorValue.g,
        var.payload.__bindgen_anon_1.colorValue.b,
        var.payload.__bindgen_anon_1.colorValue.a,
      ))
    }
  }

  pub fn color_ints(r: i32, g: i32, b: i32, a: i32) -> Result<Var, &'static str> {
    // ensure all values are in range [0, 255]
    if r < 0 || r > 255 {
      return Err("r is out of range");
    }
    if g < 0 || g > 255 {
      return Err("g is out of range");
    }
    if b < 0 || b > 255 {
      return Err("b is out of range");
    }
    if a < 0 || a > 255 {
      return Err("a is out of range");
    }
    Ok(Var::color_u8s(r as u8, g as u8, b as u8, a as u8))
  }

  pub fn color_floats(r: f32, g: f32, b: f32, a: f32) -> Result<Var, &'static str> {
    // ensure all values are in range [0.0, 1.0]
    if r < 0.0 || r > 1.0 {
      return Err("r is out of range");
    }
    if g < 0.0 || g > 1.0 {
      return Err("g is out of range");
    }
    if b < 0.0 || b > 1.0 {
      return Err("b is out of range");
    }
    if a < 0.0 || a > 1.0 {
      return Err("a is out of range");
    }
    Ok(Var::color_u8s(
      (r * 255.0) as u8,
      (g * 255.0) as u8,
      (b * 255.0) as u8,
      (a * 255.0) as u8,
    ))
  }

  /// To be used while &str is in scope.
  /// Such string likely doesn't have NULL terminator!
  /// CloneVar is safe but the rest might not be!
  pub fn ephemeral_string(s: &str) -> Var {
    let len = s.len();
    let p = if len > 0 {
      s.as_ptr()
    } else {
      core::ptr::null()
    };
    SHVar {
      valueType: SHType_String,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_2: SHVarPayload__bindgen_ty_1__bindgen_ty_2 {
            stringValue: p as *const std::os::raw::c_char,
            stringLen: len as u32,
            stringCapacity: 0,
          },
        },
      },
      ..Default::default()
    }
  }

  pub fn ephemeral_slice(b: &[u8]) -> Var {
    SHVar {
      valueType: SHType_Bytes,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_4: SHVarPayload__bindgen_ty_1__bindgen_ty_4 {
            bytesValue: b.as_ptr() as *mut u8,
            bytesSize: b.len() as u32,
            bytesCapacity: 0,
          },
        },
      },
      ..Default::default()
    }
  }

  /// Caller must call cloneVar to increase strong reference count
  pub fn new_ref_counted<T: RCObjectVar>(obj: T, info: &Type) -> Var {
    let rc = Box::new(RefCounted::<T> {
      rc: AtomicU32::new(0),
      weak_rc: AtomicU32::new(1),
      value: Some(obj),
    });
    unsafe {
      Var {
        valueType: SHType_Object,
        flags: SHVAR_FLAGS_USES_OBJINFO as u16,
        __bindgen_anon_1: SHVar__bindgen_ty_1 {
          objectInfo: T::get_info(),
        },
        payload: SHVarPayload {
          __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
            __bindgen_anon_1: SHVarPayload__bindgen_ty_1__bindgen_ty_1 {
              objectValue: Box::into_raw(rc) as *mut _,
              objectVendorId: info.details.object.vendorId,
              objectTypeId: info.details.object.typeId,
            },
          },
        },
        ..Default::default()
      }
    }
  }

  pub unsafe fn weak_object_var(dst: &mut Var, src: &Var) {
    assert_eq!(src.valueType, SHType_Object);
    assert!(!src.__bindgen_anon_1.objectInfo.is_null());
    assert!(!(*src.__bindgen_anon_1.objectInfo).weakReference.is_none());

    destroyVar(dst);
    dst.valueType = SHType_Object;
    dst.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue =
      src.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue;
    dst.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId =
      src.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId;
    dst.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId =
      src.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId;
    dst.flags |= SHVAR_FLAGS_USES_OBJINFO as u16 | SHVAR_FLAGS_WEAK_OBJECT as u16;
    dst.__bindgen_anon_1.objectInfo = src.__bindgen_anon_1.objectInfo;
    unsafe {
      (*dst.__bindgen_anon_1.objectInfo)
        .weakReference
        .unwrap_unchecked()(dst.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue);
    }
  }

  pub unsafe fn from_ref_counted_object<T>(var: &Var, info: &Type) -> Result<*mut T, &'static str> {
    if var.valueType != SHType_Object
      || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId
        != info.details.object.vendorId
      || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId != info.details.object.typeId
    {
      Err("Failed to cast Var into custom ref counted object")
    } else {
      let aptr = var.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue as *mut RefCounted<T>;
      if let Some(value) = &mut (*aptr).value {
        Ok(value as *mut T)
      } else {
        Err("Failed to cast Var into custom ref counted object: value is None")
      }
    }
  }

  pub fn new_object<T>(obj: &Rc<T>, info: &Type) -> Var {
    unsafe {
      Var {
        valueType: SHType_Object,
        payload: SHVarPayload {
          __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
            __bindgen_anon_1: SHVarPayload__bindgen_ty_1__bindgen_ty_1 {
              objectValue: obj as *const Rc<T> as *mut Rc<T> as SHPointer,
              objectVendorId: info.details.object.vendorId,
              objectTypeId: info.details.object.typeId,
            },
          },
        },
        ..Default::default()
      }
    }
  }

  pub unsafe fn new_object_from_ptr<T>(obj: *const T, info: &Type) -> Var {
    Var {
      valueType: SHType_Object,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_1: SHVarPayload__bindgen_ty_1__bindgen_ty_1 {
            objectValue: obj as *mut T as SHPointer,
            objectVendorId: info.details.object.vendorId,
            objectTypeId: info.details.object.typeId,
          },
        },
      },
      ..Default::default()
    }
  }

  pub unsafe fn new_object_from_raw_ptr(obj: SHPointer, info: &Type) -> Var {
    Var {
      valueType: SHType_Object,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          __bindgen_anon_1: SHVarPayload__bindgen_ty_1__bindgen_ty_1 {
            objectValue: obj,
            objectVendorId: info.details.object.vendorId,
            objectTypeId: info.details.object.typeId,
          },
        },
      },
      ..Default::default()
    }
  }

  pub fn from_object_as_clone<T>(var: &Var, info: &Type) -> Result<Rc<T>, &'static str> {
    // use this to store the smart pointer in order to keep it alive
    // this will not allow mutable references btw
    unsafe {
      if var.valueType != SHType_Object
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId
          != info.details.object.vendorId
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId != info.details.object.typeId
      {
        Err("Failed to cast Var into custom Rc<T> object")
      } else {
        let aptr = var.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue as *mut Rc<T>;
        let at = Rc::clone(&*aptr);
        Ok(at)
      }
    }
  }

  // This pattern is often used in shards storing Rcs of Vars
  pub fn get_mut_from_clone<'a, T>(c: &Option<Rc<Option<T>>>) -> Result<&'a mut T, &'static str> {
    let c = c.as_ref().ok_or("No Var reference found")?;
    let c = Rc::as_ptr(c) as *mut Option<T>;
    let c = unsafe { (*c).as_mut().ok_or("Failed to unwrap Rc-ed reference")? };
    Ok(c)
  }

  // This pattern is often used in shards storing Rcs of Vars
  pub fn get_mut_from_clone1<'a, T>(c: &Option<Rc<T>>) -> Result<&'a mut T, &'static str> {
    let c = c.as_ref().ok_or("No Var reference found")?;
    let c = Rc::as_ptr(c) as *mut T;
    Ok(unsafe { &mut *c })
  }

  pub fn from_object_mut_ref<'a, T>(var: &Var, info: &Type) -> Result<&'a mut T, &'static str> {
    // used to use the object once, when it comes from a simple pointer
    unsafe {
      if var.valueType != SHType_Object
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId
          != info.details.object.vendorId
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId != info.details.object.typeId
      {
        Err("Failed to cast Var into custom &mut T object")
      } else {
        let aptr = var.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue as *mut Rc<T>;
        let p = Rc::as_ptr(&*aptr);
        let mp = p as *mut T;
        Ok(&mut *mp)
      }
    }
  }

  pub fn from_object_ptr_mut_ref<'a, T>(var: &Var, info: &Type) -> Result<&'a mut T, &'static str> {
    // used to use the object once, when it comes from a Rc
    unsafe {
      if var.valueType != SHType_Object
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId
          != info.details.object.vendorId
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId != info.details.object.typeId
      {
        Err("Failed to cast Var into custom &mut T object")
      } else {
        let aptr = var.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue as *mut T;
        Ok(&mut *aptr)
      }
    }
  }

  pub fn from_object_ptr_ref<'a, T>(var: &Var, info: &Type) -> Result<&'a T, &'static str> {
    // used to use the object once, when it comes from a Rc
    unsafe {
      if var.valueType != SHType_Object
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectVendorId
          != info.details.object.vendorId
        || var.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId != info.details.object.typeId
      {
        Err("Failed to cast Var into custom &mut T object")
      } else {
        let aptr = var.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue as *mut T;
        Ok(&*aptr)
      }
    }
  }

  pub fn push<T: Into<Var>>(&mut self, _val: T) {
    unimplemented!();
  }

  pub fn try_push<T: TryInto<Var>>(&mut self, _val: T) {
    unimplemented!();
  }

  pub fn is_seq(&self) -> bool {
    self.valueType == SHType_Seq
  }

  pub fn is_table(&self) -> bool {
    self.valueType == SHType_Table
  }

  pub fn is_none(&self) -> bool {
    self.valueType == SHType_None
  }

  pub fn is_bool(&self) -> bool {
    self.valueType == SHType_Bool
  }

  pub fn is_int(&self) -> bool {
    self.valueType == SHType_Int
  }

  pub fn is_float(&self) -> bool {
    self.valueType == SHType_Float
  }

  pub fn is_string(&self) -> bool {
    self.valueType == SHType_String
  }

  pub fn is_bytes(&self) -> bool {
    self.valueType == SHType_Bytes
  }

  pub fn is_path(&self) -> bool {
    self.valueType == SHType_Path
  }

  pub fn is_context_var(&self) -> bool {
    self.valueType == SHType_ContextVar
  }

  pub fn enum_value(&self) -> Result<i32, &'static str> {
    if self.valueType != SHType_Enum {
      Err("Variable is not an enum")
    } else {
      unsafe { Ok(self.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue) }
    }
  }

  pub fn as_seq(&self) -> Result<&SeqVar, &'static str> {
    if self.valueType != SHType_Seq {
      Err("Variable is not a sequence")
    } else {
      Ok(unsafe { &*(self as *const Var as *const SeqVar) })
    }
  }

  /// The returned SeqVar needs to be wrapped in ClonedVar or destroyed with destroyVar or ownership should be delegated to another Var
  /// SeqVar WON'T call DROP
  pub fn as_mut_seq_creating(&mut self) -> Result<&mut SeqVar, &'static str> {
    if self.valueType != SHType_Seq {
      if self.valueType == SHType_None {
        let sv = SeqVar::new();
        *self = sv.0;
        Ok(unsafe { &mut *(self as *mut Var as *mut SeqVar) })
      } else {
        Err("Variable is not a sequence")
      }
    } else {
      Ok(unsafe { &mut *(self as *mut Var as *mut SeqVar) })
    }
  }

  pub fn as_mut_seq(&mut self) -> Result<&mut SeqVar, &'static str> {
    if self.valueType != SHType_Seq {
      Err("Variable is not a sequence")
    } else {
      Ok(unsafe { &mut *(self as *mut Var as *mut SeqVar) })
    }
  }

  pub fn as_table(&self) -> Result<&TableVar, &'static str> {
    if self.valueType != SHType_Table {
      Err("Variable is not a table")
    } else {
      Ok(unsafe { &*(self as *const Var as *const TableVar) })
    }
  }

  /// The returned TableVar needs to be wrapped in ClonedVar or destroyed with destroyVar or ownership should be delegated to another Var
  /// TableVar WON'T call DROP
  pub fn as_mut_table_creating(&mut self) -> Result<&mut TableVar, &'static str> {
    if self.valueType != SHType_Table {
      if self.valueType == SHType_None {
        let sv = TableVar::new();
        *self = sv.0;
        Ok(unsafe { &mut *(self as *mut Var as *mut TableVar) })
      } else {
        Err("Variable is not a table")
      }
    } else {
      Ok(unsafe { &mut *(self as *mut Var as *mut TableVar) })
    }
  }

  pub fn as_mut_table(&mut self) -> Result<&mut TableVar, &'static str> {
    if self.valueType != SHType_Table {
      Err("Variable is not a table")
    } else {
      Ok(unsafe { &mut *(self as *mut Var as *mut TableVar) })
    }
  }

  pub fn as_str(&self) -> Result<&str, &'static str> {
    self.try_into()
  }

  pub fn serialize(&self) -> ClonedVar {
    ClonedVar(unsafe { shards_serialize_var(self) })
  }
}

impl TryFrom<&Var> for SHString {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_String
      && var.valueType != SHType_Path
      && var.valueType != SHType_ContextVar
    {
      Err("Expected String, Path or ContextVar variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue) }
    }
  }
}

impl TryFrom<&Var> for std::string::String {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_String
      && var.valueType != SHType_Path
      && var.valueType != SHType_ContextVar
    {
      Err("Expected String, Path or ContextVar variable, but casting failed.")
    } else {
      unsafe {
        let cstr = CStr::from_ptr(var.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue);
        Ok(std::string::String::from(cstr.to_str().unwrap()))
      }
    }
  }
}

impl TryFrom<&Var> for CString {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_String
      && var.valueType != SHType_Path
      && var.valueType != SHType_ContextVar
    {
      Err("Expected String, Path or ContextVar variable, but casting failed.")
    } else {
      unsafe {
        let cstr = CStr::from_ptr(var.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue);
        // we need to do this to own the string, this is kinda a burden tho
        Ok(CString::new(cstr.to_str().unwrap()).unwrap())
      }
    }
  }
}

impl TryFrom<&Var> for &CStr {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_String
      && var.valueType != SHType_Path
      && var.valueType != SHType_ContextVar
    {
      Err("Expected String, Path or ContextVar variable, but casting failed.")
    } else {
      unsafe {
        Ok(CStr::from_ptr(
          var.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue as *mut std::os::raw::c_char,
        ))
      }
    }
  }
}

impl TryFrom<&Var> for Option<CString> {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_String
      && var.valueType != SHType_Path
      && var.valueType != SHType_ContextVar
      && var.valueType != SHType_None
    {
      Err("Expected None, String, Path or ContextVar variable, but casting failed.")
    } else if var.is_none() {
      Ok(None)
    } else {
      Ok(Some(
        var.try_into().unwrap_or_else(|_| CString::new("").unwrap()),
      ))
    }
  }
}

impl TryFrom<&Var> for &[u8] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Bytes {
      Err("Expected Bytes, but casting failed.")
    } else {
      unsafe {
        if var.payload.__bindgen_anon_1.__bindgen_anon_4.bytesSize == 0 {
          return Ok(&[]);
        }
        Ok(core::slice::from_raw_parts_mut(
          var.payload.__bindgen_anon_1.__bindgen_anon_4.bytesValue,
          var.payload.__bindgen_anon_1.__bindgen_anon_4.bytesSize as usize,
        ))
      }
    }
  }
}

impl TryFrom<&Var> for &str {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_String
      && var.valueType != SHType_Path
      && var.valueType != SHType_ContextVar
    {
      Err("Expected String, Path or ContextVar variable, but casting failed.")
    } else {
      unsafe {
        if var.payload.__bindgen_anon_1.__bindgen_anon_2.stringLen == 0 {
          return Ok("");
        }
      }
      std::str::from_utf8(unsafe {
        slice::from_raw_parts(
          var.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue as *const u8,
          var.payload.__bindgen_anon_1.__bindgen_anon_2.stringLen as usize,
        )
      })
      .map_err(|_| "Expected valid UTF-8 string, but casting failed.")
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a SHImage {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Image {
      Err("Expected Image variable, but casting failed.")
    } else {
      unsafe {
        Ok(
          &var
            .payload
            .__bindgen_anon_1
            .imageValue
            .as_ref()
            .unwrap_unchecked(),
        )
      }
    }
  }
}

// 64-bit precision :i64
impl TryFrom<&Var> for i64 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.intValue) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut i64 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.intValue) }
    }
  }
}

// 64-bit precision :u64
impl TryFrom<&Var> for u64 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe {
        Ok(u64::from_ne_bytes(
          (var.payload.__bindgen_anon_1.intValue).to_ne_bytes(),
        ))
      }
    }
  }
}

impl TryFrom<&Var> for u128 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(value: &Var) -> Result<Self, Self::Error> {
    if value.valueType != SHType_Int16 {
      return Err("Expected Int16 variable, but casting failed.");
    }
    // so this is simply our Int16 16 bytes to u128
    let value: [i8; 16] = unsafe { value.payload.__bindgen_anon_1.int16Value };
    // reinterpret as u128
    Ok(unsafe { std::mem::transmute(value) })
  }
}

// 64-bit precision :usize
impl TryFrom<&Var> for usize {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe {
        var
          .payload
          .__bindgen_anon_1
          .intValue
          .try_into()
          .map_err(|_| "Int conversion failed, likely out of range (usize)")
      }
    }
  }
}

// 64-bit precision :f64
impl TryFrom<&Var> for f64 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float {
      Err("Expected Float variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.floatValue) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut f64 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float {
      Err("Expected Float variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.floatValue) }
    }
  }
}

// 64-bit precision :[i64;2]
impl TryFrom<&Var> for (i64, i64) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int2Value[0],
          var.payload.__bindgen_anon_1.int2Value[1],
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a [i64; 2] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe { Ok(&var.payload.__bindgen_anon_1.int2Value) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut [i64; 2] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.int2Value) }
    }
  }
}

// 64-bit precision :[u64;2]
impl TryFrom<&Var> for (u64, u64) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u64::from_ne_bytes((var.payload.__bindgen_anon_1.int2Value[0]).to_ne_bytes()),
          u64::from_ne_bytes((var.payload.__bindgen_anon_1.int2Value[1]).to_ne_bytes()),
        ))
      }
    }
  }
}

// 64-bit precision :[f64;2]
impl TryFrom<&Var> for (f64, f64) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float2 {
      Err("Expected Float2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.float2Value[0],
          var.payload.__bindgen_anon_1.float2Value[1],
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a [f64; 2] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float2 {
      Err("Expected Float2 variable, but casting failed.")
    } else {
      unsafe { Ok(&var.payload.__bindgen_anon_1.float2Value) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut [f64; 2] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float2 {
      Err("Expected Float2 variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.float2Value) }
    }
  }
}

// 32-bit precision :i32
impl TryFrom<&Var> for i32 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.intValue as i32) }
    }
  }
}

// 32-bit precision :[i32;2]
impl TryFrom<&Var> for (i32, i32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int2Value[0] as i32,
          var.payload.__bindgen_anon_1.int2Value[1] as i32,
        ))
      }
    }
  }
}

// 32-bit precision :[i32;3]
impl TryFrom<&Var> for (i32, i32, i32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int3 {
      Err("Expected Int3 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int3Value[0],
          var.payload.__bindgen_anon_1.int3Value[1],
          var.payload.__bindgen_anon_1.int3Value[2],
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a [i32; 3] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int3 {
      Err("Expected Int3 variable, but casting failed.")
    } else {
      unsafe {
        Ok(core::mem::transmute(
          &var.payload.__bindgen_anon_1.int3Value,
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut [i32; 3] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int3 {
      Err("Expected Int3 variable, but casting failed.")
    } else {
      unsafe {
        Ok(core::mem::transmute(
          &mut var.payload.__bindgen_anon_1.int3Value,
        ))
      }
    }
  }
}

// 32-bit precision :[i32;4]
impl TryFrom<&Var> for (i32, i32, i32, i32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int4 {
      Err("Expected Int4 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int4Value[0],
          var.payload.__bindgen_anon_1.int4Value[1],
          var.payload.__bindgen_anon_1.int4Value[2],
          var.payload.__bindgen_anon_1.int4Value[3],
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a [i32; 4] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int4 {
      Err("Expected Int4 variable, but casting failed.")
    } else {
      unsafe { Ok(&var.payload.__bindgen_anon_1.int4Value) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut [i32; 4] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int4 {
      Err("Expected Int4 variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.int4Value) }
    }
  }
}

// 32-bit precision :u32
impl TryFrom<&Var> for u32 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.intValue as u32) }
    }
  }
}

// 32-bit precision :[u32;2]
impl TryFrom<&Var> for (u32, u32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int2Value[0] as i32).to_ne_bytes()),
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int2Value[1] as i32).to_ne_bytes()),
        ))
      }
    }
  }
}

// 32-bit precision :[u32;3]
impl TryFrom<&Var> for (u32, u32, u32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int3 {
      Err("Expected Int3 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int3Value[0]).to_ne_bytes()),
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int3Value[1]).to_ne_bytes()),
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int3Value[2]).to_ne_bytes()),
        ))
      }
    }
  }
}

// 32-bit precision :[u32;4]
impl TryFrom<&Var> for (u32, u32, u32, u32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int4 {
      Err("Expected Int4 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[0]).to_ne_bytes()),
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[1]).to_ne_bytes()),
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[2]).to_ne_bytes()),
          u32::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[3]).to_ne_bytes()),
        ))
      }
    }
  }
}

// 32-bit precision :f32
impl TryFrom<&Var> for f32 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float {
      Err("Expected Float variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.floatValue as f32) }
    }
  }
}

// 32-bit precision :[f32;2]
impl TryFrom<&Var> for (f32, f32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float2 {
      Err("Expected Float2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.float2Value[0] as f32,
          var.payload.__bindgen_anon_1.float2Value[1] as f32,
        ))
      }
    }
  }
}

// 32-bit precision :[f32;3]
impl TryFrom<&Var> for (f32, f32, f32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float3 {
      Err("Expected Float3 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.float3Value[0],
          var.payload.__bindgen_anon_1.float3Value[1],
          var.payload.__bindgen_anon_1.float3Value[2],
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a [f32; 3] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float3 {
      Err("Expected Float3 variable, but casting failed.")
    } else {
      unsafe {
        Ok(core::mem::transmute(
          &var.payload.__bindgen_anon_1.float3Value,
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut [f32; 3] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float3 {
      Err("Expected Float3 variable, but casting failed.")
    } else {
      unsafe {
        Ok(core::mem::transmute(
          &mut var.payload.__bindgen_anon_1.float3Value,
        ))
      }
    }
  }
}

// 32-bit precision :[f32;4]
impl TryFrom<&Var> for (f32, f32, f32, f32) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float4 {
      Err("Expected Float4 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.float4Value[0],
          var.payload.__bindgen_anon_1.float4Value[1],
          var.payload.__bindgen_anon_1.float4Value[2],
          var.payload.__bindgen_anon_1.float4Value[3],
        ))
      }
    }
  }
}

impl<'a> TryFrom<&'a Var> for &'a [f32; 4] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float4 {
      Err("Expected Float4 variable, but casting failed.")
    } else {
      unsafe { Ok(&var.payload.__bindgen_anon_1.float4Value) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut [f32; 4] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float4 {
      Err("Expected Float4 variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.float4Value) }
    }
  }
}

// 16-bit precision :i16
impl TryFrom<&Var> for i16 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.intValue as i16) }
    }
  }
}

// 16-bit precision :[i16;2]
impl TryFrom<&Var> for (i16, i16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int2Value[0] as i16,
          var.payload.__bindgen_anon_1.int2Value[1] as i16,
        ))
      }
    }
  }
}

// 16-bit precision :[i16;3]
impl TryFrom<&Var> for (i16, i16, i16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int3 {
      Err("Expected Int3 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int3Value[0] as i16,
          var.payload.__bindgen_anon_1.int3Value[1] as i16,
          var.payload.__bindgen_anon_1.int3Value[2] as i16,
        ))
      }
    }
  }
}

// 16-bit precision :[i16;4]
impl TryFrom<&Var> for (i16, i16, i16, i16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int4 {
      Err("Expected Int4 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          var.payload.__bindgen_anon_1.int4Value[0] as i16,
          var.payload.__bindgen_anon_1.int4Value[1] as i16,
          var.payload.__bindgen_anon_1.int4Value[2] as i16,
          var.payload.__bindgen_anon_1.int4Value[3] as i16,
        ))
      }
    }
  }
}

// 16-bit precision :u16
impl TryFrom<&Var> for u16 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int {
      Err("Expected Int variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.intValue as u16) }
    }
  }
}

// 16-bit precision :[u16;2]
impl TryFrom<&Var> for (u16, u16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int2 {
      Err("Expected Int2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int2Value[0] as i16).to_ne_bytes()),
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int2Value[1] as i16).to_ne_bytes()),
        ))
      }
    }
  }
}

// 16-bit precision :[u16;3]
impl TryFrom<&Var> for (u16, u16, u16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int3 {
      Err("Expected Int3 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int3Value[0] as i16).to_ne_bytes()),
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int3Value[1] as i16).to_ne_bytes()),
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int3Value[2] as i16).to_ne_bytes()),
        ))
      }
    }
  }
}

// 16-bit precision :[u16;4]
impl TryFrom<&Var> for (u16, u16, u16, u16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Int4 {
      Err("Expected Int4 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[0] as i16).to_ne_bytes()),
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[1] as i16).to_ne_bytes()),
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[2] as i16).to_ne_bytes()),
          u16::from_ne_bytes((var.payload.__bindgen_anon_1.int4Value[3] as i16).to_ne_bytes()),
        ))
      }
    }
  }
}

// 16-bit precision :f16
impl TryFrom<&Var> for half::f16 {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float {
      Err("Expected Float variable, but casting failed.")
    } else {
      unsafe { Ok(half::f16::from_f64(var.payload.__bindgen_anon_1.floatValue)) }
    }
  }
}

// 16-bit precision :[f16;2]
impl TryFrom<&Var> for (half::f16, half::f16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float2 {
      Err("Expected Float2 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          half::f16::from_f64(var.payload.__bindgen_anon_1.float2Value[0]),
          half::f16::from_f64(var.payload.__bindgen_anon_1.float2Value[1]),
        ))
      }
    }
  }
}

// 16-bit precision :[f16;3]
impl TryFrom<&Var> for (half::f16, half::f16, half::f16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float3 {
      Err("Expected Float3 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          half::f16::from_f32(var.payload.__bindgen_anon_1.float3Value[0]),
          half::f16::from_f32(var.payload.__bindgen_anon_1.float3Value[1]),
          half::f16::from_f32(var.payload.__bindgen_anon_1.float3Value[2]),
        ))
      }
    }
  }
}

// 16-bit precision :[f16;4]
impl TryFrom<&Var> for (half::f16, half::f16, half::f16, half::f16) {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Float4 {
      Err("Expected Float4 variable, but casting failed.")
    } else {
      unsafe {
        Ok((
          half::f16::from_f32(var.payload.__bindgen_anon_1.float4Value[0]),
          half::f16::from_f32(var.payload.__bindgen_anon_1.float4Value[1]),
          half::f16::from_f32(var.payload.__bindgen_anon_1.float4Value[2]),
          half::f16::from_f32(var.payload.__bindgen_anon_1.float4Value[3]),
        ))
      }
    }
  }
}

impl TryFrom<&Var> for SHColor {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Color {
      Err("Expected Color variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.colorValue) }
    }
  }
}

impl TryFrom<&Var> for bool {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Bool {
      Err("Expected Bool variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.boolValue) }
    }
  }
}

impl<'a> TryFrom<&'a mut Var> for &'a mut bool {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &'a mut Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Bool {
      Err("Expected Bool variable, but casting failed.")
    } else {
      unsafe { Ok(&mut var.payload.__bindgen_anon_1.boolValue) }
    }
  }
}

impl TryFrom<&Var> for &[Var] {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Seq {
      Err("Expected Seq variable, but casting failed.")
    } else {
      unsafe {
        let elems = var.payload.__bindgen_anon_1.seqValue.elements;
        let len = var.payload.__bindgen_anon_1.seqValue.len;
        if len == 0 {
          return Ok(&[]);
        }
        let res = std::slice::from_raw_parts(elems, len as usize);
        Ok(res)
      }
    }
  }
}

impl TryFrom<&Var> for WireRef {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Wire {
      Err("Expected Wire variable, but casting failed.")
    } else {
      unsafe { Ok(WireRef(var.payload.__bindgen_anon_1.wireValue)) }
    }
  }
}

impl TryFrom<&Var> for TableVar {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Table {
      Err("Expected Table variable, but casting failed.")
    } else {
      Ok(TableVar(*var))
    }
  }
}

impl TryFrom<&Var> for SeqVar {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Seq {
      Err("Expected Seq variable, but casting failed.")
    } else {
      Ok(SeqVar(*var))
    }
  }
}

impl From<WireRef> for Var {
  #[inline(always)]
  fn from(wire: WireRef) -> Self {
    Var {
      valueType: SHType_Wire,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { wireValue: wire.0 },
      },
      ..Default::default()
    }
  }
}

impl TryFrom<&Var> for ShardRef {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_ShardRef {
      Err("Expected Shard variable, but casting failed.")
    } else {
      unsafe { Ok(ShardRef(var.payload.__bindgen_anon_1.shardValue)) }
    }
  }
}

impl AsRef<Var> for Var {
  #[inline(always)]
  fn as_ref(&self) -> &Var {
    self
  }
}

impl AsRef<Var> for ClonedVar {
  #[inline(always)]
  fn as_ref(&self) -> &Var {
    &self.0
  }
}

pub struct ParamVar {
  pub parameter: ClonedVar,
  pub pointee: *mut Var,
}

impl ParamVar {
  pub fn new(initial_value: Var) -> ParamVar {
    ParamVar {
      parameter: initial_value.into(),
      pointee: std::ptr::null_mut(),
    }
  }

  pub fn new_named(name: &str) -> ParamVar {
    let mut var = ParamVar::default();
    var.set_name(name);
    var
  }

  pub fn cleanup(&mut self, _ctx: Option<&Context>) {
    unsafe {
      if self.parameter.0.valueType == SHType_ContextVar {
        (*Core).releaseVariable.unwrap_unchecked()(self.pointee);
      }
      self.pointee = std::ptr::null_mut();
    }
  }

  pub fn warmup(&mut self, context: &Context) {
    if self.parameter.0.valueType == SHType_ContextVar {
      assert_eq!(self.pointee, std::ptr::null_mut());
      unsafe {
        let ctx = context as *const SHContext as *mut SHContext;
        self.pointee = (*Core).referenceVariable.unwrap_unchecked()(
          ctx,
          SHStringWithLen {
            string: self
              .parameter
              .0
              .payload
              .__bindgen_anon_1
              .__bindgen_anon_2
              .stringValue,
            len: self
              .parameter
              .0
              .payload
              .__bindgen_anon_1
              .__bindgen_anon_2
              .stringLen as u64,
          },
        );
      }
    } else {
      self.pointee = &mut self.parameter.0 as *mut Var;
    }
    assert_ne!(self.pointee, std::ptr::null_mut());
  }

  pub fn set_fast_unsafe(&mut self, value: &Var) {
    // avoid overwrite refcount
    assert_ne!(self.pointee, std::ptr::null_mut());
    unsafe {
      // store flags and rc
      let rc = (*self.pointee).refcount;
      let flags = (*self.pointee).flags;
      // assign
      (*self.pointee) = *value;
      // restore flags and rc
      (*self.pointee).flags = flags;
      (*self.pointee).refcount = rc;
    }
  }

  pub fn set_cloning(&mut self, value: &Var) {
    unsafe { (*Core).cloneVar.unwrap_unchecked()(self.pointee, value) };
    // notice we don't need to fix up ref-counting because cloning does not touch that
  }

  pub fn get(&self) -> &Var {
    assert_ne!(self.pointee, std::ptr::null_mut());
    unsafe { &*self.pointee }
  }

  // Users should never fully overwrite or flags will be lost, unless taken care of
  pub fn get_mut(&mut self) -> &mut Var {
    assert_ne!(self.pointee, std::ptr::null_mut());
    unsafe { &mut *self.pointee }
  }

  pub fn try_get(&self) -> Option<&Var> {
    if self.pointee.is_null() {
      None
    } else {
      Some(unsafe { &*self.pointee })
    }
  }

  pub fn assign(&mut self, value: &Var) {
    self.parameter = value.into();
  }

  pub fn set_param(&mut self, value: &Var) -> Result<(), &'static str> {
    self.assign(value);
    Ok(())
  }

  pub fn get_param(&self) -> Var {
    self.parameter.0
  }

  pub fn is_variable(&self) -> bool {
    self.parameter.0.valueType == SHType_ContextVar
  }

  pub fn is_none(&self) -> bool {
    self.parameter.0.valueType == SHType_None
  }

  pub fn set_name(&mut self, name: &str) {
    let s = Var::ephemeral_string(name);
    self.parameter = s.into(); // clone it!
    self.parameter.0.valueType = SHType_ContextVar;
  }

  pub fn get_name(&self) -> *const std::os::raw::c_char {
    if self.is_variable() {
      (&self.parameter.0)
        .try_into()
        .expect("parameter name is a string")
    } else {
      std::ptr::null()
    }
  }
}

impl Default for ParamVar {
  fn default() -> Self {
    ParamVar::new(().into())
  }
}

impl Drop for ParamVar {
  fn drop(&mut self) {
    self.cleanup(None);
  }
}

// ShardsVar

#[derive(Default)]
pub struct ShardsVar {
  param: ClonedVar,
  shards: Vec<ShardRef>,
  compose_result: Option<SHComposeResult>,
  native_shards: Shards,
}

impl Drop for ShardsVar {
  fn drop(&mut self) {
    self.destroy();
  }
}

unsafe extern "C" fn shardsvar_compose_cb(
  errorShard: *const Shard,
  errorTxt: SHStringWithLen,
  nonfatalWarning: SHBool,
  userData: *mut c_void,
) {
  let shard_name = CStr::from_ptr((*errorShard).name.unwrap()(errorShard as *mut _));
  let msg = std::str::from_utf8(unsafe {
    if errorTxt.len == 0 {
      &[]
    } else {
      slice::from_raw_parts(errorTxt.string as *const u8, errorTxt.len as usize)
    }
  })
  .unwrap();
  if !nonfatalWarning {
    shlog_error!(
      "Fatal error: {} shard: {}",
      msg,
      shard_name.to_str().unwrap()
    );
    let failed = userData as *mut bool;
    *failed = true;
  } else {
    shlog_error!("Error: {} shard: {}", msg, shard_name.to_str().unwrap());
  }
}

impl ShardsVar {
  fn destroy(&mut self) {
    for shard in &self.shards {
      if let Err(e) = shard.cleanup(None) {
        shlog!("Errors during shard cleanup: {}", e);
      }
    }
    self.shards.clear();
    destroyVar(&mut self.param.0);

    // clear old results if any
    if let Some(compose_result) = self.compose_result {
      unsafe {
        (*Core).expTypesFree.unwrap_unchecked()(&compose_result.exposedInfo as *const _ as *mut _);
        (*Core).expTypesFree.unwrap_unchecked()(&compose_result.requiredInfo as *const _ as *mut _);
      }
      self.compose_result = None;
    }
  }

  #[inline(always)]
  pub fn cleanup(&mut self, ctx: Option<&Context>) {
    for shard in self.shards.iter().rev() {
      if let Err(e) = shard.cleanup(ctx) {
        shlog!("Errors during shard cleanup: {}", e);
      }
    }
  }

  #[inline(always)]
  pub fn warmup(&self, context: &Context) -> Result<(), &'static str> {
    for shard in self.shards.iter() {
      if let Err(e) = shard.warmup(context) {
        shlog!("Errors during shard warmup: {}", e);
        return Err(e);
      }
    }
    Ok(())
  }

  pub fn set_param(&mut self, value: &Var) -> Result<(), &'static str> {
    self.destroy(); // destroy old blocks

    self.param = value.into(); // clone it

    if let Ok(s) = Seq::try_from(self.param.0.as_ref()) {
      for shard in s.iter() {
        self.shards.push(shard.as_ref().try_into()?);
      }
    } else if let Ok(s) = ShardRef::try_from(&self.param.0) {
      self.shards.push(s);
    } else if value.is_none() {
      // we allow none
      return Ok(());
    } else {
      return Err("Expected sequence or shard variable, but casting failed.");
    }

    self.native_shards = Shards {
      elements: self.shards.as_mut_ptr() as *mut *mut _,
      len: self.shards.len() as u32,
      cap: 0,
    };

    Ok(())
  }

  #[inline(always)]
  pub fn get_param(&self) -> Var {
    self.param.0
  }

  #[inline(always)]
  pub fn compose(&mut self, data: &InstanceData) -> Result<&ComposeResult, &'static str> {
    // clear old results if any
    if let Some(compose_result) = self.compose_result {
      unsafe {
        (*Core).expTypesFree.unwrap_unchecked()(&compose_result.exposedInfo as *const _ as *mut _);
        (*Core).expTypesFree.unwrap_unchecked()(&compose_result.requiredInfo as *const _ as *mut _);
      }
      self.compose_result = None;
    }

    if self.param.0.is_none() {
      self.compose_result = Some(Default::default());
      return Ok(self.compose_result.as_ref().unwrap());
    }

    let failed = false;

    let mut result = unsafe { (*Core).composeShards.unwrap_unchecked()(self.native_shards, *data) };

    if result.failed {
      let msg: &str = (&result.failureMessage).try_into().unwrap();
      shlog!("Compose failed with error {}", msg);
      destroyVar(&mut result.failureMessage);
      Err("Composition failed.")
    } else if failed {
      Err("Composition failed.")
    } else {
      self.compose_result = Some(result);
      Ok(self.compose_result.as_ref().unwrap())
    }
  }

  #[inline(always)]
  pub fn activate(&self, context: &Context, input: &Var, output: &mut Var) -> WireState {
    if self.param.0.is_none() {
      return WireState::Continue;
    }

    unsafe {
      (*Core).runShards.unwrap_unchecked()(
        self.native_shards,
        context as *const _ as *mut _,
        input,
        output,
      )
      .into()
    }
  }

  #[inline(always)]
  pub fn activate_handling_return(
    &self,
    context: &Context,
    input: &Var,
    output: &mut Var,
  ) -> WireState {
    if self.param.0.is_none() {
      return WireState::Continue;
    }

    unsafe {
      (*Core).runShards2.unwrap_unchecked()(
        self.native_shards,
        context as *const _ as *mut _,
        input,
        output,
      )
      .into()
    }
  }

  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.param.0.is_none()
  }

  #[inline(always)]
  pub fn get_exposing(&self) -> Option<&[ExposedInfo]> {
    self.compose_result.map(|compose_result| unsafe {
      let elems = compose_result.exposedInfo.elements;
      let len = compose_result.exposedInfo.len;
      if len == 0 {
        &[]
      } else {
        std::slice::from_raw_parts(elems, len as usize)
      }
    })
  }

  #[inline(always)]
  pub fn get_requiring(&self) -> Option<&[ExposedInfo]> {
    self.compose_result.map(|compose_result| unsafe {
      let elems = compose_result.requiredInfo.elements;
      let len = compose_result.requiredInfo.len;
      if len == 0 {
        &[]
      } else {
        std::slice::from_raw_parts(elems, len as usize)
      }
    })
  }
}

impl From<&ShardsVar> for Shards {
  fn from(v: &ShardsVar) -> Self {
    v.native_shards
  }
}

impl From<&ShardsVar> for &SHVar {
  fn from(v: &ShardsVar) -> Self {
    (&v.param).into()
  }
}

impl From<&ShardsVar> for Var {
  fn from(v: &ShardsVar) -> Self {
    (&v.param).into()
  }
}

impl From<&ClonedVar> for &SHVar {
  fn from(value: &ClonedVar) -> Self {
    unsafe { transmute(value) }
  }
}

impl From<&ClonedVar> for Var {
  fn from(value: &ClonedVar) -> Self {
    let v: &Var = unsafe { transmute(value) };
    *v
  }
}

impl From<&ParamVar> for &SHVar {
  fn from(value: &ParamVar) -> Self {
    unsafe { transmute(value) }
  }
}

impl From<&ParamVar> for Var {
  fn from(value: &ParamVar) -> Self {
    let v: &Var = unsafe { transmute(value) };
    *v
  }
}

impl IntoIterator for TableVar {
  type Item = (Var, Var);
  type IntoIter = TableIterator;

  fn into_iter(self) -> Self::IntoIter {
    unsafe {
      let it = TableIterator {
        table: self.0.payload.__bindgen_anon_1.tableValue,
        citer: [0; 64],
      };
      (*it.table.api).tableGetIterator.unwrap_unchecked()(
        it.table,
        &it.citer as *const _ as *mut _,
      );
      it
    }
  }
}

impl IntoIterator for SeqVar {
  type Item = SHVar;
  type IntoIter = SeqVarIterator;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl IntoIterator for &SeqVar {
  type Item = SHVar;
  type IntoIter = SeqVarIterator;

  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

pub struct ExposedTypesIterator {
  elements: *mut SHExposedTypeInfo,
  length: usize,
  index: usize,
}

impl Iterator for ExposedTypesIterator {
  type Item = SHExposedTypeInfo;

  fn next(&mut self) -> Option<Self::Item> {
    if self.index < self.length {
      Some({
        let ret_index = self.index;
        self.index += 1;
        if self.length == 0 {
          return None;
        }
        unsafe { slice::from_raw_parts_mut(self.elements, self.length)[ret_index] }
      })
    } else {
      None
    }
  }
}

impl IntoIterator for &SHExposedTypesInfo {
  type Item = SHExposedTypeInfo;
  type IntoIter = ExposedTypesIterator;

  fn into_iter(self) -> Self::IntoIter {
    ExposedTypesIterator {
      elements: self.elements,
      index: 0,
      length: self.len as usize,
    }
  }
}

impl SHExposedTypesInfo {
  pub fn iter(&self) -> ExposedTypesIterator {
    (&self).into_iter()
  }
}

// Strings / SHStrings

pub struct Strings {
  pub s: SHStrings, // Don't derive clone, won't work, it will double free
  owned: bool,
}

impl Drop for Strings {
  fn drop(&mut self) {
    if self.owned {
      unsafe {
        (*Core).stringsFree.unwrap_unchecked()(&self.s as *const SHStrings as *mut SHStrings);
      }
    }
  }
}

impl Strings {
  pub const fn new() -> Self {
    Self {
      s: SHStrings {
        elements: core::ptr::null_mut(),
        len: 0,
        cap: 0,
      },
      owned: true,
    }
  }

  pub fn set_len(&mut self, len: usize) {
    unsafe {
      (*Core).stringsResize.unwrap_unchecked()(
        &self.s as *const SHStrings as *mut SHStrings,
        len.try_into().unwrap(),
      );
    }
  }

  pub fn push(&mut self, value: &str) {
    let str = value.as_ptr() as *const std::os::raw::c_char;
    unsafe {
      (*Core).stringsPush.unwrap_unchecked()(&self.s as *const SHStrings as *mut SHStrings, &str);
    }
  }

  pub fn insert(&mut self, index: usize, value: &str) {
    let str = value.as_ptr() as *const std::os::raw::c_char;
    unsafe {
      (*Core).stringsInsert.unwrap_unchecked()(
        &self.s as *const SHStrings as *mut SHStrings,
        index.try_into().unwrap(),
        &str,
      );
    }
  }

  pub fn len(&self) -> usize {
    self.s.len.try_into().unwrap()
  }

  pub fn is_empty(&self) -> bool {
    self.s.len == 0
  }

  pub fn pop(&mut self) -> Option<&str> {
    unsafe {
      if !self.is_empty() {
        let v =
          (*Core).stringsPop.unwrap_unchecked()(&self.s as *const SHStrings as *mut SHStrings);
        Some(CStr::from_ptr(v).to_str().unwrap())
      } else {
        None
      }
    }
  }

  pub fn clear(&mut self) {
    unsafe {
      (*Core).stringsResize.unwrap_unchecked()(&self.s as *const SHStrings as *mut SHStrings, 0);
    }
  }
}

impl Default for Strings {
  fn default() -> Self {
    Strings::new()
  }
}

impl AsRef<Strings> for Strings {
  #[inline(always)]
  fn as_ref(&self) -> &Strings {
    self
  }
}

pub type OptionalStrings = Vec<OptionalString>;

impl From<&OptionalStrings> for SHOptionalStrings {
  fn from(vec: &OptionalStrings) -> Self {
    SHOptionalStrings {
      elements: vec.as_ptr() as *mut SHOptionalString,
      len: vec.len() as u32,
      cap: 0,
    }
  }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct SeqVar(pub Var);

/// A wrapper around `SeqVar` that automatically destroys the variable when it goes out of scope.
#[repr(transparent)] // force it same size of original
pub struct AutoSeqVar(pub SeqVar);

impl Drop for AutoSeqVar {
  fn drop(&mut self) {
    destroyVar(&mut self.0 .0);
  }
}

impl AutoSeqVar {
  /// Creates a new `AutoSeqVar`.
  pub fn new() -> AutoSeqVar {
    AutoSeqVar(SeqVar::new())
  }

  /// Extracts the `Var` from the `AutoSeqVar` and leaks it.
  ///
  /// This function destroys the `AutoSeqVar` and returns the `Var` contained within it.
  /// The `Var` will not be destroyed when it goes out of scope.
  pub fn leak(&mut self) -> Var {
    std::mem::replace(&mut self.0 .0, Var::default())
  }
}

/// A wrapper around `TableVar` that automatically destroys the variable when it goes out of scope.
#[repr(transparent)] // force it same size of original
pub struct AutoTableVar(pub TableVar);

impl Debug for AutoTableVar {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "AutoTableVar")
  }
}

impl Drop for AutoTableVar {
  fn drop(&mut self) {
    destroyVar(&mut self.0 .0);
  }
}

impl AutoTableVar {
  /// Creates a new `AutoTableVar`.
  pub fn new() -> AutoTableVar {
    AutoTableVar(TableVar::new())
  }

  /// Extracts the `Var` from the `AutoTableVar` and leaks it.
  ///
  /// This function destroys the `AutoTableVar` and returns the `Var` contained within it.
  /// The `Var` will not be destroyed when it goes out of scope.
  pub fn leak(&mut self) -> Var {
    std::mem::replace(&mut self.0 .0, Var::default())
  }

  pub fn to_cloned(self) -> ClonedVar {
    unsafe { std::mem::transmute(self) }
  }
}

pub struct SeqVarIterator {
  s: SeqVar,
  i: u32,
}

impl Iterator for SeqVarIterator {
  fn next(&mut self) -> Option<Self::Item> {
    unsafe {
      let res = if self.i < self.s.0.payload.__bindgen_anon_1.seqValue.len {
        Some(
          *self
            .s
            .0
            .payload
            .__bindgen_anon_1
            .seqValue
            .elements
            .offset(self.i.try_into().unwrap()),
        )
      } else {
        None
      };
      self.i += 1;
      res
    }
  }
  type Item = Var;
}

impl DoubleEndedIterator for SeqVarIterator {
  fn next_back(&mut self) -> Option<Self::Item> {
    unsafe {
      let res = if self.i < self.s.0.payload.__bindgen_anon_1.seqValue.len {
        Some(
          *self.s.0.payload.__bindgen_anon_1.seqValue.elements.offset(
            (self.s.0.payload.__bindgen_anon_1.seqValue.len - self.i - 1)
              .try_into()
              .unwrap(),
          ),
        )
      } else {
        None
      };
      self.i += 1;
      res
    }
  }
}

impl Index<usize> for SeqVar {
  #[inline(always)]
  fn index(&self, idx: usize) -> &Self::Output {
    let idx_u32: u32 = idx.try_into().unwrap();
    let len = unsafe { self.0.payload.__bindgen_anon_1.seqValue.len };
    if idx_u32 < len {
      unsafe {
        &*self
          .0
          .payload
          .__bindgen_anon_1
          .seqValue
          .elements
          .offset(idx.try_into().unwrap())
      }
    } else {
      panic!("Index out of range");
    }
  }
  type Output = Var;
}

impl IndexMut<usize> for SeqVar {
  #[inline(always)]
  fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
    let idx_u32: u32 = idx.try_into().unwrap();
    let len = unsafe { self.0.payload.__bindgen_anon_1.seqValue.len };
    if idx_u32 < len {
      unsafe {
        &mut *self
          .0
          .payload
          .__bindgen_anon_1
          .seqValue
          .elements
          .offset(idx.try_into().unwrap())
      }
    } else {
      panic!("Index out of range");
    }
  }
}

impl SeqVar {
  #[inline(always)]
  pub(crate) fn new() -> SeqVar {
    SeqVar {
      0: Var {
        payload: SHVarPayload {
          __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
            seqValue: SHSeq {
              elements: 0 as *mut SHVar,
              len: 0,
              cap: 0,
            },
          },
        },
        valueType: SHType_Seq,
        ..Default::default()
      },
    }
  }

  #[inline(always)]
  /// Creates a new `SeqVar` that IS NOT DROPPED when it goes out of scope.
  /// The resulting variable should be wrapped inside a `ClonedVar(v)` or `destroyVar` should be called.
  pub fn leaking_new() -> SeqVar {
    Self::new()
  }

  #[inline(always)]
  pub fn wrap(var: Var) -> SeqVar {
    SeqVar { 0: var }
  }

  #[inline(always)]
  pub fn set_len(&mut self, len: usize) {
    unsafe {
      (*Core).seqResize.unwrap_unchecked()(
        &self.0.payload.__bindgen_anon_1.seqValue as *const SHSeq as *mut SHSeq,
        len.try_into().unwrap(),
      );
    }
  }

  #[inline(always)]
  pub fn push(&mut self, value: &Var) {
    // we need to clone to own the memory shards side
    let idx = self.len();
    self.set_len(idx + 1);
    cloneVar(&mut self[idx], &value);
  }

  #[inline(always)]
  pub fn emplace(&mut self, value: ClonedVar) {
    // we need to clone to own the memory shards side
    let idx = self.len();
    self.set_len(idx + 1);
    let v = &mut self[idx];
    *v = value.0;
    // now make sure value is not dropped
    std::mem::forget(value);
  }

  #[inline(always)]
  pub fn insert(&mut self, index: usize, value: &Var) {
    // we need to clone to own the memory shards side
    let mut tmp = SHVar::default();
    cloneVar(&mut tmp, &value);
    unsafe {
      (*Core).seqInsert.unwrap_unchecked()(
        &self.0.payload.__bindgen_anon_1.seqValue as *const SHSeq as *mut SHSeq,
        index.try_into().unwrap(),
        &tmp,
      );
    }
  }

  #[inline(always)]
  pub fn len(&self) -> usize {
    unsafe {
      self
        .0
        .payload
        .__bindgen_anon_1
        .seqValue
        .len
        .try_into()
        .unwrap()
    }
  }

  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  #[inline(always)]
  pub fn pop(&mut self) -> Option<ClonedVar> {
    unsafe {
      if !self.is_empty() {
        let v = (*Core).seqPop.unwrap_unchecked()(
          &self.0.payload.__bindgen_anon_1.seqValue as *const SHSeq as *mut SHSeq,
        );
        Some(transmute(v))
      } else {
        None
      }
    }
  }

  #[inline(always)]
  pub fn remove(&mut self, index: usize) {
    unsafe {
      (*Core).seqSlowDelete.unwrap_unchecked()(
        &self.0.payload.__bindgen_anon_1.seqValue as *const SHSeq as *mut SHSeq,
        index.try_into().unwrap(),
      );
    }
  }

  #[inline(always)]
  pub fn remove_fast(&mut self, index: usize) {
    unsafe {
      (*Core).seqFastDelete.unwrap_unchecked()(
        &self.0.payload.__bindgen_anon_1.seqValue as *const SHSeq as *mut SHSeq,
        index.try_into().unwrap(),
      );
    }
  }

  #[inline(always)]
  pub fn clear(&mut self) {
    unsafe {
      (*Core).seqResize.unwrap_unchecked()(
        &self.0.payload.__bindgen_anon_1.seqValue as *const SHSeq as *mut SHSeq,
        0,
      );
    }
  }

  #[inline(always)]
  pub fn iter(&self) -> SeqVarIterator {
    SeqVarIterator {
      s: self.clone(),
      i: 0,
    }
  }

  #[inline(always)]
  pub fn next_mut(&mut self) -> &mut Var {
    let idx = self.len();
    self.set_len(idx + 1);
    &mut self[idx]
  }
}

// Seq / SHSeq

pub struct Seq {
  s: SHSeq, // Don't derive clone, won't work, it will double free
  owned: bool,
}

impl Drop for Seq {
  fn drop(&mut self) {
    if self.owned {
      unsafe {
        (*Core).seqFree.unwrap_unchecked()(&self.s as *const SHSeq as *mut SHSeq);
      }
    }
  }
}

pub struct SeqIterator<'a> {
  s: &'a Seq,
  i: u32,
}

impl<'a> Iterator for SeqIterator<'a> {
  fn next(&mut self) -> Option<Self::Item> {
    let res = if self.i < self.s.s.len {
      unsafe { Some(&*self.s.s.elements.offset(self.i.try_into().unwrap())) }
    } else {
      None
    };
    self.i += 1;
    res
  }
  type Item = &'a Var;
}

impl<'a> DoubleEndedIterator for SeqIterator<'a> {
  fn next_back(&mut self) -> Option<Self::Item> {
    let res = if self.i < self.s.s.len {
      unsafe {
        Some(
          &*self
            .s
            .s
            .elements
            .offset((self.s.s.len - self.i - 1).try_into().unwrap()),
        )
      }
    } else {
      None
    };
    self.i += 1;
    res
  }
}

impl Index<usize> for SHSeq {
  #[inline(always)]
  fn index(&self, idx: usize) -> &Self::Output {
    let idx_u32: u32 = idx.try_into().unwrap();
    if idx_u32 < self.len {
      unsafe { &*self.elements.offset(idx.try_into().unwrap()) }
    } else {
      panic!("Index out of range");
    }
  }
  type Output = Var;
}

impl Index<usize> for Seq {
  #[inline(always)]
  fn index(&self, idx: usize) -> &Self::Output {
    &self.s[idx]
  }
  type Output = Var;
}

impl IndexMut<usize> for SHSeq {
  #[inline(always)]
  fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
    let idx_u32: u32 = idx.try_into().unwrap();
    if idx_u32 < self.len {
      unsafe { &mut *self.elements.offset(idx.try_into().unwrap()) }
    } else {
      panic!("Index out of range");
    }
  }
}

impl IndexMut<usize> for Seq {
  #[inline(always)]
  fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
    &mut self.s[idx]
  }
}

impl Seq {
  pub const fn new() -> Seq {
    Seq {
      s: SHSeq {
        elements: core::ptr::null_mut(),
        len: 0,
        cap: 0,
      },
      owned: true,
    }
  }

  pub fn set_len(&mut self, len: usize) {
    unsafe {
      (*Core).seqResize.unwrap_unchecked()(
        &self.s as *const SHSeq as *mut SHSeq,
        len.try_into().unwrap(),
      );
    }
  }

  pub fn push(&mut self, value: &Var) {
    // we need to clone to own the memory shards side
    let idx = self.len();
    self.set_len(idx + 1);
    cloneVar(&mut self[idx], &value);
  }

  pub fn insert(&mut self, index: usize, value: &Var) {
    // we need to clone to own the memory shards side
    let mut tmp = SHVar::default();
    cloneVar(&mut tmp, &value);
    unsafe {
      (*Core).seqInsert.unwrap_unchecked()(
        &self.s as *const SHSeq as *mut SHSeq,
        index.try_into().unwrap(),
        &tmp,
      );
    }
  }

  pub fn len(&self) -> usize {
    self.s.len.try_into().unwrap()
  }

  pub fn is_empty(&self) -> bool {
    self.s.len == 0
  }

  pub fn pop(&mut self) -> Option<ClonedVar> {
    unsafe {
      if !self.is_empty() {
        let v = (*Core).seqPop.unwrap_unchecked()(&self.s as *const SHSeq as *mut SHSeq);
        Some(transmute(v))
      } else {
        None
      }
    }
  }

  pub fn remove(&mut self, index: usize) {
    unsafe {
      (*Core).seqSlowDelete.unwrap_unchecked()(
        &self.s as *const SHSeq as *mut SHSeq,
        index.try_into().unwrap(),
      );
    }
  }

  pub fn remove_fast(&mut self, index: usize) {
    unsafe {
      (*Core).seqFastDelete.unwrap_unchecked()(
        &self.s as *const SHSeq as *mut SHSeq,
        index.try_into().unwrap(),
      );
    }
  }

  pub fn clear(&mut self) {
    unsafe {
      (*Core).seqResize.unwrap_unchecked()(&self.s as *const SHSeq as *mut SHSeq, 0);
    }
  }

  pub fn iter(&self) -> SeqIterator {
    SeqIterator { s: self, i: 0 }
  }
}

impl Default for Seq {
  fn default() -> Self {
    Seq::new()
  }
}

impl Into<SHSeq> for &Seq {
  fn into(self) -> SHSeq {
    self.s
  }
}

impl AsRef<Seq> for Seq {
  #[inline(always)]
  fn as_ref(&self) -> &Seq {
    self
  }
}

impl From<&Seq> for Var {
  #[inline(always)]
  fn from(s: &Seq) -> Self {
    SHVar {
      valueType: SHType_Seq,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { seqValue: s.s },
      },
      ..Default::default()
    }
  }
}

impl TryFrom<&mut Var> for Seq {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(v: &mut Var) -> Result<Self, Self::Error> {
    // in this case allow None type, we might be a new variable from a Table or Seq
    if v.valueType == SHType_None {
      v.valueType = SHType_Seq;
    }

    if v.valueType != SHType_Seq {
      Err("Expected Seq variable, but casting failed.")
    } else {
      unsafe {
        Ok(Seq {
          s: v.payload.__bindgen_anon_1.seqValue,
          owned: false,
        })
      }
    }
  }
}

impl TryFrom<&Var> for Seq {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(v: &Var) -> Result<Self, Self::Error> {
    if v.valueType != SHType_Seq {
      Err("Expected Seq variable, but casting failed.")
    } else {
      unsafe {
        Ok(Seq {
          s: v.payload.__bindgen_anon_1.seqValue,
          owned: false,
        })
      }
    }
  }
}

#[repr(transparent)]
pub struct TableVar(pub Var);

impl TableVar {
  #[inline(always)]
  pub(crate) fn new() -> TableVar {
    TableVar(Var {
      valueType: SHType_Table,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 {
          tableValue: unsafe { (*Core).tableNew.unwrap_unchecked()() },
        },
      },
      ..Default::default()
    })
  }

  /// Creates a new `SeqVar` that IS NOT DROPPED when it goes out of scope.
  /// The resulting variable should be wrapped inside a `ClonedVar(v)` or `destroyVar` should be called.
  #[inline(always)]
  pub fn leaking_new() -> TableVar {
    Self::new()
  }

  #[inline(always)]
  pub fn insert(&mut self, k: Var, v: &Var) -> Option<Var> {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      if (*t.api).tableContains.unwrap_unchecked()(t, k) {
        let p = (*t.api).tableAt.unwrap_unchecked()(t, k);
        let old = *p;
        cloneVar(&mut *p, &v);
        Some(old)
      } else {
        let p = (*t.api).tableAt.unwrap_unchecked()(t, k);
        cloneVar(&mut *p, &v);
        None
      }
    }
  }

  #[inline(always)]
  pub fn emplace(&mut self, k: Var, v: ClonedVar) {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let p = (*t.api).tableAt.unwrap_unchecked()(t, k);
      *p = v.0;
      std::mem::forget(v);
    }
  }

  #[inline(always)]
  pub fn insert_fast(&mut self, k: Var, v: &Var) {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let p = (*t.api).tableAt.unwrap_unchecked()(t, k);
      cloneVar(&mut *p, &v);
    }
  }

  #[inline(always)]
  pub fn insert_fast_static(&mut self, k: &str, v: &Var) {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let str_key = Var::ephemeral_string(k);
      let p = (*t.api).tableAt.unwrap_unchecked()(t, str_key);
      cloneVar(&mut *p, &v);
    }
  }

  #[inline(always)]
  pub fn get_mut(&self, k: Var) -> Option<&mut Var> {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      if (*t.api).tableContains.unwrap_unchecked()(t, k) {
        let p = (*t.api).tableAt.unwrap_unchecked()(t, k);
        Some(&mut *p)
      } else {
        None
      }
    }
  }

  #[inline(always)]
  pub fn get_mut_fast(&mut self, k: Var) -> &mut Var {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      &mut *(*t.api).tableAt.unwrap_unchecked()(t, k)
    }
  }

  #[inline(always)]
  pub fn get_mut_fast_static(&mut self, k: &'static str) -> &mut Var {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let str_key = Var::ephemeral_string(k);
      &mut *(*t.api).tableAt.unwrap_unchecked()(t, str_key)
    }
  }

  #[inline(always)]
  pub fn get(&self, k: Var) -> Option<&Var> {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      if (*t.api).tableContains.unwrap_unchecked()(t, k) {
        let p = (*t.api).tableAt.unwrap_unchecked()(t, k);
        Some(&*p)
      } else {
        None
      }
    }
  }

  #[inline(always)]
  pub fn get_static(&self, k: &'static str) -> Option<&Var> {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let key_str = Var::ephemeral_string(k);
      if (*t.api).tableContains.unwrap_unchecked()(t, key_str) {
        let p = (*t.api).tableAt.unwrap_unchecked()(t, key_str);
        Some(&*p)
      } else {
        None
      }
    }
  }

  #[inline(always)]
  pub fn get_fast_static(&self, k: &'static str) -> &Var {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let key_str = Var::ephemeral_string(k);
      &*(*t.api).tableAt.unwrap_unchecked()(t, key_str)
    }
  }

  #[inline(always)]
  pub fn len(&self) -> usize {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      (*t.api).tableSize.unwrap_unchecked()(t) as usize
    }
  }

  #[inline(always)]
  pub fn iter(&self) -> TableIterator {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      let it = TableIterator {
        table: t,
        citer: [0; 64],
      };
      (*t.api).tableGetIterator.unwrap_unchecked()(t, &it.citer as *const _ as *mut _);
      it
    }
  }

  pub fn as_table(&self) -> &Table {
    unsafe {
      let tab_ptr = self.0.payload.__bindgen_anon_1.tableValue.opaque as *const Table;
      &*tab_ptr
    }
  }

  pub fn clear(&mut self) {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      (*t.api).tableClear.unwrap_unchecked()(t);
    }
  }

  pub fn remove(&mut self, k: Var) {
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      (*t.api).tableRemove.unwrap_unchecked()(t, k);
    }
  }

  pub fn remove_static(&mut self, k: &'static str) {
    let k = Var::ephemeral_string(k);
    unsafe {
      let t = self.0.payload.__bindgen_anon_1.tableValue;
      (*t.api).tableRemove.unwrap_unchecked()(t, k);
    }
  }
}

// Table / SHTable

pub struct Table {
  pub t: SHTable, // note, don't derive clone... cos won't work
  owned: bool,
}

impl Drop for Table {
  fn drop(&mut self) {
    if self.owned {
      unsafe {
        (*self.t.api).tableFree.unwrap_unchecked()(self.t);
      }
    }
  }
}

unsafe extern "C" fn table_foreach_callback(
  key: *const ::std::os::raw::c_char,
  value: *mut SHVar,
  userData: *mut ::std::os::raw::c_void,
) -> SHBool {
  let ptrs = userData as *mut (&mut Vec<&str>, &mut Vec<Var>);
  let cstr = CStr::from_ptr(key);
  (*ptrs).0.push(cstr.to_str().unwrap());
  (*ptrs).1.push(*value);
  true // false aborts iteration
}

impl Default for Table {
  fn default() -> Self {
    Self::new()
  }
}

impl AsRef<Table> for Table {
  fn as_ref(&self) -> &Table {
    self
  }
}

impl Table {
  pub fn new() -> Table {
    unsafe {
      Table {
        t: (*Core).tableNew.unwrap_unchecked()(),
        owned: true,
      }
    }
  }

  pub fn from_sh_table(t: SHTable) -> Table {
    Table { t, owned: false }
  }

  pub fn insert(&mut self, k: Var, v: &Var) -> Option<Var> {
    unsafe {
      if (*self.t.api).tableContains.unwrap_unchecked()(self.t, k) {
        let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
        let old = *p;
        cloneVar(&mut *p, &v);
        Some(old)
      } else {
        let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
        cloneVar(&mut *p, &v);
        None
      }
    }
  }

  pub fn insert_fast(&mut self, k: Var, v: &Var) {
    unsafe {
      let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
      cloneVar(&mut *p, &v);
    }
  }

  pub fn insert_fast_static(&mut self, k: &str, v: &Var) {
    unsafe {
      let k = Var::ephemeral_string(k);
      let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
      cloneVar(&mut *p, &v);
    }
  }

  pub fn get_mut(&self, k: Var) -> Option<&mut Var> {
    unsafe {
      if (*self.t.api).tableContains.unwrap_unchecked()(self.t, k) {
        let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
        Some(&mut *p)
      } else {
        None
      }
    }
  }

  pub fn get_mut_fast(&mut self, k: Var) -> &mut Var {
    unsafe { &mut *(*self.t.api).tableAt.unwrap_unchecked()(self.t, k) }
  }

  pub fn get_mut_fast_static(&mut self, k: &'static str) -> &mut Var {
    unsafe {
      let k = Var::ephemeral_string(k);
      &mut *(*self.t.api).tableAt.unwrap_unchecked()(self.t, k)
    }
  }

  pub fn get(&self, k: Var) -> Option<&Var> {
    unsafe {
      if (*self.t.api).tableContains.unwrap_unchecked()(self.t, k) {
        let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
        Some(&*p)
      } else {
        None
      }
    }
  }

  pub fn get_static(&self, k: &'static str) -> Option<&Var> {
    let k = Var::ephemeral_string(k);
    unsafe {
      if (*self.t.api).tableContains.unwrap_unchecked()(self.t, k) {
        let p = (*self.t.api).tableAt.unwrap_unchecked()(self.t, k);
        Some(&*p)
      } else {
        None
      }
    }
  }

  pub fn get_fast_static(&self, k: &'static str) -> &Var {
    let k = Var::ephemeral_string(k);
    unsafe { &*(*self.t.api).tableAt.unwrap_unchecked()(self.t, k) }
  }

  pub fn len(&self) -> usize {
    unsafe { (*self.t.api).tableSize.unwrap_unchecked()(self.t) as usize }
  }

  pub fn iter(&self) -> TableIterator {
    unsafe {
      let it = TableIterator {
        table: self.t,
        citer: [0; 64],
      };
      (*self.t.api).tableGetIterator.unwrap_unchecked()(self.t, &it.citer as *const _ as *mut _);
      it
    }
  }

  pub fn remove(&mut self, k: Var) {
    unsafe {
      (*self.t.api).tableRemove.unwrap_unchecked()(self.t, k);
    }
  }

  pub fn remove_static(&mut self, k: &'static str) {
    let k = Var::ephemeral_string(k);
    unsafe {
      (*self.t.api).tableRemove.unwrap_unchecked()(self.t, k);
    }
  }
}

pub struct TableIterator {
  table: SHTable,
  citer: SHTableIterator,
}

impl Iterator for TableIterator {
  type Item = (Var, Var);
  fn next(&mut self) -> Option<Self::Item> {
    unsafe {
      let k: Var = Var::default();
      let v: Var = Var::default();
      let hasValue = (*(self.table.api)).tableNext.unwrap_unchecked()(
        self.table,
        &self.citer as *const _ as *mut _,
        &k as *const _ as *mut _,
        &v as *const _ as *mut _,
      );
      if hasValue {
        Some((k, v))
      } else {
        None
      }
    }
  }
}

impl From<SHTable> for Table {
  fn from(t: SHTable) -> Self {
    Table { t, owned: false }
  }
}

impl TryFrom<&mut Var> for Table {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &mut Var) -> Result<Self, Self::Error> {
    // in this case allow None type, we might be a new variable from a Table or Seq
    if var.valueType == SHType_None {
      var.valueType = SHType_Table;
    }

    if var.valueType != SHType_Table {
      Err("Expected Table variable, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.tableValue.into()) }
    }
  }
}

impl TryFrom<&Var> for Table {
  type Error = &'static str;

  #[inline(always)]
  fn try_from(var: &Var) -> Result<Self, Self::Error> {
    if var.valueType != SHType_Table {
      Err("Expected Table, but casting failed.")
    } else {
      unsafe { Ok(var.payload.__bindgen_anon_1.tableValue.into()) }
    }
  }
}

impl From<&Table> for Var {
  fn from(t: &Table) -> Self {
    SHVar {
      valueType: SHType_Table,
      payload: SHVarPayload {
        __bindgen_anon_1: SHVarPayload__bindgen_ty_1 { tableValue: t.t },
      },
      ..Default::default()
    }
  }
}

impl PartialEq for Var {
  fn eq(&self, other: &Self) -> bool {
    if self.valueType != other.valueType {
      false
    } else {
      unsafe { (*Core).isEqualVar.unwrap_unchecked()(self as *const _, other as *const _) }
    }
  }
}

impl Eq for Var {}

impl PartialEq for Type {
  fn eq(&self, other: &Type) -> bool {
    if self.basicType != other.basicType {
      false
    } else {
      unsafe { (*Core).isEqualType.unwrap_unchecked()(self as *const _, other as *const _) }
    }
  }
}

pub const FRAG_CC: i32 = fourCharacterCode(*b"frag");

pub static INT_TYPES_SLICE: &[Type] = &[common_type::int];
pub static INT_OR_NONE_TYPES_SLICE: &[Type] = &[common_type::int, common_type::none];
pub static INT2_TYPES_SLICE: &[Type] = &[common_type::int2];
pub static FLOAT_TYPES_SLICE: &[Type] = &[common_type::float];
pub static FLOAT_OR_NONE_TYPES_SLICE: &[Type] = &[common_type::float, common_type::none];
pub static FLOAT2_TYPES_SLICE: &[Type] = &[common_type::float2];
pub static FLOAT3_TYPES_SLICE: &[Type] = &[common_type::float3];
pub static BOOL_TYPES_SLICE: &[Type] = &[common_type::bool];
pub static BOOL_OR_NONE_SLICE: &[Type] = &[common_type::bool, common_type::none];
pub static BOOL_OR_VAR_SLICE: &[Type] = &[common_type::bool, common_type::bool_var];
pub static BOOL_VAR_OR_NONE_SLICE: &[Type] =
  &[common_type::bool, common_type::bool_var, common_type::none];
pub static STRING_TYPES_SLICE: &[Type] = &[common_type::string];
pub static STRING_OR_NONE_SLICE: &[Type] = &[common_type::string, common_type::none];
pub static STRING_VAR_OR_NONE_SLICE: &[Type] = &[
  common_type::string,
  common_type::string_var,
  common_type::none,
];

pub static ANY_TABLE_VAR_NONE_SLICE: &[Type] = &[
  common_type::any_table,
  common_type::any_table_var,
  common_type::none,
];

// TODO share those from C++ ones to reduce binary size
lazy_static! {
  pub static ref ANY_TYPES: Vec<Type> = vec![common_type::any];
  pub static ref WIRE_TYPES: Vec<Type> = vec![common_type::wire];
  pub static ref ANYS_TYPES: Vec<Type> = vec![common_type::anys];
  pub static ref ANY_TABLE_VAR_TYPES: Vec<Type> =
    vec![common_type::any_table, common_type::any_table_var];
  pub static ref ANY_TABLE_TYPES: Vec<Type> = vec![common_type::any_table];
  pub static ref SEQ_OF_ANY_TABLE: Type = Type::seq(&ANY_TABLE_TYPES);
  pub static ref SEQ_OF_ANY_TABLE_TYPES: Vec<Type> = vec![*SEQ_OF_ANY_TABLE, common_type::none];
  pub static ref NONE_TYPES: Vec<Type> = vec![common_type::none];
  pub static ref STRING_TYPES: Vec<Type> = vec![common_type::string];
  pub static ref STRINGS_TYPES: Vec<Type> = vec![common_type::strings];
  pub static ref SEQ_OF_STRINGS: Type = Type::seq(&STRINGS_TYPES);
  pub static ref SEQ_OF_STRINGS_TYPES: Vec<Type> = vec![*SEQ_OF_STRINGS];
  pub static ref COLOR_TYPES: Vec<Type> = vec![common_type::color];
  pub static ref INT_TYPES: Vec<Type> = vec![common_type::int];
  pub static ref INT2_TYPES: Vec<Type> = vec![common_type::int2];
  pub static ref INT3_TYPES: Vec<Type> = vec![common_type::int3];
  pub static ref INT4_TYPES: Vec<Type> = vec![common_type::int4];
  pub static ref INT16_TYPES: Vec<Type> = vec![common_type::int16];
  pub static ref FLOAT_TYPES: Vec<Type> = vec![common_type::float];
  pub static ref SEQ_OF_INT: Type = Type::seq(&INT_TYPES);
  pub static ref SEQ_OF_INT_TYPES: Vec<Type> = vec![*SEQ_OF_INT];
  pub static ref SEQ_OF_SEQ_OF_INT: Type = Type::seq(&SEQ_OF_INT_TYPES);
  pub static ref SEQ_OF_SEQ_OF_INT_TYPES: Vec<Type> = vec![*SEQ_OF_SEQ_OF_INT];
  pub static ref SEQ_OF_FLOAT: Type = Type::seq(&FLOAT_TYPES);
  pub static ref SEQ_OF_FLOAT_TYPES: Vec<Type> = vec![*SEQ_OF_FLOAT];
  pub static ref SEQ_OF_SEQ_OF_FLOAT: Type = Type::seq(&SEQ_OF_FLOAT_TYPES);
  pub static ref SEQ_OF_SEQ_OF_FLOAT_TYPES: Vec<Type> = vec![*SEQ_OF_SEQ_OF_FLOAT];
  pub static ref FLOAT2_TYPES: Vec<Type> = vec![common_type::float2];
  pub static ref FLOAT3_TYPES: Vec<Type> = vec![common_type::float3];
  pub static ref FLOAT4_TYPES: Vec<Type> = vec![common_type::float4];
  pub static ref BOOL_TYPES: Vec<Type> = vec![common_type::bool];
  pub static ref BYTES_TYPES: Vec<Type> = vec![common_type::bytes];
  pub static ref BYTES_OR_STRING_TYPES: Vec<Type> = vec![common_type::bytes, common_type::string];
  pub static ref FLOAT4X4_TYPE: Type = {
    let mut t = common_type::float4s;
    t.fixedSize = 4;
    t
  };
  pub static ref FLOAT4X4_TYPES: Vec<Type> = vec![*FLOAT4X4_TYPE];
  pub static ref FLOAT4X4S_TYPE: Type = Type::seq(&FLOAT4X4_TYPES);
  pub static ref FLOAT4X4orS_TYPES: Vec<Type> = vec![*FLOAT4X4_TYPE, *FLOAT4X4S_TYPE];
  pub static ref FLOAT3X3_TYPE: Type = {
    let mut t = common_type::float3s;
    t.fixedSize = 3;
    t
  };
  pub static ref FLOAT3X3_TYPES: Vec<Type> = vec![*FLOAT3X3_TYPE];
  pub static ref FLOAT3X3S_TYPE: Type = Type::seq(&FLOAT3X3_TYPES);
  pub static ref FLOAT4X2_TYPE: Type = {
    let mut t = common_type::float4s;
    t.fixedSize = 2;
    t
  };
  pub static ref FLOAT4X2_TYPES: Vec<Type> = vec![*FLOAT4X2_TYPE];
  pub static ref FLOAT4X2S_TYPE: Type = Type::seq(&FLOAT4X2_TYPES);
  pub static ref ENUM_TYPE: Type = {
    let mut t = common_type::enumeration;
    t.details.enumeration = SHEnumTypeInfo {
      vendorId: FRAG_CC,
      typeId: fourCharacterCode(*b"type"),
    };
    t
  };
  pub static ref ENUM_TYPES: Vec<Type> = vec![*ENUM_TYPE];
  pub static ref ENUMS_TYPE: Type = Type::seq(&ENUM_TYPES);
  pub static ref ENUMS_TYPES: Vec<Type> = vec![*ENUMS_TYPE];
  pub static ref IMAGE_TYPES: Vec<Type> = vec![common_type::image];
  pub static ref SHARDS_OR_NONE_TYPES: Vec<Type> =
    vec![common_type::none, common_type::shard, common_type::shards];
  pub static ref SEQ_OF_SHARDS: Type = Type::seq(&SHARDS_OR_NONE_TYPES);
  pub static ref SEQ_OF_SHARDS_TYPES: Vec<Type> = vec![*SEQ_OF_SHARDS];
  pub static ref SEQ_OF_SEQ_OF_ANY: Type = Type::seq(&ANYS_TYPES);
  pub static ref SEQ_OF_SEQ_OF_ANY_TYPES: Vec<Type> = vec![*SEQ_OF_SEQ_OF_ANY];
  pub static ref SEQ_OF_INT_OR_FLOAT_TYPES: Vec<Type> = vec![*SEQ_OF_INT, *SEQ_OF_FLOAT];
}

macro_rules! test_to_from_vec1 {
  ($type:ty, $value:expr, $msg:literal) => {
    let fromNum: $type = $value;
    let asVar: Var = fromNum.try_into().unwrap();
    let intoNum: $type = <$type>::try_from(&asVar).unwrap();
    assert_eq!(fromNum, intoNum, $msg);
  };
}

macro_rules! test_to_from_vec2 {
  ($type:ty, $value:expr, $msg:literal) => {
    let fromNum: ($type, $type) = ($value, $value);
    let asVar: Var = fromNum.try_into().unwrap();
    let intoNum: ($type, $type) = <($type, $type)>::try_from(&asVar).unwrap();
    assert_eq!(fromNum, intoNum, $msg);
  };
}

macro_rules! test_to_from_vec3 {
  ($type:ty, $value:expr, $msg:literal) => {
    let fromNum: ($type, $type, $type) = ($value, $value, $value);
    let asVar: Var = fromNum.try_into().unwrap();
    let intoNum: ($type, $type, $type) = <($type, $type, $type)>::try_from(&asVar).unwrap();
    assert_eq!(fromNum, intoNum, $msg);
  };
}

macro_rules! test_to_from_vec4 {
  ($type:ty, $value:expr, $msg:literal) => {
    let fromNum: ($type, $type, $type, $type) = ($value, $value, $value, $value);
    let asVar: Var = fromNum.try_into().unwrap();
    let intoNum: ($type, $type, $type, $type) =
      <($type, $type, $type, $type)>::try_from(&asVar).unwrap();
    assert_eq!(fromNum, intoNum, $msg);
  };
}

#[test]
fn precision_conversion() {
  test_to_from_vec1!(i64, i64::MAX, "i64 conversion failed");
  test_to_from_vec1!(i64, i64::MIN, "i64 conversion failed");
  test_to_from_vec1!(u64, u64::MAX, "u64 conversion failed"); // Don't panic!
  test_to_from_vec1!(u64, u64::MIN, "u64 conversion failed");
  test_to_from_vec1!(f64, f64::MAX, "f64 conversion failed");
  test_to_from_vec1!(f64, f64::MIN, "f64 conversion failed");
  test_to_from_vec1!(f64, f64::MIN_POSITIVE, "f64 conversion failed");
  test_to_from_vec1!(f64, f64::EPSILON, "f64 conversion failed");
  test_to_from_vec1!(f64, f64::INFINITY, "f64 conversion failed");

  test_to_from_vec2!(i64, i64::MAX, "[i64,2] conversion failed");
  test_to_from_vec2!(i64, i64::MIN, "[i64,2] conversion failed");
  test_to_from_vec2!(u64, u64::MAX, "[u64,2] conversion failed"); // Don't panic!
  test_to_from_vec2!(u64, u64::MIN, "[u64,2] conversion failed");
  test_to_from_vec2!(f64, f64::MAX, "[f64,2] conversion failed");
  test_to_from_vec2!(f64, f64::MIN, "[f64,2] conversion failed");
  test_to_from_vec2!(f64, f64::MIN_POSITIVE, "[f64,2] conversion failed");
  test_to_from_vec2!(f64, f64::EPSILON, "[f64,2] conversion failed");
  test_to_from_vec2!(f64, f64::INFINITY, "[f64,2] conversion failed");

  test_to_from_vec3!(i32, i32::MAX, "[i32,3] conversion failed");
  test_to_from_vec3!(i32, i32::MIN, "[i32,3] conversion failed");
  test_to_from_vec3!(u32, u32::MAX, "[u32,3] conversion failed"); // Don't panic!
  test_to_from_vec3!(u32, u32::MIN, "[u32,3] conversion failed");
  test_to_from_vec3!(f32, f32::MAX, "[f32,3] conversion failed");
  test_to_from_vec3!(f32, f32::MIN, "[f32,3] conversion failed");
  test_to_from_vec3!(f32, f32::MIN_POSITIVE, "[f32,3] conversion failed");
  test_to_from_vec3!(f32, f32::EPSILON, "[f32,3] conversion failed");
  test_to_from_vec3!(f32, f32::INFINITY, "[f32,3] conversion failed");

  test_to_from_vec4!(i32, i32::MAX, "[i32,4] conversion failed");
  test_to_from_vec4!(i32, i32::MIN, "[i32,4] conversion failed");
  test_to_from_vec4!(u32, u32::MAX, "[u32,4] conversion failed"); // Don't panic!
  test_to_from_vec4!(u32, u32::MIN, "[u32,4] conversion failed");
  test_to_from_vec4!(f32, f32::MAX, "[f32,4] conversion failed");
  test_to_from_vec4!(f32, f32::MIN, "[f32,4] conversion failed");
  test_to_from_vec4!(f32, f32::MIN_POSITIVE, "[f32,4] conversion failed");
  test_to_from_vec4!(f32, f32::EPSILON, "[f32,4] conversion failed");
  test_to_from_vec4!(f32, f32::INFINITY, "[f32,4] conversion failed");

  test_to_from_vec4!(i16, i16::MAX, "[i16,4] conversion failed");
  test_to_from_vec4!(i16, i16::MIN, "[i16,4] conversion failed");
  test_to_from_vec4!(u16, u16::MAX, "[u16,4] conversion failed");
  test_to_from_vec4!(u16, u16::MIN, "[u16,4] conversion failed");
  test_to_from_vec4!(
    half::f16,
    half::f16::ZERO,
    "[half::f16,4] conversion failed"
  );
  test_to_from_vec4!(half::f16, half::f16::MAX, "[half::f16,4] conversion failed");
  test_to_from_vec4!(half::f16, half::f16::MIN, "[half::f16,4] conversion failed");
  test_to_from_vec4!(
    half::f16,
    half::f16::MIN_POSITIVE,
    "[half::f16,4] conversion failed"
  );
  test_to_from_vec4!(
    half::f16,
    half::f16::EPSILON,
    "[half::f16,4] conversion failed"
  );
  test_to_from_vec4!(
    half::f16,
    half::f16::INFINITY,
    "[half::f16,4] conversion failed"
  );
}
