/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */
use crate::shard::legacy_shard_construct;
use crate::shard::shard_construct;
use crate::shard::LegacyShard;
use crate::shard::{Shard, ShardGenerated, ShardGeneratedOverloads};
use crate::shardsc::Shard as SHShard;
use crate::shardsc::*;
use crate::shlog_debug;
use crate::types::ClonedVar;
use crate::types::Context;
use crate::types::DerivedType;
use crate::types::ExternalVar;
use crate::types::InstanceData;
use crate::types::Mesh;
use crate::types::ParameterInfo;
use crate::types::Parameters;
use crate::types::RCObjectVar;
use crate::types::Var;
use crate::types::WireRef;
use crate::types::WireState;
use crate::util::from_raw_parts_allow_null;
use core::convert::TryInto;
use core::ffi::c_void;
use core::slice;
use futures_util::pin_mut;
use std::ffi::CStr;
use std::ffi::CString;
use std::future::Future;
use std::os::raw::c_char;

const ABI_VERSION: u32 = 0x20200101;

pub const MIN_STACK_SIZE: usize = crate::shardsc::SH_MIN_STACK_SIZE as usize;

pub static mut Core: *mut SHCore = core::ptr::null_mut();
pub static mut ScriptEnvCreate: Option<
  unsafe extern "C" fn(path: *const ::std::os::raw::c_char) -> *mut ::core::ffi::c_void,
> = None;
pub static mut ScriptEnvCreateSub: Option<
  unsafe extern "C" fn(parent_env: *mut ::core::ffi::c_void) -> *mut ::core::ffi::c_void,
> = None;
pub static mut ScriptEnvDestroy: Option<unsafe extern "C" fn(env: *mut ::core::ffi::c_void)> = None;
pub static mut ScriptEval: Option<
  unsafe extern "C" fn(
    env: *mut ::core::ffi::c_void,
    script: *const ::std::os::raw::c_char,
    output: *mut SHVar,
  ) -> bool,
> = None;
static mut init_done: bool = false;

#[cfg(feature = "dllshard")]
mod internal_core_init {
  extern crate dlopen;
  use super::*;
  use crate::core::SHCore;
  use crate::core::ABI_VERSION;
  use dlopen::symbor::Library;

  fn try_load_dlls() -> Option<Library> {
    if let Ok(lib) = Library::open("libshards.dylib") {
      Some(lib)
    } else if let Ok(lib) = Library::open("libshards.so") {
      Some(lib)
    } else if let Ok(lib) = Library::open("libshards.dll") {
      Some(lib)
    } else {
      None
    }
  }

  pub static mut SHDLL: Option<Library> = None;

  pub unsafe fn initScripting(lib: &Library) {
    let fun = lib.symbol::<unsafe extern "C" fn(
      path: *const ::std::os::raw::c_char,
    ) -> *mut ::core::ffi::c_void>("shLispCreate");
    if let Ok(fun) = fun {
      ScriptEnvCreate = Some(*fun);
    } else {
      // short circuit here
      return;
    }

    let fun = lib.symbol::<unsafe extern "C" fn(env: *mut ::core::ffi::c_void)>("shLispDestroy");
    ScriptEnvDestroy = Some(*fun.unwrap());

    let fun = lib
      .symbol::<unsafe extern "C" fn(env: *mut ::core::ffi::c_void) -> *mut ::core::ffi::c_void>(
        "shLispCreateSub",
      );
    ScriptEnvCreateSub = Some(*fun.unwrap());

    let fun = lib.symbol::<unsafe extern "C" fn(
      env: *mut ::core::ffi::c_void,
      script: *const ::std::os::raw::c_char,
      output: *mut SHVar,
    ) -> bool>("shLispEval");
    ScriptEval = Some(*fun.unwrap());

    // trigger initializations... fix me in the future to something more elegant
    let current_dir = std::env::current_dir().unwrap();
    let current_dir = current_dir.to_str().unwrap();
    let current_dir = std::ffi::CString::new(current_dir).unwrap();
    let env = ScriptEnvCreate.unwrap()(current_dir.as_ptr());
    ScriptEnvDestroy.unwrap()(env);
  }

  pub unsafe fn initInternal() {
    let exe = Library::open_self().ok().unwrap();

    let exefun = exe
      .symbol::<unsafe extern "C" fn(abi_version: u32) -> *mut SHCore>("shardsInterface")
      .ok();
    if let Some(fun) = exefun {
      // init scripting first if possible!
      initScripting(&exe);
      Core = fun(ABI_VERSION);
      if Core.is_null() {
        panic!("Failed to aquire shards interface, version not compatible.");
      }
    } else {
      let lib = try_load_dlls().unwrap();
      let fun = lib
        .symbol::<unsafe extern "C" fn(abi_version: u32) -> *mut SHCore>("shardsInterface")
        .unwrap();
      // init scripting first if possible!
      initScripting(&lib);
      Core = fun(ABI_VERSION);
      if Core.is_null() {
        panic!("Failed to aquire shards interface, version not compatible.");
      }
      SHDLL = Some(lib);
    }
  }
}

#[cfg(not(feature = "dllshard"))]
mod internal_core_init {
  pub unsafe fn initInternal() {}
}

#[inline(always)]
pub fn init() {
  unsafe {
    if !init_done {
      internal_core_init::initInternal();
      init_done = true;
    }
  }
}

#[inline(always)]
pub fn sleep(seconds: f64) {
  unsafe {
    (*Core).sleep.unwrap_unchecked()(seconds);
  }
}

#[inline(always)]
pub fn step_count(context: &SHContext) -> u64 {
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    (*Core).getStepCount.unwrap_unchecked()(ctx)
  }
}

#[inline(always)]
pub fn suspend(context: &SHContext, seconds: f64) -> WireState {
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    (*Core).suspend.unwrap_unchecked()(ctx, seconds).into()
  }
}

#[inline(always)]
pub fn getState(context: &SHContext) -> WireState {
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    (*Core).getState.unwrap_unchecked()(ctx).into()
  }
}

#[inline(always)]
pub fn abortWire(context: &SHContext, message: &str) {
  let msg = SHStringWithLen {
    string: message.as_ptr() as *const c_char,
    len: message.len() as u64,
  };
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    (*Core).abortWire.unwrap_unchecked()(ctx, msg);
  }
}

// SHBool shards_cancel_abort(SHContext *context)
// Not exposed in the C API, but used internally
extern "C" {
  fn shards_cancel_abort(context: *mut SHContext) -> SHBool;
}

#[inline(always)]
pub fn cancel_abort(context: &SHContext) -> bool {
  unsafe { shards_cancel_abort(context as *const SHContext as *mut SHContext) }
}

#[inline(always)]
pub fn register_legacy_shard<T: Default + LegacyShard>() {
  unsafe {
    (*Core).registerShard.unwrap_unchecked()(
      T::registerName().as_ptr() as *const c_char,
      Some(legacy_shard_construct::<T>),
    );
  }
}

#[inline(always)]
pub fn register_shard<T: Default + ShardGenerated + Shard + ShardGeneratedOverloads>() {
  unsafe {
    (*Core).registerShard.unwrap_unchecked()(
      T::register_name().as_ptr() as *const c_char,
      Some(shard_construct::<T>),
    );
  }
}

pub trait EnumRegister {
  fn register();
}

pub fn register_enum<T: EnumRegister>() {
  T::register();
}

pub fn getShards() -> Vec<&'static CStr> {
  unsafe {
    let shard_names = (*Core).getShards.unwrap_unchecked()();
    let mut res = Vec::new();
    let len = shard_names.len;
    let slice = from_raw_parts_allow_null(shard_names.elements, len.try_into().unwrap());
    for name in slice.iter() {
      res.push(CStr::from_ptr(*name));
    }
    (*Core).stringsFree.unwrap_unchecked()(&shard_names as *const SHStrings as *mut SHStrings);
    res
  }
}

#[inline(always)]
pub fn getRootPath() -> &'static str {
  unsafe {
    CStr::from_ptr((*Core).getRootPath.unwrap_unchecked()())
      .to_str()
      .unwrap()
  }
}

#[inline(always)]
pub fn createShardPtr(name: &str) -> ShardPtr {
  let name = SHStringWithLen {
    string: name.as_ptr() as *const c_char,
    len: name.len() as u64,
  };
  unsafe { (*Core).createShard.unwrap_unchecked()(name) }
}

#[inline(always)]
pub fn cloneVar(dst: &mut Var, src: &Var) {
  unsafe {
    (*Core).cloneVar.unwrap_unchecked()(dst, src);
  }
}

#[inline(always)]
pub fn destroyVar(v: &mut Var) {
  unsafe {
    (*Core).destroyVar.unwrap_unchecked()(v);
  }
}

pub fn readCachedString(id: u32) -> &'static str {
  unsafe {
    let s = (*Core).readCachedString.unwrap_unchecked()(id);
    CStr::from_ptr(s.string).to_str().unwrap()
  }
}

pub fn writeCachedString(id: u32, string: &'static str) -> &'static str {
  unsafe {
    let s = (*Core).writeCachedString.unwrap_unchecked()(
      id,
      string.as_ptr() as *const std::os::raw::c_char,
    );
    CStr::from_ptr(s.string).to_str().unwrap()
  }
}

pub fn readCachedString1(id: u32) -> SHOptionalString {
  unsafe { (*Core).readCachedString.unwrap_unchecked()(id) }
}

pub fn writeCachedString1(id: u32, string: &'static str) -> SHOptionalString {
  unsafe {
    (*Core).writeCachedString.unwrap_unchecked()(id, string.as_ptr() as *const std::os::raw::c_char)
  }
}

pub fn deriveType(var: &Var, data: &InstanceData, mutable_: bool) -> DerivedType {
  let t = unsafe {
    (*Core).deriveTypeInfo.unwrap_unchecked()(var as *const _, data as *const _, mutable_)
  };
  DerivedType(t)
}

#[macro_export]
macro_rules! shccstr {
  ($string:literal) => {
    if cfg!(debug_assertions) {
      shards::core::writeCachedString1($crate::crc32!($string), shards::cstr!($string))
    } else {
      shards::core::readCachedString1($crate::crc32!($string))
    }
  };
}

pub fn referenceMutVariable(context: &SHContext, name: SHStringWithLen) -> &mut SHVar {
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    let shptr = (*Core).referenceVariable.unwrap_unchecked()(ctx, name);
    shptr.as_mut().unwrap()
  }
}

pub fn referenceVariable(context: &SHContext, name: SHStringWithLen) -> &SHVar {
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    let shptr = (*Core).referenceVariable.unwrap_unchecked()(ctx, name);
    shptr.as_mut().unwrap()
  }
}

pub fn releaseMutVariable(var: &mut SHVar) {
  unsafe {
    let v = var as *mut SHVar;
    (*Core).releaseVariable.unwrap_unchecked()(v);
  }
}

pub fn releaseVariable(var: &SHVar) {
  unsafe {
    let v = var as *const SHVar as *mut SHVar;
    (*Core).releaseVariable.unwrap_unchecked()(v);
  }
}

pub fn register_object_type_internal(vendorId: i32, typeId: i32, info: SHObjectInfo) {
  unsafe {
    (*Core).registerObjectType.unwrap_unchecked()(vendorId, typeId, info);
  }
}

pub fn register_object_type<T: RCObjectVar>(vendor_id: i32, type_id: i32) {
  let info = T::get_info();
  register_object_type_internal(vendor_id, type_id, unsafe { *info });
}

pub fn register_enum_internal(vendorId: i32, typeId: i32, info: SHEnumInfo) {
  unsafe {
    (*Core).registerEnumType.unwrap_unchecked()(vendorId, typeId, info);
  }
}

pub fn findEnumInfo(vendorId: i32, typeId: i32) -> Option<SHEnumInfo> {
  let info = unsafe { (*Core).findEnumInfo.unwrap_unchecked()(vendorId, typeId) };
  if info == std::ptr::null() {
    None
  } else {
    Some(unsafe { *info })
  }
}

pub fn findEnumId(name: &str) -> Option<i64> {
  let s = SHStringWithLen {
    string: name.as_ptr() as *const c_char,
    len: name.len() as u64,
  };
  let info = unsafe { (*Core).findEnumId.unwrap_unchecked()(s) };
  if info == 0 {
    None
  } else {
    Some(info)
  }
}

pub fn findObjectInfo(vendorId: i32, typeId: i32) -> *const SHObjectInfo {
  unsafe { (*Core).findObjectInfo.unwrap_unchecked()(vendorId, typeId) }
}

pub fn type2Name(t: SHType) -> *const c_char {
  unsafe { (*Core).type2Name.unwrap_unchecked()(t) }
}

impl WireRef {
  pub fn set_external(&self, name: &str, var: &mut ExternalVar) {
    let name = SHStringWithLen {
      string: name.as_ptr() as *const c_char,
      len: name.len() as u64,
    };
    unsafe {
      let ev = SHExternalVariable {
        var: &var.0 as *const _ as *mut _,
        type_: std::ptr::null(),
      };
      (*Core).setExternalVariable.unwrap_unchecked()(self.0, name, &ev as *const _ as *mut _);
    }
  }

  pub fn remove_external(&self, name: &str) {
    let name = SHStringWithLen {
      string: name.as_ptr() as *const c_char,
      len: name.len() as u64,
    };
    unsafe {
      (*Core).removeExternalVariable.unwrap_unchecked()(self.0, name);
    }
  }

  pub fn get_result(&self) -> Result<Option<ClonedVar>, &str> {
    let info = unsafe { (*Core).getWireInfo.unwrap_unchecked()(self.0) };

    if !info.isRunning {
      if info.failed {
        let slice = unsafe {
          from_raw_parts_allow_null(
            info.failureMessage.string as *const u8,
            info.failureMessage.len as usize,
          )
        };
        let msg = std::str::from_utf8(slice).unwrap();
        Err(msg)
      } else {
        unsafe { Ok(Some((*info.finalOutput).into())) }
      }
    } else {
      Ok(None)
    }
  }
}

//--------------------------------------------------------------------------------------------------

pub trait BlockingShard {
  fn activate_blocking(&mut self, context: &Context, input: &Var) -> Result<Var, &'static str>;
  fn cancel_activation(&mut self, _context: &Context) {}
}

struct AsyncCallData<T: BlockingShard> {
  caller: *mut T,
  input: *const SHVar,
}

unsafe extern "C" fn activate_blocking_c_call<T: BlockingShard>(
  context: *mut SHContext,
  arg2: *mut c_void,
) -> SHVar {
  let data = arg2 as *mut AsyncCallData<T>;
  let res = (*(*data).caller).activate_blocking(&*context, &*(*data).input);
  match res {
    Ok(value) => value,
    Err(error) => {
      shlog_debug!("activate_blocking failure detected"); // in case error leads to crash
      shlog_debug!("activate_blocking failure: {}", error);
      // we fail on another thread so we cannot call abortWire directly as it would race
      let error = Var::ephemeral_string(error);
      let mut strongClone = Var::default();
      cloneVar(&mut strongClone, &error);
      strongClone.flags = SHVAR_FLAGS_ABORT as u16;
      strongClone
    }
  }
}

unsafe extern "C" fn cancel_blocking_c_call<T: BlockingShard>(
  context: *mut SHContext,
  arg2: *mut c_void,
) {
  let data = arg2 as *mut AsyncCallData<T>;
  (*(*data).caller).cancel_activation(&*context);
}

unsafe extern "C" fn activate_future_c_call<
  F: Future<Output = Result<R, &'static str>> + Send + 'static,
  R: Into<ClonedVar>,
>(
  _context: *mut SHContext,
  arg2: *mut c_void,
) -> SHVar {
  let f = arg2 as *mut F;
  let res = futures::executor::block_on(f.read()); // this will consume f
  match res {
    Ok(value) => {
      // SAFETY: We are unsafely managing memory here on purpose because run_future returns a ClonedVar
      let mut cloned = value.into();
      let value = std::mem::take(&mut cloned.0);
      std::mem::forget(cloned);
      value
    }
    Err(error) => {
      shlog_debug!("activate_future failure detected"); // in case error leads to crash
      shlog_debug!("activate_future failure: {}", error);
      // we fail on another thread so we cannot call abortWire directly as it would race
      let error = Var::ephemeral_string(error);
      let mut strongClone = Var::default();
      cloneVar(&mut strongClone, &error);
      strongClone.flags = SHVAR_FLAGS_ABORT as u16;
      strongClone
    }
  }
}

pub fn run_future<
  'a,
  F: Future<Output = Result<R, &'static str>> + Send + 'static,
  R: Into<ClonedVar>,
>(
  context: &'a SHContext,
  f: F,
) -> Result<ClonedVar, &'static str> {
  unsafe {
    let ctx = context as *const SHContext as *mut SHContext;
    let data_ptr = &f as *const F as *mut F as *mut c_void;
    // see note in activate_future_c_call
    let result = ClonedVar((*Core).asyncActivate.unwrap_unchecked()(
      ctx,
      data_ptr,
      Some(activate_future_c_call::<F, R>),
      None,
    ));

    // before exiting, we need to forget f, otherwise we double free
    std::mem::forget(f);

    if result.0.flags & SHVAR_FLAGS_ABORT as u16 != 0 {
      Err("Failed to run future")
    } else {
      Ok(result)
    }
  }
}

pub fn run_blocking<'a, T>(caller: &'a mut T, context: &'a SHContext, input: &'a SHVar) -> SHVar
where
  T: BlockingShard,
{
  unsafe {
    let data = AsyncCallData {
      caller: caller as *mut T,
      input: input as *const SHVar,
    };
    let ctx = context as *const SHContext as *mut SHContext;
    let data_ptr = &data as *const AsyncCallData<T> as *mut AsyncCallData<T> as *mut c_void;
    (*Core).asyncActivate.unwrap_unchecked()(
      ctx,
      data_ptr,
      Some(activate_blocking_c_call::<T>),
      Some(cancel_blocking_c_call::<T>),
    )
  }
}

//--------------------------------------------------------------------------------------------------

impl core::fmt::Debug for SHVar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SHVar")
      .field(
        "payload",
        &SHVarPayloadDebugView {
          payload: &self.payload,
          valueType: self.valueType,
        },
      )
      .field("valueType", &self.valueType)
      .field("innerType", &self.innerType)
      .finish_non_exhaustive()
  }
}

struct SHVarPayloadDebugView<'a> {
  payload: &'a SHVarPayload,
  valueType: SHType,
}

impl core::fmt::Debug for SHVarPayloadDebugView<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    unsafe {
      match self.valueType {
        SHType_None => f.write_str("None"),
        SHType_Any => f.write_str("Any"),
        SHType_Enum => f
          .debug_struct("SHEnum")
          .field(
            "value",
            &self.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue,
          )
          .field(
            "vendorId",
            &self.payload.__bindgen_anon_1.__bindgen_anon_3.enumVendorId,
          )
          .field(
            "typeId",
            &self.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId,
          )
          .finish(),
        SHType_Bool => write!(f, "{:?}", self.payload.__bindgen_anon_1.boolValue),
        SHType_Int => write!(f, "{:?}", self.payload.__bindgen_anon_1.intValue),
        SHType_Int2 => write!(f, "{:?}", self.payload.__bindgen_anon_1.int2Value),
        SHType_Int3 => write!(f, "{:?}", self.payload.__bindgen_anon_1.int3Value),
        SHType_Int4 => write!(f, "{:?}", self.payload.__bindgen_anon_1.int4Value),
        SHType_Int8 => write!(f, "{:?}", self.payload.__bindgen_anon_1.int8Value),
        SHType_Int16 => write!(f, "{:?}", self.payload.__bindgen_anon_1.int16Value),
        SHType_Float => write!(f, "{:?}", self.payload.__bindgen_anon_1.floatValue),
        SHType_Float2 => write!(f, "{:?}", self.payload.__bindgen_anon_1.float2Value),
        SHType_Float3 => write!(f, "{:?}", self.payload.__bindgen_anon_1.float3Value),
        SHType_Float4 => write!(f, "{:?}", self.payload.__bindgen_anon_1.float4Value),
        SHType_Color => write!(f, "{:?}", self.payload.__bindgen_anon_1.colorValue),
        SHType_ShardRef => write!(f, "{:p}", self.payload.__bindgen_anon_1.shardValue),
        SHType_Bytes => f
          .debug_struct("SHBytes")
          .field(
            "value",
            &format_args!(
              "{:p}",
              self.payload.__bindgen_anon_1.__bindgen_anon_4.bytesValue
            ),
          )
          .field(
            "size",
            &self.payload.__bindgen_anon_1.__bindgen_anon_4.bytesSize,
          )
          .field(
            "capacity",
            &self.payload.__bindgen_anon_1.__bindgen_anon_4.bytesCapacity,
          )
          .finish(),
        SHType_String => write!(
          f,
          "{:?}",
          CStr::from_ptr(self.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue)
            .to_str()
            .unwrap()
        ),
        SHType_Path => write!(
          f,
          "Path({:?})",
          CStr::from_ptr(self.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue)
            .to_str()
            .unwrap()
        ),
        SHType_ContextVar => write!(
          f,
          "ContextVar({:?})",
          CStr::from_ptr(self.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue)
            .to_str()
            .unwrap()
        ),
        SHType_Image => write!(f, "{:?}", self.payload.__bindgen_anon_1.imageValue),
        SHType_Seq => write!(f, "{:?}", self.payload.__bindgen_anon_1.seqValue),
        SHType_Table => write!(f, "{:?}", self.payload.__bindgen_anon_1.tableValue),
        SHType_Wire => write!(f, "{:p}", self.payload.__bindgen_anon_1.wireValue),
        SHType_Object => f
          .debug_struct("SHObject")
          .field(
            "value",
            &format_args!(
              "{:p}",
              self.payload.__bindgen_anon_1.__bindgen_anon_1.objectValue
            ),
          )
          .field(
            "vendorId",
            &self
              .payload
              .__bindgen_anon_1
              .__bindgen_anon_1
              .objectVendorId,
          )
          .field(
            "typeId",
            &self.payload.__bindgen_anon_1.__bindgen_anon_1.objectTypeId,
          )
          .finish(),
        SHType_Audio => write!(f, "{:?}", self.payload.__bindgen_anon_1.audioValue),
        _ => f.write_str("???"),
      }
    }
  }
}

use std::hash::{Hash, Hasher};

impl Hash for Var {
  fn hash<H: Hasher>(&self, state: &mut H) {
    let hash = unsafe { (*Core).hashVar.unwrap_unchecked()(self as *const _) };
    unsafe {
      state.write_u64(hash.payload.__bindgen_anon_1.int2Value[0] as u64);
      state.write_u64(hash.payload.__bindgen_anon_1.int2Value[1] as u64);
    }
  }
}

pub fn hash_var(v: &Var) -> Var {
  unsafe { (*Core).hashVar.unwrap_unchecked()(v as *const _) }
}

/// A struct that will execute a closure when it goes out of scope.
pub struct Defer<F: FnOnce()> {
  cleanup: Option<F>,
}

impl<F: FnOnce()> Defer<F> {
  /// Create a new Defer instance with the given closure.
  pub fn new(cleanup: F) -> Self {
    Defer {
      cleanup: Some(cleanup),
    }
  }
}

impl<F: FnOnce()> Drop for Defer<F> {
  fn drop(&mut self) {
    if let Some(cleanup) = self.cleanup.take() {
      cleanup();
    }
  }
}

/// Helper function to create a Defer instance.
pub fn defer<F: FnOnce()>(cleanup: F) -> Defer<F> {
  Defer::new(cleanup)
}
