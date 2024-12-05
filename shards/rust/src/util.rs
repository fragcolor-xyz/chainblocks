use std::ffi::{CStr, CString};

use crate::{
  core::{deriveType, referenceVariable, releaseVariable},
  types::{
    Context, DerivedType, ExposedInfo, ExposedTypes, ParamVar, SeqVar, ShardsVar, TableVar, Type,
    Var,
  },
  SHExposedTypeInfo, SHExposedTypesInfo, SHInstanceData, SHString, SHVar,
};

pub enum TypeOrDerived {
  Derived(DerivedType),
  Static(Type),
}

impl<'a> From<&'a TypeOrDerived> for &'a Type {
  fn from(value: &'a TypeOrDerived) -> &'a Type {
    match value {
      TypeOrDerived::Derived(t) => &t.0,
      TypeOrDerived::Static(t) => &t,
    }
  }
}

pub fn get_param_var_type(
  instance_data: &SHInstanceData,
  var: &ParamVar,
) -> Result<TypeOrDerived, &'static str> {
  if var.is_variable() {
    let exp_type = find_exposed_variable(&instance_data.shared, &var.get_param())?
      .ok_or("Could not find exposed variable for parameter")?;
    Ok(TypeOrDerived::Static(exp_type.exposedType))
  } else {
    Ok(TypeOrDerived::Derived(deriveType(
      &var.get_param(),
      instance_data,
      false,
    )))
  }
}

pub fn find_exposed_variable(
  shared: &SHExposedTypesInfo,
  var: &SHVar,
) -> Result<Option<SHExposedTypeInfo>, &'static str> {
  let var_name: &str = var
    .try_into()
    .map_err(|_x| "Invalid context variable name")?;
  for entry in shared {
    let cstr = unsafe { CStr::from_ptr(entry.name) };
    if var_name == cstr.to_str().map_err(|_x| "invalid string")? {
      return Ok(Some(ExposedInfo::new(
        unsafe { var.payload.__bindgen_anon_1.__bindgen_anon_2.stringValue },
        entry.exposedType,
      )));
    }
  }
  return Ok(None);
}

pub fn collect_required_variables(
  shared: &SHExposedTypesInfo,
  out: &mut ExposedTypes,
  var: &SHVar,
) -> Result<(), &'static str> {
  match var.valueType {
    crate::SHType_ContextVar => {
      if let Some(exposed_var) = find_exposed_variable(shared, var)? {
        out.push(exposed_var);
      }
    }
    crate::SHType_Seq => {
      for v in SeqVar(*var) {
        collect_required_variables(shared, out, &v)?;
      }
    }
    crate::SHType_Table => {
      for (_k, v) in TableVar(*var) {
        collect_required_variables(shared, out, &v)?;
      }
    }
    _ => {}
  }
  Ok(())
}

/// Adds required variables from inside a ShardsVar
pub fn require_shards_contents(required: &mut ExposedTypes, contents: &ShardsVar) -> bool {
  if !contents.is_empty() {
    if let Some(requiring) = contents.get_requiring() {
      for exp in requiring {
        required.push(*exp);
      }
      true
    } else {
      false
    }
  } else {
    false
  }
}

/// Adds exposed variables from within a ShardsVar
pub fn expose_shards_contents(exposed: &mut ExposedTypes, contents: &ShardsVar) -> bool {
  if !contents.is_empty() {
    if let Some(exposing) = contents.get_exposing() {
      for exp in exposing {
        exposed.push(*exp);
      }
      true
    } else {
      false
    }
  } else {
    false
  }
}

/// Use to resolve variables for ContextVars nested inside parameters
pub fn get_or_var<'a: 'c, 'b: 'c, 'c>(v: &'a Var, ctx: &'b Context) -> &'c Var {
  if v.is_context_var() {
    unsafe {
      let str = v.payload.__bindgen_anon_1.__bindgen_anon_2;
      let result = referenceVariable(
        ctx,
        crate::SHStringWithLen {
          string: str.stringValue,
          len: str.stringLen as u64,
        },
      );
      releaseVariable(result);
      result
    }
  } else {
    &v
  }
}

pub fn merge_exposed_types(exposed: &mut ExposedTypes, types: &SHExposedTypesInfo) {
  for t in types {
    exposed.push(t);
  }
}

/// Creates a slice from a raw pointer and a length, allowing a null pointer when length is 0.
///
/// # Safety
///
/// The caller must ensure that the pointer is valid and points to `len` elements of type `T`.
/// If `len` is 0, the pointer can be null. The resulting slice must not be used after the memory
/// it points to has been deallocated or modified.
///
/// # Panics
///
/// This function does not panic.
#[inline]
pub unsafe fn from_raw_parts_allow_null<'a, T>(data: *const T, len: usize) -> &'a [T] {
  if len == 0 {
      &[]
  } else {
      std::slice::from_raw_parts(data, len)
  }
}