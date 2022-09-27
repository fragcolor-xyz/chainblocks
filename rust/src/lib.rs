/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#![cfg_attr(all(target_os = "windows", target_arch = "x86"), feature(abi_thiscall))]

#![feature(allocator_api)]

extern crate approx;

#[macro_use]
extern crate lazy_static;

extern crate compile_time_crc32;

pub mod core;
#[macro_use]
pub mod shard;
mod shardsc;
#[macro_use]
pub mod types;
// order matters
#[cfg(feature = "shards")]
pub mod shards;


use crate::core::Core;
use crate::shard::Shard;
pub use crate::shardsc::*;
use crate::types::Types;
use crate::types::Var;

use std::ffi::CString;

pub const fn four_character_code(val: [u8; 4]) -> i32 {
  let unsigned_val = ((val[0] as u32) << 24 & 0xff000000)
    | ((val[1] as u32) << 16 & 0x00ff0000)
    | ((val[2] as u32) << 8 & 0x0000ff00)
    | ((val[3] as u32) & 0x000000ff);
  unsigned_val as i32
}

pub trait IntoVar {
  fn into_var(self) -> Var;
}

impl IntoVar for &str {
  fn into_var(self) -> Var {
    Var::ephemeral_string(self)
  }
}

impl<T: Into<Var>> IntoVar for T
where
  Var: From<T>,
{
  fn into_var(self) -> Var {
    Var::from(self)
  }
}

#[cfg(feature = "scripting")]
pub mod scripting {
  use crate::types::ClonedVar;
  use std::convert::TryInto;

  pub struct ScriptEnv(pub *mut ::core::ffi::c_void);

  impl Drop for ScriptEnv {
    fn drop(&mut self) {
      unsafe {
        crate::core::ScriptEnvDestroy.unwrap()(self.0);
      }
    }
  }

  pub fn new_env() -> ScriptEnv {
    unsafe {
      let current_dir = std::env::current_dir().unwrap();
      let current_dir = current_dir.to_str().unwrap();
      let current_dir = std::ffi::CString::new(current_dir).unwrap();
      let env = crate::core::ScriptEnvCreate.unwrap()(current_dir.as_ptr());
      ScriptEnv(env)
    }
  }

  pub fn new_sub_env(env: &ScriptEnv) -> ScriptEnv {
    unsafe {
      let cenv = crate::core::ScriptEnvCreateSub.unwrap()(env.0);
      ScriptEnv(cenv)
    }
  }

  #[macro_export]
  macro_rules! shards_no_env {
    ($code:expr) => {
      unsafe {
        let code = std::ffi::CString::new($code).expect("CString failed...");
        let mut output = ClonedVar::default();
        let ok = $crate::core::shLispEval(::core::ptr::null_mut(), code.as_ptr(), &mut output.0);
        if ok {
          Some(output)
        } else {
          None
        }
      }
    };
  }

  #[macro_export]
  macro_rules! shards {
    ($code:expr) => {{
      thread_local! {
        static ENV: $crate::scripting::ScriptEnv = {
          let current_dir = std::env::current_dir().unwrap();
          let current_dir = current_dir.to_str().unwrap();
          let current_dir = std::ffi::CString::new(current_dir).unwrap();
          unsafe { $crate::scripting::ScriptEnv($crate::core::ScriptEnvCreate.unwrap()(current_dir.as_ptr())) }
        }
      }
      unsafe {
        let code = std::ffi::CString::new($code).expect("CString failed...");
        let mut output = ClonedVar::default();
        let ok = ENV.with(|env| $crate::core::ScriptEval.unwrap()((*env).0, code.as_ptr(), &mut output.0));
        if ok {
          Some(output)
        } else {
          None
        }
      }
    }};
  }

  #[macro_export]
  macro_rules! shards_env {
    ($env:expr, $code:expr) => {{
      unsafe {
        let code = std::ffi::CString::new($code).expect("CString failed...");
        let mut output = ClonedVar::default();
        let env = $env.0;
        let ok = $crate::core::ScriptEval.unwrap()(env, code.as_ptr(), &mut output.0);
        if ok {
          Some(output)
        } else {
          None
        }
      }
    }};
  }

  pub fn test_shl() {
    // the string is stored at compile time, ideally we should compress them all!
    let res = shards!(include_str!("test.edn")).unwrap();
    let res: &str = res.0.as_ref().try_into().unwrap();
    assert_eq!(res, "Hello");
  }
}

#[cfg(feature = "shards")]
#[no_mangle]
pub extern "C" fn registerRustShards(core: *mut SHCore) {
  unsafe {
    Core = core;
  }

  #[cfg(not(target_arch = "wasm32"))]
  shards::http::register_shards();

  shards::casting::register_shards();
  shards::hash::register_shards();
  shards::ecdsa::register_shards();
  shards::physics::simulation::register_shards();
  shards::physics::shapes::register_shards();
  shards::physics::rigidbody::register_shards();
  shards::physics::queries::register_shards();
  shards::physics::forces::register_shards();
  shards::svg::register_shards();
  shards::eth::register_shards();
  shards::csv::register_shards();
  shards::curve25519::register_shards();
  shards::substrate::register_shards();
  shards::chachapoly::register_shards();
  shards::gui::register_shards();
  shards::date::register_shards();
  shards::onnx::register_shards();

  #[cfg(not(any(target_arch = "wasm32", target_os = "ios")))]
  shards::browse::register_shards();
}

#[no_mangle]
pub extern "C" fn runRuntimeTests() {
  #[cfg(feature = "scripting")]
  scripting::test_shl();
}
