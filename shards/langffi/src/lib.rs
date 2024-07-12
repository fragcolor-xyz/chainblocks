use shards::core::{register_enum, register_legacy_shard, register_shard};
use shards::util::from_raw_parts_allow_null;
use shards::SHStringWithLen;
use shards::{shlog_error, types::*};
use shards_lang::cli::process_args;
use shards_lang::eval::{self, *};
use shards_lang::read::AST_TYPE;
use shards_lang::{ast::*, RcStrWrapper};
use shards_lang::{print, read};
use std::collections::HashMap;
use std::ffi::{c_char, CString};

#[repr(C)]
pub struct SHLError {
  message: *mut c_char,
  line: u32,
  column: u32,
}

#[repr(C)]
pub struct SHLAst {
  /// of Program ast object, ref counted, count at 0 when returned, receiver must clone it!
  ast: Var,
  error: *mut SHLError,
}

#[repr(C)]
pub struct SHLWire {
  wire: *mut Wire,
  error: *mut SHLError,
}

#[no_mangle]
pub extern "C" fn shards_init(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }
}

#[no_mangle]
pub extern "C" fn shards_read(
  name: SHStringWithLen,
  code: SHStringWithLen,
  base_path: SHStringWithLen,
) -> SHLAst {
  let name: &str = name.into();
  let code = code.into();
  let base_path: &str = base_path.into();
  let result = read::read(code, name, base_path);

  match result {
    Ok(p) => SHLAst {
      ast: Var::new_ref_counted(p, &AST_TYPE),
      error: std::ptr::null_mut(),
    },
    Err(error) => {
      shlog_error!("{:?}", error);
      let error_message = CString::new(error.message).unwrap();
      let shards_error = SHLError {
        message: error_message.into_raw(),
        line: error.loc.line,
        column: error.loc.column,
      };
      SHLAst {
        ast: Var::default(),
        error: Box::into_raw(Box::new(shards_error)),
      }
    }
  }
}

#[no_mangle]
pub extern "C" fn shards_load_ast(bytes: *mut u8, size: u32) -> SHLAst {
  let bytes = unsafe { from_raw_parts_allow_null(bytes, size as usize) };
  let decoded_bin: Result<Program, _> = flexbuffers::from_slice(bytes);
  match decoded_bin {
    Ok(p) => SHLAst {
      ast: Var::new_ref_counted(p, &AST_TYPE),
      error: std::ptr::null_mut(),
    },
    Err(error) => {
      shlog_error!("{:?}", error);
      let error_message = CString::new(error.to_string()).unwrap();
      let shards_error = SHLError {
        message: error_message.into_raw(),
        line: 0,
        column: 0,
      };
      SHLAst {
        ast: Var::default(),
        error: Box::into_raw(Box::new(shards_error)),
      }
    }
  }
}

#[no_mangle]
pub extern "C" fn shards_save_ast(ast: *mut Program) -> Var {
  let ast = unsafe { &*ast };
  let encoded_bin = flexbuffers::to_vec(&ast).unwrap();
  let v: ClonedVar = encoded_bin.as_slice().into();
  let inner = v.0;
  std::mem::forget(v);
  inner
}

#[no_mangle]
pub extern "C" fn shards_create_env(namespace: SHStringWithLen) -> *mut EvalEnv {
  if namespace.len == 0 {
    Box::into_raw(Box::new(EvalEnv::new(None, None)))
  } else {
    let namespace: &str = namespace.into();
    Box::into_raw(Box::new(EvalEnv::new(Some(namespace.into()), None)))
  }
}

#[no_mangle]
pub extern "C" fn shards_forbid_shard(env: *mut EvalEnv, name: SHStringWithLen) {
  let env = unsafe { &mut *env };
  let name: &str = name.into();
  env.forbidden_funcs.insert(Identifier {
    name: RcStrWrapper::from(name),
    namespaces: Vec::new(),
  });
}

#[no_mangle]
pub extern "C" fn shards_free_env(env: *mut EvalEnv) {
  unsafe {
    drop(Box::from_raw(env));
  }
}

#[no_mangle]
pub extern "C" fn shards_create_sub_env(
  env: *mut EvalEnv,
  namespace: SHStringWithLen,
) -> *mut EvalEnv {
  let env = unsafe { &mut *env };
  if namespace.len == 0 {
    Box::into_raw(Box::new(EvalEnv::new(None, Some(env))))
  } else {
    let namespace: &str = namespace.into();
    Box::into_raw(Box::new(EvalEnv::new(Some(namespace.into()), Some(env))))
  }
}

#[no_mangle]
pub extern "C" fn shards_eval_env(env: *mut EvalEnv, ast: &Var) -> *mut SHLError {
  let ast = unsafe {
    &mut *Var::from_ref_counted_object::<Program>(ast, &AST_TYPE).expect("A valid AST variable.")
  };
  let env = unsafe { &mut *env };
  for stmt in &ast.sequence.statements {
    if let Err(error) = eval::eval_statement(stmt, env, new_cancellation_token()) {
      shlog_error!("{:?}", error);
      let error_message = CString::new(error.message).unwrap();
      let shards_error = SHLError {
        message: error_message.into_raw(),
        line: error.loc.line,
        column: error.loc.column,
      };
      return Box::into_raw(Box::new(shards_error));
    }
  }
  core::ptr::null_mut()
}

/// It will consume the env
#[no_mangle]
pub extern "C" fn shards_transform_env(env: *mut EvalEnv, name: SHStringWithLen) -> SHLWire {
  let name = name.into();
  let mut env = unsafe { Box::from_raw(env) };
  let res = eval::transform_env(&mut env, name);
  match res {
    Ok(wire) => SHLWire {
      wire: Box::into_raw(Box::new(wire)),
      error: std::ptr::null_mut(),
    },
    Err(error) => {
      shlog_error!("{:?}", error);
      let error_message = CString::new(error.message).unwrap();
      let shards_error = SHLError {
        message: error_message.into_raw(),
        line: error.loc.line,
        column: error.loc.column,
      };
      SHLWire {
        wire: std::ptr::null_mut(),
        error: Box::into_raw(Box::new(shards_error)),
      }
    }
  }
}

#[no_mangle]
pub extern "C" fn shards_transform_envs(
  env: *mut *mut EvalEnv,
  len: usize,
  name: SHStringWithLen,
) -> SHLWire {
  let name = name.into();
  let envs = unsafe { std::slice::from_raw_parts_mut(env, len) };
  let mut deref_envs = Vec::with_capacity(len);
  for &env in envs.iter() {
    let env = unsafe { Box::from_raw(env) };
    deref_envs.push(env);
  }
  let res = eval::transform_envs(deref_envs.iter_mut().map(|x| x.as_mut()), name);
  match res {
    Ok(wire) => SHLWire {
      wire: Box::into_raw(Box::new(wire)),
      error: std::ptr::null_mut(),
    },
    Err(error) => {
      shlog_error!("{:?}", error);
      let error_message = CString::new(error.message).unwrap();
      let shards_error = SHLError {
        message: error_message.into_raw(),
        line: error.loc.line,
        column: error.loc.column,
      };
      SHLWire {
        wire: std::ptr::null_mut(),
        error: Box::into_raw(Box::new(shards_error)),
      }
    }
  }
}

#[no_mangle]
pub extern "C" fn shards_eval(ast: &Var, name: SHStringWithLen) -> SHLWire {
  let name = name.into();
  // we just want a reference to the sequence, not ownership
  let ast = unsafe {
    &mut *Var::from_ref_counted_object::<Program>(ast, &AST_TYPE).expect("A valid AST variable.")
  };
  let result = eval::eval(
    &ast.sequence,
    name,
    HashMap::new(),
    new_cancellation_token(),
  );
  match result {
    Ok(wire) => SHLWire {
      wire: Box::into_raw(Box::new(wire)),
      error: std::ptr::null_mut(),
    },
    Err(error) => {
      shlog_error!("{:?}", error);
      let error_message = CString::new(error.message).unwrap();
      let shards_error = SHLError {
        message: error_message.into_raw(),
        line: error.loc.line,
        column: error.loc.column,
      };
      SHLWire {
        wire: std::ptr::null_mut(),
        error: Box::into_raw(Box::new(shards_error)),
      }
    }
  }
}

#[no_mangle]
pub extern "C" fn shards_print_ast(ast: &Var) -> Var {
  let ast = unsafe {
    &mut *Var::from_ref_counted_object::<Program>(ast, &AST_TYPE).expect("A valid AST variable.")
  };
  let s = print::print_ast(&ast.sequence);
  let s = Var::ephemeral_string(&s);
  let mut v = Var::default();
  shards::core::cloneVar(&mut v, &s);
  v
}

#[no_mangle]
pub extern "C" fn shards_free_wire(wire: *mut Wire) {
  unsafe {
    drop(Box::from_raw(wire));
  }
}

#[no_mangle]
pub extern "C" fn shards_free_error(error: *mut SHLError) {
  unsafe {
    drop(CString::from_raw((*error).message));
    drop(Box::from_raw(error));
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_langffi_langffi(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }

  register_shard::<read::ReadShard>();
  register_legacy_shard::<eval::EvalShard>();
  register_enum::<read::AstType>();
  register_shard::<print::ShardsPrintShard>();
}

/// Please note it will consume `from` but not `to`
#[no_mangle]
pub extern "C" fn shards_merge_envs(from: *mut EvalEnv, to: *mut EvalEnv) -> *mut SHLError {
  let from = unsafe { Box::from_raw(from) };
  let to = unsafe { &mut *to };
  if let Err(e) = merge_env(*from, to) {
    shlog_error!("{:?}", e);
    let error_message = CString::new(e.message).unwrap();
    let shards_error = SHLError {
      message: error_message.into_raw(),
      line: e.loc.line,
      column: e.loc.column,
    };
    Box::into_raw(Box::new(shards_error))
  } else {
    std::ptr::null_mut()
  }
}

#[no_mangle]
pub extern "C" fn setup_panic_hook() {
  // Had to put this in this crate otherwise we would have duplicated symbols
  // Set a custom panic hook to break into the debugger.
  #[cfg(debug_assertions)]
  std::panic::set_hook(Box::new(|info| {
    // Print the panic info to standard error.
    eprintln!("Panic occurred: {:?}", info);
    // Trigger a breakpoint.
    #[cfg(unix)]
    unsafe {
      libc::raise(libc::SIGTRAP);
    }
    #[cfg(windows)]
    unsafe {
      windows::Win32::System::Diagnostics::Debug::DebugBreak();
    }
  }));
}

#[no_mangle]
pub extern "C" fn shards_process_args(
  argc: i32,
  argv: *const *const c_char,
  no_cancellation: bool,
) -> i32 {
  process_args(argc, argv, no_cancellation)
}
