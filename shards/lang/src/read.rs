use crate::custom_state::CustomStateContainer;
use crate::{ast::*, RcStrWrapper};
use core::convert::TryInto;
use pest::iterators::Pair;
use pest::Parser;
use shards::shard::Shard;
use shards::types::{
  common_type, AutoSeqVar, AutoTableVar, ClonedVar, Context, ExposedTypes, InstanceData, ParamVar, SeqVar, Type, Types, Var, FRAG_CC, SEQ_OF_STRINGS, SEQ_OF_STRINGS_TYPES, STRINGS_TYPES, STRING_TYPES, STRING_VAR_OR_NONE_SLICE
};
use shards::{
  fourCharacterCode, ref_counted_object_type_impl, shard, shard_impl, shlog_debug, shlog_error,
  shlog_trace,
};
use std::cell::{Ref, RefCell};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct ReadEnv {
  name: RcStrWrapper,
  script_directory: String,
  include_directories: Vec<String>,
  included: RefCell<HashSet<RcStrWrapper>>,
  dependencies: RefCell<Vec<String>>,
  parent: Option<*const ReadEnv>,
}

impl ReadEnv {
  pub fn new_cwd(name: &str) -> Self {
    Self::new(name, ".".to_string(), vec![".".to_string()])
  }
  pub fn new_root(name: &str, script_directory: String) -> Self {
    Self::new(name, script_directory.clone(), vec![script_directory])
  }
  pub fn new(name: &str, script_directory: String, include_directories: Vec<String>) -> Self {
    Self {
      name: name.to_owned().into(),
      script_directory: script_directory,
      include_directories: include_directories,
      included: RefCell::new(HashSet::new()),
      dependencies: RefCell::new(Vec::new()),
      parent: None,
    }
  }

  pub fn resolve_file(&self, name: &str) -> Result<PathBuf, String> {
    let script_dir = Path::new(&self.script_directory);
    let file_path = script_dir.join(name);
    if let Ok(canonical) = std::fs::canonicalize(&file_path) {
      shlog_debug!("Found include {:?}", file_path);
      return Ok(canonical);
    }

    shlog_debug!("Tried include {:?} (not found)", file_path);

    self.resolve_include_path(name)
  }

  fn resolve_include_path(&self, name: &str) -> Result<PathBuf, String> {
    for dir in &self.include_directories {
      let script_dir = Path::new(&dir);
      let file_path = script_dir.join(name);
      let canonical = std::fs::canonicalize(&file_path);
      if let Ok(canonical) = canonical {
        shlog_debug!("Found include {:?}", file_path);
        return Ok(canonical);
      }
      shlog_debug!("Tried include {:?} (not found)", file_path);
    }
    if let Some(parent) = self.parent {
      unsafe {
        return (*parent).resolve_include_path(name);
      }
    }
    return Err(format!("File not found: {}", name).to_string());
  }
}

pub fn get_dependencies<'a>(env: &'a ReadEnv) -> Ref<'_, Vec<String>> {
  env.dependencies.borrow()
}

pub fn get_root_env<'a>(env: &'a ReadEnv) -> &'a ReadEnv {
  let mut node: *const ReadEnv = env;
  unsafe {
    while (*node).parent.is_some() {
      node = (*node).parent.unwrap();
    }
    &*node
  }
}

fn check_included<'a>(name: &'a RcStrWrapper, env: &'a ReadEnv) -> bool {
  if env.included.borrow().contains(name) {
    true
  } else if let Some(parent) = env.parent {
    check_included(name, unsafe { &*parent })
  } else {
    false
  }
}

fn extract_identifier(pair: Pair<Rule>) -> Result<Identifier, ShardsError> {
  // so this can be either a simple identifier or a complex identifier
  // complex identifiers are separated by '/'
  // we want to return a vector of identifiers
  let mut identifiers = Vec::new();
  for pair in pair.into_inner() {
    let rule = pair.as_rule();
    match rule {
      Rule::LowIden => identifiers.push(pair.as_str().to_owned().into()),
      _ => return Err(("Unexpected rule in Identifier.", pair.as_span().start_pos()).into()),
    }
  }
  Ok(Identifier {
    name: identifiers.pop().unwrap(), // qed
    namespaces: identifiers,
    custom_state: CustomStateContainer::new(),
  })
}

fn process_assignment(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Assignment, ShardsError> {
  let pos = pair.as_span().start_pos();
  if pair.as_rule() != Rule::Assignment {
    return Err(
      (
        "Expected an Assignment rule, but found a different rule.",
        pos,
      )
        .into(),
    );
  }

  let mut inner = pair.into_inner();

  let pipeline = if let Some(next) = inner.peek() {
    if next.as_rule() == Rule::Pipeline {
      process_pipeline(
        inner
          .next()
          .ok_or(("Expected a Pipeline in Assignment, but found none.", pos).into())?,
        env,
      )?
    } else {
      Pipeline {
        blocks: vec![Block {
          content: BlockContent::Empty,
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }],
      }
    }
  } else {
    unreachable!("Assignment should have at least one inner rule.")
  };

  let assignment_op = inner
    .next()
    .ok_or(
      (
        "Expected an AssignmentOp in Assignment, but found none.",
        pos,
      )
        .into(),
    )?
    .as_str();

  let iden = inner
    .next()
    .ok_or(("Expected an Identifier in Assignment, but found none.", pos).into())?;

  let identifier = extract_identifier(iden)?;

  match assignment_op {
    "=" => Ok(Assignment::AssignRef(pipeline, identifier)),
    ">=" => Ok(Assignment::AssignSet(pipeline, identifier)),
    ">" => Ok(Assignment::AssignUpd(pipeline, identifier)),
    ">>" => Ok(Assignment::AssignPush(pipeline, identifier)),
    _ => Err(("Unexpected assignment operator.", pos).into()),
  }
}

enum FunctionValue {
  Const(Value),
  Function(Function),
  Program(Program),
}

fn process_function(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<FunctionValue, ShardsError> {
  let pos = pair.as_span().start_pos();

  let mut inner = pair.into_inner();
  let exp = inner
    .next()
    .ok_or(("Expected a Name or Const in Shard", pos).into())?;

  let pos = exp.as_span().start_pos();

  match exp.as_rule() {
    Rule::UppIden => {
      // Definitely a Shard!
      let identifier = Identifier {
        name: exp.as_str().to_owned().into(),
        namespaces: Vec::new(),
        custom_state: CustomStateContainer::new(),
      };
      let next = inner.next();

      let params = match next {
        Some(pair) => {
          if pair.as_rule() == Rule::Params {
            Some(process_params(pair, env)?)
          } else {
            return Err(("Expected Params in Shard", pos).into());
          }
        }
        None => None,
      };

      Ok(FunctionValue::Function(Function {
        name: identifier,
        params,
        custom_state: CustomStateContainer::new(),
      }))
    }
    Rule::VarName => {
      // Many other things...!
      let identifier = extract_identifier(exp)?;
      let next = inner.next();

      let params = match next {
        Some(pair) => {
          if pair.as_rule() == Rule::Params {
            Some(process_params(pair, env)?)
          } else {
            return Err(("Expected Params in Shard", pos).into());
          }
        }
        None => None,
      };

      if identifier.namespaces.is_empty() {
        let name = identifier.name.as_str().to_owned();
        match name.as_str() {
          "include" => {
            let params = params.ok_or(("Expected 2-3 parameters", pos).into())?;
            let n_params = params.len();

            let file_name = if n_params > 0 && params[0].name.is_none() {
              Some(&params[0])
            } else {
              params
                .iter()
                .find(|param| param.name.as_deref() == Some("File"))
            };
            let file_name = file_name.ok_or(("Expected a file name (File:)", pos).into())?;
            let file_name = match &file_name.value {
              Value::String(s) => Ok(s),
              _ => Err(("Expected a string value for File", pos).into()),
            }?;

            let once = if n_params > 1 && params[0].name.is_none() && params[1].name.is_none() {
              Some(&params[1])
            } else {
              params
                .iter()
                .find(|param| param.name.as_deref() == Some("Once"))
            };

            let once = once
              .map(|param| match &param.value {
                Value::Boolean(b) => Ok(*b),
                _ => Err(("Expected a boolean value for Once", pos).into()),
              })
              .unwrap_or(Ok(false))?;

            let fallback = if n_params > 2
              && params[0].name.is_none()
              && params[1].name.is_none()
              && params[2].name.is_none()
            {
              Some(&params[2])
            } else {
              params
                .iter()
                .find(|param| param.name.as_deref() == Some("Fallback"))
            };

            let fallback = fallback
              .map(|param| match &param.value {
                Value::String(s) => Ok(s.as_str()),
                _ => Err(("Expected a string value for Fallback", pos).into()),
              })
              .transpose()?;

            let file_path = env
              .resolve_file(file_name)
              .or_else(|_| {
                fallback.map(|fb| env.resolve_file(fb)).unwrap_or(Err(
                  format!("File {} not found and no fallback provided", file_name).to_string(),
                ))
              })
              .map_err(|x| (x, pos).into())?;

            let file_path_str = file_path
              .to_str()
              .ok_or(("Failed to convert file path to string", pos).into())?
              .to_owned();

            let rc_path = file_path_str.into();

            if once && check_included(&rc_path, env) {
              return Ok(FunctionValue::Const(Value::None(())));
            }

            shlog_trace!("Including file {:?}", file_path);
            {
              // Insert this into the root map so it gets tracked globally
              let root_env = get_root_env(env);
              root_env.dependencies.borrow_mut().push(rc_path.to_string());
              root_env.included.borrow_mut().insert(rc_path);
            }

            // read string from file
            let mut code = std::fs::read_to_string(&file_path)
              .map_err(|e| (format!("Failed to read file {:?}: {}", file_path, e), pos).into())?;
            // add new line at the end of the file to be able to parse it correctly
            code.push('\n');

            let parent = file_path.parent().unwrap_or(Path::new("."));
            let successful_parse = ShardsParser::parse(Rule::Program, &code)
              .map_err(|e| (format!("Failed to parse file {:?}: {}", file_path, e), pos).into())?;
            let mut sub_env: ReadEnv = ReadEnv::new(
              file_path.to_str().unwrap(), // should be qed...
              parent
                .to_str()
                .ok_or(
                  (
                    format!("Failed to convert file path {:?} to string", parent),
                    pos,
                  )
                    .into(),
                )?
                .into(),
              Vec::new(),
            );
            sub_env.parent = Some(env);
            let program = process_program(
              successful_parse.into_iter().next().unwrap(), // should be qed because of success parse
              &mut sub_env,
            )?;

            Ok(FunctionValue::Program(program))
          }
          "env" => {
            // read from environment variable
            let params = params.ok_or(("Expected 1 parameter", pos).into())?;
            let n_params = params.len();

            let name = if n_params > 0 && params[0].name.is_none() {
              Some(&params[0])
            } else {
              params
                .iter()
                .find(|param| param.name.as_deref() == Some("Name"))
            };
            let name = name.ok_or(("Expected an environment variable name", pos).into())?;
            let name = match &name.value {
              Value::String(s) => Ok(s),
              _ => Err(("Expected a string value", pos).into()),
            }?;

            let value = std::env::var(name.as_str()).unwrap_or("".to_string());

            Ok(FunctionValue::Const(Value::String(value.into())))
          }
          "read" => {
            let params = params.ok_or(("Expected 2 parameters", pos).into())?;
            let n_params = params.len();

            let file_name = if n_params > 0 && params[0].name.is_none() {
              Some(&params[0])
            } else {
              params
                .iter()
                .find(|param| param.name.as_deref() == Some("File"))
            };
            let file_name = file_name.ok_or(("Expected a file name (File:)", pos).into())?;
            let file_name = match &file_name.value {
              Value::String(s) => Ok(s),
              _ => Err(("Expected a string value for File", pos).into()),
            }?;

            let as_bytes = if n_params > 1 && params[0].name.is_none() && params[1].name.is_none() {
              Some(&params[1])
            } else {
              params
                .iter()
                .find(|param| param.name.as_deref() == Some("Bytes"))
            };

            let as_bytes = as_bytes
              .map(|param| match &param.value {
                Value::Boolean(b) => Ok(*b),
                _ => Err(("Expected a boolean value for Bytes", pos).into()),
              })
              .unwrap_or(Ok(false))?;

            let file_path = env.resolve_file(file_name).map_err(|x| (x, pos).into())?;

            {
              // Insert this into the root map so it gets tracked globally
              let root_env = get_root_env(env);
              root_env.dependencies.borrow_mut().push(file_path.to_string_lossy().to_string());
            }

            if as_bytes {
              // read bytes from file
              let bytes = std::fs::read(&file_path)
                .map_err(|e| (format!("Failed to read file {:?}: {}", file_path, e), pos).into())?;
              Ok(FunctionValue::Const(Value::Bytes(bytes.into())))
            } else {
              // read string from file
              let string = std::fs::read_to_string(&file_path)
                .map_err(|e| (format!("Failed to read file {:?}: {}", file_path, e), pos).into())?;
              Ok(FunctionValue::Const(Value::String(string.into())))
            }
          }
          _ => Ok(FunctionValue::Function(Function {
            name: identifier,
            params,
            custom_state: CustomStateContainer::new(),
          })),
        }
      } else {
        Ok(FunctionValue::Function(Function {
          name: identifier,
          params,
          custom_state: CustomStateContainer::new(),
        }))
      }
    }
    _ => Err(
      (
        format!("Unexpected rule {:?} in Function.", exp.as_rule()),
        pos,
      )
        .into(),
    ),
  }
}

fn process_take_table(
  pair: Pair<Rule>,
  _env: &mut ReadEnv,
) -> Result<(Identifier, Vec<RcStrWrapper>), ShardsError> {
  let pos = pair.as_span().start_pos();
  // first is the identifier which has to be VarName
  // followed by N Iden which are the keys

  let mut inner = pair.into_inner();
  let identity = inner
    .next()
    .ok_or(("Expected an identifier in TakeTable", pos).into())?;

  let identifier = extract_identifier(identity)?;

  let mut keys = Vec::new();
  for pair in inner {
    let pos = pair.as_span().start_pos();
    match pair.as_rule() {
      Rule::Iden => keys.push(pair.as_str().to_owned().into()),
      _ => return Err(("Expected an identifier in TakeTable", pos).into()),
    }
  }

  // wrap the shards into an Expr Sequence
  Ok((identifier, keys))
}

fn process_take_seq(
  pair: Pair<Rule>,
  _env: &mut ReadEnv,
) -> Result<(Identifier, Vec<u32>), ShardsError> {
  let pos = pair.as_span().start_pos();
  // first is the identifier which has to be VarName
  // followed by N Integer which are the indices

  let mut inner = pair.into_inner();
  let identity = inner
    .next()
    .ok_or(("Expected an identifier in TakeSeq", pos).into())?;

  let identifier = extract_identifier(identity)?;

  let mut indices = Vec::new();
  for pair in inner {
    let pos = pair.as_span().start_pos();
    match pair.as_rule() {
      Rule::Integer => {
        let value = pair
          .as_str()
          .parse()
          .map_err(|_| ("Failed to parse Integer", pos).into())?;
        indices.push(value);
      }
      _ => return Err(("Expected an integer in TakeSeq", pos).into()),
    }
  }

  Ok((identifier, indices))
}

fn process_pipeline(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Pipeline, ShardsError> {
  let pos = pair.as_span().start_pos();
  if pair.as_rule() != Rule::Pipeline {
    return Err(("Expected a Pipeline rule, but found a different rule.", pos).into());
  }

  let mut blocks = Vec::new();

  for pair in pair.into_inner() {
    let pos = pair.as_span().start_pos();
    let rule = pair.as_rule();
    match rule {
      Rule::EvalExpr => blocks.push(Block {
        content: BlockContent::EvalExpr(process_sequence(
          pair
            .into_inner()
            .next()
            .ok_or(("Expected an eval time expression, but found none.", pos).into())?,
          env,
        )?),
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      Rule::Expr => blocks.push(Block {
        content: BlockContent::Expr(process_sequence(
          pair
            .into_inner()
            .next()
            .ok_or(("Expected an expression, but found none.", pos).into())?,
          env,
        )?),
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      Rule::Shard => match process_function(pair, env)? {
        FunctionValue::Const(value) => blocks.push(Block {
          content: BlockContent::Const(value),
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }),
        FunctionValue::Function(func) => blocks.push(Block {
          content: BlockContent::Shard(func),
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }),
        FunctionValue::Program(program) => blocks.push(Block {
          content: BlockContent::Program(program),
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }),
      },
      Rule::Func => match process_function(pair, env)? {
        FunctionValue::Const(value) => blocks.push(Block {
          content: BlockContent::Const(value),
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }),
        FunctionValue::Function(func) => blocks.push(Block {
          content: BlockContent::Func(func),
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }),
        FunctionValue::Program(program) => blocks.push(Block {
          content: BlockContent::Program(program),
          line_info: Some(pos.into()),
          custom_state: CustomStateContainer::new(),
        }),
      },
      Rule::TakeTable => blocks.push(Block {
        content: {
          let pair = process_take_table(pair, env)?;
          BlockContent::TakeTable(pair.0, pair.1)
        },
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      Rule::TakeSeq => blocks.push(Block {
        content: {
          let pair = process_take_seq(pair, env)?;
          BlockContent::TakeSeq(pair.0, pair.1)
        },
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      Rule::ConstValue => blocks.push(Block {
        // this is an indirection, process_value will handle the case of a ConstValue
        content: BlockContent::Const(process_value(pair, env)?),
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      Rule::Enum => blocks.push(Block {
        content: BlockContent::Const(process_value(pair, env)?),
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      Rule::Shards => blocks.push(Block {
        content: BlockContent::Shards(process_sequence(
          pair
            .into_inner()
            .next()
            .ok_or(("Expected an expression, but found none.", pos).into())?,
          env,
        )?),
        line_info: Some(pos.into()),
        custom_state: CustomStateContainer::new(),
      }),
      _ => return Err((format!("Unexpected rule ({:?}) in Pipeline.", rule), pos).into()),
    }
  }
  Ok(Pipeline { blocks })
}

fn process_statement(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Statement, ShardsError> {
  let pos = pair.as_span().start_pos();
  match pair.as_rule() {
    Rule::Assignment => process_assignment(pair, env).map(Statement::Assignment),
    Rule::Pipeline => process_pipeline(pair, env).map(Statement::Pipeline),
    _ => Err(("Expected an Assignment or a Pipeline", pos).into()),
  }
}

pub(crate) fn process_sequence(
  pair: Pair<Rule>,
  env: &mut ReadEnv,
) -> Result<Sequence, ShardsError> {
  let statements = pair
    .into_inner()
    .map(|x| process_statement(x, env))
    .collect::<Result<Vec<_>, _>>()?;
  Ok(Sequence {
    statements,
    custom_state: CustomStateContainer::new(),
  })
}

pub fn process_program(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Program, ShardsError> {
  let pos = pair.as_span().start_pos();
  if pair.as_rule() != Rule::Program {
    return Err(("Expected a Program rule, but found a different rule.", pos).into());
  }
  let pair = pair.into_inner().next().unwrap(); // parsed qed
  Ok(Program {
    sequence: process_sequence(pair, env)?,
    metadata: Metadata {
      name: env.name.clone(),
      debug_info: RefCell::new(DebugInfo::default()),
    },
  })
}

fn process_value(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Value, ShardsError> {
  let pos = pair.as_span().start_pos();
  match pair.as_rule() {
    Rule::ConstValue => {
      // unwrap the inner rule
      let pair = pair.into_inner().next().unwrap(); // parsed qed
      process_value(pair, env)
    }
    Rule::None => Ok(Value::None(())),
    Rule::Boolean => {
      // check if string content is true or false
      let bool_str = pair.as_str();
      if bool_str == "true" {
        Ok(Value::Boolean(true))
      } else if bool_str == "false" {
        Ok(Value::Boolean(false))
      } else {
        Err(("Expected a boolean value", pos).into())
      }
    }
    Rule::VarName => Ok(Value::Identifier(extract_identifier(pair)?)),
    Rule::Enum => {
      let text = pair.as_str();
      let splits: Vec<_> = text.split("::").collect();
      if splits.len() != 2 {
        return Err(("Expected an enum value", pos).into());
      }
      let enum_name = splits[0].to_owned();
      let variant_name = splits[1].to_owned();
      Ok(Value::Enum(enum_name.into(), variant_name.into()))
    }
    Rule::Number => process_number(
      pair
        .into_inner()
        .next()
        .ok_or(("Expected a Number value", pos).into())?,
      env,
    )
    .map(Value::Number),
    Rule::String => {
      let inner = pair.into_inner().next().unwrap(); // parsed qed
      match inner.as_rule() {
        Rule::SimpleString => Ok(Value::String({
          let full_str = inner.as_str();
          // remove quotes AND
          // with this case we need to transform escaped characters
          // so we need to iterate over the string
          let mut chars: std::str::Chars = full_str[1..full_str.len() - 1].chars();
          let mut new_str = String::new();
          while let Some(c) = chars.next() {
            if c == '\\' {
              // we need to check the next character
              let c = chars
                .next()
                .ok_or(("Unexpected end of string", pos).into())?;
              match c {
                'n' => new_str.push('\n'),
                'r' => new_str.push('\r'),
                't' => new_str.push('\t'),
                '\\' => new_str.push('\\'),
                '"' => new_str.push('"'),
                '\'' => new_str.push('\''),
                _ => return Err((format!("Unexpected escaped character {:?}", c), pos).into()),
              }
            } else {
              new_str.push(c);
            }
          }
          new_str.into()
        })),
        Rule::ComplexString => Ok(Value::String({
          let full_str = inner.as_str().to_owned();
          // remove triple quotes
          full_str[3..full_str.len() - 3].to_owned().into()
        })),
        _ => unreachable!(),
      }
    }
    Rule::Seq => {
      let values = pair
        .into_inner()
        .map(|value| {
          let pos = value.as_span().start_pos();
          process_value(
            value
              .into_inner()
              .next()
              .ok_or(("Expected a Value in the sequence", pos).into())?,
            env,
          )
        })
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Value::Seq(values))
    }
    Rule::Table => {
      let pairs = pair
        .into_inner()
        .map(|pair| {
          assert_eq!(pair.as_rule(), Rule::TableEntry);

          let mut inner = pair.into_inner();

          let key = inner.next().unwrap(); // should not fail
          assert_eq!(key.as_rule(), Rule::TableKey);
          let pos = key.as_span().start_pos();
          let key = key
            .into_inner()
            .next()
            .ok_or(("Expected a Table key", pos).into())?;
          let key = match key.as_rule() {
            Rule::None => Value::None(()),
            Rule::Iden => Value::String(key.as_str().to_owned().into()),
            Rule::VarName => Value::Identifier(extract_identifier(key)?),
            Rule::ConstValue => process_value(
              key.into_inner().next().unwrap(), // parsed qed
              env,
            )?,
            _ => {
              eprintln!("Unexpected rule in TableKey: {:?}", key.as_rule());
              unreachable!()
            }
          };

          let value = inner
            .next()
            .ok_or(("Expected a value in TableEntry", pos).into())?;
          let pos = value.as_span().start_pos();
          let value = process_value(
            value
              .into_inner()
              .next()
              .ok_or(("Expected a value in TableEntry", pos).into())?,
            env,
          )?;
          Ok((key, value))
        })
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Value::Table(pairs))
    }
    Rule::Shards => process_sequence(
      pair
        .into_inner()
        .next()
        .ok_or(("Expected a Sequence in Value", pos).into())?,
      env,
    )
    .map(Value::Shards),
    Rule::Shard => match process_function(pair, env)? {
      FunctionValue::Function(func) => Ok(Value::Shard(func)),
      _ => Err(("Invalid Shard in value", pos).into()),
    },
    Rule::EvalExpr => process_sequence(
      pair
        .into_inner()
        .next()
        .ok_or(("Expected a Sequence in Value", pos).into())?,
      env,
    )
    .map(Value::EvalExpr),
    Rule::Expr => process_sequence(
      pair
        .into_inner()
        .next()
        .ok_or(("Expected a Sequence in Value", pos).into())?,
      env,
    )
    .map(Value::Expr),
    Rule::TakeTable => {
      let pair = process_take_table(pair, env)?;
      Ok(Value::TakeTable(pair.0, pair.1))
    }
    Rule::TakeSeq => {
      let pair = process_take_seq(pair, env)?;
      Ok(Value::TakeSeq(pair.0, pair.1))
    }
    Rule::Func => match process_function(pair, env)? {
      FunctionValue::Const(val) => return Ok(val),
      FunctionValue::Function(func) => Ok(Value::Func(func)),
      _ => Err(("Function cannot be used as value", pos).into()),
    },
    _ => Err(
      (
        format!("Unexpected rule ({:?}) in Value", pair.as_rule()),
        pos,
      )
        .into(),
    ),
  }
}

fn process_number(pair: Pair<Rule>, _env: &mut ReadEnv) -> Result<Number, ShardsError> {
  let pos = pair.as_span().start_pos();
  match pair.as_rule() {
    Rule::Integer => Ok(Number::Integer(
      pair
        .as_str()
        .parse()
        .map_err(|_| ("Failed to parse Integer", pos).into())?,
    )),
    Rule::Float => Ok(Number::Float(
      pair
        .as_str()
        .parse()
        .map_err(|_| ("Failed to parse Float", pos).into())?,
    )),
    Rule::Hexadecimal => Ok(Number::Hexadecimal(pair.as_str().to_owned().into())),
    _ => Err(("Unexpected rule in Number", pos).into()),
  }
}

fn process_param(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Param, ShardsError> {
  let pos = pair.as_span().start_pos();
  if pair.as_rule() != Rule::Param {
    return Err(("Expected a Param rule", pos).into());
  }

  let mut inner = pair.into_inner();
  let first = inner
    .next()
    .ok_or(("Expected a ParamName or Value in Param", pos).into())?;
  let pos = first.as_span().start_pos();
  let (param_name, param_value) = if first.as_rule() == Rule::ParamName {
    let name = first.as_str().to_owned();
    let name = name[0..name.len() - 1].to_owned().into();
    let value = process_value(
      inner
        .next()
        .ok_or(("Expected a Value in Param", pos).into())?
        .into_inner()
        .next()
        .ok_or(("Expected a Value in Param", pos).into())?,
      env,
    )?;
    (Some(name), value)
  } else {
    (
      None,
      process_value(
        first
          .into_inner()
          .next()
          .ok_or(("Expected a Value in Param", pos).into())?,
        env,
      )?,
    )
  };

  Ok(Param {
    name: param_name,
    value: param_value,
    custom_state: CustomStateContainer::new(),
    is_default: None,
  })
}

fn process_params(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Vec<Param>, ShardsError> {
  pair.into_inner().map(|x| process_param(x, env)).collect()
}

pub fn parse(code: &str) -> Result<pest::iterators::Pairs<'_, Rule>, ShardsError> {
  profiling::scope!("parse");

  ShardsParser::parse(Rule::Program, code).map_err(|e| {
    (
      format!("Failed to parse file: {}", e),
      LineInfo { line: 0, column: 0 },
    )
      .into()
  })
}

pub fn read_with_env(code: &str, env: &mut ReadEnv) -> Result<Program, ShardsError> {
  profiling::scope!("read_with_env");

  let successful_parse: pest::iterators::Pairs<'_, Rule> = {
    ShardsParser::parse(Rule::Program, code).map_err(|e| {
      (
        format!("Failed to parse file {:?}: {}", env.script_directory, e),
        LineInfo { line: 0, column: 0 },
      )
        .into()
    })?
  };
  process_program(
    successful_parse.into_iter().next().unwrap(), // parsed qed
    env,
  )
}

pub fn read(
  code: &str,
  name: &str,
  path: String,
  include_dirs: Vec<String>,
) -> Result<Program, ShardsError> {
  let mut env = ReadEnv::new(name, path, include_dirs);
  read_with_env(&code, &mut env)
}

use lazy_static::lazy_static;

lazy_static! {
  pub static ref AST_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"ASTa")); // last letter used as version
  pub static ref AST_TYPE_VEC: Vec<Type> = vec![*AST_TYPE];
  pub static ref AST_VAR_TYPE: Type = Type::context_variable(&AST_TYPE_VEC);
  pub static ref READ_OUTPUT_TYPES: Vec<Type> = vec![common_type::string, common_type::bytes, *AST_TYPE];
}

#[derive(shards::shards_enum)]
#[enum_info(
  b"ASTt",
  "AstType",
  "Variants of AST representation of a Shards program."
)]
pub enum AstType {
  #[enum_value("Binary AST as Bytes type")]
  Bytes = 0x0,
  #[enum_value("JSON String type AST")]
  Json = 0x1,
  #[enum_value("Live Object AST to be used within a live environment")]
  Object = 0x2,
}

lazy_static! {
  pub static ref STRINGS_VAR: Type = Type::context_variable(&STRING_TYPES);
}

#[derive(shard)]
#[shard_info(
  "Shards.Read",
  "Reads the textual representation of a Shards program and outputs the binary or json AST representation.",
)]
pub struct ReadShard {
  output: ClonedVar,
  #[shard_param(
    "OutputType",
    "Determines the type of AST to be outputted.",
    ASTTYPE_TYPES
  )]
  output_type: ClonedVar,
  #[shard_param(
    "BasePath",
    "The base path used when interpreting file references.",
    STRING_VAR_OR_NONE_SLICE
  )]
  base_path: ParamVar,
  #[shard_param(
    "Include",
    "The list of include paths.",
    [common_type::strings, *STRINGS_VAR]
  )]
  include: ParamVar,
  #[shard_required]
  required_variables: ExposedTypes,
}

impl Default for ReadShard {
  fn default() -> Self {
    Self {
      output: ClonedVar::default(),
      output_type: ClonedVar::from(AstType::Bytes),
      base_path: ParamVar::new(Var::ephemeral_string(".")),
      include: ParamVar::new(SeqVar::leaking_new().0),
      required_variables: ExposedTypes::default(),
    }
  }
}

ref_counted_object_type_impl!(Program);

#[shard_impl]
impl Shard for ReadShard {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &READ_OUTPUT_TYPES
  }

  fn warmup(&mut self, _context: &Context) -> Result<(), &str> {
    self.warmup_helper(_context)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn compose(&mut self, _data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(_data)?;

    match self.output_type.0.as_ref().try_into() {
      Ok(AstType::Bytes) => Ok(common_type::bytes),
      Ok(AstType::Json) => Ok(common_type::string),
      Ok(AstType::Object) => Ok(*AST_TYPE),
      Err(_) => {
        shlog_error!("Invalid output type for ReadShard");
        Err("Invalid output type for ReadShard")
      }
    }
  }

  fn activate(&mut self, _: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let code: &str = input.try_into()?;

    let output_type = self.output_type.0.as_ref().try_into() as Result<AstType, _>;

    let parsed = ShardsParser::parse(Rule::Program, code).map_err(|e| {
      shlog_error!("Failed to parse shards code: {}", e);
      "Failed to parse Shards code"
    })?;

    let bp_var = self.base_path.get();
    let base_path = if bp_var.is_none() {
      "."
    } else {
      bp_var.try_into()?
    };

    let mut includes: Vec<String> = Vec::new();
    for inc in self.include.get().as_seq()? {
      includes.push((&inc).try_into()?);
    }

    let prog = process_program(
      parsed.into_iter().next().unwrap(), // parsed qed
      &mut ReadEnv::new("", base_path.to_string(), includes),
    )
    .map_err(|e| {
      shlog_error!("Failed to process shards code: {:?}", e);
      "Failed to tokenize Shards code"
    })?;

    match output_type {
      Ok(AstType::Bytes) => {
        // Serialize using flexbuffers
        let encoded_bin: Vec<u8> = flexbuffers::to_vec(&prog).map_err(|e| {
          shlog_error!("Failed to serialize shards code: {}", e);
          "Failed to serialize Shards code"
        })?;

        self.output = encoded_bin.as_slice().into();
      }
      Ok(AstType::Json) => {
        // Serialize using json
        let encoded_json = serde_json::to_string(&prog).map_err(|e| {
          shlog_error!("Failed to serialize shards code: {}", e);
          "Failed to serialize Shards code"
        })?;

        let s = Var::ephemeral_string(encoded_json.as_str());
        self.output = s.into();
      }
      Ok(AstType::Object) => {
        self.output = Var::new_ref_counted(prog, &AST_TYPE).into();
      }
      Err(_) => {
        shlog_error!("Invalid output type for ReadShard");
        return Err("Invalid output type for ReadShard");
      }
    }

    Ok(Some(self.output.0))
  }
}

// Shards.Errors
// A shard to fetch all errors from a live AstType::Object input

#[derive(shards::shard)]
#[shard_info("Shards.Errors", "Fetches all errors from a live AST Object input")]
pub struct ShardsErrorsShard {
  #[shard_required]
  required: ExposedTypes,

  output: AutoSeqVar,
}

impl Default for ShardsErrorsShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      output: AutoSeqVar::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for ShardsErrorsShard {
  fn input_types(&mut self) -> &Types {
    &AST_TYPE_VEC
  }

  fn output_types(&mut self) -> &Types {
    &STRINGS_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let object = unsafe { &*Var::from_ref_counted_object::<Program>(input, &AST_TYPE)? };
    self.output.0.clear();
    self.process_sequence(&object.sequence);
    Ok(Some(self.output.0 .0))
  }
}

impl ShardsErrorsShard {
  fn process_sequence(&mut self, seq: &Sequence) {
    for child in seq.statements.iter() {
      match child {
        Statement::Assignment(x) => match x {
          Assignment::AssignRef(p, _)
          | Assignment::AssignSet(p, _)
          | Assignment::AssignUpd(p, _)
          | Assignment::AssignPush(p, _) => self.process_pipeline(p),
        },
        Statement::Pipeline(p) => {
          self.process_pipeline(p);
        }
      }
    }
    seq.custom_state.with::<ShardsError, _, _>(|e| {
      let mut table = AutoTableVar::new();
      table
        .0
        .insert_fast_static("message", &Var::ephemeral_string(e.message.as_str()));
      table.0.insert_fast_static("line", &e.loc.line.into());
      table.0.insert_fast_static("column", &e.loc.column.into());
      self.output.0.emplace(table.to_cloned());
    });
  }

  fn process_pipeline(&mut self, pipeline: &Pipeline) {
    for block in pipeline.blocks.iter() {
      match &block.content {
        BlockContent::Shard(f) | BlockContent::Func(f) => {
          if let Some(params) = &f.params {
            for param in params {
              self.process_value(&param.value);
            }
          }
          f.custom_state.with::<ShardsError, _, _>(|e| {
            let mut table = AutoTableVar::new();
            table
              .0
              .insert_fast_static("message", &Var::ephemeral_string(e.message.as_str()));
            table.0.insert_fast_static("line", &e.loc.line.into());
            table.0.insert_fast_static("column", &e.loc.column.into());
            self.output.0.emplace(table.to_cloned());
          });
        }
        BlockContent::Shards(s) | BlockContent::EvalExpr(s) | BlockContent::Expr(s) => {
          self.process_sequence(s)
        }
        BlockContent::Const(v) => self.process_value(v),
        _ => (),
      }
    }
  }

  fn process_value(&mut self, value: &Value) {
    match value {
      Value::None(_) => {}
      Value::Identifier(identifier) => {
        identifier
          .custom_state
          .with::<ShardsError, _, _>(|e: &ShardsError| {
            let mut table = AutoTableVar::new();
            table
              .0
              .insert_fast_static("message", &Var::ephemeral_string(e.message.as_str()));
            table.0.insert_fast_static("line", &e.loc.line.into());
            table.0.insert_fast_static("column", &e.loc.column.into());
            self.output.0.emplace(table.to_cloned());
          });
      }
      Value::Boolean(_) => {}
      Value::Enum(_, _) => {}
      Value::Number(_) => {}
      Value::String(_) => {}
      Value::Bytes(_) => {}
      Value::Int2(_) | Value::Int3(_) | Value::Int4(_) | Value::Int8(_) | Value::Int16(_) => {}
      Value::Float2(_) | Value::Float3(_) | Value::Float4(_) => {}
      Value::Seq(seq) => {
        for item in seq {
          self.process_value(item);
        }
      }
      Value::Table(table) => {
        for (key, value) in table {
          self.process_value(key);
          self.process_value(value);
        }
      }
      Value::Shard(function) | Value::Func(function) => {
        function.custom_state.with::<ShardsError, _, _>(|e| {
          let mut table = AutoTableVar::new();
          table
            .0
            .insert_fast_static("message", &Var::ephemeral_string(e.message.as_str()));
          table.0.insert_fast_static("line", &e.loc.line.into());
          table.0.insert_fast_static("column", &e.loc.column.into());
          self.output.0.emplace(table.to_cloned());
        });
      }
      Value::Shards(sequence) | Value::EvalExpr(sequence) | Value::Expr(sequence) => {
        self.process_sequence(sequence);
      }
      Value::TakeTable(identifier, _) | Value::TakeSeq(identifier, _) => {
        identifier.custom_state.with::<ShardsError, _, _>(|e| {
          let mut table = AutoTableVar::new();
          table
            .0
            .insert_fast_static("message", &Var::ephemeral_string(e.message.as_str()));
          table.0.insert_fast_static("line", &e.loc.line.into());
          table.0.insert_fast_static("column", &e.loc.column.into());
          self.output.0.emplace(table.to_cloned());
        });
      }
    }
  }
}

#[test]
fn test_parsing1() {
  // use std::num::NonZeroUsize;
  // pest::set_call_limit(NonZeroUsize::new(25000));
  // let code = include_str!("nested.shs");
  let code = include_str!("sample1.shs");
  let successful_parse = ShardsParser::parse(Rule::Program, code).unwrap();
  let mut env = ReadEnv::new_cwd("");
  let seq = process_program(successful_parse.into_iter().next().unwrap(), &mut env).unwrap();
  let seq = seq.sequence;

  // Serialize using flexbuffers
  let encoded_bin: Vec<u8> = flexbuffers::to_vec(&seq).unwrap();

  // Deserialize using flexbuffers
  let decoded_bin: Sequence = flexbuffers::from_slice(&encoded_bin).unwrap();

  // Serialize using json
  let encoded_json = serde_json::to_string(&seq).unwrap();
  println!("Json Serialized = {}", encoded_json);

  let encoded_json2 = serde_json::to_string(&decoded_bin).unwrap();
  assert_eq!(encoded_json, encoded_json2);

  // Deserialize using json
  let decoded_json: Sequence = serde_json::from_str(&encoded_json).unwrap();

  let encoded_bin2: Vec<u8> = flexbuffers::to_vec(&decoded_json).unwrap();
  assert_eq!(encoded_bin, encoded_bin2);
}

#[test]
fn test_parsing2() {
  let code = include_str!("explained.shs");
  let successful_parse = ShardsParser::parse(Rule::Program, code).unwrap();
  let mut env = ReadEnv::new_cwd("");
  let seq = process_program(successful_parse.into_iter().next().unwrap(), &mut env).unwrap();
  let seq = seq.sequence;

  // Serialize using flexbuffers
  let encoded_bin: Vec<u8> = flexbuffers::to_vec(&seq).unwrap();

  // Deserialize using flexbuffers
  let decoded_bin: Sequence = flexbuffers::from_slice(&encoded_bin).unwrap();

  // Serialize using json
  let encoded_json = serde_json::to_string(&seq).unwrap();
  println!("Json Serialized = {}", encoded_json);

  let encoded_json2 = serde_json::to_string(&decoded_bin).unwrap();
  assert_eq!(encoded_json, encoded_json2);

  // Deserialize using json
  let decoded_json: Sequence = serde_json::from_str(&encoded_json).unwrap();

  let encoded_bin2: Vec<u8> = flexbuffers::to_vec(&decoded_json).unwrap();
  assert_eq!(encoded_bin, encoded_bin2);
}
