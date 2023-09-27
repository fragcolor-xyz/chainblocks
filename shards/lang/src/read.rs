use crate::{ast::*, RcStrWrapper};
use core::convert::TryInto;
use pest::iterators::Pair;
use pest::Parser;
use shards::shard;
use shards::shard::Shard;
use shards::types::{
  common_type, ClonedVar, Context, ExposedTypes, InstanceData, ParamVar, Parameters, Type, Types,
  Var, BOOL_TYPES_SLICE, STRING_TYPES, STRING_VAR_OR_NONE_SLICE,
};
use shards::{cstr, shard_impl, shccstr, shlog, shlog_error};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub struct ReadEnv {
  name: RcStrWrapper,
  root_directory: String,
  script_directory: String,
  included: HashSet<RcStrWrapper>,
  dependencies: Vec<String>,
  parent: Option<*const ReadEnv>,
}

impl ReadEnv {
  pub(crate) fn new(name: &str, root_directory: &str, script_directory: &str) -> Self {
    Self {
      name: name.into(),
      root_directory: root_directory.to_string(),
      script_directory: script_directory.to_string(),
      included: HashSet::new(),
      dependencies: Vec::new(),
      parent: None,
    }
  }

  pub fn resolve_file(&self, name: &str) -> Result<PathBuf, String> {
    let script_dir = Path::new(&self.script_directory);
    let file_path = script_dir.join(name);
    shlog!("Trying include {:?}", file_path);
    let mut file_path_r = std::fs::canonicalize(&file_path);

    // Try from root
    if file_path_r.is_err() {
      let root_dir = Path::new(&self.root_directory);
      let file_path = root_dir.join(name);
      shlog!("Trying include {:?}", file_path);
      file_path_r = std::fs::canonicalize(&file_path);
    }

    Ok(
      file_path_r
        .map_err(|e| return format!("Failed to canonicalize file {:?}: {}", name, e).to_string())?,
    )
  }
}

pub fn get_dependencies<'a>(env: &'a ReadEnv) -> core::slice::Iter<'a, String> {
  env.dependencies.iter()
}

fn check_included<'a>(name: &'a RcStrWrapper, env: &'a ReadEnv) -> bool {
  if env.included.contains(name) {
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
      Rule::LowIden => identifiers.push(pair.as_str().into()),
      _ => return Err(("Unexpected rule in Identifier.", pair.as_span().start_pos()).into()),
    }
  }
  Ok(Identifier {
    name: identifiers.pop().unwrap(), // qed
    namespaces: identifiers,
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

fn process_vector(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Value, ShardsError> {
  assert_eq!(pair.as_rule(), Rule::Vector);
  let pos = pair.as_span().start_pos();
  let inner = pair.into_inner();
  let mut values = Vec::new();
  for pair in inner {
    let pos = pair.as_span().start_pos();
    match pair.as_rule() {
      Rule::Number => values.push(process_number(
        pair
          .into_inner()
          .next()
          .ok_or(("Expected a number", pos).into())?,
        env,
      )?),
      _ => return Err(("Unexpected rule in Vector.", pos).into()),
    }
  }
  // now check that all the values are the same type
  // and if so, return a Const with a vector value
  // now, vectors are always 2+ values, so we can safely check first
  let first = values.first().unwrap(); // qed
  let is_int = match first {
    Number::Float(_) => false,
    _ => true,
  };
  if is_int {
    match values.len() {
      2 => {
        let mut vec = Vec::new();
        for value in values {
          match value {
            Number::Integer(i) => vec.push(i),
            _ => unreachable!(),
          }
        }
        Ok(Value::Int2(vec.try_into().unwrap())) // qed
      }
      3 => {
        let mut vec: Vec<i32> = Vec::new();
        for value in values {
          match value {
            Number::Integer(i) => vec.push(i.try_into().map_err(|_| {
              (
                "Expected a signed integer that fits in 32 bits, but found one that doesn't.",
                pos,
              )
                .into()
            })?),
            _ => unreachable!(),
          }
        }
        Ok(Value::Int3(vec.try_into().unwrap())) // qed
      }
      4 => {
        let mut vec: Vec<i32> = Vec::new();
        for value in values {
          match value {
            Number::Integer(i) => vec.push(i.try_into().map_err(|_| {
              (
                "Expected a signed integer that fits in 32 bits, but found one that doesn't.",
                pos,
              )
                .into()
            })?),
            _ => unreachable!(),
          }
        }
        Ok(Value::Int4(vec.try_into().unwrap())) // qed
      }
      8 => {
        let mut vec: Vec<i16> = Vec::new();
        for value in values {
          match value {
            Number::Integer(i) => vec.push(i.try_into().map_err(|_| {
              (
                "Expected a signed integer that fits in 16 bits, but found one that doesn't.",
                pos,
              )
                .into()
            })?),
            _ => unreachable!(),
          }
        }
        Ok(Value::Int8(vec.try_into().unwrap())) // qed
      }
      16 => {
        let mut vec: Vec<i8> = Vec::new();
        for value in values {
          match value {
            Number::Integer(i) => vec.push(i.try_into().map_err(|_| {
              (
                "Expected a signed integer that fits in 8 bits, but found one that doesn't.",
                pos,
              )
                .into()
            })?),
            _ => unreachable!(),
          }
        }
        Ok(Value::Int16(vec.try_into().unwrap())) // qed
      }
      _ => Err(("Expected an int vector of 2, 3, 4, 8, or 16 numbers.", pos).into()),
    }
  } else {
    match values.len() {
      2 => {
        let mut vec = Vec::new();
        for value in values {
          match value {
            Number::Float(f) => vec.push(f),
            _ => unreachable!(),
          }
        }
        Ok(Value::Float2(vec.try_into().unwrap())) // qed
      }
      3 => {
        let mut vec: Vec<f32> = Vec::new();
        for value in values {
          match value {
            Number::Float(f) => vec.push(f as f32),
            _ => unreachable!(),
          }
        }
        Ok(Value::Float3(vec.try_into().unwrap())) // qed
      }
      4 => {
        let mut vec: Vec<f32> = Vec::new();
        for value in values {
          match value {
            Number::Float(f) => vec.push(f as f32),
            _ => unreachable!(),
          }
        }
        Ok(Value::Float4(vec.try_into().unwrap())) // qed
      }
      _ => Err(("Expected a float vector of 2, 3, 4 numbers.", pos).into()),
    }
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
        name: exp.as_str().into(),
        namespaces: Vec::new(),
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
        let name = identifier.name.as_str();
        match name {
          "include" => {
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

            let file_path = env.resolve_file(file_name).map_err(|x| (x, pos).into())?;
            let file_path_str = file_path
              .to_str()
              .ok_or(("Failed to convert file path to string", pos).into())?;

            let rc_path = file_path_str.into();

            if once && check_included(&rc_path, env) {
              return Ok(FunctionValue::Const(Value::None));
            }

            shlog!("Including file {:?}", file_path);

            env.dependencies.push(rc_path.to_string());
            env.included.insert(rc_path);

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
              &env.root_directory,
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
            );
            sub_env.parent = Some(env);
            let program = process_program(
              successful_parse.into_iter().next().unwrap(), // should be qed because of success parse
              &mut sub_env,
            )?;

            Ok(FunctionValue::Program(program))
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

            env
              .dependencies
              .push(file_path.to_string_lossy().to_string());

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
          })),
        }
      } else {
        Ok(FunctionValue::Function(Function {
          name: identifier,
          params,
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
      Rule::Iden => keys.push(pair.as_str().into()),
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
      Rule::Vector => {
        // in this case we want a Const with a vector value
        let value = process_vector(pair, env)?;
        blocks.push(Block {
          content: BlockContent::Const(value),
          line_info: Some(pos.into()),
        });
      }
      Rule::EvalExpr => blocks.push(Block {
        content: BlockContent::EvalExpr(process_sequence(
          pair
            .into_inner()
            .next()
            .ok_or(("Expected an eval time expression, but found none.", pos).into())?,
          env,
        )?),
        line_info: Some(pos.into()),
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
      }),
      Rule::Shard => match process_function(pair, env)? {
        FunctionValue::Const(value) => blocks.push(Block {
          content: BlockContent::Const(value),
          line_info: Some(pos.into()),
        }),
        FunctionValue::Function(func) => blocks.push(Block {
          content: BlockContent::Shard(func),
          line_info: Some(pos.into()),
        }),
        FunctionValue::Program(program) => blocks.push(Block {
          content: BlockContent::Program(program),
          line_info: Some(pos.into()),
        }),
      },
      Rule::Func => match process_function(pair, env)? {
        FunctionValue::Const(value) => blocks.push(Block {
          content: BlockContent::Const(value),
          line_info: Some(pos.into()),
        }),
        FunctionValue::Function(func) => blocks.push(Block {
          content: BlockContent::Func(func),
          line_info: Some(pos.into()),
        }),
        FunctionValue::Program(program) => blocks.push(Block {
          content: BlockContent::Program(program),
          line_info: Some(pos.into()),
        }),
      },
      Rule::TakeTable => blocks.push(Block {
        content: {
          let pair = process_take_table(pair, env)?;
          BlockContent::TakeTable(pair.0, pair.1)
        },
        line_info: Some(pos.into()),
      }),
      Rule::TakeSeq => blocks.push(Block {
        content: {
          let pair = process_take_seq(pair, env)?;
          BlockContent::TakeSeq(pair.0, pair.1)
        },
        line_info: Some(pos.into()),
      }),
      Rule::ConstValue => blocks.push(Block {
        // this is an indirection, process_value will handle the case of a ConstValue
        content: BlockContent::Const(process_value(pair, env)?),
        line_info: Some(pos.into()),
      }),
      Rule::Enum => blocks.push(Block {
        content: BlockContent::Const(process_value(pair, env)?),
        line_info: Some(pos.into()),
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
  Ok(Sequence { statements })
}

pub(crate) fn process_program(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Program, ShardsError> {
  let pos = pair.as_span().start_pos();
  if pair.as_rule() != Rule::Program {
    return Err(("Expected a Program rule, but found a different rule.", pos).into());
  }
  let pair = pair.into_inner().next().unwrap(); // parsed qed
  Ok(Program {
    sequence: process_sequence(pair, env)?,
    metadata: Metadata {
      name: env.name.clone(),
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
    Rule::None => Ok(Value::None),
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
      let enum_name = splits[0];
      let variant_name = splits[1];
      Ok(Value::Enum(enum_name.into(), variant_name.into()))
    }
    Rule::Vector => process_vector(pair, env),
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
          let mut chars = full_str[1..full_str.len() - 1].chars();
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
          let full_str = inner.as_str();
          // remove triple quotes
          full_str[3..full_str.len() - 3].into()
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
            Rule::None => Value::None,
            Rule::Iden => Value::String(key.as_str().into()),
            Rule::Value => process_value(
              key.into_inner().next().unwrap(), // parsed qed
              env,
            )?,
            _ => unreachable!(),
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
    Rule::Hexadecimal => Ok(Number::Hexadecimal(pair.as_str().into())),
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
    let name = first.as_str();
    let name = name[0..name.len() - 1].into();
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
  })
}

fn process_params(pair: Pair<Rule>, env: &mut ReadEnv) -> Result<Vec<Param>, ShardsError> {
  pair.into_inner().map(|x| process_param(x, env)).collect()
}

pub fn read_with_env(code: &str, env: &mut ReadEnv) -> Result<Program, ShardsError> {
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

pub fn read(code: &str, name: &str, path: &str) -> Result<Program, ShardsError> {
  let mut env = ReadEnv::new(name, "", path);
  read_with_env(&code, &mut env)
}

use lazy_static::lazy_static;

lazy_static! {
  pub static ref READ_OUTPUT_TYPES: Vec<Type> = vec![common_type::string, common_type::bytes];
  static ref READ_PARAMETERS: Parameters = vec![(
    cstr!("Json"),
    shccstr!("If the output should be a json AST string instead of binary."),
    BOOL_TYPES_SLICE
  )
    .into()];
}

#[derive(shard, Default)]
#[shard_info(
  "Shards.Read",
  "Reads a Shards program and outputs a binary or json AST."
)]
pub(crate) struct ReadShard {
  output: ClonedVar,
  #[shard_param(
    "Json",
    "If the output should be a json AST string instead of binary.",
    BOOL_TYPES_SLICE
  )]
  as_json: ClonedVar,
  #[shard_param(
    "BasePath",
    "The base path to use when interpreting file references.",
    STRING_VAR_OR_NONE_SLICE
  )]
  base_path: ParamVar,
  #[shard_required]
  required_variables: ExposedTypes,
}

impl ReadShard {
  fn get_as_json(&self) -> bool {
    if self.as_json.0.is_bool() {
      (&self.as_json.0).try_into().unwrap()
    } else {
      false
    }
  }
}

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

  fn cleanup(&mut self) -> Result<(), &str> {
    self.cleanup_helper()?;
    Ok(())
  }

  fn compose(&mut self, _data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(_data)?;

    if self.get_as_json() {
      Ok(common_type::string)
    } else {
      Ok(common_type::bytes)
    }
  }

  fn activate(&mut self, _: &Context, input: &Var) -> Result<Var, &str> {
    let code: &str = input.try_into()?;

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

    let seq = process_program(
      parsed.into_iter().next().unwrap(), // parsed qed
      &mut ReadEnv::new("", "", base_path),
    )
    .map_err(|e| {
      shlog_error!("Failed to process shards code: {:?}", e);
      "Failed to tokenize Shards code"
    })?;

    if self.get_as_json() {
      // Serialize using json
      let encoded_json = serde_json::to_string(&seq).map_err(|e| {
        shlog_error!("Failed to serialize shards code: {}", e);
        "Failed to serialize Shards code"
      })?;

      let s = Var::ephemeral_string(encoded_json.as_str());
      self.output = s.into();
    } else {
      // Serialize using bincode
      let encoded_bin: Vec<u8> = bincode::serialize(&seq).map_err(|e| {
        shlog_error!("Failed to serialize shards code: {}", e);
        "Failed to serialize Shards code"
      })?;

      self.output = encoded_bin.as_slice().into();
    }

    Ok(self.output.0)
  }
}

#[test]
fn test_parsing1() {
  // use std::num::NonZeroUsize;
  // pest::set_call_limit(NonZeroUsize::new(25000));
  // let code = include_str!("nested.shs");
  let code = include_str!("sample1.shs");
  let successful_parse = ShardsParser::parse(Rule::Program, code).unwrap();
  let mut env = ReadEnv::new("", ".", ".");
  let seq = process_program(successful_parse.into_iter().next().unwrap(), &mut env).unwrap();

  // Serialize using bincode
  let encoded_bin: Vec<u8> = bincode::serialize(&seq).unwrap();

  // Deserialize using bincode
  let decoded_bin: Sequence = bincode::deserialize(&encoded_bin[..]).unwrap();

  // Serialize using json
  let encoded_json = serde_json::to_string(&seq).unwrap();
  println!("Json Serialized = {}", encoded_json);

  let encoded_json2 = serde_json::to_string(&decoded_bin).unwrap();
  assert_eq!(encoded_json, encoded_json2);

  // Deserialize using json
  let decoded_json: Sequence = serde_json::from_str(&encoded_json).unwrap();

  let encoded_bin2: Vec<u8> = bincode::serialize(&decoded_json).unwrap();
  assert_eq!(encoded_bin, encoded_bin2);
}

#[test]
fn test_parsing2() {
  let code = include_str!("explained.shs");
  let successful_parse = ShardsParser::parse(Rule::Program, code).unwrap();
  let mut env = ReadEnv::new("", ".", ".");
  let seq = process_program(successful_parse.into_iter().next().unwrap(), &mut env).unwrap();

  // Serialize using bincode
  let encoded_bin: Vec<u8> = bincode::serialize(&seq).unwrap();

  // Deserialize using bincode
  let decoded_bin: Sequence = bincode::deserialize(&encoded_bin[..]).unwrap();

  // Serialize using json
  let encoded_json = serde_json::to_string(&seq).unwrap();
  println!("Json Serialized = {}", encoded_json);

  let encoded_json2 = serde_json::to_string(&decoded_bin).unwrap();
  assert_eq!(encoded_json, encoded_json2);

  // Deserialize using json
  let decoded_json: Sequence = serde_json::from_str(&encoded_json).unwrap();

  let encoded_bin2: Vec<u8> = bincode::serialize(&decoded_json).unwrap();
  assert_eq!(encoded_bin, encoded_bin2);
}
