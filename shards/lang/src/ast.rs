#![allow(non_upper_case_globals)]

use pest::Position;
use serde::{Deserialize, Serialize};
use shards::{
  types::Var, SHType_Bool, SHType_Bytes, SHType_Float, SHType_Int, SHType_None, SHType_String,
};

use crate::{RcBytesWrapper, RcStrWrapper};

#[derive(Parser)]
#[grammar = "shards.pest"]
pub struct ShardsParser;

#[derive(Serialize, Deserialize, Debug, Copy, Clone, Default)]
pub struct LineInfo {
  pub line: u32,
  pub column: u32,
}

#[derive(Debug)]
pub struct ShardsError {
  pub message: String,
  pub loc: LineInfo,
}

impl<'a> Into<ShardsError> for (&str, Position<'a>) {
  fn into(self) -> ShardsError {
    let (message, pos) = self;
    let line = pos.line_col().0;
    let column = pos.line_col().1;
    ShardsError {
      message: message.to_string(),
      loc: LineInfo {
        line: line as u32,
        column: column as u32,
      },
    }
  }
}

impl<'a> Into<ShardsError> for (String, Position<'a>) {
  fn into(self) -> ShardsError {
    let (message, pos) = self;
    let line = pos.line_col().0;
    let column = pos.line_col().1;
    ShardsError {
      message,
      loc: LineInfo {
        line: line as u32,
        column: column as u32,
      },
    }
  }
}

impl<'a> Into<ShardsError> for (&str, LineInfo) {
  fn into(self) -> ShardsError {
    let (message, pos) = self;
    ShardsError {
      message: message.to_string(),
      loc: pos,
    }
  }
}

impl<'a> Into<ShardsError> for (String, LineInfo) {
  fn into(self) -> ShardsError {
    let (message, pos) = self;
    ShardsError { message, loc: pos }
  }
}

impl<'a> Into<LineInfo> for Position<'a> {
  fn into(self) -> LineInfo {
    let line = self.line_col().0;
    let column = self.line_col().1;
    LineInfo {
      line: line as u32,
      column: column as u32,
    }
  }
}

impl Into<(u32, u32)> for LineInfo {
  fn into(self) -> (u32, u32) {
    (self.line, self.column)
  }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Number {
  Integer(i64),
  Float(f64),
  Hexadecimal(RcStrWrapper),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct Identifier {
  pub name: RcStrWrapper,
  pub namespaces: Vec<RcStrWrapper>,
}

impl Identifier {
  pub fn resolve(&self) -> RcStrWrapper {
    if self.namespaces.is_empty() {
      return self.name.clone();
    } else {
      // go thru all namespaces and concatenate them with "/" finally add name
      let mut result = String::new();
      for namespace in &self.namespaces {
        result.push_str(&namespace);
        result.push('/');
      }
      result.push_str(&self.name);
      result.into()
    }
  }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Value {
  None,
  Identifier(Identifier),
  Boolean(bool),
  Enum(RcStrWrapper, RcStrWrapper),
  Number(Number),
  String(RcStrWrapper),
  Bytes(RcBytesWrapper),
  Int2([i64; 2]),
  Int3([i32; 3]),
  Int4([i32; 4]),
  Int8([i16; 8]),
  Int16([i8; 16]),
  Float2([f64; 2]),
  Float3([f32; 3]),
  Float4([f32; 4]),
  Seq(Vec<Value>),
  Table(Vec<(Value, Value)>),
  Shard(Function),
  Shards(Sequence),
  EvalExpr(Sequence),
  Expr(Sequence),
  TakeTable(Identifier, Vec<RcStrWrapper>),
  TakeSeq(Identifier, Vec<u32>),
  Func(Function),
}

impl TryFrom<Var> for Value {
  type Error = &'static str;

  fn try_from(value: Var) -> Result<Self, Self::Error> {
    match value.valueType {
      SHType_None => Ok(Value::None),
      SHType_Bool => Ok(Value::Boolean(value.as_ref().try_into().unwrap())),
      SHType_Int => Ok(Value::Number(Number::Integer(
        value.as_ref().try_into().unwrap(),
      ))),
      SHType_Float => Ok(Value::Number(Number::Float(
        value.as_ref().try_into().unwrap(),
      ))),
      SHType_String => {
        let s: &str = value.as_ref().try_into().unwrap();
        Ok(Value::String(s.into()))
      }
      SHType_Bytes => {
        let b: &[u8] = value.as_ref().try_into().unwrap();
        Ok(Value::Bytes(b.into()))
      }
      _ => Err("Unsupported type"),
    }
  }
}

impl Value {
  pub fn get_identifier(&self) -> Option<&Identifier> {
    match self {
      Value::Identifier(identifier) => Some(identifier),
      _ => None,
    }
  }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Param {
  pub name: Option<RcStrWrapper>,
  pub value: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
  pub name: Identifier,
  pub params: Option<Vec<Param>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BlockContent {
  Empty,
  Shard(Function),                          // Rule: Shard
  Shards(Sequence),                         // Rule: Shards
  Const(Value),                             // Rules: ConstValue, Vector
  TakeTable(Identifier, Vec<RcStrWrapper>), // Rule: TakeTable
  TakeSeq(Identifier, Vec<u32>),            // Rule: TakeSeq
  EvalExpr(Sequence),                       // Rule: EvalExpr
  Expr(Sequence),                           // Rule: Expr
  Func(Function),                           // Rule: BuiltIn
  Program(Program), // @include files, this is a sequence that will include itself when evaluated
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Block {
  pub content: BlockContent,
  pub line_info: Option<LineInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Pipeline {
  pub blocks: Vec<Block>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Assignment {
  AssignRef(Pipeline, Identifier),
  AssignSet(Pipeline, Identifier),
  AssignUpd(Pipeline, Identifier),
  AssignPush(Pipeline, Identifier),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Statement {
  Assignment(Assignment),
  Pipeline(Pipeline),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Metadata {
  pub name: RcStrWrapper,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequence {
  pub statements: Vec<Statement>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Program {
  pub sequence: Sequence,
  pub metadata: Metadata,
}
