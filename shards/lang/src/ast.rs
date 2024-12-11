#![allow(non_upper_case_globals)]

use crate::{custom_state::CustomStateContainer, RcBytesWrapper, RcStrWrapper};
use core::{fmt, hash::Hash};
use pest::Position;
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use shards::{
  types::Var, SHType_Bool, SHType_Bytes, SHType_Float, SHType_Int, SHType_None, SHType_String,
};
use std::{cell::RefCell, collections::HashMap, fmt::Debug, hash::Hasher};

#[derive(Parser)]
#[grammar = "shards.pest"]
pub struct ShardsParser;

#[derive(Serialize, Deserialize, Debug, Copy, Clone, Default, PartialEq)]
pub struct LineInfo {
  pub line: u32,
  pub column: u32,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename = "Num")]
pub enum Number {
  #[serde(rename = "int")]
  Integer(i64),
  #[serde(rename = "float")]
  Float(f64),
  #[serde(rename = "hex")]
  Hexadecimal(RcStrWrapper),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename = "Iden")]
pub struct Identifier {
  pub name: RcStrWrapper,
  #[serde(default)]
  #[serde(rename = "ns")]
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub namespaces: Vec<RcStrWrapper>,

  #[serde(skip)]
  pub custom_state: CustomStateContainer,
}

impl fmt::Display for Identifier {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if self.namespaces.is_empty() {
      write!(f, "{}", self.name)
    } else {
      write!(
        f,
        "{}::{}",
        self
          .namespaces
          .iter()
          .map(|s| s.as_ref())
          .collect::<Vec<&str>>()
          .join("/"),
        self.name
      )
    }
  }
}

impl Eq for Identifier {}

impl Hash for Identifier {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.name.hash(state);
    self.namespaces.hash(state);
  }
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Value {
  #[serde(rename = "none")]
  None(()), // We leave it as (), to achieve serializer consistency
  #[serde(rename = "id")]
  Identifier(Identifier),
  #[serde(rename = "bool")]
  Boolean(bool),
  #[serde(rename = "enum")]
  Enum(RcStrWrapper, RcStrWrapper),
  #[serde(rename = "num")]
  Number(Number),
  #[serde(rename = "str")]
  String(RcStrWrapper),
  #[serde(rename = "bytes")]
  Bytes(RcBytesWrapper),
  #[serde(rename = "i2")]
  Int2([i64; 2]),
  #[serde(rename = "i3")]
  Int3([i32; 3]),
  #[serde(rename = "i4")]
  Int4([i32; 4]),
  #[serde(rename = "i8")]
  Int8([i16; 8]),
  #[serde(rename = "i16")]
  Int16([i8; 16]),
  #[serde(rename = "f2")]
  Float2([f64; 2]),
  #[serde(rename = "f3")]
  Float3([f32; 3]),
  #[serde(rename = "f4")]
  Float4([f32; 4]),
  #[serde(rename = "seq")]
  Seq(Vec<Value>),
  #[serde(rename = "table")]
  Table(Vec<(Value, Value)>),
  #[serde(rename = "sh")]
  Shard(Function),
  #[serde(rename = "shs")]
  Shards(Sequence),
  #[serde(rename = "eExpr")]
  EvalExpr(Sequence),
  #[serde(rename = "expr")]
  Expr(Sequence),
  #[serde(rename = "tt")]
  TakeTable(Identifier, Vec<RcStrWrapper>),
  #[serde(rename = "ts")]
  TakeSeq(Identifier, Vec<u32>),
  #[serde(rename = "func")]
  Func(Function),
}

impl TryFrom<Var> for Value {
  type Error = &'static str;

  fn try_from(value: Var) -> Result<Self, Self::Error> {
    match value.valueType {
      SHType_None => Ok(Value::None(())),
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

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
  pub name: Option<RcStrWrapper>,
  pub value: Value,

  pub is_default: Option<bool>, // This is used to determine if the param is default or not, optional

  pub custom_state: CustomStateContainer,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Function {
  #[serde(flatten)]
  pub name: Identifier,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub params: Option<Vec<Param>>,

  #[serde(skip)]
  pub custom_state: CustomStateContainer,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BlockContent {
  #[serde(rename = "none")]
  Empty,
  #[serde(rename = "sh")]
  Shard(Function), // Rule: Shard
  #[serde(rename = "shs")]
  Shards(Sequence), // Rule: Shards
  #[serde(rename = "const")]
  Const(Value), // Rules: ConstValue, Vector
  #[serde(rename = "tt")]
  TakeTable(Identifier, Vec<RcStrWrapper>), // Rule: TakeTable
  #[serde(rename = "ts")]
  TakeSeq(Identifier, Vec<u32>), // Rule: TakeSeq
  #[serde(rename = "eExpr")]
  EvalExpr(Sequence), // Rule: EvalExpr
  #[serde(rename = "expr")]
  Expr(Sequence), // Rule: Expr
  #[serde(rename = "func")]
  Func(Function), // Rule: BuiltIn
  #[serde(rename = "prog")]
  Program(Program), // @include files, this is a sequence that will include itself when evaluated
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
  pub content: BlockContent,
  pub line_info: Option<LineInfo>,

  pub custom_state: CustomStateContainer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pipeline {
  pub blocks: Vec<Block>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Assignment {
  #[serde(rename = "ref")]
  AssignRef(Pipeline, Identifier),
  #[serde(rename = "set")]
  AssignSet(Pipeline, Identifier),
  #[serde(rename = "upd")]
  AssignUpd(Pipeline, Identifier),
  #[serde(rename = "push")]
  AssignPush(Pipeline, Identifier),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
  Assignment(Assignment),
  Pipeline(Pipeline),
}

impl Statement {
  pub fn as_pipeline_mut(&mut self) -> Option<&mut Pipeline> {
    match self {
      Statement::Pipeline(pipeline) => Some(pipeline),
      _ => None,
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DebugPtr {
  Function(*const Function),
  Identifier(*const Identifier),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DebugInfo {
  pub id_to_functions: HashMap<u64, DebugPtr>,
  pub id_counter: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Metadata {
  pub name: RcStrWrapper,

  #[serde(skip)]
  pub debug_info: RefCell<DebugInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
  pub statements: Vec<Statement>,

  pub custom_state: CustomStateContainer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
  pub sequence: Sequence,
  pub metadata: Metadata,
}

pub trait RewriteFunction {
  fn rewrite_function(&self, function: &Function) -> Option<Function>;
}

impl Serialize for Sequence {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    self.statements.serialize(serializer)
  }
}

impl<'de> Deserialize<'de> for Sequence {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let statements = Vec::deserialize(deserializer)?;
    Ok(Sequence {
      statements,
      custom_state: CustomStateContainer::new(),
    })
  }
}

impl Serialize for Program {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::SerializeStruct;
    let mut state = serializer.serialize_struct("Program", 2)?;
    state.serialize_field("metadata", &self.metadata)?;
    state.serialize_field("sequence", &self.sequence.statements)?;
    state.end()
  }
}

impl<'de> Deserialize<'de> for Program {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    #[derive(Deserialize)]
    struct ProgramHelper {
      metadata: Metadata,
      sequence: Vec<Statement>,
    }

    let helper = ProgramHelper::deserialize(deserializer)?;
    Ok(Program {
      metadata: helper.metadata,
      sequence: Sequence {
        statements: helper.sequence,
        custom_state: CustomStateContainer::new(),
      },
    })
  }
}

// Custom serialization for Pipeline
impl Serialize for Pipeline {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    // Directly serialize the blocks vector
    self.blocks.serialize(serializer)
  }
}

// Custom deserialization for Pipeline
impl<'de> Deserialize<'de> for Pipeline {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    // Directly deserialize into a Vec<Block>
    let blocks = Vec::<Block>::deserialize(deserializer)?;
    Ok(Pipeline { blocks })
  }
}

// Custom serialization for Statement
impl Serialize for Statement {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    match self {
      Statement::Assignment(assignment) => assignment.serialize(serializer),
      Statement::Pipeline(pipeline) => pipeline.serialize(serializer),
    }
  }
}

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};

impl<'de> Deserialize<'de> for Statement {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    struct StatementVisitor;

    impl<'de> Visitor<'de> for StatementVisitor {
      type Value = Statement;

      fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a map (for Assignment) or a sequence (for Pipeline)")
      }

      fn visit_seq<V>(self, seq: V) -> Result<Self::Value, V::Error>
      where
        V: SeqAccess<'de>,
      {
        let pipeline = Pipeline::deserialize(de::value::SeqAccessDeserializer::new(seq))?;
        Ok(Statement::Pipeline(pipeline))
      }

      fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
      where
        M: MapAccess<'de>,
      {
        let assignment = Assignment::deserialize(de::value::MapAccessDeserializer::new(map))?;
        Ok(Statement::Assignment(assignment))
      }
    }

    deserializer.deserialize_any(StatementVisitor)
  }
}

impl Serialize for Block {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::SerializeStruct;
    let mut state = serializer.serialize_struct("Block", 2)?;

    // Flatten the content
    match &self.content {
      BlockContent::Empty => state.serialize_field("none", &()),
      BlockContent::Shard(func) => state.serialize_field("sh", func),
      BlockContent::Shards(seq) => state.serialize_field("shs", seq),
      BlockContent::Const(val) => state.serialize_field("const", val),
      BlockContent::TakeTable(id, vec) => state.serialize_field("tt", &(id, vec)),
      BlockContent::TakeSeq(id, vec) => state.serialize_field("ts", &(id, vec)),
      BlockContent::EvalExpr(seq) => state.serialize_field("eExpr", seq),
      BlockContent::Expr(seq) => state.serialize_field("expr", seq),
      BlockContent::Func(func) => state.serialize_field("func", func),
      BlockContent::Program(prog) => state.serialize_field("prog", prog),
    }?;

    if let Some(line_info) = &self.line_info {
      state.serialize_field("line_info", line_info)?;
    }

    state.end()
  }
}

impl<'de> Deserialize<'de> for Block {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    enum Field {
      None,
      Sh,
      Shs,
      Const,
      Tt,
      Ts,
      EExpr,
      Expr,
      Func,
      Prog,
      LineInfo,
    }

    impl<'de> Deserialize<'de> for Field {
      fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
      where
        D: Deserializer<'de>,
      {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
          type Value = Field;

          fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("`none`, `sh`, `shs`, `const`, `tt`, `ts`, `eExpr`, `expr`, `func`, `prog`, or `line_info`")
          }

          fn visit_str<E>(self, value: &str) -> Result<Field, E>
          where
            E: de::Error,
          {
            match value {
              "none" => Ok(Field::None),
              "sh" => Ok(Field::Sh),
              "shs" => Ok(Field::Shs),
              "const" => Ok(Field::Const),
              "tt" => Ok(Field::Tt),
              "ts" => Ok(Field::Ts),
              "eExpr" => Ok(Field::EExpr),
              "expr" => Ok(Field::Expr),
              "func" => Ok(Field::Func),
              "prog" => Ok(Field::Prog),
              "line_info" => Ok(Field::LineInfo),
              _ => Err(de::Error::unknown_field(value, FIELDS)),
            }
          }
        }

        deserializer.deserialize_identifier(FieldVisitor)
      }
    }

    struct BlockVisitor;

    impl<'de> Visitor<'de> for BlockVisitor {
      type Value = Block;

      fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("struct Block")
      }

      fn visit_map<V>(self, mut map: V) -> Result<Block, V::Error>
      where
        V: MapAccess<'de>,
      {
        let mut content = None;
        let mut line_info = None;

        while let Some(key) = map.next_key()? {
          match key {
            Field::None => {
              map.next_value::<()>()?;
              content = Some(BlockContent::Empty);
            }
            Field::Sh => {
              let value = map.next_value()?;
              content = Some(BlockContent::Shard(value));
            }
            Field::Shs => {
              let value = map.next_value()?;
              content = Some(BlockContent::Shards(value));
            }
            Field::Const => {
              let value = map.next_value()?;
              content = Some(BlockContent::Const(value));
            }
            Field::Tt => {
              let (id, vec): (Identifier, Vec<RcStrWrapper>) = map.next_value()?;
              content = Some(BlockContent::TakeTable(id, vec));
            }
            Field::Ts => {
              let (id, vec): (Identifier, Vec<u32>) = map.next_value()?;
              content = Some(BlockContent::TakeSeq(id, vec));
            }
            Field::EExpr => {
              let value = map.next_value()?;
              content = Some(BlockContent::EvalExpr(value));
            }
            Field::Expr => {
              let value = map.next_value()?;
              content = Some(BlockContent::Expr(value));
            }
            Field::Func => {
              let value = map.next_value()?;
              content = Some(BlockContent::Func(value));
            }
            Field::Prog => {
              let value = map.next_value()?;
              content = Some(BlockContent::Program(value));
            }
            Field::LineInfo => {
              let value = map.next_value()?;
              line_info = Some(value);
            }
          }
        }

        let content = content.ok_or_else(|| de::Error::missing_field("content"))?;

        Ok(Block {
          content,
          line_info,
          custom_state: CustomStateContainer::new(),
        })
      }
    }

    const FIELDS: &[&str] = &[
      "none", "sh", "shs", "const", "tt", "ts", "eExpr", "expr", "func", "prog",
    ];
    deserializer.deserialize_struct("Block", FIELDS, BlockVisitor)
  }
}

impl Serialize for Param {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let mut state = serializer.serialize_struct("Param", 2)?;

    // Serialize the name if it exists
    if let Some(name) = &self.name {
      state.serialize_field("name", name)?;
    }

    // Flatten the value field
    match &self.value {
      Value::None(_) => state.serialize_field("none", &()),
      Value::Identifier(id) => state.serialize_field("id", id),
      Value::Boolean(b) => state.serialize_field("bool", b),
      Value::Enum(e1, e2) => state.serialize_field("enum", &(e1, e2)),
      Value::Number(n) => state.serialize_field("num", n),
      Value::String(s) => state.serialize_field("str", s),
      Value::Bytes(b) => state.serialize_field("bytes", b),
      Value::Int2(arr) => state.serialize_field("i2", arr),
      Value::Int3(arr) => state.serialize_field("i3", arr),
      Value::Int4(arr) => state.serialize_field("i4", arr),
      Value::Int8(arr) => state.serialize_field("i8", arr),
      Value::Int16(arr) => state.serialize_field("i16", arr),
      Value::Float2(arr) => state.serialize_field("f2", arr),
      Value::Float3(arr) => state.serialize_field("f3", arr),
      Value::Float4(arr) => state.serialize_field("f4", arr),
      Value::Seq(seq) => state.serialize_field("seq", seq),
      Value::Table(table) => state.serialize_field("table", table),
      Value::Shard(func) => state.serialize_field("sh", func),
      Value::Shards(seq) => state.serialize_field("shs", seq),
      Value::EvalExpr(seq) => state.serialize_field("eExpr", seq),
      Value::Expr(seq) => state.serialize_field("expr", seq),
      Value::TakeTable(id, vec) => state.serialize_field("tt", &(id, vec)),
      Value::TakeSeq(id, vec) => state.serialize_field("ts", &(id, vec)),
      Value::Func(func) => state.serialize_field("func", func),
    }?;

    state.end()
  }
}

impl<'de> Deserialize<'de> for Param {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    enum Field {
      Name,
      None,
      Id,
      Bool,
      Enum,
      Num,
      Str,
      Bytes,
      I2,
      I3,
      I4,
      I8,
      I16,
      F2,
      F3,
      F4,
      Seq,
      Table,
      Sh,
      Shs,
      EExpr,
      Expr,
      Tt,
      Ts,
      Func,
    }

    impl<'de> Deserialize<'de> for Field {
      fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
      where
        D: Deserializer<'de>,
      {
        struct FieldVisitor;

        impl<'de> Visitor<'de> for FieldVisitor {
          type Value = Field;

          fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a valid Param field")
          }

          fn visit_str<E>(self, value: &str) -> Result<Field, E>
          where
            E: de::Error,
          {
            match value {
              "name" => Ok(Field::Name),
              "none" => Ok(Field::None),
              "id" => Ok(Field::Id),
              "bool" => Ok(Field::Bool),
              "enum" => Ok(Field::Enum),
              "num" => Ok(Field::Num),
              "str" => Ok(Field::Str),
              "bytes" => Ok(Field::Bytes),
              "i2" => Ok(Field::I2),
              "i3" => Ok(Field::I3),
              "i4" => Ok(Field::I4),
              "i8" => Ok(Field::I8),
              "i16" => Ok(Field::I16),
              "f2" => Ok(Field::F2),
              "f3" => Ok(Field::F3),
              "f4" => Ok(Field::F4),
              "seq" => Ok(Field::Seq),
              "table" => Ok(Field::Table),
              "sh" => Ok(Field::Sh),
              "shs" => Ok(Field::Shs),
              "eExpr" => Ok(Field::EExpr),
              "expr" => Ok(Field::Expr),
              "tt" => Ok(Field::Tt),
              "ts" => Ok(Field::Ts),
              "func" => Ok(Field::Func),
              _ => Err(de::Error::unknown_field(value, FIELDS)),
            }
          }
        }

        deserializer.deserialize_identifier(FieldVisitor)
      }
    }

    struct ParamVisitor;

    impl<'de> Visitor<'de> for ParamVisitor {
      type Value = Param;

      fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("struct Param")
      }

      fn visit_map<V>(self, mut map: V) -> Result<Param, V::Error>
      where
        V: MapAccess<'de>,
      {
        let mut name = None;
        let mut value = None;

        while let Some(key) = map.next_key()? {
          match key {
            Field::Name => name = Some(map.next_value()?),
            Field::None => {
              map.next_value::<()>()?;
              value = Some(Value::None(()));
            }
            Field::Id => value = Some(Value::Identifier(map.next_value()?)),
            Field::Bool => value = Some(Value::Boolean(map.next_value()?)),
            Field::Enum => {
              let (e1, e2): (RcStrWrapper, RcStrWrapper) = map.next_value()?;
              value = Some(Value::Enum(e1, e2));
            }
            Field::Num => value = Some(Value::Number(map.next_value()?)),
            Field::Str => value = Some(Value::String(map.next_value()?)),
            Field::Bytes => value = Some(Value::Bytes(map.next_value()?)),
            Field::I2 => value = Some(Value::Int2(map.next_value()?)),
            Field::I3 => value = Some(Value::Int3(map.next_value()?)),
            Field::I4 => value = Some(Value::Int4(map.next_value()?)),
            Field::I8 => value = Some(Value::Int8(map.next_value()?)),
            Field::I16 => value = Some(Value::Int16(map.next_value()?)),
            Field::F2 => value = Some(Value::Float2(map.next_value()?)),
            Field::F3 => value = Some(Value::Float3(map.next_value()?)),
            Field::F4 => value = Some(Value::Float4(map.next_value()?)),
            Field::Seq => value = Some(Value::Seq(map.next_value()?)),
            Field::Table => value = Some(Value::Table(map.next_value()?)),
            Field::Sh => value = Some(Value::Shard(map.next_value()?)),
            Field::Shs => value = Some(Value::Shards(map.next_value()?)),
            Field::EExpr => value = Some(Value::EvalExpr(map.next_value()?)),
            Field::Expr => value = Some(Value::Expr(map.next_value()?)),
            Field::Tt => {
              let (id, vec): (Identifier, Vec<RcStrWrapper>) = map.next_value()?;
              value = Some(Value::TakeTable(id, vec));
            }
            Field::Ts => {
              let (id, vec): (Identifier, Vec<u32>) = map.next_value()?;
              value = Some(Value::TakeSeq(id, vec));
            }
            Field::Func => value = Some(Value::Func(map.next_value()?)),
          }
        }

        let value = value.ok_or_else(|| de::Error::missing_field("value"))?;

        Ok(Param {
          name,
          value,
          is_default: None,
          custom_state: CustomStateContainer::new(),
        })
      }
    }

    const FIELDS: &[&str] = &[
      "name", "none", "id", "bool", "enum", "num", "str", "bytes", "i2", "i3", "i4", "i8", "i16",
      "f2", "f3", "f4", "seq", "table", "sh", "shs", "eExpr", "expr", "tt", "ts", "func",
    ];
    deserializer.deserialize_struct("Param", FIELDS, ParamVisitor)
  }
}
