use shards::shard::Shard;
use shards::shlog_error;
use shards::types::ClonedVar;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::Type;
use shards::types::Types;
use shards::types::Var;
use shards::types::STRING_TYPES;
use shards::SHContext;

use crate::ast::*;
use crate::ast_visitor::*;
use crate::formatter::format_str;
use crate::read;
use crate::read::AST_TYPE;

impl Identifier {
  fn to_string(&self) -> String {
    if self.namespaces.is_empty() {
      self.name.to_string()
    } else {
      let namespaces_str = self
        .namespaces
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<String>>()
        .join("/");
      format!("{}/{}", namespaces_str, self.name)
    }
  }
}

impl Number {
  fn to_string(&self) -> String {
    match self {
      Number::Integer(i) => format!("{}", i),
      Number::Float(f) => {
        if f.fract() == 0.0 {
          format!("{}.0", f)
        } else {
          format!("{}", f)
        }
      }
      Number::Hexadecimal(s) => format!("{}", s),
    }
  }
}

fn format_f64(f: f64) -> String {
  if f.fract() == 0.0 {
    format!("{:.1}", f)
  } else {
    format!("{}", f)
  }
}

fn format_f32(f: f32) -> String {
  if f.fract() == 0.0 {
    format!("{:.1}", f)
  } else {
    format!("{}", f)
  }
}

pub struct PrintVisitor {
  output: String,
}

impl PrintVisitor {
  pub fn new() -> Self {
    PrintVisitor {
      output: String::new(),
    }
  }

  pub fn result(self) -> String {
    self.output
  }

  fn write(&mut self, text: &str) {
    self.output.push_str(text);
  }
}

impl AstVisitor for PrintVisitor {
  fn visit_program(&mut self, program: &Program) {
    program.metadata.accept(self);
    program.sequence.accept(self);
  }

  fn visit_sequence(&mut self, sequence: &Sequence) {
    for statement in &sequence.statements {
      statement.accept(self);
    }
  }

  fn visit_statement(&mut self, statement: &Statement) {
    match statement {
      Statement::Assignment(assignment) => assignment.accept(self),
      Statement::Pipeline(pipeline) => pipeline.accept(self),
    }
    self.write("\n")
  }

  fn visit_assignment(&mut self, assignment: &Assignment) {
    match assignment {
      Assignment::AssignRef(pipeline, identifier) => {
        pipeline.accept(self);
        self.write(&format!(" = {}", identifier.to_string()));
      }
      Assignment::AssignSet(pipeline, identifier) => {
        pipeline.accept(self);
        self.write(&format!(" >= {}", identifier.to_string()));
      }
      Assignment::AssignUpd(pipeline, identifier) => {
        pipeline.accept(self);
        self.write(&format!(" > {}", identifier.to_string()));
      }
      Assignment::AssignPush(pipeline, identifier) => {
        pipeline.accept(self);
        self.write(&format!(" >> {}", identifier.to_string()));
      }
    }
  }

  fn visit_pipeline(&mut self, pipeline: &Pipeline) {
    for (i, block) in pipeline.blocks.iter().enumerate() {
      if i > 0 {
        self.write(" | ");
      }
      block.accept(self);
    }
  }

  fn visit_block(&mut self, block: &Block) {
    match &block.content {
      BlockContent::Empty => self.write(""),
      BlockContent::Shard(function) => function.accept(self),
      BlockContent::Func(function) => {
        self.write("@");
        function.accept(self);
      }
      BlockContent::Shards(sequence) => {
        self.write("{");
        sequence.accept(self);
        self.write("}");
      }
      BlockContent::Const(value) => value.accept(self),
      BlockContent::TakeTable(identifier, path) => {
        let path = path.iter().map(|p| p.to_string()).collect::<Vec<String>>();
        self.write(&format!("{}:{}", identifier.to_string(), path.join(":")));
      }
      BlockContent::TakeSeq(identifier, path) => {
        self.write(&format!(
          "{}:{}",
          identifier.to_string(),
          path
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<String>>()
            .join(":")
        ));
      }
      BlockContent::EvalExpr(sequence) => {
        self.write("#(");
        sequence.accept(self);
        self.write(")");
      }
      BlockContent::Expr(sequence) => {
        self.write("(");
        sequence.accept(self);
        self.write(")");
      }
      BlockContent::Program(program) => {
        program.sequence.accept(self);
      }
    }
  }

  fn visit_function(&mut self, function: &Function) {
    self.write(&function.name.to_string());
    if let Some(params) = &function.params {
      let params_str = params
        .iter()
        .map(|v| {
          let mut visitor = PrintVisitor::new();
          v.accept(&mut visitor);
          visitor.result()
        })
        .collect::<Vec<String>>()
        .join(" ");
      if !params_str.is_empty() {
        self.write(&format!("({})", params_str));
      }
    }
  }

  fn visit_param(&mut self, param: &Param) {
    match param.is_default {
      Some(true) => {} // do nothing, omit default params
      _ => {
        if let Some(name) = &param.name {
          self.write(&format!("{}: ", name));
        }
        param.value.accept(self);
      }
    }
  }

  fn visit_identifier(&mut self, identifier: &Identifier) {
    self.write(&identifier.to_string());
  }

  fn visit_value(&mut self, value: &Value) {
    match value {
      Value::None(_) => self.write("none"),
      Value::Identifier(identifier) => self.visit_identifier(identifier),
      Value::Boolean(b) => self.write(&format!("{}", b)),
      Value::Enum(a, b) => self.write(&format!("{}::{}", a, b)),
      Value::Number(number) => self.write(&number.to_string()),
      Value::String(s) => {
        let escaped = s.replace("\"", "\\\"");
        self.write(&format!("\"{}\"", escaped));
      }
      Value::Bytes(b) => {
        let bytes_slice = b.0.as_ref();
        let hex_string: String = bytes_slice
          .iter()
          .map(|byte| format!("{:02x}", byte))
          .collect();
        self.write(&format!("@bytes(0x{})", hex_string));
      }
      Value::Int2(arr) => self.write(&format!("@i2({} {})", arr[0], arr[1])),
      Value::Int3(arr) => self.write(&format!("@i3({} {} {})", arr[0], arr[1], arr[2])),
      Value::Int4(arr) => self.write(&format!("@i4({} {} {} {})", arr[0], arr[1], arr[2], arr[3])),
      Value::Float2(arr) => self.write(&format!(
        "@f2({} {})",
        format_f64(arr[0]),
        format_f64(arr[1])
      )),
      Value::Float3(arr) => self.write(&format!(
        "@f3({} {} {})",
        format_f32(arr[0]),
        format_f32(arr[1]),
        format_f32(arr[2])
      )),
      Value::Float4(arr) => self.write(&format!(
        "@f4({} {} {} {})",
        format_f32(arr[0]),
        format_f32(arr[1]),
        format_f32(arr[2]),
        format_f32(arr[3])
      )),
      Value::Seq(values) => {
        let values_str = values
          .iter()
          .map(|v| {
            let mut visitor = PrintVisitor::new();
            v.accept(&mut visitor);
            visitor.result()
          })
          .collect::<Vec<String>>()
          .join(" ");
        self.write(&format!("[{}]", values_str));
      }
      Value::TakeTable(identifier, path) => {
        let path = path.iter().map(|p| p.to_string()).collect::<Vec<String>>();
        let path = path.join(":");
        self.write(&format!("{}:{}", identifier.to_string(), path));
      }
      Value::TakeSeq(identifier, path) => {
        let path_str = path
          .iter()
          .map(|p| p.to_string())
          .collect::<Vec<String>>()
          .join(":");
        self.write(&format!("{}:{}", identifier.to_string(), path_str));
      }
      Value::Int8(arr) => self.write(&format!(
        "@i8({} {} {} {} {} {} {} {})",
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]
      )),
      Value::Int16(arr) => self.write(&format!(
        "@i16({} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {})",
        arr[0],
        arr[1],
        arr[2],
        arr[3],
        arr[4],
        arr[5],
        arr[6],
        arr[7],
        arr[8],
        arr[9],
        arr[10],
        arr[11],
        arr[12],
        arr[13],
        arr[14],
        arr[15]
      )),
      Value::Table(values) => {
        let values_str = values
          .iter()
          .map(|(k, v)| {
            let mut key_visitor = PrintVisitor::new();
            k.accept(&mut key_visitor);
            let key_str = key_visitor.result();

            let mut value_visitor = PrintVisitor::new();
            v.accept(&mut value_visitor);
            let value_str = value_visitor.result();

            format!("{}: {}", key_str, value_str)
          })
          .collect::<Vec<String>>()
          .join(" ");
        self.write(&format!("{{{}}}", values_str));
      }
      Value::Shards(seq) => {
        self.write("{\n");
        seq.accept(self);
        self.write("}");
      }
      Value::Shard(func) => func.accept(self),
      Value::EvalExpr(seq) => {
        self.write("#(");
        seq.accept(self);
        self.write(")");
      }
      Value::Expr(seq) => {
        self.write("(");
        seq.accept(self);
        self.write(")");
      }
      Value::Func(func) => {
        self.write("@");
        func.accept(self);
      }
    }
  }

  fn visit_metadata(&mut self, metadata: &Metadata) {
    self.write(&metadata.name.to_string());
  }
}

pub fn print_ast(ast: &Sequence) -> String {
  let mut visitor = PrintVisitor::new();
  ast.accept(&mut visitor);
  visitor.result()
}

#[derive(shards::shard)]
#[shard_info("Shards.Print", "Pretty prints shards AST into a string.")]
pub struct ShardsPrintShard {
  #[shard_required]
  required: ExposedTypes,

  output: ClonedVar,
}

impl Default for ShardsPrintShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for ShardsPrintShard {
  fn input_types(&mut self) -> &Types {
    &read::READ_OUTPUT_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn warmup(&mut self, ctx: &SHContext) -> Result<(), &str> {
    self.warmup_helper(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&SHContext>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &SHContext, input: &Var) -> Result<Option<Var>, &str> {
    // ok we have 3 types, string (json), bytes (binary), and object (Program Rc etc)
    let string_ast: Result<&str, _> = input.try_into();
    let bytes_ast: Result<&[u8], _> = input.try_into();
    let object_ast = unsafe { Var::from_ref_counted_object::<Program>(input, &AST_TYPE) };
    match (string_ast, bytes_ast, object_ast) {
      (Ok(string), _, _) => {
        let program = serde_json::from_str::<Program>(string).map_err(|e| {
          shlog_error!("Failed to deserialize shards code: {}", e);
          "Failed to deserialize Shards code"
        })?;
        let printed = print_ast(&program.sequence);
        let formatted = format_str(&printed).map_err(|e| {
          shlog_error!("Failed to format shards code: {}", e);
          "Failed to format Shards code"
        })?;
        self.output = Var::ephemeral_string(formatted.as_str()).into();
      }
      (_, Ok(bytes), _) => {
        let program = flexbuffers::from_slice::<Program>(bytes).map_err(|e| {
          shlog_error!("Failed to deserialize shards code: {}", e);
          "Failed to deserialize Shards code"
        })?;
        let printed = print_ast(&program.sequence);
        let formatted = format_str(&printed).map_err(|e| {
          shlog_error!("Failed to format shards code: {}", e);
          "Failed to format Shards code"
        })?;
        self.output = Var::ephemeral_string(formatted.as_str()).into();
      }
      (_, _, Ok(rc)) => {
        let program = unsafe { &*rc };
        let printed = print_ast(&program.sequence);
        let formatted = format_str(&printed).map_err(|e| {
          shlog_error!("Failed to format shards code: {}", e);
          "Failed to format Shards code"
        })?;
        self.output = Var::ephemeral_string(formatted.as_str()).into();
      }
      _ => return Err("Invalid input type"),
    }
    Ok(Some(self.output.0))
  }
}

#[test]
fn test1() {
  use pest::Parser;
  // use std::num::NonZeroUsize;
  // pest::set_call_limit(NonZeroUsize::new(25000));
  // let code = include_str!("nested.shs");
  let code = include_str!("sample1.shs");
  let successful_parse = crate::ShardsParser::parse(Rule::Program, code).unwrap();
  let mut env = crate::read::ReadEnv::new("", ".", ".");
  let seq =
    crate::read::process_program(successful_parse.into_iter().next().unwrap(), &mut env).unwrap();
  let seq = seq.sequence;

  let printed = print_ast(&seq);

  println!("{}", printed);
}
