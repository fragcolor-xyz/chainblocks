use std::{
  fs::{self, File},
  io::Read,
  path::Path,
};

use crate::rule_visitor::RuleVisitor;
use pest::iterators::Pair;

pub type Rule = crate::ast::Rule;

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum NewlineStyle {
  LF,
  CRLF,
}

impl NewlineStyle {
  pub fn from_index(idx: usize) -> Self {
    match idx {
      0 => Self::LF,
      1 => Self::CRLF,
      _ => Self::LF,
    }
  }
  pub fn to_str(&self) -> &str {
    match self {
      Self::LF => "\n",
      Self::CRLF => "\r\n",
    }
  }
  pub fn push_to_str(&self, s: &mut String) {
    match self {
      Self::LF => s.push('\n'),
      Self::CRLF => {
        s.push('\r');
        s.push('\n');
      }
    }
  }
}

// Identifies the last written value type
#[derive(PartialEq, Debug)]
enum FormatterTop {
  None,
  Atom,
  Comment,
  LineFunc,
}

#[derive(Clone, Copy, Default, Debug)]
struct CollectionStyling {
  start_line: usize,
  // The indent for the first item of this collections, subsequent items will use the same indent
  indent: Option<usize>,
  // Joined collection, where braces are connected to the items e.g.:
  // {a:1
  //  b:2}
  is_joined: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
enum Context {
  Unknown,
  Pipeline,
  Seq(CollectionStyling),
  Table(CollectionStyling),
}

pub struct Options {
  pub indent: usize,
}

impl Default for Options {
  fn default() -> Self {
    Self { indent: 2 }
  }
}

pub struct FormatterVisitor<'a> {
  out: &'a mut dyn std::io::Write,
  options: Options,
  top: FormatterTop,
  line_length: usize,
  line_counter: usize,
  depth: usize,
  last_char: usize,
  context_stack: Vec<Context>,
  pub newline_style: NewlineStyle,

  input: String,
}

enum UserLine {
  Newline,
  Comment(String),
}

#[derive(Default)]
struct UserStyling {
  lines: Vec<UserLine>,
}

struct QuoteState {
  start: usize,
  num_quotes: i32,
}

impl<'a> FormatterVisitor<'a> {
  pub fn new(out: &'a mut dyn std::io::Write, input: &str) -> Self {
    Self {
      out,
      options: Options::default(),
      top: FormatterTop::None,
      line_length: 0,
      line_counter: 0,
      depth: 0,
      input: input.into(),
      last_char: 0,
      context_stack: vec![Context::Unknown],
      newline_style: NewlineStyle::LF,
    }
  }

  fn get_context(&self) -> Context {
    *self.context_stack.last().unwrap()
  }

  fn get_context_mut(&mut self) -> &mut Context {
    self.context_stack.last_mut().unwrap()
  }

  fn determine_collection_styling(&self) -> CollectionStyling {
    let mut styling = CollectionStyling::default();
    styling.start_line = self.line_counter;
    styling
  }

  fn get_indent_opts(&self) -> usize {
    let ctx = self.get_context();
    match ctx {
      Context::Pipeline | Context::Unknown => self.depth * self.options.indent,
      Context::Seq(s) | Context::Table(s) => {
        if let Some(i) = s.indent {
          i
        } else {
          self.depth * self.options.indent
        }
      }
    }
  }

  fn with_context<F: FnOnce(&mut Self)>(&mut self, ctx: Context, inner: F) -> Context {
    self.context_stack.push(ctx);
    inner(self);
    self.context_stack.pop().unwrap()
  }

  fn set_last_char(&mut self, ptr: usize) {
    self.last_char = ptr;
  }

  fn extract_styling(&self, until: usize) -> Option<UserStyling> {
    let from = self.last_char;
    if until <= from {
      return None;
    }

    let mut us: UserStyling = UserStyling::default();
    let mut comment_start: Option<usize> = None;
    let interpolated = &self.input[from..until];
    for (i, c) in interpolated.char_indices() {
      if c == ';' && comment_start.is_none() {
        comment_start = Some(i + from + 1);
      } else if c == '\n' {
        if let Some(start) = comment_start {
          let mut comment = &self.input[start..i + from];
          if comment.ends_with('\r') {
            comment = &comment[..comment.len() - 1];
          }
          let comment_str = comment.to_string();
          us.lines.push(UserLine::Comment(comment_str));
          comment_start = None;
        } else {
          us.lines.push(UserLine::Newline);
        }
      }
    }
    if let Some(start) = comment_start {
      let comment = &self.input[start..until];
      us.lines.push(UserLine::Comment(comment.into()));
    }
    if !us.lines.is_empty() {
      return Some(us);
    }
    None
  }

  fn interpolate(&mut self, new_pair: &Pair<Rule>) {
    self.interpolate_at_pos(new_pair.as_span().start());
  }

  // Will interpolate any user-defined comments and newline styling
  // up until the given position
  fn interpolate_at_pos(&mut self, ptr: usize) {
    self.interpolate_at_pos_ext(ptr, false);
  }

  fn interpolate_at_pos_ext(&mut self, ptr: usize, strip_final_newline: bool) {
    if let Some(us) = self.extract_styling(ptr) {
      for (i, line) in us.lines.iter().enumerate() {
        match line {
          UserLine::Newline => {
            if i == us.lines.len() - 1 && strip_final_newline {
              continue;
            }
            self.newline();
          }
          UserLine::Comment(line) => {
            self.write(&format!(";{}", line), FormatterTop::Comment);
            if i < us.lines.len() - 1 {
              self.newline();
            }
          }
        }
      }
    }
    self.set_last_char(ptr);
  }

  fn write_raw(&mut self, s: &str) {
    for c in s.chars() {
      if c == '\n' {
        self.line_length = 0;
      } else {
        self.line_length += 1;
      }
    }
    self.out.write_all(s.as_bytes()).unwrap();
  }

  fn write(&mut self, s: &str, top: FormatterTop) {
    self.write_pre_space();
    self.write_raw(s);
    self.top = top;
  }

  fn newline(&mut self) {
    let nls = self.newline_style;
    self.write_raw(nls.to_str());
    self.line_length = 0;
    self.line_counter += 1;
    for _ in 0..self.get_indent_opts() {
      self.write_raw(" ");
    }
    self.top = FormatterTop::None;
  }

  // Insert a newline before any closing bracket for tables/seq/pipelines/etc.
  // if the opening bracket is on the same line as an item, the closing bracked will also be on the same line as the last item
  fn opt_pre_closing_newline(&mut self, ctx: Context) {
    match ctx {
      Context::Seq(s) | Context::Table(s) => {
        if s.is_joined.is_none() {
          // Empty collection, no newline
          return;
        }
        if s.is_joined.is_some_and(|x| x) {
          // no newline
          return;
        }
      }
      _ => {}
    }
    self.newline();
  }

  fn get_next_atom_col(&self) -> usize {
    match self.top {
      FormatterTop::Comment => self.get_indent_opts(),
      FormatterTop::Atom => self.line_length + 1,
      _ => self.line_length,
    }
  }

  fn write_pre_space(&mut self) {
    match self.top {
      FormatterTop::None => {}
      FormatterTop::Atom => {
        self.write_raw(" ");
      }
      FormatterTop::Comment => {
        self.newline();
      }
      FormatterTop::LineFunc => {
        self.newline();
      }
    }
    self.top = FormatterTop::None;
  }

  // Write while spacing between previous atoms
  fn write_atom(&mut self, s: &str) {
    self.write(s, FormatterTop::Atom);
  }

  // Write without spacing between atomics (for symbols and such)
  fn write_joined(&mut self, s: &str) {
    match self.top {
      FormatterTop::Atom | FormatterTop::None => {
        self.write_raw(s);
      }
      FormatterTop::Comment | FormatterTop::LineFunc => {
        self.newline();
        self.write_raw(s);
      }
    }
    self.top = FormatterTop::Atom;
  }

  fn filter<'b>(&mut self, v: &'b str) -> String {
    let mut result = String::new();
    let mut comment = false;
    let mut quote_open: Option<QuoteState> = None;
    let mut quote_close: Option<QuoteState> = None;
    for (i, c) in v.chars().enumerate() {
      if !comment && c == '"' {
        match &mut quote_open {
          None => {
            quote_open = Some(QuoteState {
              start: i,
              num_quotes: 1,
            });
          }
          Some(qo) => {
            if qo.num_quotes < 3 && (i - 1) == qo.start {
              qo.num_quotes += 1;
              qo.start = i;
            } else {
              match &mut quote_close {
                None => {
                  quote_close = Some(QuoteState {
                    start: i,
                    num_quotes: 1,
                  });
                }
                Some(qc) => {
                  qc.num_quotes += 1;
                  if qc.num_quotes == qo.num_quotes {
                    quote_close = None;
                    quote_open = None;
                  }
                }
              }
            }
          }
        };
      } else {
        quote_close = None;
      }
      if quote_open.is_some() {
        result.push(c);
        continue;
      }
      if c == ';' {
        comment = true;
      }
      if c == '\n' || c == '\r' {
        comment = false;
        continue;
      }
      if c == ' ' || c == '\t' {
        continue; // Ignore whitespace
      }
      if !comment {
        result.push(c);
      }
    }
    result
  }

  fn write_func_after_open<F: FnOnce(&mut Self)>(&mut self, pair: &Pair<Rule>, inner: F) {
    let start_line = self.line_counter;
    let omit_indent = omit_shard_param_indent(pair.clone());
    if omit_indent {
      inner(self);
      self.interpolate_at_pos_ext(pair.as_span().end() - 1, true);
    } else {
      self.depth += 1;
      inner(self);
      self.interpolate_at_pos_ext(pair.as_span().end() - 1, true);
      self.depth -= 1;
    }

    if !omit_indent && start_line != self.line_counter {
      self.newline();
    }
    self.write_joined(")");
  }

  fn update_collection_first_item(&mut self) {
    let ll = self.get_next_atom_col();
    let lc = self.line_counter;
    match self.get_context_mut() {
      Context::Seq(s) | Context::Table(s) => {
        if s.indent.is_none() {
          s.indent = Some(ll);
          s.is_joined = Some(lc == s.start_line);
        }
      }
      _ => {}
    }
  }
}

// Checks if this is a function without any parameters
fn omit_shard_params(func: Pair<Rule>) -> bool {
  let mut inner = func.clone().into_inner();
  let _name = inner.next().unwrap();
  if let Some(params) = inner.next() {
    let params: pest::iterators::Pairs<'_, Rule> = params.into_inner();
    let num_params = params.len();
    if num_params == 0 {
      return true;
    }
  } else {
    return true;
  }
  return false;
}

fn omit_builtin_params(func: Pair<Rule>) -> bool {
  let mut inner = func.clone().into_inner();
  let _name = inner.next().unwrap();
  if let Some(_) = inner.next() {
    return false;
  }
  return true;
}

// Checks if all param starting points are on the same line as the function name
// Examples are:
// Once({... , If(Something {..., etc.
fn omit_shard_param_indent(func: Pair<Rule>) -> bool {
  let mut inner = func.clone().into_inner();
  let _name = inner.next().unwrap();
  let start_line = _name.line_col().0;
  if let Some(params) = inner.next() {
    let params = params.into_inner();
    let end_line = params
      .clone()
      .last()
      .map(|x| x.as_span().end_pos().line_col().0)
      .unwrap_or(start_line);
    for param in params {
      let mut param_inner = param.into_inner();
      let v = if param_inner.len() == 2 {
        param_inner.next();
        param_inner.next()
      } else {
        param_inner.next()
      }
      .unwrap()
      .into_inner()
      .next()
      .unwrap();

      let col = v.line_col().0;
      if col != start_line && col != end_line {
        return false;
      }
    }
  }
  return true;
}

impl<'a> RuleVisitor for FormatterVisitor<'a> {
  fn v_pipeline<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, inner: T) {
    self.with_context(Context::Pipeline, |s| {
      s.interpolate(&pair);
      inner(s);
      // s.interpolate_at_pos(pair.as_span().end())
    });
  }
  fn v_stmt<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, inner: T) {
    self.interpolate(&pair);
    inner(self);
  }
  fn v_value(&mut self, pair: Pair<Rule>) {
    self.interpolate(&pair);
    self.update_collection_first_item();

    let ctx = self.get_context();
    if self.top == FormatterTop::Atom {
      if let Context::Pipeline = ctx {
        self.write_atom("|");
      }
    }

    let inner = pair.clone().into_inner();
    if inner.len() > 0 {
      self.write_atom(inner.as_str());
      let last = inner.last().unwrap();
      self.set_last_char(last.as_span().end());
    } else {
      self.write_atom(pair.as_str());
      self.set_last_char(pair.as_span().end());
    }
  }
  fn v_assign<T: FnOnce(&mut Self)>(
    &mut self,
    pair: Pair<Rule>,
    op: Pair<Rule>,
    to: Pair<Rule>,
    inner: T,
  ) {
    self.interpolate(&pair);
    inner(self);
    let op = self.filter(op.as_str());
    let to = self.filter(to.as_str());
    self.write_atom(&format!("{} {}", op, to));
  }
  fn v_func<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, name: Pair<Rule>, inner: T) {
    let ctx = self.get_context();
    self.with_context(Context::Unknown, |_self| {
      _self.interpolate_at_pos(pair.as_span().start());

      if _self.top == FormatterTop::Atom {
        if let Context::Pipeline = ctx {
          _self.write_atom("|");
        }
      }

      let name_str = _self.filter(name.as_str());
      match pair.as_rule() {
        Rule::Func => {
          if omit_builtin_params(pair.clone()) {
            // This is most likely a @define
            _self.write_atom(&format!("@{}", name_str));
          } else {
            _self.write(&format!("@{}(", name_str), FormatterTop::None);
            _self.write_func_after_open(&pair, inner);
            if name_str == "wire"
              || name_str == "define"
              || name_str == "template"
              || name_str == "mesh"
              || name_str == "schedule"
            {
              _self.top = FormatterTop::LineFunc;
            }
          }
        }
        _ => {
          if omit_shard_params(pair.clone()) {
            _self.write_atom(&format!("{}", name_str));
          } else {
            _self.write(&format!("{}(", name_str), FormatterTop::None);
            _self.write_func_after_open(&pair, inner);
          }
        }
      }

      let last = pair.clone().into_inner().last().unwrap();
      let last = if last.as_rule() == Rule::VarName {
        // In case this is an argumentless builtin '@something', expand the identifier to get the last token before whitespace
        last.into_inner().last().unwrap()
      } else {
        last
      };
      _self.set_last_char(last.as_span().end());
    });
  }

  fn v_param<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, kw: Option<Pair<Rule>>, inner: T) {
    self.with_context(Context::Unknown, |_self| {
      _self.interpolate(&pair);
      if let Some(kw) = kw {
        let kw_str = _self.filter(kw.as_str());
        _self.write_atom(&format!("{}", kw_str));
        _self.interpolate_at_pos(kw.as_span().end());
      }
      inner(_self);
    });
  }
  fn v_expr<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, inner_pair: Pair<Rule>, inner: T) {
    self.interpolate(&pair);
    self.write("(", FormatterTop::None);
    let starting_line = self.line_counter;
    self.depth += 1;
    {
      inner(self);
      self.interpolate_at_pos_ext(inner_pair.as_span().end(), true);
    }
    self.depth -= 1;
    if self.line_counter != starting_line {
      self.newline();
    }
    self.write_joined(")");
  }
  fn v_eval_expr<T: FnOnce(&mut Self)>(
    &mut self,
    pair: Pair<Rule>,
    inner_pair: Pair<Rule>,
    inner: T,
  ) {
    self.interpolate(&pair);
    self.write("#(", FormatterTop::None);
    let starting_line = self.line_counter;
    self.depth += 1;
    {
      inner(self);
      self.interpolate_at_pos_ext(inner_pair.as_span().end(), true);
    }
    self.depth -= 1;
    if self.line_counter != starting_line {
      self.newline();
    }
    self.write_joined(")");
  }
  fn v_shards<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, inner: T) {
    self.with_context(Context::Pipeline, |_self| {
      let start = pair.as_span().start();
      _self.interpolate_at_pos(start);
      _self.write("{", FormatterTop::None);
      let starting_line = _self.line_counter;

      _self.depth += 1;
      {
        inner(_self);
        let end = pair.as_span().end();
        _self.interpolate_at_pos_ext(end, true);
      }
      _self.depth -= 1;
      if _self.line_counter != starting_line {
        _self.newline();
      }
      _self.write_joined("}");
    });
  }
  fn v_table<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, inner: T) {
    self.interpolate(&pair);
    self.write("{", FormatterTop::None);
    let ctx = Context::Table(self.determine_collection_styling());
    let ctx = self.with_context(ctx, |_self| {
      _self.depth += 1;
      inner(_self);
      _self.interpolate_at_pos_ext(pair.as_span().end(), true);
      _self.depth -= 1;
    });
    self.opt_pre_closing_newline(ctx);
    self.write_joined("}");
  }
  fn v_seq<T: FnOnce(&mut Self)>(&mut self, pair: Pair<Rule>, inner: T) {
    self.interpolate(&pair);
    self.write("[", FormatterTop::None);

    let ctx = Context::Seq(self.determine_collection_styling());
    let ctx = self.with_context(ctx, |_self| {
      _self.depth += 1;
      inner(_self);
      _self.interpolate_at_pos_ext(pair.as_span().end(), true);
      _self.depth -= 1;
    });
    self.opt_pre_closing_newline(ctx);
    self.write_joined("]");
  }
  fn v_table_val<TK: FnOnce(&mut Self), TV: FnOnce(&mut Self)>(
    &mut self,
    pair: Pair<Rule>,
    _key: Pair<Rule>,
    inner_key: TK,
    inner_val: TV,
  ) {
    self.interpolate(&_key);
    inner_key(self);
    self.write_joined(":");
    self.interpolate(&pair);
    inner_val(self);
  }
  fn v_take_seq(&mut self, pair: Pair<Rule>) {
    let str = self.filter(pair.as_str());
    self.write_atom(&str);
  }
  fn v_take_table(&mut self, pair: Pair<Rule>) {
    let str = self.filter(pair.as_str());
    self.write_atom(&str);
  }
  fn v_end(&mut self, pair: Pair<Rule>) {
    // Manually done to measure final newline
    let mut has_final_newline = false;
    if let Some(us) = self.extract_styling(pair.as_span().end()) {
      for (i, line) in us.lines.iter().enumerate() {
        match line {
          UserLine::Newline => {
            if i == us.lines.len() - 1 {
              continue;
            }
            self.newline();
            has_final_newline = true;
          }
          UserLine::Comment(line) => {
            self.write(&format!(";{}", line), FormatterTop::Comment);
            if i < us.lines.len() - 1 {
              self.newline();
              has_final_newline = true;
            } else {
              has_final_newline = false;
            }
          }
        }
      }
    }

    // Make sure to append final newline
    if !has_final_newline {
      let nls = self.newline_style;
      self.write_raw(nls.to_str());
    }
  }
}

pub fn detect_newline_style(input: &str) -> NewlineStyle {
  // Find at least a few lines to determine newline style
  let mut weights: [u32; 2] = [0, 0];
  let mut total = 0;
  let mut chars = input.chars().peekable();

  while let Some(c) = chars.next() {
    if c == '\n' {
      weights[0] += 1;
      total += 1;
    } else if c == '\r' {
      if chars.peek() == Some(&'\n') {
        weights[1] += 1;
        chars.next(); // consume the '\n'
        total += 1;
      } else {
        weights[0] += 1; // treat lone '\r' as LF
        total += 1;
      }
    }

    if total >= 3 {
      break;
    }
  }

  // Sort weights
  let mut weight_idx = [0, 1];
  weight_idx.sort_by_key(|&i| weights[i]);
  NewlineStyle::from_index(weight_idx[1])
}

pub fn format_str(input: &str) -> Result<String, crate::error::Error> {
  let mut buf = std::io::BufWriter::new(Vec::new());

  let newline_style = detect_newline_style(input);
  // eprintln!("newline_style: {:?}", newline_style);

  // Fix ending newline
  let mut in_str_buf: String;
  let input_ref = if input.ends_with('\n') || input.ends_with('\r') {
    input
  } else {
    in_str_buf = String::new();
    in_str_buf.reserve(input.len() + 4);
    in_str_buf.push_str(input);
    newline_style.push_to_str(&mut in_str_buf);
    &in_str_buf
  };

  let mut v = FormatterVisitor::new(&mut buf, &input_ref);
  v.newline_style = newline_style;
  crate::rule_visitor::process(&input_ref, &mut v)?;

  Ok(String::from_utf8(buf.into_inner()?)?)
}

fn strequal_ignore_line_endings(a: &str, b: &str) -> bool {
  let a = a.replace("\r\n", "\n");
  let b = b.replace("\r\n", "\n");
  a == b
}

fn strequal(a: &str, b: &str) -> bool {
  a == b
}

fn format_file_validate(input_path: &Path) -> Result<(), crate::error::Error> {
  let input_str = String::from_utf8(fs::read(input_path)?)?;
  let expected = input_path.with_extension("expected.shs");

  if let Ok(f) = File::open(expected.clone()) {
    eprintln!("Validating test {:?} against {:?}", input_path, expected);

    let mut f = f;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let expected_str = String::from_utf8(buf)?;

    let formatted = format_str(&input_str)?;
    if !strequal(&formatted, &expected_str) {
      return Err(format!("Test output does not match, was: {}", formatted).into());
    }

    let reformatted = format_str(&formatted)?;
    if !strequal(&reformatted, &formatted) {
      return Err(
        format!(
          "Formatted output changes after formatting twice: {}",
          reformatted
        )
        .into(),
      );
    }
  } else {
    eprintln!("Generating expected result for test {:?}", input_path);
    let formatted = format_str(&input_str)?;
    fs::write(expected, formatted)?;
  }

  Ok(())
}

pub fn run_tests() -> Result<(), crate::error::Error> {
  let mut any_failure = false;
  let test_files = fs::read_dir(".")?;
  for file in test_files {
    let p = &file?.path();
    if p
      .file_name()
      .is_some_and(|x| x.to_str().unwrap().ends_with(".expected.shs"))
    {
      continue;
    }

    if !p
      .extension()
      .is_some_and(|x| x.to_ascii_lowercase() == "shs")
    {
      continue;
    }

    if let Err(e) = format_file_validate(p) {
      eprintln!("Test {:?} failed: {}", p, e);
      any_failure = true
    }
  }

  if any_failure {
    Err("Test failures".into())
  } else {
    Ok(())
  }
}
