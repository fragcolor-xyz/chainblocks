/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

use crate::core::log;
use crate::core::registerShard;
use crate::shard::Shard;
use crate::types::common_type;
use crate::types::ClonedVar;
use crate::types::Context;
use crate::types::OptionalString;
use crate::types::ParamVar;
use crate::types::Parameters;
use crate::types::Seq;
use crate::types::Table;
use crate::types::Type;
use crate::types::BOOL_TYPES_SLICE;
use crate::types::SEQ_OF_STRINGS_TYPES;
use crate::types::STRING_TYPES;
use crate::types::STRING_TYPES_SLICE;
use crate::CString;
use crate::Types;
use crate::Var;
use core::convert::TryFrom;
use core::convert::TryInto;
use ext_csv::Reader;
use ext_csv::ReaderBuilder;
use ext_csv::WriterBuilder;

lazy_static! {
  static ref PARAMETERS: Parameters = vec![
    (
      cstr!("NoHeader"),
      shccstr!("Whether the shard should parse the first row as data, instead of header."),
      BOOL_TYPES_SLICE
    )
      .into(),
    (
      cstr!("Separator"),
      shccstr!("The character to use as fields separator."),
      STRING_TYPES_SLICE
    )
      .into()
  ];
  static ref STR_HELP: OptionalString =
    OptionalString(shccstr!("A multiline string in CSV format."));
  static ref SEQ_HELP: OptionalString = OptionalString(shccstr!(
    "A sequence of rows, with each row being a sequence of strings."
  ));
}

struct CSVRead {
  output: Seq,
  no_header: bool,
  separator: CString,
}

impl Default for CSVRead {
  fn default() -> Self {
    CSVRead {
      output: Seq::new(),
      no_header: false,
      separator: CString::new(",").unwrap(),
    }
  }
}

impl Shard for CSVRead {
  fn registerName() -> &'static str {
    cstr!("CSV.Read")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("CSV.Read-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "CSV.Read"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "Reads a CSV string and outputs the data as a sequence of strings in a sequence of rows."
    ))
  }

  fn inputHelp(&mut self) -> OptionalString {
    *STR_HELP
  }

  fn outputHelp(&mut self) -> OptionalString {
    *SEQ_HELP
  }

  fn inputTypes(&mut self) -> &std::vec::Vec<Type> {
    &STRING_TYPES
  }

  fn outputTypes(&mut self) -> &std::vec::Vec<Type> {
    &SEQ_OF_STRINGS_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.no_header = value.try_into()?),
      1 => Ok(self.separator = value.try_into()?),
      _ => unreachable!(),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.no_header.into(),
      1 => self.separator.as_ref().into(),
      _ => unreachable!(),
    }
  }

  fn activate(&mut self, _: &Context, input: &Var) -> Result<Var, &str> {
    let text: &str = input.try_into()?;
    let mut reader = ReaderBuilder::new()
      .has_headers(!self.no_header)
      .delimiter(self.separator.as_bytes()[0] as u8)
      .from_reader(text.as_bytes());
    for row in reader.records() {
      let record = row.map_err(|e| {
        shlog!("{}", e);
        "Failed to parse CSV"
      })?;
      let mut output = Seq::new();
      for field in record.iter() {
        let field = CString::new(field).map_err(|e| {
          shlog!("{}", e);
          "Failed to parse CSV record string"
        })?;
        output.push(&field.as_ref().into());
      }
      self.output.push(&output.as_ref().into());
    }
    Ok(self.output.as_ref().into())
  }
}

struct CSVWrite {
  output: ClonedVar,
  no_header: bool,
  separator: CString,
}

impl Default for CSVWrite {
  fn default() -> Self {
    CSVWrite {
      output: ().into(),
      no_header: false,
      separator: CString::new(",").unwrap(),
    }
  }
}

impl Shard for CSVWrite {
  fn registerName() -> &'static str {
    cstr!("CSV.Write")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("CSV.Write-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "CSV.Write"
  }

  fn help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "Reads a sequence of strings in a sequence of rows and outputs the data as a CSV string."
    ))
  }

  fn inputHelp(&mut self) -> OptionalString {
    *SEQ_HELP
  }

  fn outputHelp(&mut self) -> OptionalString {
    *STR_HELP
  }

  fn inputTypes(&mut self) -> &Vec<Type> {
    &SEQ_OF_STRINGS_TYPES
  }

  fn outputTypes(&mut self) -> &Vec<Type> {
    &STRING_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => Ok(self.no_header = value.try_into().unwrap()),
      1 => Ok(self.separator = value.try_into().unwrap()),
      _ => unreachable!(),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.no_header.into(),
      1 => self.separator.as_ref().into(),
      _ => unreachable!(),
    }
  }

  fn activate(&mut self, _: &Context, input: &Var) -> Result<Var, &str> {
    let mut writer = WriterBuilder::new()
      .has_headers(!self.no_header)
      .delimiter(self.separator.as_bytes()[0] as u8)
      .from_writer(Vec::new());
    let input: Seq = input.try_into()?;
    for row in input.iter() {
      let row: Seq = row.as_ref().try_into()?;
      let mut record = Vec::new();
      for field in row.iter() {
        let field: &str = field.as_ref().try_into()?;
        record.push(field);
      }
      writer.write_record(&record).map_err(|e| {
        shlog!("{}", e);
        "Failed to write to CSV"
      })?;
    }
    let output = writer.into_inner().map_err(|e| {
      shlog!("{}", e);
      "Failed to write to CSV"
    })?;
    let output = CString::new(output).map_err(|e| {
      shlog!("{}", e);
      "Failed to write to CSV"
    })?;
    let output: Var = output.as_ref().into();
    self.output.assign(&output);
    Ok(self.output.0)
  }
}

pub fn registerShards() {
  registerShard::<CSVRead>();
  registerShard::<CSVWrite>();
}
