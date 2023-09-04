/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

use shards::core::register_legacy_shard;
use shards::shard::LegacyShard;
use shards::types::common_type;
use shards::types::ClonedVar;
use shards::types::Context;
use shards::types::ParamVar;
use shards::types::Parameters;
use shards::types::Seq;

use shards::types::Type;
use shards::types::BOOL_TYPES_SLICE;
use shards::types::STRING_OR_NONE_SLICE;
use shards::types::STRING_VAR_OR_NONE_SLICE;

use shards::types::Var;

use ethabi::Contract;
use ethabi::ParamType;
use ethabi::Token;
use ethereum_types::U256;
use std::convert::TryInto;

lazy_static! {
  static ref INPUT_TYPES: Vec<Type> = vec![common_type::anys];
  static ref DECODE_INPUT_TYPES: Vec<Type> = vec![common_type::bytes, common_type::string];
  static ref OUTPUT_TYPES: Vec<Type> = vec![common_type::bytes];
  static ref DECODE_OUTPUT_TYPES: Vec<Type> = vec![common_type::anys];
  static ref PARAMETERS: Parameters = vec![
    (
      cstr!("ABI"),
      shccstr!("The contract's json ABI."),
      STRING_VAR_OR_NONE_SLICE
    )
      .into(),
    (
      cstr!("Name"),
      shccstr!("The name of the method to call."),
      STRING_OR_NONE_SLICE
    )
      .into()
  ];
}

fn var_to_token(var: &Var, param_type: &ParamType) -> Result<Token, &'static str> {
  match param_type {
    ParamType::Uint(size) => {
      if *size > 64 && *size <= 256 {
        let bytes: &[u8] = var.try_into()?;
        let u: U256 = bytes.into();
        let u: [u8; 32] = u.into();
        Ok(Token::Uint(u.into()))
      } else if *size <= 64 {
        let uint: u64 = var.try_into()?;
        Ok(Token::Uint(uint.into()))
      } else {
        Err("Invalid Uint size")
      }
    }
    ParamType::Address => {
      let s: String = var.try_into()?;
      Ok(Token::Address(
        s.parse().map_err(|_| "Invalid address string")?,
      ))
    }
    ParamType::Bytes => {
      let bytes: &[u8] = var.try_into()?;
      Ok(Token::Bytes(bytes.to_vec()))
    }
    ParamType::FixedBytes(size) => {
      let bytes: &[u8] = var.try_into()?;
      let mut u = Vec::new();
      u.resize(*size, 0);
      u[..bytes.len()].clone_from_slice(&bytes[..bytes.len()]);
      Ok(Token::FixedBytes(u))
    }
    ParamType::String => {
      let s: String = var.try_into()?;
      Ok(Token::String(s))
    }
    ParamType::Bool => {
      let b: bool = var.try_into()?;
      Ok(Token::Bool(b))
    }
    ParamType::Array(inner_type) => {
      if !var.is_seq() {
        Err("Expected a sequence")
      } else {
        let mut tokens = Vec::new();
        let s: Seq = var.try_into()?;
        for v in s.iter() {
          tokens.push(var_to_token(&v, inner_type)?);
        }
        Ok(Token::Array(tokens))
      }
    }
    _ => Err("Failed to convert Var into Token - matched case not implemented"),
  }
}

fn token_to_var(token: Token) -> Result<ClonedVar, &'static str> {
  match token {
    Token::Uint(uint) => {
      let bytes: [u8; 32] = uint.into();
      Ok(bytes[..].into())
    }
    Token::Address(address) => {
      let hex = address.as_bytes();
      let mut hex = hex::encode(hex);
      hex.insert_str(0, "0x");
      let s = Var::ephemeral_string(hex.as_str());
      Ok(s.into()) // Clone
    }
    Token::Bytes(bytes) | Token::FixedBytes(bytes) => Ok(bytes.as_slice().into()),
    Token::Array(tokens) | Token::FixedArray(tokens) => {
      let mut s = Seq::new();
      for token in tokens {
        let v = token_to_var(token)?;
        s.push(&v.0);
      }
      Ok(s.as_ref().into())
    }
    _ => Err("Failed to convert Token into Var - matched case not implemented"),
  }
}

#[derive(Default)]
struct EncodeCall {
  abi: ParamVar,
  call_name: ParamVar,
  current_abi: Option<ClonedVar>,
  contract: Option<Contract>,
  output: Vec<u8>,
  input: Vec<Token>,
}

impl LegacyShard for EncodeCall {
  fn registerName() -> &'static str {
    cstr!("Eth.EncodeCall")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("Eth.EncodeCall-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "Eth.EncodeCall"
  }

  fn inputTypes(&mut self) -> &std::vec::Vec<Type> {
    &INPUT_TYPES
  }

  fn outputTypes(&mut self) -> &std::vec::Vec<Type> {
    &OUTPUT_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => self.abi.set_param(value),
      1 => self.call_name.set_param(value),
      _ => unreachable!(),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.abi.get_param(),
      1 => self.call_name.get_param(),
      _ => unreachable!(),
    }
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.abi.warmup(context);
    self.call_name.warmup(context);
    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.abi.cleanup();
    self.call_name.cleanup();
    Ok(())
  }

  fn activate(&mut self, _: &Context, input: &Var) -> Result<Var, &str> {
    let abi = self.abi.get();
    let call_name = self.call_name.get();

    // process abi, create contract if needed
    if let Some(current_abi) = &self.current_abi {
      // abi might change on the fly, so we need to check it
      if self.abi.is_variable() && *abi != current_abi.0 {
        self.contract = serde_json::from_str(abi.try_into()?).map_err(|e| {
          shlog!("{}", e);
          "Failed to parse abi json string"
        })?;
        self.current_abi = Some(abi.into());
      }
    } else {
      self.contract = serde_json::from_str(abi.try_into()?).map_err(|e| {
        shlog!("{}", e);
        "Failed to parse abi json string"
      })?;
      self.current_abi = Some(abi.into());
    }

    // encode the call
    if let Some(contract) = &self.contract {
      let name: String = call_name.try_into()?;
      let func = if let Some(func) = &contract.functions.get(&name) {
        if !func.is_empty() {
          Ok(&func[0])
        } else {
          Err("Function not found")
        }
      } else {
        Err("Function not found")
      }?;

      let inputs: Seq = input.try_into()?;
      if inputs.len() != func.inputs.len() {
        return Err("Invalid number of input parameters");
      }

      self.input.clear();

      for (idx, input_type) in func.inputs.iter().enumerate() {
        let value = inputs[idx];
        self.input.push(var_to_token(&value, &input_type.kind)?);
      }

      self.output = func.encode_input(self.input.as_slice()).map_err(|e| {
        shlog!("{}", e);
        "Failed to encode call input"
      })?;
      Ok(self.output.as_slice().into())
    } else {
      Err("Contract is missing")
    }
  }
}

lazy_static! {
  static ref DECODE_PARAMETERS: Parameters = {
    let mut v = vec![(
      cstr!("Input"),
      shccstr!("If the input is the actual function call transaction input rather than the result of the call."),
      BOOL_TYPES_SLICE
    )
      .into()];
    v.insert(0, (*PARAMETERS)[1]);
    v.insert(0, (*PARAMETERS)[0]);
    v
  };
}

#[derive(Default)]
struct DecodeCall {
  abi: ParamVar,
  call_name: ParamVar,
  current_abi: Option<ClonedVar>,
  contract: Option<Contract>,
  output: Seq,
  is_input: bool,
}

impl LegacyShard for DecodeCall {
  fn registerName() -> &'static str {
    cstr!("Eth.DecodeCall")
  }

  fn hash() -> u32 {
    compile_time_crc32::crc32!("Eth.DecodeCall-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "Eth.DecodeCall"
  }

  fn inputTypes(&mut self) -> &std::vec::Vec<Type> {
    &DECODE_INPUT_TYPES
  }

  fn outputTypes(&mut self) -> &std::vec::Vec<Type> {
    &DECODE_OUTPUT_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&DECODE_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    match index {
      0 => self.abi.set_param(value),
      1 => self.call_name.set_param(value),
      2 => Ok(self.is_input = value.try_into()?),
      _ => unreachable!(),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.abi.get_param(),
      1 => self.call_name.get_param(),
      2 => self.is_input.into(),
      _ => unreachable!(),
    }
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.abi.warmup(context);
    self.call_name.warmup(context);
    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.abi.cleanup();
    self.call_name.cleanup();
    Ok(())
  }

  fn activate(&mut self, _: &Context, input: &Var) -> Result<Var, &str> {
    let abi = self.abi.get();
    let call_name = self.call_name.get();

    // process abi, create contract if needed
    if let Some(current_abi) = &self.current_abi {
      // abi might change on the fly, so we need to check it
      if self.abi.is_variable() && *abi != current_abi.0 {
        self.contract = serde_json::from_str(abi.try_into()?).map_err(|e| {
          shlog!("{}", e);
          "Failed to parse abi json string"
        })?;
        self.current_abi = Some(abi.into());
      }
    } else {
      self.contract = serde_json::from_str(abi.try_into()?).map_err(|e| {
        shlog!("{}", e);
        "Failed to parse abi json string"
      })?;
      self.current_abi = Some(abi.into());
    }

    // encode the call
    if let Some(contract) = &self.contract {
      let name: String = call_name.try_into()?;
      let func = if let Some(func) = &contract.functions.get(&name) {
        if !func.is_empty() {
          Ok(&func[0])
        } else {
          Err("Function not found")
        }
      } else {
        Err("Function not found")
      }?;

      let decoded = {
        let str_input: Result<&str, &str> = input.try_into();
        if self.is_input {
          if let Ok(str_input) = str_input {
            let bytes = hex::decode(str_input.trim_start_matches("0x").trim_start_matches("0X"))
              .map_err(|e| {
                shlog!("{}", e);
                "Failed to parse input hex string"
              })?;
            func.decode_input(bytes.as_slice()).map_err(|e| {
              shlog!("{}", e);
              "Failed to parse input bytes"
            })
          } else {
            let input: &[u8] = input.try_into()?;
            func.decode_input(input).map_err(|e| {
              shlog!("{}", e);
              "Failed to parse input bytes"
            })
          }
        } else if let Ok(str_input) = str_input {
          let bytes = hex::decode(str_input.trim_start_matches("0x").trim_start_matches("0X"))
            .map_err(|e| {
              shlog!("{}", e);
              "Failed to parse input hex string"
            })?;
          func.decode_output(bytes.as_slice()).map_err(|e| {
            shlog!("{}", e);
            "Failed to parse input bytes"
          })
        } else {
          let input: &[u8] = input.try_into()?;
          func.decode_output(input).map_err(|e| {
            shlog!("{}", e);
            "Failed to parse input bytes"
          })
        }
      }?;

      self.output.clear();

      for token in decoded {
        let value: ClonedVar = token_to_var(token)?;
        self.output.push(&value.0);
      }

      Ok(self.output.as_ref().into())
    } else {
      Err("Contract is missing")
    }
  }
}

pub fn register_shards() {
  register_legacy_shard::<EncodeCall>();
  register_legacy_shard::<DecodeCall>();
}
