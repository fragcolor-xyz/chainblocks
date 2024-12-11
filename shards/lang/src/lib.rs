extern crate pest;
#[macro_use]
extern crate pest_derive;

extern crate clap;

pub mod ast;
pub mod ast_visitor;
pub mod cli;
pub mod custom_state;
pub mod directory;
mod error;
pub mod eval;
mod formatter;
pub mod print;
pub mod read;
pub mod rule_visitor;

use crate::ast::*;

use core::fmt;

use std::ops::Deref;

use shards::types::{AutoShardRef, ClonedVar};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BytesWrapper(Vec<u8>);

impl BytesWrapper {
  /// Creates a new `BytesWrapper` from a `Vec<u8>`.
  pub fn new(bytes: Vec<u8>) -> Self {
    BytesWrapper(bytes)
  }

  /// Converts the `BytesWrapper` into the inner `Vec<u8>`.
  pub fn into_inner(self) -> Vec<u8> {
    self.0
  }

  /// Returns a reference to the inner byte slice.
  pub fn as_slice(&self) -> &[u8] {
    &self.0
  }

  /// Returns a mutable reference to the inner `Vec<u8>`.
  pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
    &mut self.0
  }

  /// Converts the `BytesWrapper` into a `Vec<u8>`.
  pub fn to_vec(&self) -> Vec<u8> {
    self.0.clone()
  }
}

impl From<Vec<u8>> for BytesWrapper {
  fn from(bytes: Vec<u8>) -> Self {
    BytesWrapper::new(bytes)
  }
}

impl From<&'static [u8]> for BytesWrapper {
  fn from(bytes: &'static [u8]) -> Self {
    BytesWrapper::new(bytes.to_vec())
  }
}

impl Serialize for BytesWrapper {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    serializer.serialize_bytes(&self.0)
  }
}

impl<'de> Deserialize<'de> for BytesWrapper {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
    Ok(BytesWrapper::new(bytes))
  }
}

impl fmt::Display for BytesWrapper {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // Display the bytes in hexadecimal format for readability
    for byte in &self.0 {
      write!(f, "{:02x}", byte)?;
    }
    Ok(())
  }
}

impl Deref for BytesWrapper {
  type Target = [u8];

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StrWrapper(String);

impl StrWrapper {
  pub fn new(s: String) -> Self {
    StrWrapper(s)
  }

  pub fn into_inner(self) -> String {
    self.0
  }

  pub fn as_str(&self) -> &str {
    &self.0
  }

  pub fn as_mut_str(&mut self) -> &mut String {
    &mut self.0
  }

  pub fn to_string(&self) -> String {
    self.0.clone()
  }
}

impl From<String> for StrWrapper {
  fn from(s: String) -> Self {
    StrWrapper::new(s)
  }
}

impl From<&str> for StrWrapper {
  fn from(s: &str) -> Self {
    StrWrapper::new(s.to_string())
  }
}

impl Serialize for StrWrapper {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    serializer.serialize_str(&self.0)
  }
}

impl<'de> Deserialize<'de> for StrWrapper {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let s = String::deserialize(deserializer)?;
    Ok(StrWrapper::new(s))
  }
}

impl std::fmt::Display for StrWrapper {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl std::ops::Deref for StrWrapper {
  type Target = str;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

pub struct ParamHelper<'a> {
  params: &'a [Param],
}

impl<'a> ParamHelper<'a> {
  pub fn new(params: &'a [Param]) -> Self {
    Self { params }
  }

  pub fn get_param_by_name_or_index(&self, param_name: &str, index: usize) -> Option<&'a Param> {
    let named_param_encountered = self.params.iter().take(index + 1).any(|p| p.name.is_some());

    if named_param_encountered {
      // If we've encountered a named parameter up to this index, we only look for the parameter by name
      self
        .params
        .iter()
        .find(|param| param.name.as_deref() == Some(param_name))
    } else if index < self.params.len() {
      // If no named parameters encountered and index is valid, return the parameter at that index
      Some(&self.params[index])
    } else {
      // If index is out of bounds, we look for a parameter with the given name
      self
        .params
        .iter()
        .find(|param| param.name.as_deref() == Some(param_name))
    }
  }
}

pub struct ParamHelperMut<'a> {
  params: &'a mut [Param],
}

impl<'a> ParamHelperMut<'a> {
  pub fn new(params: &'a mut [Param]) -> Self {
    Self { params }
  }

  pub fn get_param_by_name_or_index_mut(
    &mut self,
    param_name: &str,
    index: usize,
  ) -> Option<&mut Param> {
    let named_param_encountered = self.params.iter().take(index + 1).any(|p| p.name.is_some());

    if named_param_encountered {
      // If we've encountered a named parameter up to this index, we only look for the parameter by name
      self
        .params
        .iter_mut()
        .find(|param| param.name.as_deref() == Some(param_name))
    } else if index < self.params.len() {
      // If no named parameters encountered and index is valid, return the parameter at that index
      Some(&mut self.params[index])
    } else {
      // If index is out of bounds, we look for a parameter with the given name
      self
        .params
        .iter_mut()
        .find(|param| param.name.as_deref() == Some(param_name))
    }
  }
}

pub trait ShardsExtension {
  fn name(&self) -> &str;
  fn process_to_var(
    &mut self,
    func: &Function,
    line_info: LineInfo,
  ) -> Result<ClonedVar, ShardsError>;
  fn process_to_shard(
    &mut self,
    func: &Function,
    line_info: LineInfo,
  ) -> Result<AutoShardRef, ShardsError>;
}
