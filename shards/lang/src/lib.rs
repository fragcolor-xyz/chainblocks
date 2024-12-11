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

use std::borrow::Cow;
use std::ops::Deref;

use shards::types::{AutoShardRef, ClonedVar};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct RcBytesWrapper(Rc<Cow<'static, [u8]>>);

impl Serialize for RcBytesWrapper {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    serializer.serialize_bytes(&self.0)
  }
}

impl<'de> Deserialize<'de> for RcBytesWrapper {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
    Ok(RcBytesWrapper(Rc::new(Cow::Owned(bytes))))
  }
}

impl RcBytesWrapper {
  pub fn new<S: Into<Cow<'static, [u8]>>>(s: S) -> Self {
    RcBytesWrapper(Rc::new(s.into()))
  }

  pub fn to_vec(&self) -> Vec<u8> {
    self.0.to_vec()
  }

  pub fn as_slice(&self) -> &[u8] {
    &self.0
  }

  pub fn to_mut(&mut self) -> &mut Vec<u8> {
    let cow = Rc::make_mut(&mut self.0);
    cow.to_mut()
  }
}

impl From<&'static [u8]> for RcBytesWrapper {
  fn from(s: &'static [u8]) -> Self {
    RcBytesWrapper::new(Cow::Borrowed(s))
  }
}

impl From<Vec<u8>> for RcBytesWrapper {
  fn from(s: Vec<u8>) -> Self {
    RcBytesWrapper::new(Cow::Owned(s))
  }
}

impl PartialEq for RcBytesWrapper {
  fn eq(&self, other: &RcBytesWrapper) -> bool {
    self.0 == other.0
  }
}

impl Eq for RcBytesWrapper {}

impl PartialEq<[u8]> for RcBytesWrapper {
  fn eq(&self, other: &[u8]) -> bool {
    *self.0 == other
  }
}

impl Hash for RcBytesWrapper {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.hash(state)
  }
}

impl fmt::Display for RcBytesWrapper {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:?}", self.0)
  }
}

impl Deref for RcBytesWrapper {
  type Target = [u8];
  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

#[derive(Debug, Clone)]
pub struct RcStrWrapper(Rc<Cow<'static, str>>);

impl Serialize for RcStrWrapper {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    serializer.serialize_str(&self.0)
  }
}

impl<'de> Deserialize<'de> for RcStrWrapper {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let s = String::deserialize(deserializer)?;
    Ok(RcStrWrapper(Rc::new(Cow::Owned(s))))
  }
}

impl RcStrWrapper {
  pub fn new<S: Into<Cow<'static, str>>>(s: S) -> Self {
    RcStrWrapper(Rc::new(s.into()))
  }

  pub fn to_string(&self) -> String {
    self.0.to_string()
  }

  pub fn as_str(&self) -> &str {
    &self.0
  }

  pub fn to_mut(&mut self) -> &mut String {
    let cow = Rc::make_mut(&mut self.0);
    cow.to_mut()
  }
}

impl From<&'static str> for RcStrWrapper {
  fn from(s: &'static str) -> Self {
    RcStrWrapper::new(Cow::Borrowed(s))
  }
}

impl From<String> for RcStrWrapper {
  fn from(s: String) -> Self {
    RcStrWrapper::new(Cow::Owned(s))
  }
}

impl Eq for RcStrWrapper {}

impl PartialEq<RcStrWrapper> for RcStrWrapper {
  fn eq(&self, other: &RcStrWrapper) -> bool {
    self.0 == other.0
  }
}

impl PartialEq<str> for RcStrWrapper {
  fn eq(&self, other: &str) -> bool {
    *self.0 == other
  }
}

impl Hash for RcStrWrapper {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.hash(state)
  }
}

impl fmt::Display for RcStrWrapper {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl Deref for RcStrWrapper {
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
