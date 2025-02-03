/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2020 Fragcolor Pte. Ltd. */

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[macro_use]
extern crate shards;

#[macro_use]
extern crate lazy_static;

use reqwest::RequestBuilder;
use reqwest::Response;
use shards::core::register_legacy_shard;
use shards::core::register_shard;
use shards::core::run_future;
use shards::fourCharacterCode;
use shards::shard::LegacyShard;
use shards::shard::Shard;
use shards::types::common_type;

use shards::types::AutoTableVar;
use shards::types::ClonedVar;
use shards::types::Context;
use shards::types::ExposedInfo;
use shards::types::ExposedTypes;
use shards::types::InstanceData;
use shards::types::OptionalString;
use shards::types::ParamVar;
use shards::types::Parameters;
use shards::types::RawString;
use shards::types::Table;
use shards::types::Type;
use shards::types::Types;
use shards::types::BOOL_TYPES_SLICE;
use shards::types::BYTES_TYPES;
use shards::types::FRAG_CC;
use shards::types::INT_TYPES_SLICE;
use shards::types::NONE_TYPES;

use core::time::Duration;
use shards::types::Var;

use reqwest::header::{HeaderName, HeaderValue};
use std::convert::TryInto;

use std::collections::HashMap;

use std::sync::{Arc, Mutex};

fn print_error(e: &dyn std::error::Error) {
  shlog_error!("Error: {}", e);
  let mut source = e.source();
  while let Some(e) = source {
      shlog_error!("Caused by: {}", e);
      source = e.source();
  }
}

lazy_static! {
  static ref TOKIO_RUNTIME: Arc<Mutex<tokio::runtime::Runtime>> = Arc::new(Mutex::new(
    tokio::runtime::Builder::new_multi_thread()
      .worker_threads(4)
      .enable_all()
      .build()
      .expect("Failed to create Tokio runtime")
  ));
  static ref GET_INPUT_TYPES: Vec<Type> = vec![common_type::none, common_type::string_table];
  static ref POST_INPUT_TYPES: Vec<Type> = vec![
    common_type::none,
    common_type::string_table,
    common_type::bytes,
    common_type::string
  ];
  static ref _STR_FULL_OUTPUT_TYPES: Vec<Type> = vec![
    common_type::int,
    common_type::string_table,
    common_type::string
  ];
  static ref _BYTES_FULL_OUTPUT_TYPES: Vec<Type> = vec![
    common_type::int,
    common_type::string_table,
    common_type::bytes
  ];
  static ref FULL_OUTPUT_KEYS: Vec<Var> = vec![
    shstr!("status").into(),
    shstr!("headers").into(),
    shstr!("body").into()
  ];
  static ref STR_FULL_OUTPUT_TTYPE: Type = Type::table(&FULL_OUTPUT_KEYS, &_STR_FULL_OUTPUT_TYPES);
  static ref BYTES_FULL_OUTPUT_TTYPE: Type =
    Type::table(&FULL_OUTPUT_KEYS, &_BYTES_FULL_OUTPUT_TYPES);
  static ref STREAM_TYPE: Type = Type::object(FRAG_CC, fourCharacterCode(*b"htst"));
  static ref STREAM_TYPE_VEC: Vec<Type> = vec![*STREAM_TYPE];
  static ref STREAM_TYPE_VAR: Type = Type::context_variable(&STREAM_TYPE_VEC);
  static ref ALL_OUTPUT_TYPES: Vec<Type> = vec![
    *BYTES_FULL_OUTPUT_TTYPE,
    common_type::bytes,
    *STR_FULL_OUTPUT_TTYPE,
    common_type::string,
    *STREAM_TYPE
  ];
  static ref GET_PARAMETERS: Parameters = vec![
    (cstr!("URL"), shccstr!("The url to request to."), URL_TYPES).into(),
    (
      cstr!("Headers"),
      shccstr!("If a table of headers is provided, it will be used as is; if no headers are provided, a Content-Type header will be derived based on the input type."),
      HEADERS_TYPES
    )
      .into(),
    (
      cstr!("Timeout"),
      shccstr!("How many seconds to wait for the request to complete."),
      INT_TYPES_SLICE
    )
      .into(),
    (
      cstr!("Bytes"),
      shccstr!("If instead of a string the shard should output bytes."),
      BOOL_TYPES_SLICE
    )
      .into(),
    (
      cstr!("FullResponse"),
      shccstr!(
        "If the output should be a table with the full response, including headers and status."
      ),
      BOOL_TYPES_SLICE
    )
      .into(),
    (
      cstr!("AcceptInvalidCerts"),
      shccstr!(
        "If we should ignore invalid certificates. This is useful for testing but should not be used in production."
      ),
      BOOL_TYPES_SLICE
    )
      .into(),
    (
      cstr!("Retry"),
      shccstr!("How many times to retry the request if it fails."),
      INT_TYPES_SLICE
    )
      .into(),
    (
      cstr!("KeepAlive"),
      shccstr!("If the client instance should be kept alive, allowing connection reuse for multiple requests. The client won't be closed until this shard cleans up."),
      BOOL_TYPES_SLICE
    )
      .into(),
    (
      cstr!("Streaming"),
      shccstr!("If the response should be streamed, in which case the output will be an object to use with the Http.Stream shard."),
      BOOL_TYPES_SLICE
    )
      .into(),
    (
      cstr!("Backoff"),
      shccstr!("How many seconds to wait between retries. Defaults to 1 second."),
      INT_TYPES_SLICE
    )
      .into(),
    (
      cstr!("ConnectionTimeout"),
      shccstr!("How many seconds to wait for a connection to be established. Defaults to 10 seconds."),
      INT_TYPES_SLICE
    )
      .into(),
      
  ];
}

struct OurResponse(Response);
ref_counted_object_type_impl!(OurResponse);

static URL_TYPES: &[Type] = &[common_type::string, common_type::string_var];
static HEADERS_TYPES: &[Type] = &[
  common_type::none,
  common_type::string_table,
  common_type::string_table_var,
];

type Client = reqwest::Client;

struct RequestBase {
  client: Option<Client>,
  url: ParamVar,
  headers: ParamVar,
  output: ClonedVar,
  timeout: u64,
  retry: u64,
  backoff: u64,
  as_bytes: bool,
  full_response: bool,
  invalid_certs: bool,
  keep_alive: bool,
  required: ExposedTypes,
  streaming: bool,
  connection_timeout: u64,
}

impl Default for RequestBase {
  fn default() -> Self {
    Self {
      client: None,
      url: ParamVar::new(Var::ephemeral_string("")),
      headers: ParamVar::new(().into()),
      output: ClonedVar::default(),
      timeout: 10,
      retry: 0,
      backoff: 1,
      as_bytes: false,
      full_response: false,
      invalid_certs: false,
      keep_alive: false,
      required: Vec::new(),
      streaming: false,
      connection_timeout: 10,
    }
  }
}

impl RequestBase {
  fn _outputTypes(&mut self) -> &Types {
    &ALL_OUTPUT_TYPES
  }

  fn _parameters(&mut self) -> Option<&Parameters> {
    Some(&GET_PARAMETERS)
  }

  fn _setParam(&mut self, index: i32, value: &Var) -> Result<(), &'static str> {
    match index {
      0 => self.url.set_param(value),
      1 => self.headers.set_param(value),
      2 => Ok(self.timeout = value.try_into().map_err(|_x| "Failed to set timeout")?),
      3 => Ok(self.as_bytes = value.try_into().map_err(|_x| "Failed to set as_bytes")?),
      4 => Ok(
        self.full_response = value
          .try_into()
          .map_err(|_x| "Failed to set full_response")?,
      ),
      5 => Ok(
        self.invalid_certs = value
          .try_into()
          .map_err(|_x| "Failed to set invalid_certs")?,
      ),
      6 => Ok(self.retry = value.try_into().map_err(|_x| "Failed to set retry")?),
      7 => Ok(self.keep_alive = value.try_into().map_err(|_x| "Failed to set keep_alive")?),
      8 => Ok(self.streaming = value.try_into().map_err(|_x| "Failed to set streaming")?),
      9 => Ok(self.backoff = value.try_into().map_err(|_x| "Failed to set backoff")?),
      10 => Ok(self.connection_timeout = value.try_into().map_err(|_x| "Failed to set connection_timeout")?),
      _ => unreachable!(),
    }
  }

  fn _getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.url.get_param(),
      1 => self.headers.get_param(),
      2 => self.timeout.try_into().expect("A valid integer in range"),
      3 => self.as_bytes.into(),
      4 => self.full_response.into(),
      5 => self.invalid_certs.into(),
      6 => self.retry.try_into().expect("A valid integer in range"),
      7 => self.keep_alive.into(),
      8 => self.streaming.into(),
      9 => self.backoff.try_into().expect("A valid integer in range"),
      10 => self.connection_timeout.try_into().expect("A valid integer in range"),
      _ => unreachable!(),
    }
  }

  fn _requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.required.clear();

    if self.headers.is_variable() {
      let exp_info = ExposedInfo {
        exposedType: common_type::string_table,
        name: self.headers.get_name(),
        help: shccstr!("The headers associated with the request."),
        ..ExposedInfo::default()
      };
      self.required.push(exp_info);
    }
    if self.url.is_variable() {
      let exp_info = ExposedInfo {
        exposedType: common_type::string,
        name: self.url.get_name(),
        help: shccstr!("The url to request to."),
        ..ExposedInfo::default()
      };
      self.required.push(exp_info);
    }

    Some(&self.required)
  }

  fn _open_client(&mut self) -> Result<(), &'static str> {
    if self.client.is_none() {
      self.client = Some(
        reqwest::Client::builder()
          .danger_accept_invalid_certs(self.invalid_certs)
          .read_timeout(Duration::from_secs(self.timeout))
          .connect_timeout(Duration::from_secs(self.connection_timeout))
          .build()
          .map_err(|e| {
            print_error(&e);
            "Failed to create client"
          })?,
      );
    }
    Ok(())
  }

  fn _close_client(&mut self) {
    self.client = None;
  }

  fn _warmup(&mut self, context: &Context) -> Result<(), &'static str> {
    if self.keep_alive {
      // open client early in this case
      self._open_client()?;
    }
    self.url.warmup(context);
    self.headers.warmup(context);
    Ok(())
  }

  fn _cleanup(&mut self, ctx: Option<&Context>) {
    self.url.cleanup(ctx);
    self.headers.cleanup(ctx);
    self._close_client();
    self.output = ClonedVar::default();
  }

  fn _compose(&mut self, _data: &InstanceData) -> Result<Type, &'static str> {
    let output_type = if self.streaming {
      // for now don't support full response for streaming
      if self.full_response {
        return Err("Full response not supported when streaming");
      }
      *STREAM_TYPE_VAR
    } else if self.as_bytes {
      if self.full_response {
        *BYTES_FULL_OUTPUT_TTYPE
      } else {
        common_type::bytes
      }
    } else if self.full_response {
      *STR_FULL_OUTPUT_TTYPE
    } else {
      common_type::string
    };
    Ok(output_type)
  }

  fn _finalize(&mut self, context: &Context, request: RequestBuilder) -> Result<(), &'static str> {
    let as_bytes = self.as_bytes;
    let full_response = self.full_response;
    let streaming = self.streaming;

    let result = run_future(context, async move {
      let runtime = TOKIO_RUNTIME.clone();
      let request = request; // Capture request in the async block

      // Lock the runtime briefly to spawn the task
      let task = {
        let runtime = runtime.lock().unwrap();
        runtime.spawn(async move {
          let response = request.send().await.map_err(|e| {
            print_error(&e);
            "Failed to send the request"
          })?;

          if !full_response && !response.status().is_success() {
            shlog_error!("Request failed with status {}", response.status());
            let err_text = response.text().await.map_err(|e| {
              print_error(&e);
              "Failed to decode the failure response"
            })?;
            let err_text = if err_text.len() > 1024 {
              format!("{}...", err_text.chars().take(1024).collect::<String>())
            } else {
              err_text
            };
            shlog_error!(
              "Request failed with body {}",
              err_text
            );
            return Err("Request failed");
          }

          if streaming {
            // When streaming, we return a ref counted object with the response it self
            let response_object = Var::new_ref_counted(OurResponse(response), &*STREAM_TYPE);
            return Ok(response_object.into());
          }

          let mut output_table = AutoTableVar::new();

          if full_response {
            output_table
              .0
              .insert_fast_static("status", &response.status().as_u16().into());

            let headers = output_table
              .0
              .get_mut_fast_static("headers")
              .as_mut_table_creating()
              .unwrap();
            for (key, value) in response.headers() {
              let key = Var::ephemeral_string(key.as_str());
              let value = Var::ephemeral_string(value.to_str().map_err(|e| {
                print_error(&e);
                "Failed to decode the response"
              })?);
              headers.insert_fast(key, &value);
            }
          }

          let content: ClonedVar = if as_bytes {
            let bytes = response.bytes().await.map_err(|e| {
              print_error(&e);
              "Failed to decode the response"
            })?;

            bytes.as_ref().into()
          } else {
            let str = response.text().await.map_err(|e| {
              print_error(&e);
              "Failed to decode the response"
            })?;

            let shards_str = Var::ephemeral_string(str.as_str());
            shards_str.into()
          };

          let result: ClonedVar = if full_response {
            output_table
              .0
              .insert_fast_static("body", &content.0);
            output_table.to_cloned()
          } else {
            content
          };

          Ok(result)
        })
      };

      // Await the spawned task outside the lock
      task.await.map_err(|e| {
        print_error(&e);
        "Failed to join task"
      })?
    })?;

    self.output = result;

    Ok(())
  }
}

macro_rules! get_like {
  ($shard_name:ident, $call:ident, $name_str:literal, $hash:literal) => {
    #[derive(Default)]
    struct $shard_name {
      rb: RequestBase,
      on_worker_thread: bool,
    }

    impl LegacyShard for $shard_name {
      fn registerName() -> &'static str {
        cstr!($name_str)
      }

      fn name(&mut self) -> &str {
        $name_str
      }

      fn hash() -> u32 {
        compile_time_crc32::crc32!($hash)
      }

      fn help(&mut self) -> OptionalString {
        if $name_str == "Http.Get" {
          OptionalString(shccstr!("This shard sends a GET request to the specified URL and outputs the response."))
        } else if $name_str == "Http.Head" {
          OptionalString(shccstr!("This shard sends a HEAD request to the specified URL and outputs the response."))
        } else {
          OptionalString(shccstr!("this shard sends the respective HTTP request to the specified URL and outputs the response."))
        }
      }

      fn inputHelp(&mut self) -> OptionalString {
        OptionalString(shccstr!("The input for this shard should either be none or an optional string table of query parameters to append to the URL."))
      }

      fn outputHelp(&mut self) -> OptionalString {
        if $name_str == "Http.Get" {
          OptionalString(shccstr!("The output is the response from the server through the GET request."))
        } else if $name_str == "Http.Head" {
          OptionalString(shccstr!("The output is the headers of the response from the server through the HEAD request."))
        } else {
          OptionalString(shccstr!("The output is the response from the server through the respective HTTP request."))
        }
      }

      fn inputTypes(&mut self) -> &Types {
        &GET_INPUT_TYPES
      }

      fn outputTypes(&mut self) -> &Types {
        self.rb._outputTypes()
      }

      fn parameters(&mut self) -> Option<&Parameters> {
        self.rb._parameters()
      }

      fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
        self.rb._requiredVariables()
      }

      fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &'static str> {
        self.rb._setParam(index, value)
      }

      fn getParam(&mut self, index: i32) -> Var {
        self.rb._getParam(index)
      }

      fn warmup(&mut self, context: &Context) -> Result<(), &'static str> {
        self.rb._warmup(context)
      }

      fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &'static str> {
        self.rb._cleanup(ctx);
        Ok(())
      }

      fn hasCompose() -> bool {
        true
      }

      fn compose(&mut self, data: &InstanceData) -> Result<Type, &'static str> {
        self.on_worker_thread = data.onWorkerThread;
        self.rb._compose(data)
      }

      fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
        if !self.rb.keep_alive {
          self.rb._open_client()?;
        }

        let request = self.rb.url.get();
        let request_string: &str = request.try_into()?;
        let mut request = self.rb.client.as_ref().unwrap().$call(request_string);

        let headers = self.rb.headers.get();
        if !headers.is_none() {
          let headers_table: Table = headers.try_into()?;
          for (k, v) in headers_table.iter() {
            let key: &str = k.as_ref().try_into()?;
            let hname: HeaderName = key
              .try_into()
              .map_err(|_| "Could not convert into HeaderName")?;
            let hvalue = HeaderValue::from_str(v.as_ref().try_into()?)
              .map_err(|_| "Could not convert into HeaderValue")?;
            request = request.header(hname, hvalue);
          }
        }

        if !input.is_none() {
          let input_table: Table = input.try_into()?;
          for (k, v) in input_table.iter() {
            let key: &str = k.as_ref().try_into()?;
            let value: &str = v.as_ref().try_into()?;
            request = request.query(&[(key, value)]);
          }
        }

        if self.rb.retry == 0 {
          let _ = self.rb._finalize(context, request)?;
          return Ok(Some(self.rb.output.0));
        } else {
          let mut retries = self.rb.retry;
          loop {
            let request = request.try_clone().ok_or_else(|| {
              shlog_error!("Failed to clone the request");
              "Failed to clone the request"
            })?;

            let result = self.rb._finalize(context, request);

            if let Ok(()) = result {
              return Ok(Some(self.rb.output.0));
            }

            // Check if retries are exhausted
            if retries == 0 {
              return Err("Request failed");
            } else {
              if shards::core::cancel_abort(context) {
                shlog_debug!("Retrying request, {} tries left", retries);
                shards::core::suspend(context, self.rb.backoff as f64); // use backoff instead of hardcoded 1.0
                retries -= 1;
              } else {
                return Err("Cannot retry request, wire might have been aborted");
              }
            }
          }
        }
      }
    }
  };
}

macro_rules! post_like {
  ($shard_name:ident, $call:ident, $name_str:literal, $hash:literal) => {
    #[derive(Default)]
    struct $shard_name {
      rb: RequestBase,
      on_worker_thread: bool,
    }

    impl LegacyShard for $shard_name {
      fn registerName() -> &'static str {
        cstr!($name_str)
      }

      fn name(&mut self) -> &str {
        $name_str
      }

      fn hash() -> u32 {
        compile_time_crc32::crc32!($hash)
      }

      fn help(&mut self) -> OptionalString {
        if $name_str == "Http.Post" {
          OptionalString(shccstr!("This shard sends a HTTP POST request to the specified URL and outputs the response."))
        } else if $name_str == "Http.Put" {
          OptionalString(shccstr!("This shard sends a HTTP PUT request to the specified URL and outputs the response."))
        } else if $name_str == "Http.Patch" {
          OptionalString(shccstr!("This shard sends a HTTP PATCH request to the specified URL and outputs the response."))
        } else if $name_str == "Http.Delete" {
          OptionalString(shccstr!("This shard sends a HTTP DELETE request to the specified URL and outputs the response."))
        } else {
          OptionalString(shccstr!("this shard sends the respective HTTP request to the specified URL and outputs the response."))
        }
      }

      fn inputHelp(&mut self) -> OptionalString {
        if $name_str == "Http.Post" {
          OptionalString(shccstr!("The input for this shard should either be none, string, bytes, or string table to send in the body of the POST request."))
        } else if $name_str == "Http.Put" {
          OptionalString(shccstr!("The input for this shard should either be none, string, bytes, or string table to send in the body of the PUT request."))
        } else if $name_str == "Http.Patch" {
          OptionalString(shccstr!("The input for this shard should either be none, string, bytes, or string table to send in the body of the PATCH request."))
        } else if $name_str == "Http.Delete" {
          OptionalString(shccstr!("The input for this shard should either be none, string, bytes, or string table to send in the body of the DELETE request."))
        } else {
          OptionalString(shccstr!("The input for this shard should either be none, string, bytes, or string table to send in the body of the respective HTTP request."))
        }
      }

      fn outputHelp(&mut self) -> OptionalString {
        if $name_str == "Http.Post" {
          OptionalString(shccstr!("The output is the response from the server through the POST request as a string, byte array, or table (if the FullResponse parameter is set to true)."))
        } else if $name_str == "Http.Put" {
          OptionalString(shccstr!("The output is the response from the server through the PUT request as a string, byte array, or table (if the FullResponse parameter is set to true)."))
        } else if $name_str == "Http.Patch" {
          OptionalString(shccstr!("The output is the response from the server through the PATCH request as a string, byte array, or table (if the FullResponse parameter is set to true)."))
        } else if $name_str == "Http.Delete" {
          OptionalString(shccstr!("The output is the response from the server through the DELETE request as a string, byte array, or table (if the FullResponse parameter is set to true)."))
        } else {
          OptionalString(shccstr!("The output is the response from the server through the respective HTTP request as a string, byte array, or table (if the FullResponse parameter is set to true)."))
        }
      }

      fn inputTypes(&mut self) -> &Types {
        &POST_INPUT_TYPES
      }

      fn outputTypes(&mut self) -> &Types {
        self.rb._outputTypes()
      }

      fn parameters(&mut self) -> Option<&Parameters> {
        self.rb._parameters()
      }

      fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
        self.rb._requiredVariables()
      }

      fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &'static str> {
        self.rb._setParam(index, value)
      }

      fn getParam(&mut self, index: i32) -> Var {
        self.rb._getParam(index)
      }

      fn warmup(&mut self, context: &Context) -> Result<(), &'static str> {
        self.rb._warmup(context)
      }

      fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &'static str> {
        self.rb._cleanup(ctx);
        Ok(())
      }

      fn hasCompose() -> bool {
        true
      }

      fn compose(&mut self, data: &InstanceData) -> Result<Type, &'static str> {
        self.on_worker_thread = data.onWorkerThread;
        self.rb._compose(data)
      }

      fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
        if !self.rb.keep_alive {
          self.rb._open_client()?;
        }

        let request = self.rb.url.get();
        let request_string: &str = request.try_into()?;

        let mut request = self.rb.client.as_ref().unwrap().$call(request_string);

        let headers = self.rb.headers.get();

        if !input.is_none() {
          // .form ( kv table )
          let input_table: Result<Table, &'static str> = input.try_into();
          if let Ok(input_table) = input_table {
            // default to this in this case but users can edit under
            if headers.is_none() {
              request = request.header("content-type", "application/x-www-form-urlencoded");
            }

            let mut params = HashMap::new();
            for (k, v) in input_table.iter() {
              let key: &str = k.as_ref().try_into()?;
              let value: &str = v.as_ref().try_into()?;
              params.insert(key, value);
            }
            request = request.form(&params);
          } else {
            // .body ( string )
            let input_string: Result<&str, &'static str> = input.try_into();
            if let Ok(input_string) = input_string {
              // default to this in this case but users can edit under
              if headers.is_none() {
                request = request.header("content-type", "application/json");
              }

              request = request.body(input_string);
            } else {
              // .body ( bytes )
              let input_bytes: Result<&[u8], &'static str> = input.try_into();
              if let Ok(input_bytes) = input_bytes {
                // default to this in this case but users can edit under
                if headers.is_none() {
                  request = request.header("content-type", "application/octet-stream");
                }

                request = request.body(input_bytes);
              } else {
                return Err("Invalid input type");
              }
            }
          }
        }

        if !headers.is_none() {
          let headers_table: Table = headers.as_ref().try_into()?;
          for (k, v) in headers_table.iter() {
            let key: &str = k.as_ref().try_into()?;
            let hname: HeaderName = key
              .try_into()
              .map_err(|_| "Could not convert into HeaderName")?;
            let hvalue = HeaderValue::from_str(v.as_ref().try_into()?)
              .map_err(|_| "Could not convert into HeaderValue")?;
            request = request.header(hname, hvalue);
          }
        }

        if self.rb.retry == 0 {
          let _ = self.rb._finalize(context, request)?;
          return Ok(Some(self.rb.output.0));
        } else {
          let mut retries = self.rb.retry;
          loop {
            let request = request.try_clone().ok_or_else(|| {
              shlog_error!("Failed to clone the request");
              "Failed to clone the request"
            })?;

            let result = self.rb._finalize(context, request);

            if let Ok(()) = result {
              return Ok(Some(self.rb.output.0));
            }

            // Check if retries are exhausted
            if retries == 0 {
              return Err("Request failed");
            } else {
              if shards::core::cancel_abort(context) {
                shlog_debug!("Retrying request, {} tries left", retries);
                shards::core::suspend(context, self.rb.backoff as f64); // use backoff instead of hardcoded 1.0
                retries -= 1;
              } else {
                return Err("Cannot retry request, wire might have been aborted");
              }
            }
          }
        }
      }
    }
  };
}

get_like!(Get, get, "Http.Get", "Http.Get-rust-0x20200101");
get_like!(Head, head, "Http.Head", "Http.Head-rust-0x20200101");
post_like!(Post, post, "Http.Post", "Http.Post-rust-0x20200101");
post_like!(Put, put, "Http.Put", "Http.Put-rust-0x20200101");
post_like!(Patch, patch, "Http.Patch", "Http.Patch-rust-0x20200101");
post_like!(Delete, delete, "Http.Delete", "Http.Delete-rust-0x20200101");

#[derive(shards::shard)]
#[shard_info("Http.Stream", "Reads data from a previously opened stream.")]
struct HttpStreamShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Stream", "The stream to read from.", [*STREAM_TYPE_VAR])]
  stream: ParamVar,

  output: ClonedVar,
}

impl Default for HttpStreamShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      stream: ParamVar::default(),
      output: ClonedVar::default(),
    }
  }
}

#[shards::shard_impl]
impl Shard for HttpStreamShard {
  fn input_types(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &BYTES_TYPES
  }

  fn output_help(&mut self) -> OptionalString {
    shccstr!("Bytes read from the stream. When the stream is exhausted, this will return an empty byte array.").into()
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &'static str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &'static str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &'static str> {
    self.compose_helper(data)?;
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, context: &Context, _input: &Var) -> Result<Option<Var>, &'static str> {
    let stream = *self.stream.get();
    let result = run_future(context, async move {
      let stream = unsafe { Var::from_ref_counted_object::<OurResponse>(&stream, &*STREAM_TYPE) };
      let response = unsafe { &mut (*stream?).0 };
      let runtime = TOKIO_RUNTIME.clone();
      let task = {
        let runtime = runtime.lock().unwrap();
        runtime.spawn(async move {
          let bytes = response
            .chunk()
            .await
            .map_err(|e| {
              print_error(&e);
              "Failed to read from stream"
            })?;
          if let Some(bytes) = bytes {
            Ok(ClonedVar::new_bytes(&bytes))
          } else {
            Ok(ClonedVar::new_bytes(&[]))
          }
        })
      };
      // Await the spawned task outside the lock
      task.await.map_err(|e| {
        print_error(&e);
        "Failed to join task"
      })?
    })?;
    self.output = result;
    Ok(Some(self.output.0))
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_http_rust(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }

  register_legacy_shard::<Get>();
  register_legacy_shard::<Head>();
  register_legacy_shard::<Post>();
  register_legacy_shard::<Put>();
  register_legacy_shard::<Patch>();
  register_legacy_shard::<Delete>();

  register_shard::<HttpStreamShard>();
}
