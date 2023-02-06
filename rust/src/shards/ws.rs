use crate::{
  core::{run_blocking, registerShard, BlockingShard},
  types::{
    ClonedVar, Context, ExposedInfo, ExposedTypes, ParamVar, Parameters, Type, Types, Var,
    ANYS_TYPES, FRAG_CC, NONE_TYPES, STRING_TYPES,
  },
  Shard,
};
use std::net::TcpStream;
use std::rc::Rc;
use std::thread::spawn;
use tungstenite::{
  accept, handshake::client::Response, stream::MaybeTlsStream, Message, WebSocket,
};

static WS_CLIENT_TYPE: Type = Type::object(FRAG_CC, 2004042604); // 'wsCl'
static WS_CLIENT_SLICE: &'static [Type] = &[WS_CLIENT_TYPE];
static WS_CLIENT_VAR: Type = Type::context_variable(WS_CLIENT_SLICE);

lazy_static! {
  static ref WS_CLIENT_VEC: Types = vec![WS_CLIENT_TYPE];
  static ref WS_CLIENT_VAR_VEC: Types = vec![WS_CLIENT_VAR];
  static ref WS_CLIENT_USER_PARAMS: Parameters = vec![(
    cstr!("Client"),
    cstr!("The WebSocket client instance."),
    &WS_CLIENT_VAR_VEC[..],
  )
    .into(),];
}

type WsClient = WebSocket<MaybeTlsStream<TcpStream>>;

#[derive(Default)]
struct Client {
  ws: Rc<Option<WsClient>>,
}

impl Shard for Client {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("WS.Client")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("WS.Client-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "WS.Client"
  }

  fn inputTypes(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &WS_CLIENT_VEC
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    Ok(run_blocking(self, context, input))
  }
}

impl BlockingShard for Client {
  fn activate_blocking(&mut self, _context: &Context, input: &Var) -> Result<Var, &str> {
    let addr: &str = input.try_into()?;
    let ws = tungstenite::client::connect(addr).map_err(|e| {
      shlog!("{}", e);
      "Failed to connect to server."
    })?;
    // we need to keep this alive here, otherwise it will be dropped at the end of this function
    self.ws = Rc::new(Some(ws.0));
    let var_ws = Var::new_object(&self.ws, &WS_CLIENT_TYPE);
    Ok(var_ws)
  }
}

#[derive(Default)]
struct ClientUser {
  ws: Rc<Option<WsClient>>,
  instance: ParamVar,
  requiring: Vec<ExposedInfo>,
}

impl ClientUser {
  fn set_param(&mut self, index: i32, value: &Var) {
    match index {
      0 => self.instance.set_param(value),
      _ => panic!("Invalid index {}", index),
    }
  }

  fn get_param(&self, index: i32) -> Var {
    match index {
      0 => self.instance.get_param(),
      _ => panic!("Invalid index {}", index),
    }
  }

  fn required_variables(&mut self) -> Option<&ExposedTypes> {
    self.requiring.clear();

    let exp_info = ExposedInfo {
      exposedType: WS_CLIENT_TYPE,
      name: self.instance.get_name(),
      help: cstr!("The exposed Websocket.").into(),
      ..ExposedInfo::default()
    };
    self.requiring.push(exp_info);

    Some(&self.requiring)
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.instance.warmup(context);
    Ok(())
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.instance.cleanup();
    Ok(())
  }

  fn activate(&mut self) -> Result<&mut WsClient, &str> {
    if self.ws.is_none() {
      let ws = self.instance.get();
      self.ws = Var::from_object_as_clone(*ws, &WS_CLIENT_TYPE)?;
    }

    let ws = Var::from_object_mut_ref(*self.instance.get(), &WS_CLIENT_TYPE)?;

    if let Some(ws) = ws {
      Ok(ws)
    } else {
      Err("No WebSocket client.")
    }
  }
}

#[derive(Default)]
struct ReadString {
  client: ClientUser,
  text: ClonedVar,
}

impl Shard for ReadString {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("WS.ReadString")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("WS.ReadString-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "WS.ReadString"
  }

  fn inputTypes(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&WS_CLIENT_USER_PARAMS)
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.client.required_variables()
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    self.client.set_param(index, value);
    Ok(())
  }

  fn getParam(&mut self, index: i32) -> Var {
    self.client.get_param(index)
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.client.warmup(context)
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.client.cleanup()
  }

  fn activate(&mut self, context: &Context, input: &Var) -> Result<Var, &str> {
    Ok(run_blocking(self, context, input))
  }
}

impl BlockingShard for ReadString {
  fn activate_blocking(&mut self, _context: &Context, _input: &Var) -> Result<Var, &str> {
    let ws = self.client.activate()?;

    loop {
      let msg = ws.read_message().map_err(|e| {
        shlog!("{}", e);
        "Failed to read message."
      })?;

      match msg {
        Message::Text(text) => {
          self.text = text.into();
          return Ok(self.text.0);
        }
        Message::Binary(_) => {
          return Err("Received binary message, expected text.");
        }
        Message::Close(_) => {
          return Err("Received close message.");
        }
        Message::Ping(_) => {
          // shlog_debug!("Received ping message, sending pong.");
          ws.write_message(Message::Pong(vec![])).map_err(|e| {
            shlog!("{}", e);
            "Failed to write message."
          })?;
        }
        Message::Pong(_) => {
          return Err("Received pong message, expected text.");
        }
        _ => return Err("Invalid message type."),
      }
    }
  }
}

#[derive(Default)]
struct WriteString {
  client: ClientUser,
}

impl Shard for WriteString {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("WS.WriteString")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("WS.WriteString-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "WS.WriteString"
  }

  fn inputTypes(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&WS_CLIENT_USER_PARAMS)
  }

  fn requiredVariables(&mut self) -> Option<&ExposedTypes> {
    self.client.required_variables()
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &str> {
    self.client.set_param(index, value);
    Ok(())
  }

  fn getParam(&mut self, index: i32) -> Var {
    self.client.get_param(index)
  }

  fn warmup(&mut self, context: &Context) -> Result<(), &str> {
    self.client.warmup(context)
  }

  fn cleanup(&mut self) -> Result<(), &str> {
    self.client.cleanup()
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Var, &str> {
    // TODO make this run on a worker thread but not using BlockingShard as would overlap/crash
    let ws = self.client.activate()?;

    let msg: &str = input.try_into()?;

    ws.write_message(Message::Text(msg.to_string()))
      .map_err(|e| {
        shlog!("{}", e);
        "Failed to send message."
      })?;

    Ok(*input)
  }
}

pub fn registerShards() {
  registerShard::<Client>();
  registerShard::<ReadString>();
  registerShard::<WriteString>();
}