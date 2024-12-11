// prevent upper case globals
#![allow(non_upper_case_globals)]

use directory::{get_global_map, get_global_name_btree};
use egui::*;
use nanoid::nanoid;
use std::cell::RefCell;
use std::sync::mpsc;

use crate::{
  util::{get_current_parent_opt, require_parents},
  widgets::drag_value::CustomDragValue,
  PARENTS_UI_NAME,
};
use shards::{
  core::register_shard,
  shard::Shard,
  types::{
    ClonedVar, Context as ShardsContext, ExposedTypes, InstanceData, OptionalString, ParamVar,
    SeqVar, Type, Types, Var, BOOL_TYPES, NONE_TYPES,
  },
  SHType_Any, SHType_Bool, SHType_Bytes, SHType_Color, SHType_ContextVar, SHType_Enum,
  SHType_Float, SHType_Float2, SHType_Float3, SHType_Float4, SHType_Int, SHType_Int16, SHType_Int2,
  SHType_Int3, SHType_Int4, SHType_Int8, SHType_None, SHType_Seq, SHType_ShardRef, SHType_String,
  SHType_Table, SHType_Wire,
};

use shards_lang::{
  ast::*,
  ast_visitor::*,
  custom_state::*,
  directory,
  read::{AST_TYPE, AST_VAR_TYPE},
  ParamHelperMut, RcStrWrapper,
};

use num_traits::{Float, FromPrimitive, PrimInt, Zero};

// ok here is the thing with names, we need to format them better, when defined they are in PascalCase
// we convert them to use spaces between words instead
thread_local! {
  // Define a thread-local storage for the cache of formatted names
  static NAME_CACHE: RefCell<String> = RefCell::new(String::new());
}

// Function to convert PascalCase to spaced format
fn pascal_to_spaced(pascal: &str, buffer: &mut String) {
  // Clear the buffer to reuse it for new formatted output
  buffer.clear();
  let mut first = true;
  let chars: Vec<char> = pascal.chars().collect();
  let length = chars.len();

  for i in 0..length {
    let c = chars[i];
    if c.is_uppercase() {
      // Check if the current uppercase letter is part of an acronym
      let is_acronym =
        (i + 1 < length && chars[i + 1].is_uppercase()) || (i > 0 && chars[i - 1].is_uppercase());

      if !first && !is_acronym {
        buffer.push(' '); // Insert a space before uppercase letters except the first
      }

      if i == 0 || is_acronym {
        buffer.push(c); // Keep the first letter or acronym letters capitalized
      } else {
        buffer.push(c.to_ascii_lowercase()); // Convert subsequent uppercase letters to lowercase
      }
    } else {
      buffer.push(c); // Append lowercase or other characters directly
    }
    first = false;
  }
}

// Function to obtain a reference to the formatted name
fn get_formatted_name_ref(name: &str) -> &'static str {
  NAME_CACHE.with(|cache| {
    // Access the thread-local cache and get a mutable reference to it
    let mut cache = cache.borrow_mut();

    // Format the name and update the cache with the new formatted string
    pascal_to_spaced(name, &mut cache);

    // Obtain a reference to the updated cache
    let cache_ref: &str = &cache;

    // Convert the reference to a `Box<str>` and then leak it to get a `'static` reference
    // `Box::leak` creates a leakable `Box<str>` that lives for the entire lifetime of the program
    // This is generally safe here because the storage is thread-local and won't be freed
    Box::leak(cache_ref.to_string().into_boxed_str())
  })
}

fn var_to_value(var: &Var) -> Result<Value, String> {
  match var.valueType {
    SHType_None => Ok(Value::None(())),
    SHType_Bool => Ok(Value::Boolean(unsafe {
      var.payload.__bindgen_anon_1.boolValue
    })),
    SHType_Int => Ok(Value::Number(Number::Integer(unsafe {
      var.payload.__bindgen_anon_1.intValue
    }))),
    SHType_Float => Ok(Value::Number(Number::Float(unsafe {
      var.payload.__bindgen_anon_1.floatValue
    }))),
    SHType_String => {
      let string = unsafe { var.payload.__bindgen_anon_1.string };
      let string_slice =
        unsafe { std::slice::from_raw_parts(string.elements as *const u8, string.len as usize) };
      let string = unsafe { std::str::from_utf8_unchecked(string_slice) };
      Ok(Value::String(string.into()))
    }
    SHType_Bytes => {
      let bytes = unsafe {
        std::slice::from_raw_parts(
          var.payload.__bindgen_anon_1.__bindgen_anon_4.bytesValue,
          var.payload.__bindgen_anon_1.__bindgen_anon_4.bytesSize as usize,
        )
      };
      Ok(Value::Bytes(bytes.into()))
    }
    SHType_Float2 => {
      let float2 = unsafe { var.payload.__bindgen_anon_1.float2Value };
      Ok(Value::Float2([float2[0], float2[1]]))
    }
    SHType_Float3 => {
      let float3 = unsafe { var.payload.__bindgen_anon_1.float3Value };
      Ok(Value::Float3([float3[0], float3[1], float3[2]]))
    }
    SHType_Float4 => {
      let float4 = unsafe { var.payload.__bindgen_anon_1.float4Value };
      Ok(Value::Float4([float4[0], float4[1], float4[2], float4[3]]))
    }
    SHType_Int2 => {
      let int2 = unsafe { var.payload.__bindgen_anon_1.int2Value };
      Ok(Value::Int2([int2[0], int2[1]]))
    }
    SHType_Int3 => {
      let int3 = unsafe { var.payload.__bindgen_anon_1.int3Value };
      Ok(Value::Int3([int3[0], int3[1], int3[2]]))
    }
    SHType_Int4 => {
      let int4 = unsafe { var.payload.__bindgen_anon_1.int4Value };
      Ok(Value::Int4([int4[0], int4[1], int4[2], int4[3]]))
    }
    SHType_Int8 => {
      let int8 = unsafe { var.payload.__bindgen_anon_1.int8Value };
      Ok(Value::Int8([
        int8[0], int8[1], int8[2], int8[3], int8[4], int8[5], int8[6], int8[7],
      ]))
    }
    SHType_Int16 => {
      let int16 = unsafe { var.payload.__bindgen_anon_1.int16Value };
      Ok(Value::Int16([
        int16[0], int16[1], int16[2], int16[3], int16[4], int16[5], int16[6], int16[7], int16[8],
        int16[9], int16[10], int16[11], int16[12], int16[13], int16[14], int16[15],
      ]))
    }
    SHType_Enum => {
      // not so simple as the Value::Enum:
      // 1. we need to derive name and value from the actual numeric values
      // 2. values can be sparse so we need both labels and values
      let enums_from_ids = get_global_map()
        .0
        .get_fast_static("enums-from-ids")
        .as_table()
        .unwrap();
      let enum_type_id = unsafe { var.payload.__bindgen_anon_1.__bindgen_anon_3.enumTypeId };
      let enum_vendor_id = unsafe { var.payload.__bindgen_anon_1.__bindgen_anon_3.enumVendorId };
      let enum_value = unsafe { var.payload.__bindgen_anon_1.__bindgen_anon_3.enumValue };
      // ok so the composite value in c++ is made like this:
      // int64_t id = (int64_t)vendorId << 32 | typeId;
      let id = (enum_vendor_id as i64) << 32 | enum_type_id as i64;
      let enum_info = enums_from_ids
        .get(Var::from(id))
        .map(|x| x.as_table().unwrap());
      if let Some(enum_info) = enum_info {
        let labels = enum_info.get_fast_static("labels").as_seq().unwrap();
        let values = enum_info.get_fast_static("values").as_seq().unwrap();
        let index = values.iter().position(|x| x == enum_value.into()).unwrap();
        let name: &str = enum_info
          .get_fast_static("name")
          .as_ref()
          .try_into()
          .unwrap();
        let label: &str = labels[index].as_ref().try_into().unwrap();
        Ok(Value::Enum(name.into(), label.into()))
      } else {
        Err(format!("Enum not found: {}", id))
      }
    }
    SHType_Seq => {
      let seq = var.as_seq().unwrap();
      let mut values = Vec::new();
      for value in seq {
        values.push(var_to_value(&value)?);
      }
      Ok(Value::Seq(values))
    }
    SHType_Table => {
      let table = var.as_table().unwrap();
      let mut map = Vec::new();
      for (key, value) in table.iter() {
        let key = var_to_value(&key)?;
        let value = var_to_value(&value)?;
        map.push((key, value));
      }
      Ok(Value::Table(map))
    }
    SHType_ShardRef => unreachable!("ShardRef type not supported"),
    SHType_Wire => unreachable!("Wire type not supported"),
    SHType_ContextVar => {
      let string = unsafe { var.payload.__bindgen_anon_1.string };
      let string_slice =
        unsafe { std::slice::from_raw_parts(string.elements as *const u8, string.len as usize) };
      let string = unsafe { std::str::from_utf8_unchecked(string_slice) };
      // ok we need to derive namespace from the string, we define then using `/` such as a/b/c
      let mut namespaces = string
        .split('/')
        .map(|x| x.into())
        .collect::<Vec<RcStrWrapper>>();
      // name is last part, we can just pop it and remove it from the namespaces without cloning in one go
      let name = namespaces.pop().unwrap();
      Ok(Value::Identifier(Identifier {
        name: name.into(),
        namespaces,
        custom_state: CustomStateContainer::new(),
      }))
    }
    _ => Err(format!("Unsupported Var type: {:?}", var.valueType)),
  }
}

enum SwapStateResult<T> {
  Done(T),
  Continue,
  Close,
}

#[derive(Debug, Clone, PartialEq)]
struct SwapStateCommon {
  id: Id,
  receiver: Option<UniqueReceiver<ClonedVar>>,
  window_pos: Pos2,
}

#[derive(Debug, Clone, PartialEq)]
struct ParamSwapState {
  common: SwapStateCommon,
  param: *mut Param,
  types: *const SeqVar,
}

#[derive(Debug)]
struct BlockSwapState {
  common: SwapStateCommon,
  block: *mut Block,
  search_string: String,
  previous_search_string: String,
  search_results: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct BlockState {
  selected: bool,
  id: Id,
}

#[derive(Debug, Clone, PartialEq)]
struct FunctionState {
  params_sorted: bool,
}

enum SwapState {
  Param(ParamSwapState),
  Block(BlockSwapState),
}

#[derive(Debug)]
struct UniqueReceiver<T> {
  receiver: Option<mpsc::Receiver<T>>,
}

impl<T> Clone for UniqueReceiver<T> {
  fn clone(&self) -> Self {
    UniqueReceiver { receiver: None }
  }
}

impl<T> PartialEq for UniqueReceiver<T> {
  fn eq(&self, _other: &Self) -> bool {
    // no-op
    true
  }
}

impl<T> UniqueReceiver<T> {
  fn new(receiver: mpsc::Receiver<T>) -> Self {
    UniqueReceiver {
      receiver: Some(receiver),
    }
  }

  fn get_mut(&mut self) -> Option<&mut mpsc::Receiver<T>> {
    self.receiver.as_mut()
  }
}

#[derive(Debug, Clone, PartialEq)]
struct SequenceState {
  selected: bool,
}

// common state
pub struct Context {
  swap_state: Option<SwapState>,
  seqs_zoom_stack: Vec<*mut Sequence>,
  seqs_stack: Vec<*mut Sequence>,
  has_changed: bool,
  non_error_stroke: Option<egui::Stroke>,
}

pub struct VisualAst<'a> {
  context: &'a mut Context,
  ui: &'a mut Ui,
  parent_selected: bool,
}

// Helper function to determine if a color is light
fn is_light_color(color: egui::Color32) -> bool {
  let brightness = 0.299 * color.r() as f32 + 0.587 * color.g() as f32 + 0.114 * color.b() as f32;
  brightness > 128.0
}

fn emoji(s: &str) -> egui::RichText {
  egui::RichText::new(s)
    .family(egui::FontFamily::Proportional)
    // .strong()
    .size(14.0)
}

fn chars(s: &str) -> egui::RichText {
  egui::RichText::new(s)
    .family(egui::FontFamily::Proportional)
    .strong()
    .size(15.0)
}

fn get_first_shard_ref<'a>(ast: &'a mut Sequence) -> Option<&'a mut Function> {
  for statement in &mut ast.statements {
    match statement {
      Statement::Assignment(assignment) => {
        if let Assignment::AssignRef(pipeline, _) = assignment {
          if let BlockContent::Shard(shard) = &mut pipeline.blocks[0].content {
            return Some(shard);
          }
        }
      }
      Statement::Pipeline(pipeline) => {
        if let BlockContent::Shard(shard) = &mut pipeline.blocks[0].content {
          return Some(shard);
        }
      }
    }
  }
  None
}
fn get_last_shard_ref<'a>(ast: &'a mut Sequence) -> Option<&'a mut Function> {
  if ast.statements.len() < 2 {
    return None;
  }
  for statement in ast.statements.iter_mut().rev() {
    match statement {
      Statement::Assignment(assignment) => {
        if let Assignment::AssignRef(pipeline, _) = assignment {
          if let BlockContent::Shard(shard) = &mut pipeline.blocks.last_mut()?.content {
            return Some(shard);
          }
        }
      }
      Statement::Pipeline(pipeline) => {
        if let BlockContent::Shard(shard) = &mut pipeline.blocks.last_mut()?.content {
          return Some(shard);
        }
      }
    }
  }
  None
}

impl<'a> VisualAst<'a> {
  pub fn with_parent_selected(
    context: &'a mut Context,
    ui: &'a mut Ui,
    parent_selected: bool,
  ) -> Self {
    VisualAst {
      context,
      ui,
      parent_selected,
    }
  }

  pub fn render(&mut self, ast: &mut Sequence) -> Option<Response> {
    ast.accept_mut(self)
  }

  fn populate_params(x: &mut Function, info: &SeqVar) {
    // only if we really need to sort
    let mut params = x.params.take();
    // reset current params
    x.params = Some(Vec::new());

    // helper if needed as well
    let mut helper = params.as_mut().map(|x| ParamHelperMut::new(x));

    for (idx, param) in info.into_iter().enumerate() {
      let param = param.as_table().unwrap();
      let name: &str = param.get_fast_static("name").try_into().unwrap();

      let new_param = helper
        .as_mut()
        .and_then(|ast| ast.get_param_by_name_or_index_mut(name, idx))
        .cloned();

      let param_to_add = new_param.unwrap_or_else(|| {
        let default_value = param.get_fast_static("default");
        Param {
          name: Some(name.into()),
          value: var_to_value(&default_value).unwrap(),
          custom_state: CustomStateContainer::new(),
          is_default: Some(true),
        }
      });

      x.params.as_mut().unwrap().push(param_to_add);
    }

    // set flag to true
    x.custom_state.with_mut::<FunctionState, _, _>(|s| {
      s.params_sorted = true;
    });
  }

  fn mutate_shard(&mut self, x: &mut Function) -> Option<Response> {
    let params_sorted = x.custom_state.with_or_insert_with(
      || FunctionState {
        params_sorted: false,
      },
      |x| x.params_sorted,
    );

    // // check if we have a result from a pending operation
    // let has_result = state
    //   .receiver
    //   .as_mut()
    //   .and_then(|r| r.get_mut())
    //   .map(|r| r.try_recv());

    // if let Some(Ok(result)) = has_result {
    //   shlog_debug!("Got result: {:?}", result);
    //   // reset the receiver
    //   state.receiver = None;
    // }

    let directory = directory::get_global_map();
    let shards = directory.0.get_fast_static("shards");
    let shards = shards.as_table().unwrap();
    let shard_name = x.name.name.as_str();
    let shard_name_var = Var::ephemeral_string(shard_name);
    let shard_info = shards.get(shard_name_var).and_then(|x| x.as_table().ok());

    let help_text: Option<&str> = shard_info.map(|x| {
      x.get_fast_static("help")
        .try_into()
        .expect("A shard's help text must be a string!")
    });
    let help_text = help_text
      .map(|x| {
        if x.is_empty() {
          "No help text provided."
        } else {
          x
        }
      })
      .unwrap_or("No help text provided.");

    let error_text = x
      .custom_state
      .with::<ShardsError, _, _>(|s| s.message.clone());

    let color = shard_info.map(|x| {
      let c = x.get_fast_static("color");
      Var::color_bytes(c).unwrap_or((255, 255, 255, 255))
    });

    let shard_name_rt = if let Some(color) = color {
      let bg_color = egui::Color32::from_rgb(color.0, color.1, color.2);

      // Determine text color based on background brightness
      let text_color = if is_light_color(bg_color) {
        egui::Color32::BLACK
      } else {
        egui::Color32::WHITE
      };

      egui::RichText::new(shard_name)
        .size(14.0)
        .strong()
        .family(egui::FontFamily::Monospace)
        .background_color(bg_color)
        .color(text_color)
    } else {
      egui::RichText::new(shard_name)
        .size(14.0)
        .strong()
        .family(egui::FontFamily::Monospace)
        .italics()
    };

    let response = if let Some(error_text) = error_text {
      let help_text_rich = RichText::new(help_text).color(Color32::WHITE);
      let error_text_rich = RichText::new(error_text).color(Color32::LIGHT_RED);

      self.ui.label(shard_name_rt).on_hover_ui(|ui: &mut Ui| {
        ui.label(help_text_rich);
        ui.add_space(5.0);
        ui.label(error_text_rich);
      })
    } else {
      self.ui.label(shard_name_rt).on_hover_ui(|ui: &mut Ui| {
        ui.label(RichText::new(help_text).color(Color32::WHITE));
      })
    };

    let params = shard_info.and_then(|x| x.get_fast_static("parameters").as_seq().ok());

    if !params_sorted {
      if let Some(params) = params {
        Self::populate_params(x, params);
      }
    }

    if self.parent_selected {
      if let Some(params) = params {
        // We have documentation...
        if !params.is_empty() {
          for (idx, param) in params.into_iter().enumerate() {
            let param = param.as_table().unwrap();
            let name: &str = param
              .get_fast_static("name")
              .try_into()
              .expect("A shard's parameter name must be a string!");
            let name = get_formatted_name_ref(name);
            let help_text: &str = param
              .get_fast_static("help")
              .try_into()
              .expect("A shard's parameter help text must be a string!");
            let types = param
              .get_fast_static("types")
              .as_seq()
              .expect("A shard's parameter types must be a sequence!");
            let help_text = if help_text.is_empty() {
              "No help text provided."
            } else {
              help_text
            };
            let default_value = param.get_fast_static("default");
            let is_at_default_value = x
              .params
              .as_ref()
              .map(|params| {
                let param = &params[idx];
                let default_value = var_to_value(&default_value).unwrap();
                &param.value == &default_value
              })
              .unwrap_or(false);
            egui::CollapsingHeader::new(name.as_str())
              .default_open(!is_at_default_value)
              .show(self.ui, |ui| {
                ui.horizontal(|ui| {
                  // button to reset to default
                  if ui
                    .button(emoji("🔄"))
                    .on_hover_text("Reset to default value.")
                    .clicked()
                  {
                    // reset to default
                    shlog_debug!("Resetting: {} to default value.", name);
                    x.params.as_mut().map(|params| {
                      params[idx].value = var_to_value(&default_value).unwrap();
                    });
                    self.context.has_changed = true;
                  }
                  if ui
                    .button(emoji("🔧"))
                    .on_hover_text("Change value type.")
                    .clicked()
                  {
                    // let (sender, receiver) = mpsc::channel();
                    // let query = Var::ephemeral_string("").into();
                    // get_global_visual_shs_channel_sender()
                    //   .send((query, sender))
                    //   .unwrap();
                    // x.custom_state.get_mut::<FunctionState>().unwrap().receiver =
                    //   Some(UniqueReceiver::new(receiver));

                    // open a dialog to change the value

                    let mouse_pos = ui
                      .ctx()
                      .input(|i| i.pointer.hover_pos().unwrap_or_default());
                    self.context.swap_state = Some(SwapState::Param(ParamSwapState {
                      common: SwapStateCommon {
                        id: Id::new(nanoid!()),
                        receiver: None,
                        window_pos: mouse_pos,
                      },
                      param: &mut x.params.as_mut().unwrap()[idx],
                      types: types as *const SeqVar,
                    }));
                  }
                });

                x.params.as_mut().map(|params| {
                  let param = &mut params[idx];

                  // this will be useful later in print for example!
                  param.is_default = Some(is_at_default_value);

                  // draw the value
                  let mut mutator =
                    VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                  param.value.accept_mut(&mut mutator);

                  // process changes
                  let new_value =
                    if let Some(SwapState::Param(swap_state)) = &mut self.context.swap_state {
                      if swap_state.param == param {
                        match select_value_modal(ui, swap_state) {
                          SwapStateResult::<Value>::Done(f) => {
                            self.context.has_changed = true;
                            self.context.swap_state = None;
                            Some(f)
                          }
                          SwapStateResult::Close => {
                            self.context.swap_state = None;
                            None
                          }
                          _ => None,
                        }
                      } else {
                        None
                      }
                    } else {
                      None
                    };

                  if let Some(new_value) = new_value {
                    param.value = new_value;
                  }
                });
              })
              .header_response
              .on_hover_text(help_text);
          }
        }
      } else {
        // no documentation just render the values
        if let Some(params) = &mut x.params {
          for param in params {
            let mut mutator =
              VisualAst::with_parent_selected(self.context, self.ui, self.parent_selected);
            param.value.accept_mut(&mut mutator);
          }
        }
      }
    } else {
      // preview the first param THAT IS NOT None if it exists
      self.ui.add_enabled_ui(false, |ui| {
        if let Some(params) = &mut x.params {
          for param in params {
            if let Value::None(_) = param.value {
              continue;
            }
            let mut mutator =
              VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
            param.value.accept_mut(&mut mutator);
            break;
          }
        }
      });
    }
    Some(response)
  }

  fn mutate_color(&mut self, x: &mut Function) -> Option<Response> {
    let params = x.params.as_mut().expect("params should exist");

    // Initialize the color array with default values (255 for R, G, B, and A)
    let mut color_bytes = [255u8; 4];

    // Populate color_bytes based on the input params
    for (i, param) in params.iter().enumerate().take(4) {
      color_bytes[i] = match &param.value {
        Value::Number(Number::Integer(v)) => *v as u8,
        Value::Number(Number::Float(v)) => (*v * 255.0).clamp(0.0, 255.0) as u8,
        Value::Number(Number::Hexadecimal(s)) if s.starts_with("0x") => {
          u8::from_str_radix(&s[2..4], 16).unwrap_or(0)
        }
        _ => unreachable!(),
      };
    }

    // Show color picker
    let response = self
      .ui
      .color_edit_button_srgba_unmultiplied(&mut color_bytes);

    // Write the values back to params
    for (i, &byte) in color_bytes.iter().enumerate().take(4) {
      if i < params.len() {
        match &mut params[i].value {
          Value::Number(Number::Integer(val)) => {
            *val = byte as i64;
          }
          Value::Number(Number::Float(val)) => {
            *val = (byte as f64) / 255.0;
          }
          Value::Number(Number::Hexadecimal(s)) if s.starts_with("0x") => {
            let s = s.to_mut();
            s.replace_range(2..4, &format!("{:02x}", byte));
          }
          _ => unreachable!(),
        }
      }
    }

    Some(response)
  }
}

fn has_type(types: &SeqVar, expected_type: u8) -> bool {
  types.iter().any(|x| {
    let info = x.as_table().expect("A type must be a table!");
    let type_int: i64 = info
      .get_fast_static("type")
      .try_into()
      .expect("A type must have a type!");
    type_int as u8 == SHType_Any || type_int as u8 == expected_type
  })
}

fn select_shard_modal(ui: &mut Ui, swap_state: &mut BlockSwapState) -> SwapStateResult<Block> {
  egui::Area::new(swap_state.common.id)
    .order(egui::Order::Foreground)
    .pivot(Align2::LEFT_TOP)
    .fixed_pos(swap_state.common.window_pos)
    .show(ui.ctx(), |ui| {
      egui::Frame::menu(ui.style())
        .show(ui, |ui| {
          let mut widest = 0.0;
          let result = ui.horizontal(|ui| {
            if ui
              .button("Bool")
              .on_hover_text("A boolean value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Boolean(false)),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("Int")
              .on_hover_text("An integer value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Number(Number::Integer(0))),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui.button("Float").on_hover_text("A float value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Number(Number::Float(0.0))),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("String")
              .on_hover_text("A string value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::String("".into())),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui.button("Bytes").on_hover_text("A bytes value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Bytes(Vec::new().into())),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            SwapStateResult::Continue
          });
          widest = widest.max(result.response.rect.width());
          let result = result.inner;
          match result {
            SwapStateResult::Done(_) => return result,
            _ => {}
          }
          let result = ui.horizontal(|ui| {
            if ui
              .button("Float2")
              .on_hover_text("A float2 value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Float2([0.0, 0.0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("Float3")
              .on_hover_text("A float3 value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Float3([0.0, 0.0, 0.0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("Float4")
              .on_hover_text("A float4 value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Float4([0.0, 0.0, 0.0, 0.0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui.button("Int2").on_hover_text("An int2 value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Int2([0, 0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui.button("Int3").on_hover_text("An int3 value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Int3([0, 0, 0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            SwapStateResult::Continue
          });
          widest = widest.max(result.response.rect.width());
          let result = result.inner;
          match result {
            SwapStateResult::Done(_) => return result,
            _ => {}
          }
          let result = ui.horizontal(|ui| {
            if ui.button("Int4").on_hover_text("An int4 value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Int4([0, 0, 0, 0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui.button("Int8").on_hover_text("An int8 value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Int8([0, 0, 0, 0, 0, 0, 0, 0])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("Int16")
              .on_hover_text("An int16 value.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Int16([0; 16])),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("Seq")
              .on_hover_text("A sequence of values.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Seq(Vec::new())),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui
              .button("Table")
              .on_hover_text("A table of key-value pairs.")
              .clicked()
            {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Table(Vec::new())),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            if ui.button("Color").on_hover_text("A color value.").clicked() {
              return SwapStateResult::Done(Block {
                content: BlockContent::Const(Value::Func(Function {
                  name: Identifier {
                    name: "color".into(),
                    namespaces: Vec::new(),
                    custom_state: CustomStateContainer::new(),
                  },
                  params: Some(vec![
                    // 4 Number/Integer values
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                  ]),
                  custom_state: CustomStateContainer::new(),
                })),
                line_info: None,
                custom_state: CustomStateContainer::new(),
              });
            }
            SwapStateResult::Continue
          });
          widest = widest.max(result.response.rect.width());
          let result = result.inner;
          match result {
            SwapStateResult::Done(_) => return result,
            _ => {}
          }

          ui.add_sized([widest, 1.0], egui::Separator::default().horizontal());

          ui.label("Search for a shard:");
          let len = swap_state.search_string.len();
          TextEdit::singleline(&mut swap_state.search_string)
            .clip_text(false)
            .desired_width(if len == 0 { 40.0 } else { 0.0 })
            .ui(ui);
          let result = ui
            .horizontal(|ui| {
              if ui.button("Cancel").clicked() {
                SwapStateResult::<Block>::Close
              } else {
                SwapStateResult::Continue
              }
            })
            .inner;
          match result {
            SwapStateResult::Close => return SwapStateResult::Close,
            _ => {}
          }

          let prefix = &swap_state.search_string;

          if *prefix != swap_state.previous_search_string {
            swap_state.search_results.clear();
            let shards = get_global_name_btree();
            for shard in shards.range(prefix.to_string()..) {
              if shard.starts_with(prefix) {
                // exit if more than 100
                if swap_state.search_results.len() > 100 {
                  break;
                }
                swap_state.search_results.push(shard.clone());
              } else {
                break;
              }
            }
          }

          ui.add_sized([widest, 1.0], egui::Separator::default().horizontal());

          let maybe_block = {
            for result in swap_state.search_results.iter() {
              if ui.selectable_label(false, result).clicked() {
                return SwapStateResult::Done(Block {
                  content: BlockContent::Shard(Function {
                    name: Identifier {
                      name: result.clone().into(),
                      namespaces: Vec::new(),
                      custom_state: CustomStateContainer::new(),
                    },
                    params: None,
                    custom_state: CustomStateContainer::new(),
                  }),
                  line_info: None,
                  custom_state: CustomStateContainer::new(),
                });
              }
            }
            SwapStateResult::Continue
          };

          swap_state.previous_search_string = prefix.clone();

          maybe_block
        })
        .inner
    })
    .inner
}

fn select_value_modal(ui: &mut Ui, swap_state: &mut ParamSwapState) -> SwapStateResult<Value> {
  egui::Area::new(swap_state.common.id)
    .order(egui::Order::Foreground)
    .pivot(Align2::LEFT_TOP)
    .fixed_pos(swap_state.common.window_pos)
    .show(ui.ctx(), |ui| {
      egui::Frame::menu(ui.style())
        .show(ui, |ui| {
          let types = unsafe { &*swap_state.types };

          let mut result = SwapStateResult::Continue;

          ui.horizontal(|ui| {
            ui.add_enabled_ui(has_type(&types, SHType_None), |ui| {
              if ui.button("None").on_hover_text("A none value.").clicked() {
                result = SwapStateResult::Done(Value::None(()));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Bool), |ui| {
              if ui
                .button("Bool")
                .on_hover_text("A boolean value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Boolean(false));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Int), |ui| {
              if ui
                .button("Int")
                .on_hover_text("An integer value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Number(Number::Integer(0)));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Float), |ui| {
              if ui.button("Float").on_hover_text("A float value.").clicked() {
                result = SwapStateResult::Done(Value::Number(Number::Float(0.0)));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_String), |ui| {
              if ui
                .button("String")
                .on_hover_text("A string value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::String("".into()));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Bytes), |ui| {
              if ui.button("Bytes").on_hover_text("A bytes value.").clicked() {
                result = SwapStateResult::Done(Value::Bytes(Vec::new().into()));
              }
            });
          });

          if let SwapStateResult::Done(_) = result {
            return result;
          }

          ui.horizontal(|ui| {
            ui.add_enabled_ui(has_type(&types, SHType_Float2), |ui| {
              if ui
                .button("Float2")
                .on_hover_text("A float2 value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Float2([0.0, 0.0]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Float3), |ui| {
              if ui
                .button("Float3")
                .on_hover_text("A float3 value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Float3([0.0, 0.0, 0.0]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Float4), |ui| {
              if ui
                .button("Float4")
                .on_hover_text("A float4 value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Float4([0.0, 0.0, 0.0, 0.0]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Int2), |ui| {
              if ui.button("Int2").on_hover_text("An int2 value.").clicked() {
                result = SwapStateResult::Done(Value::Int2([0, 0]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Int3), |ui| {
              if ui.button("Int3").on_hover_text("An int3 value.").clicked() {
                result = SwapStateResult::Done(Value::Int3([0, 0, 0]));
              }
            });
          });

          if let SwapStateResult::Done(_) = result {
            return result;
          }

          ui.horizontal(|ui| {
            ui.add_enabled_ui(has_type(&types, SHType_Int4), |ui| {
              if ui.button("Int4").on_hover_text("An int4 value.").clicked() {
                result = SwapStateResult::Done(Value::Int4([0, 0, 0, 0]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Int8), |ui| {
              if ui.button("Int8").on_hover_text("An int8 value.").clicked() {
                result = SwapStateResult::Done(Value::Int8([0, 0, 0, 0, 0, 0, 0, 0]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Int16), |ui| {
              if ui
                .button("Int16")
                .on_hover_text("An int16 value.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Int16([0; 16]));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Seq), |ui| {
              if ui
                .button("Seq")
                .on_hover_text("A sequence of values.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Seq(Vec::new()));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Table), |ui| {
              if ui
                .button("Table")
                .on_hover_text("A table of key-value pairs.")
                .clicked()
              {
                result = SwapStateResult::Done(Value::Table(Vec::new()));
              }
            });

            ui.add_enabled_ui(has_type(&types, SHType_Color), |ui| {
              if ui.button("Color").on_hover_text("A color value.").clicked() {
                result = SwapStateResult::Done(Value::Func(Function {
                  name: Identifier {
                    name: "color".into(),
                    namespaces: Vec::new(),
                    custom_state: CustomStateContainer::new(),
                  },
                  params: Some(vec![
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                    Param {
                      name: None,
                      value: Value::Number(Number::Integer(255)),
                      custom_state: CustomStateContainer::new(),
                      is_default: Some(false),
                    },
                  ]),
                  custom_state: CustomStateContainer::new(),
                }));
              }
            });
          });

          if let SwapStateResult::Done(_) = result {
            return result;
          }

          if ui.button("Cancel").clicked() {
            return SwapStateResult::Close;
          }

          SwapStateResult::Continue
        })
        .inner
    })
    .inner
}

impl<'a> AstMutator<Option<Response>> for VisualAst<'a> {
  fn visit_program(&mut self, program: &mut Program) -> Option<Response> {
    program.metadata.accept_mut(self);
    program.sequence.accept_mut(self)
  }

  fn visit_sequence(&mut self, sequence: &mut Sequence) -> Option<Response> {
    sequence.custom_state.with_or_insert_with(
      || SequenceState {
        selected: self.parent_selected,
      },
      |x| x.selected = self.parent_selected,
    );

    self.context.seqs_stack.push(sequence as *mut Sequence);

    if self.parent_selected && self.context.seqs_zoom_stack.len() > 0 {
      let top_most = self.context.seqs_zoom_stack.last().unwrap();
      // if our current sequence is not the top most sequence
      if *top_most != sequence as *mut Sequence {
        if self
          .ui
          .button(emoji("🔍"))
          .on_hover_text("Zoom in.")
          .clicked()
        {
          self.context.seqs_zoom_stack.push(sequence as *mut Sequence);
        }
      }
    }

    for statement in &mut sequence.statements {
      statement.accept_mut(self);
    }

    if self.parent_selected {
      self.context.seqs_stack.pop();
      Some(
        self
          .ui
          .horizontal(|ui| {
            let response = ui.button(emoji("➕")).on_hover_text("Add new statement.");
            if response.clicked() {
              // add a new statement
              sequence.statements.push(Statement::Pipeline(Pipeline {
                blocks: vec![Block {
                  content: BlockContent::Shard(Function {
                    name: Identifier {
                      name: "Pass".into(),
                      namespaces: Vec::new(),
                      custom_state: CustomStateContainer::new(),
                    },
                    params: None,
                    custom_state: CustomStateContainer::new(),
                  }),
                  line_info: None,
                  custom_state: CustomStateContainer::new(),
                }],
              }));

              // and immediately trigger a swap request
              // switch to shard selection
              let new_block = sequence
                .statements
                .last_mut()
                .unwrap()
                .as_pipeline_mut()
                .unwrap()
                .blocks
                .last_mut()
                .unwrap();
              let window_pos = ui
                .ctx()
                .input(|i| i.pointer.hover_pos().unwrap_or_default());
              self.context.swap_state = Some(SwapState::Block(BlockSwapState {
                common: SwapStateCommon {
                  id: Id::new(nanoid!()),
                  receiver: None,
                  window_pos,
                },
                block: new_block as *mut Block,
                search_string: "".into(),
                previous_search_string: "".into(),
                search_results: Vec::new(),
              }));

              self.context.has_changed = true;
            }
            response
          })
          .inner,
      )
    } else {
      self.context.seqs_stack.pop();
      Some(
        self
          .ui
          .allocate_response(egui::Vec2::ZERO, egui::Sense::hover()),
      )
    }
  }

  fn visit_statement(&mut self, statement: &mut Statement) -> Option<Response> {
    self
      .ui
      .with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
        let mut mutator = VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
        let is_expanded = match statement {
          Statement::Assignment(assignment) => {
            assignment.accept_mut(&mut mutator);
            false
          }
          Statement::Pipeline(pipeline) => {
            pipeline.accept_mut(&mut mutator);
            pipeline
              .blocks
              .last()
              .and_then(|x| x.custom_state.with::<BlockState, _, _>(|x| x.selected))
              .unwrap_or(false)
          }
        };
        if self.parent_selected && is_expanded {
          Some(
            ui.horizontal(|ui| {
              let response = ui.button(emoji("➕")).on_hover_text("Add new statement.");
              if response.clicked() {
                // add a new statement
                let pipeline = match statement {
                  Statement::Pipeline(pipeline) => pipeline,
                  _ => unreachable!(),
                };
                pipeline.blocks.push(Block {
                  content: BlockContent::Shard(Function {
                    name: Identifier {
                      name: "Pass".into(),
                      namespaces: Vec::new(),
                      custom_state: CustomStateContainer::new(),
                    },
                    params: None,
                    custom_state: CustomStateContainer::new(),
                  }),
                  line_info: None,
                  custom_state: CustomStateContainer::new(),
                });

                // and immediately trigger a swap request
                // switch to shard selection
                let new_block = pipeline.blocks.last_mut().unwrap();
                let window_pos = ui
                  .ctx()
                  .input(|i| i.pointer.hover_pos().unwrap_or_default());
                self.context.swap_state = Some(SwapState::Block(BlockSwapState {
                  common: SwapStateCommon {
                    id: Id::new(nanoid!()),
                    receiver: None,
                    window_pos,
                  },
                  block: new_block as *mut Block,
                  search_string: "".into(),
                  previous_search_string: "".into(),
                  search_results: Vec::new(),
                }));

                self.context.has_changed = true;
              }
              response
            })
            .inner,
          )
        } else {
          Some(ui.allocate_response(egui::Vec2::ZERO, egui::Sense::hover()))
        }
      })
      .inner
  }

  fn visit_assignment(&mut self, assignment: &mut Assignment) -> Option<Response> {
    let mut combined_response = None;
    let (r_a, r_b) = match assignment {
      Assignment::AssignRef(pipeline, identifier) => (pipeline.accept_mut(self), {
        self.ui.label(chars("="));
        identifier.accept_mut(self)
      }),
      Assignment::AssignSet(pipeline, identifier) => (pipeline.accept_mut(self), {
        self.ui.label(chars(">="));
        identifier.accept_mut(self)
      }),
      Assignment::AssignUpd(pipeline, identifier) => (pipeline.accept_mut(self), {
        self.ui.label(chars(">"));
        identifier.accept_mut(self)
      }),
      Assignment::AssignPush(pipeline, identifier) => (pipeline.accept_mut(self), {
        self.ui.label(chars(">>"));
        identifier.accept_mut(self)
      }),
    };
    if let Some(a) = r_a {
      if let Some(b) = r_b {
        combined_response = Some(a.union(b));
      } else {
        combined_response = Some(a);
      }
    }
    combined_response.map(|x| {
      if x.clicked() {
        self.context.has_changed = true;
      }
      x
    })
  }

  fn visit_pipeline(&mut self, pipeline: &mut Pipeline) -> Option<Response> {
    let mut final_response: Option<Response> = None;
    let mut i = 0;
    while i < pipeline.blocks.len() {
      final_response = match self.visit_block(&mut pipeline.blocks[i]) {
        (BlockAction::Remove, r) => {
          self.context.has_changed = true;
          pipeline.blocks.remove(i);
          // if the blocks are empty, we should remove the pipeline
          if pipeline.blocks.is_empty() {
            let parent_sequence = self.context.seqs_stack.last().unwrap();
            let parent_sequence = unsafe { &mut **parent_sequence };
            // find pipeline index and remove it
            let pipeline_index = parent_sequence
              .statements
              .iter()
              .position(|x| match x {
                Statement::Pipeline(x) => x as *const Pipeline == pipeline as *const Pipeline,
                _ => false,
              })
              .unwrap();
            parent_sequence.statements.remove(pipeline_index);
          }
          r
        }
        (BlockAction::Keep, r) => {
          i += 1;
          r
        }
        (BlockAction::Duplicate, r) => {
          self.context.has_changed = true;
          let block = pipeline.blocks[i].clone();
          block.custom_state.with_mut::<BlockState, _, _>(|x| {
            x.id = Id::new(nanoid!(16));
          });
          pipeline.blocks.insert(i, block);
          i += 2;
          r
        }
        (BlockAction::Swap(block), r) => {
          self.context.has_changed = true;
          pipeline.blocks.get_mut(i).map(|x| {
            let selected = x
              .custom_state
              .with_mut::<BlockState, _, _>(|x| x.selected)
              .unwrap_or(false);
            *x = block;
            x.custom_state.with_or_insert_with(
              || BlockState {
                selected,
                id: Id::new(nanoid!(16)),
              },
              |x| x.selected = selected,
            )
          });
          i += 1;
          r
        }
      };

      // if let Some(response) = &response {
      //   draw_arrow_head(
      //     self.ui,
      //     response.rect,
      //     response.rect.translate(Vec2::new(10.0, 0.0)),
      //   );
      // }
    }

    final_response
  }

  fn visit_block(&mut self, block: &mut Block) -> (BlockAction, Option<Response>) {
    let (selected, id) = {
      block.custom_state.with_or_insert_with(
        || BlockState {
          selected: false,
          id: Id::new(nanoid!(16)),
        },
        |s| (s.selected, s.id),
      )
    };

    let mut action = BlockAction::Keep;
    let mut selected = selected;
    let error = match &block.content {
      BlockContent::Shard(f) | BlockContent::Func(f) => {
        f.custom_state.with::<ShardsError, _, _>(|x| x.clone())
      }
      _ => None,
    };

    if error.is_some() {
      self.ui.style_mut().visuals.window_stroke = egui::Stroke::new(1.5, Color32::RED);
    } else {
      self.ui.style_mut().visuals.window_stroke = self.context.non_error_stroke.unwrap();
    };

    let response = self
      .ui
      .push_id(id, |ui| {
        let response = egui::Frame::window(ui.style())
          .show(ui, |ui| {
            ui.vertical(|ui| {
              if selected {
                ui.horizontal(|ui| {
                  if ui.button(emoji("🔒")).on_hover_text("Collapse.").clicked() {
                    selected = false;
                  }
                  if ui
                    .button(emoji("🔧"))
                    .on_hover_text("Change shard type.")
                    .clicked()
                  {
                    // switch to shard selection
                    let window_pos = ui
                      .ctx()
                      .input(|i| i.pointer.hover_pos().unwrap_or_default());
                    self.context.swap_state = Some(SwapState::Block(BlockSwapState {
                      common: SwapStateCommon {
                        id: Id::new(nanoid!()),
                        receiver: None,
                        window_pos,
                      },
                      block: block as *mut Block,
                      search_string: "".into(),
                      previous_search_string: "".into(),
                      search_results: Vec::new(),
                    }));
                  }
                  if ui.button(emoji("🗐")).on_hover_text("Duplicate.").clicked() {
                    action = BlockAction::Duplicate;
                  }
                  if ui.button(emoji("🗑")).on_hover_text("Delete.").clicked() {
                    action = BlockAction::Remove;
                  }
                });
              }
              match &mut block.content {
                BlockContent::Empty => Some(ui.label(emoji("⬊"))),
                BlockContent::Shard(x) => {
                  let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                  mutator.mutate_shard(x)
                }
                BlockContent::Expr(x) => {
                  ui.style_mut().visuals.widgets.noninteractive.bg_stroke =
                    egui::Stroke::new(1.0, Color32::from_rgb(173, 216, 230));
                  render_shards_group(self.context, ui, selected, x)
                }
                BlockContent::EvalExpr(x) => {
                  ui.style_mut().visuals.widgets.noninteractive.bg_stroke =
                    egui::Stroke::new(1.0, Color32::from_rgb(200, 180, 255));
                  render_shards_group(self.context, ui, selected, x)
                }
                BlockContent::Shards(x) => render_shards_group(self.context, ui, selected, x),
                BlockContent::Const(x) => {
                  let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                  x.accept_mut(&mut mutator)
                }
                BlockContent::TakeTable(x, y) => {
                  let new_value = transform_take_table(x, y);
                  block.content = BlockContent::Expr(new_value);
                  let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                  block.accept_mut(&mut mutator).1
                }
                BlockContent::TakeSeq(x, y) => {
                  let new_value = transform_take_seq(x, y);
                  block.content = BlockContent::Expr(new_value);
                  let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                  block.accept_mut(&mut mutator).1
                }
                BlockContent::Func(x) => match x.name.name.as_str() {
                  "color" => {
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    let response = mutator.mutate_color(x);
                    if response.as_ref().unwrap().changed() {
                      self.context.has_changed = true;
                    }
                    response
                  }
                  "i2" => {
                    let y = ints_func_to_ints(x);
                    block.content = BlockContent::Const(Value::Int2(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "i3" => {
                    let y = ints_func_to_ints(x);
                    block.content = BlockContent::Const(Value::Int3(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "i4" => {
                    let y = ints_func_to_ints(x);
                    block.content = BlockContent::Const(Value::Int4(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "i8" => {
                    let y = ints_func_to_ints(x);
                    block.content = BlockContent::Const(Value::Int8(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "i16" => {
                    let y = ints_func_to_ints(x);
                    block.content = BlockContent::Const(Value::Int16(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "f2" => {
                    let y = floats_func_to_floats(x);
                    block.content = BlockContent::Const(Value::Float2(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "f3" => {
                    let y = floats_func_to_floats(x);
                    block.content = BlockContent::Const(Value::Float3(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  "f4" => {
                    let y = floats_func_to_floats(x);
                    block.content = BlockContent::Const(Value::Float4(y));
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    block.accept_mut(&mut mutator).1
                  }
                  _ => {
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    mutator.mutate_shard(x)
                  }
                },
                BlockContent::Program(x) => {
                  ui.group(|ui| {
                    let mut mutator = VisualAst::with_parent_selected(self.context, ui, selected);
                    x.accept_mut(&mut mutator)
                  })
                  .inner
                }
              }
            })
            .inner
          })
          .response;

        if !selected {
          if ui
            .interact(response.rect, response.id, Sense::click())
            .clicked()
          {
            selected = !selected;
          }
        }
        response
      })
      .inner;

    block
      .custom_state
      .with_mut::<BlockState, _, _>(|x| x.selected = selected);

    if let Some(SwapState::Block(swap_state)) = &mut self.context.swap_state {
      if block as *mut _ == swap_state.block {
        match select_shard_modal(self.ui, swap_state) {
          SwapStateResult::Done(new_block) => {
            action = BlockAction::Swap(new_block);
            self.context.swap_state = None;
            self.context.has_changed = true;
          }
          SwapStateResult::Close => {
            self.context.swap_state = None;
          }
          _ => {}
        }
      }
    }

    (action, Some(response))
  }

  fn visit_function(&mut self, _function: &mut Function) -> Option<Response> {
    unreachable!("Function should not be visited directly.")
  }

  fn visit_param(&mut self, param: &mut Param) -> Option<Response> {
    if let Some(name) = &param.name {
      self.ui.label(name.as_str());
    }
    param.value.accept_mut(self)
  }

  fn visit_identifier(&mut self, identifier: &mut Identifier) -> Option<Response> {
    // kiss, for now we support only 1 level of namespace properly, in eval and most of all fbl.
    let response = self
      .ui
      .horizontal(|ui| {
        if identifier.namespaces.is_empty() {
          if ui
            .button(emoji("➕"))
            .on_hover_text("Add namespace.")
            .clicked()
          {
            identifier.namespaces.push("default".into());
            self.context.has_changed = true;
          }
          let x = identifier.name.to_mut();
          egui::TextEdit::singleline(x)
            .clip_text(false)
            .desired_width(0.0)
            .ui(ui)
        } else {
          let first = &mut identifier.namespaces[0];
          let len = first.len();
          let first = first.to_mut();
          if ui
            .horizontal(|ui| {
              let remove = if ui
                .button(emoji("🗑"))
                .on_hover_text("Remove Namespace")
                .clicked()
              {
                self.context.has_changed = true;
                true
              } else {
                false
              };

              if egui::TextEdit::singleline(first)
                .clip_text(false)
                .desired_width(if len == 0 { 70.0 } else { 0.0 })
                .hint_text("namespace")
                .ui(ui)
                .changed()
              {
                self.context.has_changed = true;
              }
              remove
            })
            .inner
          {
            identifier.namespaces.clear();
          }
          let x = identifier.name.to_mut();
          egui::TextEdit::singleline(x)
            .clip_text(false)
            .desired_width(0.0)
            .ui(ui)
        }
      })
      .inner;
    Some(response).map(|x| {
      if x.changed() {
        self.context.has_changed = true;
      }
      x
    })
  }

  fn visit_value(&mut self, value: &mut Value) -> Option<Response> {
    self
      .ui
      .push_id(egui::Id::new(value as *const _), |ui| {
        match value {
          Value::None(_) => Some(ui.label("None")),
          Value::Identifier(x) => {
            let mut mutator = VisualAst::with_parent_selected(self.context, ui, true);
            x.accept_mut(&mut mutator)
          }
          Value::Boolean(x) => Some(ui.checkbox(x, if *x { "true" } else { "false" })),
          Value::Enum(x, y) => {
            let enum_data = get_global_map();
            let enum_data = enum_data.0.get_fast_static("enums").as_table().unwrap();
            let x_var = Var::ephemeral_string(x);
            let enum_data = enum_data.get(x_var).map(|x| x.as_table().unwrap());
            if let Some(enum_data) = enum_data {
              let labels = enum_data.get_fast_static("labels").as_seq().unwrap();
              // render the first part as a constant label, while the second as a single choice select
              let x = get_formatted_name_ref(x);
              let response = ui.label(x.as_str());

              // render labels in a combo box
              let mut selected_index = labels
                .iter()
                .position(|label| label == Var::ephemeral_string(y))
                .unwrap_or(0);
              let previous_index = selected_index;
              let selected: &str = labels[selected_index].as_ref().try_into().unwrap();
              let selected = get_formatted_name_ref(selected);
              egui::ComboBox::from_label("")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                  for (index, label) in labels.iter().enumerate() {
                    let label: &str = label.as_ref().try_into().unwrap();
                    ui.selectable_value(&mut selected_index, index, label);
                  }
                });

              if previous_index != selected_index {
                let selected: &str = labels[selected_index].as_ref().try_into().unwrap();
                *y = selected.into();
              }

              Some(response)
            } else {
              Some(ui.label("Invalid enum"))
            }
          }
          Value::Number(x) => match x {
            Number::Integer(x) => Some(ui.add(CustomDragValue::new(x))),
            Number::Float(x) => Some(ui.add(CustomDragValue::new(x))),
            Number::Hexadecimal(x) => {
              // we need a mini embedded text editor
              let prev_value = x.clone();
              let response = TextEdit::singleline(x.to_mut())
                .clip_text(false)
                .desired_width(0.0)
                .ui(ui);

              if response.changed() {
                // ensure that the string is a valid hexadecimal number, if not revert to previous value
                // consider we always prefix with 0x, slice off the 0x prefix
                // parse the string as a u64 in hexadecimal format
                let parsed = u64::from_str_radix(&x[2..], 16);
                if parsed.is_err() {
                  *x = prev_value;
                }
              }
              Some(response)
            }
          },
          Value::String(x) => {
            // if long we should use a multiline text editor
            // if short we should use a single line text editor
            let x = x.to_mut();
            Some(if x.len() > 32 {
              ui.text_edit_multiline(x)
            } else {
              let len = x.len();
              TextEdit::singleline(x)
                .hint_text("String")
                .clip_text(false)
                .desired_width(if len == 0 { 40.0 } else { 0.0 })
                .ui(ui)
            })
          }
          Value::Bytes(x) => {
            let bytes = x.to_mut();
            let mut len = bytes.len();
            let response = ui.label(format!("Bytes (len: {})", len));
            if self.parent_selected {
              ui.add(CustomDragValue::new(&mut len));

              // check if we need to resize
              if len != bytes.len() {
                bytes.resize(len, 0);
              }

              let mem_range = 0..bytes.len();

              if len > 0 {
                use egui_memory_editor::MemoryEditor;
                let mut is_open = true;
                let mut memory_editor = MemoryEditor::new().with_address_range("Memory", mem_range);
                // Show a read-only window
                memory_editor.window_ui(
                  ui.ctx(),
                  &mut is_open,
                  bytes,
                  |mem, addr| mem[addr].into(),
                  |mem, addr, val| mem[addr] = val,
                );
              }
            }
            Some(response)
          }
          Value::Int2(x) => {
            // just 2 ints wrapped in a horizontal layout
            Some(
              ui.horizontal(|ui| {
                ui.add(CustomDragValue::new(&mut x[0]));
                ui.add(CustomDragValue::new(&mut x[1]));
              })
              .response,
            )
          }
          Value::Int3(x) => {
            // just 3 ints wrapped in a horizontal layout
            Some(
              ui.horizontal(|ui| {
                ui.add(CustomDragValue::new(&mut x[0]));
                ui.add(CustomDragValue::new(&mut x[1]));
                ui.add(CustomDragValue::new(&mut x[2]));
              })
              .response,
            )
          }
          Value::Int4(x) => {
            // just 4 ints wrapped in a horizontal layout
            Some(
              ui.horizontal(|ui| {
                ui.add(CustomDragValue::new(&mut x[0]));
                ui.add(CustomDragValue::new(&mut x[1]));
                ui.add(CustomDragValue::new(&mut x[2]));
                ui.add(CustomDragValue::new(&mut x[3]));
              })
              .response,
            )
          }
          Value::Int8(x) => {
            // just 8 ints wrapped in 2 horizontal layouts
            // First 4
            let response = ui
              .horizontal(|ui| {
                for i in 0..4 {
                  ui.add(CustomDragValue::new(&mut x[i]));
                }
              })
              .response;
            Some(
              response.union(
                ui.horizontal(|ui| {
                  for i in 4..8 {
                    ui.add(CustomDragValue::new(&mut x[i]));
                  }
                })
                .response,
              ),
            )
          }
          Value::Int16(x) => {
            // just 16 ints wrapped in 8 horizontal layout
            // First 4
            let response = ui
              .horizontal(|ui| {
                for i in 0..4 {
                  ui.add(CustomDragValue::new(&mut x[i]));
                }
              })
              .response;
            // Second 4
            let response = response.union(
              ui.horizontal(|ui| {
                for i in 4..8 {
                  ui.add(CustomDragValue::new(&mut x[i]));
                }
              })
              .response,
            );
            // Third 4
            let response = response.union(
              ui.horizontal(|ui| {
                for i in 8..12 {
                  ui.add(CustomDragValue::new(&mut x[i]));
                }
              })
              .response,
            );
            // Fourth 4
            let response = response.union(
              ui.horizontal(|ui| {
                for i in 12..16 {
                  ui.add(CustomDragValue::new(&mut x[i]));
                }
              })
              .response,
            );
            Some(response)
          }
          Value::Float2(x) => {
            // just 2 floats wrapped in a horizontal layout
            Some(
              ui.horizontal(|ui| {
                ui.add(CustomDragValue::new(&mut x[0]));
                ui.add(CustomDragValue::new(&mut x[1]));
              })
              .response,
            )
          }
          Value::Float3(x) => {
            // just 3 floats wrapped in a horizontal layout
            Some(
              ui.horizontal(|ui| {
                ui.add(CustomDragValue::new(&mut x[0]));
                ui.add(CustomDragValue::new(&mut x[1]));
                ui.add(CustomDragValue::new(&mut x[2]));
              })
              .response,
            )
          }
          Value::Float4(x) => {
            // just 4 floats wrapped in a horizontal layout
            Some(
              ui.horizontal(|ui| {
                ui.add(CustomDragValue::new(&mut x[0]));
                ui.add(CustomDragValue::new(&mut x[1]));
                ui.add(CustomDragValue::new(&mut x[2]));
                ui.add(CustomDragValue::new(&mut x[3]));
              })
              .response,
            )
          }
          Value::Seq(x) => {
            let len = x.len();
            let response = ui.label(format!("Seq (len: {})", len));
            if self.parent_selected {
              let mut idx = 0;
              x.retain_mut(|value| {
                let mut to_keep = true;
                ui.horizontal(|ui| {
                  ui.label(format!("{}:", idx));
                  idx += 1;
                  ui.vertical(|ui| {
                    let mut mutator =
                      VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                    let response = value.accept_mut(&mut mutator).unwrap();
                    response.context_menu(|ui| {
                      // change type
                      if ui
                        .button(emoji("Change 🔧"))
                        .on_hover_text("Change value type.")
                        .clicked()
                      {
                        // open a dialog to change the value
                        ui.close_menu();
                      }
                      if ui
                        .button(emoji("Remove 🗑"))
                        .on_hover_text("Remove value.")
                        .clicked()
                      {
                        to_keep = false;
                        ui.close_menu();
                      }
                    });
                  });
                });
                to_keep
              });
              let response = ui.button(emoji("➕")).on_hover_text("Add new value.");
              if response.clicked() {
                x.push(Value::None(()));
              }
            } else {
              // give a peek of FIRST..LAST
              if !x.is_empty() {
                ui.horizontal(|ui| {
                  // first is qed
                  let first = x.first_mut().unwrap();
                  let mut mutator =
                    VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                  first.accept_mut(&mut mutator).unwrap();
                  // no ctx menu this time
                  // second is not qed
                  if x.len() > 1 {
                    if let Some(second) = x.last_mut() {
                      ui.label(emoji("⬌"));
                      let mut mutator =
                        VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                      second.accept_mut(&mut mutator).unwrap();
                      // no ctx menu this time
                    }
                  }
                });
              }
            }
            Some(response)
          }
          Value::Table(x) => {
            let len = x.len();
            let response = ui.label(format!("Table (len: {})", len));
            if self.parent_selected {
              // like sequence but key value pairs
              ui.label("Key-Value pairs");
              x.retain_mut(|(key, value)| {
                let mut to_keep = true;
                ui.horizontal(|ui| {
                  let mut mutator =
                    VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                  let response = key.accept_mut(&mut mutator).unwrap();
                  response.context_menu(|ui| {
                    // change type
                    if ui
                      .button(emoji("Change 🔧"))
                      .on_hover_text("Change key type.")
                      .clicked()
                    {
                      // open a dialog to change the value
                      ui.close_menu();
                    }
                    if ui
                      .button(emoji("Remove 🗑"))
                      .on_hover_text("Remove key.")
                      .clicked()
                    {
                      to_keep = false;
                      ui.close_menu();
                    }
                  });
                  ui.vertical(|ui| {
                    let mut mutator =
                      VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                    let response = value.accept_mut(&mut mutator).unwrap();
                    response.context_menu(|ui| {
                      // change type
                      if ui
                        .button(emoji("Change 🔧"))
                        .on_hover_text("Change value type.")
                        .clicked()
                      {
                        // open a dialog to change the value
                        ui.close_menu();
                      }
                      if ui
                        .button(emoji("Remove 🗑"))
                        .on_hover_text("Remove value.")
                        .clicked()
                      {
                        to_keep = false;
                        ui.close_menu();
                      }
                    });
                  });
                });
                to_keep
              });
              let response = ui
                .button(emoji("➕"))
                .on_hover_text("Add new key value pair.");
              if response.clicked() {
                x.push((Value::None(()), Value::None(())));
              }
            } else {
              // like seq but just preview first and last key (not values)
              if !x.is_empty() {
                ui.horizontal(|ui| {
                  // first is qed
                  let first = x.first_mut().unwrap();
                  let mut mutator =
                    VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                  first.0.accept_mut(&mut mutator).unwrap();
                  // no ctx menu this time
                  // second is not qed
                  if x.len() > 1 {
                    if let Some(second) = x.last_mut() {
                      ui.label(emoji("⬌"));
                      let mut mutator =
                        VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
                      second.0.accept_mut(&mut mutator).unwrap();
                      // no ctx menu this time
                    }
                  }
                });
              }
            }
            Some(response)
          }
          Value::Shard(x) => {
            /*

            ### Don’t try too hard to satisfy TEXT version.
            Such as eliding `{}` when single shard or Omitting params which are at default value, etc
            We can have a pass when we turn AST into text to apply such EYE CANDY.

            Turn it into a Shards value instead
            */
            let shard = x.clone();
            *value = Value::Shards(Sequence {
              statements: vec![Statement::Pipeline(Pipeline {
                blocks: vec![Block {
                  content: BlockContent::Shard(shard),
                  line_info: None,
                  custom_state: CustomStateContainer::new(),
                }],
              })],
              custom_state: CustomStateContainer::new(),
            });
            let mut mutator =
              VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
            mutator.visit_value(value)
          }
          Value::Shards(x) => {
            ui.group(|ui| {
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              x.accept_mut(&mut mutator)
            })
            .inner
          }
          Value::EvalExpr(x) => {
            ui.style_mut().visuals.widgets.noninteractive.bg_stroke =
              egui::Stroke::new(1.0, Color32::from_rgb(200, 180, 255));
            ui.group(|ui| {
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              x.accept_mut(&mut mutator)
            })
            .inner
          }
          Value::Expr(x) => {
            ui.style_mut().visuals.widgets.noninteractive.bg_stroke =
              egui::Stroke::new(1.0, Color32::from_rgb(173, 216, 230));
            ui.group(|ui| {
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              x.accept_mut(&mut mutator)
            })
            .inner
          }
          Value::TakeTable(x, y) => {
            let new_value = transform_take_table(x, y);
            *value = Value::Expr(new_value);
            let mut mutator =
              VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
            mutator.visit_value(value)
          }
          Value::TakeSeq(x, y) => {
            let new_value = transform_take_seq(x, y);
            *value = Value::Expr(new_value);
            let mut mutator =
              VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
            mutator.visit_value(value)
          }
          Value::Func(x) => match x.name.name.as_str() {
            "color" => {
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.mutate_color(x)
            }
            "i2" => {
              let y = ints_func_to_ints(x);
              *value = Value::Int2(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "i3" => {
              let y = ints_func_to_ints(x);
              *value = Value::Int3(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "i4" => {
              let y = ints_func_to_ints(x);
              *value = Value::Int4(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "i8" => {
              let y = ints_func_to_ints(x);
              *value = Value::Int4(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "i16" => {
              let y = ints_func_to_ints(x);
              *value = Value::Int4(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "f2" => {
              let y = floats_func_to_floats(x);
              *value = Value::Float2(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "f3" => {
              let y = floats_func_to_floats(x);
              *value = Value::Float3(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            "f4" => {
              let y = floats_func_to_floats(x);
              *value = Value::Float4(y);
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.visit_value(value)
            }
            _ => {
              let mut mutator =
                VisualAst::with_parent_selected(self.context, ui, self.parent_selected);
              mutator.mutate_shard(x)
            }
          },
        }
      })
      .inner
      .map(|x| {
        if x.changed() {
          self.context.has_changed = true;
        }
        x
      })
  }

  fn visit_metadata(&mut self, _metadata: &mut Metadata) -> Option<Response> {
    None
  }
}

fn func_to_numbers<T, const SIZE: usize>(x: &mut Function) -> [T; SIZE]
where
  T: FromPrimitive + Copy + Zero,
{
  // Initialize array with zeros
  let mut y = [T::zero(); SIZE];

  for (i, param) in x.params.as_mut().unwrap().iter_mut().take(SIZE).enumerate() {
    y[i] = match &param.value {
      Value::Number(num) => match num {
        Number::Integer(x) => T::from_i64(*x).unwrap_or_else(T::zero),
        Number::Float(x) => T::from_f64(*x).unwrap_or_else(T::zero),
        Number::Hexadecimal(x) => {
          T::from_i64(i64::from_str_radix(&x[2..], 16).unwrap_or(0)).unwrap_or_else(T::zero)
        }
      },
      _ => T::zero(),
    };
  }
  y
}

// Specialized function for integer types
fn ints_func_to_ints<T, const SIZE: usize>(x: &mut Function) -> [T; SIZE]
where
  T: PrimInt + FromPrimitive,
{
  func_to_numbers(x)
}

// Specialized function for float types
fn floats_func_to_floats<T, const SIZE: usize>(x: &mut Function) -> [T; SIZE]
where
  T: Float + FromPrimitive,
{
  func_to_numbers(x)
}

fn transform_take_table(x: &mut Identifier, y: &mut Vec<RcStrWrapper>) -> Sequence {
  // substitute with a Expr sequence
  let mut blocks = vec![Block {
    content: BlockContent::Shard(Function {
      name: Identifier {
        name: "Get".into(),
        namespaces: Vec::new(),
        custom_state: CustomStateContainer::new(),
      },
      params: Some(vec![Param {
        name: None,
        value: Value::Identifier(x.clone()),
        custom_state: CustomStateContainer::new(),
        is_default: Some(false),
      }]),
      custom_state: CustomStateContainer::new(),
    }),
    line_info: None,
    custom_state: CustomStateContainer::new(),
  }];
  for y in y.iter() {
    // add Take shard for each
    blocks.push(Block {
      content: BlockContent::Shard(Function {
        name: Identifier {
          name: "Take".into(),
          namespaces: Vec::new(),
          custom_state: CustomStateContainer::new(),
        },
        params: Some(vec![Param {
          name: None,
          value: Value::String(y.clone()),
          custom_state: CustomStateContainer::new(),
          is_default: Some(false),
        }]),
        custom_state: CustomStateContainer::new(),
      }),
      line_info: None,
      custom_state: CustomStateContainer::new(),
    });
  }
  let new_value = Sequence {
    statements: vec![Statement::Pipeline(Pipeline { blocks })],
    custom_state: CustomStateContainer::new(),
  };
  new_value
}

fn transform_take_seq(x: &mut Identifier, y: &mut Vec<u32>) -> Sequence {
  // same as take table but integer keys
  // substitute with a Expr sequence
  let mut blocks = vec![Block {
    content: BlockContent::Shard(Function {
      name: Identifier {
        name: "Get".into(),
        namespaces: Vec::new(),
        custom_state: CustomStateContainer::new(),
      },
      params: Some(vec![Param {
        name: None,
        value: Value::Identifier(x.clone()),
        custom_state: CustomStateContainer::new(),
        is_default: Some(false),
      }]),
      custom_state: CustomStateContainer::new(),
    }),
    line_info: None,
    custom_state: CustomStateContainer::new(),
  }];
  for y in y.iter() {
    // add Take shard for each
    blocks.push(Block {
      content: BlockContent::Shard(Function {
        name: Identifier {
          name: "Take".into(),
          namespaces: Vec::new(),
          custom_state: CustomStateContainer::new(),
        },
        params: Some(vec![Param {
          name: None,
          value: Value::Number(Number::Integer(*y as i64)),
          custom_state: CustomStateContainer::new(),
          is_default: Some(false),
        }]),
        custom_state: CustomStateContainer::new(),
      }),
      line_info: None,
      custom_state: CustomStateContainer::new(),
    });
  }
  let new_value = Sequence {
    statements: vec![Statement::Pipeline(Pipeline { blocks })],
    custom_state: CustomStateContainer::new(),
  };
  new_value
}

fn render_shards_group(
  context: &mut Context,
  ui: &mut Ui,
  selected: bool,
  x: &mut Sequence,
) -> Option<Response> {
  ui.group(|ui| {
    // if not selected, let's render just first and last shard as previews
    if selected {
      let mut mutator = VisualAst::with_parent_selected(context, ui, selected);
      x.accept_mut(&mut mutator)
    } else {
      Some(
        ui.horizontal(|ui| {
          if let Some(first) = get_first_shard_ref(x) {
            let mut mutator = VisualAst::with_parent_selected(context, ui, selected);
            mutator.mutate_shard(first);
          }
          if let Some(last) = get_last_shard_ref(x) {
            ui.label(emoji("⬌"));
            let mut mutator = VisualAst::with_parent_selected(context, ui, selected);
            mutator.mutate_shard(last);
          }
        })
        .response,
      )
    }
  })
  .inner
}

#[derive(shards::shard)]
#[shard_info("UI.Shards", "A Shards program AST visual editor.")]
pub struct UIShardsShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_warmup]
  parents: ParamVar,

  #[shard_param("AST", "The Shards AST object to edit in real time, this shard will manipulate and edit this variable in place.", [*AST_VAR_TYPE])]
  ast_object: ParamVar,

  ast: ClonedVar,
  context: Context,
}

impl Default for UIShardsShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      parents: ParamVar::new_named(PARENTS_UI_NAME),
      ast_object: ParamVar::default(),
      ast: ClonedVar::default(),
      context: Context {
        swap_state: None,
        seqs_zoom_stack: Vec::new(),
        seqs_stack: Vec::new(),
        has_changed: false,
        non_error_stroke: None,
      },
    }
  }
}

#[shards::shard_impl]
impl Shard for UIShardsShard {
  fn input_types(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn input_help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "No input required, the AST is manipulated in place."
    ))
  }

  fn output_types(&mut self) -> &Types {
    &BOOL_TYPES
  }

  fn output_help(&mut self) -> OptionalString {
    OptionalString(shccstr!(
      "True if the AST has been modified, false otherwise."
    ))
  }

  fn warmup(&mut self, ctx: &ShardsContext) -> Result<(), &str> {
    self.warmup_helper(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&ShardsContext>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;

    self.ast = ClonedVar::default(); // drop the AST object

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    if !self.ast_object.is_variable() {
      return Err("AST object is not a variable");
    }

    require_parents(&mut self.required);

    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &ShardsContext, _input: &Var) -> Result<Option<Var>, &str> {
    let ast_var = self.ast_object.get();
    let ast = unsafe { &mut *Var::from_ref_counted_object::<Program>(ast_var, &AST_TYPE)? };
    if self.ast.0 != *ast_var {
      self.ast = ast_var.into(); // take ownership/increase refcount
      self.context.seqs_zoom_stack.clear();
      let seq_ptr = &mut ast.sequence as *mut Sequence;
      self.context.seqs_zoom_stack.push(seq_ptr);
      self.context.seqs_stack.clear();
      self.context.swap_state = None;
    }

    self.context.has_changed = false;

    let ui = get_current_parent_opt(self.parents.get())?.ok_or("No parent UI")?;
    self.context.non_error_stroke = Some(ui.style().visuals.window_stroke);

    // // Set the minimum and maximum size of the UI
    // // This allows us to have a fully user controlled UI/Window
    // let x = ui.available_size_before_wrap().x;
    // let y = ui.available_size_before_wrap().y;
    // let min_max = egui::Vec2::new(x, y);
    // ui.set_min_size(min_max);
    // ui.set_max_size(min_max);

    egui::ScrollArea::new([true, true]).show(ui, |ui| {
      // go backward / zoom out
      if self.context.seqs_zoom_stack.len() > 1 {
        if ui.button(emoji("⬅")).on_hover_text("Zoom out.").clicked() {
          self.context.seqs_zoom_stack.pop();
        }
      }
      let root = unsafe { &mut **self.context.seqs_zoom_stack.last_mut().unwrap() };
      let mut mutator = VisualAst::with_parent_selected(
        &mut self.context,
        ui,
        root
          .custom_state
          .with_or_insert_with(|| SequenceState { selected: true }, |x| x.selected),
      );
      root.accept_mut(&mut mutator);
    });

    Ok(Some(self.context.has_changed.into()))
  }
}

pub fn register_shards() {
  register_shard::<UIShardsShard>();
}
