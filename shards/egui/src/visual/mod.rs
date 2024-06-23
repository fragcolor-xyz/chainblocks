// prevent upper case globals
#![allow(non_upper_case_globals)]

use directory::get_global_visual_shs_channel_sender;
use egui::*;
use egui_extras::{Column, TableBuilder};
use nanoid::nanoid;
use std::{any::Any, sync::mpsc};

use crate::{
  util::{get_current_parent_opt, require_parents},
  widgets::drag_value::CustomDragValue,
  PARENTS_UI_NAME,
};
use shards::{
  core::register_shard,
  shard::Shard,
  types::{
    ClonedVar, Context as ShardsContext, ExposedTypes, InstanceData, ParamVar, Type, Types, Var,
    NONE_TYPES,
  },
  SHType_Bool, SHType_Bytes, SHType_ContextVar, SHType_Enum, SHType_Float, SHType_Float2,
  SHType_Float3, SHType_Float4, SHType_Int, SHType_Int16, SHType_Int2, SHType_Int3, SHType_Int4,
  SHType_Int8, SHType_None, SHType_Seq, SHType_ShardRef, SHType_String, SHType_Table, SHType_Wire,
};

use pest::Parser;

use shards_lang::{ast::*, ast_visitor::*, ParamHelperMut};

mod directory;

fn draw_arrow_head(ui: &mut egui::Ui, from: Rect, to: Rect) {
  let painter = ui.painter();

  // Calculate arrow position
  let arrow_x = (from.right() + to.left()) / 2.0;
  let arrow_y = from.center().y; // Align with the vertical center of the frames
  let arrow_pos = pos2(arrow_x, arrow_y);

  // Arrow dimensions
  let arrow_width = 8.0;
  let arrow_height = 12.0;

  // Calculate arrow points
  let tip = arrow_pos + Vec2::new(arrow_height / 2.0, 0.0);
  let left = arrow_pos + Vec2::new(-arrow_height / 2.0, -arrow_width / 2.0);
  let right = arrow_pos + Vec2::new(-arrow_height / 2.0, arrow_width / 2.0);

  // Draw arrow
  painter.add(egui::Shape::convex_polygon(
    vec![tip, left, right],
    egui::Color32::WHITE,
    Stroke::new(1.0, egui::Color32::WHITE),
  ));
}

fn var_to_value(var: &Var) -> Result<Value, String> {
  match var.valueType {
    SHType_None => Ok(Value::None),
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
    SHType_Enum => Ok(Value::String("TODO".into())),
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
    SHType_ShardRef => Ok(Value::String("TODO".into())),
    SHType_Wire => Ok(Value::String("TODO".into())),
    SHType_ContextVar => {
      let string = unsafe { var.payload.__bindgen_anon_1.string };
      let string_slice =
        unsafe { std::slice::from_raw_parts(string.elements as *const u8, string.len as usize) };
      let string = unsafe { std::str::from_utf8_unchecked(string_slice) };
      Ok(Value::Identifier(Identifier {
        name: string.into(),
        namespaces: Vec::new(),
      }))
    }
    _ => Err(format!("Unsupported Var type: {:?}", var.valueType)),
  }
}

#[derive(Debug, Clone, PartialEq)]
struct BlockState {
  selected: bool,
  id: Id,
}

impl_custom_any!(BlockState);

#[derive(Debug, Clone, PartialEq)]
struct FunctionState {
  params_sorted: bool,
  standalone_state: BlockState, // used when a single shard is used as parameter
  receiver: Option<UniqueReceiver<ClonedVar>>,
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

impl_custom_any!(FunctionState);

pub struct VisualAst<'a> {
  ui: &'a mut Ui,
  parent_selected: bool,
}

// Helper function to determine if a color is light
fn is_light_color(color: egui::Color32) -> bool {
  let brightness = 0.299 * color.r() as f32 + 0.587 * color.g() as f32 + 0.114 * color.b() as f32;
  brightness > 128.0
}

impl<'a> VisualAst<'a> {
  pub fn new(ui: &'a mut Ui) -> Self {
    VisualAst {
      ui,
      parent_selected: false,
    }
  }

  pub fn with_parent_selected(ui: &'a mut Ui, parent_selected: bool) -> Self {
    VisualAst {
      ui,
      parent_selected,
    }
  }

  pub fn render(&mut self, ast: &mut Sequence) -> Option<Response> {
    ast.accept_mut(self)
  }

  fn mutate_shard(&mut self, x: &mut Function) -> Option<Response> {
    let selected = self.parent_selected;

    let state = x.get_or_insert_custom_state(|| FunctionState {
      params_sorted: false,
      standalone_state: BlockState {
        selected: false,
        id: Id::new(nanoid!(16)),
      },
      receiver: None,
    });
    let params_sorted = state.params_sorted;

    // check if we have a result from a pending operation
    let has_result = state
      .receiver
      .as_mut()
      .and_then(|r| r.get_mut())
      .map(|r| r.try_recv());

    if let Some(Ok(result)) = has_result {
      shlog_debug!("Got result: {:?}", result);
      // reset the receiver
      state.receiver = None;
    }

    let directory = directory::get_global_map();
    let shards = directory.0.get_fast_static("shards");
    let shards = shards.as_table().unwrap();
    let shard_name = x.name.name.as_str();
    let shard_name_var = Var::ephemeral_string(shard_name);
    if let Some(shard) = shards.get(shard_name_var) {
      let shard = shard.as_table().unwrap();

      let help_text: &str = shard.get_fast_static("help").try_into().unwrap();
      let help_text = if help_text.is_empty() {
        "No help text provided."
      } else {
        help_text
      };

      let color = shard.get_fast_static("color");
      let color = Var::color_bytes(&color).unwrap();
      let bg_color = egui::Color32::from_rgb(color.0, color.1, color.2);

      // Determine text color based on background brightness
      let text_color = if is_light_color(bg_color) {
        egui::Color32::BLACK
      } else {
        egui::Color32::WHITE
      };

      let response = egui::CollapsingHeader::new(
        egui::RichText::new(shard_name)
          .size(14.0)
          .strong()
          .family(egui::FontFamily::Monospace)
          .background_color(bg_color)
          .color(text_color),
      )
      .open(Some(selected))
      .default_open(false)
      .show_background(true)
      .show(self.ui, |ui| {
        let params = shard.get_fast_static("parameters").as_seq().unwrap();
        if !params.is_empty() {
          let mut params_copy = if !params_sorted {
            // only if we really need to sort
            let params = x.params.clone();
            // reset current params
            x.params = Some(Vec::new());
            Some(params)
          } else {
            None
          };

          // helper if needed as well
          let mut helper = params_copy
            .as_mut()
            .map(|params| params.as_mut().map(|x| ParamHelperMut::new(x)));

          if !params_sorted {
            for (idx, param) in params.into_iter().enumerate() {
              let param = param.as_table().unwrap();
              let name: &str = param.get_fast_static("name").try_into().unwrap();

              let new_param = helper
                .as_mut()
                .and_then(|h| h.as_mut())
                .and_then(|ast| ast.get_param_by_name_or_index_mut(name, idx))
                .cloned();

              let param_to_add = new_param.unwrap_or_else(|| {
                let default_value = param.get_fast_static("default");
                Param {
                  name: Some(name.into()),
                  value: var_to_value(&default_value).unwrap(),
                }
              });

              x.params.as_mut().unwrap().push(param_to_add);
            }

            // set flag to true
            x.get_custom_state::<FunctionState>().unwrap().params_sorted = true;
          }

          for (idx, param) in params.into_iter().enumerate() {
            let param = param.as_table().unwrap();
            let name: &str = param.get_fast_static("name").try_into().unwrap();
            let help_text: &str = param.get_fast_static("help").try_into().unwrap();
            let help_text = if help_text.is_empty() {
              "No help text provided."
            } else {
              help_text
            };
            egui::CollapsingHeader::new(name)
              .default_open(false)
              .show(ui, |ui| {
                ui.horizontal(|ui| {
                  // button to reset to default
                  if ui
                    .button("🔄")
                    .on_hover_text("Reset to default value.")
                    .clicked()
                  {
                    // reset to default
                    let default_value = param.get_fast_static("default");
                    x.params.as_mut().map(|params| {
                      params[idx].value = var_to_value(&default_value).unwrap();
                    });
                  }
                  if ui
                    .button("🔧")
                    .on_hover_text("Change value type.")
                    .clicked()
                  {
                    // open a dialog to change the value
                    let (sender, receiver) = mpsc::channel();
                    let query = Var::ephemeral_string("").into();
                    get_global_visual_shs_channel_sender()
                      .send((query, sender))
                      .unwrap();
                    x.get_custom_state::<FunctionState>().unwrap().receiver =
                      Some(UniqueReceiver::new(receiver));
                  }
                });

                // draw the value
                x.params.as_mut().map(|params| {
                  let mut mutator = VisualAst::with_parent_selected(ui, selected);
                  params[idx].value.accept_mut(&mut mutator);
                });
              })
              .header_response
              .on_hover_text(help_text);
          }
        }
      });
      let response = response.header_response.on_hover_text(help_text);
      Some(response)
    } else {
      Some(self.ui.label("Unknown shard"))
    }
  }

  fn render_sub_window<F>(
    &mut self,
    id: Id,
    selected: bool,
    content_renderer: F,
  ) -> (bool, BlockAction, Option<Response>)
  where
    F: FnOnce(&mut Ui) -> Option<Response>,
  {
    let mut action = BlockAction::Keep;
    let mut selected = selected;

    let response = self
      .ui
      .push_id(id, |ui| {
        let response = egui::Frame::window(ui.style())
          .show(ui, |ui| {
            ui.vertical(|ui| {
              if selected {
                ui.horizontal(|ui| {
                  if ui.button("🔒").clicked() {
                    selected = false;
                  }
                  if ui.button("S").clicked() {
                    // switch
                  }
                  if ui.button("D").clicked() {
                    action = BlockAction::Duplicate;
                  }
                  if ui.button("X").clicked() {
                    action = BlockAction::Remove;
                  }
                });
              }
              content_renderer(ui)
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

    (selected, action, Some(response))
  }
}

impl<'a> AstMutator<Option<Response>> for VisualAst<'a> {
  fn visit_program(&mut self, program: &mut Program) -> Option<Response> {
    program.metadata.accept_mut(self);
    program.sequence.accept_mut(self)
  }

  fn visit_sequence(&mut self, sequence: &mut Sequence) -> Option<Response> {
    for statement in &mut sequence.statements {
      statement.accept_mut(self);
    }
    Some(self.ui.button("➕").on_hover_text("Add new statement."))
  }

  fn visit_statement(&mut self, statement: &mut Statement) -> Option<Response> {
    self
      .ui
      .horizontal(|ui| {
        let mut mutator = VisualAst::new(ui);
        match statement {
          Statement::Assignment(assignment) => assignment.accept_mut(&mut mutator),
          Statement::Pipeline(pipeline) => pipeline.accept_mut(&mut mutator),
        };
        Some(ui.button("➕").on_hover_text("Add new statement."))
      })
      .inner
  }

  fn visit_assignment(&mut self, assignment: &mut Assignment) -> Option<Response> {
    let mut combined_response = None;
    let (r_a, r_b) = match assignment {
      Assignment::AssignRef(pipeline, identifier) => {
        (pipeline.accept_mut(self), identifier.accept_mut(self))
      }
      Assignment::AssignSet(pipeline, identifier) => {
        (pipeline.accept_mut(self), identifier.accept_mut(self))
      }
      Assignment::AssignUpd(pipeline, identifier) => {
        (pipeline.accept_mut(self), identifier.accept_mut(self))
      }
      Assignment::AssignPush(pipeline, identifier) => {
        (pipeline.accept_mut(self), identifier.accept_mut(self))
      }
    };
    if let Some(a) = r_a {
      if let Some(b) = r_b {
        combined_response = Some(a.union(b));
      } else {
        combined_response = Some(a);
      }
    }
    combined_response
  }

  fn visit_pipeline(&mut self, pipeline: &mut Pipeline) -> Option<Response> {
    let mut final_response: Option<Response> = None;
    let mut i = 0;
    while i < pipeline.blocks.len() {
      let response = match self.visit_block(&mut pipeline.blocks[i]) {
        (BlockAction::Remove, r) => {
          pipeline.blocks.remove(i);
          r
        }
        (BlockAction::Keep, r) => {
          i += 1;
          r
        }
        (BlockAction::Duplicate, r) => {
          let mut block = pipeline.blocks[i].clone();
          block.get_custom_state::<BlockState>().map(|x| {
            x.id = Id::new(nanoid!(16));
          });
          pipeline.blocks.insert(i, block);
          i += 2;
          r
        }
        (BlockAction::Swap(block), r) => {
          pipeline.blocks[i] = block;
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

      if let Some(previous_response) = final_response.take() {
        if let Some(response) = response {
          final_response = Some(previous_response.union(response))
        } else {
          final_response = Some(previous_response)
        }
      }
    }

    final_response
  }

  fn visit_block(&mut self, block: &mut Block) -> (BlockAction, Option<Response>) {
    let (selected, id) = {
      let state = block.get_or_insert_custom_state(|| BlockState {
        selected: false,
        id: Id::new(nanoid!(16)),
      });
      (state.selected, state.id)
    };

    let (selected, action, response) =
      self.render_sub_window(id, selected, |ui| match &mut block.content {
        BlockContent::Empty => Some(ui.separator()),
        BlockContent::Shard(x) => {
          let mut mutator = VisualAst::with_parent_selected(ui, selected);
          mutator.mutate_shard(x)
        }
        BlockContent::Expr(x) | BlockContent::EvalExpr(x) | BlockContent::Shards(x) => {
          ui.group(|ui| {
            let mut mutator = VisualAst::new(ui);
            x.accept_mut(&mut mutator)
          })
          .inner
        }
        BlockContent::Const(x) => {
          let mut mutator = VisualAst::new(ui);
          x.accept_mut(&mut mutator)
        }
        BlockContent::TakeTable(_, _) => todo!(),
        BlockContent::TakeSeq(_, _) => todo!(),
        BlockContent::Func(x) => {
          let mut mutator = VisualAst::with_parent_selected(ui, selected);
          mutator.mutate_shard(x)
        }
        BlockContent::Program(x) => {
          ui.group(|ui| {
            let mut mutator = VisualAst::new(ui);
            x.accept_mut(&mut mutator)
          })
          .inner
        }
      });

    let state = block.get_custom_state::<BlockState>().unwrap();
    state.selected = selected;

    (action, response)
  }

  fn visit_function(&mut self, function: &mut Function) -> Option<Response> {
    self.mutate_shard(function)
  }

  fn visit_param(&mut self, param: &mut Param) -> Option<Response> {
    if let Some(name) = &param.name {
      self.ui.label(name.as_str());
    }
    param.value.accept_mut(self)
  }

  fn visit_identifier(&mut self, identifier: &mut Identifier) -> Option<Response> {
    Some(self.ui.label(identifier.name.as_str()))
  }

  fn visit_value(&mut self, value: &mut Value) -> Option<Response> {
    match value {
      Value::None => Some(self.ui.label("None")),
      Value::Identifier(x) => Some(self.ui.label(x.name.as_str())),
      Value::Boolean(x) => Some(self.ui.checkbox(x, if *x { "true" } else { "false" })),
      Value::Enum(_, _) => todo!(),
      Value::Number(x) => match x {
        Number::Integer(x) => Some(self.ui.add(CustomDragValue::new(x))),
        Number::Float(x) => Some(self.ui.add(CustomDragValue::new(x))),
        Number::Hexadecimal(_x) => todo!(),
      },
      Value::String(x) => {
        // if long we should use a multiline text editor
        // if short we should use a single line text editor
        let x = x.to_mut();
        Some(if x.len() > 16 {
          self.ui.text_edit_multiline(x)
        } else {
          let text = x.as_str();
          let text_width = 10.0 * text.chars().count() as f32;
          let width = text_width + 20.0; // Add some padding
          TextEdit::singleline(x).desired_width(width).ui(self.ui)
        })
      }
      Value::Bytes(_) => todo!(),
      Value::Int2(_) => todo!(),
      Value::Int3(_) => todo!(),
      Value::Int4(_) => todo!(),
      Value::Int8(_) => todo!(),
      Value::Int16(_) => todo!(),
      Value::Float2(_) => todo!(),
      Value::Float3(_) => todo!(),
      Value::Float4(_) => todo!(),
      Value::Seq(x) => {
        if x.len() > 4 {
          egui::ScrollArea::new([true, true])
            .show(self.ui, |ui| {
              let mut mutator = VisualAst::new(ui);
              for value in x.iter_mut() {
                value.accept_mut(&mut mutator);
              }
              let response = ui.button("➕").on_hover_text("Add new value.");
              if response.clicked() {
                x.push(Value::None);
              }
              Some(response)
            })
            .inner
        } else {
          for value in x.iter_mut() {
            value.accept_mut(self);
          }
          let response = self.ui.button("➕").on_hover_text("Add new value.");
          if response.clicked() {
            x.push(Value::None);
          }
          Some(response)
        }
      }
      Value::Table(x) => {
        let table = TableBuilder::new(self.ui)
          .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
          .striped(true)
          .column(Column::auto())
          .column(Column::auto())
          .header(16.0, |mut row| {
            row.col(|ui| {
              ui.label("Key");
            });
            row.col(|ui| {
              ui.label("Value");
            });
          });
        table.body(|body| {
          let len = x.len();
          body.rows(16.0, len, |mut row| {
            let index = row.index();
            row.col(|ui| {
              let mut mutator = VisualAst::new(ui);
              x[index].0.accept_mut(&mut mutator);
            });
            row.col(|ui| {
              let mut mutator = VisualAst::new(ui);
              x[index].1.accept_mut(&mut mutator);
            });
          });
        });
        None
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
              custom_state: None,
            }],
          })],
        });
        self.visit_value(value)
      }
      Value::Shards(x) => {
        let mut mutator = VisualAst::new(self.ui);
        x.accept_mut(&mut mutator)
      }
      Value::EvalExpr(_) => todo!(),
      Value::Expr(_) => todo!(),
      Value::TakeTable(_, _) => todo!(),
      Value::TakeSeq(_, _) => todo!(),
      Value::Func(x) => match x.name.name.as_str() {
        "color" => {
          let mut mutator = VisualAst::new(self.ui);
          x.accept_mut(&mut mutator)
        }
        _ => {
          let mut mutator = VisualAst::new(self.ui);
          x.accept_mut(&mut mutator)
        }
      },
    }
  }

  fn visit_metadata(&mut self, _metadata: &mut Metadata) -> Option<Response> {
    None
  }
}

#[derive(shards::shard)]
#[shard_info("UI.Shards", "A Shards program editor")]
pub struct UIShardsShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_warmup]
  parents: ParamVar,

  ast: Sequence,
}

impl Default for UIShardsShard {
  fn default() -> Self {
    let code = include_str!("simple.shs");
    let successful_parse = ShardsParser::parse(Rule::Program, code).unwrap();
    let mut env = shards_lang::read::ReadEnv::new("", ".", ".");
    let seq =
      shards_lang::read::process_program(successful_parse.into_iter().next().unwrap(), &mut env)
        .unwrap();
    let seq = seq.sequence;
    Self {
      required: ExposedTypes::new(),
      parents: ParamVar::new_named(PARENTS_UI_NAME),
      ast: seq,
    }
  }
}

#[shards::shard_impl]
impl Shard for UIShardsShard {
  fn input_types(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn warmup(&mut self, ctx: &ShardsContext) -> Result<(), &str> {
    self.warmup_helper(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&ShardsContext>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;

    require_parents(&mut self.required);

    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &ShardsContext, _input: &Var) -> Result<Var, &str> {
    let ui = get_current_parent_opt(self.parents.get())?.ok_or("No parent UI")?;
    egui::ScrollArea::new([true, true]).show(ui, |ui| {
      let mut mutator = VisualAst::new(ui);
      self.ast.accept_mut(&mut mutator);
    });
    Ok(Var::default())
  }
}

pub fn register_shards() {
  register_shard::<UIShardsShard>();
}
