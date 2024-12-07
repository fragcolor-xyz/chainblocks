/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2021 Fragcolor Pte. Ltd. */

use shards::core::register_shard;
use shards::shard::Shard;
use shards::types::{common_type, AutoSeqVar, ClonedVar, ParamVar, STRINGS_TYPES, STRING_TYPES};
use shards::types::{Context, ExposedTypes, InstanceData, Type, Types, Var};
use htmd::HtmlToMarkdown;

use pulldown_cmark::{
  BlockQuoteKind, CodeBlockKind, Event, LinkType, MetadataBlockKind, Options, Parser, Tag, TagEnd,
};

#[derive(shards::shard)]
#[shard_info("Markdown.Parse", "A markdown commonmark pull parser.")]
struct MarkdownParseShard {
  #[shard_required]
  required: ExposedTypes,

  #[shard_param("Reset", "Reset the parser", [common_type::bool, common_type::bool_var])]
  reset: ParamVar,

  input: ClonedVar,
  parser: Option<Parser<'static>>,
  output: AutoSeqVar,
}

impl Default for MarkdownParseShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      input: ClonedVar::default(),
      parser: None,
      output: AutoSeqVar::new(),
      reset: ParamVar::new(false.into()),
    }
  }
}

#[shards::shard_impl]
impl Shard for MarkdownParseShard {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &STRINGS_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;

    self.parser = None; // this is important to reset parser's state

    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let reset: bool = self.reset.get().as_ref().try_into()?;
    if self.parser.is_none() || reset {
      self.input = input.into();
      let input: &str = self.input.as_ref().try_into()?;
      self.parser = Some(Parser::new_ext(input, Options::all()));
    }

    self.output.0.clear();
    let next = self.parser.as_mut().unwrap().next();
    if let Some(event) = next {
      match event {
        Event::Start(tag) => match tag {
          Tag::Paragraph => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Paragraph");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::Heading {
            level,
            id,
            classes,
            attrs,
          } => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Heading");
            let level = level.to_string();
            let c = Var::ephemeral_string(&level);
            self.output.0.push(&a);
            self.output.0.push(&b);
            self.output.0.push(&c);
            if let Some(id) = id {
              let id = format!("#{}", id);
              let d = Var::ephemeral_string(&id);
              self.output.0.push(&d);
            }
            for class in classes {
              let class = format!(".{}", class);
              let d = Var::ephemeral_string(&class);
              self.output.0.push(&d);
            }
            for (name, value) in attrs {
              if let Some(value) = value {
                let attr = format!("{}={}", name, value);
                let d = Var::ephemeral_string(&attr);
                self.output.0.push(&d);
              } else {
                let attr = format!("{}", name);
                let d = Var::ephemeral_string(&attr);
                self.output.0.push(&d);
              }
            }
          }
          Tag::BlockQuote(kind) => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("BlockQuote");
            self.output.0.push(&a);
            self.output.0.push(&b);
            if let Some(kind) = kind {
              match kind {
                BlockQuoteKind::Note => {
                  let c = Var::ephemeral_string("Note");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Tip => {
                  let c = Var::ephemeral_string("Tip");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Important => {
                  let c = Var::ephemeral_string("Important");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Warning => {
                  let c = Var::ephemeral_string("Warning");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Caution => {
                  let c = Var::ephemeral_string("Caution");
                  self.output.0.push(&c);
                }
              }
            }
          }
          Tag::CodeBlock(kind) => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("CodeBlock");
            self.output.0.push(&a);
            self.output.0.push(&b);
            match kind {
              CodeBlockKind::Indented => {
                let c = Var::ephemeral_string("Indented");
                self.output.0.push(&c);
              }
              CodeBlockKind::Fenced(language) => {
                let c = Var::ephemeral_string("Fenced");
                self.output.0.push(&c);
                let d = Var::ephemeral_string(&language);
                self.output.0.push(&d);
              }
            }
          }
          Tag::HtmlBlock => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("HtmlBlock");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::List(first) => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("List");
            let c = Var::ephemeral_string(if first.is_some() {
              "Ordered"
            } else {
              "Unordered"
            });
            self.output.0.push(&a);
            self.output.0.push(&b);
            self.output.0.push(&c);
          }
          Tag::Item => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Item");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::FootnoteDefinition(text) => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("FootnoteDefinition");
            let c = Var::ephemeral_string(&text);
            self.output.0.push(&a);
            self.output.0.push(&b);
            self.output.0.push(&c);
          }
          Tag::DefinitionList => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("DefinitionList");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::DefinitionListTitle => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("DefinitionListTitle");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::DefinitionListDefinition => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("DefinitionListDefinition");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::Table(_) => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Table");
            self.output.0.push(&a);
            self.output.0.push(&b);
            // alignment of columns ignored for simplicity
          }
          Tag::TableHead => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("TableHead");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::TableRow => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("TableRow");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::TableCell => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("TableCell");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::Emphasis => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Emphasis");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::Strong => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Strong");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::Strikethrough => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Strikethrough");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          Tag::Link {
            link_type,
            dest_url,
            title,
            id,
          } => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Link");
            self.output.0.push(&a);
            self.output.0.push(&b);
            match link_type {
              LinkType::Inline => {
                let c = Var::ephemeral_string("Inline");
                self.output.0.push(&c);
              }
              LinkType::Reference => {
                let c = Var::ephemeral_string("Reference");
                self.output.0.push(&c);
              }
              LinkType::ReferenceUnknown => {
                let c = Var::ephemeral_string("ReferenceUnknown");
                self.output.0.push(&c);
              }
              LinkType::Collapsed => {
                let c = Var::ephemeral_string("Collapsed");
                self.output.0.push(&c);
              }
              LinkType::CollapsedUnknown => {
                let c = Var::ephemeral_string("CollapsedUnknown");
                self.output.0.push(&c);
              }
              LinkType::Shortcut => {
                let c = Var::ephemeral_string("Shortcut");
                self.output.0.push(&c);
              }
              LinkType::ShortcutUnknown => {
                let c = Var::ephemeral_string("ShortcutUnknown");
                self.output.0.push(&c);
              }
              LinkType::Autolink => {
                let c = Var::ephemeral_string("Autolink");
                self.output.0.push(&c);
              }
              LinkType::Email => {
                let c = Var::ephemeral_string("Email");
                self.output.0.push(&c);
              }
            }
            let d = Var::ephemeral_string(&dest_url);
            self.output.0.push(&d);
            let e = Var::ephemeral_string(&title);
            self.output.0.push(&e);
            let f = Var::ephemeral_string(&id);
            self.output.0.push(&f);
          }
          Tag::Image {
            link_type,
            dest_url,
            title,
            id,
          } => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("Image");
            self.output.0.push(&a);
            self.output.0.push(&b);
            match link_type {
              LinkType::Inline => {
                let c = Var::ephemeral_string("Inline");
                self.output.0.push(&c);
              }
              LinkType::Reference => {
                let c = Var::ephemeral_string("Reference");
                self.output.0.push(&c);
              }
              LinkType::ReferenceUnknown => {
                let c = Var::ephemeral_string("ReferenceUnknown");
                self.output.0.push(&c);
              }
              LinkType::Collapsed => {
                let c = Var::ephemeral_string("Collapsed");
                self.output.0.push(&c);
              }
              LinkType::CollapsedUnknown => {
                let c = Var::ephemeral_string("CollapsedUnknown");
                self.output.0.push(&c);
              }
              LinkType::Shortcut => {
                let c = Var::ephemeral_string("Shortcut");
                self.output.0.push(&c);
              }
              LinkType::ShortcutUnknown => {
                let c = Var::ephemeral_string("ShortcutUnknown");
                self.output.0.push(&c);
              }
              LinkType::Autolink => {
                let c = Var::ephemeral_string("Autolink");
                self.output.0.push(&c);
              }
              LinkType::Email => {
                let c = Var::ephemeral_string("Email");
                self.output.0.push(&c);
              }
            }
            let d = Var::ephemeral_string(&dest_url);
            self.output.0.push(&d);
            let e = Var::ephemeral_string(&title);
            self.output.0.push(&e);
            let f = Var::ephemeral_string(&id);
            self.output.0.push(&f);
          }
          Tag::MetadataBlock(kind) => {
            let a = Var::ephemeral_string("Start");
            let b = Var::ephemeral_string("MetadataBlock");
            self.output.0.push(&a);
            self.output.0.push(&b);
            match kind {
              MetadataBlockKind::YamlStyle => {
                let c = Var::ephemeral_string("YamlStyle");
                self.output.0.push(&c);
              }
              MetadataBlockKind::PlusesStyle => {
                let c = Var::ephemeral_string("PlusesStyle");
                self.output.0.push(&c);
              }
            }
          }
        },
        Event::End(tag) => match tag {
          TagEnd::Paragraph => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Paragraph");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Heading(level) => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Heading");
            let level = level.to_string();
            let c = Var::ephemeral_string(&level);
            self.output.0.push(&a);
            self.output.0.push(&b);
            self.output.0.push(&c);
          }
          TagEnd::BlockQuote(kind) => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("BlockQuote");
            self.output.0.push(&a);
            self.output.0.push(&b);
            if let Some(kind) = kind {
              match kind {
                BlockQuoteKind::Note => {
                  let c = Var::ephemeral_string("Note");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Tip => {
                  let c = Var::ephemeral_string("Tip");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Important => {
                  let c = Var::ephemeral_string("Important");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Warning => {
                  let c = Var::ephemeral_string("Warning");
                  self.output.0.push(&c);
                }
                BlockQuoteKind::Caution => {
                  let c = Var::ephemeral_string("Caution");
                  self.output.0.push(&c);
                }
              }
            }
          }
          TagEnd::CodeBlock => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("CodeBlock");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::HtmlBlock => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("HtmlBlock");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::List(ordered) => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("List");
            let c = Var::ephemeral_string(if ordered { "Ordered" } else { "Unordered" });
            self.output.0.push(&a);
            self.output.0.push(&b);
            self.output.0.push(&c);
          }
          TagEnd::Item => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Item");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::FootnoteDefinition => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("FootnoteDefinition");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::DefinitionList => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("DefinitionList");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::DefinitionListTitle => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("DefinitionListTitle");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::DefinitionListDefinition => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("DefinitionListDefinition");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Table => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Table");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::TableHead => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("TableHead");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::TableRow => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("TableRow");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::TableCell => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("TableCell");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Emphasis => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Emphasis");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Strong => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Strong");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Strikethrough => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Strikethrough");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Link => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Link");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::Image => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("Image");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
          TagEnd::MetadataBlock(_) => {
            let a = Var::ephemeral_string("End");
            let b = Var::ephemeral_string("MetadataBlock");
            self.output.0.push(&a);
            self.output.0.push(&b);
          }
        },
        Event::Text(text) => {
          let a = Var::ephemeral_string("Text");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::Code(text) => {
          let a = Var::ephemeral_string("Code");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::InlineMath(text) => {
          let a = Var::ephemeral_string("InlineMath");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::DisplayMath(text) => {
          let a = Var::ephemeral_string("DisplayMath");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::Html(text) => {
          let a = Var::ephemeral_string("Html");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::InlineHtml(text) => {
          let a = Var::ephemeral_string("InlineHtml");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::FootnoteReference(text) => {
          let a = Var::ephemeral_string("FootnoteReference");
          let b = Var::ephemeral_string(&text);
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
        Event::SoftBreak => {
          let a = Var::ephemeral_string("SoftBreak");
          self.output.0.push(&a);
        }
        Event::HardBreak => {
          let a = Var::ephemeral_string("HardBreak");
          self.output.0.push(&a);
        }
        Event::Rule => {
          let a = Var::ephemeral_string("Rule");
          self.output.0.push(&a);
        }
        Event::TaskListMarker(b) => {
          let a = Var::ephemeral_string("TaskListMarker");
          let b = Var::ephemeral_string(if b { "Checked" } else { "Unchecked" });
          self.output.0.push(&a);
          self.output.0.push(&b);
        }
      }
    }

    Ok(Some(self.output.0 .0))
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_markdown_rust(core: *mut shards::shardsc::SHCore) {
  unsafe {
    shards::core::Core = core;
  }

  register_shard::<MarkdownParseShard>();
  register_shard::<MarkdownFromHTMLShard>();
}

#[derive(shards::shard)]
#[shard_info("Markdown.FromHTML", "Converts HTML to Markdown")]
struct MarkdownFromHTMLShard {
  #[shard_required]
  required: ExposedTypes,
}

impl Default for MarkdownFromHTMLShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for MarkdownFromHTMLShard {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &str> {
    self.warmup_helper(ctx)?;
    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  fn compose(&mut self, data: &InstanceData) -> Result<Type, &str> {
    self.compose_helper(data)?;
    Ok(self.output_types()[0])
  }

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &str> {
    let html: &str = input.try_into()?;
    
    match HtmlToMarkdown::new().convert(html) {
      Ok(markdown) => Ok(Some(Var::ephemeral_string(&markdown))),
      Err(_) => Err("Failed to convert HTML to Markdown")
    }
  }
}
