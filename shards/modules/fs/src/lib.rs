/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright © 2023 Fragcolor Pte. Ltd. */

#[macro_use]
extern crate shards;
#[macro_use]
extern crate lazy_static;

extern crate compile_time_crc32;

use shards::core::{register_enum, register_shard};
use shards::shard::Shard;
use shards::types::{common_type, NONE_TYPES, SEQ_OF_ANY_TABLE_TYPES};
use shards::types::{Context, ExposedTypes, InstanceData, ParamVar, Type, Types, Var};

use std::fs;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use notify::Config;
use shards::core::register_legacy_shard;
use shards::core::run_blocking;
use shards::core::BlockingShard;
use shards::core::Core;
use shards::shard::LegacyShard;
use shards::shardsc::SHCore;
use shards::types::AutoSeqVar;
use shards::types::ClonedVar;
use shards::types::ANYS_TYPES;
use shards::types::INT_TYPES;
use shards::SHInt;

use shards::types::Parameters;
use shards::types::Seq;
use shards::types::BOOL_TYPES_SLICE;
use shards::types::BOOL_VAR_OR_NONE_SLICE;
use shards::types::STRING_TYPES;
use shards::types::STRING_VAR_OR_NONE_SLICE;

lazy_static! {
  pub static ref STRINGS_VAR_OR_NONE_TYPES: Types = vec![
    common_type::strings,
    common_type::strings_var,
    common_type::none
  ];
  pub static ref STRING_OR_STRINGS_TYPES: Types = vec![common_type::string, common_type::strings];
  pub static ref SEQ_OF_STRINGS_TYPE: Type = Type::seq(&*STRING_TYPES);
  pub static ref SEQ_OF_STRINGS_TYPES: Types = vec![*SEQ_OF_STRINGS_TYPE];
  pub static ref FILTER_SEQ_TYPE: Type = Type::seq(&*SEQ_OF_STRINGS_TYPES);
  pub static ref FILTER_SEQ_TYPES: Types = vec![*FILTER_SEQ_TYPE];
  static ref SAVEFILEDIALOG_PARAMETERS: Parameters = vec![
    (
      cstr!("Filters"),
      shccstr!("To filter files based on extensions."),
      &STRINGS_VAR_OR_NONE_TYPES[..],
    )
      .into(),
    (
      cstr!("CurrentDir"),
      shccstr!("Set the current directory"),
      STRING_VAR_OR_NONE_SLICE,
    )
      .into(),
  ];
}

#[derive(shards::shard)]
#[shard_info("FS.FileDialog", "Creates a file dialog to open files")]
struct FileDialog {
  #[shard_param("Filters", "To filter files based on extensions.", [common_type::none, *FILTER_SEQ_TYPE, Type::context_variable(&*FILTER_SEQ_TYPES)])]
  filters: ParamVar,
  #[shard_param("FilterNames", "For each filter, gives the name to show of the filter.", [common_type::none, common_type::strings, common_type::strings_var])]
  filter_names: ParamVar,
  #[shard_param("CurrentDir", "Set the current directory", STRING_VAR_OR_NONE_SLICE)]
  current_dir: ParamVar,
  #[shard_param(
    "Multiple",
    "To select multiple files instead of just one.",
    BOOL_TYPES_SLICE
  )]
  multiple: ClonedVar,
  #[shard_param(
    "Folder",
    "To select a folder instead of a file.",
    BOOL_VAR_OR_NONE_SLICE
  )]
  folder: ParamVar,
  output: ClonedVar,
  #[shard_required]
  required: ExposedTypes,
}

impl Default for FileDialog {
  fn default() -> Self {
    Self {
      filters: ParamVar::default(),
      filter_names: ParamVar::default(),
      current_dir: ParamVar::default(),
      multiple: ClonedVar(Var::from(false)),
      folder: ParamVar::new(false.into()),
      output: ClonedVar::default(),
      required: ExposedTypes::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for FileDialog {
  fn input_types(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &STRING_OR_STRINGS_TYPES
  }

  fn compose(&mut self, _data: &InstanceData) -> Result<Type, &'static str> {
    self.compose_helper(_data)?;

    let multiple: bool = (&self.multiple.0).try_into().unwrap_or(false);
    if multiple {
      Ok(common_type::strings)
    } else {
      Ok(common_type::string)
    }
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &'static str> {
    self.warmup_helper(ctx)?;

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &'static str> {
    self.cleanup_helper(ctx)?;
    Ok(())
  }

  #[cfg(feature = "rfd-enabled")]
  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
    Ok(Some(run_blocking(self, context, input)))
  }

  #[cfg(not(feature = "rfd-enabled"))]
  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
    Err("FileDialog is not supported on this platform")
  }
}

#[cfg(feature = "rfd-enabled")]
impl BlockingShard for FileDialog {
  #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
  fn activate_blocking(&mut self, _context: &Context, _input: &Var) -> Result<Var, &'static str> {
    let mut dialog = rfd::FileDialog::new();
    let folder: bool = self.folder.get().try_into().unwrap_or_default();
    let filters = self.filters.get();
    let filter_names = self.filter_names.get();
    if !folder && !filters.is_none() {
      let filters: Seq = filters.try_into()?;
      let filter_names: Result<Seq, _> = filter_names.try_into();
      if let Ok(filter_names) = &filter_names {
        if filters.len() != filter_names.len() {
          return Err("FilterNames must be the same length as Filters");
        }
        for (filter, filter_name) in filters.iter().zip(filter_names.iter()) {
          let filter: Seq = filter.try_into()?;
          let filter_name: &str = filter_name.try_into()?;
          let filter = filter
            .iter()
            .map(|f| f.try_into())
            .collect::<Result<Vec<&'static str>, _>>()?;
          dialog = dialog.add_filter(filter_name, &filter);
        }
      } else {
        for filter in filters.iter() {
          let filter: Seq = filter.try_into()?;
          let filter = filter
            .iter()
            .map(|f| f.try_into())
            .collect::<Result<Vec<&'static str>, _>>()?;
          let filter_name = filter.join(", ");
          dialog = dialog.add_filter(&filter_name, &filter);
        }
      }
    }
    let current_dir = self.current_dir.get();
    if !current_dir.is_none() {
      let cd: &str = current_dir.try_into()?;
      if let Ok(start_path) = std::path::PathBuf::from(cd).canonicalize() {
        dialog = dialog.set_directory(start_path);
      }
    }

    let multiple: bool = (&self.multiple.0).try_into().unwrap_or(false);
    if multiple {
      self.pick_multiple(dialog, folder)
    } else {
      self.pick_single(dialog, folder)
    }
  }
}

#[cfg(feature = "rfd-enabled")]
impl FileDialog {
  fn pick_single(&mut self, dialog: rfd::FileDialog, folder: bool) -> Result<Var, &'static str> {
    let path = if folder {
      dialog.pick_folder()
    } else {
      dialog.pick_file()
    };

    if let Some(path) = path {
      let path = path.display().to_string();
      self.output.assign(&Var::ephemeral_string(path.as_str()));
      Ok(self.output.0)
    } else {
      Err("Operation was cancelled")
    }
  }

  fn pick_multiple(&mut self, dialog: rfd::FileDialog, folder: bool) -> Result<Var, &'static str> {
    let paths = if folder {
      dialog.pick_folders()
    } else {
      dialog.pick_files()
    };

    if let Some(paths) = paths {
      let paths: Vec<Var> = paths
        .iter()
        .map(|path| {
          let path = path.display().to_string();
          Var::ephemeral_string(path.as_str())
        })
        .collect();
      let output: Var = paths.as_slice().into();
      self.output.assign(&output);
      Ok(self.output.0)
    } else {
      Err("Operation was cancelled")
    }
  }
}

struct SaveFileDialog {
  filters: ParamVar,
  current_dir: ParamVar,
  output: ClonedVar,
}

impl Default for SaveFileDialog {
  fn default() -> Self {
    Self {
      filters: ParamVar::default(),
      current_dir: ParamVar::default(),
      output: ClonedVar::default(),
    }
  }
}

impl LegacyShard for SaveFileDialog {
  fn registerName() -> &'static str
  where
    Self: Sized,
  {
    cstr!("FS.SaveFileDialog")
  }

  fn hash() -> u32
  where
    Self: Sized,
  {
    compile_time_crc32::crc32!("FS.SaveFileDialog-rust-0x20200101")
  }

  fn name(&mut self) -> &str {
    "FS.SaveFileDialog"
  }

  fn inputTypes(&mut self) -> &Types {
    &NONE_TYPES
  }

  fn outputTypes(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn parameters(&mut self) -> Option<&Parameters> {
    Some(&SAVEFILEDIALOG_PARAMETERS)
  }

  fn setParam(&mut self, index: i32, value: &Var) -> Result<(), &'static str> {
    match index {
      0 => self.filters.set_param(value),
      1 => self.current_dir.set_param(value),
      _ => Err("Invalid parameter index"),
    }
  }

  fn getParam(&mut self, index: i32) -> Var {
    match index {
      0 => self.filters.get_param(),
      1 => self.current_dir.get_param(),
      _ => Var::default(),
    }
  }

  fn warmup(&mut self, ctx: &Context) -> Result<(), &'static str> {
    self.filters.warmup(ctx);
    self.current_dir.warmup(ctx);

    Ok(())
  }

  fn cleanup(&mut self, ctx: Option<&Context>) -> Result<(), &'static str> {
    self.current_dir.cleanup(ctx);
    self.filters.cleanup(ctx);

    Ok(())
  }

  #[cfg(feature = "rfd-enabled")]
  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
    Ok(Some(run_blocking(self, context, input)))
  }

  #[cfg(not(feature = "rfd-enabled"))]
  fn activate(&mut self, context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
    Err("SaveFileDialog is not supported on this platform")
  }
}

#[cfg(feature = "rfd-enabled")]
impl BlockingShard for SaveFileDialog {
  fn activate_blocking(&mut self, _context: &Context, _input: &Var) -> Result<Var, &'static str> {
    let mut dialog = rfd::FileDialog::new();
    let filters = self.filters.get();
    if !filters.is_none() {
      let filters: Seq = filters.try_into()?;
      for filter in filters.iter() {
        let filter: &str = filter.try_into()?;
        dialog = dialog.add_filter(filter, &[filter]);
      }
    }
    let current_dir = self.current_dir.get();
    if !current_dir.is_none() {
      let cd: &str = current_dir.try_into()?;
      if let Ok(start_path) = std::path::PathBuf::from(cd).canonicalize() {
        dialog = dialog.set_directory(start_path);
      }
    }

    if let Some(path) = dialog.save_file() {
      let path = path.display().to_string();
      self.output.assign(&Var::ephemeral_string(path.as_str()));
      Ok(self.output.0)
    } else {
      Err("Operation was cancelled")
    }
  }
}

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

#[derive(shards::shard)]
#[shard_info("FS.Watch", "A directory watcher")]
struct NotifyShard {
  #[shard_required]
  required: ExposedTypes,
  current_path: ClonedVar,
  rx: Option<Receiver<Result<notify::Event, notify::Error>>>,
  watcher: Option<RecommendedWatcher>,
  output: AutoSeqVar,
}

impl Default for NotifyShard {
  fn default() -> Self {
    Self {
      required: ExposedTypes::new(),
      current_path: ClonedVar::default(),
      rx: None,
      watcher: None,
      output: AutoSeqVar::new(),
    }
  }
}

#[shards::shard_impl]
impl Shard for NotifyShard {
  fn input_types(&mut self) -> &Types {
    &STRING_TYPES
  }

  fn output_types(&mut self) -> &Types {
    &SEQ_OF_ANY_TABLE_TYPES
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

  fn activate(&mut self, _context: &Context, input: &Var) -> Result<Option<Var>, &'static str> {
    if self.current_path.0.as_ref() != input {
      self.current_path.assign(input);
      self.watcher = Some(self.reset()?);
    }

    self.output.0.clear();

    if let Some(rx) = &self.rx {
      loop {
        let something = rx.try_recv();
        match something {
          Ok(event) => {
            if let Ok(event) = event {
              // for now we only send events if content has changed or all/any
              let k = match event.kind {
                notify::EventKind::Access(_) => Var::ephemeral_string("Access"),
                notify::EventKind::Create(_) => Var::ephemeral_string("Create"),
                notify::EventKind::Modify(_) => Var::ephemeral_string("Modify"),
                notify::EventKind::Remove(_) => Var::ephemeral_string("Remove"),
                notify::EventKind::Any | notify::EventKind::Other => {
                  Var::ephemeral_string("Unknown")
                }
              };
              for path in event.paths {
                if let Some(path) = path.to_str() {
                  let p = Var::ephemeral_string(path);
                  let t = self.output.0.next_mut();
                  let t = t.as_mut_table_creating().unwrap();
                  // TODO, this originally used path as the key to deduplicate.
                  // We "fixed" it after, but now we got dupes :)
                  t.insert_fast_static("path", &p);
                  t.insert_fast_static("type", &k);
                }
              }
            } else if let Err(e) = event {
              shlog_error!("Error: {:?}", e);
              return Err("Error");
            }
          }
          Err(e) => match e {
            std::sync::mpsc::TryRecvError::Empty => break,
            std::sync::mpsc::TryRecvError::Disconnected => return Err("Disconnected"),
          },
        }
      }
    }

    Ok(Some(self.output.0 .0))
  }
}

impl NotifyShard {
  fn reset(&mut self) -> Result<RecommendedWatcher, &'static str> {
    let (tx, rx) = std::sync::mpsc::channel();
    self.rx = Some(rx);

    let mut watcher = RecommendedWatcher::new(tx, Config::default()).map_err(|e| {
      shlog_error!("Failed to create watcher: {}", e);
      "Failed to create watcher"
    })?;

    let path: &str = self.current_path.0.as_ref().try_into()?;
    let path = std::path::PathBuf::from(path);
    let path = fs::canonicalize(&path).map_err(|e| {
      shlog_error!("Failed to canonicalize path: {}", e);
      "Failed to canonicalize path"
    })?;

    shlog_debug!("Watching path: {:?}", path);

    watcher
      .watch(&path, RecursiveMode::Recursive)
      .map_err(|e| {
        shlog_error!("Failed to watch path: {}", e);
        "Failed to watch path"
      })?;

    Ok(watcher)
  }
}

#[no_mangle]
pub extern "C" fn shardsRegister_fs_rust(core: *mut SHCore) {
  unsafe {
    Core = core;
  }

  register_shard::<FileDialog>();
  register_legacy_shard::<SaveFileDialog>();
  register_shard::<NotifyShard>();
}
