use crate::error::Error;
use crate::read::{get_dependencies, read_with_env, ReadEnv};
use crate::{eval, formatter, Program};
use crate::{eval::eval, eval::new_cancellation_token, read::read};
use clap::{arg, Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use shards::core::{sleep, Core};
use shards::types::Mesh;
use shards::util::from_raw_parts_allow_null;
use shards::{
  fourCharacterCode, shlog, shlog_debug, shlog_error, SHCore, GIT_VERSION, SHARDS_CURRENT_ABI,
};
use std::collections::HashMap;
use std::ffi::CStr;
use std::fs;
use std::io::Write;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc};

extern "C" {
  fn shardsInterface(version: u32) -> *mut SHCore;
  fn shards_install_signal_handlers();
  fn shards_decompress_strings();
}

#[derive(Debug, clap::Args)]
struct RunArgs {
  /// The script to execute
  #[arg(value_hint = clap::ValueHint::FilePath)]
  file: String,
  /// Decompress help strings before running the script
  #[arg(long, short = 'd', default_value = "false", action)]
  decompress_strings: bool,
  /// Change the current path to the scripts's path
  #[arg(long, short = 'c', action)]
  skip_cwd: bool,
  /// List of include directories
  #[arg(long, short = 'I')]
  include: Vec<String>,
  #[arg(num_args = 0..)]
  args: Vec<String>,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
  /// Formats a shards file
  Format {
    /// The file to format
    #[arg(value_hint = clap::ValueHint::FilePath)]
    file: String,
    /// Run the formatter on the file directly
    /// by default the output will go to stdout
    #[arg(long, short = 'i', action)]
    inline: bool,
    /// Optionally an output file name
    #[arg(long, short = 'o')]
    output: Option<String>,
  },
  /// Run formatter tests
  Test {},
  /// Reads and executes a Shards file
  New(RunArgs),
  Run(RunArgs),
  /// Reads and builds a binary AST Shards file
  Build {
    /// The script to evaluate
    #[arg(value_hint = clap::ValueHint::FilePath)]
    file: String,
    /// The output file to write to
    #[arg(long, short = 'o', default_value = "out.sho")]
    output: String,
    /// Output as JSON ast
    #[arg(long, short = 'j', action)]
    json: bool,
    /// List of include directories
    #[arg(long, short = 'I')]
    include: Vec<String>,
    /// The depfile to write, in makefile readable format
    #[arg(long, short = 'd')]
    depfile: Option<String>,
  },
  AST {
    /// The script to evaluate
    #[arg(value_hint = clap::ValueHint::FilePath)]
    file: String,
    /// The output file to write to
    #[arg(long, short = 'o', default_value = "out.sho")]
    output: String,
    /// List of include directories
    #[arg(long, short = 'I')]
    include: Vec<String>,
  },
  /// Loads and executes a binary Shards file
  Load {
    /// The binary Shards file to execute
    #[arg(value_hint = clap::ValueHint::FilePath)]
    file: String,
    /// Decompress help strings before running the script
    #[arg(long, short = 'd', default_value = "false", action)]
    decompress_strings: bool,
    #[arg(num_args = 0..)]
    args: Vec<String>,
  },
}

#[derive(Debug, clap::Parser)]
#[command(name = "Shards", version = "0.1")]
#[command(about = "Shards command line tools and executor.")]
#[command(author = "Fragcolor Team")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Debug, clap::Parser)]
#[command(name = "shards run")]
struct SimpleCLI {
  #[command(flatten)]
  run_args: RunArgs,
}

pub fn process_args(argc: i32, argv: *const *const c_char, no_cancellation: bool) -> i32 {
  let cancellation_token = new_cancellation_token();

  #[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "visionos")))]
  if !no_cancellation {
    let cancellation_token_1 = cancellation_token.clone();
    let r = ctrlc::set_handler(move || {
      cancellation_token_1.store(true, atomic::Ordering::Relaxed);
    });
    if r.is_err() {
      shlog!("Failed to set ctrl-c handler");
      return 1;
    }
  }

  let args: Vec<String> = unsafe {
    from_raw_parts_allow_null(argv, argc as usize)
      .iter()
      .map(|&arg| {
        let c_str = CStr::from_ptr(arg);
        c_str.to_str().unwrap().to_owned()
      })
      .collect()
  };

  unsafe {
    shards::core::Core = shardsInterface(SHARDS_CURRENT_ABI as u32);
    shards_install_signal_handlers();
    shlog_debug!("Shards git version: {}", GIT_VERSION);
  }

  let cli = Cli::try_parse_from(args.clone());
  let res = match cli {
    Ok(cli) => match &cli.command {
      Commands::Build {
        file,
        output,
        include,
        depfile,
        json,
      } => build(file, &output, include.to_vec(), depfile.as_deref(), *json),
      Commands::AST {
        file,
        output,
        include,
      } => build(file, &output, include.to_vec(), None, true),
      Commands::Load {
        file,
        decompress_strings,
        args,
      } => load(file, args, *decompress_strings, cancellation_token),
      Commands::New(args) => execute(args, cancellation_token),
      Commands::Run(args) => execute(args, cancellation_token),
      Commands::Format {
        file,
        output,
        inline,
      } => format(file, output, *inline),
      Commands::Test {} => formatter::run_tests(),
    },
    // Try to support a simple "shards script.shs" command line in case none of the above matched
    Err(orig_err) => match SimpleCLI::try_parse_from(args) {
      Ok(cli) => execute(&cli.run_args, cancellation_token),
      Err(e) => Err(Box::new(orig_err) as Box<dyn std::error::Error>),
    },
  };

  if let Err(e) = res {
    shlog_error!("Error: {}", e);
    1
  } else {
    0
  }
}

fn format(file: &str, output: &Option<String>, inline: bool) -> Result<(), Error> {
  if output.is_some() && inline {
    return Err("Cannot use both -i and -o".into());
  }

  let mut in_str = if file == "-" {
    std::io::read_to_string(std::io::stdin()).unwrap()
  } else {
    fs::read_to_string(file)?
  };

  let newline_style = formatter::detect_newline_style(&in_str);

  // add new line at the end of the file to be able to parse it correctly
  newline_style.push_to_str(&mut in_str);

  if inline {
    let mut buf = std::io::BufWriter::new(Vec::new());
    let mut v = formatter::FormatterVisitor::new(&mut buf, &in_str);
    v.newline_style = newline_style;

    crate::rule_visitor::process(&in_str, &mut v)?;

    fs::write(file, &buf.into_inner()?[..])?;
  } else {
    let mut out_stream: Box<dyn std::io::Write> = if let Some(out) = output {
      Box::new(fs::File::create(out)?)
    } else {
      Box::new(std::io::stdout())
    };

    let mut v = formatter::FormatterVisitor::new(out_stream.as_mut(), &in_str);
    v.newline_style = newline_style;
    crate::rule_visitor::process(&in_str, &mut v)?;
  }

  std::io::stdout().flush()?;

  Ok(())
}

fn load(
  file: &str,
  args: &Vec<String>,
  decompress_strings: bool,
  cancellation_token: Arc<AtomicBool>,
) -> Result<(), Error> {
  if decompress_strings {
    unsafe {
      shards_decompress_strings();
    }
  }

  shlog!("Loading file");
  shlog!("Parsing binary file: {}", file);

  let ast = {
    // deserialize from flexbuffers, skipping the first 8 bytes
    let mut file_content = std::fs::read(file).map_err(|_| "File not found")?;
    let magic: i32 = i32::from_be_bytes([
      file_content[0],
      file_content[1],
      file_content[2],
      file_content[3],
    ]);
    assert_eq!(magic, fourCharacterCode(*b"SHRD"));
    let version = u32::from_le_bytes([
      file_content[4],
      file_content[5],
      file_content[6],
      file_content[7],
    ]);
    assert_eq!(version, SHARDS_CURRENT_ABI); // todo backwards compatibility
    file_content.drain(0..8);
    flexbuffers::from_slice(file_content.as_slice()).unwrap()
  };

  Ok(execute_seq(&args, ast, cancellation_token)?)
}

fn execute_seq(
  args: &Vec<String>,
  ast: Program,
  cancellation_token: Arc<AtomicBool>,
) -> Result<(), &'static str> {
  let mut defines = HashMap::new();

  for arg in args {
    shlog_debug!("arg: {}", arg);
    // find the first column and split it, the rest is the value
    let mut split = arg.split(':');
    let key = split.next().unwrap();
    // value should be all the rest, could contain ':' even
    let value = split.collect::<Vec<&str>>().join(":");
    // finally unescape the value if needed
    let value = value.replace("\\:", ":");
    // and remove quotes if quoted
    let value = value.trim_matches('"');
    defines.insert(key.to_owned(), value.to_owned());
  }

  let wire = {
    eval(&ast, "root", defines, cancellation_token.clone()).map_err(|e| {
      shlog!("Error: {:?}", e);
      "Failed to evaluate file"
    })?
  };
  // enlarge stack
  wire.set_stack_size(eval::EVAL_STACK_SIZE);

  let mut mesh = Mesh::default();
  if !mesh.compose(wire.0) {
    return Err("Failed to compose mesh");
  }
  mesh.schedule(wire.0, false);

  loop {
    if cancellation_token.load(atomic::Ordering::Relaxed) {
      break;
    }

    if !mesh.tick() || mesh.is_empty() {
      break;
    }

    // still yield to other threads
    sleep(0.0);
  }

  let info = wire.get_info();
  if info.failed {
    let msg = std::str::from_utf8(unsafe {
      from_raw_parts_allow_null(
        info.failureMessage.string as *const u8,
        info.failureMessage.len as usize,
      )
    })
    .unwrap();
    shlog!("Failed: {}", msg);
    Err("Failed to execute file")
  } else {
    Ok(())
  }
}

fn build(
  file: &str,
  output: &str,
  include: Vec<String>,
  depfile: Option<&str>,
  as_json: bool,
) -> Result<(), Error> {
  shlog!("Parsing file: {}", file);

  let (deps, ast) = {
    let file_path = Path::new(&file);
    let file_path = std::fs::canonicalize(file_path).unwrap();
    let mut file_content = std::fs::read_to_string(file).map_err(|_| "File not found")?;
    // add new line at the end of the file to be able to parse it correctly
    file_content.push('\n');

    // get absolute parent path of the file
    let parent_path = file_path.parent().unwrap().to_str().unwrap();

    let mut env = ReadEnv::new(
      file_path.to_str().unwrap(),
      parent_path.to_string(),
      include,
    );
    let ast = read_with_env(&file_content, &mut env).map_err(|e| {
      shlog!("Error: {:?}", e);
      "Failed to parse file"
    })?;
    let mut deps = get_dependencies(&env)
      .iter()
      .map(|x| {
        dunce::canonicalize(x)
          .map_err(|_| "Failed to canonicalize path")
          .map(|x| x.to_string_lossy().to_string())
      })
      .collect::<Result<Vec<String>, _>>()?;
    // Add the main file as well
    let p = dunce::canonicalize(file_path)
      .map_err(|_| "Failed to canonicalize path")?
      .to_string_lossy()
      .to_string();
    deps.push(p);
    (deps, ast)
  };

  // write sequence to file
  {
    let mut file = std::fs::File::create(output).unwrap();
    let mut writer = std::io::BufWriter::new(&mut file);

    if !as_json {
      // Serialize using flexbuffers
      let encoded_bin = flexbuffers::to_vec(&ast).unwrap();
      writer
        .write(fourCharacterCode(*b"SHRD").to_be_bytes().as_ref())
        .unwrap();
      writer
        .write(SHARDS_CURRENT_ABI.to_le_bytes().as_ref())
        .unwrap();
      writer.write_all(encoded_bin.as_slice()).unwrap();
    } else {
      let encoded_json = serde_json::to_string_pretty(&ast).unwrap();
      writer.write_all(encoded_json.as_bytes()).unwrap();
    }
  }

  if let Some(out_dep_file) = depfile {
    let mut file = std::fs::File::create(out_dep_file).unwrap();
    let mut writer = std::io::BufWriter::new(&mut file);

    writer.write_all(output.as_bytes()).unwrap();
    writer.write_all(b": ").unwrap();
    for dep in deps {
      writer.write_all(dep.as_bytes()).unwrap();
      writer.write_all(b" ").unwrap();
    }
  }
  Ok(())
}

fn execute(eargs: &RunArgs, cancellation_token: Arc<AtomicBool>) -> Result<(), Error> {
  let RunArgs {
    file,
    decompress_strings,
    skip_cwd,
    include: in_include_paths,
    args,
  } = eargs;

  if *decompress_strings {
    unsafe {
      shards_decompress_strings();
    }
  }

  shlog_debug!("Evaluating file: {}", file);

  let ast = {
    let file_path = Path::new(&file);
    let file_path = std::fs::canonicalize(file_path).unwrap();
    let mut file_content = std::fs::read_to_string(file).map_err(|_| "File not found")?;
    // add new line at the end of the file to be able to parse it correctly
    file_content.push('\n');

    let parent_path = file_path.parent().unwrap().to_str().unwrap();

    let mut include_paths = Vec::new();
    for path in in_include_paths {
      let path = std::path::PathBuf::from(path);
      let path = path
        .canonicalize()
        .map_err(|x| format!("Failed to canonicalize path: {} ({:?})", x, path))?;
      include_paths.push(path.to_string_lossy().to_string());
    }

    if !*skip_cwd {
      // get absolute parent path of the file
      let c_parent_path = std::ffi::CString::new(parent_path).unwrap();
      // set it as root path
      unsafe { (*Core).setRootPath.unwrap()(c_parent_path.as_ptr() as *const c_char) };
    }

    read(
      &file_content,
      file_path.to_str().unwrap(),
      parent_path.to_string(),
      include_paths,
    )
    .map_err(|e| {
      shlog!("Error: {:?}", e);
      "Failed to parse file"
    })?
  };

  Ok(execute_seq(args, ast, cancellation_token)?)
}
