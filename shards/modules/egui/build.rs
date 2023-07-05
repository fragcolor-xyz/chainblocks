extern crate bindgen;
extern crate gfx_build;

use std::env;
use std::path::PathBuf;

fn main() {
  let gfx_path = "../../gfx";
  let shards_root = "../../..";

  let builder = gfx_build::setup_bindgen_for_gfx(gfx_path, bindgen::Builder::default());

  let bindings = builder
    .header("rust_interop.hpp")
    .allowlist_type("egui::.*")
    .allowlist_var("SDLK_.*")
    .allowlist_type("SDL_KeyCode")
    .size_t_is_usize(true)
    .layout_tests(false)
    .clang_arg(format!("-I{}/include", shards_root))
    .generate()
    .expect("Unable to generate bindings");

  let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
  bindings
    .write_to_file(out_path.join("bindings.rs"))
    .expect("Couldn't write bindings!");
}
