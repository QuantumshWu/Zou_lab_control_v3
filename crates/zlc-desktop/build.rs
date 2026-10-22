//! Ships the renderer's fonts beside the executable (`fonts/`), where
//! zlc_plot::render::fonts resolves them, and gives zlc-desktop.exe the product
//! icon and its application manifest (assets/zlc-desktop.manifest: Windows 8+,
//! for the title bar's layered child window) as compiled resources (zlc-resources).
use std::path::{Path, PathBuf};

fn main() {
    let manifest = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("assets/zlc-desktop.manifest");
    println!("cargo:rerun-if-changed={}", manifest.display());
    zlc_resources::embed(Some(&std::fs::read(&manifest).expect("the manifest")));
    let source = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../zlc-plot/fonts");
    println!("cargo:rerun-if-changed={}", source.display());
    // OUT_DIR is <profile directory>/build/<package>-<hash>/out.
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let fonts = out.ancestors().nth(3).expect("the profile directory").join("fonts");
    std::fs::create_dir_all(&fonts).expect("fonts directory");
    for entry in std::fs::read_dir(&source).expect("the renderer's fonts") {
        let entry = entry.expect("font entry");
        std::fs::copy(entry.path(), fonts.join(entry.file_name())).expect("copy font");
    }
}
