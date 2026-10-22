//! Links the static FreeType 2.6.1 library built exactly like Matplotlib's
//! wheel (`freetype.lib` in `ZLC_FREETYPE_DIR`).
fn main() {
    println!("cargo:rerun-if-env-changed=ZLC_FREETYPE_DIR");
    let directory = std::env::var("ZLC_FREETYPE_DIR")
        .expect("ZLC_FREETYPE_DIR must name the directory holding freetype.lib");
    println!("cargo:rerun-if-changed={directory}/freetype.lib");
    println!("cargo:rustc-link-search=native={directory}");
    println!("cargo:rustc-link-lib=static=freetype");
}
