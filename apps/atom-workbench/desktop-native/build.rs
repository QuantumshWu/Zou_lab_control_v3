//! Use the same product resources as v3, including Explorer's executable icon.
fn main() {
    const MANIFEST: &[u8] = include_bytes!("../../../crates/zlc-launch/launch.manifest");
    println!("cargo:rerun-if-changed=../../../crates/zlc-launch/launch.manifest");
    println!("cargo:rerun-if-changed=../../../crates/zlc-desktop/assets/zlc.ico");
    println!("cargo:rerun-if-env-changed=ATOM_V3_ICON_SOURCE");
    let icon=std::env::var_os("ATOM_V3_ICON_SOURCE").map(std::path::PathBuf::from).unwrap_or_else(||std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../../crates/zlc-desktop/assets/zlc.ico"));
    println!("cargo:rerun-if-changed={}",icon.display());println!("cargo:rustc-env=ATOM_V3_ICON_FILE={}",icon.display());
    zlc_resources::embed_icon(&std::fs::read(icon).expect("current v3 product icon"),Some(MANIFEST));
}
