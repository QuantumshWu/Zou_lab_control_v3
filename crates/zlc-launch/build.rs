//! The launcher wears the product icon under each of its four names, and its
//! manifest (launch.manifest: run as the operator) -- both through zlc-resources.
fn main() {
    let manifest = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("launch.manifest");
    println!("cargo:rerun-if-changed={}", manifest.display());
    zlc_resources::embed(Some(&std::fs::read(&manifest).expect("the manifest")));
}
