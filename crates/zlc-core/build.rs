//! An executable built into a target folder outside its checkout (a test
//! harness's own) cannot find the checkout from where it sits, so it is told
//! the checkout it was compiled from. One built inside the checkout -- every
//! development build and release -- carries no machine path.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let checkout = manifest.ancestors().nth(2).expect("crates/zlc-core lies in a checkout").to_path_buf();
    if !PathBuf::from(std::env::var("OUT_DIR").unwrap()).starts_with(&checkout) {
        println!("cargo:rustc-env=ZLC_CHECKOUT_COMPILED_FROM={}", checkout.display());
    }
}
