use std::path::PathBuf;
use sha2::{Sha256,Digest};
fn main(){
    println!("cargo:rerun-if-env-changed=ATOM_STATIC_NUMERIC_DIR");
    let root=std::env::var_os("ATOM_STATIC_NUMERIC_DIR").map(PathBuf::from)
        .unwrap_or_else(||PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../.cache/native/lib"));
    for name in ["openblas.lib","atom_superlu.lib"]{let path=root.join(name);assert!(path.is_file(),"Missing native static archive: {}",path.display());println!("cargo:rerun-if-changed={}",path.display());}
    println!("cargo:rustc-link-search=native={}",root.display());
    println!("cargo:rustc-link-lib=static=atom_superlu");println!("cargo:rustc-link-lib=static=openblas");
    let source=PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../rust-solver/src");
    let mut files:Vec<_>=std::fs::read_dir(&source).unwrap().filter_map(Result::ok).map(|entry|entry.path()).filter(|path|path.extension().is_some_and(|value|value=="rs")).collect();files.sort();
    let records:Vec<_>=files.into_iter().map(|path|{println!("cargo:rerun-if-changed={}",path.display());serde_json::json!({"file":path.file_name().unwrap().to_string_lossy(),"sha256":format!("{:x}",Sha256::digest(std::fs::read(&path).unwrap()))})}).collect();
    let libraries:Vec<_>=["openblas.lib","atom_superlu.lib"].into_iter().map(|name|serde_json::json!({"file":name,"sha256":format!("{:x}",Sha256::digest(std::fs::read(root.join(name)).unwrap()))})).collect();
    let manifest=serde_json::json!({"schema":"atom-static-numerics/1","source":"same Rust solver src compiled as rlib","files":records,"libraries":libraries});
    let output=PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("numeric-source.json");std::fs::write(&output,serde_json::to_vec(&manifest).unwrap()).unwrap();
    println!("cargo:rustc-env=ATOM_NUMERIC_MANIFEST={}",output.display());
}
