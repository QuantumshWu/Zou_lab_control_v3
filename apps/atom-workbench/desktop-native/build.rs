//! Compile the same native frame/chrome modules and icon that v3 builds.
use std::{path::PathBuf,process::Command};
use sha2::{Digest,Sha256};
fn main(){
    let manifest_dir=PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=ATOM_V3_SOURCE_ROOT");
    println!("cargo:rerun-if-env-changed=ATOM_V3_ICON_SOURCE");
    let root=std::env::var_os("ATOM_V3_SOURCE_ROOT").map(PathBuf::from).unwrap_or_else(||{
        let output=Command::new("git").current_dir(&manifest_dir).args(["rev-parse","--path-format=absolute","--git-common-dir"]).output().expect("locate current shared v3 source");
        assert!(output.status.success(),"Set ATOM_V3_SOURCE_ROOT to the v3 source checkout");
        PathBuf::from(String::from_utf8(output.stdout).unwrap().trim()).parent().unwrap().to_owned()
    });
    let output=PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let mut modules=String::new();let mut files=Vec::new();
    for name in ["frame","chrome"]{
        let relative=format!("crates/zlc-desktop/src/{name}.rs");let path=root.join(&relative).canonicalize().expect("shared v3 native chrome source");
        println!("cargo:rerun-if-changed={}",path.display());
        modules+=&format!("#[path = {:?}]\nmod {name};\n",path.to_string_lossy());
        files.push(serde_json::json!({"path":relative,"sha256":format!("{:x}",Sha256::digest(std::fs::read(path).unwrap()))}));
    }
    std::fs::write(output.join("shared_window_chrome.rs"),modules).unwrap();
    let proof=output.join("shared-window-chrome.json");
    std::fs::write(&proof,serde_json::to_vec(&serde_json::json!({"schema":"zlc-shared-window-chrome/1","files":files})).unwrap()).unwrap();
    println!("cargo:rustc-env=ATOM_SHARED_CHROME_MANIFEST={}",proof.display());
    let manifest=root.join("crates/zlc-launch/launch.manifest");println!("cargo:rerun-if-changed={}",manifest.display());
    let icon=std::env::var_os("ATOM_V3_ICON_SOURCE").map(PathBuf::from).unwrap_or_else(||root.join("crates/zlc-desktop/assets/zlc.ico"));
    println!("cargo:rerun-if-changed={}",icon.display());println!("cargo:rustc-env=ATOM_V3_ICON_FILE={}",icon.display());
    zlc_resources::embed_icon(&std::fs::read(icon).expect("current v3 product icon"),Some(&std::fs::read(manifest).expect("shared desktop manifest")));
}
