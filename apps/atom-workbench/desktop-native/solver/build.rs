use std::{collections::BTreeMap,fs,path::{Component,Path,PathBuf}};
use sha2::{Sha256,Digest};
use serde_json::Value;

const POLICY:&str="native/numeric-source-inputs.json";
fn relative_path(value:&str)->PathBuf{
    assert!(!value.is_empty()&&!value.contains('\\')&&!value.contains(':')&&
        value.split('/').all(|part|!part.is_empty()&&part!="."&&part!=".."),"Invalid app-relative numeric input: {value}");
    let path=PathBuf::from(value);
    assert!(path.components().all(|part|matches!(part,Component::Normal(_))),"Numeric input must stay app-relative: {value}");
    path
}
fn watch(path:&Path){
    // An existing parent also detects optional core/build-script creation.
    let mut target=path;while !target.exists(){target=target.parent().expect("Numeric input has no existing parent");}
    println!("cargo:rerun-if-changed={}",target.display());
}
fn checked(app:&Path,path:&Path)->PathBuf{
    let metadata=fs::symlink_metadata(path).expect("Read numeric source metadata");
    assert!(!metadata.file_type().is_symlink(),"Numeric source symlinks are not provenance inputs: {}",path.display());
    let canonical=path.canonicalize().expect("Canonical numeric input");
    assert!(canonical.starts_with(app),"Numeric input escaped application source root: {}",path.display());canonical
}
fn insert(app:&Path,path:&Path,files:&mut BTreeMap<String,PathBuf>){
    let path=checked(app,path);assert!(path.is_file(),"Numeric input is not a file: {}",path.display());
    let relative=path.strip_prefix(app).unwrap().to_str().expect("UTF8 numeric source path").replace('\\',"/");files.insert(relative,path);
}
fn collect_tree(app:&Path,directory:&Path,extensions:&[&str],files:&mut BTreeMap<String,PathBuf>){
    let directory=checked(app,directory);assert!(directory.is_dir(),"Numeric source tree is not a directory");watch(&directory);
    for entry in fs::read_dir(&directory).expect("Read numeric source tree"){
        let path=entry.expect("Read numeric source entry").path();let metadata=fs::symlink_metadata(&path).expect("Numeric source metadata");
        assert!(!metadata.file_type().is_symlink(),"Numeric source tree contains a symlink: {}",path.display());
        if metadata.is_dir(){collect_tree(app,&path,extensions,files);}
        else if metadata.is_file()&&path.extension().and_then(|v|v.to_str()).is_some_and(|ext|extensions.iter().any(|wanted|ext.eq_ignore_ascii_case(wanted))){insert(app,&path,files);}
    }
}
fn source_files(app:&Path)->Vec<(String,PathBuf)>{
    let policy:Value=serde_json::from_slice(&fs::read(app.join(POLICY)).expect("Numeric provenance policy")).expect("Parse numeric provenance policy");
    assert_eq!(policy["schema"],"atom-numeric-source-inputs/1","Unsupported numeric provenance policy");let mut files=BTreeMap::new();
    for tree in policy["trees"].as_array().expect("Numeric source trees"){
        let path=app.join(relative_path(tree["path"].as_str().expect("Numeric tree path")));watch(&path);
        if !path.exists(){assert!(!tree["required"].as_bool().unwrap_or(true),"Required numeric source tree missing: {}",path.display());continue;}
        let extensions:Vec<_>=tree["extensions"].as_array().expect("Numeric tree extensions").iter().map(|x|x.as_str().expect("Numeric extension")).collect();
        collect_tree(app,&path,&extensions,&mut files);
    }
    for file in policy["files"].as_array().expect("Numeric input files"){
        let path=app.join(relative_path(file["path"].as_str().expect("Numeric file path")));watch(&path);
        if !path.exists(){assert!(!file["required"].as_bool().unwrap_or(true),"Required numeric input missing: {}",path.display());continue;}insert(app,&path,&mut files);
    }files.into_iter().collect()
}
fn main(){
    println!("cargo:rerun-if-env-changed=ATOM_STATIC_NUMERIC_DIR");
    let manifest_dir=PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let app=manifest_dir.join("../..").canonicalize().expect("Canonical Atom Workbench root");
    let root=std::env::var_os("ATOM_STATIC_NUMERIC_DIR").map(PathBuf::from).unwrap_or_else(||app.join(".cache/native/lib"));
    for name in ["openblas.lib","atom_superlu.lib"]{let path=root.join(name);assert!(path.is_file(),"Missing native static archive: {}",path.display());println!("cargo:rerun-if-changed={}",path.display());}
    println!("cargo:rustc-link-search=native={}",root.display());println!("cargo:rustc-link-lib=static=atom_superlu");println!("cargo:rustc-link-lib=static=openblas");
    let records:Vec<_>=source_files(&app).into_iter().map(|(relative,path)|{println!("cargo:rerun-if-changed={}",path.display());
        serde_json::json!({"path":relative,"sha256":format!("{:x}",Sha256::digest(fs::read(path).expect("Read numeric input")))})}).collect();
    let libraries:Vec<_>=["openblas.lib","atom_superlu.lib"].into_iter().map(|name|serde_json::json!({"file":name,"sha256":format!("{:x}",Sha256::digest(fs::read(root.join(name)).unwrap()))})).collect();
    let manifest=serde_json::json!({"schema":"atom-static-numerics/2","pathBase":"atom-workbench","inputPolicy":POLICY,
        "source":"same canonical Rust solver and population core compiled as rlibs","files":records,"libraries":libraries});
    let output=PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("numeric-source.json");fs::write(&output,serde_json::to_vec(&manifest).unwrap()).unwrap();
    println!("cargo:rustc-env=ATOM_NUMERIC_MANIFEST={}",output.display());
}
