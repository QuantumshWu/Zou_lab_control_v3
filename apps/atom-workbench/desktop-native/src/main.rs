#![windows_subsystem = "windows"]
mod host;
mod embedded_assets;
mod app_icon;
mod chrome_probe;
include!(concat!(env!("OUT_DIR"), "/shared_window_chrome.rs"));
fn main(){
    if std::env::args().any(|argument|argument=="--backend-info"){
        #[cfg(feature="embedded-solver")]
        println!("{}",serde_json::json!({"staticSolver":true,"v3IconSha256":app_icon::source_sha256(),"windowChromeSource":serde_json::from_str::<serde_json::Value>(include_str!(env!("ATOM_SHARED_CHROME_MANIFEST"))).unwrap(),"numericSource":serde_json::from_str::<serde_json::Value>(atom_lindblad_embedded::embedded_source_manifest()).unwrap()}));
        #[cfg(not(feature="embedded-solver"))]
        println!("{}",serde_json::json!({"staticSolver":false,"v3IconSha256":app_icon::source_sha256(),"windowChromeSource":serde_json::from_str::<serde_json::Value>(include_str!(env!("ATOM_SHARED_CHROME_MANIFEST"))).unwrap()}));
        return;
    }
    if std::env::args().any(|argument|argument=="--worker"){
        #[cfg(feature="embedded-solver")]
        {atom_lindblad_embedded::main();return;}
        #[cfg(not(feature="embedded-solver"))]
        {eprintln!("This development launcher does not include the static solver.");std::process::exit(1);}
    }
    host::main();
}
