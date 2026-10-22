//! Every dumped IR deserializes (env-gated: ZLC_FIGURE_DUMP = a dump.py output
//! directory, ZLC_FIGURE_BLOBS = its blob store).
use std::path::PathBuf;

#[test]
fn dumped_ir_deserializes() {
    let Ok(dir) = std::env::var("ZLC_FIGURE_DUMP") else { return };
    let blobs = std::env::var("ZLC_FIGURE_BLOBS").ok().map(PathBuf::from);
    zlc_figure::wire::set_blob_root(blobs);
    let mut count = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.to_string_lossy().ends_with(".ir.json") { continue; }
        let text = std::fs::read_to_string(&path).unwrap();
        let figure: zlc_figure::ir::Figure = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(figure.version, 1);
        count += 1;
    }
    eprintln!("{count} IR files");
    assert!(count > 0);
}
