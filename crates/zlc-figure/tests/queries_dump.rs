//! The Scene query API against matplotlib's own geometry, per dumped front
//! (env-gated: ZLC_FIGURE_DUMP or ZLC_FIGURE_CORPUS, ZLC_FIGURE_BLOBS).
mod support;

use std::path::PathBuf;

#[test]
fn dumped_queries_match_matplotlib() {
    zlc_figure::wire::set_blob_root(std::env::var("ZLC_FIGURE_BLOBS").ok().map(PathBuf::from));
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("ZLC_FIGURE_DUMP") { if !dir.is_empty() { dirs.push(dir.into()); } }
    if let Ok(root) = std::env::var("ZLC_FIGURE_CORPUS") {
        let mut sub: Vec<PathBuf> = std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
        sub.sort();
        dirs.extend(sub);
    }
    if dirs.is_empty() { return; }
    let (mut equal, mut total) = (0, 0);
    let mut bad = Vec::new();
    for dir in &dirs {
        let (e, t, b) = support::run_queries_dir(dir, 12);
        equal += e;
        total += t;
        bad.extend(b.into_iter().take(3));
    }
    for line in bad.iter().take(20) { eprintln!("  {line}"); }
    eprintln!("queries equal {equal}/{total}");
    assert_eq!(equal, total);
}

/// A dumped front whose IR does not load counts as failed, never skipped.
#[test]
fn an_ir_that_does_not_load_fails() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("queries-unloadable");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("00001.f.ir.json"), r#"{"version": 1}"#).unwrap();
    std::fs::write(dir.join("00001.f.l2.json"), r#"{"queries": {"axes": [], "texts": {}}}"#).unwrap();
    let (equal, total, bad) = support::run_queries_dir(&dir, 12);
    assert_eq!((equal, total), (0, 1), "{bad:?}");
    assert!(bad[0].contains("IR does not load"), "{bad:?}");
}
