//! The dump test over every scenario directory of a corpus dump
//! (env-gated: ZLC_FIGURE_CORPUS = dump.py --corpus output root).
mod support;

use std::path::PathBuf;

#[test]
fn corpus_streams_match_matplotlib() {
    let Ok(root) = std::env::var("ZLC_FIGURE_CORPUS") else { return };
    zlc_figure::wire::set_blob_root(std::env::var("ZLC_FIGURE_BLOBS").ok().map(PathBuf::from));
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let (mut exact, mut total, mut calls_equal, mut calls_total) = (0, 0, 0, 0);
    let mut failing = Vec::new();
    for dir in &dirs {
        let r = support::run_dir(dir, 2);
        exact += r.exact;
        total += r.total;
        calls_equal += r.calls_equal;
        calls_total += r.calls_total;
        if r.exact != r.total { failing.push((dir.file_name().unwrap().to_string_lossy().into_owned(), r)); }
    }
    for (name, r) in &failing {
        eprintln!("FAIL {name}: {}/{} fronts", r.exact, r.total);
        for (reason, n) in &r.failures { eprintln!("    {n:4} {reason}"); }
    }
    eprintln!("corpus: fronts exact {exact}/{total}; calls equal {calls_equal}/{calls_total}; scenarios failing {}", failing.len());
    assert!(total > 0);
    assert_eq!(exact, total);
}
