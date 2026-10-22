//! Rust(IR) -> calls == matplotlib's recorded L2 stream, call by call, bit for
//! bit, over one dump.py output directory (env-gated: ZLC_FIGURE_DUMP,
//! ZLC_FIGURE_BLOBS; ZLC_FIGURE_ONLY=substring filters files;
//! ZLC_FIGURE_VERBOSE=n prints n differing paths per failing front).
mod support;

use std::path::PathBuf;

#[test]
fn dump_streams_match_matplotlib() {
    let Ok(dir) = std::env::var("ZLC_FIGURE_DUMP") else { return };
    zlc_figure::wire::set_blob_root(std::env::var("ZLC_FIGURE_BLOBS").ok().map(PathBuf::from));
    let r = support::run_dir(std::path::Path::new(&dir), 6);
    eprintln!("fronts exact {}/{}; calls equal {}/{}", r.exact, r.total, r.calls_equal, r.calls_total);
    for (reason, n) in &r.failures { eprintln!("  {n:5} {reason}"); }
    assert!(r.total > 0);
    assert_eq!(r.exact, r.total, "not every front matches");
}
