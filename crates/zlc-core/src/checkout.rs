//! The checkout a program serves: the folder its UI page, `science\src`, the
//! FPGA tree, fonts and recipes are read from at run time. It is found from
//! where the executable sits -- a release in `bin\engine\` and a build in
//! `target\<triple>\release\` both lie inside the checkout they serve -- so an
//! executable built on one machine runs from another machine's clone. What a
//! running program writes stays in the checkout too, in [`local_root`].
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Whether `folder` is the root of this repository's checkout.
fn is_checkout(folder: &Path) -> bool {
    folder.join("Cargo.toml").is_file() && folder.join("crates").join("zlc-core").is_dir() && folder.join("science").join("src").is_dir()
}

/// The checkout an executable (or any file) at `path` lies in.
pub fn of(path: &Path) -> Option<PathBuf> { path.ancestors().skip(1).find(|folder| is_checkout(folder)).map(Path::to_path_buf) }

/// The nearest checkout above the running executable. An executable built
/// outside every checkout (a test harness's own build directory) serves the
/// checkout it was compiled from (build.rs); nothing else carries that path.
pub fn root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let exe = std::env::current_exe().unwrap_or_default();
        of(&exe).or_else(|| option_env!("ZLC_CHECKOUT_COMPILED_FROM").map(PathBuf::from))
            .unwrap_or_else(|| exe.parent().map(Path::to_path_buf).unwrap_or_default())
    })
}

/// `relative` (a `/`-separated path) inside the checkout.
pub fn path(relative: &str) -> PathBuf { relative.split('/').fold(root().to_path_buf(), |path, part| path.join(part)) }

/// The folder a checkout's running programs write in, ignored by git: `.local` (see [`local`]).
pub const LOCAL: &str = ".local";

/// This checkout's runtime folder, `<checkout>\.local`; `ZLC_V3_LOCAL` names
/// another (a harness that keeps several instruments of one checkout apart).
pub fn local_root() -> PathBuf {
    std::env::var_os("ZLC_V3_LOCAL").filter(|value| !value.is_empty()).map(PathBuf::from).unwrap_or_else(|| root().join(LOCAL))
}

/// One subfolder of [`local_root`]: `webview2` (the windows' browser profile),
/// `mcp` (the MCP token and this checkout's running-instrument records),
/// `logs` (zlc-desktop's output when nobody redirected it),
/// `vivado` (the pulse server's JTAG session), and the plugins' `simulation`,
/// `slm_gpu_cache` and `cupy` caches.
pub fn local(part: &str) -> PathBuf { local_root().join(part) }
