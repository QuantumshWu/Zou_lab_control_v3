# Portable Rust

`install-portable-rust.ps1` installs Rust and Cargo 1.99.1 (host
`x86_64-pc-windows-gnu`, LLVM 23.1.1) into the checkout's `.tools` (or
`-ToolsRoot`) from the official `rustc`, `cargo`, `rust-std` and `rust-mingw`
archives of the stable release dated 2026-10-15, each checked against its SHA-256
from that release's manifest. No Rust or Visual Studio installer runs; no
registry, global PATH, default Rust or system configuration changes.
`tools\dev_setup.ps1` (`bin\scripts\dev_setup.bat`, installation.exe's Advanced card) calls it first, then adds the MSVC
target, lld-link, the SDK libraries, FreeType and Node that the desktop build
needs.

`build-release.ps1` is the toolchain's smoke check: it builds this folder's
dependency-free package offline into `target\` here and runs it (`zlc-v3 Rust
release smoke passed: sum=5050`), restoring the process environment afterwards.

Sources: [Rust Windows toolchain guidance](https://rust-lang.github.io/rustup/installation/windows.html)
and [official stable release manifest](https://static.rust-lang.org/dist/channel-rust-stable.toml).
