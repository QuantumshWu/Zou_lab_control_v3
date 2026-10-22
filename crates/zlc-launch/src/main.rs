//! The four programs at the root of a checkout -- experiment.exe,
//! figure_viewer.exe, pulse_editor.exe and installation.exe -- are this one
//! launcher under four names (tools\release.ps1 copies it). It starts the
//! engine, bin\engine\zlc-desktop.exe, in the mode its own file name says,
//! hands it the file dropped on it, and ends; the engine reports its own
//! errors. It has no window and no console.
//!
//! The engine is not in git: bin\release.json names the release this checkout
//! runs. A downloaded engine waiting in bin\engine.next goes in first. An
//! engine that is missing or of another release is never started:
//! installation.exe puts this checkout's release in place in a console window
//! of its own (tools\fetch_engine.ps1) and then opens, and the other three open
//! installation.exe for it.
#![windows_subsystem = "windows"]
use std::ffi::OsString;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if let Err(error) = launch() {
        let text: Vec<u16> = error.encode_utf16().chain([0]).collect();
        let title: Vec<u16> = "Zou lab control".encode_utf16().chain([0]).collect();
        use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW};
        unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND) };
        std::process::exit(1);
    }
}

fn launch() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|error| format!("Cannot tell where this program is: {error}"))?;
    let name = exe.file_stem().and_then(|stem| stem.to_str()).unwrap_or_default().to_ascii_lowercase();
    let root = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    let bin = root.join("bin");
    let engine = bin.join("engine").join("zlc-desktop.exe");
    // A file dropped on the program: Explorer gives its full path, a shell may give one relative to it.
    let file: Option<OsString> = std::env::args_os().nth(1).map(|arg| std::path::absolute(&arg).map(PathBuf::into_os_string).unwrap_or(arg));
    let mode: &[&str] = match name.as_str() {
        "experiment" => if file.is_some() { &["--open"] } else { &[] },
        "figure_viewer" => &["--figure-viewer"],
        "pulse_editor" => &["--pulse-editor"],
        "installation" => &["--installer"],
        other => return Err(format!("{other}.exe is none of experiment, figure_viewer, pulse_editor and installation: rename it back.")),
    };
    // A downloaded engine goes in before anything starts (a closing engine is given a few seconds).
    if bin.join("engine.next").is_dir() { fetch(&root, &["-Swap", "10"], false)?; }
    if !current(&bin) {
        if name != "installation" {
            Command::new(exe.with_file_name("installation.exe")).args(["--for", &name]).spawn()
                .map_err(|error| format!("This checkout's program is not in place, and installation.exe, which puts it there, does not start: {error}"))?;
            return Ok(());
        }
        let asker = std::env::args().skip_while(|arg| arg != "--for").nth(1).unwrap_or_else(|| name.clone());
        // The console window said why it failed and waited for the operator.
        if !fetch(&root, &["-For", &asker], true)?.success() { std::process::exit(1); }
        if !current(&bin) { return Err(format!("{} is still not this checkout's release (bin\\release.json): start installation.exe again once every Zou lab program is closed.", engine.display())); }
    }
    let mut command = Command::new(&engine);
    command.args(mode);
    if name != "installation" { command.args(file); }
    command.spawn().map_err(|error| format!("Cannot start {}: {error}", engine.display()))?;
    Ok(())
}

/// Whether bin\engine is the release bin\release.json names (the same source stamp in its
/// SOURCE). A checkout that names no release runs the engine it has.
fn current(bin: &Path) -> bool {
    if !bin.join("engine").join("zlc-desktop.exe").is_file() { return false; }
    let Ok(record) = std::fs::read_to_string(bin.join("release.json")) else { return true };
    let have = std::fs::read_to_string(bin.join("engine").join("SOURCE")).ok()
        .and_then(|text| text.lines().find_map(|line| line.strip_prefix("source ").map(|stamp| stamp.trim().to_owned())));
    field(&record, "source").is_some_and(|wanted| Some(wanted) == have)
}

/// A string value of tools\release.ps1's JSON (one level, no escapes in its values).
fn field(json: &str, key: &str) -> Option<String> {
    let rest = &json[json.find(&format!("\"{key}\""))? + key.len() + 2..];
    let rest = rest[rest.find(':')? + 1..].trim_start();
    let value = rest.strip_prefix('"')?;
    Some(value[..value.find('"')?].to_owned())
}

/// tools\fetch_engine.ps1 with these arguments, waited for: in a console window of its own when
/// `shown` (its output is the operator's progress; a failure waits for Enter), else hidden. Windows
/// parked off-screen (ZLC_V3_WINDOW_AT) show none.
fn fetch(root: &Path, arguments: &[&str], shown: bool) -> Result<std::process::ExitStatus, String> {
    const CREATE_NEW_CONSOLE: u32 = 0x10;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let shown = shown && std::env::var_os("ZLC_V3_WINDOW_AT").is_none();
    let script = root.join("tools").join("fetch_engine.ps1");
    Command::new("powershell.exe").args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]).arg(&script).args(arguments)
        .args(shown.then_some("-Pause"))
        .creation_flags(if shown { CREATE_NEW_CONSOLE } else { CREATE_NO_WINDOW })
        .status().map_err(|error| format!("Cannot run {}: {error}", script.display()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_field_is_read_from_release_json_as_powershell_writes_it() {
        let record = "{\r\n    \"version\":  \"2026.10.22.1\",\r\n    \"source\":  \"ab12\",\r\n    \"asset\":  {\r\n        \"name\":  \"zlc-engine-2026.10.22.1.zip\",\r\n        \"size\":  123\r\n    }\r\n}";
        assert_eq!(super::field(record, "source").as_deref(), Some("ab12"));
        assert_eq!(super::field(record, "version").as_deref(), Some("2026.10.22.1"));
        assert_eq!(super::field(record, "size"), None);
        assert_eq!(super::field(record, "commit"), None);
    }
}
