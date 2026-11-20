//! Restart the device node that owns a serial port (`restart_serial_port`, zlc_workbench::PortRestart).
//!
//! The node is the present Ports-class device whose `PortName` is the port: for a CH340 or CP210x
//! the USB device itself, for an FTDI adapter its FTDIBUS node. It is restarted with
//! `pnputil /restart-device "<instance id>"`, which needs an administrator: run elevated through
//! ShellExecuteExW's `runas` verb, so Windows shows its one confirmation (nothing privileged stays
//! installed), hidden, on the caller's thread -- never the window's -- and waited for within
//! [`RESTART_WAIT`]. pnputil's words are kept (through `cmd`, into a file of its own) so a failure
//! says what pnputil said.
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    DICS_FLAG_GLOBAL, DIGCF_PRESENT, DIREG_DEV, GUID_DEVCLASS_PORTS, SP_DEVINFO_DATA, SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInfo,
    SetupDiGetClassDevsW, SetupDiGetDeviceInstanceIdW, SetupDiOpenDevRegKey,
};
use windows_sys::Win32::Foundation::{CloseHandle, ERROR_CANCELLED, GetLastError, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::Globalization::{CP_OEMCP, MultiByteToWideChar};
use windows_sys::Win32::System::Com::{COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoInitializeEx, CoUninitialize};
use windows_sys::Win32::System::Registry::{KEY_READ, REG_SZ, RegCloseKey, RegQueryValueExW};
use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;
use windows_sys::Win32::System::Threading::{GetExitCodeProcess, TerminateProcess, WaitForSingleObject};
use windows_sys::Win32::UI::Shell::{SEE_MASK_NOASYNC, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW, ShellExecuteExW};
use windows_sys::Win32::UI::WindowsAndMessaging::SW_HIDE;
use zlc_pulse::serial::com_name;
use zlc_workbench::PortRestart;

/// How long a restart may take from pnputil's start until the port is there again: pnputil stops
/// and starts one node in seconds; a node that will not stop (a handle still open on it) is not
/// waited for longer. The operator's answer to Windows' confirmation comes before and is not timed.
pub const RESTART_WAIT: Duration = Duration::from_secs(60);

/// How often the restarted port is looked for while it comes back.
const PORT_POLL: Duration = Duration::from_millis(100);

fn wide(text: &str) -> Vec<u16> { text.encode_utf16().chain(std::iter::once(0)).collect() }

fn text_of(units: &[u16]) -> String { String::from_utf16_lossy(&units[..units.iter().position(|unit| *unit == 0).unwrap_or(units.len())]) }

/// The present Ports-class device nodes: (instance id, `PortName`) of each that names a port.
fn port_nodes() -> Vec<(String, String)> {
    let set = unsafe { SetupDiGetClassDevsW(&GUID_DEVCLASS_PORTS, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT) };
    if set == INVALID_HANDLE_VALUE as isize { return Vec::new(); }
    let mut nodes = Vec::new();
    for index in 0.. {
        let mut device: SP_DEVINFO_DATA = unsafe { std::mem::zeroed() };
        device.cbSize = std::mem::size_of::<SP_DEVINFO_DATA>() as u32;
        if unsafe { SetupDiEnumDeviceInfo(set, index, &mut device) } == 0 { break; }
        let mut id = [0u16; 512];
        if unsafe { SetupDiGetDeviceInstanceIdW(set, &device, id.as_mut_ptr(), id.len() as u32, std::ptr::null_mut()) } == 0 { continue; }
        let key = unsafe { SetupDiOpenDevRegKey(set, &device, DICS_FLAG_GLOBAL, 0, DIREG_DEV, KEY_READ) };
        if key.is_null() || key as isize == INVALID_HANDLE_VALUE as isize { continue; }
        let (name, mut kind) = (wide("PortName"), 0u32);
        let mut port = [0u16; 64];
        let mut size = (port.len() * 2) as u32;
        let read = unsafe { RegQueryValueExW(key, name.as_ptr(), std::ptr::null(), &mut kind, port.as_mut_ptr().cast(), &mut size) };
        unsafe { RegCloseKey(key) };
        if read == 0 && kind == REG_SZ { nodes.push((text_of(&id), text_of(&port))); }
    }
    unsafe { SetupDiDestroyDeviceInfoList(set) };
    nodes
}

/// The node among `nodes` (instance id, `PortName`) that owns `port`, however either spells it.
fn owner<'a>(nodes: &'a [(String, String)], port: &str) -> Option<&'a str> {
    nodes.iter().find(|(_, name)| com_name(name) == com_name(port)).map(|(id, _)| id.as_str())
}

/// `cmd`'s arguments that run `pnputil` on `node` with all it says written to `output`: one
/// quoted command line (`/s`: cmd strips only the outer quotes; `/v:off`: no `!` expansion). None
/// when a path or the id holds what cmd would not take literally inside quotes (`"`, `%`).
fn command_line(pnputil: &Path, node: &str, output: &Path) -> Option<String> {
    let (pnputil, output) = (pnputil.display().to_string(), output.display().to_string());
    if [pnputil.as_str(), node, output.as_str()].iter().any(|part| part.contains(['"', '%'])) { return None; }
    Some(format!(r#"/d /v:off /s /c ""{pnputil}" /restart-device "{node}" > "{output}" 2>&1""#))
}

/// What pnputil wrote, as text: UTF-16 when it says so, UTF-8 when it is, else the console's code page.
fn decoded(bytes: &[u8]) -> String {
    if let Some(units) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return String::from_utf16_lossy(&units.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])).collect::<Vec<_>>());
    }
    if let Ok(text) = std::str::from_utf8(bytes) { return text.to_owned(); }
    let length = unsafe { MultiByteToWideChar(CP_OEMCP, 0, bytes.as_ptr(), bytes.len() as i32, std::ptr::null_mut(), 0) };
    let mut units = vec![0u16; length.max(0) as usize];
    let written = unsafe { MultiByteToWideChar(CP_OEMCP, 0, bytes.as_ptr(), bytes.len() as i32, units.as_mut_ptr(), length) };
    String::from_utf16_lossy(&units[..written.max(0) as usize])
}

/// pnputil's lines, as one: each non-blank one, in order, with its exit code.
fn words(said: &str, code: u32) -> String {
    let lines: Vec<&str> = said.lines().map(str::trim).filter(|line| !line.is_empty()).collect();
    if lines.is_empty() { format!("pnputil exited with code {code}") } else { format!("{} (pnputil exit code {code})", lines.join("; ")) }
}

fn system_folder() -> PathBuf {
    let mut folder = [0u16; 260];
    let length = unsafe { GetSystemDirectoryW(folder.as_mut_ptr(), folder.len() as u32) } as usize;
    PathBuf::from(String::from_utf16_lossy(&folder[..length.min(folder.len())]))
}

/// How an elevated run went: declined at the confirmation, could not start, or its exit code
/// and what it wrote.
enum Elevated { Declined, NotStarted(String), Ended(u32, String), Overran }

/// `cmd.exe` with `arguments`, elevated (Windows asks the operator), hidden; waited for within `wait`.
fn run_elevated(arguments: &str, output: &Path, wait: Duration) -> Elevated {
    let (verb, file, parameters) = (wide("runas"), wide(&system_folder().join("cmd.exe").display().to_string()), wide(arguments));
    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
    info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC;
    (info.lpVerb, info.lpFile, info.lpParameters, info.nShow) = (verb.as_ptr(), file.as_ptr(), parameters.as_ptr(), SW_HIDE);
    // The shell may hand the verb to an extension: COM on this thread first, as ShellExecuteEx asks.
    let com = unsafe { CoInitializeEx(std::ptr::null(), (COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) as u32) };
    let started = unsafe { ShellExecuteExW(&mut info) } != 0;
    let error = unsafe { GetLastError() };
    if com >= 0 { unsafe { CoUninitialize() }; }
    if !started {
        if error == ERROR_CANCELLED { return Elevated::Declined; }
        return Elevated::NotStarted(std::io::Error::from_raw_os_error(error as i32).to_string());
    }
    let process = info.hProcess;
    if process.is_null() { return Elevated::NotStarted("Windows gave no process to wait for".into()); }
    let ended = unsafe { WaitForSingleObject(process, wait.as_millis().min(u128::from(u32::MAX - 1)) as u32) } == WAIT_OBJECT_0;
    let mut code = 0u32;
    if ended { unsafe { GetExitCodeProcess(process, &mut code) }; } else { unsafe { TerminateProcess(process, 1) }; }
    unsafe { CloseHandle(process) };
    if !ended { return Elevated::Overran; }
    Elevated::Ended(code, std::fs::read(output).map(|bytes| decoded(&bytes)).unwrap_or_default())
}

/// Restart the node that owns `port` and wait for the port to be there again (the caller's
/// thread blocks: never the window's).
pub fn restart(port: &str) -> PortRestart {
    let nodes = port_nodes();
    let Some(node) = owner(&nodes, port).map(str::to_owned) else { return PortRestart::Absent };
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|since| since.as_nanos()).unwrap_or_default();
    let output = std::env::temp_dir().join(format!("zlc-port-restart-{}-{stamp}.txt", std::process::id()));
    let Some(arguments) = command_line(&system_folder().join("pnputil.exe"), &node, &output) else {
        return PortRestart::Failed { words: format!("{node} cannot be handed to pnputil on a command line"), node };
    };
    let began = Instant::now();
    let ran = run_elevated(&arguments, &output, RESTART_WAIT);
    let _ = std::fs::remove_file(&output);
    match ran {
        Elevated::Declined => PortRestart::Declined { node },
        Elevated::NotStarted(why) => PortRestart::Failed { words: format!("pnputil could not be started: {why}"), node },
        Elevated::Overran => PortRestart::Failed { words: format!("pnputil did not finish within {} s", RESTART_WAIT.as_secs()), node },
        Elevated::Ended(code, said) if code != 0 => PortRestart::Failed { words: words(&said, code), node },
        Elevated::Ended(..) => {
            // Back once its node names the port again; the open that follows says if it is not.
            while owner(&port_nodes(), port).is_none() && began.elapsed() < RESTART_WAIT { std::thread::sleep(PORT_POLL); }
            PortRestart::Restarted { node }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The node that owns a port is the Ports-class node naming it, in any spelling; a port none
    /// names is absent.
    #[test]
    fn the_node_that_owns_a_port_is_the_one_naming_it() {
        let nodes = [(r"USB\VID_1A86&PID_7523\5&2F0D1A6F&0&3".to_owned(), "COM10".to_owned()),
                     (r"FTDIBUS\VID_0403+PID_6001+A50285BIA\0000".to_owned(), "com4".to_owned()),
                     (r"ACPI\PNP0501\0".to_owned(), "COM1".to_owned())];
        assert_eq!(owner(&nodes, r"\\.\com10"), Some(r"USB\VID_1A86&PID_7523\5&2F0D1A6F&0&3"));
        assert_eq!(owner(&nodes, "COM4"), Some(r"FTDIBUS\VID_0403+PID_6001+A50285BIA\0000"));
        assert_eq!(owner(&nodes, "COM7"), None);
    }

    /// pnputil is run through cmd with its words kept: one quoted line, the id's `&` inside quotes;
    /// a part cmd would expand or end early on is not handed over.
    #[test]
    fn pnputil_runs_on_one_quoted_line_with_its_words_kept() {
        let line = command_line(Path::new(r"C:\Windows\system32\pnputil.exe"), r"USB\VID_1A86&PID_7523\5&2F0D1A6F&0&3", Path::new(r"C:\Temp\out.txt"));
        assert_eq!(line.as_deref(), Some(r#"/d /v:off /s /c ""C:\Windows\system32\pnputil.exe" /restart-device "USB\VID_1A86&PID_7523\5&2F0D1A6F&0&3" > "C:\Temp\out.txt" 2>&1""#));
        assert_eq!(command_line(Path::new("pnputil.exe"), r#"USB\X"Y"#, Path::new("out.txt")), None);
        assert_eq!(command_line(Path::new("pnputil.exe"), r"USB\X", Path::new(r"C:\%TEMP%\out.txt")), None);
        assert_eq!(words("Microsoft PnP Utility\r\n\r\nRestarting device:  USB\\X\r\nFailed to restart device: Access is denied.\r\n", 5),
                   r"Microsoft PnP Utility; Restarting device:  USB\X; Failed to restart device: Access is denied. (pnputil exit code 5)");
        assert_eq!(words("", 1), "pnputil exited with code 1");
        assert_eq!(decoded(&[0xFF, 0xFE, b'o', 0, b'k', 0]), "ok");
        assert_eq!(decoded("设备".as_bytes()), "设备");
    }

    /// The workbench's word for a port its driver refused is the shared serial path's class.
    #[test]
    fn the_refused_class_is_the_serial_paths() {
        assert_eq!(zlc_workbench::REFUSED, zlc_pulse::serial::PortFault::Refused.words());
    }
}
