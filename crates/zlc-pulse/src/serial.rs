//! A Windows COM port as the pulse board's UART link, how the board's port is found, and what
//! an open that fails says.
//!
//! Opened with DTR and RTS low and kept low: opening the pulse streamer's port
//! must never reset or drive another instrument on the bus. Reads wait at most
//! 10 ms per slice so a deadline and a Stop stay responsive.
//!
//! Every serial open of the application words its failure one way ([`OpenFailure`], twinned in
//! Python as `zlc_atom.devices.open_serial_port`): the OS's error class, its code and text, and
//! the serial ports this PC does have. Each failure is also told to the [`FaultWatch`]es held
//! meanwhile: the open or search that holds one carries them to the core (`serial_faults`).
use std::fmt;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    DIGCF_PRESENT, GUID_DEVCLASS_PORTS, SP_DEVINFO_DATA, SPDRP_FRIENDLYNAME, SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInfo,
    SetupDiGetClassDevsW, SetupDiGetDeviceRegistryPropertyW,
};
use windows_sys::Win32::Devices::Communication::{
    COMMTIMEOUTS, COMSTAT, ClearCommError, DCB, EscapeCommFunction, GetCommState, PURGE_RXABORT, PURGE_RXCLEAR, PurgeComm,
    SetCommState, SetCommTimeouts, SetupComm, CLRDTR, CLRRTS, NOPARITY, ONESTOPBIT,
};
use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_ATTRIBUTE_NORMAL, OPEN_EXISTING, ReadFile, WriteFile};
use windows_sys::Win32::System::Registry::{HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_SZ, RegCloseKey, RegEnumValueW, RegOpenKeyExW};

use crate::device::PulseStreamer;
use crate::error::{Error, Kind, Result};
use crate::json::Json;
use crate::model::Target;
use crate::transport::{Event, RegisterTransport, aborted};
use crate::uart::{MAX_FRAME_WORDS, UartLink, UartRegisterTransport, extract_reply, reply_frame_len};
use crate::wire::StreamerParams;

fn wide(text: &str) -> Vec<u16> { text.encode_utf16().chain(std::iter::once(0)).collect() }

fn last_error(what: &str) -> Error { Error::new(Kind::Io, format!("{what}: {}", std::io::Error::last_os_error())) }

/// Why a serial port could not be opened, by the OS's error code: one table for every serial open
/// of the application (Python's twin: `zlc_atom.devices.SERIAL_FAULTS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortFault { Absent, Busy, Refused, Other }

impl PortFault {
    pub fn of(code: u32) -> PortFault {
        match code {
            // ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, ERROR_NO_SUCH_DEVICE, ERROR_DEVICE_NOT_CONNECTED: no such port here.
            2 | 3 | 433 | 1167 => PortFault::Absent,
            // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION: a COM port opens once, and another program holds it.
            5 | 32 => PortFault::Busy,
            // ERROR_INVALID_FUNCTION, ERROR_GEN_FAILURE: the name exists and its driver refused the open.
            1 | 31 => PortFault::Refused,
            _ => PortFault::Other,
        }
    }

    pub fn words(self) -> &'static str {
        match self { PortFault::Absent => "not present", PortFault::Busy => "in use by another program",
                     PortFault::Refused => "driver refused", PortFault::Other => "refused" }
    }

    /// The error type an open failing so is raised as, which the workbench reads: a port that cannot
    /// be used at all is a stale record of where the device was, and Init looks for the device again;
    /// a port another program holds still has the device behind it, and looking elsewhere finds nothing.
    pub fn error_type(self) -> &'static str { if self == PortFault::Busy { "SerialPortBusy" } else { "SerialPortUnusable" } }
}

/// A serial port that could not be opened, with what the OS said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFailure { pub port: String, pub fault: PortFault, pub code: u32, pub detail: String }

impl OpenFailure {
    /// `step` failed on `port` with the calling thread's last OS error.
    fn last(port: &str, step: &str) -> OpenFailure {
        let error = std::io::Error::last_os_error();
        let code = error.raw_os_error().unwrap_or(0) as u32;
        let text = error.to_string();
        let text = text.trim_end_matches(&format!(" (os error {code})")).trim_end_matches(['.', '\u{3002}', ' ']).to_owned();
        let failure = OpenFailure { port: port.to_owned(), fault: PortFault::of(code), code, detail: if step.is_empty() { text } else { format!("{step}: {text}") } };
        noticed(&failure);
        failure
    }

    /// What the core is told of it beside the answer of the open or search that met it
    /// (`serial_faults`, PLUGIN_PROTOCOL §7): `{port, fault, code}`, as Python's `zlc_plugin.faults` says one.
    pub fn fact(&self) -> Json {
        Json::object(vec![("port", Json::str(com_name(&self.port))), ("fault", Json::str(self.fault.words())), ("code", Json::Int(self.code.into()))])
    }

    /// The class alone, as a probe lists it beside the other ports it tried.
    pub fn short(&self) -> String { format!("could not open ({})", self.fault.words()) }
}

/// The one wording of a failed serial open: `COM10 could not be opened: driver refused (os error 1:
/// Incorrect function) -- serial ports on this PC: COM3 (USB-SERIAL CH340)`. What precedes ` -- ` is
/// the failure; what follows, the ports there are to choose from.
impl fmt::Display for OpenFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} could not be opened: {} (os error {}: {}) -- {}", self.port, self.fault.words(), self.code, self.detail, ports_here())
    }
}

impl From<OpenFailure> for Error { fn from(failure: OpenFailure) -> Error { Error::new(Kind::Io, failure.to_string()) } }

/// The watches held now, each told every serial open that fails in this process meanwhile.
static WATCHES: Mutex<Vec<Weak<Mutex<Vec<OpenFailure>>>>> = Mutex::new(Vec::new());

/// The serial opens that fail while it is held, on any thread (a search's probe thread too):
/// held around one open or search, whose answer then carries them to the core (Python's twin:
/// `zlc_plugin.faults.watching_serial_faults`).
pub struct FaultWatch(Arc<Mutex<Vec<OpenFailure>>>);

impl FaultWatch {
    pub fn start() -> FaultWatch {
        let met = Arc::new(Mutex::new(Vec::new()));
        let mut watches = WATCHES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        watches.retain(|watch| watch.strong_count() > 0);
        watches.push(Arc::downgrade(&met));
        FaultWatch(met)
    }

    /// Each failed open met so far, in order.
    pub fn met(&self) -> Vec<OpenFailure> { self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone() }
}

fn noticed(failure: &OpenFailure) {
    let watches: Vec<_> = WATCHES.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).iter().filter_map(Weak::upgrade).collect();
    for met in watches { met.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(failure.clone()); }
}

/// A failed open another process met for this one (the local pulse server's `serial_faults`):
/// told to the watches held here as one met here is.
pub fn told(port: &str, code: u32) { noticed(&OpenFailure { port: port.to_owned(), fault: PortFault::of(code), code, detail: String::new() }) }

/// The serial ports this PC has, with their friendly names where Windows has them:
/// `serial ports on this PC: COM3 (USB-SERIAL CH340), COM4`, or that it has none.
pub fn ports_here() -> String {
    // In name order, as the Python twin lists them (a probe goes USB first; a listing is for reading).
    let mut ports = list_com_ports();
    ports.sort();
    if ports.is_empty() { return "this PC has no serial port".into(); }
    let names = friendly_names();
    let named: Vec<String> = ports.iter().map(|port| match names.iter().find(|(name, _)| com_name(name) == com_name(port)) {
        Some((_, friendly)) => format!("{port} ({friendly})"),
        None => port.clone(),
    }).collect();
    format!("serial ports on this PC: {}", named.join(", "))
}

/// The present Ports-class devices' friendly names by their port (`USB-SERIAL CH340 (COM3)` is
/// COM3's `USB-SERIAL CH340`), as Device Manager shows them.
fn friendly_names() -> Vec<(String, String)> {
    let set = unsafe { SetupDiGetClassDevsW(&GUID_DEVCLASS_PORTS, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT) };
    if set == INVALID_HANDLE_VALUE as isize { return Vec::new(); }
    let mut names = Vec::new();
    for index in 0.. {
        let mut device: SP_DEVINFO_DATA = unsafe { std::mem::zeroed() };
        device.cbSize = std::mem::size_of::<SP_DEVINFO_DATA>() as u32;
        if unsafe { SetupDiEnumDeviceInfo(set, index, &mut device) } == 0 { break; }
        let mut buffer = [0u16; 256];
        let ok = unsafe { SetupDiGetDeviceRegistryPropertyW(set, &device, SPDRP_FRIENDLYNAME, std::ptr::null_mut(), buffer.as_mut_ptr().cast(),
                                                             (buffer.len() * 2) as u32, std::ptr::null_mut()) };
        if ok == 0 { continue; }
        let text = String::from_utf16_lossy(&buffer[..buffer.iter().position(|unit| *unit == 0).unwrap_or(buffer.len())]);
        if let Some((friendly, port)) = text.trim_end().strip_suffix(')').and_then(|rest| rest.rsplit_once(" (")) {
            names.push((port.to_owned(), friendly.trim().to_owned()));
        }
    }
    unsafe { SetupDiDestroyDeviceInfoList(set) };
    names
}

pub struct ComPortLink { port: String, baud: u64, handle: HANDLE, last_shortfall: String }

// The handle is used by one thread at a time (the transport holds its lock).
unsafe impl Send for ComPortLink {}

impl ComPortLink {
    pub fn new(port: &str, baud: u64) -> Result<ComPortLink> {
        if port.trim().is_empty() { return Err(Error::new(Kind::Value, "UART port is required")); }
        if baud == 0 { return Err(Error::new(Kind::Value, "UART baud must be a positive integer")); }
        Ok(ComPortLink { port: port.trim().to_owned(), baud, handle: INVALID_HANDLE_VALUE, last_shortfall: String::new() })
    }

    fn timeouts(&self, read_ms: u32, write_ms: u32) -> Result<()> {
        let timeouts = COMMTIMEOUTS { ReadIntervalTimeout: u32::MAX, ReadTotalTimeoutMultiplier: u32::MAX,
                                      ReadTotalTimeoutConstant: read_ms.max(1), WriteTotalTimeoutMultiplier: 0, WriteTotalTimeoutConstant: write_ms.max(1) };
        if unsafe { SetCommTimeouts(self.handle, &timeouts) } == 0 { return Err(last_error("Cannot set UART timeouts")); }
        Ok(())
    }

    fn queued(&self) -> usize {
        let mut errors = 0u32;
        let mut status: COMSTAT = unsafe { std::mem::zeroed() };
        if unsafe { ClearCommError(self.handle, &mut errors, &mut status) } == 0 { return 0; }
        status.cbInQue as usize
    }

    fn write_all(&self, payload: &[u8], deadline: Instant) -> Result<()> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err(Error::new(Kind::Timeout, "UART write exceeded its deadline")); }
        self.timeouts(10, remaining.as_millis().min(u128::from(u32::MAX)) as u32)?;
        let mut written = 0u32;
        let ok = unsafe { WriteFile(self.handle, payload.as_ptr(), payload.len() as u32, &mut written, std::ptr::null_mut()) };
        if ok == 0 || written as usize != payload.len() {
            return Err(Error::new(Kind::Timeout, format!("UART write timed out on {} at {} baud: {written} of {} byte(s) accepted",
                                                         self.port, self.baud, payload.len())));
        }
        Ok(())
    }
}

impl Drop for ComPortLink {
    fn drop(&mut self) { let _ = self.close(); }
}

impl ComPortLink {
    /// Open the port as [`UartLink::open`] does, saying why it could not be in the shared wording.
    pub fn open_port(&mut self) -> std::result::Result<(), OpenFailure> {
        if self.handle != INVALID_HANDLE_VALUE { return Ok(()); }
        let path = wide(&format!(r"\\.\{}", self.port));
        let handle = unsafe { CreateFileW(path.as_ptr(), GENERIC_READ | GENERIC_WRITE, 0, std::ptr::null(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, std::ptr::null_mut()) };
        if handle == INVALID_HANDLE_VALUE { return Err(OpenFailure::last(&self.port, "")); }
        self.handle = handle;
        let configured = (|| -> std::result::Result<(), OpenFailure> {
            unsafe { SetupComm(handle, 1 << 16, 1 << 16) };
            let mut dcb: DCB = unsafe { std::mem::zeroed() };
            dcb.DCBlength = std::mem::size_of::<DCB>() as u32;
            if unsafe { GetCommState(handle, &mut dcb) } == 0 { return Err(OpenFailure::last(&self.port, "GetCommState")); }
            dcb.BaudRate = self.baud as u32;
            dcb.ByteSize = 8;
            dcb.Parity = NOPARITY;
            dcb.StopBits = ONESTOPBIT;
            // fBinary only: no parity check, no flow control, DTR and RTS disabled (low).
            dcb._bitfield = 1;
            if unsafe { SetCommState(handle, &dcb) } == 0 { return Err(OpenFailure::last(&self.port, "SetCommState")); }
            unsafe { EscapeCommFunction(handle, CLRDTR); EscapeCommFunction(handle, CLRRTS); }
            self.timeouts(50, 1000).map_err(|_| OpenFailure::last(&self.port, "SetCommTimeouts"))
        })();
        if let Err(failure) = configured { let _ = self.close(); return Err(failure); }
        Ok(())
    }
}

impl UartLink for ComPortLink {
    fn open(&mut self) -> Result<()> { Ok(self.open_port()?) }

    /// Release the port; the handle is forgotten only once it is released.
    fn close(&mut self) -> Result<()> {
        if self.handle == INVALID_HANDLE_VALUE { return Ok(()); }
        if unsafe { CloseHandle(self.handle) } == 0 { return Err(last_error(&format!("could not close port {}", self.port))); }
        self.handle = INVALID_HANDLE_VALUE;
        Ok(())
    }

    fn port(&self) -> String { self.port.clone() }
    fn baud(&self) -> u64 { self.baud }
    fn last_shortfall(&self) -> String { self.last_shortfall.clone() }

    fn write_batch(&mut self, requests: &[Vec<u8>], deadline: Instant, stop: Option<&Event>) -> Result<Vec<Vec<u8>>> {
        if stop.is_some_and(Event::is_set) { return Err(aborted("UART request")); }
        if self.handle == INVALID_HANDLE_VALUE { return Err(Error::new(Kind::Runtime, "serial link is not open")); }
        if requests.is_empty() { return Ok(Vec::new()); }
        unsafe { PurgeComm(self.handle, PURGE_RXCLEAR | PURGE_RXABORT) };
        self.last_shortfall.clear();
        let payload: Vec<u8> = requests.iter().enumerate().flat_map(|(index, request)| {
            let gap: &[u8] = if index == 0 { &[] } else { &[0xFF; 8] };
            gap.iter().chain(request.iter()).copied().collect::<Vec<_>>()
        }).collect();
        self.write_all(&payload, deadline)?;
        let mut pending: std::collections::BTreeSet<u8> = requests.iter().map(|request| request[3]).collect();
        let mut buffer = Vec::new();
        let mut replies = Vec::new();
        let mut read_bytes = 0usize;
        let mut chunk = vec![0u8; reply_frame_len(MAX_FRAME_WORDS)];
        while !pending.is_empty() {
            let now = Instant::now();
            if now >= deadline { break; }
            if stop.is_some_and(Event::is_set) { return Err(aborted("UART read")); }
            let slice = (deadline - now).min(Duration::from_millis(10));
            self.timeouts(slice.as_millis().max(1) as u32, 1000)?;
            let wanted = self.queued().clamp(1, chunk.len());
            let mut got = 0u32;
            if unsafe { ReadFile(self.handle, chunk.as_mut_ptr(), wanted as u32, &mut got, std::ptr::null_mut()) } == 0 {
                return Err(last_error(&format!("UART read on {}", self.port)));
            }
            if got == 0 { continue; }
            read_bytes += got as usize;
            buffer.extend_from_slice(&chunk[..got as usize]);
            while !pending.is_empty() {
                let Some(frame) = extract_reply(&mut buffer) else { break };
                if pending.remove(&frame[3]) { replies.push(frame); }
            }
        }
        if !pending.is_empty() {
            self.last_shortfall = if read_bytes == 0 { "no bytes".into() }
                else { format!("{} of {} replies, {} byte(s) left unframed", replies.len(), requests.len(), buffer.len()) };
        }
        Ok(replies)
    }
}

/// The COM ports this machine lists (`HKLM\HARDWARE\DEVICEMAP\SERIALCOMM`),
/// USB adapters first: a board's UART is a USB bridge.
pub fn list_com_ports() -> Vec<String> {
    let mut ports: Vec<(bool, String)> = Vec::new();
    let key_path = wide(r"HARDWARE\DEVICEMAP\SERIALCOMM");
    let mut key: HKEY = std::ptr::null_mut();
    if unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, key_path.as_ptr(), 0, KEY_READ, &mut key) } != 0 { return Vec::new(); }
    for index in 0.. {
        let mut name = [0u16; 256];
        let mut name_length = name.len() as u32;
        let mut data = [0u8; 256];
        let mut data_length = data.len() as u32;
        let mut kind = 0u32;
        let status = unsafe { RegEnumValueW(key, index, name.as_mut_ptr(), &mut name_length, std::ptr::null_mut(), &mut kind,
                                            data.as_mut_ptr(), &mut data_length) };
        if status != 0 { break; }
        if kind != REG_SZ { continue; }
        let device = String::from_utf16_lossy(&name[..name_length as usize]);
        let units: Vec<u16> = data[..data_length as usize].chunks(2).map(|pair| u16::from_le_bytes([pair[0], *pair.get(1).unwrap_or(&0)])).collect();
        let port = String::from_utf16_lossy(&units).trim_end_matches('\0').to_owned();
        let upper = device.to_ascii_uppercase();
        let usb = ["USB", "VCP", "CH34", "SILAB", "PROLIFIC", "FTDI", "CP210"].iter().any(|token| upper.contains(token));
        if !port.is_empty() && !ports.iter().any(|(_, existing)| *existing == port) { ports.push((usb, port)); }
    }
    unsafe { RegCloseKey(key) };
    ports.sort_by_key(|(usb, _)| !*usb);
    ports.into_iter().map(|(_, port)| port).collect()
}

/// A COM port as one name whatever its spelling: upper-case, without `\\.\` (`\\.\com3` is `COM3`).
pub fn com_name(port: &str) -> String { port.trim().trim_start_matches(r"\\.\").to_uppercase() }

/// `ports` in their order, less those `excluded` names (any spelling): (the free ones, the ones passed over).
pub fn free_ports(ports: &[String], excluded: &[String]) -> (Vec<String>, Vec<String>) {
    let excluded: Vec<String> = excluded.iter().map(|port| com_name(port)).collect();
    ports.iter().cloned().partition(|port| !excluded.contains(&com_name(port)))
}

/// What a probe of one port came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probe {
    /// The board answered with this build's word-63 fingerprint.
    Matched,
    /// The port could not be opened at all; nothing was asked.
    NotOpened(OpenFailure),
    /// The port opened and no board of this build answered (`timeout`, `fingerprint mismatch`, ...).
    NoBoard(String),
    /// Nothing to open: the port or the baud is not one (`UART port is required`).
    Invalid(String),
}

/// One port a probe tried, as the lines say it: `COM10: could not open (driver refused)`,
/// `COM4: opened, no board answered (timeout)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt { pub port: String, pub probe: Probe }

impl fmt::Display for Attempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.probe {
            Probe::Matched => write!(f, "{}: word63 fingerprint matched", self.port),
            Probe::NotOpened(failure) => write!(f, "{}: {}", self.port, failure.short()),
            Probe::NoBoard(why) => write!(f, "{}: opened, no board answered ({why})", self.port),
            Probe::Invalid(why) => write!(f, "{}: {why}", self.port),
        }
    }
}

/// The board's UART among `ports`, found as v2's server found it: each port opened with the board's
/// handshake (open and the word-63 fingerprint, 0.5 s) and released again, all of them at once
/// ([`search_ports`]). The first in `ports`' order that answers, and every port tried with what it came to.
pub fn find_board_uart(ports: &[String], baud: u64, geometry: StreamerParams, clock_hz: f64, target: &Arc<Target>) -> (Option<String>, Vec<Attempt>) {
    search_ports(ports, |port| match ComPortLink::new(port, baud) {
        Err(error) => Probe::Invalid(error.message),
        Ok(mut link) => match link.open_port() {
            Err(failure) => Probe::NotOpened(failure),
            Ok(()) => {
                let transport: Arc<dyn RegisterTransport> = Arc::new(UartRegisterTransport::new(Box::new(link), Duration::from_millis(500)));
                let opened = PulseStreamer::new(transport.clone(), geometry, clock_hz, target.clone()).and_then(|streamer| streamer.open());
                let _ = transport.close();
                match opened {
                    Ok(()) => Probe::Matched,
                    Err(error) if error.message.contains("layout mismatch") => Probe::NoBoard("fingerprint mismatch".into()),
                    Err(error) if error.kind == Kind::Timeout => Probe::NoBoard("timeout".into()),
                    Err(error) => Probe::NoBoard(error.message),
                }
            }
        },
    })
}

/// One search over `ports`: each probed once, all at once on threads of their own, so the search
/// takes about one port's probe however many there are. It returns once every probe has released
/// its port, with the attempts in `ports`' order and, of the ports that answered, the first in that order.
pub fn search_ports(ports: &[String], probe: impl Fn(&str) -> Probe + Sync) -> (Option<String>, Vec<Attempt>) {
    let probe = &probe;
    let attempts: Vec<Attempt> = std::thread::scope(|scope| {
        let probes: Vec<_> = ports.iter().map(|port| (port, scope.spawn(move || probe(port)))).collect();
        probes.into_iter().map(|(port, probing)| Attempt {
            port: port.clone(),
            probe: probing.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
        }).collect()
    });
    (attempts.iter().find(|attempt| attempt.probe == Probe::Matched).map(|attempt| attempt.port.clone()), attempts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bound port is one name however it is spelled, and the probe never reaches it.
    #[test]
    fn a_probe_passes_over_the_bound_ports_in_any_spelling() {
        assert_eq!(com_name(r" \\.\com3 "), "COM3");
        assert_eq!(com_name("COM12"), "COM12");
        let listed: Vec<String> = ["COM5", "com3", r"\\.\COM4", "COM12"].map(str::to_owned).to_vec();
        let (free, passed) = free_ports(&listed, &[r"\\.\COM3".to_owned(), "com4".to_owned(), "COM1".to_owned()]);
        assert_eq!((free, passed), (vec!["COM5".to_owned(), "COM12".to_owned()], vec!["com3".to_owned(), r"\\.\COM4".to_owned()]));
        assert_eq!(free_ports(&listed, &[]).0, listed, "nothing bound: every port, in its order");
        let lane = ["d0".to_owned()];
        let target = Target::new(&lane, vec![crate::model::PortSpec::new("probe", "digital", &lane, "", None, None, None, None, None).unwrap()], Vec::new()).unwrap();
        assert_eq!(find_board_uart(&[], 115_200, StreamerParams::default(), 1.0e8, &target), (None, Vec::new()), "no free port: nothing opened, nothing tried");
    }

    /// A port that is not there is said with its class and the ports this PC does have, and a probe
    /// tells it apart from a port that opened with no board behind it.
    #[test]
    fn a_failed_open_names_its_class_and_the_ports_here() {
        let absent = "COM250";
        let mut link = ComPortLink::new(absent, 115_200).unwrap();
        let failure = link.open_port().unwrap_err();
        assert_eq!((failure.fault, failure.code, failure.fault.error_type()), (PortFault::Absent, 2, "SerialPortUnusable"));
        let said = failure.to_string();
        let (failed, listed) = said.split_once(" -- ").unwrap();
        assert!(failed.starts_with("COM250 could not be opened: not present (os error 2: "), "{said}");
        assert_eq!(listed, ports_here());
        assert!(listed == "this PC has no serial port" || listed.starts_with("serial ports on this PC: "), "{listed}");
        assert_eq!(UartLink::open(&mut link).unwrap_err().message, link.open_port().unwrap_err().to_string());
        let lane = ["d0".to_owned()];
        let target = Target::new(&lane, vec![crate::model::PortSpec::new("probe", "digital", &lane, "", None, None, None, None, None).unwrap()], Vec::new()).unwrap();
        let (found, attempts) = find_board_uart(&[absent.to_owned()], 115_200, StreamerParams::default(), 1.0e8, &target);
        assert_eq!((found, attempts.len(), attempts[0].to_string()), (None, 1, "COM250: could not open (not present)".to_owned()));
        assert_eq!(Attempt { port: "COM4".into(), probe: Probe::NoBoard("timeout".into()) }.to_string(), "COM4: opened, no board answered (timeout)");
        // One table with Python's (zlc_atom.devices.SERIAL_FAULTS): the OS codes by class.
        let classes: Vec<(u32, &str)> = [2, 3, 433, 1167, 5, 32, 1, 31, 87].iter().map(|code| (*code, PortFault::of(*code).words())).collect();
        assert_eq!(classes, [(2, "not present"), (3, "not present"), (433, "not present"), (1167, "not present"), (5, "in use by another program"),
                             (32, "in use by another program"), (1, "driver refused"), (31, "driver refused"), (87, "refused")]);
        assert_eq!((PortFault::Busy.error_type(), PortFault::Refused.error_type()), ("SerialPortBusy", "SerialPortUnusable"));
    }

    /// One search probes its ports at once: it takes about one port's probe, not their sum, tries
    /// each port once, reports the attempts in the listing's order, and of two that answer takes the
    /// first listed -- even when a later one answers sooner.
    #[test]
    fn a_search_probes_its_ports_at_once_and_reports_them_in_order() {
        let listed: Vec<String> = ["COM13", "COM7", "COM8", "COM10", "COM11", "COM12"].map(str::to_owned).to_vec();
        let timeout = Duration::from_millis(400);
        let asked = std::sync::Mutex::new(Vec::new());
        let began = Instant::now();
        let (found, attempts) = search_ports(&listed, |port| {
            asked.lock().unwrap().push(port.to_owned());
            match port {
                // The later-listed board answers at once; the earlier one only after a while.
                "COM12" => Probe::Matched,
                "COM10" => { std::thread::sleep(timeout / 2); Probe::Matched }
                "COM7" => Probe::NotOpened(OpenFailure { port: port.into(), fault: PortFault::Refused, code: 1, detail: "Incorrect function".into() }),
                _ => { std::thread::sleep(timeout); Probe::NoBoard("timeout".into()) }
            }
        });
        let took = began.elapsed();
        assert!(took >= timeout && took < timeout * 2, "about one port's timeout, not the sum: {took:?}");
        assert_eq!(found.as_deref(), Some("COM10"), "the first listed of those that answer");
        assert_eq!(attempts.iter().map(|attempt| attempt.port.as_str()).collect::<Vec<_>>(), listed, "in the listing's order");
        assert_eq!(attempts.iter().map(ToString::to_string).collect::<Vec<_>>(),
                   ["COM13: opened, no board answered (timeout)", "COM7: could not open (driver refused)", "COM8: opened, no board answered (timeout)",
                    "COM10: word63 fingerprint matched", "COM11: opened, no board answered (timeout)", "COM12: word63 fingerprint matched"]);
        let mut asked = asked.into_inner().unwrap();
        asked.sort();
        let mut each = listed.clone();
        each.sort();
        assert_eq!(asked, each, "each port probed once");
    }

    /// A failed open is told to every watch held meanwhile, whichever thread met it, and to none
    /// started after it; one another process met is told as one met here, classed by its code.
    #[test]
    fn a_watch_hears_every_failed_open_while_it_is_held() {
        // Ports no other test opens: tests share the process, and a watch hears them all.
        let ours = |watch: &FaultWatch| watch.met().iter().map(OpenFailure::fact).map(|fact| fact.compact_text())
            .filter(|fact| fact.contains("COM251") || fact.contains("COM252")).collect::<Vec<_>>();
        let watch = FaultWatch::start();
        std::thread::spawn(|| ComPortLink::new("com251", 115_200).unwrap().open_port().unwrap_err()).join().unwrap();
        let later = FaultWatch::start();
        told("COM252", 31);
        assert_eq!(ours(&watch), [r#"{"port":"COM251","fault":"not present","code":2}"#, r#"{"port":"COM252","fault":"driver refused","code":31}"#]);
        assert_eq!(ours(&later), [r#"{"port":"COM252","fault":"driver refused","code":31}"#]);
        drop(watch);
        told("COM252", 1);
        assert_eq!(ours(&later).len(), 2, "the watches still held hear on");
    }
}
