//! A Windows COM port as the pulse board's UART link.
//!
//! Opened with DTR and RTS low and kept low: opening the pulse streamer's port
//! must never reset or drive another instrument on the bus. Reads wait at most
//! 10 ms per slice so a deadline and a Stop stay responsive.
use std::time::{Duration, Instant};

use windows_sys::Win32::Devices::Communication::{
    COMMTIMEOUTS, COMSTAT, ClearCommError, DCB, EscapeCommFunction, GetCommState, PURGE_RXABORT, PURGE_RXCLEAR, PurgeComm,
    SetCommState, SetCommTimeouts, SetupComm, CLRDTR, CLRRTS, NOPARITY, ONESTOPBIT,
};
use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_ATTRIBUTE_NORMAL, OPEN_EXISTING, ReadFile, WriteFile};
use windows_sys::Win32::System::Registry::{HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_SZ, RegCloseKey, RegEnumValueW, RegOpenKeyExW};

use crate::error::{Error, Kind, Result};
use crate::transport::{Event, aborted};
use crate::uart::{MAX_FRAME_WORDS, UartLink, extract_reply, reply_frame_len};

fn wide(text: &str) -> Vec<u16> { text.encode_utf16().chain(std::iter::once(0)).collect() }

fn last_error(what: &str) -> Error { Error::new(Kind::Io, format!("{what}: {}", std::io::Error::last_os_error())) }

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

impl UartLink for ComPortLink {
    fn open(&mut self) -> Result<()> {
        if self.handle != INVALID_HANDLE_VALUE { return Ok(()); }
        let path = wide(&format!(r"\\.\{}", self.port));
        let handle = unsafe { CreateFileW(path.as_ptr(), GENERIC_READ | GENERIC_WRITE, 0, std::ptr::null(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, std::ptr::null_mut()) };
        if handle == INVALID_HANDLE_VALUE { return Err(last_error(&format!("could not open port {}", self.port))); }
        self.handle = handle;
        let configured = (|| -> Result<()> {
            unsafe { SetupComm(handle, 1 << 16, 1 << 16) };
            let mut dcb: DCB = unsafe { std::mem::zeroed() };
            dcb.DCBlength = std::mem::size_of::<DCB>() as u32;
            if unsafe { GetCommState(handle, &mut dcb) } == 0 { return Err(last_error("GetCommState")); }
            dcb.BaudRate = self.baud as u32;
            dcb.ByteSize = 8;
            dcb.Parity = NOPARITY;
            dcb.StopBits = ONESTOPBIT;
            // fBinary only: no parity check, no flow control, DTR and RTS disabled (low).
            dcb._bitfield = 1;
            if unsafe { SetCommState(handle, &dcb) } == 0 { return Err(last_error("SetCommState")); }
            unsafe { EscapeCommFunction(handle, CLRDTR); EscapeCommFunction(handle, CLRRTS); }
            self.timeouts(50, 1000)
        })();
        if let Err(error) = configured { let _ = self.close(); return Err(error); }
        Ok(())
    }

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
