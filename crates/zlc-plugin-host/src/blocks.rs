//! The core's block registry (PLUGIN_PROTOCOL.md §6).
//!
//! Every shared-memory block that crosses the core is counted here: each
//! `Lend` the core keeps, and each lend a plugin received through the core,
//! is one hold. When the last hold goes, the block's owner is told it may
//! write the block again. The core maps a block read-only once and borrows
//! it into Runtime as an `OwnedBuffer` without copying.
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::{Arc, Mutex};

use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile,
    MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, OpenFileMappingW, PAGE_READWRITE, UnmapViewOfFile, VirtualQuery};
use zlc_core::ForeignBytes;

pub const GRANULARITY: usize = 64 << 10;

/// One mapped view of a named block; unmapped when the last user drops it.
pub(crate) struct Mapping { address: usize, size: usize, handle: usize }

fn wide(name: &str) -> Vec<u16> { name.encode_utf16().chain([0]).collect() }

impl Mapping {
    fn view(handle: HANDLE, access: u32, name: &str) -> Result<Self, String> {
        let view = unsafe { MapViewOfFile(handle, access, 0, 0, 0) };
        if view.Value.is_null() {
            let error = io::Error::last_os_error();
            unsafe { CloseHandle(handle) };
            return Err(format!("Cannot map block {name}: {error}"));
        }
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        unsafe { VirtualQuery(view.Value, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
        Ok(Self { address: view.Value as usize, size: info.RegionSize, handle: handle as usize })
    }

    fn open(name: &str) -> Result<Self, String> {
        let handle = unsafe { OpenFileMappingW(FILE_MAP_READ, 0, wide(name).as_ptr()) };
        if handle.is_null() { return Err(format!("Cannot open block {name}: {}", io::Error::last_os_error())); }
        Self::view(handle, FILE_MAP_READ, name)
    }

    fn create(name: &str, size: usize) -> Result<Self, String> {
        let handle = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE,
            (size as u64 >> 32) as u32, size as u32, wide(name).as_ptr()) };
        if handle.is_null() { return Err(format!("Cannot create block {name}: {}", io::Error::last_os_error())); }
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            unsafe { CloseHandle(handle) };
            return Err(format!("Block {name} already exists"));
        }
        Self::view(handle, FILE_MAP_WRITE, name)
    }

    fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.address as *const u8, self.size) } }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.address as *mut _ });
            CloseHandle(self.handle as HANDLE);
        }
    }
}

/// `holds` counts every holder; `lent` the owner's own lends since its last release.
struct Entry { owner: u32, mapping: Arc<Mapping>, holds: usize, lent: usize }

#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    held: HashMap<(u32, String), usize>,
    free: HashMap<usize, Vec<String>>,
    gone: HashSet<u32>,
    next: u64,
}

/// Tell `owner` (a plugin pid) that it may reuse these blocks.
pub(crate) type Release = Box<dyn Fn(u32, Vec<String>) + Send + Sync>;

pub struct Registry { session: String, me: u32, state: Mutex<State>, release: Release }

/// One hold of a block by the core; its bytes back zero-copy `OwnedBuffer`s.
pub struct Lend { registry: Arc<Registry>, name: String, mapping: Arc<Mapping> }

// The owner writes a block only after the registry released every hold of
// it, and this Lend is one of those holds; the view stays mapped while it lives.
unsafe impl ForeignBytes for Lend {
    fn bytes(&self) -> &[u8] { self.mapping.bytes() }
    fn as_any(&self) -> &dyn Any { self }
}

impl Lend {
    pub fn name(&self) -> &str { &self.name }
}

impl Drop for Lend {
    fn drop(&mut self) { self.registry.unhold(&self.name, None); }
}

impl Registry {
    pub(crate) fn new(session: String, release: Release) -> Arc<Self> {
        Arc::new(Self { session, me: std::process::id(), state: Mutex::new(State::default()), release })
    }

    /// A registry of core blocks only (no plugin to tell): for tools and tests.
    pub fn standalone(session: &str) -> Arc<Self> { Self::new(session.to_owned(), Box::new(|_, _| {})) }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> { self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

    /// The owning pid named in a block of this session.
    pub fn owner_of(&self, name: &str) -> Result<u32, String> {
        let rest = name.strip_prefix("Local\\zlc.").and_then(|rest| rest.strip_prefix(self.session.as_str()))
            .and_then(|rest| rest.strip_prefix('.')).ok_or_else(|| format!("{name} is not a block of this session"))?;
        let (pid, counter) = rest.split_once('.').ok_or_else(|| format!("{name} has no block counter"))?;
        counter.parse::<u64>().map_err(|_| format!("{name} has no block counter"))?;
        pid.parse().map_err(|_| format!("{name} names no owner process"))
    }

    /// A message from plugin `sender` lent `name` to the core: one more hold.
    /// The owner counts every lend it sends, so it is released once per lend.
    pub fn lend(self: &Arc<Self>, name: &str, sender: u32) -> Result<Lend, String> {
        let owner = self.owner_of(name)?;
        let mut state = self.lock();
        let mapping = match state.entries.get_mut(name) {
            Some(entry) => {
                entry.holds += 1;
                entry.lent += usize::from(sender == owner);
                entry.mapping.clone()
            }
            None => {
                let mapping = Arc::new(Mapping::open(name)?);
                state.entries.insert(name.to_owned(), Entry { owner, mapping: mapping.clone(), holds: 1,
                    lent: usize::from(sender == owner) });
                mapping
            }
        };
        Ok(Lend { registry: self.clone(), name: name.to_owned(), mapping })
    }

    /// The core lends a block it holds on to plugin `holder`.
    pub fn hold_for(&self, holder: u32, name: &str) -> Result<(), String> {
        let mut state = self.lock();
        let entry = state.entries.get_mut(name).ok_or_else(|| format!("{name} is not a registered block"))?;
        entry.holds += 1;
        *state.held.entry((holder, name.to_owned())).or_default() += 1;
        Ok(())
    }

    /// Plugin `holder` released lends it received.
    pub fn released_by(&self, holder: u32, names: &[String]) {
        for name in names { self.unhold(name, Some(holder)); }
    }

    /// A plugin process ended: its holds go, and its own blocks are never reused.
    pub fn holder_gone(&self, holder: u32) {
        let names: Vec<(String, usize)> = {
            let mut state = self.lock();
            state.gone.insert(holder);
            let held: Vec<_> = state.held.keys().filter(|(pid, _)| *pid == holder).cloned().collect();
            held.into_iter().filter_map(|key| state.held.remove(&key).map(|count| (key.1, count))).collect()
        };
        for (name, count) in names {
            for _ in 0..count { self.unhold(&name, None); }
        }
        let mut state = self.lock();
        state.entries.retain(|_, entry| entry.owner != holder || entry.holds > 0);
    }

    fn unhold(&self, name: &str, holder: Option<u32>) {
        let notice = {
            let mut state = self.lock();
            if let Some(pid) = holder {
                let key = (pid, name.to_owned());
                match state.held.get_mut(&key) {
                    Some(count) if *count > 1 => *count -= 1,
                    Some(_) => { state.held.remove(&key); }
                    None => return,
                }
            }
            let Some(entry) = state.entries.get_mut(name) else { return };
            entry.holds = entry.holds.saturating_sub(1);
            if entry.holds > 0 { return; }
            let (owner, lent) = (entry.owner, std::mem::take(&mut entry.lent));
            if owner == self.me {
                let size = entry.mapping.size;
                state.free.entry(size).or_default().push(name.to_owned());
                None
            } else if state.gone.contains(&owner) {
                state.entries.remove(name);
                None
            } else {
                (lent > 0).then_some((owner, lent))
            }
        };
        if let Some((owner, lent)) = notice { (self.release)(owner, vec![name.to_owned(); lent]); }
    }

    /// Copy core-owned bytes into a core block, held once by the returned Lend.
    pub fn export(self: &Arc<Self>, bytes: &[u8]) -> Result<Lend, String> {
        self.export_with(bytes.len(), |block| { block.copy_from_slice(bytes); Ok(()) })
    }

    /// A core block of at least `len` bytes whose first `len` bytes `fill`
    /// writes (a network read straight into it), held once by the returned Lend.
    pub fn export_with(self: &Arc<Self>, len: usize, fill: impl FnOnce(&mut [u8]) -> Result<(), String>) -> Result<Lend, String> {
        let size = len.max(1).div_ceil(GRANULARITY) * GRANULARITY;
        let (name, mapping) = {
            let mut state = self.lock();
            match state.free.get_mut(&size).and_then(Vec::pop) {
                Some(name) => { let mapping = state.entries[&name].mapping.clone(); (name, mapping) }
                None => {
                    state.next += 1;
                    let name = format!("Local\\zlc.{}.{}.{}", self.session, self.me, state.next);
                    let mapping = Arc::new(Mapping::create(&name, size)?);
                    state.entries.insert(name.clone(), Entry { owner: self.me, mapping: mapping.clone(), holds: 0, lent: 0 });
                    (name, mapping)
                }
            }
        };
        // Out of the free list with no hold: nobody else reads or reuses the block
        // while it is written, and the registry is not locked meanwhile.
        let block = unsafe { std::slice::from_raw_parts_mut(mapping.address as *mut u8, len) };
        if let Err(error) = fill(block) {
            self.lock().free.entry(size).or_default().push(name);
            return Err(error);
        }
        self.lock().entries.get_mut(&name).expect("reserved block").holds = 1;
        Ok(Lend { registry: self.clone(), name, mapping })
    }

    /// Outstanding holds of a block (diagnostics and tests).
    pub fn holds(&self, name: &str) -> usize { self.lock().entries.get(name).map_or(0, |entry| entry.holds) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use zlc_core::OwnedBuffer;

    #[test]
    fn lends_borrow_without_copying_and_the_last_hold_releases_to_the_owner() {
        let (sender, released) = mpsc::channel();
        let sender = Mutex::new(sender);
        let registry = Registry::new("test".into(), Box::new(move |owner, names| { sender.lock().unwrap().send((owner, names)).unwrap(); }));
        // A core block stands in for a plugin's: its name says the owner.
        let exported = registry.export(&[7u8; 300]).unwrap();
        let name = exported.name().to_owned();
        assert_eq!(registry.owner_of(&name).unwrap(), std::process::id());
        let owner: Arc<dyn ForeignBytes> = Arc::new(exported);
        let values = OwnedBuffer::<u8>::from_foreign(owner.clone(), 0, 300).unwrap();
        assert_eq!(values.as_slice(), &[7u8; 300][..]);
        registry.hold_for(4242, &name).unwrap();
        assert_eq!(registry.holds(&name), 2);
        drop((owner, values));
        assert_eq!(registry.holds(&name), 1);
        registry.released_by(4242, &[name.clone()]);
        assert_eq!(registry.holds(&name), 0);
        // A released core block is reused for the next export of its size.
        let again = registry.export(&[1u8; 10]).unwrap();
        assert_eq!(again.name(), name);
        assert!(released.try_recv().is_err(), "core blocks are recycled, not announced");
        registry.hold_for(77, &name).unwrap();
        registry.holder_gone(77);
        assert_eq!(registry.holds(&name), 1);
        assert!(registry.owner_of("Local\\zlc.other.1.2").is_err());
    }
}
