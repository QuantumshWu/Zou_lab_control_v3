//! ZIP files as CPython 3.13's `zipfile` writes them for an NPZ on Windows
//! (`allowZip64=True`, every member opened with `force_zip64=True`, the 1980
//! timestamp, a seekable stream) and reads them back, with its refusals.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::deflate::Deflater;
use crate::error::{ArchiveError, ErrorKind, Result};
use crate::json::py_repr_str;

const ZIP64_LIMIT: u64 = (1 << 31) - 1;
const ZIP_FILECOUNT_LIMIT: usize = (1 << 16) - 1;
const LOCAL_SIGNATURE: &[u8; 4] = b"PK\x03\x04";
const CENTRAL_SIGNATURE: &[u8; 4] = b"PK\x01\x02";
const END_SIGNATURE: &[u8; 4] = b"PK\x05\x06";
const END64_SIGNATURE: &[u8; 4] = b"PK\x06\x06";
const LOCATOR_SIGNATURE: &[u8; 4] = b"PK\x06\x07";
const LOCAL_SIZE: usize = 30;
const CENTRAL_SIZE: usize = 46;
const END_SIZE: usize = 22;
const END64_SIZE: usize = 56;
const LOCATOR_SIZE: usize = 20;
const MAX_COMMENT: usize = (1 << 16) - 1;
const ZIP64_VERSION: u8 = 45;
const DOS_DATE: u16 = 1 << 5 | 1;
const FLAG_UTF8: u16 = 1 << 11;
const ATTRIBUTES: u32 = 0o600 << 16;
pub const STORED: u16 = 0;
pub const DEFLATED: u16 = 8;

fn encode_name(name: &str) -> (Vec<u8>, u16) {
    if name.is_ascii() { (name.as_bytes().to_vec(), 0) } else { (name.as_bytes().to_vec(), FLAG_UTF8) }
}

/// `ZipInfo.FileHeader(zip64=True)`.
fn local_header(name: &[u8], flags: u16, method: u16, crc: u32, compressed: u64, size: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(LOCAL_SIZE + name.len() + 20);
    out.extend_from_slice(LOCAL_SIGNATURE);
    out.extend_from_slice(&[ZIP64_VERSION, 0]);
    for value in [flags, method, 0, DOS_DATE] { out.extend_from_slice(&value.to_le_bytes()); }
    for value in [crc, u32::MAX, u32::MAX] { out.extend_from_slice(&value.to_le_bytes()); }
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&compressed.to_le_bytes());
    out
}

struct Written { name: Vec<u8>, flags: u16, method: u16, crc: u32, compressed: u64, size: u64, offset: u64 }

/// A `ZipFile(stream, "w", allowZip64=True)` writing members one at a time.
pub struct ZipWriter<W: Write + Seek> { out: W, members: Vec<Written>, start_dir: u64 }

impl<W: Write + Seek> ZipWriter<W> {
    pub fn new(mut out: W) -> std::io::Result<Self> {
        let start_dir = out.stream_position()?;
        Ok(Self { out, members: Vec::new(), start_dir })
    }

    /// One `open(ZipInfo(name), "w", force_zip64=True)` member: `parts` are
    /// the successive `write` calls, compressed per call as zipfile does.
    pub fn write_member(&mut self, name: &str, deflate: bool, parts: &[&[u8]]) -> std::io::Result<()> {
        let (encoded, flags) = encode_name(name);
        let method = if deflate { DEFLATED } else { STORED };
        let offset = self.start_dir;
        self.out.seek(SeekFrom::Start(offset))?;
        self.out.write_all(&local_header(&encoded, flags, method, 0, 0, 0))?;
        let mut hasher = crc32fast::Hasher::new();
        let mut size = 0u64;
        let mut compressed = 0u64;
        let mut deflater = deflate.then(Deflater::new);
        for part in parts {
            hasher.update(part);
            size += part.len() as u64;
            match &mut deflater {
                Some(deflater) => {
                    deflater.compress(part);
                    let bytes = deflater.take_output();
                    compressed += bytes.len() as u64;
                    self.out.write_all(&bytes)?;
                }
                None => { compressed += part.len() as u64; self.out.write_all(part)?; }
            }
        }
        if let Some(deflater) = deflater {
            let bytes = deflater.finish();
            compressed += bytes.len() as u64;
            self.out.write_all(&bytes)?;
        }
        let crc = hasher.finalize();
        self.start_dir = self.out.stream_position()?;
        self.out.seek(SeekFrom::Start(offset))?;
        self.out.write_all(&local_header(&encoded, flags, method, crc, compressed, size))?;
        self.out.seek(SeekFrom::Start(self.start_dir))?;
        self.members.push(Written { name: encoded, flags, method, crc, compressed, size, offset });
        Ok(())
    }

    /// `close()`: the central directory and end records.
    pub fn finish(mut self) -> std::io::Result<W> {
        self.out.seek(SeekFrom::Start(self.start_dir))?;
        let mut directory = Vec::new();
        for member in &self.members {
            let mut extra: Vec<u64> = Vec::new();
            let (mut compressed, mut size) = (member.compressed as u32, member.size as u32);
            if member.size > ZIP64_LIMIT || member.compressed > ZIP64_LIMIT {
                extra.extend([member.size, member.compressed]);
                (compressed, size) = (u32::MAX, u32::MAX);
            }
            let mut offset = member.offset as u32;
            if member.offset > ZIP64_LIMIT { extra.push(member.offset); offset = u32::MAX; }
            let mut extra_data = Vec::new();
            if !extra.is_empty() {
                extra_data.extend_from_slice(&1u16.to_le_bytes());
                extra_data.extend_from_slice(&(8 * extra.len() as u16).to_le_bytes());
                for value in extra { extra_data.extend_from_slice(&value.to_le_bytes()); }
            }
            directory.extend_from_slice(CENTRAL_SIGNATURE);
            directory.extend_from_slice(&[ZIP64_VERSION, 0, ZIP64_VERSION, 0]);
            for value in [member.flags, member.method, 0, DOS_DATE] { directory.extend_from_slice(&value.to_le_bytes()); }
            for value in [member.crc, compressed, size] { directory.extend_from_slice(&value.to_le_bytes()); }
            for value in [member.name.len() as u16, extra_data.len() as u16, 0, 0, 0] { directory.extend_from_slice(&value.to_le_bytes()); }
            directory.extend_from_slice(&ATTRIBUTES.to_le_bytes());
            directory.extend_from_slice(&offset.to_le_bytes());
            directory.extend_from_slice(&member.name);
            directory.extend_from_slice(&extra_data);
        }
        self.out.write_all(&directory)?;
        let end = self.start_dir + directory.len() as u64;
        let (mut count, mut size, mut offset) = (self.members.len() as u64, directory.len() as u64, self.start_dir);
        if self.members.len() > ZIP_FILECOUNT_LIMIT || offset > ZIP64_LIMIT || size > ZIP64_LIMIT {
            let mut record = Vec::with_capacity(END64_SIZE + LOCATOR_SIZE);
            record.extend_from_slice(END64_SIGNATURE);
            record.extend_from_slice(&((END64_SIZE - 12) as u64).to_le_bytes());
            record.extend_from_slice(&45u16.to_le_bytes());
            record.extend_from_slice(&45u16.to_le_bytes());
            record.extend_from_slice(&0u32.to_le_bytes());
            record.extend_from_slice(&0u32.to_le_bytes());
            for value in [count, count, size, offset] { record.extend_from_slice(&value.to_le_bytes()); }
            record.extend_from_slice(LOCATOR_SIGNATURE);
            record.extend_from_slice(&0u32.to_le_bytes());
            record.extend_from_slice(&end.to_le_bytes());
            record.extend_from_slice(&1u32.to_le_bytes());
            self.out.write_all(&record)?;
            count = count.min(0xffff);
            size = size.min(0xffff_ffff);
            offset = offset.min(0xffff_ffff);
        }
        let mut record = Vec::with_capacity(END_SIZE);
        record.extend_from_slice(END_SIGNATURE);
        for value in [0u16, 0, count as u16, count as u16] { record.extend_from_slice(&value.to_le_bytes()); }
        record.extend_from_slice(&(size as u32).to_le_bytes());
        record.extend_from_slice(&(offset as u32).to_le_bytes());
        record.extend_from_slice(&0u16.to_le_bytes());
        self.out.write_all(&record)?;
        self.out.flush()?;
        Ok(self.out)
    }
}

// ------------------------------------------------------------- reading

const CP437_HIGH: &str = "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{a0}";

fn cp437(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| if *byte < 0x80 { *byte as char } else { CP437_HIGH.chars().nth(usize::from(*byte - 0x80)).unwrap() }).collect()
}

/// `_sanitize_filename` on Windows: up to the first NUL, separators as `/`.
fn sanitize(name: &str) -> String { name.split('\0').next().unwrap_or("").replace('\\', "/") }

fn u16_at(bytes: &[u8], at: usize) -> u16 { u16::from_le_bytes([bytes[at], bytes[at + 1]]) }
fn u32_at(bytes: &[u8], at: usize) -> u32 { u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) }
fn u64_at(bytes: &[u8], at: usize) -> u64 { u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) }

const MAX_EXTRACT_VERSION: u8 = 63;
const MIN_READ_SIZE: i128 = 4096;
/// `ZipExtFile.MAX_N` (`1 << 31 - 1`).
const MAX_N: usize = 1 << 30;

/// One central-directory entry (`ZipInfo`), sizes and offsets as Python ints.
#[derive(Clone, Debug)]
pub struct ZipInfo {
    pub filename: String,
    orig_filename: String,
    flags: u16,
    method: u16,
    crc: u32,
    compress_size: i128,
    file_size: i128,
    header_offset: i128,
    end_offset: i128,
}

/// `io.DEFAULT_BUFFER_SIZE`, the buffer of the `open(file, "rb")` zipfile reads.
const READ_BUFFER: i128 = 8192;

/// The archive file as `_SharedFile` reads it: a seek, then up to `n` bytes.
struct Shared { file: std::fs::File, size: u64 }

impl Shared {
    fn read_at(&mut self, at: i128, n: i128) -> Result<Vec<u8>> {
        let invalid = || ArchiveError::new(ErrorKind::Os, "[Errno 22] Invalid argument");
        // BufferedReader.seek: an offset beyond off_t, then lseek.
        if at > i128::from(i64::MAX) || at < i128::from(i64::MIN) { return Err(ArchiveError::value("cannot fit 'int' into an offset-sized integer")); }
        if at < 0 { return Err(invalid()); }
        if n <= 0 { return Ok(Vec::new()); }
        // Its raw read (one 8192-byte buffer, or whole buffers) may not end
        // beyond the largest file offset.
        let request = if n <= READ_BUFFER { READ_BUFFER } else { n - n % READ_BUFFER };
        if at + request > i128::from(i64::MAX) { return Err(invalid()); }
        let available = (i128::from(self.size) - at).max(0);
        let take = n.min(available).max(0) as usize;
        if take == 0 { return Ok(Vec::new()); }
        let os = |error: std::io::Error| ArchiveError::os(&error, None);
        self.file.seek(SeekFrom::Start(at as u64)).map_err(os)?;
        let mut buffer = vec![0u8; take];
        self.file.read_exact(&mut buffer).map_err(os)?;
        Ok(buffer)
    }
}

/// `_EndRecData` and `_EndRecData64`: `(directory size, directory offset,
/// record location)`, or `None` when no end record is found.
fn end_record(shared: &mut Shared) -> Result<Option<(i128, i128, i128)>> {
    let file_size = i128::from(shared.size);
    if file_size < END_SIZE as i128 { return Ok(None); }
    let tail = shared.read_at(file_size - END_SIZE as i128, END_SIZE as i128)?;
    let (record, location) = if &tail[..4] == END_SIGNATURE && tail[20..22] == [0, 0] {
        (tail, file_size - END_SIZE as i128)
    } else {
        let start = (file_size - (MAX_COMMENT + END_SIZE) as i128).max(0);
        let data = shared.read_at(start, (MAX_COMMENT + END_SIZE) as i128)?;
        let Some(found) = data.windows(4).rposition(|window| window == END_SIGNATURE) else { return Ok(None) };
        if found + END_SIZE > data.len() { return Ok(None); }
        (data[found..found + END_SIZE].to_vec(), start + found as i128)
    };
    let (size, offset) = (i128::from(u32_at(&record, 12)), i128::from(u32_at(&record, 16)));
    // `_EndRecData64`'s short reads are OSErrors, which make no zip file.
    let unknown = || ArchiveError::zip("File is not a zip file");
    let mut at = location - LOCATOR_SIZE as i128;
    if at < 0 { return Ok(Some((size, offset, location))); }
    let locator = shared.read_at(at, LOCATOR_SIZE as i128)?;
    if locator.len() != LOCATOR_SIZE { return Err(unknown()); }
    if &locator[..4] != LOCATOR_SIGNATURE { return Ok(Some((size, offset, location))); }
    let (disk, relative, disks) = (u32_at(&locator, 4), i128::from(u64_at(&locator, 8)), u32_at(&locator, 16));
    if disk != 0 || disks > 1 { return Err(ArchiveError::zip("zipfiles that span multiple disks are not supported")); }
    at -= END64_SIZE as i128;
    if relative > at { return Err(ArchiveError::zip("Corrupt zip64 end of central directory locator")); }
    let mut extra = at - relative;
    let mut data = shared.read_at(relative, END64_SIZE as i128)?;
    if data.len() != END64_SIZE { return Err(unknown()); }
    if &data[..4] != END64_SIGNATURE && relative != at {
        extra = 0;
        data = shared.read_at(at, END64_SIZE as i128)?;
        if data.len() != END64_SIZE { return Err(unknown()); }
    }
    if &data[..4] != END64_SIGNATURE { return Err(ArchiveError::zip("Zip64 end of central directory record not found")); }
    let (record_size, directory_size, directory_offset) = (i128::from(u64_at(&data, 4)), i128::from(u64_at(&data, 40)), i128::from(u64_at(&data, 48)));
    if directory_offset + directory_size != relative || record_size + 12 != END64_SIZE as i128 + extra {
        return Err(ArchiveError::zip("Corrupt zip64 end of central directory record"));
    }
    Ok(Some((directory_size, directory_offset, at - extra)))
}

/// `ZipInfo._decodeExtra(filename_crc)`: ZIP64 sizes and offset, and an
/// Info-ZIP Unicode path.
fn decode_extra(info: &mut ZipInfo, mut extra: &[u8], filename_crc: u32) -> Result<()> {
    while extra.len() >= 4 {
        let (kind, len) = (u16_at(extra, 0), usize::from(u16_at(extra, 2)));
        if len + 4 > extra.len() { return Err(ArchiveError::zip(format!("Corrupt extra field {kind:04x} (size={len})"))); }
        let data = &extra[4..len + 4];
        if kind == 0x0001 {
            let mut rest = data;
            let mut take = |field: &str| -> Result<i128> {
                if rest.len() < 8 { return Err(ArchiveError::zip(format!("Corrupt zip64 extra field. {field} not found."))); }
                let value = i128::from(u64_at(rest, 0));
                rest = &rest[8..];
                Ok(value)
            };
            if info.file_size == 0xFFFF_FFFF_FFFF_FFFF || info.file_size == 0xFFFF_FFFF { info.file_size = take("File size")?; }
            if info.compress_size == 0xFFFF_FFFF { info.compress_size = take("Compress size")?; }
            if info.header_offset == 0xFFFF_FFFF { info.header_offset = take("Header offset")?; }
        } else if kind == 0x7075 {
            if data.len() < 5 { return Err(ArchiveError::zip("Corrupt unicode path extra field (0x7075)")); }
            if data[0] == 1 && u32_at(data, 1) == filename_crc {
                let name = crate::pystr::decode_utf8(&data[5..])
                    .map_err(|_| ArchiveError::zip("Corrupt unicode path extra field (0x7075): invalid utf-8 bytes"))?;
                if !name.is_empty() { info.filename = sanitize(&name); }
            }
        }
        extra = &extra[len + 4..];
    }
    Ok(())
}

/// A `ZipFile` opened for reading.
pub struct ZipReader { shared: Shared, entries: Vec<ZipInfo>, by_name: std::collections::HashMap<String, usize> }

impl ZipReader {
    /// `ZipFile(path)`.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let display = path.to_string_lossy();
        let file = std::fs::File::open(path).map_err(|error| ArchiveError::io(&error, &display))?;
        Self::from_file(file)
    }

    /// `ZipFile(fileobj)`: `_RealGetContents`.
    pub fn from_file(file: std::fs::File) -> Result<Self> {
        let size = file.metadata().map_err(|error| ArchiveError::os(&error, None))?.len();
        let mut shared = Shared { file, size };
        let (size_cd, offset_cd, location) = end_record(&mut shared)?.ok_or_else(|| ArchiveError::zip("File is not a zip file"))?;
        let concat = location - size_cd - offset_cd;
        let start_dir = offset_cd + concat;
        if start_dir < 0 { return Err(ArchiveError::zip("Bad offset for central directory")); }
        let directory = shared.read_at(start_dir, size_cd)?;
        let mut cursor = 0usize;
        let mut read = |len: usize| -> Vec<u8> {
            let from = cursor.min(directory.len());
            let to = from.saturating_add(len).min(directory.len());
            cursor = to;
            directory[from..to].to_vec()
        };
        let mut entries: Vec<ZipInfo> = Vec::new();
        let mut by_name = std::collections::HashMap::new();
        let mut total: i128 = 0;
        while total < size_cd {
            let record = read(CENTRAL_SIZE);
            if record.len() != CENTRAL_SIZE { return Err(ArchiveError::zip("Truncated central directory")); }
            if &record[..4] != CENTRAL_SIGNATURE { return Err(ArchiveError::zip("Bad magic number for central directory")); }
            let (name_len, extra_len, comment_len) = (usize::from(u16_at(&record, 28)), usize::from(u16_at(&record, 30)), usize::from(u16_at(&record, 32)));
            let raw_name = read(name_len);
            let flags = u16_at(&record, 8);
            let filename = if flags & FLAG_UTF8 != 0 { crate::pystr::decode_utf8(&raw_name)? } else { cp437(&raw_name) };
            let extra = read(extra_len);
            read(comment_len);
            if record[6] > MAX_EXTRACT_VERSION {
                return Err(ArchiveError::new(ErrorKind::NotImplemented, format!("zip file version {:.1}", f64::from(record[6]) / 10.0)));
            }
            let mut info = ZipInfo {
                filename: sanitize(&filename), orig_filename: filename, flags, method: u16_at(&record, 10), crc: u32_at(&record, 16),
                compress_size: i128::from(u32_at(&record, 20)), file_size: i128::from(u32_at(&record, 24)),
                header_offset: i128::from(u32_at(&record, 42)), end_offset: 0,
            };
            decode_extra(&mut info, &extra, crc32fast::hash(&raw_name))?;
            info.header_offset += concat;
            by_name.insert(info.filename.clone(), entries.len());
            entries.push(info);
            total += (CENTRAL_SIZE + name_len + extra_len + comment_len) as i128;
        }
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by_key(|index| entries[*index].header_offset);
        let mut end = start_dir;
        for index in order.into_iter().rev() { entries[index].end_offset = end; end = entries[index].header_offset; }
        Ok(Self { shared, entries, by_name })
    }

    /// `namelist()`.
    pub fn names(&self) -> Vec<&str> { self.entries.iter().map(|entry| entry.filename.as_str()).collect() }

    /// `open(name)`: the local header checked, a stream over the member.
    pub fn open_member(&mut self, name: &str) -> Result<ZipExtFile<'_>> {
        let index = *self.by_name.get(name)
            .ok_or_else(|| ArchiveError::new(ErrorKind::Key, py_repr_str(&format!("There is no item named {} in the archive", py_repr_str(name)))))?;
        let info = self.entries[index].clone();
        let header = self.shared.read_at(info.header_offset, LOCAL_SIZE as i128)?;
        if header.len() != LOCAL_SIZE { return Err(ArchiveError::zip("Truncated file header")); }
        if &header[..4] != LOCAL_SIGNATURE { return Err(ArchiveError::zip("Bad magic number for file header")); }
        let (name_len, extra_len) = (i128::from(u16_at(&header, 26)), i128::from(u16_at(&header, 28)));
        let raw_name = self.shared.read_at(info.header_offset + LOCAL_SIZE as i128, name_len)?;
        let data_start = info.header_offset + LOCAL_SIZE as i128 + raw_name.len() as i128 + extra_len;
        if info.flags & (1 << 5) != 0 { return Err(ArchiveError::new(ErrorKind::NotImplemented, "compressed patched data (flag bit 5)")); }
        if info.flags & (1 << 6) != 0 { return Err(ArchiveError::new(ErrorKind::NotImplemented, "strong encryption (flag bit 6)")); }
        let header_name = if u16_at(&header, 6) & FLAG_UTF8 != 0 { crate::pystr::decode_utf8(&raw_name)? } else { cp437(&raw_name) };
        if header_name != info.orig_filename {
            return Err(ArchiveError::zip(format!("File name in directory {} and header {} differ.", py_repr_str(&info.orig_filename), crate::json::py_repr_bytes(&raw_name))));
        }
        if data_start + info.compress_size > info.end_offset && info.end_offset != info.header_offset {
            return Err(ArchiveError::zip(format!("Overlapped entries: {} (possible zip bomb)", py_repr_str(&info.orig_filename))));
        }
        if info.flags & 1 != 0 {
            return Err(ArchiveError::new(ErrorKind::Runtime, format!("File {} is encrypted, password required for extraction", py_repr_str(name))));
        }
        let decompressor = match info.method {
            STORED => None,
            DEFLATED => Some(crate::inflate::Decompress::new()),
            // zipfile reads these through bz2 and lzma, which the native reader does not carry.
            12 => return Err(ArchiveError::new(ErrorKind::Runtime, "Compression requires the (missing) bz2 module")),
            14 => return Err(ArchiveError::new(ErrorKind::Runtime, "Compression requires the (missing) lzma module")),
            _ => return Err(ArchiveError::new(ErrorKind::NotImplemented, "That compression method is not supported")),
        };
        Ok(ZipExtFile {
            shared: &mut self.shared, position: data_start, name: info.filename, decompressor, compress_left: info.compress_size,
            left: info.file_size, eof: false, buffer: Vec::new(), offset: 0, expected_crc: info.crc, running_crc: 0,
        })
    }

    /// `read(name)`: the whole member.
    pub fn read(&mut self, name: &str) -> Result<Vec<u8>> { self.open_member(name)?.read_all() }
}

/// `ZipExtFile`: one member read as zipfile reads it, in 4 KiB or larger
/// compressed reads, with the CRC-32 checked the moment its end is reached.
pub struct ZipExtFile<'a> {
    shared: &'a mut Shared,
    position: i128,
    name: String,
    decompressor: Option<crate::inflate::Decompress>,
    compress_left: i128,
    left: i128,
    eof: bool,
    buffer: Vec<u8>,
    offset: usize,
    expected_crc: u32,
    running_crc: u32,
}

impl ZipExtFile<'_> {
    /// `read(n)` for `n >= 0`.
    pub fn read(&mut self, n: usize) -> Result<Vec<u8>> {
        let end = n.saturating_add(self.offset);
        if end < self.buffer.len() {
            let out = self.buffer[self.offset..end].to_vec();
            self.offset = end;
            return Ok(out);
        }
        let mut n = end - self.buffer.len();
        let mut out = self.buffer[self.offset..].to_vec();
        self.buffer.clear();
        self.offset = 0;
        while n > 0 && !self.eof {
            let data = self.read1(n)?;
            if n < data.len() {
                out.extend_from_slice(&data[..n]);
                self.buffer = data;
                self.offset = n;
                break;
            }
            n -= data.len();
            if out.is_empty() { out = data; } else { out.extend_from_slice(&data); }
        }
        Ok(out)
    }

    /// `read()`: everything to the end.
    pub fn read_all(&mut self) -> Result<Vec<u8>> {
        let mut out = self.buffer[self.offset..].to_vec();
        self.buffer.clear();
        self.offset = 0;
        while !self.eof {
            let data = self.read1(MAX_N)?;
            if out.is_empty() { out = data; } else { out.extend_from_slice(&data); }
        }
        Ok(out)
    }

    /// `_read1(n)`.
    fn read1(&mut self, n: usize) -> Result<Vec<u8>> {
        if self.eof || n == 0 { return Ok(Vec::new()); }
        let mut data = match &self.decompressor {
            Some(decompressor) => {
                let mut data = decompressor.unconsumed_tail().to_vec();
                if n > data.len() {
                    let more = self.read2(n - data.len())?;
                    if data.is_empty() { data = more; } else { data.extend(more); }
                }
                data
            }
            None => self.read2(n)?,
        };
        match &mut self.decompressor {
            None => self.eof = self.compress_left <= 0,
            Some(decompressor) => {
                data = decompressor.decompress(&data, n.max(MIN_READ_SIZE as usize))?;
                self.eof = decompressor.eof() || (self.compress_left <= 0 && decompressor.unconsumed_tail().is_empty());
                if self.eof { data.extend(decompressor.flush()); }
            }
        }
        data.truncate(self.left.clamp(0, data.len() as i128) as usize);
        self.left -= data.len() as i128;
        if self.left <= 0 { self.eof = true; }
        let mut hasher = crc32fast::Hasher::new_with_initial(self.running_crc);
        hasher.update(&data);
        self.running_crc = hasher.finalize();
        if self.eof && self.running_crc != self.expected_crc {
            return Err(ArchiveError::zip(format!("Bad CRC-32 for file {}", py_repr_str(&self.name))));
        }
        Ok(data)
    }

    /// `_read2(n)`: the next compressed bytes from the file.
    fn read2(&mut self, n: usize) -> Result<Vec<u8>> {
        if self.compress_left <= 0 { return Ok(Vec::new()); }
        let want = (n as i128).max(MIN_READ_SIZE).min(self.compress_left);
        let data = self.shared.read_at(self.position, want)?;
        self.position += data.len() as i128;
        self.compress_left -= data.len() as i128;
        if data.is_empty() { return Err(ArchiveError::new(ErrorKind::Eof, "")); }
        Ok(data)
    }
}
