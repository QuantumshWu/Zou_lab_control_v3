//! zlib 1.3.1's raw inflate (`inflate.c`, `inffast.c`, `inftrees.c`; 32 KiB
//! window, `INFLATE_STRICT` off), and CPython's `zlib.decompressobj(-15)`
//! around it: the same output blocks per `inflate()` call, so a corrupt
//! stream fails with zlib's message at the same point, and input is consumed
//! and left over exactly as `unconsumed_tail` reports it.

use crate::error::{ArchiveError, ErrorKind, Result};

#[derive(Clone, Copy, Default)]
struct Code { op: u8, bits: u8, val: u16 }

const ENOUGH_LENS: usize = 852;
const ENOUGH_DISTS: usize = 592;
const ENOUGH: usize = ENOUGH_LENS + ENOUGH_DISTS;
const MAXBITS: usize = 15;
const WBITS: u32 = 15;

#[derive(Clone, Copy, PartialEq, Eq)]
enum CodeType { Codes, Lens, Dists }

/// `inflate_table`: 0 on success, -1 for an invalid code, +1 when ENOUGH is not.
fn inflate_table(kind: CodeType, lens: &[u16], table: &mut [Code], next: &mut usize, bits: &mut u32, work: &mut [u16]) -> i32 {
    const LBASE: [u16; 31] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0];
    const LEXT: [u16; 31] = [16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 16, 203, 77];
    const DBASE: [u16; 32] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
                              8193, 12289, 16385, 24577, 0, 0];
    const DEXT: [u16; 32] = [16, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
                             28, 28, 29, 29, 64, 64];
    let codes = lens.len();
    let mut count = [0u16; MAXBITS + 1];
    let mut offs = [0u16; MAXBITS + 1];
    for &len in lens { count[len as usize] += 1; }
    let mut root = *bits as usize;
    let mut max = MAXBITS;
    while max >= 1 && count[max] == 0 { max -= 1; }
    if root > max { root = max; }
    if max == 0 {
        let here = Code { op: 64, bits: 1, val: 0 };
        table[*next] = here;
        table[*next + 1] = here;
        *next += 2;
        *bits = 1;
        return 0;
    }
    let mut min = 1;
    while min < max && count[min] == 0 { min += 1; }
    if root < min { root = min; }
    let mut left: i32 = 1;
    for len in 1..=MAXBITS {
        left <<= 1;
        left -= i32::from(count[len]);
        if left < 0 { return -1; }
    }
    if left > 0 && (kind == CodeType::Codes || max != 1) { return -1; }
    offs[1] = 0;
    for len in 1..MAXBITS { offs[len + 1] = offs[len] + count[len]; }
    for (sym, &len) in lens.iter().enumerate().take(codes) {
        if len != 0 { work[offs[len as usize] as usize] = sym as u16; offs[len as usize] += 1; }
    }
    let (base, extra, matched): (&[u16], &[u16], u32) = match kind {
        CodeType::Codes => (&[], &[], 20),
        CodeType::Lens => (&LBASE, &LEXT, 257),
        CodeType::Dists => (&DBASE, &DEXT, 0),
    };
    let start = *next;
    let mut huff: u32 = 0;
    let mut sym = 0usize;
    let mut len = min;
    let mut next_at = start;
    let mut curr = root;
    let mut drop = 0usize;
    let mut low: u32 = u32::MAX;
    let mut used: usize = 1 << root;
    let mask: u32 = (used - 1) as u32;
    if (kind == CodeType::Lens && used > ENOUGH_LENS) || (kind == CodeType::Dists && used > ENOUGH_DISTS) { return 1; }
    loop {
        let symbol = u32::from(work[sym]);
        let here = if symbol + 1 < matched {
            Code { op: 0, bits: (len - drop) as u8, val: work[sym] }
        } else if symbol >= matched {
            Code { op: extra[(symbol - matched) as usize] as u8, bits: (len - drop) as u8, val: base[(symbol - matched) as usize] }
        } else {
            Code { op: 32 + 64, bits: (len - drop) as u8, val: 0 }
        };
        let incr = 1u32 << (len - drop);
        let mut fill = 1u32 << curr;
        let saved = fill;
        loop {
            fill -= incr;
            table[next_at + ((huff >> drop) + fill) as usize] = here;
            if fill == 0 { break; }
        }
        let mut incr = 1u32 << (len - 1);
        while huff & incr != 0 { incr >>= 1; }
        if incr != 0 { huff &= incr - 1; huff += incr; } else { huff = 0; }
        sym += 1;
        count[len] -= 1;
        if count[len] == 0 {
            if len == max { break; }
            len = lens[work[sym] as usize] as usize;
        }
        if len > root && (huff & mask) != low {
            if drop == 0 { drop = root; }
            next_at += saved as usize;
            curr = len - drop;
            let mut left: i32 = 1 << curr;
            while curr + drop < max {
                left -= i32::from(count[curr + drop]);
                if left <= 0 { break; }
                curr += 1;
                left <<= 1;
            }
            used += 1 << curr;
            if (kind == CodeType::Lens && used > ENOUGH_LENS) || (kind == CodeType::Dists && used > ENOUGH_DISTS) { return 1; }
            low = huff & mask;
            table[start + low as usize] = Code { op: curr as u8, bits: root as u8, val: (next_at - start) as u16 };
        }
    }
    if huff != 0 { table[next_at + huff as usize] = Code { op: 64, bits: (len - drop) as u8, val: 0 }; }
    *next = start + used;
    *bits = root as u32;
    0
}

/// The fixed-block tables (`fixedtables`, as `inffixed.h` holds them).
fn fixed_tables() -> &'static (Vec<Code>, Vec<Code>) {
    static FIXED: std::sync::OnceLock<(Vec<Code>, Vec<Code>)> = std::sync::OnceLock::new();
    FIXED.get_or_init(|| {
        let mut lens = [0u16; 288];
        for (sym, len) in lens.iter_mut().enumerate() { *len = match sym { 0..=143 => 8, 144..=255 => 9, 256..=279 => 7, _ => 8 }; }
        let mut work = [0u16; 288];
        let mut lencode = vec![Code::default(); 512];
        let (mut next, mut bits) = (0, 9);
        inflate_table(CodeType::Lens, &lens, &mut lencode, &mut next, &mut bits, &mut work);
        // inffixed.h marks the two unused length symbols as plain invalid codes.
        for (index, entry) in lencode.iter_mut().enumerate() { if index & 127 == 99 { entry.op = 64; } }
        let mut distcode = vec![Code::default(); 32];
        let (mut next, mut bits) = (0, 5);
        inflate_table(CodeType::Dists, &[5u16; 32], &mut distcode, &mut next, &mut bits, &mut work);
        (lencode, distcode)
    })
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Mode { Type, Stored, Copy, Table, LenLens, CodeLens, Len, LenExt, Dist, DistExt, Match, Lit, Check, Done, Bad }

/// What `inflate()` returns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status { Ok, StreamEnd, BufError, DataError }

/// One `z_stream` + `inflate_state` in raw mode.
pub struct Inflate {
    mode: Mode,
    last: bool,
    hold: u64,
    bits: u32,
    length: u32,
    offset: u32,
    extra: u32,
    /// Offsets of the length/literal and distance tables in `codes`.
    lencode: usize,
    distcode: usize,
    lenbits: u32,
    distbits: u32,
    ncode: u32,
    nlen: u32,
    ndist: u32,
    have: u32,
    lens: [u16; 320],
    work: [u16; 288],
    codes: Box<[Code; ENOUGH]>,
    window: Vec<u8>,
    wsize: u32,
    whave: u32,
    wnext: u32,
    /// `strm->msg`.
    pub msg: Option<&'static str>,
}

impl Default for Inflate {
    fn default() -> Self { Self::new() }
}

impl Inflate {
    /// `inflateInit2(strm, -15)`.
    pub fn new() -> Self {
        Self {
            mode: Mode::Type, last: false, hold: 0, bits: 0, length: 0, offset: 0, extra: 0, lencode: 0, distcode: 0,
            lenbits: 0, distbits: 0, ncode: 0, nlen: 0, ndist: 0, have: 0, lens: [0; 320], work: [0; 288], codes: Box::new([Code::default(); ENOUGH]),
            window: Vec::new(), wsize: 0, whave: 0, wnext: 0, msg: None,
        }
    }

    fn code(&self, table: usize, index: usize) -> Code { self.codes[table + index] }

    /// `updatewindow(strm, end, copy)` with the last `copy` bytes of `output`.
    fn update_window(&mut self, output: &[u8], copy: usize) {
        if self.window.is_empty() { self.window = vec![0; 1 << WBITS]; }
        if self.wsize == 0 { self.wsize = 1 << WBITS; self.wnext = 0; self.whave = 0; }
        let wsize = self.wsize as usize;
        let end = output.len();
        if copy >= wsize {
            self.window.copy_from_slice(&output[end - wsize..end]);
            self.wnext = 0;
            self.whave = self.wsize;
        } else {
            let wnext = self.wnext as usize;
            let dist = (wsize - wnext).min(copy);
            self.window[wnext..wnext + dist].copy_from_slice(&output[end - copy..end - copy + dist]);
            let rest = copy - dist;
            if rest != 0 {
                self.window[..rest].copy_from_slice(&output[end - rest..end]);
                self.wnext = rest as u32;
                self.whave = self.wsize;
            } else {
                self.wnext += dist as u32;
                if self.wnext == self.wsize { self.wnext = 0; }
                if self.whave < self.wsize { self.whave += dist as u32; }
            }
        }
    }

    /// `inflate(strm, flush)` over `input` into `output`: the status, bytes
    /// consumed and bytes written.
    pub fn inflate(&mut self, input: &[u8], output: &mut [u8], finish: bool) -> (Status, usize, usize) {
        let mut next = 0usize;
        let mut have = input.len() as u32;
        let mut put = 0usize;
        let mut left = output.len() as u32;
        let mut hold = self.hold;
        let mut bits = self.bits;
        let start_in = have;
        let out = left;
        let mut ret = Status::Ok;
        const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
        macro_rules! pull { ($l:lifetime) => {{ if have == 0 { break $l; } have -= 1; hold += u64::from(input[next]) << bits; next += 1; bits += 8; }}; }
        macro_rules! need { ($l:lifetime, $n:expr) => {{ while bits < ($n) as u32 { pull!($l); } }}; }
        macro_rules! take { ($n:expr) => { (hold as u32) & ((1u32 << ($n)) - 1) }; }
        macro_rules! drop_bits { ($n:expr) => {{ hold >>= ($n); bits -= ($n) as u32; }}; }
        macro_rules! bad { ($message:expr) => {{ self.msg = Some($message); self.mode = Mode::Bad; continue; }}; }
        'leave: loop {
            match self.mode {
                Mode::Type => {
                    if self.last {
                        hold >>= bits & 7;
                        bits -= bits & 7;
                        self.mode = Mode::Check;
                        continue;
                    }
                    need!('leave, 3);
                    self.last = take!(1) != 0;
                    drop_bits!(1);
                    match take!(2) {
                        0 => self.mode = Mode::Stored,
                        1 => {
                            // The fixed tables, copied where dynamic ones are built.
                            let (lens, dists) = fixed_tables();
                            self.codes[..512].copy_from_slice(lens);
                            self.codes[512..544].copy_from_slice(dists);
                            self.lencode = 0;
                            self.lenbits = 9;
                            self.distcode = 512;
                            self.distbits = 5;
                            self.mode = Mode::Len;
                        }
                        2 => self.mode = Mode::Table,
                        _ => { self.msg = Some("invalid block type"); self.mode = Mode::Bad; }
                    }
                    drop_bits!(2);
                }
                Mode::Stored => {
                    hold >>= bits & 7;
                    bits -= bits & 7;
                    need!('leave, 32);
                    if (hold & 0xffff) != ((hold >> 16) ^ 0xffff) { bad!("invalid stored block lengths"); }
                    self.length = (hold & 0xffff) as u32;
                    hold = 0;
                    bits = 0;
                    self.mode = Mode::Copy;
                }
                Mode::Copy => {
                    let mut copy = self.length;
                    if copy != 0 {
                        if copy > have { copy = have; }
                        if copy > left { copy = left; }
                        if copy == 0 { break 'leave; }
                        let copy = copy as usize;
                        output[put..put + copy].copy_from_slice(&input[next..next + copy]);
                        have -= copy as u32;
                        next += copy;
                        left -= copy as u32;
                        put += copy;
                        self.length -= copy as u32;
                        continue;
                    }
                    self.mode = Mode::Type;
                }
                Mode::Table => {
                    need!('leave, 14);
                    self.nlen = take!(5) + 257;
                    drop_bits!(5);
                    self.ndist = take!(5) + 1;
                    drop_bits!(5);
                    self.ncode = take!(4) + 4;
                    drop_bits!(4);
                    if self.nlen > 286 || self.ndist > 30 { bad!("too many length or distance symbols"); }
                    self.have = 0;
                    self.mode = Mode::LenLens;
                }
                Mode::LenLens => {
                    while self.have < self.ncode {
                        need!('leave, 3);
                        self.lens[ORDER[self.have as usize]] = take!(3) as u16;
                        self.have += 1;
                        drop_bits!(3);
                    }
                    while self.have < 19 { self.lens[ORDER[self.have as usize]] = 0; self.have += 1; }
                    let (mut next_code, mut lenbits) = (0usize, 7u32);
                    let lens = self.lens;
                    let ret_code = inflate_table(CodeType::Codes, &lens[..19], &mut self.codes[..], &mut next_code, &mut lenbits, &mut self.work);
                    self.lencode = 0;
                    self.lenbits = lenbits;
                    if ret_code != 0 { bad!("invalid code lengths set"); }
                    self.have = 0;
                    self.mode = Mode::CodeLens;
                }
                Mode::CodeLens => {
                    while self.have < self.nlen + self.ndist {
                        let mut here;
                        loop {
                            here = self.code(self.lencode, take!(self.lenbits) as usize);
                            if u32::from(here.bits) <= bits { break; }
                            pull!('leave);
                        }
                        if here.val < 16 {
                            drop_bits!(here.bits);
                            self.lens[self.have as usize] = here.val;
                            self.have += 1;
                        } else {
                            let (len, copy);
                            if here.val == 16 {
                                need!('leave, u32::from(here.bits) + 2);
                                drop_bits!(here.bits);
                                if self.have == 0 { self.msg = Some("invalid bit length repeat"); self.mode = Mode::Bad; break; }
                                len = self.lens[self.have as usize - 1];
                                copy = 3 + take!(2);
                                drop_bits!(2);
                            } else if here.val == 17 {
                                need!('leave, u32::from(here.bits) + 3);
                                drop_bits!(here.bits);
                                len = 0;
                                copy = 3 + take!(3);
                                drop_bits!(3);
                            } else {
                                need!('leave, u32::from(here.bits) + 7);
                                drop_bits!(here.bits);
                                len = 0;
                                copy = 11 + take!(7);
                                drop_bits!(7);
                            }
                            if self.have + copy > self.nlen + self.ndist { self.msg = Some("invalid bit length repeat"); self.mode = Mode::Bad; break; }
                            for _ in 0..copy { self.lens[self.have as usize] = len; self.have += 1; }
                        }
                    }
                    if self.mode == Mode::Bad { continue; }
                    if self.lens[256] == 0 { bad!("invalid code -- missing end-of-block"); }
                    let lens = self.lens;
                    let (nlen, ndist) = (self.nlen as usize, self.ndist as usize);
                    let mut next_code = 0usize;
                    let mut lenbits = 9u32;
                    let ret_code = inflate_table(CodeType::Lens, &lens[..nlen], &mut self.codes[..], &mut next_code, &mut lenbits, &mut self.work);
                    self.lencode = 0;
                    self.lenbits = lenbits;
                    if ret_code != 0 { bad!("invalid literal/lengths set"); }
                    let dist_start = next_code;
                    let mut distbits = 6u32;
                    let ret_code = inflate_table(CodeType::Dists, &lens[nlen..nlen + ndist], &mut self.codes[..], &mut next_code, &mut distbits, &mut self.work);
                    self.distcode = dist_start;
                    self.distbits = distbits;
                    if ret_code != 0 { bad!("invalid distances set"); }
                    self.mode = Mode::Len;
                }
                Mode::Len => {
                    if have >= 6 && left >= 258 {
                        self.hold = hold;
                        self.bits = bits;
                        let (consumed, written) = self.inflate_fast(&input[next..next + have as usize], output, put, left as usize, out as usize);
                        next += consumed;
                        have -= consumed as u32;
                        put += written;
                        left -= written as u32;
                        hold = self.hold;
                        bits = self.bits;
                        continue;
                    }
                    let mut here;
                    loop {
                        here = self.code(self.lencode, take!(self.lenbits) as usize);
                        if u32::from(here.bits) <= bits { break; }
                        pull!('leave);
                    }
                    if here.op != 0 && (here.op & 0xf0) == 0 {
                        let last = here;
                        loop {
                            let index = usize::from(last.val) + ((take!(u32::from(last.bits) + u32::from(last.op)) >> last.bits) as usize);
                            here = self.code(self.lencode, index);
                            if u32::from(last.bits) + u32::from(here.bits) <= bits { break; }
                            pull!('leave);
                        }
                        drop_bits!(last.bits);
                    }
                    drop_bits!(here.bits);
                    self.length = u32::from(here.val);
                    if here.op == 0 { self.mode = Mode::Lit; continue; }
                    if here.op & 32 != 0 { self.mode = Mode::Type; continue; }
                    if here.op & 64 != 0 { bad!("invalid literal/length code"); }
                    self.extra = u32::from(here.op) & 15;
                    self.mode = Mode::LenExt;
                }
                Mode::LenExt => {
                    if self.extra != 0 {
                        need!('leave, self.extra);
                        self.length += take!(self.extra);
                        drop_bits!(self.extra);
                    }
                    self.mode = Mode::Dist;
                }
                Mode::Dist => {
                    let mut here;
                    loop {
                        here = self.code(self.distcode, take!(self.distbits) as usize);
                        if u32::from(here.bits) <= bits { break; }
                        pull!('leave);
                    }
                    if (here.op & 0xf0) == 0 {
                        let last = here;
                        loop {
                            let index = usize::from(last.val) + ((take!(u32::from(last.bits) + u32::from(last.op)) >> last.bits) as usize);
                            here = self.code(self.distcode, index);
                            if u32::from(last.bits) + u32::from(here.bits) <= bits { break; }
                            pull!('leave);
                        }
                        drop_bits!(last.bits);
                    }
                    drop_bits!(here.bits);
                    if here.op & 64 != 0 { bad!("invalid distance code"); }
                    self.offset = u32::from(here.val);
                    self.extra = u32::from(here.op) & 15;
                    self.mode = Mode::DistExt;
                }
                Mode::DistExt => {
                    if self.extra != 0 {
                        need!('leave, self.extra);
                        self.offset += take!(self.extra);
                        drop_bits!(self.extra);
                    }
                    self.mode = Mode::Match;
                }
                Mode::Match => {
                    if left == 0 { break 'leave; }
                    let written = out - left;
                    let mut copy;
                    if self.offset > written {
                        copy = self.offset - written;
                        if copy > self.whave { bad!("invalid distance too far back"); }
                        let from = if copy > self.wnext { copy -= self.wnext; (self.wsize - copy) as usize } else { (self.wnext - copy) as usize };
                        if copy > self.length { copy = self.length; }
                        if copy > left { copy = left; }
                        left -= copy;
                        self.length -= copy;
                        let copy = copy as usize;
                        output[put..put + copy].copy_from_slice(&self.window[from..from + copy]);
                        put += copy;
                    } else {
                        let from = put - self.offset as usize;
                        copy = self.length;
                        if copy > left { copy = left; }
                        left -= copy;
                        self.length -= copy;
                        for index in 0..copy as usize { output[put + index] = output[from + index]; }
                        put += copy as usize;
                    }
                    if self.length == 0 { self.mode = Mode::Len; }
                }
                Mode::Lit => {
                    if left == 0 { break 'leave; }
                    output[put] = self.length as u8;
                    put += 1;
                    left -= 1;
                    self.mode = Mode::Len;
                }
                Mode::Check => { self.mode = Mode::Done; }
                Mode::Done => { ret = Status::StreamEnd; break 'leave; }
                Mode::Bad => { ret = Status::DataError; break 'leave; }
            }
        }
        self.hold = hold;
        self.bits = bits;
        let written = (out - left) as usize;
        if self.wsize != 0 || (written != 0 && self.mode < Mode::Bad && (self.mode < Mode::Check || !finish)) {
            self.update_window(&output[..put], written);
        }
        let consumed = (start_in - have) as usize;
        if ((consumed == 0 && written == 0) || finish) && ret == Status::Ok { ret = Status::BufError; }
        (ret, consumed, written)
    }

    /// `inflate_fast(strm, start)`: decode while at least 6 input bytes and
    /// 258 output bytes are available. Returns bytes consumed and written.
    fn inflate_fast(&mut self, input: &[u8], output: &mut [u8], put: usize, left: usize, start: usize) -> (usize, usize) {
        let mut inp = 0usize;
        let last = input.len() - 5;
        let mut out = put;
        let beg = put - (start - left);
        let end = put + (left - 257);
        let (wsize, whave, wnext) = (self.wsize as usize, self.whave as usize, self.wnext as usize);
        let mut hold = self.hold;
        let mut bits = self.bits;
        let lmask = (1u64 << self.lenbits) - 1;
        let dmask = (1u64 << self.distbits) - 1;
        let (lencode, distcode) = (self.lencode, self.distcode);
        'outer: loop {
            if bits < 15 {
                hold += u64::from(input[inp]) << bits; inp += 1; bits += 8;
                hold += u64::from(input[inp]) << bits; inp += 1; bits += 8;
            }
            let mut here = self.code(lencode, (hold & lmask) as usize);
            loop {
                // dolen
                let op = u32::from(here.bits);
                hold >>= op;
                bits -= op;
                let op = u32::from(here.op);
                if op == 0 {
                    output[out] = here.val as u8;
                    out += 1;
                    break;
                } else if op & 16 != 0 {
                    let mut len = u32::from(here.val);
                    let op = op & 15;
                    if op != 0 {
                        if bits < op { hold += u64::from(input[inp]) << bits; inp += 1; bits += 8; }
                        len += (hold as u32) & ((1u32 << op) - 1);
                        hold >>= op;
                        bits -= op;
                    }
                    if bits < 15 {
                        hold += u64::from(input[inp]) << bits; inp += 1; bits += 8;
                        hold += u64::from(input[inp]) << bits; inp += 1; bits += 8;
                    }
                    let mut here = self.code(distcode, (hold & dmask) as usize);
                    loop {
                        // dodist
                        let op = u32::from(here.bits);
                        hold >>= op;
                        bits -= op;
                        let op = u32::from(here.op);
                        if op & 16 != 0 {
                            let mut dist = u32::from(here.val);
                            let op = op & 15;
                            if bits < op {
                                hold += u64::from(input[inp]) << bits; inp += 1; bits += 8;
                                if bits < op { hold += u64::from(input[inp]) << bits; inp += 1; bits += 8; }
                            }
                            dist += (hold as u32) & ((1u32 << op) - 1);
                            hold >>= op;
                            bits -= op;
                            let dist = dist as usize;
                            let mut len = len as usize;
                            let produced = out - beg;
                            if dist > produced {
                                let mut op = dist - produced;
                                if op > whave {
                                    self.msg = Some("invalid distance too far back");
                                    self.mode = Mode::Bad;
                                    break 'outer;
                                }
                                let mut from;
                                let mut from_window = true;
                                if wnext == 0 {
                                    from = wsize - op;
                                    if op < len {
                                        len -= op;
                                        output[out..out + op].copy_from_slice(&self.window[from..from + op]);
                                        out += op;
                                        from = out - dist;
                                        from_window = false;
                                    }
                                } else if wnext < op {
                                    from = wsize + wnext - op;
                                    op -= wnext;
                                    if op < len {
                                        len -= op;
                                        output[out..out + op].copy_from_slice(&self.window[from..from + op]);
                                        out += op;
                                        from = 0;
                                        if wnext < len {
                                            let op = wnext;
                                            len -= op;
                                            output[out..out + op].copy_from_slice(&self.window[from..from + op]);
                                            out += op;
                                            from = out - dist;
                                            from_window = false;
                                        }
                                    }
                                } else {
                                    from = wnext - op;
                                    if op < len {
                                        len -= op;
                                        output[out..out + op].copy_from_slice(&self.window[from..from + op]);
                                        out += op;
                                        from = out - dist;
                                        from_window = false;
                                    }
                                }
                                if from_window {
                                    output[out..out + len].copy_from_slice(&self.window[from..from + len]);
                                } else {
                                    copy_match(output, from, out, len);
                                }
                                out += len;
                            } else {
                                copy_match(output, out - dist, out, len);
                                out += len;
                            }
                            break;
                        } else if op & 64 == 0 {
                            let index = usize::from(here.val) + ((hold as u32) & ((1u32 << op) - 1)) as usize;
                            here = self.code(distcode, index);
                            continue;
                        } else {
                            self.msg = Some("invalid distance code");
                            self.mode = Mode::Bad;
                            break 'outer;
                        }
                    }
                    break;
                } else if op & 64 == 0 {
                    let index = usize::from(here.val) + ((hold as u32) & ((1u32 << op) - 1)) as usize;
                    here = self.code(lencode, index);
                    continue;
                } else if op & 32 != 0 {
                    self.mode = Mode::Type;
                    break 'outer;
                } else {
                    self.msg = Some("invalid literal/length code");
                    self.mode = Mode::Bad;
                    break 'outer;
                }
            }
            if !(inp < last && out < end) { break; }
        }
        let len = (bits >> 3) as usize;
        inp -= len;
        bits -= (len as u32) << 3;
        hold &= (1u64 << bits) - 1;
        self.hold = hold;
        self.bits = bits;
        (inp, out - put)
    }
}

/// Copy `len` bytes from `from` to `to` (`from < to`) front to back, as a
/// match repeats bytes it is still writing.
fn copy_match(output: &mut [u8], from: usize, to: usize, len: usize) {
    let dist = to - from;
    if dist >= len {
        output.copy_within(from..from + len, to);
        return;
    }
    // The first `dist` bytes repeat: copy doubling runs of what is written.
    let mut done = 0;
    while done < len {
        let run = (dist + done).min(len - done);
        output.copy_within(from..from + run, to + done);
        done += run;
    }
}

/// `zlib.decompressobj(-15)`.
pub struct Decompress { z: Inflate, unconsumed_tail: Vec<u8>, unused_data: Vec<u8>, eof: bool }

impl Default for Decompress {
    fn default() -> Self { Self::new() }
}

fn zlib_error(status: Status, msg: Option<&str>, context: &str) -> ArchiveError {
    let code = match status { Status::DataError => -3, Status::BufError => -5, _ => 0 };
    let text = msg.unwrap_or(match status { Status::BufError => "incomplete or truncated stream", _ => "invalid input data" });
    ArchiveError::new(ErrorKind::Zlib, format!("Error {code} {context}: {text}"))
}

impl Decompress {
    pub fn new() -> Self { Self { z: Inflate::new(), unconsumed_tail: Vec::new(), unused_data: Vec::new(), eof: false } }
    pub fn unconsumed_tail(&self) -> &[u8] { &self.unconsumed_tail }
    pub fn unused_data(&self) -> &[u8] { &self.unused_data }
    pub fn eof(&self) -> bool { self.eof }

    /// `save_unconsumed_input` after consuming `consumed` bytes of `data`.
    /// At the stream end the leftover joins `unused_data` while `next_in`
    /// stays put, so a tail that was already set keeps those bytes too.
    fn save(&mut self, data: &[u8], consumed: usize, status: Status, had_tail: bool) {
        let remaining = &data[consumed..];
        let mut avail = remaining.len();
        if status == Status::StreamEnd && avail > 0 {
            self.unused_data.extend_from_slice(remaining);
            avail = 0;
        }
        if avail > 0 || had_tail { self.unconsumed_tail = remaining.to_vec(); }
    }

    /// `inflate` over output blocks until the input is used, the output
    /// reaches `limit`, the stream ends or fails: output, bytes consumed and
    /// the last status.
    fn run(&mut self, data: &[u8], first: usize, limit: usize, finish: bool) -> (Vec<u8>, usize, Status) {
        // The space handed to each inflate call is not observable: CPython's
        // blocks end the loop where any sizes do (input used, limit reached,
        // stream end or error). Start near the expected size, then double.
        let mut output = vec![0u8; first.max(data.len().saturating_mul(3)).min(1 << 26).min(limit)];
        let (mut filled, mut consumed, mut status) = (0usize, 0usize, Status::Ok);
        loop {
            if filled == output.len() {
                if output.len() == limit { break; }
                let size = output.len().min(limit - output.len());
                output.resize(output.len() + size, 0);
            }
            let end = output.len();
            let (result, used, written) = self.z.inflate(&data[consumed..], &mut output[filled..end], finish);
            consumed += used;
            filled += written;
            status = result;
            if !matches!(result, Status::Ok | Status::BufError | Status::StreamEnd) || filled != end { break; }
        }
        output.truncate(filled);
        (output, consumed, status)
    }

    /// `decompress(data, max_length)`; `max_length` 0 is unlimited.
    pub fn decompress(&mut self, data: &[u8], max_length: usize) -> Result<Vec<u8>> {
        let limit = if max_length == 0 { usize::MAX } else { max_length };
        let (output, consumed, status) = self.run(data, 32 << 10, limit, false);
        let had_tail = !self.unconsumed_tail.is_empty();
        self.save(data, consumed, status, had_tail);
        if status == Status::StreamEnd {
            self.eof = true;
        } else if status != Status::Ok && status != Status::BufError {
            return Err(zlib_error(status, self.z.msg, "while decompressing data"));
        }
        Ok(output)
    }

    /// `flush()`: the unconsumed tail finished; a data error ends it quietly.
    pub fn flush(&mut self) -> Vec<u8> {
        let data = std::mem::take(&mut self.unconsumed_tail);
        let (output, consumed, status) = self.run(&data, 16 << 10, usize::MAX, true);
        self.save(&data, consumed, status, !data.is_empty());
        if status == Status::StreamEnd { self.eof = true; }
        output
    }
}
