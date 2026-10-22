//! zlib 1.3.1's raw deflate at level 1 (`deflate_fast`; windowBits 15,
//! memLevel 8, default strategy), ported statement for statement so the
//! compressed bytes equal CPython's `zlib.compressobj(1, DEFLATED, -15)` for
//! the same sequence of `compress` calls, and `zlib.compress(data, 1)` has
//! the same length.

const W_SIZE: usize = 1 << 15;
const W_MASK: usize = W_SIZE - 1;
const WINDOW_SIZE: usize = 2 * W_SIZE;
const HASH_SIZE: usize = 1 << 15;
const HASH_MASK: u32 = (HASH_SIZE - 1) as u32;
const HASH_SHIFT: u32 = 5;
const LIT_BUFSIZE: usize = 1 << 14;
const SYM_END: usize = (LIT_BUFSIZE - 1) * 3;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;
const MAX_DIST: usize = W_SIZE - MIN_LOOKAHEAD;
const WIN_INIT: usize = MAX_MATCH;
/// Level 1: good 4, lazy (max insert length) 4, nice 8, chain 4.
const GOOD_MATCH: usize = 4;
const MAX_INSERT_LENGTH: usize = 4;
const NICE_MATCH: usize = 8;
const MAX_CHAIN: u32 = 4;

const LENGTH_CODES: usize = 29;
const LITERALS: usize = 256;
const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const END_BLOCK: usize = 256;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;

const EXTRA_LBITS: [u32; LENGTH_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const EXTRA_DBITS: [u32; D_CODES] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u32; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [usize; BL_CODES] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// `ct_data`: `fc` is Freq or Code and `dl` is Dad or Len, overlaid as the union overlays them.
#[derive(Clone, Copy, Default)]
struct Ct { fc: u16, dl: u16 }

struct Tables {
    static_ltree: [Ct; L_CODES + 2],
    static_dtree: [Ct; D_CODES],
    dist_code: [u8; 512],
    length_code: [u8; MAX_MATCH - MIN_MATCH + 1],
    base_length: [u32; LENGTH_CODES],
    base_dist: [u32; D_CODES],
}

fn bi_reverse(mut code: u32, mut len: u32) -> u32 {
    let mut result = 0;
    loop {
        result |= code & 1;
        code >>= 1;
        result <<= 1;
        len -= 1;
        if len == 0 { return result >> 1; }
    }
}

fn gen_codes(tree: &mut [Ct], max_code: i32, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code: u32 = 0;
    for bits in 1..=MAX_BITS {
        code = (code + u32::from(bl_count[bits - 1])) << 1;
        next_code[bits] = code as u16;
    }
    for ct in tree.iter_mut().take((max_code + 1).max(0) as usize) {
        let len = ct.dl as usize;
        if len == 0 { continue; }
        ct.fc = bi_reverse(u32::from(next_code[len]), len as u32) as u16;
        next_code[len] = next_code[len].wrapping_add(1);
    }
}

/// `tr_static_init`.
fn tables() -> &'static Tables {
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let mut t = Tables { static_ltree: [Ct::default(); L_CODES + 2], static_dtree: [Ct::default(); D_CODES], dist_code: [0; 512],
                             length_code: [0; MAX_MATCH - MIN_MATCH + 1], base_length: [0; LENGTH_CODES], base_dist: [0; D_CODES] };
        let mut length = 0usize;
        let mut code = 0usize;
        while code < LENGTH_CODES - 1 {
            t.base_length[code] = length as u32;
            for _ in 0..(1 << EXTRA_LBITS[code]) { t.length_code[length] = code as u8; length += 1; }
            code += 1;
        }
        t.length_code[length - 1] = code as u8;
        let mut dist = 0usize;
        code = 0;
        while code < 16 {
            t.base_dist[code] = dist as u32;
            for _ in 0..(1 << EXTRA_DBITS[code]) { t.dist_code[dist] = code as u8; dist += 1; }
            code += 1;
        }
        dist >>= 7;
        while code < D_CODES {
            t.base_dist[code] = (dist << 7) as u32;
            for _ in 0..(1 << (EXTRA_DBITS[code] - 7)) { t.dist_code[256 + dist] = code as u8; dist += 1; }
            code += 1;
        }
        let mut bl_count = [0u16; MAX_BITS + 1];
        for (n, ct) in t.static_ltree.iter_mut().enumerate() {
            ct.dl = match n { 0..=143 => 8, 144..=255 => 9, 256..=279 => 7, _ => 8 };
            bl_count[ct.dl as usize] += 1;
        }
        gen_codes(&mut t.static_ltree, (L_CODES + 1) as i32, &bl_count);
        for (n, ct) in t.static_dtree.iter_mut().enumerate() { ct.dl = 5; ct.fc = bi_reverse(n as u32, 5) as u16; }
        t
    })
}

fn d_code(dist: usize, t: &Tables) -> usize { if dist < 256 { t.dist_code[dist] as usize } else { t.dist_code[256 + (dist >> 7)] as usize } }

/// `static_tree_desc`.
struct StaticDesc { stree: Option<&'static [Ct]>, extra: &'static [u32], base: usize, elems: usize, max_length: usize }

fn l_desc() -> StaticDesc { StaticDesc { stree: Some(&tables().static_ltree), extra: &EXTRA_LBITS, base: LITERALS + 1, elems: L_CODES, max_length: MAX_BITS } }
fn d_desc() -> StaticDesc { StaticDesc { stree: Some(&tables().static_dtree), extra: &EXTRA_DBITS, base: 0, elems: D_CODES, max_length: MAX_BITS } }
fn bl_desc() -> StaticDesc { StaticDesc { stree: None, extra: &EXTRA_BLBITS, base: 0, elems: BL_CODES, max_length: MAX_BL_BITS } }

/// The heap and the bit-length statistics one tree construction uses.
struct Builder {
    heap: [i32; HEAP_SIZE],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; HEAP_SIZE],
    bl_count: [u16; MAX_BITS + 1],
    opt_len: i64,
    static_len: i64,
}

impl Builder {
    fn smaller(&self, tree: &[Ct], n: usize, m: usize) -> bool {
        tree[n].fc < tree[m].fc || (tree[n].fc == tree[m].fc && self.depth[n] <= self.depth[m])
    }

    fn pqdownheap(&mut self, tree: &[Ct], mut k: usize) {
        let v = self.heap[k] as usize;
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len && self.smaller(tree, self.heap[j + 1] as usize, self.heap[j] as usize) { j += 1; }
            if self.smaller(tree, v, self.heap[j] as usize) { break; }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v as i32;
    }

    fn gen_bitlen(&mut self, tree: &mut [Ct], max_code: i32, desc: &StaticDesc) {
        self.bl_count = [0; MAX_BITS + 1];
        let mut overflow = 0i32;
        tree[self.heap[self.heap_max] as usize].dl = 0;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h] as usize;
            let mut bits = tree[tree[n].dl as usize].dl as usize + 1;
            if bits > desc.max_length { bits = desc.max_length; overflow += 1; }
            tree[n].dl = bits as u16;
            h += 1;
            if n as i32 > max_code { continue; }
            self.bl_count[bits] += 1;
            let xbits = if n >= desc.base { i64::from(desc.extra[n - desc.base]) } else { 0 };
            let f = i64::from(tree[n].fc);
            self.opt_len += f * (bits as i64 + xbits);
            if let Some(stree) = desc.stree { self.static_len += f * (i64::from(stree[n].dl) + xbits); }
        }
        if overflow == 0 { return; }
        loop {
            let mut bits = desc.max_length - 1;
            while self.bl_count[bits] == 0 { bits -= 1; }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[desc.max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 { break; }
        }
        let mut bits = desc.max_length;
        while bits != 0 {
            let mut n = self.bl_count[bits];
            while n != 0 {
                h -= 1;
                let m = self.heap[h] as usize;
                if m as i32 > max_code { continue; }
                if tree[m].dl as usize != bits {
                    self.opt_len += (bits as i64 - i64::from(tree[m].dl)) * i64::from(tree[m].fc);
                    tree[m].dl = bits as u16;
                }
                n -= 1;
            }
            bits -= 1;
        }
    }

    /// `build_tree`: returns the largest code with nonzero frequency.
    fn build_tree(&mut self, tree: &mut [Ct], desc: &StaticDesc) -> i32 {
        let mut max_code: i32 = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for n in 0..desc.elems {
            if tree[n].fc != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n as i32;
                max_code = n as i32;
                self.depth[n] = 0;
            } else {
                tree[n].dl = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 { max_code += 1; max_code as usize } else { 0 };
            self.heap_len += 1;
            self.heap[self.heap_len] = node as i32;
            tree[node].fc = 1;
            self.depth[node] = 0;
            self.opt_len -= 1;
            if let Some(stree) = desc.stree { self.static_len -= i64::from(stree[node].dl); }
        }
        let mut n = self.heap_len / 2;
        while n >= 1 { self.pqdownheap(tree, n); n -= 1; }
        let mut node = desc.elems;
        loop {
            let n = self.heap[1] as usize;
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(tree, 1);
            let m = self.heap[1] as usize;
            self.heap_max -= 1;
            self.heap[self.heap_max] = n as i32;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m as i32;
            tree[node].fc = tree[n].fc.wrapping_add(tree[m].fc);
            self.depth[node] = self.depth[n].max(self.depth[m]).wrapping_add(1);
            tree[n].dl = node as u16;
            tree[m].dl = node as u16;
            self.heap[1] = node as i32;
            node += 1;
            self.pqdownheap(tree, 1);
            if self.heap_len < 2 { break; }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(tree, max_code, desc);
        let counts = self.bl_count;
        gen_codes(tree, max_code, &counts);
        max_code
    }
}

/// `scan_tree`: bit-length code frequencies of one tree.
fn scan_tree(tree: &mut [Ct], bl_tree: &mut [Ct], max_code: i32) {
    let mut prevlen: i32 = -1;
    let mut nextlen = i32::from(tree[0].dl);
    let mut count = 0;
    let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
    tree[(max_code + 1) as usize].dl = 0xffff;
    for n in 0..=max_code as usize {
        let curlen = nextlen;
        nextlen = i32::from(tree[n + 1].dl);
        count += 1;
        if count < max_count && curlen == nextlen { continue; }
        let bump = |slot: &mut Ct, by: u16| slot.fc = slot.fc.wrapping_add(by);
        if count < min_count {
            bump(&mut bl_tree[curlen as usize], count as u16);
        } else if curlen != 0 {
            if curlen != prevlen { bump(&mut bl_tree[curlen as usize], 1); }
            bump(&mut bl_tree[REP_3_6], 1);
        } else if count <= 10 {
            bump(&mut bl_tree[REPZ_3_10], 1);
        } else {
            bump(&mut bl_tree[REPZ_11_138], 1);
        }
        count = 0;
        prevlen = curlen;
        (max_count, min_count) = if nextlen == 0 { (138, 3) } else if curlen == nextlen { (6, 3) } else { (7, 4) };
    }
}

#[derive(PartialEq)]
enum Flush { None, Finish }

/// One raw deflate stream at level 1.
pub struct Deflater {
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    ins_h: u32,
    strstart: usize,
    block_start: isize,
    lookahead: usize,
    insert: usize,
    match_start: usize,
    match_length: usize,
    high_water: usize,
    dyn_ltree: [Ct; HEAP_SIZE],
    dyn_dtree: [Ct; 2 * D_CODES + 1],
    bl_tree: [Ct; 2 * BL_CODES + 1],
    builder: Builder,
    sym_buf: Vec<u8>,
    sym_next: usize,
    bi_buf: u16,
    bi_valid: u32,
    out: Vec<u8>,
}

impl Default for Deflater { fn default() -> Self { Self::new() } }

impl Deflater {
    pub fn new() -> Self {
        let mut deflater = Self {
            window: vec![0; WINDOW_SIZE], prev: vec![0; W_SIZE], head: vec![0; HASH_SIZE], ins_h: 0, strstart: 0, block_start: 0,
            lookahead: 0, insert: 0, match_start: 0, match_length: MIN_MATCH - 1, high_water: 0,
            dyn_ltree: [Ct::default(); HEAP_SIZE], dyn_dtree: [Ct::default(); 2 * D_CODES + 1], bl_tree: [Ct::default(); 2 * BL_CODES + 1],
            builder: Builder { heap: [0; HEAP_SIZE], heap_len: 0, heap_max: 0, depth: [0; HEAP_SIZE], bl_count: [0; MAX_BITS + 1], opt_len: 0, static_len: 0 },
            sym_buf: vec![0; LIT_BUFSIZE * 3], sym_next: 0, bi_buf: 0, bi_valid: 0, out: Vec::new(),
        };
        deflater.init_block();
        deflater
    }

    /// The compressed bytes produced so far.
    pub fn take_output(&mut self) -> Vec<u8> { std::mem::take(&mut self.out) }

    /// `compress(data)`: deflate with `Z_NO_FLUSH`.
    pub fn compress(&mut self, data: &[u8]) {
        if data.is_empty() { return; }
        let mut input = data;
        self.deflate_fast(&mut input, Flush::None);
    }

    /// `flush()`: deflate with `Z_FINISH`; the bytes not yet taken.
    pub fn finish(mut self) -> Vec<u8> {
        let mut input: &[u8] = &[];
        self.deflate_fast(&mut input, Flush::Finish);
        self.out
    }

    fn update_hash(h: u32, c: u8) -> u32 { ((h << HASH_SHIFT) ^ u32::from(c)) & HASH_MASK }

    fn slide_hash(&mut self) {
        for entry in self.head.iter_mut().chain(self.prev.iter_mut()) {
            let m = *entry as usize;
            *entry = if m >= W_SIZE { (m - W_SIZE) as u16 } else { 0 };
        }
    }

    fn fill_window(&mut self, input: &mut &[u8]) {
        loop {
            let mut more = WINDOW_SIZE - self.lookahead - self.strstart;
            if self.strstart >= W_SIZE + MAX_DIST {
                self.window.copy_within(W_SIZE..W_SIZE + W_SIZE - more, 0);
                self.match_start = self.match_start.wrapping_sub(W_SIZE);
                self.strstart -= W_SIZE;
                self.block_start -= W_SIZE as isize;
                if self.insert > self.strstart { self.insert = self.strstart; }
                self.slide_hash();
                more += W_SIZE;
            }
            if input.is_empty() { break; }
            let n = more.min(input.len());
            let at = self.strstart + self.lookahead;
            self.window[at..at + n].copy_from_slice(&input[..n]);
            *input = &input[n..];
            self.lookahead += n;
            if self.lookahead + self.insert >= MIN_MATCH {
                let mut str = self.strstart - self.insert;
                self.ins_h = u32::from(self.window[str]);
                self.ins_h = Self::update_hash(self.ins_h, self.window[str + 1]);
                while self.insert != 0 {
                    self.ins_h = Self::update_hash(self.ins_h, self.window[str + MIN_MATCH - 1]);
                    self.prev[str & W_MASK] = self.head[self.ins_h as usize];
                    self.head[self.ins_h as usize] = str as u16;
                    str += 1;
                    self.insert -= 1;
                    if self.lookahead + self.insert < MIN_MATCH { break; }
                }
            }
            if !(self.lookahead < MIN_LOOKAHEAD && !input.is_empty()) { break; }
        }
        if self.high_water < WINDOW_SIZE {
            let curr = self.strstart + self.lookahead;
            if self.high_water < curr {
                let init = (WINDOW_SIZE - curr).min(WIN_INIT);
                self.window[curr..curr + init].fill(0);
                self.high_water = curr + init;
            } else if self.high_water < curr + WIN_INIT {
                let init = (curr + WIN_INIT - self.high_water).min(WINDOW_SIZE - self.high_water);
                self.window[self.high_water..self.high_water + init].fill(0);
                self.high_water += init;
            }
        }
    }

    fn insert_string(&mut self, str: usize) -> usize {
        self.ins_h = Self::update_hash(self.ins_h, self.window[str + MIN_MATCH - 1]);
        let head = self.head[self.ins_h as usize];
        self.prev[str & W_MASK] = head;
        self.head[self.ins_h as usize] = str as u16;
        head as usize
    }

    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let mut chain_length = MAX_CHAIN;
        let scan = self.strstart;
        let mut best_len = MIN_MATCH - 1;
        let limit = if self.strstart > MAX_DIST { self.strstart - MAX_DIST } else { 0 };
        let strend = scan + MAX_MATCH;
        let window = &self.window;
        let mut scan_end1 = window[scan + best_len - 1];
        let mut scan_end = window[scan + best_len];
        // prev_length (always MIN_MATCH - 1 at level 1) never reaches good_match.
        if MIN_MATCH - 1 >= GOOD_MATCH { chain_length >>= 2; }
        let nice_match = NICE_MATCH.min(self.lookahead);
        loop {
            let m = cur_match;
            if window[m + best_len] == scan_end && window[m + best_len - 1] == scan_end1 && window[m] == window[scan] && window[m + 1] == window[scan + 1] {
                // scan[2] == match[2] by the hash; compare from scan + 3, eight per bounds check.
                let mut s = scan + 2;
                let mut q = m + 2;
                'compare: loop {
                    for _ in 0..8 {
                        s += 1;
                        q += 1;
                        if window[s] != window[q] { break 'compare; }
                    }
                    if s >= strend { break; }
                }
                let len = MAX_MATCH - (strend - s);
                if len > best_len {
                    self.match_start = cur_match;
                    best_len = len;
                    if len >= nice_match { break; }
                    scan_end1 = window[scan + best_len - 1];
                    scan_end = window[scan + best_len];
                }
            }
            cur_match = self.prev[cur_match & W_MASK] as usize;
            if cur_match <= limit { break; }
            chain_length -= 1;
            if chain_length == 0 { break; }
        }
        best_len.min(self.lookahead)
    }

    fn deflate_fast(&mut self, input: &mut &[u8], flush: Flush) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window(input);
                if self.lookahead < MIN_LOOKAHEAD && flush == Flush::None { return; }
                if self.lookahead == 0 { break; }
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH { hash_head = self.insert_string(self.strstart); }
            if hash_head != 0 && self.strstart - hash_head <= MAX_DIST { self.match_length = self.longest_match(hash_head); }
            let bflush = if self.match_length >= MIN_MATCH {
                let bflush = self.tally_dist(self.strstart - self.match_start, self.match_length - MIN_MATCH);
                self.lookahead -= self.match_length;
                if self.match_length <= MAX_INSERT_LENGTH && self.lookahead >= MIN_MATCH {
                    self.match_length -= 1;
                    loop {
                        self.strstart += 1;
                        self.insert_string(self.strstart);
                        self.match_length -= 1;
                        if self.match_length == 0 { break; }
                    }
                    self.strstart += 1;
                } else {
                    self.strstart += self.match_length;
                    self.match_length = 0;
                    self.ins_h = u32::from(self.window[self.strstart]);
                    self.ins_h = Self::update_hash(self.ins_h, self.window[self.strstart + 1]);
                }
                bflush
            } else {
                let bflush = self.tally_lit(self.window[self.strstart]);
                self.lookahead -= 1;
                self.strstart += 1;
                bflush
            };
            if bflush { self.flush_block(false); }
        }
        // Only Z_FINISH leaves the loop: Z_NO_FLUSH returned for more input above.
        self.insert = self.strstart.min(MIN_MATCH - 1);
        self.flush_block(true);
    }

    /// `FLUSH_BLOCK_ONLY` (all pending output is taken at once).
    fn flush_block(&mut self, last: bool) {
        let stored = (self.block_start >= 0).then_some(self.block_start as usize);
        let stored_len = (self.strstart as isize - self.block_start) as usize;
        self.tr_flush_block(stored, stored_len, last);
        self.block_start = self.strstart as isize;
        self.bi_flush();
    }

    fn init_block(&mut self) {
        for ct in &mut self.dyn_ltree[..L_CODES] { ct.fc = 0; }
        for ct in &mut self.dyn_dtree[..D_CODES] { ct.fc = 0; }
        for ct in &mut self.bl_tree[..BL_CODES] { ct.fc = 0; }
        self.dyn_ltree[END_BLOCK].fc = 1;
        self.builder.opt_len = 0;
        self.builder.static_len = 0;
        self.sym_next = 0;
    }

    fn tally_lit(&mut self, c: u8) -> bool {
        self.sym_buf[self.sym_next..self.sym_next + 3].copy_from_slice(&[0, 0, c]);
        self.sym_next += 3;
        let slot = &mut self.dyn_ltree[c as usize];
        slot.fc = slot.fc.wrapping_add(1);
        self.sym_next == SYM_END
    }

    fn tally_dist(&mut self, distance: usize, length: usize) -> bool {
        let t = tables();
        let dist = distance as u16;
        self.sym_buf[self.sym_next..self.sym_next + 3].copy_from_slice(&[dist as u8, (dist >> 8) as u8, length as u8]);
        self.sym_next += 3;
        let l = &mut self.dyn_ltree[t.length_code[length] as usize + LITERALS + 1];
        l.fc = l.fc.wrapping_add(1);
        let d = &mut self.dyn_dtree[d_code(dist as usize - 1, t)];
        d.fc = d.fc.wrapping_add(1);
        self.sym_next == SYM_END
    }

    fn send_bits(&mut self, value: u32, length: u32) {
        if self.bi_valid > 16 - length {
            self.bi_buf |= (value << self.bi_valid) as u16;
            self.out.extend_from_slice(&self.bi_buf.to_le_bytes());
            self.bi_buf = (value >> (16 - self.bi_valid)) as u16;
            self.bi_valid += length - 16;
        } else {
            self.bi_buf |= (value << self.bi_valid) as u16;
            self.bi_valid += length;
        }
    }

    fn send_code(&mut self, ct: Ct) { self.send_bits(u32::from(ct.fc), u32::from(ct.dl)); }

    fn bi_flush(&mut self) {
        if self.bi_valid == 16 {
            self.out.extend_from_slice(&self.bi_buf.to_le_bytes());
            self.bi_buf = 0;
            self.bi_valid = 0;
        } else if self.bi_valid >= 8 {
            self.out.push(self.bi_buf as u8);
            self.bi_buf >>= 8;
            self.bi_valid -= 8;
        }
    }

    fn bi_windup(&mut self) {
        if self.bi_valid > 8 { self.out.extend_from_slice(&self.bi_buf.to_le_bytes()); } else if self.bi_valid > 0 { self.out.push(self.bi_buf as u8); }
        self.bi_buf = 0;
        self.bi_valid = 0;
    }

    fn send_tree(&mut self, dynamic_l: bool, max_code: i32) {
        let len = |this: &Self, n: usize| i32::from(if dynamic_l { this.dyn_ltree[n].dl } else { this.dyn_dtree[n].dl });
        let mut prevlen: i32 = -1;
        let mut nextlen = len(self, 0);
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        for n in 0..=max_code as usize {
            let curlen = nextlen;
            nextlen = len(self, n + 1);
            count += 1;
            if count < max_count && curlen == nextlen { continue; }
            if count < min_count {
                loop { self.send_code(self.bl_tree[curlen as usize]); count -= 1; if count == 0 { break; } }
            } else if curlen != 0 {
                if curlen != prevlen { self.send_code(self.bl_tree[curlen as usize]); count -= 1; }
                self.send_code(self.bl_tree[REP_3_6]);
                self.send_bits((count - 3) as u32, 2);
            } else if count <= 10 {
                self.send_code(self.bl_tree[REPZ_3_10]);
                self.send_bits((count - 3) as u32, 3);
            } else {
                self.send_code(self.bl_tree[REPZ_11_138]);
                self.send_bits((count - 11) as u32, 7);
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 { (138, 3) } else if curlen == nextlen { (6, 3) } else { (7, 4) };
        }
    }

    fn compress_block(&mut self, dynamic: bool) {
        let t = tables();
        let l = |this: &Self, c: usize| if dynamic { this.dyn_ltree[c] } else { t.static_ltree[c] };
        let d = |this: &Self, c: usize| if dynamic { this.dyn_dtree[c] } else { t.static_dtree[c] };
        let mut sx = 0;
        while sx < self.sym_next {
            let mut dist = self.sym_buf[sx] as usize | ((self.sym_buf[sx + 1] as usize) << 8);
            let mut lc = self.sym_buf[sx + 2] as usize;
            sx += 3;
            if dist == 0 {
                self.send_code(l(self, lc));
            } else {
                let code = t.length_code[lc] as usize;
                self.send_code(l(self, code + LITERALS + 1));
                let extra = EXTRA_LBITS[code];
                if extra != 0 { lc -= t.base_length[code] as usize; self.send_bits(lc as u32, extra); }
                dist -= 1;
                let code = d_code(dist, t);
                self.send_code(d(self, code));
                let extra = EXTRA_DBITS[code];
                if extra != 0 { dist -= t.base_dist[code] as usize; self.send_bits(dist as u32, extra); }
            }
        }
        self.send_code(l(self, END_BLOCK));
    }

    fn tr_flush_block(&mut self, buf: Option<usize>, stored_len: usize, last: bool) {
        let l_max = self.builder.build_tree(&mut self.dyn_ltree, &l_desc());
        let d_max = self.builder.build_tree(&mut self.dyn_dtree, &d_desc());
        scan_tree(&mut self.dyn_ltree, &mut self.bl_tree, l_max);
        scan_tree(&mut self.dyn_dtree, &mut self.bl_tree, d_max);
        self.builder.build_tree(&mut self.bl_tree, &bl_desc());
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 && self.bl_tree[BL_ORDER[max_blindex]].dl == 0 { max_blindex -= 1; }
        self.builder.opt_len += 3 * (max_blindex as i64 + 1) + 5 + 5 + 4;
        let mut opt_lenb = (self.builder.opt_len + 3 + 7) >> 3;
        let static_lenb = (self.builder.static_len + 3 + 7) >> 3;
        if static_lenb <= opt_lenb { opt_lenb = static_lenb; }
        match buf {
            Some(start) if stored_len as i64 + 4 <= opt_lenb => {
                self.send_bits(u32::from(last), 3);
                self.bi_windup();
                self.out.extend_from_slice(&(stored_len as u16).to_le_bytes());
                self.out.extend_from_slice(&(!(stored_len as u16)).to_le_bytes());
                self.out.extend_from_slice(&self.window[start..start + stored_len]);
            }
            _ if static_lenb == opt_lenb => {
                self.send_bits(2 + u32::from(last), 3);
                self.compress_block(false);
            }
            _ => {
                self.send_bits(4 + u32::from(last), 3);
                self.send_bits((l_max + 1 - 257) as u32, 5);
                self.send_bits(d_max as u32, 5);
                self.send_bits((max_blindex + 1 - 4) as u32, 4);
                for rank in 0..=max_blindex { let len = self.bl_tree[BL_ORDER[rank]].dl; self.send_bits(u32::from(len), 3); }
                self.send_tree(true, l_max);
                self.send_tree(false, d_max);
                self.compress_block(true);
            }
        }
        self.init_block();
        if last { self.bi_windup(); }
    }
}

/// `len(zlib.compress(data, 1))`: the raw stream plus the 2-byte zlib header
/// and the 4-byte Adler-32 trailer.
pub fn zlib_compressed_len(data: &[u8]) -> usize {
    let mut deflater = Deflater::new();
    deflater.compress(data);
    deflater.finish().len() + 6
}
