//! Content names at the ABI boundary: `canonical_digest(value)` is the 128-bit
//! BLAKE2b of the compact, key-sorted UTF-8 JSON of a value. A board answers
//! with the digest of the program it holds and an editor digests what it would
//! compile to now; equal names mean the board plays what the screen shows.
use crate::json::Json;

/// `canonical_digest(tree)`: lowercase hex of BLAKE2b with a 16-byte digest.
pub fn digest(tree: &Json) -> String {
    blake2b(tree.canonical().as_bytes(), 16).iter().map(|byte| format!("{byte:02x}")).collect()
}

const IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// BLAKE2b (RFC 7693) without a key, `out_len` bytes of digest.
pub fn blake2b(data: &[u8], out_len: usize) -> Vec<u8> {
    assert!((1..=64).contains(&out_len));
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ out_len as u64;
    let blocks = if data.is_empty() { 1 } else { data.len().div_ceil(128) };
    for index in 0..blocks {
        let start = index * 128;
        let end = (start + 128).min(data.len());
        let mut block = [0u8; 128];
        block[..end - start].copy_from_slice(&data[start..end]);
        let last = index + 1 == blocks;
        compress(&mut h, &block, end as u128, last);
    }
    h.iter().flat_map(|word| word.to_le_bytes()).take(out_len).collect()
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], counter: u128, last: bool) {
    let m: [u64; 16] = std::array::from_fn(|i| u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap()));
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= counter as u64;
    v[13] ^= (counter >> 64) as u64;
    if last { v[14] = !v[14]; }
    for s in &SIGMA {
        mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 { h[i] ^= v[i] ^ v[i + 8]; }
}

fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake2b_known_answers() {
        // hashlib.blake2b(b"abc", digest_size=16) and the RFC 7693 64-byte vector.
        let hex = |bytes: Vec<u8>| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(blake2b(b"abc", 64)), "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
                                              7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923");
        assert_eq!(hex(blake2b(b"", 16)), "cae66941d9efbd404e4d88758ea67670");
    }
}
