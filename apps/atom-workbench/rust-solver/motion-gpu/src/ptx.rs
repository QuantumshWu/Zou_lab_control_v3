//! The engine's kernels: `kernels/motion.cu`, compiled once at development time by a pinned NVRTC
//! (`scripts/ptx.ps1`, `examples/compile_ptx.rs`) to PTX for compute capability 7.5, which the
//! driver's JIT compiles for the GPU it runs on. The PTX is committed and embedded; its header
//! records the SHA-256 of the options and the source it was made from, which a test checks.

/// The embedded PTX.
pub const PTX:&str=include_str!("../kernels/motion.ptx");
/// The source, embedded only to check the PTX against it.
pub const SOURCE:&str=include_str!("../kernels/motion.cu");
/// NVRTC's options: PTX for compute capability 7.5 (Turing, the oldest GPU the engine takes) with
/// IEEE single precision (no fast math).
pub const NVRTC_OPTIONS:&[&str]=&["--gpu-architecture=compute_75","--std=c++17","-default-device","--extra-device-vectorization"];
/// The pinned NVRTC: the PyPI wheel nvidia-cuda-nvrtc-cu12 of this version (its PTX ISA 8.0 loads
/// on every driver of CUDA 12.0 or later, 527.41 on Windows).
pub const NVRTC_VERSION:&str="12.0.140";

/// The hash a PTX header records: of the options, one per line, then the source, with its line
/// endings as LF (a checkout may have made them CRLF).
pub fn source_hash(source:&str)->String{
    let mut text=String::new();for o in NVRTC_OPTIONS{text.push_str(o);text.push('\n');}text.push_str(&source.replace("\r\n","\n"));
    sha256(text.as_bytes()).iter().map(|b|format!("{b:02x}")).collect()
}
/// The PTX header's line for a source.
pub fn header(source:&str)->String{format!("// atom-motion-gpu PTX from kernels/motion.cu by NVRTC {NVRTC_VERSION} ({}); source sha256 {}\n",NVRTC_OPTIONS.join(" "),source_hash(source))}

/// SHA-256 (FIPS 180-4).
pub fn sha256(data:&[u8])->[u8;32]{
    const K:[u32;64]=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,
        0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
        0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    let mut h:[u32;8]=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let mut message=data.to_vec();message.push(0x80);while message.len()%64!=56{message.push(0);}
    message.extend_from_slice(&((data.len() as u64)*8).to_be_bytes());
    for chunk in message.chunks(64){
        let mut w=[0u32;64];
        for i in 0..16{w[i]=u32::from_be_bytes([chunk[4*i],chunk[4*i+1],chunk[4*i+2],chunk[4*i+3]]);}
        for i in 16..64{
            let s0=w[i-15].rotate_right(7)^w[i-15].rotate_right(18)^(w[i-15]>>3);let s1=w[i-2].rotate_right(17)^w[i-2].rotate_right(19)^(w[i-2]>>10);
            w[i]=w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let mut v=h;
        for i in 0..64{
            let s1=v[4].rotate_right(6)^v[4].rotate_right(11)^v[4].rotate_right(25);let ch=(v[4]&v[5])^(!v[4]&v[6]);
            let t1=v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0=v[0].rotate_right(2)^v[0].rotate_right(13)^v[0].rotate_right(22);let maj=(v[0]&v[1])^(v[0]&v[2])^(v[1]&v[2]);
            let t2=s0.wrapping_add(maj);
            v=[t1.wrapping_add(t2),v[0],v[1],v[2],v[3].wrapping_add(t1),v[4],v[5],v[6]];
        }
        for i in 0..8{h[i]=h[i].wrapping_add(v[i]);}
    }
    let mut out=[0u8;32];for i in 0..8{out[4*i..4*i+4].copy_from_slice(&h[i].to_be_bytes());}out
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn sha256_matches_the_standard_vectors(){
        let hex=|d:&[u8]|sha256(d).iter().map(|b|format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(b""),"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(b"abc"),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }
    /// The committed PTX was made from the committed source with the pinned options: run
    /// scripts/ptx.ps1 after editing kernels/motion.cu.
    #[test]
    fn the_ptx_is_compiled_from_the_source(){
        let first=PTX.lines().next().unwrap_or("");
        assert_eq!(first,header(SOURCE).trim_end(),"kernels/motion.ptx is stale: run motion-gpu/scripts/ptx.ps1");
    }
}
