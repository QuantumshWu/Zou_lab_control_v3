//! The embedded PTX (src/gpu/ptx.rs), deflated: kernels/motion3d.ptx with LF line endings (whatever
//! the checkout made them), compressed into OUT_DIR, about a fifth of its size in the EXE.
fn main(){
    let ptx=std::fs::read_to_string("kernels/motion3d.ptx").expect("kernels/motion3d.ptx");
    let packed=miniz_oxide::deflate::compress_to_vec(ptx.replace("\r\n","\n").as_bytes(),10);
    std::fs::write(std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("motion3d.ptx.deflate"),packed).unwrap();
    println!("cargo:rerun-if-changed=kernels/motion3d.ptx");
}
