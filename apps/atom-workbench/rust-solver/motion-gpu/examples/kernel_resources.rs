//! Registers and local (spill) bytes per thread of every kernel: `cargo run --profile iterate --example kernel_resources`.
use atom_motion_gpu::gpu::Gpu;

fn main(){
    let gpu=Gpu::new().unwrap();
    let names:Vec<String>=atom_motion_gpu::ptx::PTX.lines().filter_map(|l|l.strip_prefix(".visible .entry ").map(|n|n.trim_end_matches('(').to_string())).collect();
    for name in names{let f=gpu.function(&name).unwrap();let(r,l)=f.resources().unwrap();println!("{name:28} registers {r:3}  local {l:5} B");}
}
