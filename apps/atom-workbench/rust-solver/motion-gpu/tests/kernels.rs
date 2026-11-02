//! The kernels against double-precision references on the CPU (needs an NVIDIA GPU).
use atom_motion_core::Complex;
use atom_motion_gpu::driver::{self,Buffer};
use atom_motion_gpu::gpu::Gpu;

fn uniform(seed:u64)->impl FnMut()->f64{
    let mut state=seed;move||{state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(state>>11) as f64/(1u64<<53) as f64-0.5}
}
fn bitrev(j:usize,n:usize)->usize{let bits=n.trailing_zeros();if bits==0{0}else{j.reverse_bits()>>(usize::BITS-bits)}}
/// The DFT of a sequence (sign −1 forward, +1 inverse, unnormalized).
fn dft(x:&[Complex],sign:f64)->Vec<Complex>{
    let n=x.len();(0..n).map(|k|x.iter().enumerate().fold(Complex::ZERO,|a,(j,&v)|a+v*Complex::cis(sign*std::f64::consts::TAU*(j*k%n) as f64/n as f64))).collect()
}
/// Multiplies the 2D spectrum of a plane (ny rows of nx) by f(kx index) g(ky index), exactly.
fn spectral(plane:&[Complex],nx:usize,ny:usize,fx:&[Complex],fy:&[Complex])->Vec<Complex>{
    let mut p=plane.to_vec();
    for r in 0..ny{let row=dft(&p[r*nx..(r+1)*nx],-1.);let row:Vec<Complex>=row.iter().zip(fx).map(|(&a,&b)|a*b).collect();let back=dft(&row,1.);
        for x in 0..nx{p[r*nx+x]=back[x]*(1./nx as f64);}}
    for c in 0..nx{let col:Vec<Complex>=(0..ny).map(|r|p[r*nx+c]).collect();let s=dft(&col,-1.);let s:Vec<Complex>=s.iter().zip(fy).map(|(&a,&b)|a*b).collect();
        let back=dft(&s,1.);for r in 0..ny{p[r*nx+c]=back[r]*(1./ny as f64);}}
    p
}
fn to_f32(v:&[Complex])->Vec<[f32;2]>{v.iter().map(|z|[z.re as f32,z.im as f32]).collect()}

/// The driver lists the devices, and a kernel round trip works: `thermal_potential` scales a
/// trajectory's first plane point by point and leaves its other planes.
#[test]
fn gpu_devices_and_a_kernel_round_trip(){
    println!("driver CUDA {}",driver::driver_version().unwrap());
    for d in driver::devices().unwrap(){println!("{d:?}");}
    let gpu=Gpu::new().unwrap();
    let f=gpu.function("thermal_potential").unwrap();
    let points=1000usize;
    let input:Vec<[f32;2]>=(0..5*points).map(|i|[i as f32,-(i as f32)]).collect();
    let state=Buffer::new(5*points*8).unwrap();gpu.stream.upload(&state,&input).unwrap();
    let factors=Buffer::new(points*4).unwrap();gpu.stream.upload(&factors,&vec![2.5f32;points]).unwrap();
    let kind=Buffer::new(4).unwrap();gpu.stream.upload(&kind,&[0i32]).unwrap();
    gpu.stream.launch(&f,(points.div_ceil(256) as u32,1,1),256,0,&[&state.pointer,&(points as i32),&1i32,&factors.pointer,&kind.pointer]).unwrap();
    let mut out=vec![[0f32;2];5*points];gpu.stream.download(&state,&mut out).unwrap();
    assert!(out.iter().enumerate().all(|(i,z)|{let s=if i<points{2.5}else{1.};z[0]==s*i as f32&&z[1]==-s*i as f32}));
}

/// The kinetic pass multiplies each plane's spectrum by its tables -- here arbitrary unit phases
/// -- and renormalizes by the trajectory's scale, skipping a lost trajectory and an F = 1 one's
/// last two components: against the exact transform, to single precision.
#[test]
fn gpu_kinetic_pass_matches_the_exact_spectral_product(){
    let gpu=Gpu::new().unwrap();
    for(nx,ny)in [(16,16),(32,32),(64,64),(64,1),(128,1),(512,1)]{
        let f=gpu.function(&format!("kinetic_{nx}x{ny}")).unwrap();
        let shared=ny*(nx+1)*8;f.allow_shared(shared).unwrap();
        let mut next=uniform(nx as u64*7+ny as u64);
        let trajectories=3;let components=[5i32,3,0];let scales=[1.25f32,0.75,2.];
        let planes=trajectories*5;let points=nx*ny;
        let data:Vec<Complex>=(0..planes*points).map(|_|Complex::new(next(),next())).collect();
        let phase=|n:usize,next:&mut dyn FnMut()->f64|->Vec<Complex>{(0..n).map(|_|Complex::cis(20.*next())).collect()};
        // A single row has no y transform: its one frequency, zero, keeps its phase.
        let(fx,fy)=(phase(nx,&mut next),if ny==1{vec![Complex::ONE]}else{phase(ny,&mut next)});
        // The tables in bit-reversed order, with the inverse transform's 1/N.
        let table=|f:&[Complex]|->Vec<[f32;2]>{let n=f.len();(0..n).map(|j|{let z=f[bitrev(j,n)]*(1./n as f64);[z.re as f32,z.im as f32]}).collect()};
        let state=Buffer::new(planes*points*8).unwrap();gpu.stream.upload(&state,&to_f32(&data)).unwrap();
        let(cb,sb,xb,yb)=(Buffer::new(12).unwrap(),Buffer::new(12).unwrap(),Buffer::new(nx*8).unwrap(),Buffer::new(ny*8).unwrap());
        gpu.stream.upload(&cb,&components).unwrap();gpu.stream.upload(&sb,&scales).unwrap();
        gpu.stream.upload(&xb,&table(&fx)).unwrap();gpu.stream.upload(&yb,&table(&fy)).unwrap();
        gpu.stream.launch(&f,(planes as u32,1,1),256,shared,&[&state.pointer,&cb.pointer,&sb.pointer,&xb.pointer,&yb.pointer]).unwrap();
        let mut out=vec![[0f32;2];planes*points];gpu.stream.download(&state,&mut out).unwrap();
        let mut worst=0f64;
        for p in 0..planes{
            let(t,c)=(p/5,p%5);let plane=&data[p*points..(p+1)*points];
            let expected:Vec<Complex>=if (c as i32)<components[t]{spectral(plane,nx,ny,&fx,&fy).iter().map(|&z|z*(scales[t] as f64)).collect()}else{plane.to_vec()};
            let scale=expected.iter().fold(0f64,|a,z|a.max(z.abs()));
            for(z,o)in expected.iter().zip(&out[p*points..(p+1)*points]){worst=worst.max((Complex::new(o[0] as f64,o[1] as f64)-*z).abs()/scale);}
        }
        println!("{nx}x{ny}: largest error {worst:.2e} of the largest value");
        assert!(worst<2e-6,"{nx}x{ny}: {worst:e}");
    }
}

/// The large grids' two passes (rows, then a tile of columns): the same spectral product.
#[test]
fn gpu_large_grid_passes_match_the_exact_spectral_product(){
    let gpu=Gpu::new().unwrap();
    for n in [128usize,256]{
        let rows=gpu.function(&format!("kinetic_rows_{n}")).unwrap();let cols=gpu.function(&format!("kinetic_cols_{n}")).unwrap();
        let shared=n*9*8;cols.allow_shared(shared).unwrap();
        let mut next=uniform(n as u64);
        let components=[3i32];let scales=[1.5f32];let planes=5;let points=n*n;
        let data:Vec<Complex>=(0..planes*points).map(|_|Complex::new(next(),next())).collect();
        let phase=|next:&mut dyn FnMut()->f64|->Vec<Complex>{(0..n).map(|_|Complex::cis(20.*next())).collect()};
        let(fx,fy)=(phase(&mut next),phase(&mut next));
        let table=|f:&[Complex]|->Vec<[f32;2]>{(0..n).map(|j|{let z=f[bitrev(j,n)]*(1./n as f64);[z.re as f32,z.im as f32]}).collect()};
        let state=Buffer::new(planes*points*8).unwrap();gpu.stream.upload(&state,&to_f32(&data)).unwrap();
        let(cb,sb,xb,yb)=(Buffer::new(4).unwrap(),Buffer::new(4).unwrap(),Buffer::new(n*8).unwrap(),Buffer::new(n*8).unwrap());
        gpu.stream.upload(&cb,&components).unwrap();gpu.stream.upload(&sb,&scales).unwrap();gpu.stream.upload(&xb,&table(&fx)).unwrap();gpu.stream.upload(&yb,&table(&fy)).unwrap();
        gpu.stream.launch(&rows,((planes*n/8) as u32,1,1),256,0,&[&state.pointer,&cb.pointer,&sb.pointer,&xb.pointer]).unwrap();
        gpu.stream.launch(&cols,((planes*n/8) as u32,1,1),256,shared,&[&state.pointer,&cb.pointer,&yb.pointer]).unwrap();
        let mut out=vec![[0f32;2];planes*points];gpu.stream.download(&state,&mut out).unwrap();
        let mut worst=0f64;
        for p in 0..planes{
            let plane=&data[p*points..(p+1)*points];
            let expected:Vec<Complex>=if (p as i32)<components[0]{spectral(plane,n,n,&fx,&fy).iter().map(|&z|z*1.5).collect()}else{plane.to_vec()};
            let scale=expected.iter().fold(0f64,|a,z|a.max(z.abs()));
            for(z,o)in expected.iter().zip(&out[p*points..(p+1)*points]){worst=worst.max((Complex::new(o[0] as f64,o[1] as f64)-*z).abs()/scale);}
        }
        println!("{n}x{n}: largest error {worst:.2e} of the largest value");
        assert!(worst<2e-6,"{n}x{n}: {worst:e}");
    }
}
