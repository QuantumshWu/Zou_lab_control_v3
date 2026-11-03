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
    for(nx,ny)in [(16,16),(32,32),(64,64),(16,256),(256,16),(32,128),(512,1)]{
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

/// The large grids' two passes (rows, then a tile of columns): the same spectral product, square
/// and long narrow grids alike.
#[test]
fn gpu_large_grid_passes_match_the_exact_spectral_product(){
    let gpu=Gpu::new().unwrap();
    for(nx,ny)in [(128usize,128usize),(256,256),(32,256),(256,64)]{
        let rows=gpu.function(&format!("kinetic_rows_{nx}x{ny}")).unwrap();let cols=gpu.function(&format!("kinetic_cols_{nx}x{ny}")).unwrap();
        // A tile of 8 columns (engine.rs TILE) in shared memory, its rows padded by one.
        let shared=ny*9*8;cols.allow_shared(shared).unwrap();
        let mut next=uniform((nx*ny) as u64);
        let components=[3i32];let scales=[1.5f32];let planes=5;let points=nx*ny;
        let data:Vec<Complex>=(0..planes*points).map(|_|Complex::new(next(),next())).collect();
        let phase=|n:usize,next:&mut dyn FnMut()->f64|->Vec<Complex>{(0..n).map(|_|Complex::cis(20.*next())).collect()};
        let(fx,fy)=(phase(nx,&mut next),phase(ny,&mut next));
        let table=|f:&[Complex]|->Vec<[f32;2]>{let n=f.len();(0..n).map(|j|{let z=f[bitrev(j,n)]*(1./n as f64);[z.re as f32,z.im as f32]}).collect()};
        let state=Buffer::new(planes*points*8).unwrap();gpu.stream.upload(&state,&to_f32(&data)).unwrap();
        let(cb,sb,xb,yb)=(Buffer::new(4).unwrap(),Buffer::new(4).unwrap(),Buffer::new(nx*8).unwrap(),Buffer::new(ny*8).unwrap());
        gpu.stream.upload(&cb,&components).unwrap();gpu.stream.upload(&sb,&scales).unwrap();gpu.stream.upload(&xb,&table(&fx)).unwrap();gpu.stream.upload(&yb,&table(&fy)).unwrap();
        gpu.stream.launch(&rows,((planes*ny/8) as u32,1,1),256,0,&[&state.pointer,&cb.pointer,&sb.pointer,&xb.pointer]).unwrap();
        // No scale for the columns: the rows applied it.
        gpu.stream.launch(&cols,((planes*nx/8) as u32,1,1),256,shared,&[&state.pointer,&cb.pointer,&0u64,&yb.pointer]).unwrap();
        let mut out=vec![[0f32;2];planes*points];gpu.stream.download(&state,&mut out).unwrap();
        let mut worst=0f64;
        for p in 0..planes{
            let plane=&data[p*points..(p+1)*points];
            let expected:Vec<Complex>=if (p as i32)<components[0]{spectral(plane,nx,ny,&fx,&fy).iter().map(|&z|z*1.5).collect()}else{plane.to_vec()};
            let scale=expected.iter().fold(0f64,|a,z|a.max(z.abs()));
            for(z,o)in expected.iter().zip(&out[p*points..(p+1)*points]){worst=worst.max((Complex::new(o[0] as f64,o[1] as f64)-*z).abs()/scale);}
        }
        println!("{nx}x{ny}: largest error {worst:.2e} of the largest value");
        assert!(worst<2e-6,"{nx}x{ny}: {worst:e}");
    }
}

/// The fused end of a step on the large grids: the columns renormalizing, then the rows with the
/// next step's local matrices at each point -- here random ones -- and each trajectory's norm per
/// tile of 256 points; trajectories of both blocks, a lost one, and a last block of fewer than
/// four (kernels FUSE): against the exact transform and products, to single precision.
#[test]
fn gpu_rows_fused_with_the_local_step_match_the_exact_product(){
    let gpu=Gpu::new().unwrap();
    for(nx,ny)in [(256usize,64usize),(32,256),(128,128)]{
        let cols=gpu.function(&format!("kinetic_cols_{nx}x{ny}")).unwrap();let fused=gpu.function(&format!("kinetic_rows_local_{nx}x{ny}")).unwrap();
        let shared=ny*9*8;cols.allow_shared(shared).unwrap();
        let mut next=uniform((nx*ny+1) as u64);
        let components=[5i32,3,0,5,3];let scales=[1.5f32,0.5,2.,0.75,1.25];
        let trajectories=components.len();let planes=5*trajectories;let points=nx*ny;let tiles=points/256;
        let data:Vec<Complex>=(0..planes*points).map(|_|Complex::new(next(),next())).collect();
        let phase=|n:usize,next:&mut dyn FnMut()->f64|->Vec<Complex>{(0..n).map(|_|Complex::cis(20.*next())).collect()};
        let(fx,fy)=(phase(nx,&mut next),phase(ny,&mut next));
        let table=|f:&[Complex]|->Vec<[f32;2]>{let n=f.len();(0..n).map(|j|{let z=f[bitrev(j,n)]*(1./n as f64);[z.re as f32,z.im as f32]}).collect()};
        // Local matrices per point, entry (i, j) of point x at [(i n + j) points + x].
        let local=|n:usize,next:&mut dyn FnMut()->f64|->Vec<Complex>{(0..n*n*points).map(|_|Complex::new(next(),next())).collect()};
        let(m5,m3)=(local(5,&mut next),local(3,&mut next));
        let up=|v:&[[f32;2]]|{let b=Buffer::new(v.len()*8).unwrap();gpu.stream.upload(&b,v).unwrap();b};
        let state=up(&to_f32(&data));let(b5,b3)=(up(&to_f32(&m5)),up(&to_f32(&m3)));let(xb,yb)=(up(&table(&fx)),up(&table(&fy)));
        let(cb,sb,partial)=(Buffer::new(4*trajectories).unwrap(),Buffer::new(4*trajectories).unwrap(),Buffer::new(4*trajectories*tiles).unwrap());
        gpu.stream.upload(&cb,&components).unwrap();gpu.stream.upload(&sb,&scales).unwrap();gpu.stream.zero(&partial).unwrap();
        #[repr(C)] struct Local{m5:u64,m3:u64}
        impl driver::Arg for Local{fn pointer(&self)->*mut std::ffi::c_void{self as *const Local as *mut std::ffi::c_void}}
        let m=Local{m5:b5.pointer,m3:b3.pointer};
        gpu.stream.launch(&cols,((planes*nx/8) as u32,1,1),256,shared,&[&state.pointer,&cb.pointer,&sb.pointer,&yb.pointer]).unwrap();
        gpu.stream.launch(&fused,(trajectories.div_ceil(4) as u32,tiles as u32,1),256,0,&[&state.pointer,&(trajectories as i32),&cb.pointer,&xb.pointer,&m,&partial.pointer]).unwrap();
        let mut out=vec![[0f32;2];planes*points];gpu.stream.download(&state,&mut out).unwrap();
        let mut norms=vec![0f32;trajectories*tiles];gpu.stream.download(&partial,&mut norms).unwrap();
        let(mut worst,mut worst_norm)=(0f64,0f64);
        for t in 0..trajectories{
            let n=components[t] as usize;
            let k:Vec<Vec<Complex>>=(0..5).map(|c|{let plane=&data[(t*5+c)*points..(t*5+c+1)*points];
                if c<n{spectral(plane,nx,ny,&fx,&fy).iter().map(|&z|z*(scales[t] as f64)).collect()}else{plane.to_vec()}}).collect();
            let mut expected=k.clone();let mut tile_norm=vec![0f64;tiles];
            for x in 0..points{
                let mm=if n==5{&m5}else{&m3};
                for i in 0..n{let z=(0..n).fold(Complex::ZERO,|a,j|a+mm[(i*n+j)*points+x]*k[j][x]);expected[i][x]=z;tile_norm[x/256]+=z.norm_sqr();}
            }
            for c in 0..5{
                let scale=expected[c].iter().fold(0f64,|a,z|a.max(z.abs()));
                for(z,o)in expected[c].iter().zip(&out[(t*5+c)*points..(t*5+c+1)*points]){worst=worst.max((Complex::new(o[0] as f64,o[1] as f64)-*z).abs()/scale);}
            }
            if n>0{for(i,e)in tile_norm.iter().enumerate(){worst_norm=worst_norm.max((norms[t*tiles+i] as f64-e).abs()/e);}}
        }
        println!("{nx}x{ny}: largest error {worst:.2e} of the largest value, of the tiles' norms {worst_norm:.2e}");
        assert!(worst<4e-6&&worst_norm<4e-6,"{nx}x{ny}: {worst:e} {worst_norm:e}");
    }
}

/// The moves between grids (kernels `move_*`, as ensemble.rs runs them): two trajectories of a
/// long narrow grid resampled onto a wider one -- against the CPU's band-limited resampling,
/// renormalized over their components -- their per-trajectory state copied and the source slots
/// emptied.
#[test]
fn gpu_moves_resample_renormalize_and_carry_the_trajectory_state(){
    use atom_motion_gpu::model::Grid;use atom_motion_gpu::regrid::{interpolation,resample,transpose};
    let gpu=Gpu::new().unwrap();
    let(from,to)=(Grid{nx:32,ny:128,dx:0.0151,dy:0.0076},Grid{nx:64,ny:64,dx:0.0107,dy:0.0107});
    let(pf,pt)=(from.points(),to.points());
    let packet=|x:f64,y:f64,c:usize|{let(a,p)=((-(x*x/0.004+y*y/0.002)).exp()*(1.+0.3*c as f64),23.*x-17.*y+c as f64);[(a*p.cos()) as f32,(a*p.sin()) as f32]};
    // Source slots 0 (F = 2) and 1 (F = 1); destination slots 2 and 0 of three.
    let components=[5i32,3];let moves=[[0i32,2],[1,0]];
    let mut source=vec![[0f32;2];2*5*pf];
    for t in 0..2{for c in 0..components[t] as usize{for q in 0..pf{let r=from.position(q);source[(t*5+c)*pf+q]=packet(r[0]+0.01*t as f64,r[1],c);}}}
    let up=|bytes:&[u8]|{let b=Buffer::new(bytes.len()).unwrap();gpu.stream.upload(&b,bytes).unwrap();b};
    let raw=|v:&[[f32;2]]|->Vec<u8>{v.iter().flat_map(|z|[z[0].to_le_bytes(),z[1].to_le_bytes()].concat()).collect()};
    let ints=|v:&[i32]|->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()};
    let floats=|v:&[f32]|->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()};
    let doubles=|v:&[f64]|->Vec<u8>{v.iter().flat_map(|x|x.to_le_bytes()).collect()};
    let matrix=|m:Vec<f64>|up(&floats(&m.iter().map(|&x|x as f32).collect::<Vec<f32>>()));
    // A_x transposed (ensemble.rs).
    let(ax,ay)=(matrix(transpose(&interpolation(from.nx,from.dx,to.nx,to.dx),to.nx,from.nx)),matrix(interpolation(from.ny,from.dy,to.ny,to.dy)));
    let src=up(&raw(&source));let dst=up(&raw(&vec![[0f32;2];3*5*pt]));
    // Per-trajectory state: ids, components, scale, log_norm, threshold, jumps, weights.
    let(src_ids,src_comp,src_scale,src_log,src_thr,src_jumps,src_w)=(up(&ints(&[7,9])),up(&ints(&components)),up(&floats(&[0.5,0.25])),up(&doubles(&[-0.1,-0.2])),up(&doubles(&[-1.5,-2.5])),up(&ints(&[3,4])),up(&floats(&[1.25,0.75])));
    let(dst_ids,dst_comp,dst_scale,dst_log,dst_thr,dst_jumps,dst_w)=(up(&ints(&[0;3])),up(&ints(&[0;3])),up(&floats(&[0.;3])),up(&doubles(&[0.;3])),up(&doubles(&[0.;3])),up(&ints(&[0;3])),up(&floats(&[0.;3])));
    #[repr(C)] struct Trajectories{ids:u64,components:u64,scale:u64,log_norm:u64,threshold:u64,jumps:u64}
    impl driver::Arg for Trajectories{fn pointer(&self)->*mut std::ffi::c_void{self as *const Trajectories as *mut std::ffi::c_void}}
    let src_traj=Trajectories{ids:src_ids.pointer,components:src_comp.pointer,scale:src_scale.pointer,log_norm:src_log.pointer,threshold:src_thr.pointer,jumps:src_jumps.pointer};
    let dst_traj=Trajectories{ids:dst_ids.pointer,components:dst_comp.pointer,scale:dst_scale.pointer,log_norm:dst_log.pointer,threshold:dst_thr.pointer,jumps:dst_jumps.pointer};
    let list=up(&ints(&moves.concat()));let norm=up(&floats(&[0.;2]));let tmp=Buffer::new(2*5*from.ny*to.nx*8).unwrap();
    let f=|name:&str|gpu.function(name).unwrap();let n=2i32;let blocks=|total:usize|(total.div_ceil(256) as u32,1,1);
    gpu.stream.launch(&f("move_rows"),blocks(2*5*from.ny*to.nx),256,0,&[&src.pointer,&(from.nx as i32),&(from.ny as i32),&src_comp.pointer,&list.pointer,&n,&ax.pointer,&(to.nx as i32),&tmp.pointer]).unwrap();
    gpu.stream.launch(&f("move_cols"),blocks(2*5*to.ny*to.nx),256,0,&[&tmp.pointer,&(from.ny as i32),&src_comp.pointer,&list.pointer,&n,&ay.pointer,&(to.nx as i32),&(to.ny as i32),&dst.pointer,&norm.pointer]).unwrap();
    gpu.stream.launch(&f("move_scale"),((5*pt).div_ceil(256*8) as u32,2,1),256,0,&[&src_comp.pointer,&list.pointer,&norm.pointer,&dst.pointer,&(pt as i32)]).unwrap();
    gpu.stream.launch(&f("move_finish"),(1,1,1),64,0,&[&src_traj,&dst_traj,&src_w.pointer,&dst_w.pointer,&list.pointer,&n]).unwrap();
    let mut out=vec![[0f32;2];3*5*pt];gpu.stream.download(&dst,&mut out).unwrap();
    let mut worst=0f64;
    for(m,&[a,b])in moves.iter().enumerate(){
        let(a,b,k)=(a as usize,b as usize,components[m] as usize);
        let planes:Vec<Vec<[f32;2]>>=(0..k).map(|c|resample(&from,&to,&source[(a*5+c)*pf..(a*5+c+1)*pf])).collect();
        let total:f64=planes.iter().flatten().map(|z|(z[0] as f64).powi(2)+(z[1] as f64).powi(2)).sum();
        let scale=planes.iter().flatten().fold(0f64,|x,z|x.max((z[0] as f64).hypot(z[1] as f64)))/total.sqrt();
        for c in 0..k{for q in 0..pt{let(e,o)=(planes[c][q],out[(b*5+c)*pt+q]);
            worst=worst.max(((o[0] as f64-e[0] as f64/total.sqrt()).hypot(o[1] as f64-e[1] as f64/total.sqrt()))/scale);}}
    }
    let read_i=|b:&Buffer,n:usize|{let mut v=vec![0i32;n];gpu.stream.download(b,&mut v).unwrap();v};
    let read_f=|b:&Buffer,n:usize|{let mut v=vec![0f32;n];gpu.stream.download(b,&mut v).unwrap();v};
    let read_d=|b:&Buffer,n:usize|{let mut v=vec![0f64;n];gpu.stream.download(b,&mut v).unwrap();v};
    println!("{}x{} -> {}x{}: largest error {worst:.2e} of the largest value",from.nx,from.ny,to.nx,to.ny);
    assert!(worst<2e-6,"{worst:e}");
    assert_eq!(read_i(&dst_ids,3),[9,0,7]);assert_eq!(read_i(&dst_comp,3),[3,0,5]);assert_eq!(read_i(&dst_jumps,3),[4,0,3]);
    assert_eq!(read_f(&dst_scale,3),[1.,0.,1.]);assert_eq!(read_f(&dst_w,3),[0.75,0.,1.25]);
    assert_eq!(read_d(&dst_log,3),[-0.2,0.,-0.1]);assert_eq!(read_d(&dst_thr,3),[-2.5,0.,-1.5]);
    assert_eq!(read_i(&src_comp,2),[0,0],"the source slots emptied");
}
