//! A trajectory moved between grids: its wavefunction is the band-limited function its samples
//! define (the periodic box's Fourier series), sampled at the new grid's points; points beyond the
//! old box are zero, and frequencies beyond either grid's Nyquist limit are dropped. A trajectory
//! is moved only while its tails near the old grid's edges are negligible.
use crate::model::Grid;

/// A (n_to × n_from, row-major, real): ψ_to(x'_i) = Σ_j A_ij ψ_from(x_j) along one axis.
pub fn interpolation(from_n:usize,from_dx:f64,to_n:usize,to_dx:f64)->Vec<f64>{
    let half=from_n as f64*from_dx/2.;
    // The frequencies 2π m/(n dx) below both Nyquist limits (the Nyquist mode itself dropped).
    let cut=(std::f64::consts::PI/from_dx).min(std::f64::consts::PI/to_dx);
    let base=std::f64::consts::TAU/(from_n as f64*from_dx);
    let modes=(1..from_n/2).filter(|&m|(m as f64*base)<cut*(1.-1e-12)).count() as f64;
    // (1 + 2 Σ_{m=1}^{M} cos m θ)/n, θ = base·d, summed in closed form (the Dirichlet kernel):
    // sin((M + ½) θ)/(n sin(θ/2)), and (2M + 1)/n where sin(θ/2) vanishes.
    let kernel=|d:f64|{let h=base*d/2.;let s=h.sin();if s.abs()<1e-12{(2.*modes+1.)/from_n as f64}else{((2.*modes+1.)*h).sin()/(from_n as f64*s)}};
    let mut a=vec![0.;to_n*from_n];
    for i in 0..to_n{
        let x=(i as f64-(to_n/2) as f64)*to_dx;
        if x< -half||x>=half{continue;}
        for j in 0..from_n{a[i*from_n+j]=kernel(x-(j as f64-(from_n/2) as f64)*from_dx);}
    }
    a
}
/// A matrix of `rows` × `cols` (row-major) transposed.
pub fn transpose(a:&[f64],rows:usize,cols:usize)->Vec<f64>{(0..cols*rows).map(|k|a[(k%rows)*cols+k/rows]).collect()}

/// One plane resampled from grid `from` to grid `to` (both of the plane, or both one row).
pub fn resample(from:&Grid,to:&Grid,plane:&[[f32;2]])->Vec<[f32;2]>{
    let ax=interpolation(from.nx,from.dx,to.nx,to.dx);
    // Rows: t[y][x'] = Σ_x A[x'][x] ψ[y][x].
    let mut t=vec![[0f64;2];from.ny*to.nx];
    for y in 0..from.ny{for i in 0..to.nx{let(mut re,mut im)=(0.,0.);
        for j in 0..from.nx{let w=ax[i*from.nx+j];if w!=0.{let z=plane[y*from.nx+j];re+=w*z[0] as f64;im+=w*z[1] as f64;}}
        t[y*to.nx+i]=[re,im];}}
    if from.ny==1{return t.iter().map(|z|[z[0] as f32,z[1] as f32]).collect();}
    let ay=interpolation(from.ny,from.dy,to.ny,to.dy);
    let mut out=vec![[0f32;2];to.ny*to.nx];
    for i in 0..to.ny{for x in 0..to.nx{let(mut re,mut im)=(0.,0.);
        for j in 0..from.ny{let w=ay[i*from.ny+j];if w!=0.{let z=t[j*to.nx+x];re+=w*z[0];im+=w*z[1];}}
        out[i*to.nx+x]=[re as f32,im as f32];}}
    out
}

#[cfg(test)]
mod tests{
    use super::*;
    /// A Gaussian wave packet with a momentum, well inside both boxes, moved to a finer and a
    /// coarser grid: the samples of the same function there.
    #[test]
    fn a_band_limited_packet_is_sampled_on_the_new_grid(){
        let packet=|x:f64,y:f64|{let(a,p)=((-(x*x+y*y)/(2.*0.04f64.powi(2))).exp(),37.*x-11.*y);[(a*p.cos()) as f32,(a*p.sin()) as f32]};
        let from=Grid{nx:64,ny:64,dx:0.0107,dy:0.0107};
        let plane:Vec<[f32;2]>=(0..from.points()).map(|p|{let r=from.position(p);packet(r[0],r[1])}).collect();
        // A finer grid, a coarser one, and a long narrow one (each axis apart).
        for to in [Grid{nx:128,ny:128,dx:0.0107/2f64.sqrt(),dy:0.0107/2f64.sqrt()},Grid{nx:32,ny:32,dx:0.0107*2f64.sqrt(),dy:0.0107*2f64.sqrt()},
            Grid{nx:32,ny:128,dx:0.0107*2f64.sqrt(),dy:0.0107/2f64.sqrt()}]{
            let out=resample(&from,&to,&plane);
            let worst=(0..to.points()).map(|p|{let r=to.position(p);let e=packet(r[0],r[1]);((out[p][0]-e[0]).powi(2)+(out[p][1]-e[1]).powi(2)).sqrt()}).fold(0f32,f32::max);
            assert!(worst<1e-5,"{} x {}: {worst:e}",to.nx,to.ny);
        }
    }
}
