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
    let modes:Vec<f64>=(1..from_n/2).map(|m|m as f64*base).filter(|&k|k<cut*(1.-1e-12)).collect();
    let mut a=vec![0.;to_n*from_n];
    for i in 0..to_n{
        let x=(i as f64-(to_n/2) as f64)*to_dx;
        if x< -half||x>=half{continue;}
        for j in 0..from_n{
            let d=x-(j as f64-(from_n/2) as f64)*from_dx;
            a[i*from_n+j]=(1.+2.*modes.iter().map(|&k|(k*d).cos()).sum::<f64>())/from_n as f64;
        }
    }
    a
}

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
