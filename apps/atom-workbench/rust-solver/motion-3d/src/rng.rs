//! Counter-based random streams: Philox4x32-10 (Salmon et al., Random123), each stream keyed by
//! the run's seed and its purpose and counted by the trajectory and the event (the jump's number,
//! the noise's step):
//! a draw depends on those alone, not on the order trajectories run in, so a port to another
//! device draws the same numbers. Uniform doubles and directions; and the standard normal's
//! quantile, for the normal deviate of a given uniform one.

/// Philox4x32's multipliers and Weyl key increments.
const M0:u32=0xD251_1F53;const M1:u32=0xCD9E_8D57;const W0:u32=0x9E37_79B9;const W1:u32=0xBB67_AE85;

/// Philox4x32-10 of a counter and a key.
pub fn philox(counter:[u32;4],key:[u32;2])->[u32;4]{
    let(mut c,mut k)=(counter,key);
    for round in 0..10{
        if round>0{k[0]=k[0].wrapping_add(W0);k[1]=k[1].wrapping_add(W1);}
        let(p0,p1)=((M0 as u64)*(c[0] as u64),(M1 as u64)*(c[2] as u64));
        c=[((p1>>32) as u32)^c[1]^k[0],p1 as u32,((p0>>32) as u32)^c[3]^k[1],p0 as u32];
    }
    c
}

/// What a stream is for: the key's tag.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Purpose{
    /// A trajectory's initial state.
    Initial=1,
    /// A group's Latin hypercube permutations (counted by the group's first trajectory).
    Latin=2,
    /// A trajectory's jump (event: the jump's number; 0 its initial threshold).
    Jump=3,
    /// A pilot trajectory's initial state.
    Pilot=4,
    /// A trajectory's momentum noise (event: the step's number).
    Noise=5,
}

/// A stream: Philox blocks at the counter (block, event, trajectory) under the key (seed, purpose).
#[derive(Clone,Debug)]
pub struct Rng{key:[u32;2],counter:[u32;4],block:[u32;4],used:usize}

impl Rng{
    /// The stream of a purpose, a trajectory and an event (below 2^32) of the run seeded `seed`.
    pub fn new(seed:u64,purpose:Purpose,trajectory:u64,event:u64)->Rng{
        let key=[(seed as u32)^(purpose as u32).wrapping_mul(W0),(seed>>32) as u32];
        Rng{key,counter:[0,event as u32,trajectory as u32,(trajectory>>32) as u32],block:[0;4],used:4}
    }
    pub fn next_u32(&mut self)->u32{
        if self.used==4{self.block=philox(self.counter,self.key);self.counter[0]=self.counter[0].wrapping_add(1);self.used=0;}
        let x=self.block[self.used];self.used+=1;x
    }
    pub fn next_u64(&mut self)->u64{let hi=self.next_u32() as u64;(hi<<32)|self.next_u32() as u64}
    /// Uniform in [0, 1).
    pub fn uniform(&mut self)->f64{(self.next_u64()>>11) as f64*(1./(1u64<<53) as f64)}
    /// Uniform in (0, 1]: a waiting time's threshold, never zero.
    pub fn open_uniform(&mut self)->f64{1.-self.uniform()}
    /// Uniform in (0, 1), both ends excluded: the middle of one of 2^53 equal cells.
    pub fn interior(&mut self)->f64{((self.next_u64()>>11) as f64+0.5)*(1./(1u64<<53) as f64)}
    /// A unit vector uniform on the sphere.
    pub fn direction(&mut self)->[f64;3]{
        let z=2.*self.uniform()-1.;let phi=std::f64::consts::TAU*self.uniform();let rho=(1.-z*z).max(0.).sqrt();let(s,c)=phi.sin_cos();[rho*c,rho*s,z]
    }
}

/// The standard normal's quantile Φ⁻¹(p) for p in (0, 1), to double precision: Wichura's AS 241
/// (PPND16), rational approximations in the centre and in two tail ranges.
pub fn normal_quantile(p:f64)->f64{
    let q=p-0.5;
    if q.abs()<=0.425{
        let r=0.180625-q*q;
        return q*(((((((2.5090809287301226727e3*r+3.3430575583588128105e4)*r+6.7265770927008700853e4)*r+4.5921953931549871457e4)*r
            +1.3731693765509461125e4)*r+1.9715909503065514427e3)*r+1.3314166789178437745e2)*r+3.3871328727963666080e0)
            /(((((((5.2264952788528545610e3*r+2.8729085735721942674e4)*r+3.9307895800092710610e4)*r+2.1213794301586595867e4)*r
            +5.3941960214247511077e3)*r+6.8718700749205790830e2)*r+4.2313330701600911252e1)*r+1.);
    }
    let r=(-(if q<0.{p}else{1.-p}).ln()).sqrt();
    let x=if r<=5.{
        let r=r-1.6;
        (((((((7.74545014278341407640e-4*r+2.27238449892691845833e-2)*r+2.41780725177450611770e-1)*r+1.27045825245236838258e0)*r
            +3.64784832476320460504e0)*r+5.76949722146069140550e0)*r+4.63033784615654529590e0)*r+1.42343711074968357734e0)
            /(((((((1.05075007164441684324e-9*r+5.47593808499534494600e-4)*r+1.51986665636164571966e-2)*r+1.48103976427480074590e-1)*r
            +6.89767334985100004550e-1)*r+1.67638483018380384940e0)*r+2.05319162663775882187e0)*r+1.)
    }else{
        let r=r-5.;
        (((((((2.01033439929228813265e-7*r+2.71155556874348757815e-5)*r+1.24266094738807843860e-3)*r+2.65321895265761230930e-2)*r
            +2.96560571828504891230e-1)*r+1.78482653991729133580e0)*r+5.46378491116411436990e0)*r+6.65790464350110377720e0)
            /(((((((2.04426310338993978564e-15*r+1.42151175831644588870e-7)*r+1.84631831751005468180e-5)*r+7.86869131145613259100e-4)*r
            +1.48753612908506148525e-2)*r+1.36929880922735805310e-1)*r+5.99832206555887937690e-1)*r+1.)
    };
    if q<0.{-x}else{x}
}

#[cfg(test)]
mod tests{
    use super::*;
    /// Random123's known-answer vectors for Philox4x32-10.
    #[test]
    fn philox_gives_random123_s_known_answers(){
        assert_eq!(philox([0;4],[0;2]),[0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]);
        assert_eq!(philox([0xffff_ffff;4],[0xffff_ffff;2]),[0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd]);
        assert_eq!(philox([0x243f6a88,0x85a308d3,0x13198a2e,0x03707344],[0xa4093822,0x299f31d0]),[0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1]);
    }
}
