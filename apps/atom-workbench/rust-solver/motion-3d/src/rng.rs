//! A small reproducible random stream per trajectory: xoshiro256++ seeded by SplitMix64 from
//! (seed, stream), uniform doubles and directions; and the standard normal's quantile, for the
//! normal deviate of a given uniform one.

#[derive(Clone,Debug)]
pub struct Rng{state:[u64;4]}

fn splitmix(x:&mut u64)->u64{
    *x=x.wrapping_add(0x9E37_79B9_7F4A_7C15);let mut z=*x;
    z=(z^(z>>30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);z=(z^(z>>27)).wrapping_mul(0x94D0_49BB_1331_11EB);z^(z>>31)
}

impl Rng{
    /// The stream `stream` of the run seeded `seed`: streams of one seed are independent.
    pub fn new(seed:u64,stream:u64)->Rng{
        let mut x=seed^stream.wrapping_mul(0xD1B5_4A32_D192_ED03);let state=[splitmix(&mut x),splitmix(&mut x),splitmix(&mut x),splitmix(&mut x)];
        Rng{state}
    }
    pub fn next_u64(&mut self)->u64{
        let s=&mut self.state;let result=(s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);let t=s[1]<<17;
        s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=s[3].rotate_left(45);result
    }
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
