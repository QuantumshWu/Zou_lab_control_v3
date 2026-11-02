//! A small reproducible random stream per trajectory: xoshiro256++ seeded by SplitMix64 from
//! (seed, stream), uniform doubles in [0, 1) and normal deviates (Box–Muller).

#[derive(Clone,Debug)]
pub struct Rng{state:[u64;4],spare:Option<f64>}

fn splitmix(x:&mut u64)->u64{
    *x=x.wrapping_add(0x9E37_79B9_7F4A_7C15);let mut z=*x;
    z=(z^(z>>30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);z=(z^(z>>27)).wrapping_mul(0x94D0_49BB_1331_11EB);z^(z>>31)
}

impl Rng{
    /// The stream `stream` of the run seeded `seed`: streams of one seed are independent.
    pub fn new(seed:u64,stream:u64)->Rng{
        let mut x=seed^stream.wrapping_mul(0xD1B5_4A32_D192_ED03);let state=[splitmix(&mut x),splitmix(&mut x),splitmix(&mut x),splitmix(&mut x)];
        Rng{state,spare:None}
    }
    pub fn next_u64(&mut self)->u64{
        let s=&mut self.state;let result=(s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);let t=s[1]<<17;
        s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=s[3].rotate_left(45);result
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self)->f64{(self.next_u64()>>11) as f64*(1./(1u64<<53) as f64)}
    /// Uniform in (0, 1]: a waiting time's threshold, never zero.
    pub fn open_uniform(&mut self)->f64{1.-self.uniform()}
    /// A standard normal deviate.
    pub fn normal(&mut self)->f64{
        if let Some(x)=self.spare.take(){return x;}
        let(u,v)=(self.open_uniform(),self.uniform());let r=(-2.*u.ln()).sqrt();let(s,c)=(std::f64::consts::TAU*v).sin_cos();
        self.spare=Some(r*s);r*c
    }
    /// A unit vector uniform on the sphere.
    pub fn direction(&mut self)->[f64;3]{
        let z=2.*self.uniform()-1.;let phi=std::f64::consts::TAU*self.uniform();let rho=(1.-z*z).max(0.).sqrt();let(s,c)=phi.sin_cos();[rho*c,rho*s,z]
    }
}
