//! The 3D model: the eliminated ground generator of `ground` at any point r of
//! the request's frame and any momentum ħκ, from tables over the focused beams' intensities and
//! the beams' fields at r.
//!
//! # Where the generator depends on r
//! H(r) = H0 + Σ_l (|a_l(r)|² − 1) O_l + Σ_l |a_l(r)|² F_l (ground.rs) depends on r only through
//! the intensities s_f = |a_f(r)|² ∈ [0, 1] of the focused beams -- a plane wave's is 1
//! everywhere -- and so does the elimination: E_b, H0g, K and every near beam's responses. The
//! near beams enter through their amplitudes a_l(r) (phase and profile), linearly per carrier.
//! The tables hold the elimination on a uniform grid of s ∈ [0, 1]^D (D focused beams), cubic in
//! each s_f, the grid doubled until the interpolation at the new nodes agrees with the exact
//! elimination to `TABLE_SHARE` of every table's scale.
//!
//! # A carrier's field basis
//! Each near beam's coupling V_l is the dipole operator contracted with the beam's field, so the
//! couplings of one carrier span at most three matrices. An orthonormal basis U_k of that span
//! (Gram–Schmidt, Frobenius inner product) gives V_l = Σ_k c_lk U_k and -- the responses being linear
//! in the coupling column by column -- R_l = Σ_k c_lk T_k, T_k the responses of U_k. At r the
//! carrier drives V_c = Σ_k E_k U_k and responds R_c = Σ_k E_k T_k(s), E_k(r) = Σ_l a_l(r) c_lk:
//! the carrier's field on its basis, three numbers whatever its number of beams.
//!
//! # The generator of a block at r
//! On a hyperfine block (the secular step keeps a trajectory in one), with A_k'k = U_k'†T_k and
//! Q_k'k = T_k'†D T_k,
//!
//! ```text
//! H_eff = H0g − Herm Σ_c Σ_k'k E_k'* E_k A_k'k − (i/2) Σ_c Σ_k'k E_k'* E_k Q_k'k
//! ```
//!
//! Only the Hermitian part of the light shift acts (the generator's and the force's), and
//! Herm Σ_k'k E_k'* E_k A_k'k = Σ_k'k E_k'* E_k S_k'k with S_k'k = (A_k'k + A_kk'†)/2, so S_kk' = S_k'k†
//! as Q_kk' = Q_k'k†: the tables hold S and Q for k' ≤ k only, the pairs k' < k standing for both.
//!
//! # The photons and the Doppler term
//! The photon of axis a scattered from carrier c leaves the atom in C_a R_c ψ, from the jump
//! tables J_ak = C_a T_k (all ground rows: the to-block is chosen there). Absorbing from beam l
//! raises the excited state's energy by its kinetic shift w_l = (ħ/m) k_l·κ + ħ|k_l|²/2m; to first
//! order R_l → R_l − w_l R2_l (ground.rs `first_order`), so the photon is
//! C_a Σ_k (E_k T_k − E^D_k T2_k) ψ with J2_ak = C_a T2_k and E^D_k = Σ_l a_l w_l c_lk: the
//! scattering's velocity dependence, which the jumps' mean momentum turns into Doppler friction.
//! In the no-jump generator the term changes the light shift and the scattering by kv/Δ and is
//! left out: with it, T and the populations of the 1D references TW1 and TW2 moved by less than
//! their standard errors (8192 trajectories each).
use crate::ground::Ground;
use crate::linear::Dense;
use crate::{Complex,Request,STATES};
use atom_population_core::lattice::accuracy::{A_PRIORI_SHARE,POPULATION_TOLERANCE};
use rayon::prelude::*;

pub const HBAR:f64=1.0545718176461565e-34;
pub const BOLTZMANN:f64=1.380649e-23;
/// The most focused beams the tables span (one axis of intensity each).
pub const MOST_FOCUSED:usize=3;
/// The ground manifold's states (5S₁/₂, F = 1 and 2): a photon's amplitude on each.
pub const GROUND:usize=8;
/// The most states of a hyperfine block (F = 2) and the largest rank of a carrier's couplings
/// (the dipole operator's three components).
pub const MOST_STATES:usize=5;
pub const MOST_RANK:usize=3;
/// The interpolation's error at the refined nodes, relative to each table's scale: the a-priori
/// share of the tolerance, as the population core gives its own approximations.
pub const TABLE_SHARE:f64=A_PRIORI_SHARE*POPULATION_TOLERANCE;
/// The coarsest grid of each intensity axis (nodes), the fewest a cubic stencil takes doubled once.
const FIRST_NODES:usize=5;

/// A beam's geometry in the request's frame.
#[derive(Clone,Debug)]
pub struct Geometry{pub direction:[f64;3],pub wave_number:f64,pub waist:Option<f64>}
impl Geometry{
    /// The wave vector (rad/um).
    pub fn wave_vector(&self)->[f64;3]{self.direction.map(|n|n*self.wave_number)}
    /// a(r) and ∇a(r) (1/um): the plane wave e^{ikζ}, or the paraxial Gaussian beam focused at the
    /// origin e^{ikζ} exp(−ρ²/(w² q))/q, q = 1 + iζ/z_R, z_R = k w²/2 (lib.rs `Beam::amplitude_at`),
    /// whose ∇ln a = ik n − 2(r − ζn)/(w² q) + i(ρ²/(w² q²) − 1/q) n / z_R.
    pub fn field(&self,r:[f64;3])->(Complex,[Complex;3]){
        let(k,n)=(self.wave_number,self.direction);let zeta=r[0]*n[0]+r[1]*n[1]+r[2]*n[2];let wave=Complex::cis(k*zeta);
        match self.waist{
            None=>{let g=Complex::new(0.,k)*wave;(wave,n.map(|x|g*x))}
            Some(w)=>{
                let rayleigh=k*w*w/2.;let q=Complex::new(1.,zeta/rayleigh);let inverse=Complex::ONE/q;
                let rho2=(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-zeta*zeta).max(0.);
                let a=wave*(Complex::new(-rho2/(w*w),0.)*inverse).exp()*inverse;
                let along=Complex::new(0.,k)+(Complex::new(0.,rho2/(w*w))*inverse*inverse-Complex::new(0.,1.)*inverse)*(1./rayleigh);
                let across=inverse*(-2./(w*w));
                (a,std::array::from_fn(|i|a*(along*n[i]+across*(r[i]-zeta*n[i]))))
            }
        }
    }
}

/// A carrier: its beams, their couplings' basis and coefficients, its photons' wave number.
#[derive(Clone,Debug)]
pub struct Carrier{
    pub beams:Vec<usize>,
    pub rank:usize,
    /// c_lk per beam of the carrier (in `beams` order).
    pub coefficients:Vec<[Complex;MOST_RANK]>,
    pub wave_number:f64,
}

/// A hyperfine block: its F and its states (positions in `Ground::ground_states`).
#[derive(Clone,Debug)]
pub struct Block{pub f:u8,pub states:Vec<usize>}

/// Where one block's tables lie in a node's slice.
#[derive(Clone,Debug)]
struct Layout{size:usize,h0:usize,carriers:Vec<CarrierLayout>}
#[derive(Clone,Copy,Debug)]
struct CarrierLayout{a:usize,q:usize,j:usize,j2:usize,t:usize}
impl Layout{
    fn new(nb:usize,ng:usize,ne:usize,ranks:&[usize])->Layout{
        let mut size=nb*nb;let h0=0;
        let carriers=ranks.iter().map(|&r|{let square=pairs(r)*nb*nb;let jump=3*r*ng*nb;let response=r*ne*nb;
            let l=CarrierLayout{a:size,q:size+square,j:size+2*square,j2:size+2*square+jump,t:size+2*square+2*jump};
            size+=2*square+2*jump+response;l}).collect();
        Layout{size,h0,carriers}
    }
}

/// The pairs k' ≤ k of a carrier of rank r, and pair (k', k)'s place among them.
fn pairs(r:usize)->usize{r*(r+1)/2}
fn pair(k1:usize,k:usize,r:usize)->usize{k1*r-k1*(k1.max(1)-1)/2+(k-k1)}

/// The interpolation stencil at s: per focused beam the first of its four nodes, their cubic
/// weights and the weights' derivatives in s_f; the nodes are their tensor product.
#[derive(Clone,Copy,Debug)]
pub struct Stencil{pub dimensions:usize,pub first:[usize;MOST_FOCUSED],pub weights:[[f64;4];MOST_FOCUSED],pub derivatives:[[f64;4];MOST_FOCUSED],pub stride:[usize;MOST_FOCUSED]}
impl Stencil{
    /// The number of nodes (4^D).
    pub fn count(&self)->usize{1<<(2*self.dimensions)}
    /// Node i's index, its weight and its weights' derivatives in each s_f.
    pub fn node(&self,i:usize)->(usize,f64,[f64;MOST_FOCUSED]){
        let(mut index,mut weight,mut derivative)=(0,1.,[1.;MOST_FOCUSED]);
        for f in 0..self.dimensions{let m=(i>>(2*f))&3;index+=(self.first[f]+m)*self.stride[f];weight*=self.weights[f][m];
            for g in 0..self.dimensions{derivative[g]*=if g==f{self.derivatives[f][m]}else{self.weights[f][m]};}}
        (index,weight,derivative)
    }
}

/// The fields of one carrier at (r, κ): E_k, ∇E_k (per k, per axis) and E^D_k.
#[derive(Clone,Copy,Debug,Default)]
pub struct CarrierField{pub e:[Complex;MOST_RANK],pub grad:[[Complex;3];MOST_RANK],pub doppler:[Complex;MOST_RANK]}

/// The most carriers (frequencies of near light) and beams a scene may have.
pub const MOST_CARRIERS:usize=8;
pub const MOST_BEAMS:usize=32;

/// Everything at (r, κ) that does not depend on the internal state.
#[derive(Clone,Copy,Debug)]
pub struct Point{
    pub r:[f64;3],
    pub kappa:[f64;3],
    pub stencil:Stencil,
    /// ∇s_f (1/um).
    pub intensity_gradient:[[f64;3];MOST_FOCUSED],
    pub fields:[CarrierField;MOST_CARRIERS],
}

/// A block's generator at a point: H_eff less `scalar`, the scalar part of its Hermitian part (a
/// phase), and the matrices the Ehrenfest force reads: ∂H0g/∂s_f, Σ E_k'* E_k ∂A_k'k/∂s_f (per
/// focused beam) and Σ ∇_a(E_k'* E_k) A_k'k (per axis), all n_b × n_b, re/im interleaved.
#[derive(Clone,Copy,Debug)]
pub struct Local{pub hamiltonian:[Complex;25],pub scalar:f64,pub potential:[[f64;50];MOST_FOCUSED],pub shift:[[f64;50];MOST_FOCUSED],pub field:[[f64;50];3]}

/// acc += c x for interleaved complex vectors of W/2 entries.
#[inline(always)]
fn caxpy<const W:usize>(c:Complex,x:&[f64;50],acc:&mut [f64;50]){
    for j in (0..W).step_by(2){let(u,v)=(x[j],x[j+1]);acc[j]+=c.re*u-c.im*v;acc[j+1]+=c.re*v+c.im*u;}
}
/// ⟨ψ|X|ψ⟩ for an interleaved N × N matrix X.
#[inline(always)]
fn quadratic<const N:usize>(x:&[f64;50],psi:&[Complex])->Complex{
    let mut s=Complex::ZERO;
    for a in 0..N{let mut row=Complex::ZERO;for b in 0..N{let k=2*(a*N+b);row+=Complex::new(x[k],x[k+1])*psi[b];}s+=psi[a].conj()*row;}
    s
}

/// The tables on one grid.
#[derive(Clone,Debug)]
struct Tables{nodes:usize,data:Vec<Vec<f64>>}

/// The 3D model of a request.
#[derive(Clone,Debug)]
pub struct Model{
    pub ground:Ground,
    pub geometry:Vec<Geometry>,
    /// The focused beams (indices into the beams), one intensity axis each.
    pub focused:Vec<usize>,
    pub carriers:Vec<Carrier>,
    pub blocks:Vec<Block>,
    /// ħ/m (um²/us).
    pub hbar_over_mass:f64,
    pub mass_kg:f64,
    layouts:Vec<Layout>,
    tables:Tables,
    /// The nodes per intensity axis and the interpolation's measured largest relative error.
    pub table_error:f64,
    /// The processor has AVX2 and FMA: the generator is built with them.
    wide:bool,
    /// The kinetic shift of absorption (`kinetic_shift`), always on in a solve; off only where a
    /// validation compares with the recoil-heating formula, which has no Doppler friction.
    pub doppler:bool,
}

fn frobenius_inner(a:&Dense,b:&Dense)->Complex{a.entries.iter().zip(&b.entries).fold(Complex::ZERO,|s,(x,y)|s+x.conj()**y)}

/// Cubic Lagrange weights on four consecutive nodes of a uniform grid of `nodes` points on [0, 1],
/// and their derivatives in s: the stencil's first node and the weights.
fn cubic(nodes:usize,s:f64)->(usize,[f64;4],[f64;4]){
    let h=1./(nodes-1) as f64;let x=s.clamp(0.,1.)/h;
    let first=((x.floor() as isize)-1).clamp(0,nodes as isize-4) as usize;let t=x-first as f64;
    let w=[-(t-1.)*(t-2.)*(t-3.)/6.,t*(t-2.)*(t-3.)/2.,-t*(t-1.)*(t-3.)/2.,t*(t-1.)*(t-2.)/6.];
    let d=[-((t-2.)*(t-3.)+(t-1.)*(t-3.)+(t-1.)*(t-2.))/6.,((t-2.)*(t-3.)+t*(t-3.)+t*(t-2.))/2.,
        -((t-1.)*(t-3.)+t*(t-3.)+t*(t-1.))/2.,((t-1.)*(t-2.)+t*(t-2.)+t*(t-1.))/6.];
    (first,w,d.map(|x|x/h))
}

impl Model{
    /// The model of a request: its front end, the carriers' bases and the tables.
    pub fn new(request:&Request)->Result<Model,String>{
        let ground=Ground::new(request)?;
        let geometry:Vec<Geometry>=ground.beams.iter().map(|b|Geometry{direction:b.direction,wave_number:b.wave_number_per_um,waist:b.waist_um}).collect();
        let focused:Vec<usize>=(0..geometry.len()).filter(|&l|geometry[l].waist.is_some()).collect();
        if focused.len()>MOST_FOCUSED{return Err(format!("3D 运动最多支持 {MOST_FOCUSED} 束聚焦光（场景中有 {} 束）",focused.len()));}
        // Each carrier's couplings: an orthonormal basis and the beams' coefficients on it.
        let mut carriers=vec![];let mut bases:Vec<Vec<Vec<Complex>>>=vec![];
        for c in &ground.carriers{
            let couplings:Vec<&Dense>=c.beams.iter().map(|&l|&ground.couplings[l]).collect();
            let(mut basis,mut mixing):(Vec<Dense>,Vec<Vec<Complex>>)=(vec![],vec![]);
            for(j,v)in couplings.iter().enumerate(){
                let mut u=(*v).clone();let mut g=vec![Complex::ZERO;couplings.len()];g[j]=Complex::ONE;
                // Twice, for orthogonality to round-off.
                for _ in 0..2{for(k,b)in basis.iter().enumerate(){
                    let x=frobenius_inner(b,&u);u.add_scaled(-x,b);for(gi,mi)in g.iter_mut().zip(&mixing[k]){*gi-=x**mi;}
                }}
                let norm=frobenius_inner(&u,&u).re.sqrt();let scale=frobenius_inner(v,v).re.sqrt();
                if norm<=1e-10*scale{continue;}
                basis.push(u.scaled(Complex::new(1./norm,0.)));mixing.push(g.into_iter().map(|x|x*(1./norm)).collect());
            }
            if basis.len()>MOST_RANK{return Err(format!("A carrier's couplings span {} matrices; the dipole operator has three components",basis.len()));}
            let coefficients:Vec<[Complex;MOST_RANK]>=couplings.iter().map(|v|std::array::from_fn(|k|if k<basis.len(){frobenius_inner(&basis[k],v)}else{Complex::ZERO})).collect();
            // The beams are on the span to round-off.
            for(v,c)in couplings.iter().zip(&coefficients){
                let mut d=(*v).clone();for(k,b)in basis.iter().enumerate(){d.add_scaled(-c[k],b);}
                let(e,s)=(frobenius_inner(&d,&d).re.sqrt(),frobenius_inner(v,v).re.sqrt());
                if e>1e-9*s{return Err(format!("A beam's coupling is off its carrier's span by {:.1e} of its norm",e/s));}
            }
            carriers.push(Carrier{beams:c.beams.clone(),rank:basis.len(),coefficients,wave_number:c.wave_number});bases.push(mixing);
        }
        if carriers.len()>MOST_CARRIERS||geometry.len()>MOST_BEAMS{return Err(format!("3D 运动最多支持 {MOST_CARRIERS} 个近共振频率、{MOST_BEAMS} 束光"));}
        let blocks:Vec<Block>=ground.blocks.iter().map(|b|Block{f:b.f,states:b.states.clone()}).collect();
        if blocks.iter().any(|b|b.states.len()>MOST_STATES){return Err("A hyperfine block has more than five states".into());}
        let ng=ground.ground_states.len();
        if ng!=GROUND{return Err(format!("The ground manifold has {ng} states, not {GROUND}"));}
        let ranks:Vec<usize>=carriers.iter().map(|c|c.rank).collect();
        let ne=ground.excited_states.len();let layouts:Vec<Layout>=blocks.iter().map(|b|Layout::new(b.states.len(),ng,ne,&ranks)).collect();
        let hbar_over_mass=HBAR/request.mass_kg*1e6;
        let mut model=Model{ground,geometry,focused,carriers,blocks,hbar_over_mass,mass_kg:request.mass_kg,layouts,tables:Tables{nodes:1,data:vec![]},table_error:0.,
            wide:std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("fma"),doppler:true};
        model.tabulate(&bases)?;
        Ok(model)
    }

    /// H at the intensities s of the focused beams (the plane waves at their own).
    pub fn hamiltonian_at(&self,s:&[f64])->Dense{
        let g=&self.ground;let mut h=g.hamiltonian.clone();
        for(i,&l)in self.focused.iter().enumerate(){
            h.add_scaled(Complex::new(s[i]-1.,0.),&g.outside[l]);h.add_scaled(Complex::new(s[i],0.),&g.folds[l]);
        }
        for l in (0..self.geometry.len()).filter(|l|!self.focused.contains(l)){h.add_scaled(Complex::ONE,&g.folds[l]);}
        h
    }

    /// Every block's tables at the intensities s.
    fn node(&self,s:&[f64],mixing:&[Vec<Vec<Complex>>])->Result<Vec<Vec<Complex>>,String>{
        let g=&self.ground;let e=g.eliminate(&self.hamiltonian_at(s))?;let(ne,ng)=(g.excited_states.len(),g.ground_states.len());
        // Per carrier: U_k, T_k, T2_k (ne × ng).
        let combine=|c:usize,matrices:&[Dense]|->Vec<Dense>{mixing[c].iter().map(|g_k|{let mut m=Dense::zeros(ne,ng);
            for(j,&l)in self.carriers[c].beams.iter().enumerate(){m.add_scaled(g_k[j],&matrices[l]);}m}).collect()};
        let u:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&g.couplings)).collect();
        let t:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.responses)).collect();
        let t2:Vec<Vec<Dense>>=(0..self.carriers.len()).map(|c|combine(c,&e.derivatives)).collect();
        let jump=|m:&Dense|->[Dense;3]{std::array::from_fn(|a|g.emission[a].product(m))};
        let(j,j2):(Vec<Vec<[Dense;3]>>,Vec<Vec<[Dense;3]>>)=(t.iter().map(|ts|ts.iter().map(jump).collect()).collect(),t2.iter().map(|ts|ts.iter().map(jump).collect()).collect());
        Ok(self.blocks.iter().zip(&self.layouts).map(|(block,layout)|{
            let(nb,states)=(block.states.len(),&block.states);let mut out=vec![Complex::ZERO;layout.size];
            for x in 0..nb{for y in 0..nb{out[layout.h0+x*nb+y]=e.hamiltonian[(states[x],states[y])];}}
            for(c,cl)in layout.carriers.iter().enumerate(){
                let r=self.carriers[c].rank;
                for k1 in 0..r{for k in k1..r{
                    let base=pair(k1,k,r)*nb*nb;
                    for x in 0..nb{for y in 0..nb{
                        // S_k1k = (A_k1k + A_kk1†)/2, Q_k1k.
                        let(bx,by)=(states[x],states[y]);let(mut a,mut q)=(Complex::ZERO,Complex::ZERO);
                        for i in 0..ne{a+=(u[c][k1][(i,bx)].conj()*t[c][k][(i,by)]+(u[c][k][(i,by)].conj()*t[c][k1][(i,bx)]).conj())*0.5;}
                        for jj in 0..ng{for ax in 0..3{q+=j[c][k1][ax][(jj,bx)].conj()*j[c][k][ax][(jj,by)];}}
                        out[cl.a+base+x*nb+y]=a;out[cl.q+base+x*nb+y]=q;
                    }}
                }}
                for k in 0..r{for ax in 0..3{for row in 0..ng{for y in 0..nb{
                    let at=((k*3+ax)*ng+row)*nb+y;out[cl.j+at]=j[c][k][ax][(row,states[y])];out[cl.j2+at]=j2[c][k][ax][(row,states[y])];
                }}}}
                for k in 0..r{for i in 0..ne{for y in 0..nb{out[cl.t+(k*ne+i)*nb+y]=t[c][k][(i,states[y])];}}}
            }
            out
        }).collect())
    }

    /// The intensity points of a grid of n nodes per focused beam, in node order.
    fn grid(&self,n:usize)->Vec<Vec<f64>>{
        let d=self.focused.len();(0..n.pow(d as u32)).map(|i|(0..d).map(|f|((i/n.pow(f as u32))%n) as f64/(n-1).max(1) as f64).collect()).collect()
    }
    /// The tables on a grid of n nodes per focused beam (one node without any).
    fn build(&self,n:usize,mixing:&[Vec<Vec<Complex>>])->Result<Tables,String>{
        let nodes:Vec<Vec<Vec<Complex>>>=self.grid(n).par_iter().map(|s|self.node(s,mixing)).collect::<Result<_,_>>()?;
        // data[block]: node-major, layout.size per node.
        Ok(Tables{nodes:n,data:(0..self.blocks.len()).map(|b|nodes.iter().flat_map(|node|node[b].iter().flat_map(|z|[z.re,z.im])).collect()).collect()})
    }
    /// The coarse grid's (`self.tables`) largest interpolation error at a finer grid's nodes, relative
    /// to the scale of each block's table kind.
    fn interpolation_error(&self,fine:&Tables)->f64{
        let(ng,ne)=(self.ground.ground_states.len(),self.ground.excited_states.len());let points=self.grid(fine.nodes);let mut worst=0_f64;
        for(b,layout)in self.layouts.iter().enumerate(){
            let nb=self.blocks[b].states.len();let mut kinds:Vec<(usize,usize)>=vec![(layout.h0,layout.h0+nb*nb)];
            for(c,cl)in layout.carriers.iter().enumerate(){let r=self.carriers[c].rank;let(square,jump,response)=(pairs(r)*nb*nb,3*r*ng*nb,r*ne*nb);
                kinds.extend([(cl.a,cl.a+square),(cl.q,cl.q+square),(cl.j,cl.j+jump),(cl.j2,cl.j2+jump),(cl.t,cl.t+response)]);}
            for(start,end)in kinds{
                let exact=|i:usize|&fine.data[b][2*(i*layout.size+start)..2*(i*layout.size+end)];
                let modulus=|x:&[f64]|x.chunks_exact(2).map(|z|z[0].hypot(z[1])).fold(0_f64,f64::max);
                let scale=(0..points.len()).map(|i|modulus(exact(i))).fold(0_f64,f64::max);
                if scale==0.{continue;}
                let mut out=vec![Complex::ZERO;end-start];
                for(i,s)in points.iter().enumerate(){
                    self.interpolate(b,&self.stencil(s),None,start,&mut out);
                    worst=worst.max(out.iter().zip(exact(i).chunks_exact(2)).fold(0_f64,|m,(x,y)|m.max((x.re-y[0]).hypot(x.im-y[1])))/scale);
                }
            }
        }
        worst
    }
    /// The tables on grids of FIRST_NODES, then doubled, until the cubic interpolation from the
    /// coarser grid meets TABLE_SHARE at the finer grid's nodes; the finer grid is kept.
    fn tabulate(&mut self,mixing:&[Vec<Vec<Complex>>])->Result<(),String>{
        if self.focused.is_empty(){self.tables=self.build(1,mixing)?;return Ok(());}
        self.tables=self.build(FIRST_NODES,mixing)?;
        loop{
            let n=2*self.tables.nodes-1;let fine=self.build(n,mixing)?;
            let worst=self.interpolation_error(&fine);
            self.tables=fine;self.table_error=worst;
            if worst<=TABLE_SHARE{return Ok(());}
            if n>4097{return Err(format!("The elimination's tables over the focused beams' intensities do not converge (relative error {worst:.1e} at {n} nodes)"));}
        }
    }

    /// The stencil at intensities s (one per focused beam).
    pub fn stencil(&self,s:&[f64])->Stencil{
        let n=self.tables.nodes;let mut out=Stencil{dimensions:self.focused.len(),first:[0;MOST_FOCUSED],weights:[[1.,0.,0.,0.];MOST_FOCUSED],derivatives:[[0.;4];MOST_FOCUSED],stride:[0;MOST_FOCUSED]};
        for f in 0..out.dimensions{let(first,w,d)=cubic(n,s[f]);out.first[f]=first;out.weights[f]=w;out.derivatives[f]=d;out.stride[f]=n.pow(f as u32);}
        out
    }

    /// out = the table slice [start, start + out.len()) of a block, interpolated (with the weights'
    /// derivative in s_f when `derivative` names one).
    pub fn interpolate(&self,block:usize,stencil:&Stencil,derivative:Option<usize>,start:usize,out:&mut [Complex]){
        let size=2*self.layouts[block].size;let data=&self.tables.data[block];let len=out.len();out.fill(Complex::ZERO);
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let w=match derivative{None=>w,Some(f)=>dw[f]};let base=node*size+2*start;
            for(o,x)in out.iter_mut().zip(data[base..base+2*len].chunks_exact(2)){o.re+=w*x[0];o.im+=w*x[1];}
        }
    }
    /// The same into interleaved f64, W values (2 × the entries); the one-beam stencil unrolled.
    #[inline(always)]
    fn gather<const W:usize>(&self,block:usize,stencil:&Stencil,derivative:Option<usize>,start:usize,out:&mut [f64;50]){
        let size=2*self.layouts[block].size;let data=&self.tables.data[block];let base=2*start;
        if stencil.dimensions==1{
            let w=match derivative{None=>stencil.weights[0],Some(_)=>stencil.derivatives[0]};let at=stencil.first[0]*size+base;
            let(a,b,c,d)=(&data[at..at+W],&data[at+size..at+size+W],&data[at+2*size..at+2*size+W],&data[at+3*size..at+3*size+W]);
            for j in 0..W{out[j]=w[0]*a[j]+w[1]*b[j]+w[2]*c[j]+w[3]*d[j];}
            return;
        }
        out[..W].fill(0.);
        for i in 0..stencil.count(){
            let(node,w,dw)=stencil.node(i);let w=match derivative{None=>w,Some(f)=>dw[f]};let at=node*size+base;
            for(o,x)in out[..W].iter_mut().zip(&data[at..at+W]){*o+=w*x;}
        }
    }

    /// Everything at (r, κ) but the internal state.
    pub fn point(&self,r:[f64;3],kappa:[f64;3])->Point{
        let mut s=[0.;MOST_FOCUSED];let mut gradient=[[0.;3];MOST_FOCUSED];
        for(i,&l)in self.focused.iter().enumerate(){let(a,g)=self.geometry[l].field(r);s[i]=a.norm_sqr().min(1.);gradient[i]=g.map(|x|2.*(a.conj()*x).re);}
        let mut fields=[CarrierField::default();MOST_CARRIERS];
        for(c,f)in self.carriers.iter().zip(fields.iter_mut()){
            for(j,&l)in c.beams.iter().enumerate(){
                let(a,g)=self.geometry[l].field(r);let w=self.kinetic_shift(l,kappa);
                for k in 0..c.rank{let x=c.coefficients[j][k];let ax=a*x;f.e[k]+=ax;f.doppler[k]+=ax*w;for d in 0..3{f.grad[k][d]+=g[d]*x;}}
            }
        }
        Point{r,kappa,stencil:self.stencil(&s[..self.focused.len()]),intensity_gradient:gradient,fields}
    }

    /// w_l = (ħ/m) k_l·κ + ħ|k_l|²/2m (rad/us): the kinetic energy absorbing from beam l gives.
    pub fn kinetic_shift(&self,l:usize,kappa:[f64;3])->f64{
        if !self.doppler{return 0.;}
        let g=&self.geometry[l];let k=g.wave_vector();self.hbar_over_mass*(k[0]*kappa[0]+k[1]*kappa[1]+k[2]*kappa[2]+0.5*g.wave_number*g.wave_number)
    }

    /// The block's dimension.
    pub fn dimension(&self,block:usize)->usize{self.blocks[block].states.len()}

    /// A block's generator at a point (`Local`). Γ's tables are Hermitian in (k', k), Q_kk' = Q_k'k†.
    pub fn local(&self,block:usize,p:&Point)->Local{
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(block){$($n=>if self.wide{unsafe{self.local_wide::<$n,{2*$n*$n}>(block,p)}}else{self.local_body::<$n,{2*$n*$n}>(block,p)},)*_=>unreachable!("a hyperfine block has 1 to 5 states")}}}
        sized!(1 2 3 4 5)
    }
    #[target_feature(enable="avx2,fma")]
    unsafe fn local_wide<const N:usize,const W:usize>(&self,block:usize,p:&Point)->Local{self.local_body::<N,W>(block,p)}
    #[inline(always)]
    fn local_body<const N:usize,const W:usize>(&self,block:usize,p:&Point)->Local{
        let layout=&self.layouts[block];let n2=N*N;let d=self.focused.len();
        let mut out=Local{hamiltonian:[Complex::ZERO;25],scalar:0.,potential:[[0.;50];MOST_FOCUSED],shift:[[0.;50];MOST_FOCUSED],field:[[0.;50];3]};
        let mut buf=[0.;50];let(mut m,mut gamma)=([0.;50],[0.;50]);
        // The pair (k, k1)'s tables are (k1, k)'s adjoints: c X + (c X)† for k1 < k. The force reads
        // only Re⟨ψ|·|ψ⟩, so its matrices take 2 c X for the two.
        let adjoint=|c:Complex,x:&[f64;50],acc:&mut [f64;50]|{for a in 0..N{for b in 0..N{let(i,j)=(2*(a*N+b),2*(b*N+a));let(u,v)=(x[j],-x[j+1]);acc[i]+=c.re*u+c.im*v;acc[i+1]+=c.re*v-c.im*u;}}};
        for(c,cl)in layout.carriers.iter().enumerate(){
            let r=self.carriers[c].rank;let f=&p.fields[c];
            for k1 in 0..r{for k in k1..r{
                let at=pair(k1,k,r)*n2;let ee=f.e[k1].conj()*f.e[k];let both=if k1<k{2.}else{1.};
                self.gather::<W>(block,&p.stencil,None,cl.a+at,&mut buf);caxpy::<W>(ee,&buf,&mut m);if k1<k{adjoint(ee,&buf,&mut m);}
                for a in 0..3{caxpy::<W>((f.grad[k1][a].conj()*f.e[k]+f.e[k1].conj()*f.grad[k][a])*both,&buf,&mut out.field[a]);}
                for g in 0..d{self.gather::<W>(block,&p.stencil,Some(g),cl.a+at,&mut buf);caxpy::<W>(ee*both,&buf,&mut out.shift[g]);}
                self.gather::<W>(block,&p.stencil,None,cl.q+at,&mut buf);caxpy::<W>(ee,&buf,&mut gamma);if k1<k{adjoint(ee,&buf,&mut gamma);}
            }}
        }
        self.gather::<W>(block,&p.stencil,None,layout.h0,&mut buf);
        for g in 0..d{let mut slope=[0.;50];self.gather::<W>(block,&p.stencil,Some(g),layout.h0,&mut slope);out.potential[g]=slope;}
        let at=|v:&[f64;50],a:usize,b:usize|Complex::new(v[2*(a*N+b)],v[2*(a*N+b)+1]);
        let mut scalar=0.;for a in 0..N{scalar+=buf[2*(a*N+a)]-m[2*(a*N+a)];}scalar/=N as f64;out.scalar=scalar;
        for a in 0..N{for b in 0..N{
            let herm=at(&buf,a,b)-at(&m,a,b);
            out.hamiltonian[a*N+b]=herm-at(&gamma,a,b)*Complex::new(0.,0.5)-if a==b{Complex::new(scalar,0.)}else{Complex::ZERO};
        }}
        out
    }

    /// −⟨ψ|∇H_herm|ψ⟩/⟨ψ|ψ⟩ (rad/us per um): the Ehrenfest force on ħκ. With H_herm = H0g − Herm M,
    /// M = Σ E_k'* E_k A_k'k, ⟨∇_a H_herm⟩ = Σ_f ⟨∂H0g/∂s_f⟩ ∂_a s_f − Re⟨Σ ∇_a(E_k'* E_k) A_k'k⟩
    /// − Σ_f Re⟨Σ E_k'* E_k ∂A_k'k/∂s_f⟩ ∂_a s_f (the Doppler term's own gradient, (kv/Δ) of the
    /// light shift's, is left out).
    pub fn force(&self,block:usize,p:&Point,local:&Local,psi:&[Complex])->[f64;3]{
        macro_rules! sized{($($n:literal)*)=>{match self.dimension(block){$($n=>self.force_sized::<$n>(p,local,psi),)*_=>unreachable!("a hyperfine block has 1 to 5 states")}}}
        sized!(1 2 3 4 5)
    }
    fn force_sized<const N:usize>(&self,p:&Point,local:&Local,psi:&[Complex])->[f64;3]{
        let d=self.focused.len();let norm:f64=psi[..N].iter().map(|z|z.norm_sqr()).sum();
        let mut gradient=[0.;3];
        for f in 0..d{let v=quadratic::<N>(&local.potential[f],psi).re-quadratic::<N>(&local.shift[f],psi).re;for a in 0..3{gradient[a]+=v*p.intensity_gradient[f][a];}}
        for a in 0..3{gradient[a]-=quadratic::<N>(&local.field[a],psi).re;}
        gradient.map(|g|-g/norm)
    }

    /// The photon amplitudes of every carrier and axis at a point, C_a R_c ψ over all ground states
    /// (in `ground_states` order), with the Doppler term; and the pieces their gradients and the
    /// beams' shares are made of (`Photons::gradient`, `Photons::beams`).
    pub fn photons(&self,block:usize,p:&Point,psi:&[Complex])->Photons{
        let nb=self.dimension(block);let layout=&self.layouts[block];let ng=self.ground.ground_states.len();
        let mut out=Photons{block,amplitude:vec![[[Complex::ZERO;GROUND];3];self.carriers.len()],responses:vec![],derivatives:vec![]};
        let mut buf=vec![Complex::ZERO;ng*nb];
        for(c,cl)in layout.carriers.iter().enumerate(){
            let r=self.carriers[c].rank;let fl=&p.fields[c];
            let(mut responses,mut derivatives)=([[[Complex::ZERO;GROUND];3];MOST_RANK],[[[Complex::ZERO;GROUND];3];MOST_RANK]);
            for k in 0..r{for a in 0..3{
                let at=(k*3+a)*ng*nb;
                self.interpolate(block,&p.stencil,None,cl.j+at,&mut buf);responses[k][a]=apply(&buf,psi,ng,nb);
                self.interpolate(block,&p.stencil,None,cl.j2+at,&mut buf);derivatives[k][a]=apply(&buf,psi,ng,nb);
                for row in 0..ng{out.amplitude[c][a][row]+=fl.e[k]*responses[k][a][row]-fl.doppler[k]*derivatives[k][a][row];}
            }}
            out.responses.push(responses);out.derivatives.push(derivatives);
        }
        out
    }
    /// The trap's frequencies at the focus along each axis of the frame (rad/us) for block state x:
    /// sqrt(ħ/m · d²U/dx_a²) of its potential (`potential`), by central differences over a
    /// thousandth of the narrowest waist.
    pub fn trap_frequencies(&self,block:usize,x:usize)->[f64;3]{
        let waist=self.focused.iter().filter_map(|&l|self.geometry[l].waist).fold(f64::INFINITY,f64::min);
        let h=if waist.is_finite(){1e-3*waist}else{1e-3};
        std::array::from_fn(|a|{let mut r=[0.;3];r[a]=h;let plus=self.potential(block,x,r);r[a]=-h;let minus=self.potential(block,x,r);
            ((plus+minus)/(h*h)*self.hbar_over_mass).max(0.).sqrt()})
    }

    /// The 32 populations (request order) of a block's state ψ at a point: the bare ground ones,
    /// ρ − ½{Y, ρ} with Y = Σ_c R_c†R_c, and the excited ones, Σ_c |(R_c ψ)_i|² (ground.rs's readout,
    /// with the responses at rest), for ρ = ψψ†/‖ψ‖².
    pub fn populations(&self,block:usize,p:&Point,psi:&[Complex])->[f64;STATES]{
        let g=&self.ground;let nb=self.dimension(block);let layout=&self.layouts[block];let ne=g.excited_states.len();
        let norm:f64=psi[..nb].iter().map(|z|z.norm_sqr()).sum();
        let mut out=[0.;STATES];let mut y=[Complex::ZERO;MOST_STATES];let mut buf=vec![Complex::ZERO;ne*nb];
        for(c,cl)in layout.carriers.iter().enumerate(){
            let r=self.carriers[c].rank;let f=&p.fields[c];
            // R_c ψ = Σ_k E_k T_k ψ, then R_c† (R_c ψ).
            let mut e=vec![Complex::ZERO;ne];let mut tk:Vec<Vec<Complex>>=vec![];
            for k in 0..r{self.interpolate(block,&p.stencil,None,cl.t+k*ne*nb,&mut buf);
                for i in 0..ne{let mut s=Complex::ZERO;for x in 0..nb{s+=buf[i*nb+x]*psi[x];}e[i]+=f.e[k]*s;}tk.push(buf.clone());}
            for k in 0..r{for x in 0..nb{let mut s=Complex::ZERO;for i in 0..ne{s+=tk[k][i*nb+x].conj()*e[i];}y[x]+=f.e[k].conj()*s;}}
            for(i,&state)in g.excited_states.iter().enumerate(){out[state]+=e[i].norm_sqr()/norm;}
        }
        for(x,&b)in self.blocks[block].states.iter().enumerate(){out[g.ground_states[b]]=(psi[x].norm_sqr()-(psi[x].conj()*y[x]).re)/norm;}
        out
    }

    /// The trap: block state x's potential at r relative to the focus (rad/us), the diagonal of H0g
    /// and of every beam's own light shift, |a_l(r)|² Σ c_lk'* c_lk A_k'k -- the beams' interference
    /// (their lattice) left out. A tweezer folded at second order is in H0g; one that is the frame's
    /// carrier, as when it is the only light, a near carrier: the trap is the same.
    pub fn potential(&self,block:usize,x:usize,r:[f64;3])->f64{
        let at=|r:[f64;3]|->f64{
            let nb=self.dimension(block);let layout=&self.layouts[block];let n2=nb*nb;
            let s:Vec<f64>=self.focused.iter().map(|&l|self.geometry[l].field(r).0.norm_sqr().min(1.)).collect();let stencil=self.stencil(&s);
            let mut buf=[Complex::ZERO;25];self.interpolate(block,&stencil,None,layout.h0,&mut buf[..n2]);let mut u=buf[x*nb+x].re;
            for(c,cl)in layout.carriers.iter().enumerate(){
                let carrier=&self.carriers[c];let rank=carrier.rank;
                for(j,&l)in carrier.beams.iter().enumerate(){
                    let intensity=self.geometry[l].field(r).0.norm_sqr();let coefficient=&carrier.coefficients[j];
                    for k1 in 0..rank{for k in k1..rank{
                        self.interpolate(block,&stencil,None,cl.a+pair(k1,k,rank)*n2,&mut buf[..n2]);
                        u-=intensity*(coefficient[k1].conj()*coefficient[k]*buf[x*nb+x]).re*if k1<k{2.}else{1.};
                    }}
                }
            }
            u
        };
        at(r)-at([0.;3])
    }
}

/// M ψ for an ng × nb matrix M (row-major) and a block state ψ.
fn apply(m:&[Complex],psi:&[Complex],ng:usize,nb:usize)->[Complex;GROUND]{
    let mut v=[Complex::ZERO;GROUND];for row in 0..ng{let mut s=Complex::ZERO;for y in 0..nb{s+=m[row*nb+y]*psi[y];}v[row]=s;}v
}

/// The photon amplitudes C_a R_c ψ of every carrier c and axis a (over all ground states), and per
/// carrier, rank k and axis C_a T_k ψ and C_a T2_k ψ.
#[derive(Clone,Debug)]
pub struct Photons{pub block:usize,pub amplitude:Vec<[[Complex;GROUND];3]>,responses:Vec<[[[Complex;GROUND];3];MOST_RANK]>,derivatives:Vec<[[[Complex;GROUND];3];MOST_RANK]>}
impl Photons{
    /// ∇(C_a R_c ψ) [a][axis] of carrier c: the fields' and the intensities' dependence on r (the
    /// Doppler term's own gradient left out).
    pub fn gradient(&self,model:&Model,p:&Point,psi:&[Complex],c:usize)->[[[Complex;GROUND];3];3]{
        let nb=model.dimension(self.block);let ng=model.ground.ground_states.len();let cl=model.layouts[self.block].carriers[c];
        let r=model.carriers[c].rank;let fl=&p.fields[c];let mut out=[[[Complex::ZERO;GROUND];3];3];let mut buf=vec![Complex::ZERO;ng*nb];
        for k in 0..r{for a in 0..3{
            for row in 0..ng{for x in 0..3{out[a][x][row]+=fl.grad[k][x]*self.responses[c][k][a][row];}}
            for f in 0..model.focused.len(){
                model.interpolate(self.block,&p.stencil,Some(f),cl.j+(k*3+a)*ng*nb,&mut buf);let dv=apply(&buf,psi,ng,nb);
                for row in 0..ng{for x in 0..3{out[a][x][row]+=fl.e[k]*dv[row]*p.intensity_gradient[f][x];}}
            }
        }}
        out
    }
    /// Each beam l of carrier c: its share C_a a_l (R_l − w_l R2_l) ψ [beam][a].
    pub fn beams(&self,model:&Model,p:&Point,c:usize)->Vec<[[Complex;GROUND];3]>{
        let carrier=&model.carriers[c];let ng=model.ground.ground_states.len();
        carrier.beams.iter().enumerate().map(|(j,&l)|{
            let(amplitude,w)=(model.geometry[l].field(p.r).0,model.kinetic_shift(l,p.kappa));let mut v=[[Complex::ZERO;GROUND];3];
            for k in 0..carrier.rank{let z=amplitude*carrier.coefficients[j][k];
                for a in 0..3{for row in 0..ng{v[a][row]+=z*(self.responses[c][k][a][row]-self.derivatives[c][k][a][row]*w);}}}
            v
        }).collect()
    }
}
