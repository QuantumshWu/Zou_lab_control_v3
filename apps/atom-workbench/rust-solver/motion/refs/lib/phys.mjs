// Physics and numerics shared by the references: constants (SI 2019, as motion/src/units.rs), special functions,
// quadrature, the request's own atomic data (decay rates, couplings, detunings, read from the native request so that a
// reference describes exactly the model the solver receives), the paraxial Gaussian beam, and the initial ensemble's
// exact moments (the lowered isothermal times each axis' Wigner factor, SPEC §2.5).
export const H=6.62607015e-34,HBAR=H/(2*Math.PI),KB=1.380649e-23,C=299792458;
/** rad/µs per MHz; k_B/ħ in rad/µs per µK (units.rs KT_PER_UK). */
export const RAD_PER_MHZ=2*Math.PI,KT_PER_UK=KB/HBAR*1e-12;
const N=32;

// ---------------------------------------------------------------------------------------------------------------------
// Special functions and quadrature.

/** erf(x) to ~1e-16 relative: its Maclaurin series below 2.5, else 1 − erfc by the continued fraction (Lentz). */
export function erf(x){
  if(x<0)return -erf(-x);
  if(x<2.5){let term=x,sum=x;for(let n=1;n<200;n++){term*=-x*x/n;const add=term/(2*n+1);sum+=add;if(Math.abs(add)<1e-17*Math.abs(sum))break;}return 2/Math.sqrt(Math.PI)*sum;}
  return 1-erfc(x);
}
/** erfc(x) for x ≥ 2.5 by its continued fraction (modified Lentz). */
export function erfc(x){
  if(x<2.5)return 1-erf(x);
  const tiny=1e-300;let f=x,c=x,d=0;
  for(let n=1;n<500;n++){const a=n/2;d=x+a*d;d=Math.abs(d)<tiny?tiny:d;c=x+a/c;c=Math.abs(c)<tiny?tiny:c;d=1/d;const delta=c*d;f*=delta;if(Math.abs(delta-1)<1e-16)break;}
  return Math.exp(-x*x)/(Math.sqrt(Math.PI)*f);
}
/** Γ(s) at a positive multiple of ½. */
export function gammaHalf(s){let g=(2*s)%2===0?1:Math.sqrt(Math.PI),t=(2*s)%2===0?1:0.5;while(t<s-0.25){g*=t;t+=1;}return g;}
/** The lowered isothermal's weight K(s, y) = e^y P(s, y) − y^s/Γ(s + 1) for s a positive half-integer, y ≥ 0 (ensemble
 * module of the old crate, SPEC §2.5): by its series e^y P(s,y) = y^s/Γ(s+1) Σ_{n≥0} y^n/((s+1)…(s+n)), the first term
 * cancelled exactly: K = y^s/Γ(s+1) Σ_{n≥1} y^n/((s+1)…(s+n)); scaled by e^{−shift} against overflow. */
export function lowered(s,y,shift=0){
  if(!(y>0))return 0;
  if(y<s+40){
    let term=1,sum=0,a=s;for(let n=1;n<2000;n++){a+=1;term*=y/a;sum+=term;if(term<1e-17*sum)break;}
    return Math.exp(s*Math.log(y)-shift)/gammaHalf(s+1)*sum;
  }
  // e^y P(s,y) − y^s/Γ(s+1) with e^y P = e^y − e^y Q(s,y), e^y Q = y^s h/Γ(s), h the incomplete gamma function's
  // continued fraction (Numerical Recipes gcf, Lentz); y large: e^y dominates, no cancellation.
  const tiny=1e-300;let b=y+1-s,c=1/tiny,d=1/b,h=d;
  for(let i=1;i<2000;i++){const an=-i*(i-s);b+=2;d=an*d+b;if(Math.abs(d)<tiny)d=tiny;c=b+an/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;const del=d*c;h*=del;if(Math.abs(del-1)<1e-16)break;}
  const power=Math.exp(s*Math.log(y)-shift);
  return Math.exp(y-shift)-power*(h/gammaHalf(s)+1/gammaHalf(s+1));
}
/** Gauss–Legendre nodes and weights on [a, b] (Newton on P_n). */
const legendreCache=new Map();
export function gaussLegendre(n,a=-1,b=1){
  let base=legendreCache.get(n);
  if(!base){base=[];for(let i=0;i<n;i++){let x=Math.cos(Math.PI*(i+0.75)/(n+0.5)),slope=1;for(let it=0;it<100;it++){let p0=1,p1=x;for(let k=2;k<=n;k++){const p2=((2*k-1)*x*p1-(k-1)*p0)/k;p0=p1;p1=p2;}slope=n*(x*p1-p0)/(x*x-1);const dx=p1/slope;x-=dx;if(Math.abs(dx)<=1e-15)break;}base.push([x,2/((1-x*x)*slope*slope)]);}legendreCache.set(n,base);}
  const half=(b-a)/2,mid=(a+b)/2;return base.map(([x,w])=>[mid+half*x,half*w]);
}
/** ∫_a^b f by composite Gauss–Legendre on `panels` equal panels of n points. */
export function integrate(f,a,b,{panels=16,n=16}={}){
  let s=0;const h=(b-a)/panels;for(let p=0;p<panels;p++)for(const [x,w] of gaussLegendre(n,a+p*h,a+(p+1)*h))s+=w*f(x);return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// The native request's atomic data.

/** The request's states and the index of a label. */
export function states(line){return line.model.states;}
export function index(line,label){const i=line.model.states.indexOf(label);if(i<0)throw new Error(`no state ${label}`);return i;}
const entry=(m,i,j)=>({re:m.real[i*N+j],im:m.imag[i*N+j]});
/** Γ_e (1/µs) of every state (0 for the ground states): Σ_a Σ_g |C_a[g, e]|² over the lab-axis emission operators. */
export function decayRates(line){
  const em=line.model.motion.emission,out=Array(N).fill(0);
  for(let e=0;e<N;e++)for(let g=0;g<N;g++)for(const c of em){const z=entry(c,g,e);out[e]+=z.re*z.re+z.im*z.im;}
  return out;
}
/** The polarization-summed branching rate Γ_{e→g} (1/µs): Σ_a |C_a[g, e]|². */
export function branchRate(line,e,g){let s=0;for(const c of line.model.motion.emission){const z=entry(c,g,e);s+=z.re*z.re+z.im*z.im;}return s;}
/** The co-rotating coupling X^l_{eg} of beam `id` at the focus (rad/µs): the drive is X e^{−iΔt} + h.c., so the Rabi
 * frequency of the transition is Ω = 2|X|. */
export function coupling(line,id,e,g){const b=line.model.motion.beams.find(x=>x.id===id);if(!b)throw new Error(`no beam ${id}`);return entry(b.coupling,e,g);}
/** A beam's detuning from the excited states' frame, Δ_l = 2π Σ_k index_k ν_k (rad/µs). */
export function frameDetuning(line,id){const b=line.model.motion.beams.find(x=>x.id===id),f=line.model.frequencyCoordinates.frequenciesMHz;return RAD_PER_MHZ*b.frequencyIndex.reduce((s,n,k)=>s+n*f[k],0);}
/** H0's diagonal (rad/µs) in the frames. */
export function frameEnergies(line){const h=line.model.motion.hamiltonian;return line.model.states.map((_,i)=>h.real[i*N+i]);}

// ---------------------------------------------------------------------------------------------------------------------
// Units.

/** An energy given as a frequency shift (MHz) in µK: h·ν/k_B. */
export const uKOfMHz=mhz=>H*mhz*1e6/KB*1e6;
/** ħ/m in µm²/µs; the recoil energy E_rec = ħk²/2m of wave number k (rad/µm) as a rate (rad/µs) and in µK. */
export const hbarOverMass=m=>HBAR/m*1e6;
export const recoilRate=(k,m)=>hbarOverMass(m)*k*k/2;
export const recoilUK=(k,m)=>recoilRate(k,m)/KT_PER_UK;

// ---------------------------------------------------------------------------------------------------------------------
// The paraxial Gaussian beam and the trap.

/** Rayleigh range z_R = k w₀²/2 (µm). */
export const rayleigh=(k,w0)=>k*w0*w0/2;
/** A single focused beam's trap of depth U₀ (rad/µs) for mass m: ω_ρ = √(4U₀/(m w₀²)), ω_z = √(2U₀/(m z_R²)) (rad/µs):
 * the Hessian of −U₀ (w₀/w)² e^{−2ρ²/w²} at the focus (∂²/∂ρ²: 4U₀/w₀²; ∂²/∂z²: 2U₀/z_R²). ħ/m converts the energy rate. */
export function beamTrap(U0,w0,k,m){const r=hbarOverMass(m);return {radial:Math.sqrt(4*U0*r/(w0*w0)),axial:Math.sqrt(2*U0*r/(rayleigh(k,w0)**2))};}
/** The thermal harmonic oscillator's Wigner temperature T* = (ħω/2k_B) coth(ħω/2k_BT) (µK) at ω (rad/µs), T (µK). */
export function wignerTemperature(omega,T){const h=omega/(2*KT_PER_UK*T);return h>1e-8?T*h/Math.tanh(h):T;}

/** The lowered isothermal's classical kinetic temperature over T, T_kin/T = I(5/2)/I(3/2), I(s) = ∫ d³r K(s, y(r)), y the
 * depth below the far level in k_BT at r (SPEC §2.5; old ensemble.rs): at r the bound momenta's weight is K(3/2, y) and
 * their kinetic energy K(5/2, y)·k_BT·Γ(5/2)/Γ(3/2) = (3/2)K(5/2, y)k_BT, so ⟨KE⟩ = (3/2)k_BT I(5/2)/I(3/2) and
 * T_kin = (2/3)⟨KE⟩/k_B. `depth(r)` gives y; `measure` integrates over space (see beamMeasure, latticeMeasure). */
export function kineticRatio(measure){
  const top=measure(y=>lowered(2.5,y,measure.shift)),bottom=measure(y=>lowered(1.5,y,measure.shift));return top/bottom;
}
/** The space integral ∫ d³r F(y(r)) for a single Gaussian beam of peak depth y₀ (k_BT units): with u = 2ρ²/w(z)² the
 * radial integral is (π w(z)²/2) ∫_0^{y_z} F(t) dt/t, y_z = y₀/(1 + z²/z_R²); with z = z_R tan θ the axial one is
 * (π w₀² z_R/2) ∫ sec⁴θ G(y₀ cos²θ) dθ over (−π/2, π/2), whose integrand ~ cos^{2s−2}θ at the ends (finite). */
export function beamMeasure(y0){
  const m=F=>{
    const G=Y=>{if(!(Y>0))return 0;let s=0;const lo=Math.max(Y*1e-12,1e-300);
      // ∫_0^Y F(t)/t dt with t = Y e^{−v}, v ∈ (0, ln(Y/lo)): F(t)/t dt = F(t) dv.
      const V=Math.log(Y/lo);for(const [v,w] of gaussLegendre(48,0,Math.min(V,6)))s+=w*F(Y*Math.exp(-v));
      if(V>6)for(const [v,w] of gaussLegendre(48,6,V))s+=w*F(Y*Math.exp(-v));return s;};
    return integrate(th=>{const c=Math.cos(th);return G(y0*c*c)/(c*c*c*c);},-Math.PI/2,Math.PI/2,{panels:24,n:24});
  };
  m.shift=y0;return m;
}
