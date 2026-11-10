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

// Double-precision arithmetic that carries its rounding: what the evaluation of an exact formula leaves uncertain.

/** γ_n = nε/(1 − nε), ε = 2⁻⁵³ the relative rounding of one double-precision operation: n of them in composition,
 * (1 + δ₁)…(1 + δ_n) with |δ_i| ≤ ε, stay within 1 ± γ_n (Higham, Accuracy and Stability of Numerical Algorithms, 2nd
 * ed., Lemma 3.1). */
export const roundoff=n=>n*2**-53/(1-n*2**-53);
/** A complex number as a formula's evaluation: `re`, `im` the computed value; `m` no less than the sum of the moduli of
 * the products of given numbers that its exact value is the sum of; `n` no less than the number of rounded operations
 * any of them has passed through: |computed − exact| ≤ γ_n m, to first order in ε. A given number is exact (n = 0). */
const exact=(re,im=0)=>({re,im,m:Math.hypot(re,im),n:0});
/** A sum: each partial sum is one more rounded operation on all it holds; the m add. */
const sum=(...z)=>z.reduce((a,b)=>({re:a.re+b.re,im:a.im+b.im,m:a.m+b.m,n:Math.max(a.n,b.n)+1}));
/** A product: each part is a rounded sum of two rounded products, two more operations on each factor's; the m multiply. */
const product=(...z)=>z.reduce((a,b)=>({re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re,m:a.m*b.m,n:a.n+b.n+2}));
/** Exact: the conjugate, the opposite, the product with i. */
const conjugate=a=>({...a,im:-a.im}),opposite=a=>({...a,re:-a.re,im:-a.im}),turned=a=>({...a,re:-a.im,im:a.re});
/** 1/a = conj(a)/|a|²: three more operations (|a|²'s product and sum, the division); a's own bound γ_n m is γ_n m/|a|²
 * of 1/a. */
const inverse=a=>{const d=a.re*a.re+a.im*a.im;return {re:a.re/d,im:-a.im/d,m:a.m/d,n:a.n+3};};
/** e^a: the argument's bound γ_n m is the result's relative one, that of ⌈m⌉n operations; the library's exp, cos and
 * sin (each within one unit in the last place, 2ε) and their product are five more. */
const exponential=a=>{const e=Math.exp(a.re);return {re:e*Math.cos(a.im),im:e*Math.sin(a.im),m:e,n:Math.ceil(a.m)*a.n+5};};
/** The bound γ_n m of a carried number. */
const bound=z=>roundoff(z.n)*z.m;

/** A x = y for a 3 × 3 matrix A: Gauss–Jordan elimination with partial pivoting. */
export function solve3(A,y){
  const a=A.map((r,i)=>[...r,y[i]]);
  for(let c=0;c<3;c++){let p=c;for(let r=c+1;r<3;r++)if(Math.abs(a[r][c])>Math.abs(a[p][c]))p=r;[a[c],a[p]]=[a[p],a[c]];for(let r=0;r<3;r++)if(r!==c){const f=a[r][c]/a[c][c];for(let k=c;k<4;k++)a[r][k]-=f*a[c][k];}}
  return a.map((r,i)=>r[3]/r[i]);
}

// The trap light's potential of a ground state and its trap as the solver's model states them (SPEC §1.2, §1.3), from
// the request alone.

/** The trap light's static shift of the ground state `label` (SPEC §1.2): V(r) = Σ_l′l Re[conj(a_l′(r)) a_l(r) Q_l′l] over
 * the beams of the trap beam's frequency, a_l the beam's amplitude over its focus value (`beamJet`), and (rad/µs)
 *   Q_l′l = Σ_e conj(X^l′_eb) X^l_eb/(E_b + Δ − E_e) + Σ_e conj(R^l_eb) R^l′_eb/(E_b − Δ − 2ω_f − E_e) + Σ_pq conj(E^l′_p) E^l_q [O_pq]_bb.
 * The light drives Σ_l a_l X^l e^{−iΔt} + Σ_l conj(a_l) R^l e^{+i(Δ + 2ω_f)t} + h.c. (excited ← ground; its co- and its
 * counter-rotating part in the request's frame, Δ the frequency's detuning from the frame and ω_f the frame's own, the
 * request's motion block): in second order a level is shifted by |coupling|² over the difference of its energy with
 * the photon and the excited state's -- E the request's H0, real and exact --, summed over the excited states: the
 * co-rotating conj(Σ a_l′X^l′)(Σ a_l X^l) and the counter-rotating conj(Σ conj(a_l)R^l)(Σ conj(a_l′)R^l′) both hold
 * conj(a_l′) a_l. The states outside the model shift it by the request's form in the field E = Σ_l a_l E^l,
 * Σ_pq conj(E_p) E_q O_pq on the spherical components (O_qq the |E_q|² matrix; for p < q, O_pq = (X − iY)/2 and
 * O_qp = (X + iY)/2 of the pair's Re and Im matrices, so that the pair gives Re(conj(E_p)E_q) X + Im(conj(E_p)E_q) Y).
 * Returns the beams and Q, row-major over them, each with its rounding. */
export function trapForm(line,label){
  const m=line.model.motion,b=index(line,label),E=frameEnergies(line),of=x=>String(x.frequencyIndex),trap=m.beams.find(x=>x.id===m.trapBeamId);
  const beams=m.beams.filter(x=>of(x)===of(trap)),O=m.outsideForms.find(x=>of(x)===of(trap)).matrices,given=z=>exact(z.re,z.im);
  const delta=product(exact(RAD_PER_MHZ),sum(exact(0),...trap.frequencyIndex.map((n,k)=>product(exact(n),exact(line.model.frequencyCoordinates.frequenciesMHz[k])))));
  const pair=(first,second)=>{
    const terms=[exact(0)];
    for(let e=0;e<N;e++){
      const frame=product(exact(2*RAD_PER_MHZ),sum(exact(m.frameFrequenciesMHz[e]),exact(-m.frameFrequenciesMHz[b])));
      for(const [x,y,d] of [[entry(first.coupling,e,b),entry(second.coupling,e,b),sum(exact(E[b]),delta,exact(-E[e]))],
        [entry(second.counterCoupling,e,b),entry(first.counterCoupling,e,b),sum(exact(E[b]),opposite(delta),opposite(frame),exact(-E[e]))]])
        if((x.re||x.im)&&(y.re||y.im))terms.push(product(conjugate(given(x)),given(y),inverse(d)));
    }
    for(let p=0;p<3;p++)for(let q=0;q<3;q++){
      const at=3+2*(p+q-1),Y=p===q?null:turned(given(entry(O[at+1],b,b)));
      const form=p===q?given(entry(O[q],b,b)):product(exact(0.5),sum(given(entry(O[at],b,b)),p<q?opposite(Y):Y));
      terms.push(product(conjugate(given(first.electricField[p])),given(second.electricField[q]),form));
    }
    return sum(...terms);
  };
  return {beams,Q:beams.flatMap(first=>beams.map(second=>pair(first,second)))};
}
/** A paraxial Gaussian beam focused at the origin (SPEC §1.2): a = e^{ikζ} exp(−ρ²/(w₀²q))/q, q = 1 + iζ/z_R, ζ = n·r,
 * t = r − ζn, ρ² = |t|². With u = 1/q and s = ρ²/w₀², its logarithm φ = ikζ − su + ln u has (∇ζ = n, ∇t = 1 − nnᵀ,
 * ∇ρ² = 2t, ∇u = −(i/z_R)u² n)
 *   g = ∇φ = i[k + (su² − u)/z_R] n − (2u/w₀²) t,
 *   G = ∇g = c₁(1 − nnᵀ) + c₂(ntᵀ + tnᵀ) + c₃nnᵀ,  c₁ = −2u/w₀², c₂ = 2iu²/(z_R w₀²), c₃ = (2su³ − u²)/z_R²,
 *   T = ∇G: T_abc = c₂(n_c P_ab + n_a P_bc + n_b P_ac) + c₄(n_a n_b t_c + n_a t_b n_c + t_a n_b n_c) + c₅ n_a n_b n_c,
 *       P = 1 − nnᵀ, c₄ = 4u³/(z_R²w₀²), c₅ = i(2u³ − 6su⁴)/z_R³
 * (∇c₁ = c₂n, ∇c₂ = c₄n, ∇c₃ = c₄t + c₅n). Returns a and the components of g, G, T at r, each with its rounding. */
function beamJet(beam,r){
  const length=Math.hypot(...beam.direction),n=beam.direction.map(x=>({...exact(x/length),n:3})),at=r.map(x=>exact(x)),axes=[0,1,2];
  const k=exact(beam.waveNumberPerUm),w2=product(exact(beam.waistUm),exact(beam.waistUm)),zR=product(exact(0.5),k,w2);
  const zeta=sum(...axes.map(a=>product(n[a],at[a]))),t=axes.map(a=>sum(at[a],opposite(product(zeta,n[a]))));
  const u=inverse(sum(exact(1),turned(product(zeta,inverse(zR))))),s=product(sum(...t.map(x=>product(x,x))),inverse(w2));
  const along=turned(sum(k,product(sum(product(s,u,u),opposite(u)),inverse(zR)))),c1=product(exact(-2),u,inverse(w2));
  const c2=turned(product(exact(2),u,u,inverse(product(zR,w2)))),c3=product(sum(product(exact(2),s,u,u,u),opposite(product(u,u))),inverse(product(zR,zR)));
  const c4=product(exact(4),u,u,u,inverse(product(zR,zR,w2))),c5=turned(product(sum(product(exact(2),u,u,u),opposite(product(exact(6),s,u,u,u,u))),inverse(product(zR,zR,zR))));
  const P=(a,b)=>sum(exact(+(a===b)),opposite(product(n[a],n[b])));
  return {a:product(exponential(sum(turned(product(k,zeta)),opposite(product(s,u)))),u),
    g:a=>sum(product(along,n[a]),product(c1,t[a])),
    G:(a,b)=>sum(product(c1,P(a,b)),product(c2,sum(product(n[a],t[b]),product(t[a],n[b]))),product(c3,n[a],n[b])),
    T:(a,b,c)=>sum(product(c2,sum(product(n[c],P(a,b)),product(n[a],P(b,c)),product(n[b],P(a,c)))),
      product(c4,sum(product(n[a],n[b],t[c]),product(n[a],t[b],n[c]),product(t[a],n[b],n[c]))),product(c5,n[a],n[b],n[c]))};
}
/** V and its derivatives at r. A pair's coefficient c = conj(a_l′) a_l is the exponential of conj(φ_l′) + φ_l, so with
 * γ = conj(g_l′) + g_l, Γ = conj(G_l′) + G_l, Θ = conj(T_l′) + T_l (`beamJet`):
 *   V = Σ Re[Qc], ∂_aV = Σ Re[Qc γ_a], ∂_a∂_bV = Σ Re[Qc (γ_aγ_b + Γ_ab)],
 *   ∂_b∂_a²V = Σ Re[Qc (γ_a²γ_b + 2γ_aΓ_ab + γ_bΓ_aa + Θ_aab)] (`third[a][b]`),
 * each a carried number: its real part the value, its bound the rounding. */
function trapAt(form,r){
  const jets=form.beams.map(b=>beamJet(b,r)),axes=[0,1,2],out={value:[exact(0)],gradient:axes.map(()=>[exact(0)]),hessian:axes.map(()=>axes.map(()=>[exact(0)])),third:axes.map(()=>axes.map(()=>[exact(0)]))};
  jets.forEach((first,j)=>jets.forEach((second,i)=>{
    const w=product(form.Q[j*jets.length+i],conjugate(first.a),second.a),g=axes.map(a=>sum(conjugate(first.g(a)),second.g(a))),G=(a,b)=>sum(conjugate(first.G(a,b)),second.G(a,b));
    out.value.push(w);
    for(const a of axes){
      out.gradient[a].push(product(w,g[a]));
      for(const b of axes){
        out.hessian[a][b].push(product(w,sum(product(g[a],g[b]),G(a,b))));
        out.third[a][b].push(product(w,sum(product(g[a],g[a],g[b]),product(exact(2),g[a],G(a,b)),product(g[b],G(a,a)),conjugate(first.T(a,a,b)),second.T(a,a,b))));
      }
    }
  }));
  return {value:sum(...out.value),gradient:out.gradient.map(z=>sum(...z)),hessian:out.hessian.map(row=>row.map(z=>sum(...z))),third:out.third.map(row=>row.map(z=>sum(...z)))};
}
/** The prepared trap of the ground state `label` (SPEC §1.3): the trap light's potential V (`trapForm`), its centre --
 * the minimum the descent from the focus reaches, here by Newton's steps on the analytic gradient for as long as each is
 * shorter than the one before (quadratic convergence, down to the rounding) --, and the frequencies along the lab axes
 * there, ω_a = √((ħ/m) ∂_a²V) (rad/µs). Returns the beams, the centre (µm), V's depth below the far level at the focus
 * and at the centre (rad/µs), ω and its uncertainty u, a bound of what a double-precision evaluation leaves open:
 * - the rounding of ∂_a²V's evaluation (the arithmetic's own bound) and of ω's from it (ħ/m, the product, the root and
 *   the conversion to kHz: six operations);
 * - the centre: a descent ends where V no longer falls, and V is resolved to its rounding η only, so that every point
 *   with V within η of its least value is the centre as well as any other -- within |δr| ≤ √(2η/λ) of the minimum, λ
 *   the Hessian's least eigenvalue (no less than min_a(H_aa − Σ_b≠a |H_ab|), Gershgorin), plus the |∇V|'s rounding over
 *   λ by which this evaluation's own zero of the gradient is off; over that ball ∂_a²V changes by at most
 *   |∇∂_a²V| |δr| (the next order is k|δr| of it, k the light's wave number). Where the centre is the focus by
 *   symmetry, ∇∂_a²V = 0 and nothing of it remains. */
export function preparedTrap(line,label){
  const form=trapForm(line,label),real=z=>z.re;
  let centre=[0,0,0],at=trapAt(form,centre),last=Infinity;
  const focus=-at.value.re;
  for(;;){
    const step=solve3(at.hessian.map(row=>row.map(real)),at.gradient.map(z=>-z.re)),size=Math.hypot(...step);
    if(!(size<last))break;
    last=size;centre=centre.map((x,a)=>x+step[a]);at=trapAt(form,centre);
  }
  const H=at.hessian.map(row=>row.map(real)),least=Math.min(...H.map((row,a)=>row[a]-row.reduce((s,x,b)=>s+(b===a?0:Math.abs(x)),0)));
  if(!(least>0))throw new Error(`${line.id}: the trap light's potential of ${label} has no minimum where its descent from the focus ends`);
  const off=Math.sqrt(2*bound(at.value)/least)+Math.hypot(...at.gradient.map(bound))/least,omegas=H.map((row,a)=>Math.sqrt(hbarOverMass(line.model.motion.massKg)*row[a]));
  return {beams:form.beams,centre,focus,depth:-at.value.re,omegas,
    u:omegas.map((w,a)=>w*((bound(at.hessian[a][a])+Math.hypot(...at.third[a].map(real))*off)/(2*H[a][a])+roundoff(6)))};
}
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
