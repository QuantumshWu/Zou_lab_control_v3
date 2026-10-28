//! The second-order blocks of the error propagation: per mode q of the
//! lattice, D'_q = L0' - i nu_q + sum_s K_s over the steps s whose
//! neighbour q + s is on the lattice (`kernels`). Modes with the same steps
//! share one eigensystem (their blocks differ by -i nu_q only): a class.
//!
//! In eigen-coordinates of L0' a class is diag(lambda) + sum_s K~_s, whose
//! eigendecomposition V diag(mu) V^-1 propagates any source exactly. The
//! class also stores its readouts: the population rows of E V (the mode's
//! own populations) and, per step s, of E X~_s V (what the first-order
//! transform I + S carries from the mode to its neighbour q + s).
//!
//! A class whose steps come in pairs (s, -s) is invariant under the flip J,
//! so its matrix is real in the real eigen-coordinates of L0' (the pair
//! rule of `eigenbasis`): real eigendecompositions and LUs, about a quarter
//! of the complex ones. Most couplings of diag(lambda) + K~ are far below
//! their gaps, so the eigensystem is built by clusters of strongly coupled
//! coordinates, exact within each and perturbative between them, with the
//! mixing and the values' error over the request's horizon bounded
//! (`clustered_eigensystem`).
use super::super::*;
use super::super::eigenbasis::BlockEigenbasis;
use super::super::floquet::inverse;
use super::kernels::SwKernels;
use crate::parallel::Execution;

pub struct BlockSystem{
    /// The steps whose second-order terms the class carries.
    pub steps:Vec<usize>,
    /// Eigenvalues mu (the mode's own -i nu_q not included), the
    /// eigenvectors V and V^-1 (column-major d x d, eigen-coordinates of L0').
    pub values:Vec<C>,pub vectors:Vec<C>,pub inverse:Vec<C>,
    /// Population rows (column-major n x d) of E V, and of E X~_s V per step
    /// (None for steps outside the class).
    pub readout:Vec<C>,pub step_readout:Vec<Option<Vec<C>>>,
    /// |V|_1 |V^-1|_1, and the largest cluster of the eigensystem.
    pub condition:f64,pub largest_cluster:usize,
}

impl BlockSystem{
    /// `horizon`: the longest output time (the clusters' error control).
    pub fn new(lapack:&Lapack,basis:&BlockEigenbasis,kernels:&SwKernels,steps:Vec<usize>,n:usize,horizon:f64)->Result<Self,String>{
        let d=basis.dimension();
        let mut m=vec![ZERO;d*d];
        for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}
        for &s in &steps{kernels.add_second_order(s,&mut m);}
        let closed=steps.iter().all(|&s|steps.contains(&kernels.steps[s].opposite));
        let(values,vectors,inv,largest)=if closed{clustered_eigensystem(lapack,basis,&m,d,horizon)?}else{
            let(values,vectors)=lapack.complex_eigen_right(&m,d)?;let inv=inverse(lapack,&vectors,d)?;(values,vectors,inv,d)};
        drop(m);
        let one=|a:&[C]|(0..d).map(|c|a[c*d..(c+1)*d].iter().map(|z|magnitude(*z)).sum::<f64>()).fold(0.,f64::max);
        let condition=one(&vectors)*one(&inv);
        let gemm=lapack.complex_gemm();let rows=basis.population_rows();
        let mut readout=vec![ZERO;n*d];gemm.multiply(n,d,d,ONE,rows,n,&vectors,d,ZERO,&mut readout,n)?;
        let mut step_readout:Vec<Option<Vec<C>>>=(0..kernels.steps.len()).map(|_|None).collect();
        for &s in &steps{
            let mut rx=vec![ZERO;n*d];
            for &(i,j,x)in &kernels.steps[s].entries{for r in 0..n{mul_add(&mut rx[r+n*j],rows[r+n*i],x);}}
            let mut out=vec![ZERO;n*d];gemm.multiply(n,d,d,ONE,&rx,n,&vectors,d,ZERO,&mut out,n)?;
            step_readout[s]=Some(out);
        }
        Ok(Self{steps,values,vectors,inverse:inv,readout,step_readout,condition,largest_cluster:largest})
    }
}

/// Couplings below CLUSTER_ENTRY times their gap, |m_ij| < CLUSTER_ENTRY
/// |lambda_i - lambda_j|, leave the eigencoordinates of L0' in different
/// clusters; the clusters are diagonalized exactly and the couplings between
/// them at first order (vectors) and second order (values). Measured against
/// the whole eigensystem on three, four and five lasers (20 us, 10 ms): the
/// estimates within 0.35%, the largest cluster 156-362 of 1024 (1e-3:
/// 330-552), the class built in 0.2-0.33 s instead of 0.55-1.2 s.
const CLUSTER_ENTRY:f64=1e-2;
/// Two clusters merge when a first-order mixing between them exceeds this,
/// or when the third-order error of a value, over the time its mode lives
/// (the horizon, or 1 / |Re mu| when shorter), exceeds CLUSTER_PHASE.
const CLUSTER_MIXING:f64=0.05;
const CLUSTER_PHASE:f64=1e-2;
/// Merge passes before the whole matrix is diagonalized at once.
const CLUSTER_PASSES:usize=4;

/// The eigensystem of a flip-invariant class matrix m (column-major, d x d,
/// eigen-coordinates of L0') by clusters of strongly coupled coordinates
/// (closed under the pairing): each cluster exactly (`real_eigensystem`), the
/// couplings K' between clusters, in the clusters' eigencoordinates, by
/// first-order mixing T_ba = K'_ba / (mu_a - mu_b) and the second-order values
/// mu_a + sum_b K'_ab T_ba. V = V_c (1 + T), V^-1 = (1 - T) V_c^-1 (error
/// O(T^2)). A cluster pair whose mixing or third-order value error is above
/// its bound is merged and the whole is redone; one cluster is the exact
/// eigensystem. Returns the values, V, V^-1 and the largest cluster.
fn clustered_eigensystem(lapack:&Lapack,basis:&BlockEigenbasis,m:&[C],d:usize,horizon:f64)->Result<(Vec<C>,Vec<C>,Vec<C>,usize),String>{
    let first:Vec<bool>=(0..d).map(|j|basis.partner(j)>j).collect();
    // Atoms: a pair (j, j + 1) or a single coordinate.
    let mut atom=vec![0;d];let mut atoms=0;let mut j=0;
    while j<d{atom[j]=atoms;if first[j]{atom[j+1]=atoms;j+=2;}else{j+=1;}atoms+=1;}
    let mut parent:Vec<usize>=(0..atoms).collect();
    fn root(p:&mut [usize],mut i:usize)->usize{while p[i]!=i{p[i]=p[p[i]];i=p[i];}i}
    fn join(p:&mut [usize],a:usize,b:usize){let(x,y)=(root(p,a),root(p,b));if x!=y{p[x]=y;}}
    for j in 0..d{for i in 0..d{if atom[i]==atom[j]{continue;}let v=magnitude(m[i+d*j]);if v==0.{continue;}
        let gap=magnitude(C{re:basis.lambda[i].re-basis.lambda[j].re,im:basis.lambda[i].im-basis.lambda[j].im});
        if v>=CLUSTER_ENTRY*gap{join(&mut parent,atom[i],atom[j]);}}}
    let execution=Execution::new()?;
    for pass in 0..CLUSTER_PASSES{
        // Clusters (sorted coordinates) and their exact eigensystems.
        let mut members:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
        for j in 0..d{let r=root(&mut parent,atom[j]);members.entry(r).or_default().push(j);}
        let clusters:Vec<Vec<usize>>=members.into_values().collect();
        let largest=clusters.iter().map(|c|c.len()).max().unwrap_or(0);
        if clusters.len()==1{break;}
        let indices:Vec<usize>=(0..clusters.len()).collect();
        let systems=execution.map(clusters.clone(),|c|->Result<(Vec<C>,Vec<C>,Vec<C>),String>{
            let n=c.len();let local:Vec<C>=(0..n*n).map(|k|m[c[k%n]+d*c[k/n]]).collect();
            let pairs:Vec<bool>=c.iter().map(|&j|first[j]).collect();
            real_eigensystem(lapack,&pairs,&local,n)});
        let systems=systems.into_iter().collect::<Result<Vec<_>,_>>()?;
        // Eigen-index a of cluster k: offset[k] + l.
        let mut offset=Vec::with_capacity(clusters.len());let mut owner=vec![0;d];let mut total=0;
        for(k,c)in clusters.iter().enumerate(){offset.push(total);for a in total..total+c.len(){owner[a]=k;}total+=c.len();}
        let values:Vec<C>=systems.iter().flat_map(|s|s.0.iter().copied()).collect();
        // Y = m V_c (d x d, eigen-index columns), K' = V_c^-1 Y: one task per cluster.
        let slabs=execution.map(indices.clone(),|k|->Result<Vec<C>,String>{let(c,n)=(&clusters[k],clusters[k].len());
            let mut columns=vec![ZERO;d*n];for(l,&j)in c.iter().enumerate(){columns[l*d..(l+1)*d].copy_from_slice(&m[j*d..(j+1)*d]);}
            let mut out=vec![ZERO;d*n];lapack.complex_gemm().multiply(d,n,n,ONE,&columns,d,&systems[k].1,n,ZERO,&mut out,d)?;Ok(out)});
        let mut y=vec![ZERO;d*d];for(k,slab)in slabs.into_iter().enumerate(){y[offset[k]*d..(offset[k]+clusters[k].len())*d].copy_from_slice(&slab?);}
        let blocks=execution.map(indices.clone(),|k|->Result<Vec<C>,String>{let(c,n)=(&clusters[k],clusters[k].len());
            let mut rows=vec![ZERO;n*d];for a in 0..d{for(l,&j)in c.iter().enumerate(){rows[l+n*a]=y[j+d*a];}}
            let mut out=vec![ZERO;n*d];lapack.complex_gemm().multiply(n,d,n,ONE,&systems[k].2,n,&rows,n,ZERO,&mut out,n)?;Ok(out)});
        let mut coupling=vec![ZERO;d*d];
        for(k,block)in blocks.into_iter().enumerate(){let(block,n)=(block?,clusters[k].len());for a in 0..d{for l in 0..n{coupling[offset[k]+l+d*a]=block[l+n*a];}}}
        // First-order mixing between clusters, second-order values, and the checks.
        let mut mixing=vec![ZERO;d*d];let mut shifted=values.clone();let mut merge:Vec<(usize,usize)>=Vec::new();
        for a in 0..d{
            // The mixings into eigenvector a, its value's second-order shift, and the clusters that dominate them.
            let(mut most_mixed,mut largest_mix,mut most_shifting,mut largest_shift,mut shifts)=(usize::MAX,0_f64,usize::MAX,0_f64,0_f64);
            for b in 0..d{if owner[b]==owner[a]{continue;}
                let kba=coupling[b+d*a];if !nonzero(kba){continue;}
                let gap=C{re:values[a].re-values[b].re,im:values[a].im-values[b].im};let s=gap.re*gap.re+gap.im*gap.im;
                let t=if s>0.{mul(kba,C{re:gap.re/s,im:-gap.im/s})}else{C{re:f64::INFINITY,im:0.}};
                mixing[b+d*a]=t;let shift=mul(coupling[a+d*b],t);add(&mut shifted[a],shift);
                let(tm,sm)=(magnitude(t),magnitude(shift));shifts+=sm;
                if tm>largest_mix{largest_mix=tm;most_mixed=owner[b];}
                if sm>largest_shift{largest_shift=sm;most_shifting=owner[b];}}
            let lives=if values[a].re<0.{horizon.min(1./-values[a].re)}else{horizon};
            if largest_mix>CLUSTER_MIXING{merge.push((owner[a],most_mixed));}
            else if shifts*largest_mix*lives>CLUSTER_PHASE{merge.push((owner[a],most_shifting));}
        }
        if !merge.is_empty(){
            if pass+1==CLUSTER_PASSES{break;}
            for(x,y)in merge{let(a,b)=(clusters[x][0],clusters[y][0]);join(&mut parent,atom[a],atom[b]);}
            continue;
        }
        // V = V_c (1 + T): rows of cluster k; V^-1 = (1 - T) V_c^-1: columns of cluster k.
        let parts=execution.map(indices,|k|->Result<(Vec<C>,Vec<C>),String>{let n=clusters[k].len();let gemm=lapack.complex_gemm();
            let mut right=vec![ZERO;n*d];for a in 0..d{for l in 0..n{right[l+n*a]=mixing[offset[k]+l+d*a];}}
            for l in 0..n{right[l+n*(offset[k]+l)]=ONE;}
            let mut rows=vec![ZERO;n*d];gemm.multiply(n,d,n,ONE,&systems[k].1,n,&right,n,ZERO,&mut rows,n)?;
            let mut left=vec![ZERO;d*n];for l in 0..n{for a in 0..d{let t=mixing[a+d*(offset[k]+l)];left[a+d*l]=C{re:-t.re,im:-t.im};}left[offset[k]+l+d*l]=ONE;}
            let mut columns=vec![ZERO;d*n];gemm.multiply(d,n,n,ONE,&left,d,&systems[k].2,n,ZERO,&mut columns,d)?;
            Ok((rows,columns))});
        let mut vectors=vec![ZERO;d*d];let mut inverse=vec![ZERO;d*d];
        for(k,part)in parts.into_iter().enumerate(){let((rows,columns),c)=(part?,&clusters[k]);let n=c.len();
            for a in 0..d{for(l,&j)in c.iter().enumerate(){vectors[j+d*a]=rows[l+n*a];}}
            for(l,&j)in c.iter().enumerate(){inverse[j*d..(j+1)*d].copy_from_slice(&columns[l*d..(l+1)*d]);}}
        return Ok((shifted,vectors,inverse,largest));
    }
    // One cluster, or the couplings still too strong after CLUSTER_PASSES merges: the whole.
    let(values,vectors,inverse)=real_eigensystem(lapack,&first,m,d)?;
    Ok((values,vectors,inverse,d))
}

/// The eigensystem of a flip-invariant class matrix m (column-major, complex
/// eigen-coordinates z of L0'). With the real coordinates x of a conjugate
/// pair (j, j + 1), z = T x, T = 1/2 [[1, -i], [1, i]], the matrix T^-1 m T
/// is real; its real eigenvectors VR give the complex ones V_c = VR S,
/// S = [[1, 1], [i, -i]] per eigenvalue pair, so V = T V_c and
/// V^-1 = S^-1 VR^-1 T^-1.
fn real_eigensystem(lapack:&Lapack,pairs:&[bool],m:&[C],d:usize)->Result<(Vec<C>,Vec<C>,Vec<C>),String>{
    let first=|j:usize|pairs[j];
    let i_times=|z:C|C{re:-z.im,im:z.re};
    // Y = m T (columns), then x = T^-1 Y (rows): real up to roundoff.
    let mut y=m.to_vec();
    for j in(0..d).filter(|&j|first(j)){for i in 0..d{let(a,b)=(m[i+d*j],m[i+d*(j+1)]);
        y[i+d*j]=C{re:0.5*(a.re+b.re),im:0.5*(a.im+b.im)};y[i+d*(j+1)]=i_times(C{re:0.5*(b.re-a.re),im:0.5*(b.im-a.im)});}}
    let(mut real,mut imaginary,mut size)=(vec![0.;d*d],0_f64,0_f64);
    for c in 0..d{let mut j=0;while j<d{
        if first(j){let(a,b)=(y[j+d*c],y[j+1+d*c]);let(u,v)=(C{re:a.re+b.re,im:a.im+b.im},i_times(C{re:a.re-b.re,im:a.im-b.im}));
            real[j+d*c]=u.re;real[j+1+d*c]=v.re;imaginary=imaginary.max(u.im.abs()).max(v.im.abs());size=size.max(u.re.abs()).max(v.re.abs());j+=2;}
        else{real[j+d*c]=y[j+d*c].re;imaginary=imaginary.max(y[j+d*c].im.abs());size=size.max(y[j+d*c].re.abs());j+=1;}}}
    drop(y);
    if imaginary>1e-10*size{return Err(format!("Estimator class is not flip invariant ({:e})",imaginary/size));}
    let(wr,wi,vr)=lapack.eigen(&real,d)?;drop(real);
    if vr.iter().any(|v|!v.is_finite()){return Err("Nonfinite estimator class eigenvectors".into());}
    let(lu,pivots)=lapack.factor(&vr,d)?;let mut identity=vec![0.;d*d];for k in 0..d{identity[k+d*k]=1.;}
    let vri=lapack.solve_matrix_factored(&lu,&pivots,&identity,d)?;
    let mut pair=vec![false;d];let mut l=0;while l<d{if wi[l]!=0.&&l+1<d{pair[l]=true;l+=2;}else{l+=1;}}
    let values:Vec<C>=(0..d).map(|l|if pair[l]{C{re:wr[l],im:wi[l]}}else if l>0&&pair[l-1]{C{re:wr[l],im:-wi[l-1]}}else{C{re:wr[l],im:0.}}).collect();
    // V_c = VR S (columns), V = T V_c (rows).
    let mut vc:Vec<C>=vr.iter().map(|&v|C{re:v,im:0.}).collect();
    for l in(0..d).filter(|&l|pair[l]){for i in 0..d{let(a,b)=(vr[i+d*l],vr[i+d*(l+1)]);vc[i+d*l]=C{re:a,im:b};vc[i+d*(l+1)]=C{re:a,im:-b};}}
    let mut vectors=vc.clone();
    for c in 0..d{for j in(0..d).filter(|&j|first(j)){let(a,b)=(vc[j+d*c],vc[j+1+d*c]);let ib=i_times(b);
        vectors[j+d*c]=C{re:0.5*(a.re-ib.re),im:0.5*(a.im-ib.im)};vectors[j+1+d*c]=C{re:0.5*(a.re+ib.re),im:0.5*(a.im+ib.im)};}}
    // V^-1 = S^-1 VR^-1 (rows), then T^-1 (columns).
    let mut w:Vec<C>=vri.iter().map(|&v|C{re:v,im:0.}).collect();
    for l in(0..d).filter(|&l|pair[l]){for c in 0..d{let(a,b)=(vri[l+d*c],vri[l+1+d*c]);
        w[l+d*c]=C{re:0.5*a,im:-0.5*b};w[l+1+d*c]=C{re:0.5*a,im:0.5*b};}}
    let mut inv=w.clone();
    for j in(0..d).filter(|&j|first(j)){for r in 0..d{let(a,b)=(w[r+d*j],w[r+d*(j+1)]);let ib=i_times(b);
        inv[r+d*j]=C{re:a.re+ib.re,im:a.im+ib.im};inv[r+d*(j+1)]=C{re:a.re-ib.re,im:a.im-ib.im};}}
    Ok((values,vectors,inv))
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::kernels::KernelRule;
    use super::super::super::tests::{fixture,prepared};
    #[test]
    fn estimator_class_eigensystems_diagonalize_their_blocks(){
        // A flip-invariant class (both steps of the pair: the clustered real
        // path, to its error control over a short and a long horizon) and a
        // one-sided class (the complex path, exact): m V = V diag(mu), V^-1 V = I.
        let f=fixture();let p=prepared(&f);let basis=p.eigenbasis().unwrap();let d=16;
        let k=SwKernels::new(&p,basis,KernelRule{entry:0.,horizon_rate:0.,horizon:1.,cap:f64::INFINITY}).unwrap();
        for(steps,horizon,tolerance)in[((0..k.steps.len()).collect::<Vec<_>>(),1.,1e-4),((0..k.steps.len()).collect(),1e6,1e-4),(vec![0],1.,1e-10)]{
            let mut m=vec![ZERO;d*d];for(j,&l)in basis.lambda.iter().enumerate(){m[j+d*j]=l;}for &s in &steps{k.add_second_order(s,&mut m);}
            let class=BlockSystem::new(&p.lapack,basis,&k,steps.clone(),4,horizon).unwrap();
            let scale=m.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for c in 0..d{for r in 0..d{
                let mut mv=ZERO;for l in 0..d{mul_add(&mut mv,m[r+d*l],class.vectors[l+d*c]);}
                let vm=mul(class.vectors[r+d*c],class.values[c]);
                assert!(magnitude(C{re:mv.re-vm.re,im:mv.im-vm.im})<tolerance*scale,"steps {steps:?}, horizon {horizon}: eigenpair {c}");
                let mut id=ZERO;for l in 0..d{mul_add(&mut id,class.inverse[r+d*l],class.vectors[l+d*c]);}
                assert!(magnitude(C{re:id.re-f64::from(u8::from(r==c)),im:id.im})<tolerance,"steps {steps:?}, horizon {horizon}: inverse ({r}, {c})");
            }}
        }
    }

}
