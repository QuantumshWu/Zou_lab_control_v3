//! The explicit states' dressing (ground.rs `Ground::dressing`) against the excited block's exact eigenstates at the
//! focus: for each explicit state e of every cluster and each ground state b of it, the drive into the dressed state,
//! V_eb + ½(H_eF A[F,b] + x†V^f[F,b]) (the model's), against ⟨ẽ|V|b⟩ with ẽ the excited block's eigenvector nearest to e
//! (its overlap with e largest), per carrier; and the dressed photon C(e + x) against C ẽ.
//! `cargo run --profile iterate -p atom-motion-3d --example dressing_check -- <scene>`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::model::Model;
use atom_motion_3d::linear::Dense;
use atom_motion_3d::Complex;

/// Eigenvectors of a Hermitian matrix by cyclic Jacobi rotations (columns), with the eigenvalues.
fn eigen(m:&Dense)->(Vec<f64>,Dense){
    let n=m.rows;let mut a=m.clone();let mut v=Dense::from_fn(n,n,|i,j|if i==j{Complex::ONE}else{Complex::ZERO});
    for _ in 0..100{
        let off:f64=(0..n).map(|i|(0..n).filter(|&j|j!=i).map(|j|a[(i,j)].norm_sqr()).sum::<f64>()).sum();if off<1e-28{break;}
        for p in 0..n{for q in p+1..n{
            let modulus=a[(p,q)].abs();if modulus<1e-300{continue;}
            let phase=Complex::new(a[(p,q)].re/modulus,a[(p,q)].im/modulus);
            for k in 0..n{a[(k,q)]=a[(k,q)]*phase.conj();a[(q,k)]=a[(q,k)]*phase;v[(k,q)]=v[(k,q)]*phase.conj();}
            let tau=(a[(q,q)].re-a[(p,p)].re)/(2.*modulus);let t=tau.signum()/(tau.abs()+(1.+tau*tau).sqrt());let c=1./(1.+t*t).sqrt();let s=t*c;
            for k in 0..n{let(x,y)=(a[(k,p)],a[(k,q)]);a[(k,p)]=x*c-y*s;a[(k,q)]=x*s+y*c;}
            for k in 0..n{let(x,y)=(a[(p,k)],a[(q,k)]);a[(p,k)]=x*c-y*s;a[(q,k)]=x*s+y*c;}
            for k in 0..n{let(x,y)=(v[(k,p)],v[(k,q)]);v[(k,p)]=x*c-y*s;v[(k,q)]=x*s+y*c;}
        }}
    }
    ((0..n).map(|i|a[(i,i)].re).collect(),v)
}

fn main(){
    let name=std::env::args().nth(1).unwrap();let request=scene::scene(&name);let model=Model::new(&request).unwrap();let g=&model.ground;
    let e=g.eliminate(&g.hamiltonian_with(&vec![1.;g.beams.len()])).unwrap();let ne=g.excited_states.len();
    let(_,vectors)=eigen(&e.excited);
    for(k,cl)in g.clusters.iter().enumerate(){
        let flags:Vec<bool>=(0..ne).map(|f|cl.excited.contains(&f)).collect();
        for &s in &cl.excited{
            let x=g.dressing(&e,s,&flags).unwrap();
            // The eigenvector nearest to s, its phase making its s component real positive.
            let j=(0..ne).max_by(|&a,&b|vectors[(s,a)].abs().total_cmp(&vectors[(s,b)].abs())).unwrap();
            let ph=vectors[(s,j)].conj()*(1./vectors[(s,j)].abs());let tilde:Vec<Complex>=(0..ne).map(|i|vectors[(i,j)]*ph).collect();
            let deviation=(0..ne).filter(|&f|f!=s&&!flags[f]).map(|f|(tilde[f]-x[f]).abs()).fold(0.,f64::max);
            let largest=(0..ne).filter(|&f|f!=s&&!flags[f]).map(|f|tilde[f].abs()).fold(0.,f64::max);
            let explicit_part=(0..ne).filter(|&f|f!=s&&flags[f]).map(|f|tilde[f].abs()).fold(0.,f64::max);
            println!("cluster {k} {}: dressing max |x| {:.4}, exact admixture max {:.4} (on explicit states {:.4}), largest deviation {:.2e}",request.states[g.excited_states[s]],
                x.iter().map(|z|z.abs()).fold(0.,f64::max),largest,explicit_part,deviation);
            for(c,carrier)in g.carriers.iter().enumerate(){for &b in &cl.states{
                let v=|row:usize|carrier.beams.iter().fold(Complex::ZERO,|acc,&l|acc+g.couplings[l][(row,b)]);
                let folded=|row:usize|carrier.beams.iter().fold(Complex::ZERO,|acc,&l|acc+g.folded[l][(row,b)]);
                let kept=v(s)-folded(s);if kept==Complex::ZERO{continue;}
                // A[F,b] = −R: the decoupled response of b's column.
                let a:Vec<Complex>=(0..ne).map(|f|-carrier.beams.iter().fold(Complex::ZERO,|acc,&l|acc+e.responses[l][(f,b)])).collect();
                let effective=(0..ne).filter(|&f|!flags[f]).fold(Complex::ZERO,|acc,f|acc+e.excited[(s,f)]*a[f]+x[f].conj()*folded(f))*0.5;
                let model_drive=kept+effective;let exact=(0..ne).fold(Complex::ZERO,|acc,i|acc+tilde[i].conj()*v(i));
                println!("   carrier {c} from {}: kept {:.4}, effective {:.4}, model {:.4}, exact <e~|V|b> {:.4}; relative {:+.2e}",request.states[g.ground_states[b]],
                    kept.abs(),effective.abs(),model_drive.abs(),exact.abs(),(model_drive.abs()-exact.abs())/exact.abs());
            }}
        }
    }
}
