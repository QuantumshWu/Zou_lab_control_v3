//! The rule's partition of a scene (ground.rs): its clusters -- ground states and explicit excited
//! states --, the excited couplings kept, promoted by the closure and folded with the largest
//! estimate, the model's tables and its ledger.
//! `cargo run --profile iterate -p atom-motion-3d --example partition -- <scene>... [--duration us]`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::ground::ExcitedKind;
use atom_motion_3d::model::Model;
use std::time::Instant;

fn main(){
    let args:Vec<String>=std::env::args().skip(1).collect();
    let duration=args.iter().position(|a|a=="--duration").map(|i|args[i+1].parse::<f64>().unwrap());
    for name in args.iter().filter(|a|!a.starts_with("--")&&a.parse::<f64>().is_err()){
        let mut request=scene::scene(name);if let Some(d)=duration{request.times_us=vec![0.,d];}
        let clock=Instant::now();let model=Model::new(&request).unwrap();let g=&model.ground;
        let label=|i:usize|request.states[i].clone();
        println!("{name}: model {:.2} s, duration {} us, focused {:?}, table nodes error {:.1e}",clock.elapsed().as_secs_f64(),g.duration,model.focused,model.table_error);
        for(k,c)in g.clusters.iter().enumerate(){
            println!("  cluster {k} (n {}): ground {:?} explicit {:?}; drives {}, crosses/decays, classes {}",c.dimension(),
                c.states.iter().map(|&b|label(g.ground_states[b])).collect::<Vec<_>>(),c.excited.iter().map(|&e|label(g.excited_states[e])).collect::<Vec<_>>(),c.drives.len(),c.classes.len());
        }
        let kept:Vec<_>=g.excited_items.iter().filter(|x|!x.folded).collect();let promoted=kept.iter().filter(|x|x.promoted).count();
        let worst=g.excited_items.iter().filter(|x|x.folded).max_by(|a,b|a.estimate.total_cmp(&b.estimate));
        let describe=|k:&ExcitedKind|match *k{ExcitedKind::Drive(d)=>format!("{} <- {} (carrier {})",label(g.excited_states[d.excited]),label(g.ground_states[d.ground]),d.carrier),
            ExcitedKind::Static{a,b}=>format!("{} ~ {}",label(g.excited_states[a]),label(g.excited_states[b]))};
        println!("  excited couplings: {} kept ({} promoted), {} folded; largest folded {}",kept.len(),promoted,g.excited_items.len()-kept.len(),
            worst.map_or("none".into(),|x|format!("{:.2e} {}",x.estimate,describe(&x.kind))));
        let mut top:Vec<_>=kept.clone();top.sort_by(|a,b|b.estimate.total_cmp(&a.estimate));
        for x in top.iter().take(6){println!("    kept {:.3e} (size {:.2}, distance {:.1}) {}",x.estimate,x.size,x.distance,describe(&x.kind));}
        // Folded drives between two states of one cluster: their residual in the frames.
        for x in g.excited_items.iter().filter(|x|x.folded){if let ExcitedKind::Drive(d)=x.kind{
            let k=g.cluster[d.ground];let c=&g.clusters[k];
            if let Some(p)=c.excited.iter().position(|&e|e==d.excited){let(x0,xe)=(c.position(d.ground).unwrap(),p);
                println!("    folded inside cluster {k}: {} est {:.2e} size {:.3} residual {:.3e}",describe(&x.kind),x.estimate,x.size,c.excited_frames[xe]-c.frames[x0]-g.carriers[d.carrier].detuning);}
        }}
        // The drives' saturation at the focus: 2|Σ_l V_l[e,b]|²/|E_e − E_b − δ_c − iΓ/2|² (every beam's
        // amplitude 1 there, the energies the focus's), against the rule's (every beam at its peak in phase,
        // the smallest distance over the trap).
        {let e=g.eliminate(&g.focus_intensities()).unwrap();let mut worst=(0.,0.,String::new());
        for x in &g.excited_items{if let ExcitedKind::Drive(d)=x.kind{
            let v=g.carriers[d.carrier].beams.iter().fold(atom_motion_3d::Complex::ZERO,|s,&l|s+g.couplings[l][(d.excited,d.ground)]);
            let de=atom_motion_3d::Complex::new(e.excited_energies[d.excited]-e.energies[d.ground]-g.carriers[d.carrier].detuning,-0.5*g.decay[(d.excited,d.excited)].re);
            let sat=2.*v.norm_sqr()/de.norm_sqr();let rule=2.*(x.size/x.distance).powi(2);
            if sat>worst.0{worst=(sat,rule,describe(&x.kind));}
        }}
        println!("  saturation at the focus: largest {:.3e} ({}; the rule's worst-case for it {:.3e})",worst.0,worst.2,worst.1);
        // Each ground state's total saturation S_b = 2 Σ_e |V[e,b]|²/|ΔE|² over its drives (at the focus; and the
        // rule's worst case: every beam at its peak in phase, the smallest distance over the trap), its folded part.
        let ng=g.ground_states.len();let(mut focus,mut rule,mut folded)=(vec![0.;ng],vec![0.;ng],vec![0.;ng]);
        for x in &g.excited_items{if let ExcitedKind::Drive(d)=x.kind{
            let v=g.carriers[d.carrier].beams.iter().fold(atom_motion_3d::Complex::ZERO,|s,&l|s+g.couplings[l][(d.excited,d.ground)]);
            let de=atom_motion_3d::Complex::new(e.excited_energies[d.excited]-e.energies[d.ground]-g.carriers[d.carrier].detuning,-0.5*g.decay[(d.excited,d.excited)].re);
            focus[d.ground]+=2.*v.norm_sqr()/de.norm_sqr();rule[d.ground]+=2.*(x.size/x.distance).powi(2);if x.folded{folded[d.ground]+=2.*v.norm_sqr()/de.norm_sqr();}
        }}
        let top=|v:&[f64]|v.iter().copied().enumerate().max_by(|a,b|a.1.total_cmp(&b.1)).unwrap();
        let(b,sf)=top(&focus);println!("  total saturation S_b: at the focus largest {:.3e} ({}), its folded part {:.3e}; the rule's worst case {:.3e}; largest folded at the focus {:.3e}",
            sf,label(g.ground_states[b]),folded[b],rule[b],top(&folded).1);}
        // The ground items' frequencies around the cut: the kept ones with the largest frequency.
        let mut kept_items:Vec<_>=g.items.iter().filter(|i|!i.folded).collect();kept_items.sort_by(|a,b|b.frequency.total_cmp(&a.frequency));
        println!("  ground items: cut {:.3e}; kept {} of {}; highest kept: {}",g.cut,kept_items.len(),g.items.len(),
            kept_items.iter().take(4).map(|i|format!("{:?} f {:.3e} size {:.3e}",i.kind,i.frequency,i.size)).collect::<Vec<_>>().join(" | "));
        // The items whose fold does not exist (their mixing m ≥ ½, ground.rs `exists`), the highest frequencies first.
        let mut stiff:Vec<_>=g.items.iter().filter(|i|i.size>0.&&!atom_motion_3d::ground::exists(atom_motion_3d::ground::mixing(i.size,i.frequency.abs(),g.generator_scale,f64::MIN_POSITIVE))).collect();
        stiff.sort_by(|a,b|b.frequency.total_cmp(&a.frequency));
        println!("  stiff items: {}",stiff.iter().take(4).map(|i|format!("{:?} f {:.3e} size {:.3e}",i.kind,i.frequency,i.size)).collect::<Vec<_>>().join(" | "));
        // Each ground state's total saturation with every drive folded (the rule's estimates over the reach).
        {let mut t=vec![0.;g.ground_states.len()];for x in &g.excited_items{if let ExcitedKind::Drive(d)=x.kind{t[d.ground]+=2.*(x.size/x.distance.max(1./g.duration)).powi(2);}}
        let(b,v)=t.iter().copied().enumerate().max_by(|a,b|a.1.total_cmp(&b.1)).unwrap();println!("  every drive folded: largest total saturation {v:.3e} ({}); reach intensities {:?}",label(g.ground_states[b]),g.reach.iter().map(|x|format!("{x:.3}")).collect::<Vec<_>>());}
        let(far,near,drives)=g.fold_estimates();println!("  folds: far {far:.2e}, ground couplings {near:.2e}, drives {drives:.2e}; dressing {:.2e}",model.dressing);
    }
}
