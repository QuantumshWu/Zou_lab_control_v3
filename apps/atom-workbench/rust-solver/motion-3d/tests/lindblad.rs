//! The eliminated generator against its jumps: a Lindblad generator's no-jump decay is its jumps' rate in every
//! state, i(H − H†) = Σ L†L, which keeps the trajectories' norm the survival probability. A model whose decay
//! falls below its jumps' rate beyond the budget is refused (model.rs `Model::lindblad`); where every drive
//! folds, the solve stops with why.
use atom_motion_3d::ground::Ground;
use atom_motion_3d::linear::Dense;
use atom_motion_3d::model::Model;
use atom_motion_3d::{Complex,Request};
use serde_json::Value;
#[path="../examples/scene/mod.rs"] mod scene;

/// A payload of tests/fixtures as the app sends it, on its own times.
fn request(fixture:&str)->Request{
    let f:Value=serde_json::from_str(fixture).unwrap();let p=&f["payload"];let mut r=scene::request(p);
    let(duration,samples)=(p["options"]["durationUs"].as_f64().unwrap(),p["options"]["samples"].as_u64().unwrap() as usize);
    r.times_us=(0..samples).map(|i|duration*i as f64/(samples-1) as f64).collect();r
}

/// The default scene's explicit clusters (its fold's second rung: the probe's drives into F′ = 2 kept, the
/// F = 2 sublevels two apart joined through them): the folded photons of two sublevels landing in one state
/// share a class through the explicit states' photons of their frequency, without a kept coupling between their
/// channels, and the jumps carry their cross term -- so does the decay (ground.rs `FoldedCross`). The dense
/// generator, its responses dressed by the near light (ground.rs, # The near light's dressing), is a Lindblad
/// generator to round-off at the focus and at a generic instant; it lacked 2.6e-4 rad/us of the jumps' cross
/// terms (of 38).
#[test]
fn the_explicit_clusters_decay_is_their_photons(){
    let r=request(include_str!("../../tests/fixtures/default_scene_motion.json"));
    let allowance=Ground::allowances(&r).unwrap()[1];let model=Model::with(&r,allowance).unwrap();let g=&model.ground;
    let point=[0.;3];let e=g.eliminate(&model.intensities_at_point(point)).unwrap();
    let mut checked=0;
    for k in (0..g.clusters.len()).filter(|&k|!g.clusters[k].excited.is_empty()){
        let n=g.clusters[k].dimension();
        for t in [0.,0.618]{
            let l=g.local_at(&e,k,point,t).unwrap();let h=&l.effective;
            let mut d=Dense::from_fn(n,n,|i,j|(h[(i,j)]-h[(j,i)].conj())*Complex::new(0.,1.));
            for j in &l.jumps{for m in j{d.add_scaled(-Complex::ONE,&m.adjoint_product(m));}}
            let round=64.*f64::EPSILON*n as f64*h.largest();
            assert!(d.largest()<=round,"cluster {k} at t = {t}: i(H − H†) − Σ L†L reaches {:.3e} rad/us (round-off {round:.1e})",d.largest());
            checked+=1;
        }
    }
    assert!(checked>0,"the second rung keeps explicit states");
}

/// The validation's F10 at +28 MHz, every drive folded: the folded probe–strong-beam Raman pairs' van Vleck paths
/// of the release added a decay no jump carries, below zero at the focus (−3.2 rad/us) -- the trajectories' norm
/// grew and the solve ran away (killed after 7 CPU-minutes), and the guard refused the model. Their paths resummed
/// into the responses' dressing (ground.rs, # The near light's dressing), the jumps carry the whole decay: the
/// guard passes the model, and in every cluster, at t = 0 and at a generic instant, its generator is a Lindblad
/// one to round-off at the focus, a node of the tables, and between nodes to their interpolation's error.
#[test]
fn a_strong_beam_s_folded_pairs_make_a_lindblad_generator_where_every_drive_folds(){
    let r=request(include_str!("../../tests/fixtures/far_d2_beam_probe_motion.json"));
    let model=Model::new(&r).unwrap_or_else(|e|panic!("the dressed model is refused: {e}"));
    for(x,between)in [([0.;3],false),([0.05,0.1,0.],true),([0.,0.,0.3],true)]{let p=model.point(x,[0.;3]);
        for k in 0..model.clusters(){for t in [0.,0.618]{
            let(low,_,scale)=model.decay_balance(k,&p,t);
            let tolerance=64.*f64::EPSILON*model.dimension(k) as f64*scale+if between{model.table_error*scale}else{0.};
            assert!(low>=-tolerance,"cluster {k} at {x:?}, t = {t}: i(H − H†) − Σ L†L reaches {low:.3e} rad/us (tolerance {tolerance:.1e})");
        }}
    }
}

/// The default scene's first rungs, either order, are Lindblad generators: none is refused.
#[test]
fn the_default_scene_s_rungs_are_not_refused(){
    let r=request(include_str!("../../tests/fixtures/default_scene_motion.json"));
    for allowance in Ground::allowances(&r).unwrap().into_iter().take(3){for fourth in [false,true]{
        if let Err(e)=Model::with_order(&r,allowance,fourth){panic!("the rung at {allowance:.3e} (fourth {fourth}) is refused: {e}");}
    }}
}
