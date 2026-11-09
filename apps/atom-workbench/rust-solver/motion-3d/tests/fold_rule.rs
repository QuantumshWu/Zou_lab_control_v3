//! The fold rule over the couplings between ground states (ground.rs `classify`): at the ladder's first rung every pair whose fold
//! exists folds (Stewart: its mixing m < ½), its error charged, whatever the frequencies of the others -- a van Vleck fold needs no gap
//! between what it keeps and what it folds --; a rung's allowance keeps the pairs whose m² passes it.
use atom_motion_3d::ground::{classify,exists,pair_folds,Ground,Kind,EXISTS};
use atom_motion_3d::model::Model;
use atom_motion_3d::Request;
use serde_json::Value;
#[path="../examples/scene/mod.rs"] mod scene;

/// A payload of tests/fixtures as the app sends it, on its own times.
fn request(fixture:&str)->Request{
    let f:Value=serde_json::from_str(fixture).unwrap();let p=&f["payload"];let mut r=scene::request(p);
    let(duration,samples)=(p["options"]["durationUs"].as_f64().unwrap(),p["options"]["samples"].as_u64().unwrap() as usize);
    r.times_us=(0..samples).map(|i|duration*i as f64/(samples-1) as f64).collect();r
}

/// The validation's F2, both ways the population core's gap rule failed a van Vleck fold (l0 = 1):
/// - near light on two lines and a far beam with no gap of 1/√budget above them (the case b1/115: the D1 and D2
///   light 7.1 apart, the tweezer at 95, its counter-rotating partners beyond): the rule folded nothing, and
///   the scene was refused for a counter-rotating coupling of mixing 1e-17;
/// - a foldable coupling below the largest gap (the hyperfine splitting under the D1–D2 spread): the rule kept
///   it, and the scene was refused for a static F = 1 – F = 2 coupling of mixing 1e-11.
/// At the first rung every pair that exists folds, m = g/(|ν| − 1) < ½ -- 7.1 apart with g = 2.9 (m 0.48) folds, with 3.2 (m 0.52) it stays --;
/// the classes' cut lies in the largest gap above every kept frequency whose ratio passes Stewart's bound, m = ν_lo/ν_hi < ½ (95 over 7.1).
/// A rung's allowance keeps every pair whose m² passes it: 95 apart with g = 1e-3 (m² 1.1e-10) at 1e-11, not at 1e-9.
#[test]
fn every_pair_that_exists_folds_at_the_first_rung(){
    let lines=[(0.,1.),(7.1,3.2),(95.,1e-3),(659.,1e-4),(754.,1e-6)];
    let(verdicts,cut)=classify(&lines,1.,EXISTS);
    assert_eq!(verdicts,vec![false,false,true,true,true],"{cut}");
    assert!((cut-(7.1_f64*95.).sqrt()).abs()<1e-12,"{cut}");
    let(verdicts,_)=classify(&[(0.,1.),(7.1,2.9),(95.,1e-3)],1.,f64::INFINITY);assert_eq!(verdicts,vec![false,true,true]);
    let below=[(0.5,1.),(3.,1e-6),(100.,1e-6),(120.,1e-3)];
    let(verdicts,cut)=classify(&below,1.,EXISTS);
    assert_eq!(verdicts,vec![false,true,true,true],"the item below the gap (cut {cut}) folds");
    assert_eq!(classify(&lines,1.,1e-9).0,vec![false,false,true,true,true]);assert_eq!(classify(&lines,1.,1e-11).0,vec![false,false,false,true,true]);
    assert!(exists(0.499)&&!exists(0.5)&&pair_folds(0.01,1e-4)&&!pair_folds(0.01,0.99e-4)&&!pair_folds(0.6,f64::INFINITY));
}

/// The default scene with a weak D1 beam 2 GHz off (the validation's F2 repro): near light on both D lines put
/// the ground couplings' largest gap between the hyperfine splitting and the D1–D2 spread, so the gap rule kept
/// every F = 1 – F = 2 static coupling (mixing 2.5e-11) and the scene was refused. They fold, and it builds.
#[test]
fn a_weak_beam_on_the_other_line_does_not_refuse_the_scene(){
    let r=request(include_str!("../../tests/fixtures/weak_d1_far_beam_motion.json"));
    let g=Ground::new(&r).unwrap();
    assert!(g.items.iter().filter(|i|matches!(i.kind,Kind::Static{a,b}if g.levels[a]!=g.levels[b])).all(|i|i.folded),"a static F = 1 - F = 2 coupling kept");
    Model::new(&r).unwrap();
}
