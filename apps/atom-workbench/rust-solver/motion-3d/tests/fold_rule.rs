//! The fold rule (ground.rs `classify`): every coupling whose mixing fits the budget folds, its error booked,
//! whatever the frequencies of the others -- a van Vleck fold needs no gap between what it keeps and what it folds.
use atom_motion_3d::ground::{classify,foldable,Ground};
use atom_motion_3d::ledger::BUDGET;
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
/// Every foldable item folds; the unfoldable stay, and the classes' cut stays the gap rule's.
#[test]
fn every_coupling_that_fits_the_budget_folds(){
    let lines=[(0.,1.),(7.1,1.),(95.,1e-3),(659.,1e-4),(754.,1e-6)];
    let(verdicts,cut)=classify(&lines,1.);
    assert_eq!(verdicts,vec![false,false,true,true,true],"no gap of 1/√budget above the near light: {cut}");
    assert!(cut==f64::INFINITY,"{cut}");
    let below=[(0.5,1.),(3.,1e-6),(100.,1e-6),(120.,1e-3)];
    let(verdicts,cut)=classify(&below,1.);
    assert_eq!(verdicts,vec![false,true,true,true],"the item below the gap (cut {cut}) folds");
    assert!(foldable(3.,1e-6,1.)&&!foldable(1.9,1e-6,1.)&&!foldable(3.,2.*(3.-1.)*BUDGET.sqrt(),1.));
}

/// The default scene with a weak D1 beam 2 GHz off (the validation's F2 repro): near light on both D lines put
/// the ground couplings' largest gap between the hyperfine splitting and the D1–D2 spread, so the gap rule kept
/// every F = 1 – F = 2 static coupling (mixing 2.5e-11) and the scene was refused. They fold, and it builds.
#[test]
fn a_weak_beam_on_the_other_line_does_not_refuse_the_scene(){
    let r=request(include_str!("../../tests/fixtures/weak_d1_far_beam_motion.json"));
    let g=Ground::new(&r).unwrap();
    assert!(g.items.iter().all(|i|i.folded||!foldable(i.frequency,i.size,g.generator_scale)),"a foldable coupling kept");
    Model::new(&r).unwrap();
}
