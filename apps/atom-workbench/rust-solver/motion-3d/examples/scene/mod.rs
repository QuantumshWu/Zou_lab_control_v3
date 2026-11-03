//! Payloads for the validation programs: a native solve payload (scripts/native-protocol.mjs
//! `prepareNativeSolve`) as the motion core's request.
#![allow(dead_code)]
use atom_motion_3d::{Beam,Complex,Matrix,Request,STATES};
use serde_json::Value;

fn matrix(v:&Value)->Matrix{
    let f=|k:&str|v[k].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect::<Vec<f64>>();
    Matrix{real:f("real"),imag:f("imag")}
}
/// The payload's model with its motion block, as the motion core's request.
pub fn request(payload:&Value)->Request{
    let m=&payload["model"];let motion=&m["motion"];
    let floats=|v:&Value|v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect::<Vec<f64>>();
    let beams=motion["beams"].as_array().unwrap().iter().map(|b|{let d=floats(&b["direction"]);Beam{id:b["id"].as_str().unwrap().into(),name:b["name"].as_str().unwrap().into(),
        direction:[d[0],d[1],d[2]],wave_number_per_um:b["waveNumberPerUm"].as_f64().unwrap(),waist_um:b["waistUm"].as_f64(),intensity_mw_cm2:b["intensityMWcm2"].as_f64().unwrap(),
        frequency_index:b["frequencyIndex"].as_array().unwrap().iter().map(|n|n.as_i64().unwrap() as i32).collect(),coupling:matrix(&b["coupling"]),
        counter_coupling:matrix(&b["counterCoupling"]),outside_shift:matrix(&b["outsideShift"])}}).collect::<Vec<Beam>>();
    // The trap beam the motion block names; a payload exported before it named one: the first focused beam.
    let trap_beam=match motion["trapBeamId"].as_str(){Some(id)=>beams.iter().position(|b|b.id==id).expect("trapBeamId names a beam"),
        None=>beams.iter().position(|b|b.waist_um.is_some()).unwrap_or(0)};
    let e:Vec<Matrix>=motion["emission"].as_array().unwrap().iter().map(matrix).collect();
    let(re,im)=(floats(&m["initialReal"]),floats(&m["initialImag"]));
    Request{states:m["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&motion["hamiltonian"]),
        frame_frequencies_mhz:floats(&motion["frameFrequenciesMHz"]),frequencies_mhz:floats(&m["frequencyCoordinates"]["frequenciesMHz"]),beams,
        emission:[e[0].clone(),e[1].clone(),e[2].clone()],field_vector_g:{let f=floats(&motion["fieldVectorG"]);[f[0],f[1],f[2]]},mass_kg:motion["massKg"].as_f64().unwrap(),
        initial_density:(0..STATES*STATES).map(|k|Complex::new(re[k],im[k])).collect(),temperature_uk:motion["temperatureUK"].as_f64().unwrap(),
        trap_beam,times_us:vec![]}
}
/// A scene exported as a native solve payload: `<ATOM_MOTION_SCENES>/<name>.json`.
pub fn scene(name:&str)->Request{
    let path=std::path::Path::new(&folder("ATOM_MOTION_SCENES")).join(format!("{name}.json"));
    request(&serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e|panic!("{}: {e}",path.display()))).unwrap())
}
/// A scene's 1D quantum reference (the GMW engine's long run, `LONGREF_<name>.json`: its `rows`
/// over time and its `distributions`) from `<ATOM_MOTION_REFERENCES>`.
pub fn reference(name:&str)->Value{
    let path=std::path::Path::new(&folder("ATOM_MOTION_REFERENCES")).join(format!("LONGREF_{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e|panic!("{}: {e}",path.display()))).unwrap()
}
fn folder(variable:&str)->String{std::env::var(variable).unwrap_or_else(|_|panic!("set {variable} to the validation folder"))}
