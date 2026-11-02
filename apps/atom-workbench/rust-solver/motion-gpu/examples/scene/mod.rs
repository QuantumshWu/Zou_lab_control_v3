//! Scenes for the validation programs and tests: a native payload (scripts/native-protocol.mjs
//! `prepareNativeSolve`) as the motion core's request, and its frame turned to put the plane of 2D
//! motion on the first two axes.
#![allow(dead_code)]
use atom_motion_core::{Beam,Complex,Matrix,Request,STATES};
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
        counter_coupling:matrix(&b["counterCoupling"]),outside_shift:matrix(&b["outsideShift"])}}).collect();
    let e:Vec<Matrix>=motion["emission"].as_array().unwrap().iter().map(matrix).collect();
    let(re,im)=(floats(&m["initialReal"]),floats(&m["initialImag"]));
    Request{states:m["states"].as_array().unwrap().iter().map(|s|s.as_str().unwrap().into()).collect(),hamiltonian:matrix(&motion["hamiltonian"]),
        frame_frequencies_mhz:floats(&motion["frameFrequenciesMHz"]),frequencies_mhz:floats(&m["frequencyCoordinates"]["frequenciesMHz"]),beams,
        emission:[e[0].clone(),e[1].clone(),e[2].clone()],field_vector_g:{let f=floats(&motion["fieldVectorG"]);[f[0],f[1],f[2]]},mass_kg:motion["massKg"].as_f64().unwrap(),
        initial_density:(0..STATES*STATES).map(|k|Complex::new(re[k],im[k])).collect(),temperature_uk:motion["temperatureUK"].as_f64().unwrap(),
        times_us:vec![],tolerance:0.01}
}
/// The frame turned cyclically, (a, b, c) → (c, a, b): a 1D request's motion axis (its third) becomes
/// the plane's first axis, its first perpendicular axis the plane's second.
pub fn turned(mut r:Request)->Request{
    for b in &mut r.beams{let d=b.direction;b.direction=[d[2],d[0],d[1]];}
    let[a,b,c]=r.emission.clone();r.emission=[c,a,b];r
}
pub fn load(path:&str)->Request{request(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap())}
/// The app's tweezer PGC example (TW2's in-trap parameters) as the 1D payload along x (scratch
/// export of scripts/native-protocol.mjs).
pub const PGC:&str=r"C:\Users\eadri\AppData\Local\Temp\claude\C--Users-eadri-Dropbox-WorkCode-Github-Zou-lab-control-v3\8308d66c-699b-4483-ab9b-7305d93a9f7e\scratchpad\gpu2d\scenes\pgc_x.json";
