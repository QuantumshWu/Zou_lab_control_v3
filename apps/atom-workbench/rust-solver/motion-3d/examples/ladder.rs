//! The rule's candidates of a scene (ground.rs `Ground::allowances`): per allowance the clusters, the
//! explicit excited states and the fold's charge, and the time the model takes.
//! `cargo run --profile iterate -p atom-motion-3d --example ladder -- <scene>...`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::ground::Ground;
use atom_motion_3d::model::Model;
use std::time::Instant;

fn main(){
    for name in std::env::args().skip(1){
        let request=scene::scene(&name);
        let clock=Instant::now();let allowances=Ground::allowances(&request).unwrap();
        println!("{name}: {} candidates ({:.2} s)",allowances.len(),clock.elapsed().as_secs_f64());
        for a in allowances{
            let clock=Instant::now();let model=Model::with(&request,a).unwrap();let g=&model.ground;
            let sizes:Vec<String>=g.clusters.iter().map(|c|if c.excited.is_empty(){c.dimension().to_string()}else{format!("{}({})",c.dimension(),c.excited.len())}).collect();
            println!("  allowance {a:.3e}: charge {:.3e}, clusters {}, model {:.2} s",g.charge,sizes.join("+"),clock.elapsed().as_secs_f64());
        }
    }
}
