//! The fourth-order model of a scene (model.rs `Model::with_order`): its build time, table nodes and
//! interpolation error, and the fourth order's carriers per cluster.
//! `cargo run --profile iterate -p atom-motion-3d --example fourth_build -- <scene>...`
#[path="scene/mod.rs"] mod scene;
use atom_motion_3d::model::Model;
use std::time::Instant;

fn main(){
    for name in std::env::args().skip(1){
        let request=scene::scene(&name);
        for fourth in [false,true]{
            let clock=Instant::now();
            match Model::with_order(&request,f64::INFINITY,fourth){
                Ok(m)=>println!("{name} fourth {fourth}: {:.2} s, table error {:.1e}, quartic {:.1e}, clusters {:?}",clock.elapsed().as_secs_f64(),m.table_error,m.quartic,
                    (0..m.clusters()).map(|k|m.dimension(k)).collect::<Vec<_>>()),
                Err(e)=>println!("{name} fourth {fourth}: {:.2} s, error {e}",clock.elapsed().as_secs_f64()),
            }
        }
    }
}
