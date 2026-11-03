//! The 2D solve end to end on the app's tweezer PGC example in the radial plane:
//! `cargo run --profile iterate --example solve_pgc -- [t_end_us] [spacing_us] [temperature_uK] [payload.json] [runs]` -- a payload
//! already in the plane's frame (default: the example with beams along x, turned into the plane);
//! `runs` solves in one process (the later ones find the tables of the first).
#[path="scene/mod.rs"] mod scene;
use atom_motion_gpu::solve::solve;

fn main(){
    let args:Vec<String>=std::env::args().collect();let arg=|i:usize,d:f64|args.get(i).map_or(d,|s|s.parse().unwrap());
    let(t_end,spacing,temperature)=(arg(1,1000.),arg(2,50.),arg(3,30.));
    let mut request=match args.get(4){Some(path)=>scene::load(path),None=>scene::turned(scene::load(scene::PGC))};
    request.temperature_uk=temperature;request.tolerance=0.01;
    request.times_us=(0..=((t_end/spacing).round() as usize)).map(|i|i as f64*spacing).collect();
    for run in 0..(arg(5,1.) as usize){
    if run>0{request.temperature_uk*=1.001;println!("-- run {} (same scene, another temperature)",run+1);}
    let started=std::time::Instant::now();
    let report=|s:&atom_motion_gpu::solution::Solution,label:&str|{
        let n=s.temperature_uk.len();
        for i in (0..n).step_by((n/10).max(1)).chain([n-1]){
            let pf1:f64=s.populations[i][..3].iter().sum();
            let grids:Vec<String>=s.census[i].iter().zip(&s.grids).filter(|(&n,_)|n>0).map(|(n,(x,y))|format!("{x}x{y}:{n}")).collect();
            println!("  t {:7.0}  T {:6.2}±{:4.2} / {:6.2}±{:4.2} uK  nbar {:5.2}/{:5.2}  loss {:.1e}  pF1 {:.4}  SE {:.4}  grids {}",
                i as f64*spacing,s.temperature_uk[i][0],s.temperature_errors_uk[i][0],s.temperature_uk[i][1],s.temperature_errors_uk[i][1],
                s.mean_occupation[i][0],s.mean_occupation[i][1],s.loss_probability[i],pf1,s.sample_errors[i],grids.join(" "));
        }
        println!("{label}: {} trajectories, {:.1} s (steps {:.1} s, observations {:.1} s, moves {:.1} s, tables {:.1} s); {}",s.trajectories,started.elapsed().as_secs_f64(),
            s.seconds[0],s.seconds[1],s.seconds[2],s.seconds[3],s.reason.clone().unwrap_or("targets met".into()));
    };
    let solution=solve(&request,&mut |s|{report(s,"checkpoint");true},&mut |r|{
        println!("running: {:.0} of {:.0} us, {} trajectories done, {}",r.simulated_us,r.duration_us,r.trajectories,r.remaining_ms.map_or("no estimate yet".into(),|ms|format!("about {:.0} s left",ms/1e3)));}).unwrap();
    report(&solution,"final");
    println!("{}",solution.method);
    }
}
