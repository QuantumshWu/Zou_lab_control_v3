//! The trajectories on a ladder of grids: each trajectory runs on the smallest grid that holds it.
//!
//! # The ladder
//! Grid k has N_k = N_0 2^k points per axis spaced dx_k² = 2π ħ/(N_k m ω): its half width and its
//! largest momentum are the same number of thermal widths of the trap of frequency ω, and each
//! grid reaches √2 farther in position and in momentum than the one below. A trajectory moves up
//! when the part of its ⟨x²⟩ or ⟨κ²⟩ (along either axis) beyond 0.7 of its grid's reach exceeds
//! the tolerance ε, and down when the part beyond 0.7 of the smaller grid's reach is below ε/2; the
//! checks are made at every observation. The top grid's absorbing boundary begins where the trap
//! has risen half its depth: an atom that climbs there is lost.
use crate::engine::{Carried,Engine,Internal,Settings,RECORD,REC_KX,REC_TAIL,REC_TAIL2,REC_X};
use crate::gpu::Gpu;
use crate::model::{Grid,Model};
use crate::regrid::resample;
use rayon::prelude::*;
use std::rc::Rc;

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Place{Held{class:usize,slot:usize},Lost}

pub struct Ensemble{
    gpu:Rc<Gpu>,
    template:Settings,
    pub ladder:Vec<Grid>,
    classes:Vec<Option<Engine>>,
    /// Per class, the trajectory in each slot.
    occupant:Vec<Vec<Option<usize>>>,
    /// Per class, the most slots it may take (device memory).
    limit:Vec<usize>,
    pub place:Vec<Place>,
    pub weights:Vec<f64>,
    pub tolerance:f64,
    /// The moves made, and the promotions the top grid's capacity refused.
    pub moves:usize,pub refused:usize,
}

impl Ensemble{
    /// A ladder from `ladder[0]` up, the top with `template.cap`; `memory` the device bytes the
    /// trajectories may take.
    pub fn new(gpu:Rc<Gpu>,template:Settings,ladder:Vec<Grid>,trajectories:usize,memory:usize,tolerance:f64)->Ensemble{
        // State and the two derivative planes per trajectory.
        let limit=ladder.iter().map(|g|(memory/2/(3*5*g.points()*8)).clamp(1,trajectories)).collect();
        Ensemble{gpu,template,classes:ladder.iter().map(|_|None).collect(),occupant:ladder.iter().map(|_|vec![]).collect(),limit,ladder,
            place:vec![Place::Lost;trajectories],weights:vec![1.;trajectories],tolerance,moves:0,refused:0}
    }
    fn engine(&mut self,class:usize,model:&Model)->Result<&mut Engine,String>{
        if self.classes[class].is_none(){
            let top=class+1==self.ladder.len();
            let settings=Settings{grid:self.ladder[class],trajectories:self.limit[class],cap:if top{self.template.cap}else{None},..self.template.clone()};
            let engine=Engine::new(self.gpu.clone(),model,settings)?;engine.clear()?;
            self.occupant[class]=vec![None;self.limit[class]];self.classes[class]=Some(engine);
        }
        Ok(self.classes[class].as_mut().unwrap())
    }
    /// Starts every trajectory on grid `class` in its thermal state (Engine::initialize).
    pub fn start(&mut self,class:usize,internal:&[Internal],temperature_uk:f64,model:&Model)->Result<(),String>{
        let n=self.place.len();
        if self.limit[class]<n{return Err(format!("2D 网格 {}×{} 放不下 {n} 条轨迹（显存不足）",self.ladder[class].nx,self.ladder[class].ny));}
        for c in 0..self.classes.len(){if let Some(e)=&self.classes[c]{e.clear()?;}self.occupant[c].iter_mut().for_each(|o|*o=None);}
        let engine=self.engine(class,model)?;
        let mut ids:Vec<u32>=(0..engine.capacity() as u32).collect();ids.truncate(engine.capacity());engine.set_ids(&ids)?;
        let mut all=internal.to_vec();all.resize(engine.capacity(),internal[0].clone());
        engine.initialize(&all,temperature_uk,model)?;
        // Slots beyond the trajectories stay empty.
        let weights=engine.trajectory_weights.clone();
        for slot in n..engine.capacity(){engine.take(slot)?;}
        for t in 0..n{self.place[t]=Place::Held{class,slot:t};self.occupant[class][t]=Some(t);self.weights[t]=weights[t];}
        Ok(())
    }
    fn active(&self)->Vec<usize>{(0..self.classes.len()).filter(|&c|self.occupant[c].iter().any(|o|o.is_some())).collect()}
    pub fn observe_initial(&mut self,distributions:bool)->Result<(),String>{
        for c in self.active(){self.classes[c].as_mut().unwrap().observe_initial(distributions)?;}Ok(())
    }
    pub fn step(&mut self,observe:bool,distributions:bool)->Result<(),String>{
        for c in self.active(){self.classes[c].as_mut().unwrap().step(observe,distributions)?;}Ok(())
    }
    /// The last observation's record of every held trajectory (None: lost), marking the newly lost.
    pub fn records(&mut self)->Result<Vec<Option<Vec<f32>>>,String>{
        let mut out=vec![None;self.place.len()];
        for c in self.active(){
            let(records,components)=self.classes[c].as_ref().unwrap().read_records()?;
            for(slot,o)in self.occupant[c].clone().iter().enumerate(){
                let Some(t)=*o else{continue};
                if components[slot]==0{self.place[t]=Place::Lost;self.occupant[c][slot]=None;}else{out[t]=Some(records[slot*RECORD..(slot+1)*RECORD].to_vec());}
            }
        }
        Ok(out)
    }
    /// Moves each held trajectory to the grid its tails ask for (`records` of the last observation).
    pub fn migrate(&mut self,records:&[Option<Vec<f32>>],model:&Model)->Result<(),String>{
        let eps=self.tolerance;
        let share=|r:&[f32],tail:usize|->f64{(0..4).map(|d|{let(t,m)=(r[tail+d] as f64,r[if d<2{REC_X+2+d}else{REC_KX+2+(d-2)}] as f64);if m>0.{t/m}else{0.}}).fold(0.,f64::max)};
        let mut wanted:Vec<(usize,usize,usize)>=vec![];// trajectory, from, to
        for(t,r)in records.iter().enumerate(){
            let(Some(r),Place::Held{class,..})=(r,self.place[t])else{continue};
            if share(r,REC_TAIL)>eps&&class+1<self.ladder.len(){wanted.push((t,class,class+1));}
            else if class>0&&share(r,REC_TAIL2)<eps/2.{wanted.push((t,class,class-1));}
        }
        // The planes of each move, resampled in parallel.
        let mut taken:Vec<(usize,usize,Carried)>=vec![];
        // Each grid's free slots, counted as the moves are decided (a slot freed by a move out is
        // reused only after this round's moves).
        let mut free:Vec<Option<usize>>=vec![None;self.ladder.len()];
        for &(t,from,to)in &wanted{
            let Place::Held{slot,..}=self.place[t]else{continue};
            self.engine(to,model)?;
            let f=free[to].get_or_insert_with(||self.occupant[to].iter().filter(|x|x.is_none()).count());
            if *f==0{self.refused+=1;continue;}
            *f-=1;
            let carried=self.classes[from].as_ref().unwrap().take(slot)?;self.occupant[from][slot]=None;
            taken.push((t,to,carried));
            self.place[t]=Place::Lost; // until put
        }
        let grids:Vec<(Grid,Grid)>=taken.iter().map(|(t,to,_)|(self.grid_of(*t,&wanted),self.ladder[*to])).collect();
        taken.par_iter_mut().zip(grids.par_iter()).for_each(|((_,_,c),(from,to))|{
            for plane in c.planes.iter_mut(){*plane=resample(from,to,plane);}
            let n2:f64=c.planes.iter().flat_map(|p|p.iter().map(|z|(z[0] as f64).powi(2)+(z[1] as f64).powi(2))).sum();
            let s=(1./n2.sqrt()) as f32;for plane in c.planes.iter_mut(){for z in plane.iter_mut(){z[0]*=s;z[1]*=s;}}
        });
        for(t,to,c)in taken{
            let slot=self.occupant[to].iter().position(|o|o.is_none()).unwrap();
            self.classes[to].as_ref().unwrap().put(slot,&c)?;self.occupant[to][slot]=Some(t);self.place[t]=Place::Held{class:to,slot};self.moves+=1;
        }
        Ok(())
    }
    fn grid_of(&self,t:usize,wanted:&[(usize,usize,usize)])->Grid{self.ladder[wanted.iter().find(|w|w.0==t).unwrap().1]}
    /// How many trajectories each grid holds.
    pub fn census(&self)->Vec<usize>{self.occupant.iter().map(|o|o.iter().filter(|x|x.is_some()).count()).collect()}
    pub fn engines(&self)->impl Iterator<Item=&Engine>{self.classes.iter().flatten()}
    pub fn engines_mut(&mut self)->impl Iterator<Item=&mut Engine>{self.classes.iter_mut().flatten()}
    pub fn template(&self)->&Settings{&self.template}
    pub fn reseed(&mut self,seed:u64)->Result<(),String>{self.template.seed=seed;for e in self.classes.iter_mut().flatten(){e.reseed(seed);}Ok(())}
    /// Scales the trajectories' weights (a stratified initial state's).
    pub fn scale_weights(&mut self,factors:&[f64])->Result<(),String>{
        for(t,f)in factors.iter().enumerate(){self.weights[t]*=f;}
        for c in 0..self.classes.len(){
            let Some(engine)=self.classes[c].as_mut()else{continue};
            let mut w=engine.trajectory_weights.clone();
            for(slot,o)in self.occupant[c].iter().enumerate(){if let Some(t)=*o{w[slot]=self.weights[t];}}
            engine.set_weights(&w)?;
        }
        Ok(())
    }
}

/// The ladder for a trap of frequency ω (rad/us) and ħ/m (um²/us): N = 16 .. 256.
pub fn ladder(omega:f64,hbar_over_mass:f64)->Vec<Grid>{
    [16usize,32,64,128,256].iter().map(|&n|Grid{nx:n,ny:n,dx:(std::f64::consts::TAU*hbar_over_mass/(n as f64*omega)).sqrt()}).collect()
}
