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
use crate::driver::Buffer;
use crate::engine::{Engine,Internal,Settings,RECORD,REC_KX,REC_TAIL,REC_TAIL2,REC_X};
use crate::gpu::Gpu;
use crate::model::{Grid,Model};
use crate::regrid::interpolation;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Place{Held{class:usize,slot:usize},Lost}

pub struct Ensemble{
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
    /// Wall seconds spent: launching the steps, reading the records (waiting for the GPU), moving,
    /// and building the grids' tables.
    pub seconds:[f64;4],
    /// Per pair of grids (from, to), the interpolation matrices along x and y on the device.
    matrices:HashMap<(usize,usize),(Buffer,Buffer)>,
    /// The moves' scratch on the device (the list, the half-resampled planes, the norms), grown
    /// as needed.
    scratch:[Option<Buffer>;3],
    /// Last: the device's context outlives the engines and the scratch (fields drop in order).
    gpu:Rc<Gpu>,
}

impl Ensemble{
    /// A ladder from `ladder[0]` up, the top with `template.cap`; `memory` the device bytes the
    /// trajectories may take.
    pub fn new(gpu:Rc<Gpu>,template:Settings,ladder:Vec<Grid>,trajectories:usize,memory:usize,tolerance:f64)->Ensemble{
        // State and the two derivative planes per trajectory.
        let limit=ladder.iter().map(|g|(memory/2/(3*5*g.points()*8)).clamp(1,trajectories)).collect();
        Ensemble{gpu,template,classes:ladder.iter().map(|_|None).collect(),occupant:ladder.iter().map(|_|vec![]).collect(),limit,ladder,
            place:vec![Place::Lost;trajectories],weights:vec![1.;trajectories],tolerance,moves:0,refused:0,seconds:[0.;4],matrices:HashMap::new(),scratch:[None,None,None]}
    }
    fn engine(&mut self,class:usize,model:&Model)->Result<&mut Engine,String>{
        if self.classes[class].is_none(){
            let started=std::time::Instant::now();let top=class+1==self.ladder.len();
            let settings=Settings{grid:self.ladder[class],trajectories:self.limit[class],cap:if top{self.template.cap}else{None},..self.template.clone()};
            let mut engine=Engine::new(self.gpu.clone(),model,settings)?;engine.clear()?;engine.active=0;
            self.occupant[class]=vec![None;self.limit[class]];self.classes[class]=Some(engine);
            self.seconds[3]+=started.elapsed().as_secs_f64();
        }
        Ok(self.classes[class].as_mut().unwrap())
    }
    /// Starts the trajectories (one per internal state) on grid `class` in their thermal states
    /// (Engine::initialize).
    pub fn start(&mut self,class:usize,internal:&[Internal],temperature_uk:f64,model:&Model)->Result<(),String>{
        let n=internal.len();
        self.place=vec![Place::Lost;n];self.weights=vec![1.;n];
        if self.limit[class]<n{return Err(format!("2D 网格 {}×{} 放不下 {n} 条轨迹（显存不足）",self.ladder[class].nx,self.ladder[class].ny));}
        for c in 0..self.classes.len(){if let Some(e)=self.classes[c].as_mut(){e.clear()?;e.active=0;}self.occupant[c].iter_mut().for_each(|o|*o=None);}
        let engine=self.engine(class,model)?;
        let ids:Vec<u32>=(0..engine.capacity() as u32).collect();engine.set_ids(&ids)?;
        engine.initialize(internal,temperature_uk,model)?;
        let weights=engine.trajectory_weights.clone();
        for t in 0..n{self.place[t]=Place::Held{class,slot:t};self.occupant[class][t]=Some(t);self.weights[t]=weights[t];}
        self.pack()
    }
    /// The most trajectories a grid takes.
    pub fn capacity(&self,class:usize)->usize{self.limit[class]}
    fn active(&self)->Vec<usize>{(0..self.classes.len()).filter(|&c|self.occupant[c].iter().any(|o|o.is_some())).collect()}
    pub fn observe_initial(&mut self,distributions:bool)->Result<(),String>{
        for c in self.active(){self.classes[c].as_mut().unwrap().observe_initial(distributions)?;}Ok(())
    }
    pub fn step(&mut self,observe:bool,distributions:bool)->Result<(),String>{
        let t=std::time::Instant::now();
        for c in self.active(){self.classes[c].as_mut().unwrap().step(observe,distributions)?;}
        self.seconds[0]+=t.elapsed().as_secs_f64();Ok(())
    }
    /// The last observation's record of every held trajectory (None: lost), marking the newly lost.
    pub fn records(&mut self)->Result<Vec<Option<Vec<f32>>>,String>{
        let t=std::time::Instant::now();
        let mut out=vec![None;self.place.len()];
        for c in self.active(){
            let(records,components)=self.classes[c].as_ref().unwrap().read_records()?;
            for(slot,o)in self.occupant[c].clone().iter().enumerate(){
                let Some(t)=*o else{continue};
                if slot>=components.len(){continue;}
                if components[slot]==0{self.place[t]=Place::Lost;self.occupant[c][slot]=None;}else{out[t]=Some(records[slot*RECORD..(slot+1)*RECORD].to_vec());}
            }
        }
        self.seconds[1]+=t.elapsed().as_secs_f64();
        self.pack()?;
        Ok(out)
    }
    /// Moves each held trajectory to the grid its tails ask for (`records` of the last observation).
    pub fn migrate(&mut self,records:&[Option<Vec<f32>>],model:&Model)->Result<(),String>{
        let(started,built)=(std::time::Instant::now(),self.seconds[3]);
        let eps=self.tolerance;
        let share=|r:&[f32],tail:usize|->f64{(0..4).map(|d|{let(t,m)=(r[tail+d] as f64,r[if d<2{REC_X+2+d}else{REC_KX+2+(d-2)}] as f64);if m>0.{t/m}else{0.}}).fold(0.,f64::max)};
        let mut wanted:Vec<(usize,usize,usize)>=vec![];// trajectory, from, to
        for(t,r)in records.iter().enumerate(){
            let(Some(r),Place::Held{class,..})=(r,self.place[t])else{continue};
            if share(r,REC_TAIL)>eps&&class+1<self.ladder.len(){wanted.push((t,class,class+1));}
            else if class>0&&share(r,REC_TAIL2)<eps/2.{wanted.push((t,class,class-1));}
        }
        // The slots: each grid's free ones counted as the moves are decided (a slot freed by a move
        // out is reused only in a later round).
        for &(_,_,to)in &wanted{self.engine(to,model)?;}
        let mut free:Vec<Vec<usize>>=self.occupant.iter().map(|o|(0..o.len()).rev().filter(|&s|o[s].is_none()).collect()).collect();
        // A move into a grid lands in its first free slots; the grid's active count covers them.
        let mut pairs:HashMap<(usize,usize),Vec<[i32;2]>>=HashMap::new();
        for &(t,from,to)in &wanted{
            let Place::Held{slot,..}=self.place[t]else{continue};
            let Some(target)=free[to].pop()else{self.refused+=1;continue};
            self.occupant[from][slot]=None;self.occupant[to][target]=Some(t);self.place[t]=Place::Held{class:to,slot:target};self.moves+=1;
            pairs.entry((from,to)).or_default().push([slot as i32,target as i32]);
        }
        for c in 0..self.classes.len(){if let Some(e)=self.classes[c].as_mut(){e.active=self.occupant[c].iter().rposition(|x|x.is_some()).map_or(0,|i|i+1).max(e.active);}}
        let(rows,cols,state)=(self.gpu.function("move_rows")?,self.gpu.function("move_cols")?,self.gpu.function("move_state")?);
        // The moves run on the shared stream: the engines' own (the slots packed after the
        // records) done first.
        if !pairs.is_empty(){for e in self.classes.iter().flatten(){e.synchronize()?;}}
        for((from,to),moves)in pairs{
            let(a,b)=(self.ladder[from],self.ladder[to]);
            if !self.matrices.contains_key(&(from,to)){
                let up=|m:Vec<f64>|->Result<Buffer,String>{let v:Vec<f32>=m.iter().map(|&x|x as f32).collect();let buf=Buffer::new(v.len()*4)?;self.gpu.stream.upload(&buf,&v)?;Ok(buf)};
                let pair=(up(interpolation(a.nx,a.dx,b.nx,b.dx))?,up(interpolation(a.ny,a.dx,b.ny,b.dx))?);self.matrices.insert((from,to),pair);
            }
            let(ax,ay)=&self.matrices[&(from,to)];
            let count=moves.len();
            for(k,bytes)in [count*8,count*5*a.ny*b.nx*8,count*4].into_iter().enumerate(){
                if self.scratch[k].as_ref().is_none_or(|x|x.bytes<bytes){self.scratch[k]=Some(Buffer::new(bytes)?);}
            }
            let s=&self.gpu.stream;let[Some(list),Some(tmp),Some(norm)]=&self.scratch else{unreachable!()};
            s.upload(list,&moves)?;s.zero_at(norm,0,count*4)?;
            let(source,target)=(self.classes[from].as_ref().unwrap(),self.classes[to].as_ref().unwrap());
            let(src_traj,src_state,src_weights,src_components)=source.device();let(dst_traj,dst_state,dst_weights,_)=target.device();
            let n=count as i32;let blocks=|total:usize|(total.div_ceil(256) as u32,1,1);
            s.launch(&rows,blocks(count*5*a.ny*b.nx),256,0,&[&src_state,&(a.nx as i32),&(a.ny as i32),&src_components,&list.pointer,&n,&ax.pointer,&(b.nx as i32),&tmp.pointer])?;
            s.launch(&cols,blocks(count*5*b.ny*b.nx),256,0,&[&tmp.pointer,&(a.ny as i32),&src_components,&list.pointer,&n,&ay.pointer,&(b.nx as i32),&(b.ny as i32),&dst_state,&norm.pointer])?;
            s.launch(&state,(count as u32,1,1),256,0,&[&src_traj,&dst_traj,&src_weights,&dst_weights,&list.pointer,&norm.pointer,&dst_state,&(b.points() as i32)])?;
            s.synchronize()?;
            for c in [from,to]{self.classes[c].as_mut().unwrap().touched();}
        }
        self.pack()?;
        self.seconds[2]+=started.elapsed().as_secs_f64()-(self.seconds[3]-built);
        Ok(())
    }
    /// Packs every grid's trajectories into its first slots (the passes cover only those): each
    /// hole filled from the last slot in use.
    fn pack(&mut self)->Result<(),String>{
        for c in 0..self.classes.len(){
            let Some(engine)=self.classes[c].as_mut()else{continue};
            let o=&mut self.occupant[c];
            let mut end=o.iter().rposition(|x|x.is_some()).map_or(0,|i|i+1);
            let mut hole=0;
            loop{
                while hole<end&&o[hole].is_some(){hole+=1;}
                while end>0&&o[end-1].is_none(){end-=1;}
                if hole>=end{break;}
                let t=o[end-1].take().unwrap();engine.copy_slot(end-1,hole)?;o[hole]=Some(t);self.place[t]=Place::Held{class:c,slot:hole};end-=1;
            }
            engine.active=end;
        }
        Ok(())
    }
    /// How many trajectories each grid holds.
    pub fn census(&self)->Vec<usize>{self.occupant.iter().map(|o|o.iter().filter(|x|x.is_some()).count()).collect()}
    pub fn engines(&self)->impl Iterator<Item=&Engine>{self.classes.iter().flatten()}
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
