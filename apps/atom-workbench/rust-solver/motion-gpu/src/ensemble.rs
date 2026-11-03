//! The trajectories on a ladder of grids: each trajectory runs on the smallest grid that holds it,
//! along each axis apart.
//!
//! # The ladder
//! Along each axis, level k has N_k = N_0 2^k points spaced d_k² = 2π ħ/(N_k m ω) for the trap's
//! frequency ω along that axis: its half width and its largest momentum are the same number of
//! thermal widths, and each level reaches √2 farther in position and in momentum than the one
//! below. A grid is a level along x and one along y (a class), so an atom hot along one axis and
//! cold along the other takes a long narrow grid. Along each axis apart, a trajectory moves up a
//! level when the part of its ⟨x²⟩ or ⟨κ²⟩ along that axis beyond 0.7 of its grid's reach exceeds
//! the tolerance ε, and down when the part beyond 0.7 of the next smaller level's reach is below
//! ε/2; the checks are made at every observation. The absorbing boundary begins where the trap
//! has risen half its depth (within the largest grid): an atom that climbs there is lost.
//!
//! # Device memory
//! The trajectories' grids share one budget: a grid's engine is made, and grown, as trajectories
//! move in, at most the batch. When the grids moved into would not fit, the idle memory is let go
//! first (the engines of grids nobody holds, then the free slots of the others); a move the
//! budget still cannot hold is refused (the trajectory stays, counted in `refused`). A batch
//! starts on its first grid with the last batch's grids kept, emptied, as far as the budget
//! allows; the next batch is sized so that the last one's largest footprint per trajectory fits
//! (`batch_room`).
use crate::driver::Buffer;
use crate::engine::{Engine,Internal,Settings,RECORD,REC_KX,REC_TAIL,REC_TAIL2,REC_X};
use crate::gpu::Gpu;
use crate::model::{Grid,Model};
use crate::regrid::interpolation;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Place{Held{class:usize,slot:usize},Lost}

/// The levels along each axis: points and spacing (um), N = 16 .. 256.
pub const LEVELS:usize=5;
/// The levels along an axis whose trap frequency is ω (rad/us), for ħ/m (um²/us).
pub fn levels(omega:f64,hbar_over_mass:f64)->[(usize,f64);LEVELS]{
    std::array::from_fn(|k|{let n=16usize<<k;(n,(std::f64::consts::TAU*hbar_over_mass/(n as f64*omega)).sqrt())})
}
/// The class of the levels (lx, ly), and back.
pub fn class(lx:usize,ly:usize)->usize{lx*LEVELS+ly}
pub fn class_levels(class:usize)->[usize;2]{[class/LEVELS,class%LEVELS]}

pub struct Ensemble{
    template:Settings,
    /// The grids by class (`class`).
    pub grids:Vec<Grid>,
    classes:Vec<Option<Engine>>,
    /// Per class, the trajectory in each slot.
    occupant:Vec<Vec<Option<usize>>>,
    /// The most trajectories a grid takes (the batch), and the device bytes the grids may take.
    most:usize,budget:usize,
    /// The most device bytes the batch's held trajectories took at once (`footprint`).
    peak:usize,
    pub place:Vec<Place>,
    pub weights:Vec<f64>,
    pub tolerance:f64,
    /// The moves made, and the moves the device memory refused.
    pub moves:usize,pub refused:usize,
    /// The steps of the run so far: every grid's engine counts its steps from the batch's start.
    steps:u64,
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
    /// The grids of the levels along x and y; `most` trajectories at most on a grid, `memory` the
    /// device bytes the grids may take.
    pub fn new(gpu:Rc<Gpu>,template:Settings,levels:[[(usize,f64);LEVELS];2],most:usize,memory:usize,tolerance:f64)->Ensemble{
        let grids:Vec<Grid>=(0..LEVELS*LEVELS).map(|c|{let[lx,ly]=class_levels(c);let((nx,dx),(ny,dy))=(levels[0][lx],levels[1][ly]);Grid{nx,ny,dx,dy}}).collect();
        Ensemble{gpu,template,classes:grids.iter().map(|_|None).collect(),occupant:grids.iter().map(|_|vec![]).collect(),grids,most,budget:memory,peak:0,
            place:vec![],weights:vec![],tolerance,moves:0,refused:0,steps:0,seconds:[0.;4],matrices:HashMap::new(),scratch:[None,None,None]}
    }
    /// The device bytes the grids' engines take now.
    fn allocated(&self)->usize{self.classes.iter().flatten().map(|e|e.bytes()).sum()}
    /// The trajectories grid `class` holds.
    fn held(&self,class:usize)->usize{self.occupant[class].iter().filter(|o|o.is_some()).count()}
    /// Room on the grids for the trajectories moving in (grid, count), as far as the budget
    /// allows. When the grids moved into would not fit, idle memory is let go first: the engines
    /// of grids nobody holds or moves into, then the free slots of the others, the most bytes
    /// first (their slots in use are packed at the front). A grid moved into then grows to at
    /// least its trajectories, doubling, at most `most` slots.
    fn fit(&mut self,incoming:&[(usize,usize)],model:&Model)->Result<(),String>{
        let started=std::time::Instant::now();
        let n=self.grids.len();
        let mut want:Vec<usize>=(0..n).map(|c|self.held(c)).collect();
        for &(c,k)in incoming{want[c]=(want[c]+k).min(self.most);}
        let mut targets:Vec<usize>=incoming.iter().map(|&(c,_)|c).collect();targets.sort();targets.dedup();
        let per=|g:Grid|Engine::bytes_per_trajectory(g);
        let needed:usize=targets.iter().map(|&c|{
            let have=self.occupant[c].len();
            if want[c]<=have{0}else{(want[c]-have)*per(self.grids[c])+if self.classes[c].is_none(){Engine::bytes_of_tables(self.grids[c])}else{0}}
        }).sum();
        if self.allocated()+needed>self.budget{
            let mut idle:Vec<(usize,usize)>=(0..n).filter_map(|c|{
                let e=self.classes[c].as_ref()?;
                let bytes=if want[c]==0{e.bytes()}else{e.capacity().saturating_sub(want[c])*per(self.grids[c])};
                (bytes>0).then_some((bytes,c))
            }).collect();
            idle.sort_by(|a,b|b.cmp(a));
            for(_,c)in idle{
                if self.allocated()+needed<=self.budget{break;}
                if want[c]==0{self.classes[c]=None;self.occupant[c].clear();}
                else{self.classes[c].as_mut().unwrap().resize(want[c])?;self.occupant[c].truncate(want[c]);}
            }
        }
        for c in targets{
            let have=self.occupant[c].len();if want[c]<=have{continue;}
            let g=self.grids[c];
            let tables=if self.classes[c].is_none(){Engine::bytes_of_tables(g)}else{0};
            let spare=self.budget.saturating_sub(self.allocated()+tables)/per(g);
            let target=want[c].max(2*have).min(self.most).min(have+spare);
            if target<=have{continue;}
            let made=match self.classes[c].as_mut(){
                Some(engine)=>engine.resize(target),
                None=>Engine::new(self.gpu.clone(),model,Settings{grid:g,trajectories:target,..self.template.clone()})
                    .and_then(|mut e|{e.clear()?;e.active=0;self.classes[c]=Some(e);Ok(())}),
            };
            // The device short of memory after all (another program took it): the budget is what
            // the grids have, and the moves it cannot hold are refused.
            if let Err(e)=made{if e.contains("OUT_OF_MEMORY"){self.budget=self.allocated();continue;}return Err(e);}
            self.occupant[c].resize(target,None);
        }
        self.seconds[3]+=started.elapsed().as_secs_f64();
        Ok(())
    }
    /// The device bytes the held trajectories take.
    fn footprint(&self)->usize{(0..self.grids.len()).map(|c|self.held(c)*Engine::bytes_per_trajectory(self.grids[c])).sum()}
    /// The trajectories a batch may take for its largest footprint to fit the budget beside the
    /// grids' tables, at the last batch's largest footprint per trajectory (None before a batch).
    pub fn batch_room(&self)->Option<usize>{
        let n=self.place.len();if self.peak==0||n==0{return None;}
        let tables:usize=self.classes.iter().flatten().map(|e|Engine::bytes_of_tables(e.grid())).sum();
        Some((self.budget.saturating_sub(tables) as f64*n as f64/self.peak as f64) as usize)
    }
    /// The most trajectories a batch may start with on grid `class`: the batch, as far as the
    /// budget holds them there.
    pub fn capacity(&self,class:usize)->usize{let g=self.grids[class];self.most.min(self.budget.saturating_sub(Engine::bytes_of_tables(g))/Engine::bytes_per_trajectory(g)).max(1)}
    /// Starts the trajectories (one per internal state) on grid `class` in their thermal states
    /// (Engine::initialize).
    pub fn start(&mut self,class:usize,internal:&[Internal],temperature_uk:f64,model:&Model)->Result<(),String>{
        let n=internal.len();
        self.place=vec![Place::Lost;n];self.weights=vec![1.;n];self.steps=0;self.peak=0;
        for c in 0..self.classes.len(){if let Some(e)=self.classes[c].as_mut(){e.clear()?;e.active=0;}self.occupant[c].iter_mut().for_each(|o|*o=None);}
        self.fit(&[(class,n)],model)?;
        if self.occupant[class].len()<n{let g=self.grids[class];return Err(format!("2D 网格 {}×{} 放不下 {n} 条轨迹（显存不足）",g.nx,g.ny));}
        let engine=self.classes[class].as_mut().unwrap();
        let ids:Vec<u32>=(0..engine.capacity() as u32).collect();engine.set_ids(&ids)?;
        engine.initialize(internal,temperature_uk,model)?;
        let weights=engine.trajectory_weights.clone();
        for t in 0..n{self.place[t]=Place::Held{class,slot:t};self.occupant[class][t]=Some(t);self.weights[t]=weights[t];}
        self.pack()?;
        self.peak=self.footprint();
        Ok(())
    }
    fn active(&self)->Vec<usize>{(0..self.classes.len()).filter(|&c|self.occupant[c].iter().any(|o|o.is_some())).collect()}
    pub fn observe_initial(&mut self,distributions:bool)->Result<(),String>{
        for c in self.active(){self.classes[c].as_mut().unwrap().observe_initial(distributions)?;}Ok(())
    }
    pub fn step(&mut self,observe:bool,distributions:bool)->Result<(),String>{
        let t=std::time::Instant::now();let n=self.steps;
        for c in self.active(){self.classes[c].as_mut().unwrap().step(n,observe,distributions)?;}
        self.steps+=1;
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
    /// Moves each held trajectory to the grid its tails ask for, along each axis apart (`records`
    /// of the last observation).
    pub fn migrate(&mut self,records:&[Option<Vec<f32>>],model:&Model)->Result<(),String>{
        let(started,built)=(std::time::Instant::now(),self.seconds[3]);
        let eps=self.tolerance;
        // Along axis a: the larger of the parts of ⟨a²⟩ and of ⟨κ_a²⟩ beyond the record's `tail`.
        let share=|r:&[f32],tail:usize,a:usize|->f64{[(tail+a,REC_X+2+a),(tail+2+a,REC_KX+2+a)].iter().map(|&(t,m)|{let(t,m)=(r[t] as f64,r[m] as f64);if m>0.{t/m}else{0.}}).fold(0.,f64::max)};
        let mut wanted:Vec<(usize,usize,usize)>=vec![];// trajectory, from, to
        for(t,r)in records.iter().enumerate(){
            let(Some(r),Place::Held{class:from,..})=(r,self.place[t])else{continue};
            let level=class_levels(from);let mut to=level;
            for a in 0..2{
                if share(r,REC_TAIL,a)>eps&&level[a]+1<LEVELS{to[a]+=1;}
                else if level[a]>0&&share(r,REC_TAIL2,a)<eps/2.{to[a]-=1;}
            }
            if to!=level{wanted.push((t,from,class(to[0],to[1])));}
        }
        // Room on each grid moved to, then its free slots (a slot freed by a move out is reused
        // only in a later round).
        let incoming:Vec<(usize,usize)>=wanted.iter().map(|&(_,_,to)|(to,1)).collect();
        self.fit(&incoming,model)?;
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
        // The moves run on the shared stream, in order: the engines' own work (the slots packed
        // after the records, a grown engine's copies) done first, the moves done before the
        // engines step on.
        if pairs.is_empty(){self.seconds[2]+=started.elapsed().as_secs_f64()-(self.seconds[3]-built);return Ok(());}
        for e in self.classes.iter().flatten(){e.synchronize()?;}
        let mut pairs:Vec<((usize,usize),Vec<[i32;2]>)>=pairs.into_iter().collect();pairs.sort_by_key(|p|p.0);
        for((from,to),moves)in pairs{
            let(a,b)=(self.grids[from],self.grids[to]);
            if !self.matrices.contains_key(&(from,to)){
                let up=|m:Vec<f64>|->Result<Buffer,String>{let v:Vec<f32>=m.iter().map(|&x|x as f32).collect();let buf=Buffer::new(v.len()*4)?;self.gpu.stream.upload(&buf,&v)?;Ok(buf)};
                let pair=(up(interpolation(a.nx,a.dx,b.nx,b.dx))?,up(interpolation(a.ny,a.dy,b.ny,b.dy))?);self.matrices.insert((from,to),pair);
            }
            let(ax,ay)=&self.matrices[&(from,to)];
            let count=moves.len();
            for(k,bytes)in [count*8,count*5*a.ny*b.nx*8,count*4].into_iter().enumerate(){
                // A scratch too small is replaced once the moves before it have run.
                if self.scratch[k].as_ref().is_none_or(|x|x.bytes<bytes){self.gpu.stream.synchronize()?;self.scratch[k]=Some(Buffer::new(bytes)?);}
            }
            let s=&self.gpu.stream;let[Some(list),Some(tmp),Some(norm)]=&self.scratch else{unreachable!()};
            s.upload(list,&moves)?;s.zero_at(norm,0,count*4)?;
            let(source,target)=(self.classes[from].as_ref().unwrap(),self.classes[to].as_ref().unwrap());
            let(src_traj,src_state,src_weights,src_components)=source.device();let(dst_traj,dst_state,dst_weights,_)=target.device();
            let n=count as i32;let blocks=|total:usize|(total.div_ceil(256) as u32,1,1);
            s.launch(&rows,blocks(count*5*a.ny*b.nx),256,0,&[&src_state,&(a.nx as i32),&(a.ny as i32),&src_components,&list.pointer,&n,&ax.pointer,&(b.nx as i32),&tmp.pointer])?;
            s.launch(&cols,blocks(count*5*b.ny*b.nx),256,0,&[&tmp.pointer,&(a.ny as i32),&src_components,&list.pointer,&n,&ay.pointer,&(b.nx as i32),&(b.ny as i32),&dst_state,&norm.pointer])?;
            s.launch(&state,(count as u32,1,1),256,0,&[&src_traj,&dst_traj,&src_weights,&dst_weights,&list.pointer,&norm.pointer,&dst_state,&(b.points() as i32)])?;
        }
        self.gpu.stream.synchronize()?;
        self.pack()?;
        self.peak=self.peak.max(self.footprint());
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
    /// How many trajectories each grid holds (by class).
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
