//! A solve's model for the kernels (kernels/motion3d.cu): its tables in single precision, the
//! descriptors the kernel walks -- each struct here the #[repr(C)] twin of the C struct of its name
//! --, the exact response's matrices and the step's settings. Built once per solve from model.rs
//! and ground.rs and kept on the device while its batches run.
use super::driver::{Buffer,Stream};
use crate::engine::Settings;
use crate::linear::Dense;
use crate::model::{Model,GROUND,MOST_BEAMS,MOST_CARRIERS,MOST_FOCUSED,MOST_RANK,MOST_STATES};
use crate::Complex;

/// The most states of a cluster of ground states the kernel holds (its bucket), of a cluster its exact
/// step takes (explicit excited states, beats: engine.rs # The exact step) and the most parts of its
/// components, the excited states, the most classes of a cluster's jumps and the most listed outputs
/// (lib.rs `simulate`'s distributions).
pub const MAXN:usize=MOST_STATES;
pub const MAXM:usize=16;
pub const MOST_PARTS:usize=32;
pub const NE:usize=24;
pub const MOST_CLASSES:usize=32;
pub const MOST_LISTED:usize=32;

#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Beam{
    direction:[f32;3],k:f32,wave:[f32;3],inv_w2:f32,inv_rayleigh:f32,recoil:f32,inv_pole:f32,trap:i32,
    carrier:i32,near:i32,coupled:u32,coefficients:[[f32;2];MOST_RANK],
}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Carrier{rank:i32,beams:i32,first:i32,pad:i32,detuning:f64,wave:f64}
/// The kernel's `Unit` (ground.rs `Unit`): its beams and weights' place, N, s(0) and its profile weight.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Unit{first:i32,count:i32,norm:f32,focus:f32,profile:f32,pad:i32}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Cluster{n:i32,size:i32,table:i32,h0:i32,groups:i32,ngroups:i32,classes:i32,nclasses:i32,coherent:i32,ncoherent:i32,carriers:i32,fourth:i32,frames:i32,
    positions:i32,beats:i32,nbeats:i32,n0:i32,exact:i32,parts:i32,omegas:i32,terms:i32,nterms:i32,emission:i32,excited:i32,group_parts:i32,residual_parts:i32,states:[i32;MAXM]}
/// The kernel's `Term`: a term of a cluster's explicit states (model.rs `exact_terms`): a kept drive (0), a
/// decay cross term (1), a decay (2) or a dressing's drive (3), its row, column, carrier, part, its
/// mirror's part, its tables' place, a drive's U_k.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Term{kind:i32,row:i32,col:i32,carrier:i32,part:i32,mirror:i32,at:i32,pad:i32,u:[[f32;2];MOST_RANK]}
/// The kernel's `Direct`: a class's photon from an explicit state (ground.rs `Class::direct`): its
/// state's place in the cluster, the ground state it lands in and its row in the target, its residual.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Direct{state:i32,to:i32,row:i32,pad:i32,residual:f64}
/// The kernel's `Fourth`: a cluster's fourth-order tables' places per carrier (−1: none).
#[repr(C)]#[derive(Clone,Copy,Debug)]
pub struct Fourth{
    quartic:[[i32;MOST_CARRIERS];MOST_CARRIERS],quadratic:[i32;MOST_CARRIERS],rr:[i32;MOST_CARRIERS],yw:[i32;MOST_CARRIERS],
    jumps:[[i32;MOST_CARRIERS];MOST_CARRIERS],p:[[i32;MOST_CARRIERS];MOST_CARRIERS],rr_all:[i32;MOST_CARRIERS],photons:[i32;MOST_CARRIERS],
    on:i32,carriers:[i32;MOST_CARRIERS],pad:[i32;3],
}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Group{absorbed:i32,emitted:i32,hermitian:i32,pairs:i32,npairs:i32,s:i32,q:i32,count:i32,residual:f64}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Class{target:i32,photons:i32,nphotons:i32,carriers:i32,rows:i32,direct:i32,ndirect:i32,pad:i32,wave_number:f64}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Photon{carrier:i32,col:i32,to:i32,row:i32,residual:f64}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Coherent{channels:i32,count:i32}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Channel{carrier:i32,col:i32,residual:f64}

/// The kernel's `Program`, passed by value: the descriptors' device addresses, the model's sizes and
/// the step's settings.
#[repr(C)]#[derive(Clone,Copy,Debug)]
pub struct Params{
    beams:u64,beam_list:u64,carriers:u64,clusters:u64,groups:u64,pairs:u64,classes:u64,photons:u64,coherent:u64,channels:u64,carrier_tables:u64,tables:u64,
    excited:u64,diagonal:u64,couplings:u64,emission:u64,class_rows:u64,fourths:u64,frames:u64,item_first:u64,items:u64,residuals:u64,
    units:u64,unit_beams:u64,unit_weights:u64,
    terms:u64,omegas:u64,group_parts:u64,residual_parts:u64,excited_list:u64,directs:u64,levels:u64,level_first:u64,decay_root:u64,
    nbeams:i32,ncarriers:i32,nclusters:i32,focused:i32,nodes:i32,fields:i32,pub(crate) nnear:i32,doppler:i32,loss:i32,weights:i32,nunits:i32,
    exact_at:i32,transient_at:i32,nlevels:i32,
    focused_unit:[i32;MOST_FOCUSED],stride:[i32;MOST_FOCUSED],
    field_carrier:[i32;MOST_CARRIERS*MOST_RANK],field_rank:[i32;MOST_CARRIERS*MOST_RANK],
    ground_states:[i32;GROUND],excited_states:[i32;NE],ground_cluster:[i32;GROUND],ground_position:[i32;GROUND],
    hm:f32,light2:f32,allowance:f32,truncation:f32,
    axes32:[[f32;3];3],stiffness32:[f32;3],
    hm64:f64,dt:f64,
    axes:[[f64;3];3],omega:[f64;3],stiffness:[f64;3],
    half_flow:[[f64;3];3],
    region:f64,
    node_phase:f64,beats:f64,horizon:f64,
}

/// The kernel's `State`: a trajectory between launches.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct State{
    pub(crate) q:[f64;3],pub(crate) p:[f64;3],pub(crate) time:f64,pub(crate) logn:f64,pub(crate) threshold:f64,pub(crate) log_threshold:f64,pub(crate) prefix:[f64;15],
    pub(crate) q0:[f64;3],pub(crate) p0:[f64;3],pub(crate) twin_logn:f64,pub(crate) twin_threshold:f64,pub(crate) twin_log_threshold:f64,
    pub(crate) psi:[[f32;2];MAXM],pub(crate) twin:[[f32;2];MAXN],pub(crate) doppler:f32,pub(crate) cluster:i32,pub(crate) jumps:i32,pub(crate) death:i32,
    pub(crate) index:u32,pub(crate) twin_cluster:i32,pub(crate) twin_jumps:i32,pub(crate) beats:f32,
}
/// The kernel's `Held`: the held twin's device data (all zero: no twin).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Held{pub(crate) u:u64,pub(crate) h:u64,pub(crate) l:u64,pub(crate) readout:u64,pub(crate) blocks:u64,pub(crate) jumps:u64}
/// The kernel's `Run`: a launch's wave and output intervals.
#[repr(C)]#[derive(Clone,Copy,Debug)]
pub struct Run{
    pub(crate) states:u64,pub(crate) rings:u64,pub(crate) samples:u64,pub(crate) sums:u64,pub(crate) scratch:u64,pub(crate) tasks:u64,pub(crate) profile:u64,
    pub(crate) records:u64,pub(crate) harmonic:u64,pub(crate) costs:u64,pub(crate) held:Held,pub(crate) seed:u64,
    pub(crate) outputs:i32,pub(crate) steps:i32,pub(crate) reach:i32,pub(crate) begin:i32,pub(crate) end:i32,
    pub(crate) half:[i32;3],pub(crate) ring:[i32;3],pub(crate) ring_at:[i32;3],
    pub(crate) ring_stride:i32,pub(crate) scratch_stride:i32,pub(crate) nlisted:i32,pub(crate) pad:i32,
    pub(crate) listed:[i32;MOST_LISTED],
}

/// The sizes, offsets and constants the kernel's `layout` reports, for the check that the twins and
/// the constants they share match.
pub fn layout()->[u32;25]{
    use std::mem::{offset_of,size_of};
    use super::{HARMONIC,MOST_INTERVALS,MOTION,RECORD,SLOT_RECORD,TWIN,WARPS};
    [size_of::<Beam>(),size_of::<Carrier>(),size_of::<Cluster>(),size_of::<Group>(),size_of::<Class>(),size_of::<Photon>(),size_of::<Coherent>(),size_of::<Channel>(),
        size_of::<Params>(),size_of::<State>(),size_of::<Run>(),offset_of!(Params,hm),offset_of!(Params,half_flow),offset_of!(Run,listed),offset_of!(State,psi),offset_of!(State,index),
        MOST_INTERVALS,SLOT_RECORD,RECORD,WARPS,MAXN,TWIN,MOTION,HARMONIC,size_of::<Fourth>()].map(|x|x as u32)
}

/// The model on the device: the kernel's parameters, the buffers they point into, and the most
/// photons of a cluster's jumps.
pub struct Program{pub params:Params,photons:usize,weights:usize,exact:usize,transient:usize,_buffers:Vec<Buffer>}

fn c32(z:Complex)->[f32;2]{[z.re as f32,z.im as f32]}

impl Program{
    /// The program of a model and its step's settings, uploaded; Err with the reason (Chinese, as the
    /// app shows it) when the model is beyond what the kernel holds.
    pub fn new(stream:&Stream,model:&Model,settings:&Settings)->Result<Program,String>{
        let g=&model.ground;let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
        let beyond=|what:String|->String{format!("模型超出 GPU 内核的范围（{what}）")};
        if ng!=GROUND||ne!=NE{return Err(beyond(format!("{ng} 个基态、{ne} 个激发态")));}
        if settings.free!=[true;3]{return Err(beyond("运动只沿部分坐标轴".into()));}
        // The exact step's clusters (explicit excited states, beats) in its bucket and its parts.
        if let Some(k)=(0..model.clusters()).find(|&k|(model.mixed(k)||model.beating(k))&&model.dimension(k)>MAXM){
            return Err(beyond(format!("子空间 {} 个态（{} 个激发态显式）超出精确步的 {MAXM} 个",model.dimension(k),model.dimension(k)-model.ground_dimension(k))));}
        if let Some(k)=(0..model.clusters()).find(|&k|1+model.beats(k).0.len()>MOST_PARTS){return Err(beyond(format!("子空间 {} 的 {} 个残余拍频",k,model.beats(k).0.len()/2)));}
        // The fourth order's tables are fourth2's slabs now (model.rs `FourthLayout`), which the kernel does
        // not read yet: a fourth-order model stays on the CPU until it does.
        if model.fourth{return Err(beyond("四阶折叠".into()));}
        if model.geometry.len()>MOST_BEAMS||model.carriers.len()>MOST_CARRIERS||model.focused.len()>MOST_FOCUSED{return Err(beyond("光束或强度维度太多".into()));}
        if let Some(c)=g.clusters.iter().find(|c|c.states.len()>MAXN||c.classes.len()>MOST_CLASSES){return Err(beyond(format!("子空间 {} 个态、{} 类光子",c.states.len(),c.classes.len())));}
        let hm=model.hbar_over_mass;
        // The beams: each one's carrier and place among them, and the near beams' order.
        let mut owner=vec![None;model.geometry.len()];
        for(c,carrier)in model.carriers.iter().enumerate(){for(j,&l)in carrier.beams.iter().enumerate(){owner[l]=Some((c,j));}}
        let mut near=0;
        let beams:Vec<Beam>=model.geometry.iter().enumerate().map(|(l,geometry)|{
            let k=geometry.wave_number;let(inv_w2,inv_rayleigh)=match geometry.waist{Some(w)=>(1./(w*w),2./(k*w*w)),None=>(0.,0.)};
            let mut beam=Beam{direction:geometry.direction.map(|x|x as f32),k:k as f32,wave:geometry.wave_vector().map(|x|x as f32),inv_w2:inv_w2 as f32,
                inv_rayleigh:inv_rayleigh as f32,recoil:(0.5*hm*k*k) as f32,inv_pole:0.,trap:model.trap_beams.contains(&l) as i32,carrier:-1,near:-1,coupled:0,coefficients:[[0.;2];MOST_RANK]};
            if let Some((c,j))=owner[l]{
                let pole=g.poles[c];beam.inv_pole=if pole>0.&&pole.is_finite(){(1./pole) as f32}else{0.};
                beam.carrier=c as i32;beam.near=near;near+=1;
                for b in 0..ng{if (0..ne).any(|i|g.couplings[l][(i,b)]!=Complex::ZERO){beam.coupled|=1<<b;}}
                for k in 0..model.carriers[c].rank{beam.coefficients[k]=c32(model.carriers[c].coefficients[j][k]);}
            }
            beam}).collect();
        let mut beam_list=vec![];
        let carriers:Vec<Carrier>=model.carriers.iter().enumerate().map(|(c,carrier)|{
            let first=beam_list.len() as i32;beam_list.extend(carrier.beams.iter().map(|&l|l as i32));
            Carrier{rank:carrier.rank as i32,beams:carrier.beams.len() as i32,first,pad:0,detuning:g.carriers[c].detuning,wave:carrier.wave_number}}).collect();
        // The clusters: their tables one after another, their groups, classes and coherent groups.
        let nodes=model.tables.nodes;let points=nodes.pow(model.focused.len() as u32);
        let(mut clusters,mut groups,mut pairs,mut classes,mut photons,mut class_rows,mut coherent,mut channels,mut carrier_tables,mut tables)=
            (vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![]);
        let(fourths,mut frames,weights):(Vec<Fourth>,Vec<f64>,usize)=(vec![],vec![],0);
        let(mut item_first,mut items,mut residuals):(Vec<i32>,Vec<[i32;4]>,Vec<f64>)=(vec![],vec![],vec![]);
        let(mut terms,mut omegas,mut group_parts,mut residual_parts,mut excited_list,mut directs):(Vec<Term>,Vec<f64>,Vec<i32>,Vec<i32>,Vec<i32>,Vec<Direct>)=(vec![],vec![],vec![],vec![],vec![],vec![]);
        // The exact step's room in a trajectory's scratch: per cluster its parts' H and G, their working
        // room and its norms (the kernel's `exact_segment`), the largest of every cluster's where one takes it.
        let takes=(0..model.clusters()).any(|k|model.mixed(k)||model.beating(k));let mut exact=0;
        for(k,(layout,cl))in model.layouts.iter().zip(&g.clusters).enumerate(){
            let parts=1+model.beats(k).0.len();let(n,n0)=(layout.n,layout.n0);
            if takes{exact=exact.max((5*parts+4)*n*n+4*parts);}
            let(list,u,emission)=model.exact_terms(k);
            let mut cluster=Cluster{n:n as i32,size:layout.size as i32,table:(tables.len()) as i32,h0:layout.h0 as i32,groups:groups.len() as i32,ngroups:layout.groups.len() as i32,
                classes:classes.len() as i32,nclasses:cl.classes.len() as i32,coherent:coherent.len() as i32,ncoherent:cl.coherent.len() as i32,carriers:carrier_tables.len() as i32,
                fourth:-1,frames:0,positions:item_first.len() as i32,beats:residuals.len() as i32,nbeats:0,n0:n0 as i32,exact:(model.mixed(k)||model.beating(k)) as i32,
                parts:parts as i32,omegas:omegas.len() as i32,terms:terms.len() as i32,nterms:list.len() as i32,emission:emission as i32,excited:excited_list.len() as i32,
                group_parts:group_parts.len() as i32,residual_parts:residual_parts.len() as i32,states:[0;MAXM]};
            for(x,&b)in cl.states.iter().enumerate(){cluster.states[x]=b as i32;}
            // Its beats (their period last, 0 without one), its explicit states and its terms (model.rs `exact_terms`).
            let(beats_k,period)=model.beats(k);omegas.extend(beats_k);omegas.push(period.unwrap_or(0.));excited_list.extend(cl.excited.iter().map(|&e|e as i32));
            terms.extend(list.iter().zip(&u).map(|(t,u)|Term{kind:t[0] as i32,row:t[1] as i32,col:t[2] as i32,carrier:t[3] as i32,part:t[4] as i32,mirror:t[5] as i32,at:t[6] as i32,pad:0,
                u:u.map(|z|[z.re as f32,z.im as f32])}));
            group_parts.extend(layout.groups.iter().map(|gl|model.part(k,gl.residual) as i32));
            // Its frames (its fourth-order tables: none the kernel reads yet, the model refused above).
            cluster.frames=frames.len() as i32;frames.extend(&cl.frames);
            let data=&model.tables.data[k];assert_eq!(data.len(),2*points*layout.size);
            tables.extend(data.chunks_exact(2).map(|z|[z[0] as f32,z[1] as f32]));
            for gl in &layout.groups{
                groups.push(Group{absorbed:gl.absorbed as i32,emitted:gl.emitted as i32,hermitian:gl.hermitian as i32,pairs:pairs.len() as i32,npairs:gl.pairs.len() as i32,
                    s:gl.s as i32,q:gl.q as i32,count:gl.entries.len() as i32,residual:gl.residual});
                pairs.extend(gl.pairs.iter().map(|&(k1,k)|[k1 as i32,k as i32]));
            }
            // The sparse groups' entries by the place they land at (model.rs `scatter`): per place, in the
            // groups' and pairs' order, (S's place, Q's place, kind, 0) -- the kernel's `items` -- and from
            // a pair standing for two its adjoint at the mirror place; their beats.
            let n=layout.n;let mut places:Vec<Vec<[i32;4]>>=vec![vec![];n*n];let mut beats:Vec<f64>=vec![];
            for gl in layout.groups.iter().filter(|g|g.entries.len()<n*n){
                let beat=match beats.iter().position(|&w|w==gl.residual){Some(b)=>b,None=>{beats.push(gl.residual);beats.len()-1}};
                for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                    let twice=gl.hermitian&&k1<k;let kind=(k1|k<<2|gl.absorbed<<4|gl.emitted<<7|(twice as usize)<<10|beat<<12) as i32;
                    for(j,&(a,b))in gl.entries.iter().enumerate(){
                        let at=(pi*gl.entries.len()+j) as i32;let item=[gl.s as i32+at,gl.q as i32+at,kind,0];
                        places[a*n+b].push(item);if twice{places[b*n+a].push([item[0],item[1],kind|1<<11,0]);}
                    }
                }
            }
            if beats.len()>64{return Err(beyond(format!("子空间的稀疏组有 {} 个剩余拍频",beats.len())));}
            residual_parts.extend(beats.iter().map(|&w|model.part(k,w) as i32));
            cluster.nbeats=beats.len() as i32;residuals.extend(beats);
            for place in places{item_first.push(items.len() as i32);items.extend(place);}
            item_first.push(items.len() as i32);
            // A class's photons by the row they land in and their carrier (each row's photons one run,
            // each carrier's a run within it), and where each row's run starts.
            for class in &cl.classes{
                let target=&g.clusters[class.target];
                let carriers=class.photons.iter().fold(0,|m,(ph,_)|m|1<<ph.carrier);
                let mut list:Vec<Photon>=class.photons.iter().map(|&(ph,w)|Photon{carrier:ph.carrier as i32,col:cl.position(ph.from).unwrap() as i32,to:ph.to as i32,
                    row:target.position(ph.to).unwrap() as i32,residual:w}).collect();
                list.sort_by_key(|p|(p.row,p.carrier));
                classes.push(Class{target:class.target as i32,photons:photons.len() as i32,nphotons:list.len() as i32,carriers,rows:class_rows.len() as i32,
                    direct:directs.len() as i32,ndirect:class.direct.len() as i32,pad:0,wave_number:class.wave_number});
                directs.extend(class.direct.iter().map(|&(i,to,w)|Direct{state:i as i32,to:to as i32,row:target.position(to).unwrap() as i32,pad:0,residual:w}));
                for row in 0..=target.states.len() as i32{class_rows.push(photons.len() as i32+list.iter().take_while(|p|p.row<row).count() as i32);}
                photons.extend(list);
            }
            for group in &cl.coherent{
                coherent.push(Coherent{channels:channels.len() as i32,count:group.len() as i32});
                channels.extend(group.iter().map(|&(x,w)|Channel{carrier:x.carrier as i32,col:cl.position(x.state).unwrap() as i32,residual:w}));
            }
            for c in &layout.carriers{carrier_tables.extend([c.j as i32,c.j2 as i32,c.t as i32]);}
            clusters.push(cluster);
        }
        // The exact response: H(s) = base + Σ_f s_f G_f over the focused units (ground.rs
        // `hamiltonian_with`: base = H0 + the other units' folds at s(0) − the focused units' outside shifts
        // at s(0), G_f = O_f + F_f); K = H_ee − (i/2)D. Off the diagonal in single precision, the diagonals in double.
        let mut base=g.hamiltonian.clone();
        for(u,unit)in g.units.iter().enumerate(){let s0=Complex::new(unit.focus,0.);if model.focused.contains(&u){base.add_scaled(-s0,&g.outside[u]);}else{base.add_scaled(s0,&g.folds[u]);}}
        let mut blocks:Vec<Dense>=vec![base];
        for &f in &model.focused{let mut x=g.outside[f].clone();x.add_scaled(Complex::ONE,&g.folds[f]);blocks.push(x);}
        // The units: their beams and weights, and the profile weight of those taken at the focus.
        let(mut unit_beams,mut unit_weights)=(vec![],vec![]);
        let units:Vec<Unit>=g.units.iter().enumerate().map(|(u,unit)|{let first=unit_beams.len() as i32;
            unit_beams.extend(unit.beams.iter().map(|&l|l as i32));unit_weights.extend(unit.weights.iter().map(|&w|c32(w)));
            let profile=model.profiles.iter().find(|p|p.0==u).map_or(-1.,|p|p.2 as f32);
            Unit{first,count:unit.beams.len() as i32,norm:unit.norm as f32,focus:unit.focus as f32,profile,pad:0}}).collect();
        let(mut excited,mut diagonal)=(vec![],vec![]);
        for(i,block)in blocks.iter().enumerate(){
            for(x,&a)in g.excited_states.iter().enumerate(){for(y,&b)in g.excited_states.iter().enumerate(){
                let mut z=block[(a,b)];if i==0{z+=g.decay[(x,y)]*Complex::new(0.,-0.5);}excited.push(c32(z));
            }}
            diagonal.extend(g.ground_states.iter().map(|&a|block[(a,a)].re));diagonal.extend(g.excited_states.iter().map(|&a|block[(a,a)].re));
        }
        let couplings:Vec<[f32;2]>=g.couplings.iter().flat_map(|v|v.entries.iter().map(|&z|c32(z))).collect();
        let emission:Vec<[f32;2]>=g.emission.iter().flat_map(|c|c.entries.iter().map(|&z|c32(z))).collect();
        // The settings: the trap's principal axes, their flow for dt/2 and stiffness.
        let tau=0.5*settings.dt;
        let half_flow:[[f64;3];3]=std::array::from_fn(|i|{let w=settings.frequencies[i];if w==0.{[1.,hm*tau,0.]}else{let(s,c)=(w*tau).sin_cos();[c,hm/w*s,w/hm*s]}});
        let mut field_carrier=[0;MOST_CARRIERS*MOST_RANK];let mut field_rank=[0;MOST_CARRIERS*MOST_RANK];let mut fields=0;
        for(c,carrier)in model.carriers.iter().enumerate(){for k in 0..carrier.rank{field_carrier[fields]=c as i32;field_rank[fields]=k as i32;fields+=1;}}
        // The groups of nearly degenerate excited states (ground.rs `excited_groups`) and each excited state's
        // 1/√Γ_ii (the transient's pieces, model.rs `transient`).
        let(mut levels,mut level_first)=(vec![],vec![0i32]);
        for x in &g.excited_groups{levels.extend(x.iter().map(|&i|i as i32));level_first.push(levels.len() as i32);}
        if levels.is_empty(){levels.push(0);}
        let decay_root:Vec<f32>=(0..ne).map(|i|{let d=g.decay[(i,i)].re;if d>0.{(1./d.sqrt()) as f32}else{0.}}).collect();
        fn upload<T:Copy>(stream:&Stream,buffers:&mut Vec<Buffer>,data:&[T])->Result<u64,String>{let b=Buffer::with(stream,data)?;let p=b.pointer;buffers.push(b);Ok(p)}
        let mut buffers:Vec<Buffer>=vec![];
        macro_rules! put{($v:expr)=>{upload(stream,&mut buffers,&$v)?};}
        let params=Params{
            beams:put!(beams),beam_list:put!(beam_list),carriers:put!(carriers),clusters:put!(clusters),groups:put!(groups),pairs:put!(pairs),classes:put!(classes),
            photons:put!(photons),coherent:put!(coherent),channels:put!(channels),carrier_tables:put!(carrier_tables),tables:put!(tables),
            excited:put!(excited),diagonal:put!(diagonal),couplings:put!(couplings),emission:put!(emission),class_rows:put!(class_rows),
            fourths:if fourths.is_empty(){0}else{put!(fourths)},frames:put!(frames),
            item_first:put!(item_first),items:put!(if items.is_empty(){vec![[0;4]]}else{items}),residuals:put!(if residuals.is_empty(){vec![0.]}else{residuals}),
            units:put!(units),unit_beams:put!(unit_beams),unit_weights:put!(unit_weights),
            terms:put!(if terms.is_empty(){vec![Term::default()]}else{terms}),omegas:put!(omegas),group_parts:put!(if group_parts.is_empty(){vec![0]}else{group_parts}),
            residual_parts:put!(if residual_parts.is_empty(){vec![0]}else{residual_parts}),excited_list:put!(if excited_list.is_empty(){vec![0]}else{excited_list}),
            directs:put!(if directs.is_empty(){vec![Direct::default()]}else{directs}),levels:put!(levels),level_first:put!(level_first),decay_root:put!(decay_root),
            nbeams:model.geometry.len() as i32,ncarriers:model.carriers.len() as i32,nclusters:model.layouts.len() as i32,focused:model.focused.len() as i32,
            nodes:nodes as i32,fields:fields as i32,nnear:near,doppler:model.doppler as i32,loss:settings.loss as i32,weights:0,nunits:g.units.len() as i32,
            exact_at:0,transient_at:0,nlevels:g.excited_groups.len() as i32,
            focused_unit:std::array::from_fn(|f|model.focused.get(f).map_or(0,|&u|u as i32)),stride:std::array::from_fn(|f|nodes.pow(f as u32) as i32),
            field_carrier,field_rank,ground_states:std::array::from_fn(|b|g.ground_states[b] as i32),excited_states:std::array::from_fn(|i|g.excited_states[i] as i32),
            ground_cluster:std::array::from_fn(|b|g.cluster[b] as i32),ground_position:std::array::from_fn(|b|g.clusters[g.cluster[b]].position(b).unwrap() as i32),
            hm:hm as f32,light2:(2.*g.light) as f32,allowance:settings.doppler as f32,truncation:settings.truncation as f32,
            axes32:settings.axes.map(|a|a.map(|x|x as f32)),stiffness32:settings.frequencies.map(|w|(w*w/hm) as f32),
            hm64:hm,dt:settings.dt,axes:settings.axes,omega:settings.frequencies,stiffness:settings.frequencies.map(|w|w*w/hm),half_flow,region:settings.region,
            node_phase:crate::engine::node_phase(),beats:settings.beats,horizon:settings.horizon,
        };
        let photons=g.clusters.iter().map(|c|c.classes.iter().map(|k|k.photons.len()).sum::<usize>()).max().unwrap_or(0);
        // The transient's room: every coherent group's folded amplitude on the excited states (the kernel's `transient`).
        let transient=if takes{g.clusters.iter().map(|c|c.coherent.len()*NE).max().unwrap_or(0)}else{0};
        let mut program=Program{params,photons,weights,exact,transient,_buffers:buffers};program.params.weights=(program.scratch()-weights-exact-transient) as i32;
        program.params.exact_at=(program.scratch()-exact-transient) as i32;program.params.transient_at=(program.scratch()-transient) as i32;
        Ok(program)
    }
    /// The tables' nodes per intensity axis.
    pub fn nodes(&self)->usize{self.params.nodes as usize}
    /// A jump's working space per trajectory (float2): the exact response's -- every near beam's
    /// photon tables at each cluster column --, the photons' matrices per axis and the fourth order's
    /// weights (the kernel's `Program::weights` is where they start).
    pub fn scratch(&self)->usize{self.params.nnear as usize*MAXN*3*GROUND+3*self.photons+self.weights+self.exact+self.transient}
}
