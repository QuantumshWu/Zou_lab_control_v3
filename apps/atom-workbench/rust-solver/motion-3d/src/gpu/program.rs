//! A solve's model for the kernels (kernels/motion3d.cu): its tables in single precision, the
//! descriptors the kernel walks -- each struct here the #[repr(C)] twin of the C struct of its name
//! --, the exact response's matrices and the step's settings. Built once per solve from model.rs
//! and ground.rs and kept on the device while its batches run.
use super::driver::{Buffer,Stream};
use crate::engine::Settings;
use crate::linear::Dense;
use crate::model::{Model,GROUND,MOST_BEAMS,MOST_CARRIERS,MOST_FOCUSED,MOST_RANK,MOST_STATES,RINGING_SHARE};
use crate::Complex;

/// The most states of a cluster of ground states the kernel holds (its bucket), of a cluster its exact
/// step takes (explicit excited states, beats: engine.rs # The exact step) and the most parts of its
/// components, the excited states and the most listed outputs (lib.rs `simulate`'s distributions).
pub const MAXN:usize=MOST_STATES;
pub const MAXM:usize=16;
pub const MOST_PARTS:usize=32;
pub const NE:usize=24;
pub const MOST_LISTED:usize=32;
/// A cluster's most field pairs with a first order in the kinetic shifts in the generator of a static one (the kernel's `Warp::first`; the exact step's table lies in its room)
/// and what a table holds per pair, in float2.
pub const MOST_FIRST:usize=25;
const FIRST_PAIR:usize=5;
/// A mirror item's role and where the rest of an item's fourth word starts (the kernel's ITEM_*).
const ITEM_ADJOINT:i32=1;
const ITEM_FIRST:i32=2;
const ITEM_KEPT:i32=3;
const ITEM_PAIR:i32=2;

#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Beam{
    direction:[f32;3],k:f32,wave:[f32;3],inv_w2:f32,inv_rayleigh:f32,recoil:f32,inv_pole:f32,half_pole:f32,trap:i32,
    carrier:i32,near:i32,coupled:u32,coefficients:[[f32;2];MOST_RANK],
}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Carrier{rank:i32,beams:i32,first:i32,couples:u32,detuning:f64,wave:f64}
/// The kernel's `Unit` (ground.rs `Unit`): its beams and weights' place, N, s(0) and its profile weight.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Unit{first:i32,count:i32,norm:f32,focus:f32,profile:f32,pad:i32}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Cluster{n:i32,size:i32,table:i32,h0:i32,groups:i32,ngroups:i32,classes:i32,nclasses:i32,coherent:i32,ncoherent:i32,carriers:i32,fourth:i32,frames:i32,
    positions:i32,beats:i32,nbeats:i32,n0:i32,exact:i32,parts:i32,omegas:i32,terms:i32,nterms:i32,emission:i32,excited:i32,group_parts:i32,residual_parts:i32,first_fields:i32,nfirst:i32,states:[i32;MAXM],
    shifts:u64,mixed:u32,pad:i32}
/// The kernel's `NearUnit` (ground.rs `NearUnit`): its carrier, its place among the focused units (−1: at its
/// focus intensity), N and s(0).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct NearUnit{carrier:i32,focused:i32,norm:f32,focus:f32}
/// The kernel's `Term`: a term of a cluster's explicit states (model.rs `exact_terms`): a kept drive (0), a
/// decay cross term (1), a decay (2) or a dressing's drive (3), its row, column, carrier, part, its
/// mirror's part, its tables' place (a cross term's X_k, and at `at2` its X2_k), a drive's U_k.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Term{kind:i32,row:i32,col:i32,carrier:i32,part:i32,mirror:i32,at:i32,at2:i32,u:[[f32;2];MOST_RANK]}
/// The kernel's `Direct`: a class's photon from an explicit state (ground.rs `Class::direct`): its
/// state's place in the cluster, the ground state it lands in and its row in the target, its residual.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Direct{state:i32,to:i32,row:i32,pad:i32,residual:f64}
/// The kernel's `Fourth`: where a cluster's fourth-order slabs lie (model.rs `FourthLayout`), and the carriers that bend its photons' state.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Fourth{quartic:i32,nquartic:i32,slow:i32,nslow:i32,jumps:i32,njumps:i32,factors:i32,nfactors:i32,bends:u32,pad:[i32;3]}
/// The kernel's `Quartic` (model.rs `QuarticSlab`).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Quartic{fields:[i32;4],at:i32,parts:i32,nu:f64}
/// The kernel's `Slow` (model.rs `SlowSlab`).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Slow{c1:i32,c2:i32,rr:i32,yw:i32,nu:f64}
/// The kernel's `JumpTable` (model.rs `JumpSlab`).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct JumpTable{photon:i32,inner:i32,at:i32,diagonal:i32,pairs:i32,npairs:i32}
/// The kernel's `Factor` (model.rs `FourthLayout::p`).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Factor{c2:i32,c4:i32,at:i32,pad:i32}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Group{absorbed:i32,emitted:i32,hermitian:i32,pairs:i32,npairs:i32,s:i32,q:i32,count:i32,first:i32,pad:i32,residual:f64}
/// The kernel's `FirstField`: a field pair of a group with a first order: its transfer's place, its fields and carriers, its beat.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct FirstField{transfer:i32,kind:i32,residual:f64}
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Class{target:i32,photons:i32,nphotons:i32,carriers:i32,rows:i32,direct:i32,ndirect:i32,beating:i32,wave_number:f64}
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
    excited:u64,diagonal:u64,couplings:u64,emission:u64,class_rows:u64,fourths:u64,quartics:u64,quartic_parts:u64,slows:u64,jump_tables:u64,jump_pairs:u64,factors:u64,frames:u64,item_first:u64,items:u64,residuals:u64,
    units:u64,unit_beams:u64,unit_weights:u64,
    terms:u64,omegas:u64,group_parts:u64,residual_parts:u64,excited_list:u64,directs:u64,levels:u64,level_first:u64,level_width:u64,
    kept:u64,decay:u64,near_units:u64,unit_couplings:u64,vias:u64,via_first:u64,ground_block:u64,term_floor:u64,powers:u64,first_fields:u64,
    nbeams:i32,pub(crate) ncarriers:i32,nclusters:i32,focused:i32,nodes:i32,fields:i32,pub(crate) nnear:i32,doppler:i32,loss:i32,weights:i32,nunits:i32,
    exact_at:i32,transient_at:i32,nlevels:i32,rows_at:i32,between_at:i32,classes_at:i32,
    focused_unit:[i32;MOST_FOCUSED],stride:[i32;MOST_FOCUSED],
    field_carrier:[i32;MOST_CARRIERS*MOST_RANK],field_rank:[i32;MOST_CARRIERS*MOST_RANK],
    ground_states:[i32;GROUND],excited_states:[i32;NE],ground_cluster:[i32;GROUND],ground_position:[i32;GROUND],
    near_first:[i32;MOST_CARRIERS+1],explicit_of:[u32;GROUND],partners:[u32;NE],near_focused:u32,
    hm:f32,light2:f32,allowance:f32,truncation:f32,ringing:f32,recoil_spread:f32,
    axes32:[[f32;3];3],stiffness32:[f32;3],
    hm64:f64,dt:f64,
    axes:[[f64;3];3],omega:[f64;3],stiffness:[f64;3],
    half_flow:[[f64;3];3],
    region:f64,
    lattice_axis:[f64;3],lattice_k:f64,lattice_inv_rayleigh:f64,lattice_phase:f64,
    node_phase:f64,grid_phase:f64,beats:f64,horizon:f64,
}

/// The kernel's `State`: a trajectory between launches.
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct State{
    pub(crate) q:[f64;3],pub(crate) p:[f64;3],pub(crate) time:f64,pub(crate) logn:f64,pub(crate) threshold:f64,pub(crate) log_threshold:f64,pub(crate) prefix:[f64;15],
    pub(crate) q0:[f64;3],pub(crate) p0:[f64;3],pub(crate) twin_logn:f64,pub(crate) twin_threshold:f64,pub(crate) twin_log_threshold:f64,
    pub(crate) psi:[[f32;2];MAXM],pub(crate) twin:[[f32;2];MAXN],pub(crate) cluster:i32,pub(crate) jumps:i32,pub(crate) death:i32,
    pub(crate) index:u32,pub(crate) twin_cluster:i32,pub(crate) twin_jumps:i32,pub(crate) beats:f32,
    pub(crate) interval:i32,pub(crate) step:i32,
}
/// The kernel's `Held`: the held twin's device data (all zero: no twin).
#[repr(C)]#[derive(Clone,Copy,Default,Debug)]
pub struct Held{pub(crate) u:u64,pub(crate) h:u64,pub(crate) l:u64,pub(crate) readout:u64,pub(crate) blocks:u64,pub(crate) jumps:u64}
/// The kernel's `Run`: a launch's wave and output intervals.
#[repr(C)]#[derive(Clone,Copy,Debug)]
pub struct Run{
    pub(crate) states:u64,pub(crate) rings:u64,pub(crate) samples:u64,pub(crate) sums:u64,pub(crate) scratch:u64,pub(crate) tasks:u64,pub(crate) profile:u64,
    pub(crate) records:u64,pub(crate) harmonic:u64,pub(crate) costs:u64,pub(crate) held:Held,pub(crate) seed:u64,pub(crate) budget:i64,
    pub(crate) outputs:i32,pub(crate) steps:i32,pub(crate) reach:i32,pub(crate) begin:i32,pub(crate) end:i32,
    pub(crate) half:[i32;3],pub(crate) ring:[i32;3],pub(crate) ring_at:[i32;3],
    pub(crate) ring_stride:i32,pub(crate) scratch_stride:i32,pub(crate) nlisted:i32,pub(crate) pad:i32,
    pub(crate) listed:[i32;MOST_LISTED],
}

impl Run{
    /// A launch of no wave (a check's entry, `Gpu::jump_probe`): every buffer null, the seed zero.
    pub(crate) fn empty()->Run{Run{states:0,rings:0,samples:0,sums:0,scratch:0,tasks:0,profile:0,records:0,harmonic:0,costs:0,held:Held::default(),seed:0,budget:0,
        outputs:0,steps:0,reach:0,begin:0,end:0,half:[0;3],ring:[0;3],ring_at:[0;3],ring_stride:0,scratch_stride:0,nlisted:0,pad:0,listed:[0;MOST_LISTED]}}
}

/// The sizes, offsets and constants the kernel's `layout` reports, for the check that the twins and
/// the constants they share match.
pub fn layout()->[u32;28]{
    use std::mem::{offset_of,size_of};
    use super::{HARMONIC,MOST_INTERVALS,MOTION,RECORD,SLOT_RECORD,TWIN,WARPS};
    [size_of::<Beam>(),size_of::<Carrier>(),size_of::<Cluster>(),size_of::<Group>(),size_of::<Class>(),size_of::<Photon>(),size_of::<Coherent>(),size_of::<Channel>(),
        size_of::<Params>(),size_of::<State>(),size_of::<Run>(),offset_of!(Params,hm),offset_of!(Params,half_flow),offset_of!(Run,listed),offset_of!(State,psi),offset_of!(State,index),
        MOST_INTERVALS,SLOT_RECORD,RECORD,WARPS,MAXN,TWIN,MOTION,HARMONIC,size_of::<Fourth>(),size_of::<NearUnit>(),size_of::<FirstField>(),MOST_FIRST].map(|x|x as u32)
}

/// The model on the device: the kernel's parameters, the buffers they point into, the most
/// photons of a cluster's jumps and the rest of a trajectory's scratch (`scratch`).
pub struct Program{pub params:Params,photons:usize,classes:usize,weights:usize,pub(crate) exact:usize,transient:usize,rows:usize,between:usize,_buffers:Vec<Buffer>}

fn c32(z:Complex)->[f32;2]{[z.re as f32,z.im as f32]}

impl Program{
    /// The program of a model and its step's settings, uploaded; Err with the reason (Chinese, as the
    /// app shows it) when the model is beyond what the kernel holds.
    pub fn new(stream:&Stream,model:&Model,settings:&Settings)->Result<Program,String>{
        let g=&model.ground;let(ng,ne)=(g.ground_states.len(),g.excited_states.len());
        let beyond=|what:String|->String{format!("模型超出 GPU 内核的范围（{what}）")};
        if ng!=GROUND||ne!=NE{return Err(beyond(format!("{ng} 个基态、{ne} 个激发态")));}
        if settings.free!=[true;3]{return Err(beyond("运动只沿部分坐标轴".into()));}
        // The exact step's clusters (explicit excited states, beats not all left out: engine.rs `quiet_clusters`) in its bucket; where one takes it, any cluster a jump lands it in in its parts.
        let quiet=crate::engine::quiet_clusters(model,settings);let exact_of=|k:usize|model.mixed(k)||(model.beating(k)&&quiet[k].is_none());let takes=(0..model.clusters()).any(exact_of);
        if let Some(k)=(0..model.clusters()).find(|&k|exact_of(k)&&model.dimension(k)>MAXM){
            return Err(beyond(format!("子空间 {} 个态（{} 个激发态显式）超出精确步的 {MAXM} 个",model.dimension(k),model.dimension(k)-model.ground_dimension(k))));}
        if let Some(k)=(0..model.clusters()).find(|&k|takes&&1+model.beats(k).0.len()>MOST_PARTS){return Err(beyond(format!("子空间 {} 的 {} 个残余拍频",k,model.beats(k).0.len()/2)));}
        if model.geometry.len()>MOST_BEAMS||model.carriers.len()>MOST_CARRIERS||model.focused.len()>MOST_FOCUSED{return Err(beyond("光束或强度维度太多".into()));}
        if let Some(c)=g.clusters.iter().find(|c|c.states.len()>MAXN){return Err(beyond(format!("子空间 {} 个态",c.states.len())));}
        let hm=model.hbar_over_mass;
        // The beams: each one's carrier and place among them, and the near beams' order.
        let mut owner=vec![None;model.geometry.len()];
        for(c,carrier)in model.carriers.iter().enumerate(){for(j,&l)in carrier.beams.iter().enumerate(){owner[l]=Some((c,j));}}
        let mut near=0;
        let beams:Vec<Beam>=model.geometry.iter().enumerate().map(|(l,geometry)|{
            let k=geometry.wave_number;let(inv_w2,inv_rayleigh)=match geometry.waist{Some(w)=>(1./(w*w),2./(k*w*w)),None=>(0.,0.)};
            let mut beam=Beam{direction:geometry.direction.map(|x|x as f32),k:k as f32,wave:geometry.wave_vector().map(|x|x as f32),inv_w2:inv_w2 as f32,
                inv_rayleigh:inv_rayleigh as f32,recoil:(0.5*hm*k*k) as f32,inv_pole:0.,half_pole:0.,trap:model.trap_beams.contains(&l) as i32,carrier:-1,near:-1,coupled:0,coefficients:[[0.;2];MOST_RANK]};
            if let Some((c,j))=owner[l]{
                let pole=g.poles[c];beam.inv_pole=if pole>0.&&pole.is_finite(){(1./pole) as f32}else{0.};beam.half_pole=(0.5*pole) as f32;
                beam.carrier=c as i32;beam.near=near;near+=1;
                for b in (0..ng).filter(|&b|g.couples(c,b)){if (0..ne).any(|i|g.couplings[l][(i,b)]!=Complex::ZERO){beam.coupled|=1<<b;}}
                for k in 0..model.carriers[c].rank{beam.coefficients[k]=c32(model.carriers[c].coefficients[j][k]);}
            }
            beam}).collect();
        let mut beam_list=vec![];
        let carriers:Vec<Carrier>=model.carriers.iter().enumerate().map(|(c,carrier)|{
            let first=beam_list.len() as i32;beam_list.extend(carrier.beams.iter().map(|&l|l as i32));
            Carrier{rank:carrier.rank as i32,beams:carrier.beams.len() as i32,first,couples:(0..ng).filter(|&b|g.couples(c,b)).fold(0,|m,b|m|1<<b),detuning:g.carriers[c].detuning,wave:carrier.wave_number}}).collect();
        // The clusters: their tables one after another, their groups, classes and coherent groups.
        let nodes=model.tables.nodes;let points=nodes.pow(model.focused.len() as u32);
        let(mut clusters,mut groups,mut pairs,mut classes,mut photons,mut class_rows,mut coherent,mut channels,mut carrier_tables,mut tables)=
            (vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![],vec![]);
        let(mut fourths,mut frames,mut weights):(Vec<Fourth>,Vec<f64>,usize)=(vec![],vec![],0);
        let(mut quartics,mut quartic_parts,mut slows,mut jump_tables,mut jump_pairs,mut factors):(Vec<Quartic>,Vec<i32>,Vec<Slow>,Vec<JumpTable>,Vec<i32>,Vec<Factor>)=(vec![],vec![],vec![],vec![],vec![],vec![]);
        let(mut item_first,mut items,mut residuals):(Vec<i32>,Vec<[i32;4]>,Vec<f64>)=(vec![],vec![],vec![]);
        let(mut terms,mut omegas,mut group_parts,mut residual_parts,mut excited_list,mut directs):(Vec<Term>,Vec<f64>,Vec<i32>,Vec<i32>,Vec<i32>,Vec<Direct>)=(vec![],vec![],vec![],vec![],vec![],vec![]);
        let mut first_fields:Vec<FirstField>=vec![];
        // The exact step's room in a trajectory's scratch: per cluster its parts' H and G, their working
        // room and its norms (the kernel's `exact_segment`) and its first orders' table, the largest of every cluster's where one takes it.
        let mut exact=0;
        // The carriers whose static block on a cluster mixes its responses (ground.rs `Ground::unmixed`, the kernel's
        // `carrier_responses`), bits; and the near light's shifts ΔE_a^d a cluster's exact responses take (`exact_responses`,
        // ground.rs `Ground::dress_light`), bit d·NG + a: from each of its channels (c, b) the other near carriers' on b and
        // every near carrier's on the intermediates a of its far channels; a mixed block's, the other carriers' on every
        // state of the cluster; none where the model is undressed.
        let nc=g.carriers.len();
        let mixed=|k:usize,states:&[usize]|->u32{(0..nc).filter(|&c|g.dressing&&states.iter().any(|&b|g.couples(c,b))&&!g.unmixed.get(c).is_some_and(|x|x[k])).fold(0,|m,c|m|1<<c)};
        let shifts=|k:usize,states:&[usize]|->u64{
            let mut bits=0u64;if !g.dressing{return bits;}
            for &b in states{for c in (0..nc).filter(|&c|g.couples(c,b)){
                for d in (0..nc).filter(|&d|d!=c&&g.couples(d,b)){bits|=1<<(d*GROUND+b);}
                for via in &g.vias[c][b]{for d in (0..nc).filter(|&d|g.couples(d,via.state)){bits|=1<<(d*GROUND+via.state);}}
            }}
            for c in (0..nc).filter(|&c|(mixed(k,states)>>c)&1==1){for d in (0..nc).filter(|&d|d!=c){for &b in states{if g.couples(d,b){bits|=1<<(d*GROUND+b);}}}}
            bits};
        for(k,(layout,cl))in model.layouts.iter().zip(&g.clusters).enumerate(){
            let parts=1+model.beats(k).0.len();let(n,n0)=(layout.n,layout.n0);
            let(list,u,emission)=model.exact_terms(k);
            let mut cluster=Cluster{n:n as i32,size:layout.size as i32,table:(tables.len()) as i32,h0:layout.h0 as i32,groups:groups.len() as i32,ngroups:layout.groups.len() as i32,
                classes:classes.len() as i32,nclasses:cl.classes.len() as i32,coherent:coherent.len() as i32,ncoherent:cl.coherent.len() as i32,carriers:carrier_tables.len() as i32,
                fourth:-1,frames:0,positions:item_first.len() as i32,beats:residuals.len() as i32,nbeats:0,n0:n0 as i32,exact:exact_of(k) as i32,
                parts:parts as i32,omegas:omegas.len() as i32,terms:terms.len() as i32,nterms:list.len() as i32,emission:emission as i32,excited:excited_list.len() as i32,
                group_parts:group_parts.len() as i32,residual_parts:residual_parts.len() as i32,first_fields:first_fields.len() as i32,nfirst:0,states:[0;MAXM],shifts:shifts(k,&cl.states),mixed:mixed(k,&cl.states),pad:0};
            for(x,&b)in cl.states.iter().enumerate(){cluster.states[x]=b as i32;}
            // Its beats, its explicit states and its terms (model.rs `exact_terms`).
            omegas.extend(model.beats(k).0);excited_list.extend(cl.excited.iter().map(|&e|e as i32));
            terms.extend(list.iter().zip(&u).map(|(t,u)|Term{kind:t[0] as i32,row:t[1] as i32,col:t[2] as i32,carrier:t[3] as i32,part:t[4] as i32,mirror:t[5] as i32,at:t[6] as i32,at2:t[7] as i32,
                u:u.map(|z|[z.re as f32,z.im as f32])}));
            group_parts.extend(layout.groups.iter().map(|gl|model.part(k,gl.residual) as i32));
            // Its frames and its fourth-order slabs.
            cluster.frames=frames.len() as i32;frames.extend(&cl.frames);
            if let Some(f)=&layout.fourth{
                let code=|i:usize|{let(c,k)=f.fields[i];(c*MOST_RANK+k) as i32};
                cluster.fourth=fourths.len() as i32;
                fourths.push(Fourth{quartic:quartics.len() as i32,nquartic:f.quartic.len() as i32,slow:slows.len() as i32,nslow:f.slow.len() as i32,jumps:jump_tables.len() as i32,njumps:f.jumps.len() as i32,
                    factors:factors.len() as i32,nfactors:f.p.len() as i32,bends:f.carriers.iter().fold(0,|m,&c|m|1<<c),pad:[0;3]});
                for q in &f.quartic{
                    quartics.push(Quartic{fields:[code(q.conj.0),code(q.conj.1),code(q.plain.0),code(q.plain.1)],at:q.at as i32,parts:quartic_parts.len() as i32,nu:q.nu});
                    quartic_parts.extend((0..n*n).map(|i|if q.keep[i]{q.part[i] as i32}else{-1}));
                }
                slows.extend(f.slow.iter().map(|x|Slow{c1:x.carriers.0 as i32,c2:x.carriers.1 as i32,rr:x.rr as i32,yw:x.yw as i32,nu:x.nu}));
                factors.extend(f.p.iter().map(|&(c2,c4,_,at)|Factor{c2:c2 as i32,c4:c4 as i32,at:at as i32,pad:0}));
                for j in &f.jumps{
                    jump_tables.push(JumpTable{photon:j.photon as i32,inner:j.inner as i32,at:j.at as i32,diagonal:j.diagonal as i32,pairs:jump_pairs.len() as i32,npairs:j.pairs.len() as i32});
                    jump_pairs.extend(j.pairs.iter().map(|&i|i as i32));
                }
                // The jump tables' weights and the middle factors' forms (the kernel's `fourth_weights`).
                weights=weights.max((f.jumps.len()+f.p.len())*GROUND*n);
            }
            let data=&model.tables.data[k];assert_eq!(data.len(),2*points*layout.size);
            tables.extend(data.chunks_exact(2).map(|z|[z[0] as f32,z[1] as f32]));
            // Per group 1 + its first pair's place among the cluster's field pairs with a first order (the kernel's `first_table`), its other pairs after it; 0 without one.
            let first_of:Vec<usize>=layout.groups.iter().map(|gl|{
                if gl.first.is_empty(){return 0;}
                let place=1+first_fields.len()-cluster.first_fields as usize;
                first_fields.extend(gl.pairs.iter().map(|&(k1,k)|FirstField{transfer:gl.transfer as i32,kind:(k1|k<<2|gl.absorbed<<4|gl.emitted<<7|(gl.hermitian as usize)<<10) as i32,residual:gl.residual}));place}).collect();
            cluster.nfirst=first_fields.len() as i32-cluster.first_fields;
            if cluster.exact==0&&cluster.nfirst as usize>MOST_FIRST{return Err(beyond(format!("子空间的 {} 个光场对带一阶多普勒项",cluster.nfirst)));}
            if takes{exact=exact.max((5*parts+4)*n*n+4*parts+FIRST_PAIR*cluster.nfirst as usize);}
            for(gl,&first)in layout.groups.iter().zip(&first_of){
                groups.push(Group{absorbed:gl.absorbed as i32,emitted:gl.emitted as i32,hermitian:gl.hermitian as i32,pairs:pairs.len() as i32,npairs:gl.pairs.len() as i32,
                    s:gl.s as i32,q:gl.q as i32,count:gl.entries.len() as i32,first:first as i32,pad:0,residual:gl.residual});
                pairs.extend(gl.pairs.iter().map(|&(k1,k)|[k1 as i32,k as i32]));
            }
            // The kernel's `items`, per place three lists in the groups' and pairs' order, each item (S's place, Q's place, kind, role and the pair's place):
            // 0 the sparse groups' entries that land there (model.rs `scatter`); 1 what the first order's transfer leaves of any group there, its ΔW^S and ΔW^Q in the two places;
            // 2 at their mirrors: of a pair standing for two its adjoint, of a pair between two carriers with a first order that first order's adjoint, of 1 its adjoint. Their beats.
            let n=layout.n;let mut places:Vec<[Vec<[i32;4]>;3]>=vec![[vec![],vec![],vec![]];n*n];let mut beats:Vec<f64>=vec![];
            for(gl,&first)in layout.groups.iter().zip(&first_of).filter(|(g,_)|g.entries.len()<n*n||!g.kept.is_empty()){
                let beat=match beats.iter().position(|&w|w==gl.residual){Some(b)=>b,None=>{beats.push(gl.residual);beats.len()-1}};
                let kind=|k1:usize,k:usize,twice:bool|(k1|k<<2|gl.absorbed<<4|gl.emitted<<7|(twice as usize)<<10|beat<<12) as i32;
                if gl.entries.len()<n*n{for(pi,&(k1,k))in gl.pairs.iter().enumerate(){
                    let twice=gl.hermitian&&k1<k;let kind=kind(k1,k,twice);let pair=if first>0{((first+pi) as i32)<<ITEM_PAIR}else{0};
                    for(j,&(a,b))in gl.entries.iter().enumerate(){
                        let at=(pi*gl.entries.len()+j) as i32;let(s,q)=(gl.s as i32+at,gl.q as i32+at);
                        places[a*n+b][0].push([s,q,kind,pair]);
                        if twice{places[b*n+a][2].push([s,q,kind,ITEM_ADJOINT|pair]);}else if first>0&&!gl.hermitian{places[b*n+a][2].push([s,q,kind,ITEM_FIRST|pair]);}
                    }
                }}
                for kept in &gl.kept{
                    let kind=kind(kept.pair.0,kept.pair.1,false);let count=kept.entries.len();
                    for(j,&(x,y))in kept.entries.iter().enumerate(){
                        let(s,q)=((kept.at+j) as i32,(kept.at+count+j) as i32);
                        places[x*n+y][1].push([s,q,kind,0]);places[y*n+x][2].push([s,q,kind,ITEM_KEPT]);
                    }
                }
            }
            if beats.len()>64{return Err(beyond(format!("子空间的稀疏组有 {} 个剩余拍频",beats.len())));}
            residual_parts.extend(beats.iter().map(|&w|model.part(k,w) as i32));
            cluster.nbeats=beats.len() as i32;residuals.extend(beats);
            for list in places.into_iter().flatten(){item_first.push(items.len() as i32);items.extend(list);}
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
                    direct:directs.len() as i32,ndirect:class.direct.len() as i32,beating:list.iter().any(|p|p.residual!=0.) as i32,wave_number:class.wave_number});
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
        let couplings:Vec<[f32;2]>=g.folded.iter().flat_map(|v|v.entries.iter().map(|&z|c32(z))).collect();
        // The exact response's dressing (the kernel's `exact_responses`): every beam's kept drives V_l − V_l^f and D;
        // the near units by carrier (in the units' order within one) with their folded fields W^f, NG columns of NE;
        // every channel's far channels -- none where the model is undressed.
        let kept:Vec<[f32;2]>=g.couplings.iter().zip(&g.folded).flat_map(|(v,f)|v.entries.iter().zip(&f.entries).map(|(&a,&b)|c32(a-b))).collect();
        let decay:Vec<[f32;2]>=g.decay.entries.iter().map(|&z|c32(z)).collect();
        let mut order:Vec<(usize,&crate::ground::NearUnit)>=g.near_units.iter().enumerate().filter_map(|(u,x)|x.as_ref().filter(|_|g.dressing).map(|x|(u,x))).collect();
        order.sort_by_key(|(_,x)|x.carrier);
        let near_first:[i32;MOST_CARRIERS+1]=std::array::from_fn(|d|order.iter().filter(|(_,x)|x.carrier<d).count() as i32);
        let near_units:Vec<NearUnit>=order.iter().map(|&(u,x)|NearUnit{carrier:x.carrier as i32,focused:model.focused.iter().position(|&f|f==u).map_or(-1,|f|f as i32),
            norm:g.units[u].norm as f32,focus:g.units[u].focus as f32}).collect();
        let unit_couplings:Vec<[f32;2]>=order.iter().flat_map(|(_,x)|(0..ng).flat_map(move|a|(0..ne).map(move|i|c32(x.coupling[(i,a)])))).collect();
        let(mut vias,mut via_first):(Vec<[i32;2]>,Vec<i32>)=(vec![],vec![]);
        for c in 0..nc{for b in 0..ng{via_first.push(vias.len() as i32);if g.dressing{vias.extend(g.vias[c][b].iter().map(|v|[v.carrier as i32,v.state as i32]));}}}
        via_first.push(vias.len() as i32);
        let emission:Vec<[f32;2]>=g.emission.iter().flat_map(|c|c.entries.iter().map(|&z|c32(z))).collect();
        // H's ground block, its base and per focused beam (the block's off-diagonal part, the kernel's `carrier_responses`), and per
        // channel the weight in an eigen-energy below which a state's term is left out: ground.rs `Mode::skipped`'s rule
        // at single precision's round-off of the generator, ‖V^f[:,b]‖ w/γ ≤ ε l/(3 max_b ‖V^f[:,b]‖).
        let ground_block:Vec<[f32;2]>=blocks.iter().flat_map(|block|g.ground_states.iter().flat_map(move|&a|g.ground_states.iter().map(move|&b|c32(block[(a,b)])))).collect();
        let term_floor:Vec<f32>=(0..nc).flat_map(|c|(0..ng).map(move|b|(c,b))).map(|(c,b)|{
            let strength=|s:usize|g.carriers[c].beams.iter().map(|&l|(0..ne).map(|i|g.folded[l][(i,s)].norm_sqr()).sum::<f64>().sqrt()).sum::<f64>();
            let k=g.cluster[b];let most=g.clusters[k].states.iter().map(|&s|strength(s)).fold(0.,f64::max);let floor=g.floors.get(k).copied().unwrap_or(0.);
            if strength(b)>0.{(f32::EPSILON as f64*g.generator_scale*floor/(3.*most*strength(b))) as f32}else{0.}}).collect();
        // Per channel (d, a) its near units at their focus intensity summed (the kernel's `self_energy`), lane-major.
        let powers:Vec<[f32;2]>=(0..nc).flat_map(|d|(0..ng).map(move|a|(d,a))).flat_map(|(d,a)|{let mut m=vec![Complex::ZERO;ne*ne];
            for &(u,x)in order.iter().filter(|(u,x)|x.carrier==d&&!model.focused.contains(u)){let scale=g.units[u].focus*g.units[u].norm;
                for j in 0..ne{for i in 0..ne{m[j*ne+i]+=x.coupling[(i,a)]*x.coupling[(j,a)].conj()*scale;}}}
            m.into_iter().map(c32)}).collect();
        let near_focused=order.iter().filter(|(u,_)|model.focused.contains(u)).fold(0u32,|m,(_,x)|m|1<<x.carrier);
        // The settings: the trap's principal axes, their flow for dt/2 and stiffness.
        let tau=0.5*settings.dt;
        let half_flow:[[f64;3];3]=std::array::from_fn(|i|{let w=settings.frequencies[i];if w==0.{[1.,hm*tau,0.]}else{let(s,c)=(w*tau).sin_cos();[c,hm/w*s,w/hm*s]}});
        let mut field_carrier=[0;MOST_CARRIERS*MOST_RANK];let mut field_rank=[0;MOST_CARRIERS*MOST_RANK];let mut fields=0;
        for(c,carrier)in model.carriers.iter().enumerate(){for k in 0..carrier.rank{field_carrier[fields]=c as i32;field_rank[fields]=k as i32;fields+=1;}}
        // The excited manifolds (ground.rs `excited_groups`) and each one's width, its states' mean decay (the ringing's photon, model.rs `ringing`).
        let(mut levels,mut level_first)=(vec![],vec![0i32]);
        for x in &g.excited_groups{levels.extend(x.iter().map(|&i|i as i32));level_first.push(levels.len() as i32);}
        let level_width:Vec<f64>=g.excited_groups.iter().map(|x|x.iter().map(|&i|g.decay[(i,i)].re).sum::<f64>()/x.len() as f64).collect();
        fn upload<T:Copy>(stream:&Stream,buffers:&mut Vec<Buffer>,data:&[T])->Result<u64,String>{let b=Buffer::with(stream,data)?;let p=b.pointer;buffers.push(b);Ok(p)}
        let mut buffers:Vec<Buffer>=vec![];
        macro_rules! put{($v:expr)=>{upload(stream,&mut buffers,&$v)?};}
        let params=Params{
            beams:put!(beams),beam_list:put!(beam_list),carriers:put!(carriers),clusters:put!(clusters),groups:put!(groups),pairs:put!(pairs),classes:put!(classes),
            photons:put!(photons),coherent:put!(coherent),channels:put!(channels),carrier_tables:put!(carrier_tables),tables:put!(tables),
            excited:put!(excited),diagonal:put!(diagonal),couplings:put!(couplings),emission:put!(emission),class_rows:put!(class_rows),
            fourths:if fourths.is_empty(){0}else{put!(fourths)},quartics:put!(quartics),quartic_parts:put!(quartic_parts),slows:put!(slows),jump_tables:put!(jump_tables),jump_pairs:put!(jump_pairs),factors:put!(factors),frames:put!(frames),
            item_first:put!(item_first),items:put!(if items.is_empty(){vec![[0;4]]}else{items}),residuals:put!(if residuals.is_empty(){vec![0.]}else{residuals}),
            units:put!(units),unit_beams:put!(unit_beams),unit_weights:put!(unit_weights),
            terms:put!(if terms.is_empty(){vec![Term::default()]}else{terms}),omegas:put!(omegas),group_parts:put!(if group_parts.is_empty(){vec![0]}else{group_parts}),
            residual_parts:put!(if residual_parts.is_empty(){vec![0]}else{residual_parts}),excited_list:put!(if excited_list.is_empty(){vec![0]}else{excited_list}),
            directs:put!(if directs.is_empty(){vec![Direct::default()]}else{directs}),levels:put!(levels),level_first:put!(level_first),level_width:put!(level_width),
            kept:put!(kept),decay:put!(decay),near_units:put!(if near_units.is_empty(){vec![NearUnit::default()]}else{near_units}),
            unit_couplings:put!(if unit_couplings.is_empty(){vec![[0.;2]]}else{unit_couplings}),vias:put!(if vias.is_empty(){vec![[0;2]]}else{vias}),via_first:put!(via_first),
            ground_block:put!(ground_block),term_floor:put!(term_floor),powers:put!(powers),first_fields:put!(if first_fields.is_empty(){vec![FirstField::default()]}else{first_fields}),
            nbeams:model.geometry.len() as i32,ncarriers:model.carriers.len() as i32,nclusters:model.layouts.len() as i32,focused:model.focused.len() as i32,
            nodes:nodes as i32,fields:fields as i32,nnear:near,doppler:model.doppler as i32,loss:settings.loss as i32,weights:0,nunits:g.units.len() as i32,
            exact_at:0,transient_at:0,nlevels:g.excited_groups.len() as i32,rows_at:0,between_at:0,classes_at:0,
            focused_unit:std::array::from_fn(|f|model.focused.get(f).map_or(0,|&u|u as i32)),stride:std::array::from_fn(|f|nodes.pow(f as u32) as i32),
            field_carrier,field_rank,ground_states:std::array::from_fn(|b|g.ground_states[b] as i32),excited_states:std::array::from_fn(|i|g.excited_states[i] as i32),
            ground_cluster:std::array::from_fn(|b|g.cluster[b] as i32),ground_position:std::array::from_fn(|b|g.clusters[g.cluster[b]].position(b).unwrap() as i32),
            near_first,explicit_of:std::array::from_fn(|b|g.explicit_of.get(b).map_or(0,|p|p.iter().enumerate().fold(0,|m,(e,&x)|m|(x as u32)<<e))),
            partners:std::array::from_fn(|e|g.dressed.get(e).map_or(0,|f|f.iter().fold(0,|m,&x|m|1u32<<x))),near_focused,
            hm:hm as f32,light2:(2.*g.light) as f32,allowance:settings.doppler as f32,truncation:settings.truncation as f32,ringing:RINGING_SHARE as f32,recoil_spread:crate::engine::RECOIL_SPREAD as f32,
            axes32:settings.axes.map(|a|a.map(|x|x as f32)),stiffness32:settings.frequencies.map(|w|(w*w/hm) as f32),
            hm64:hm,dt:settings.dt,axes:settings.axes,omega:settings.frequencies,stiffness:settings.frequencies.map(|w|w*w/hm),half_flow,region:settings.region,
            lattice_axis:settings.lattice.map_or([0.;3],|l|std::array::from_fn(|i|(0..3).map(|a|settings.axes[i][a]*l.axis[a]).sum())),lattice_k:settings.lattice.map_or(0.,|l|l.wave_number),lattice_inv_rayleigh:settings.lattice.map_or(0.,|l|1./l.rayleigh),lattice_phase:settings.lattice.map_or(0.,|l|l.phase),
            node_phase:crate::engine::node_phase(),grid_phase:settings.node_phase,beats:settings.beats,horizon:settings.horizon,
        };
        let photons=g.clusters.iter().map(|c|c.classes.iter().map(|k|k.photons.len()).sum::<usize>()).max().unwrap_or(0);
        // A jump's classes' weights, a float each (the kernel's `jump`): as many as the model's largest number of them, which a field splits as far as the photons' frequencies part.
        let classes=g.clusters.iter().map(|c|c.classes.len()).max().unwrap_or(0).div_ceil(2);
        // The transient's room (the kernel's `transient`, `ringing_photon`): every coherent group's folded amplitude on the excited states, and for a photon on the largest manifold's
        // m states its operators (3 axes × ng × m), per carrier and rank its channels' amplitudes (m), H's off-diagonal part or propagator and its eigenvectors (m × m), its diagonal and its phases (m).
        let most=g.excited_groups.iter().map(|x|x.len()).max().unwrap_or(0);
        let transient=g.clusters.iter().map(|c|c.coherent.len()*NE).max().unwrap_or(0)+(3*GROUND+nc*MOST_RANK+2*most+2)*most;
        // The exact responses' rows: a channel's K + Σ − z, lane-major (the kernel's `exact_responses`).
        let rows=if near>0{NE*32}else{0};
        // The mixed blocks' light shifts between a cluster's places, per carrier (the kernel's `light_shifts`).
        let between=nc*MAXN*MAXN;
        let mut program=Program{params,photons,classes,weights,exact,transient,rows,between,_buffers:buffers};let total=program.scratch();program.params.weights=(total-weights-exact-transient-rows-between) as i32;
        program.params.classes_at=program.params.weights-classes as i32;
        program.params.exact_at=(total-exact-transient-rows-between) as i32;program.params.transient_at=(total-transient-rows-between) as i32;
        program.params.rows_at=(total-rows-between) as i32;program.params.between_at=(total-between) as i32;
        Ok(program)
    }
    /// The tables' nodes per intensity axis.
    pub fn nodes(&self)->usize{self.params.nodes as usize}
    /// A jump's working space per trajectory (float2): the exact response's -- every near beam's
    /// photon tables at each cluster column --, the photons' matrices per axis, the classes' weights and the fourth order's
    /// (the kernel's `Program::classes_at` and `Program::weights` are where they start), the exact step's and the transient's
    /// room, and the exact responses' channel rows and mixed blocks' shifts (`Program::rows_at`, `Program::between_at`).
    pub fn scratch(&self)->usize{self.params.nnear as usize*MAXN*3*GROUND+3*self.photons+self.classes+self.weights+self.exact+self.transient+self.rows+self.between}
}
