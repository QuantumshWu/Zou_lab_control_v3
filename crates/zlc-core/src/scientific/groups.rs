use std::collections::{BTreeMap,BTreeSet,HashMap,HashSet};
use crate::AnyBuffer;
use super::{Tensor,reduce};
use super::reduce::{Planes,Step};
use super::tensor::product;

type Unit=(usize,usize);
struct Tally {shape:Vec<usize>,n:Vec<i64>,real:BTreeMap<String,Vec<f64>>,integer:Option<Vec<i64>>,flag:Option<Vec<bool>>}
impl Tally {
    fn new(shape:Vec<usize>,part:&Planes,operation:&str)->Result<Self,String>{
        let size=product(&shape)?;let mut real=BTreeMap::new();let mut integer=None;let mut flag=None;
        for (key,tensor) in part{if key=="n"{continue;}match &tensor.values{
            AnyBuffer::F64(_)=>{real.insert(key.clone(),vec![if key=="extreme"{if operation=="min"{f64::INFINITY}else{f64::NEG_INFINITY}}else{0.0};size]);},
            AnyBuffer::I64(_)=>integer=Some(vec![0;size]),AnyBuffer::Bool(_)=>flag=Some(vec![operation=="all";size]),_=>return Err("invalid statistic plane".into())
        }}
        Ok(Self{shape,n:vec![0;size],real,integer,flag})
    }
    fn add(&mut self,slot:usize,from:usize,part:&Planes,operation:&str)->Result<(),String>{
        let AnyBuffer::I64(counts)=&part["n"].values else{return Err("count plane is not int64".into())};
        let n1=self.n[slot];let n2=counts.as_slice()[from];let total=n1.checked_add(n2).ok_or("sample count overflows int64")?;
        let value=|key:&str| match &part[key].values{AnyBuffer::F64(v)=>v.as_slice()[from],_=>unreachable!("real statistic plane")};
        if part.contains_key("mean"){
            let mut moment=super::statistics::Moment{count:n1 as u64,mean:self.real["mean"][slot],m2:self.real["m2"][slot]};
            moment.merge(super::statistics::Moment{count:n2 as u64,mean:value("mean"),m2:value("m2")})?;
            self.real.get_mut("m2").unwrap()[slot]=moment.m2;
            self.real.get_mut("mean").unwrap()[slot]=moment.mean;
        }
        if part.contains_key("stated"){self.real.get_mut("stated").unwrap()[slot]+=value("stated");}
        if let Some(total_plane)=part.get("total"){match &total_plane.values{
            AnyBuffer::I64(v)=>{let current=&mut self.integer.as_mut().unwrap()[slot];*current=current.checked_add(v.as_slice()[from]).ok_or("boolean total overflows int64")?;},
            AnyBuffer::F64(v)=>self.real.get_mut("total").unwrap()[slot]+=v.as_slice()[from],_=>unreachable!(),
        }}
        if part.contains_key("extreme"){
            let b=value("extreme");let a=&mut self.real.get_mut("extreme").unwrap()[slot];
            if !a.is_nan() && if operation=="min"{*a>=b}else{*a<=b}{*a=b;}
        }
        if let Some(flag)=part.get("flag"){let AnyBuffer::Bool(v)=&flag.values else{unreachable!()};let current=&mut self.flag.as_mut().unwrap()[slot];
            *current=if operation=="all"{*current&&v.as_slice()[from]}else{*current||v.as_slice()[from]};}
        self.n[slot]=total;Ok(())
    }
    fn finish(self,operation:&str)->Result<Planes,String>{
        let mut planes:Planes=self.real.into_iter().map(|(k,v)|(k,Tensor::from_reals(v,self.shape.clone()))).collect();
        planes.insert("n".into(),Tensor::from_counts(self.n,self.shape.clone()));
        if let Some(v)=self.integer{planes.insert("total".into(),Tensor::from_counts(v,self.shape.clone()));}
        if let Some(v)=self.flag{planes.insert("flag".into(),Tensor::from_flags(v,self.shape.clone()));}
        reduce::result(planes,operation)
    }
}

pub struct Group {
    operation:String,domain:usize,spanning:bool,every:usize,counts_r:Vec<usize>,counts_p:Vec<usize>,source_points:usize,output_points:usize,
    lines:Option<Vec<usize>>,steps:Option<Vec<usize>>,latest:HashMap<usize,usize>,tallies:BTreeMap<Unit,Tally>,arrived:HashMap<Unit,usize>,closed:HashSet<Unit>,
}
pub struct GroupConfig {
    pub operation:String, pub domain:usize, pub spanning:bool, pub every:usize,
    pub counts_r:Vec<usize>, pub counts_p:Vec<usize>, pub source_points:usize, pub output_points:usize,
    pub lines:Option<Vec<usize>>, pub steps:Option<Vec<usize>>,
}
pub struct GroupFeed {
    pub data:Tensor, pub flags:Vec<bool>, pub sigma:Option<Tensor>,
    pub rows:Vec<usize>, pub points:Vec<usize>, pub plans:Vec<Step>, pub ordinal:usize,
}
pub struct GroupChunk { pub rows:Vec<usize>, pub columns:Vec<usize>, pub planes:Planes }
pub struct GroupState { pub open_groups:usize, pub held_bytes:usize }

impl Group {
    pub fn spans_events(&self)->bool{self.spanning}
    pub fn state(&self)->GroupState{
        let held_bytes=self.tallies.values().map(|t|t.n.len()*8+t.real.values().map(|v|v.len()*8).sum::<usize>()
            +t.integer.as_ref().map_or(0,|v|v.len()*8)+t.flag.as_ref().map_or(0,Vec::len)).sum::<usize>();
        GroupState{open_groups:self.tallies.len(),held_bytes}
    }
    pub fn new(config:GroupConfig)->Result<Self,String>{
        let GroupConfig{operation,domain,spanning,every,counts_r,counts_p,source_points,output_points,lines,steps}=config;
        if !["mean","sum","count","std","min","max","all","any"].contains(&operation.as_str()){return Err("unknown statistic".into());}
        if domain>1{return Err("groups span Repeat or Point".into());}
        if counts_r.is_empty()||counts_p.is_empty()||every==0||source_points==0||output_points==0{return Err("group dimensions must be positive".into());}
        if output_points!=counts_p.len(){return Err("group output Point size differs from its coordinate counts".into());}
        if spanning&&(domain!=0||lines.is_some()){return Err("spanning groups need one implicit Repeat line".into());}
        if lines.is_some()!=steps.is_some()||lines.as_ref().is_some_and(|v|v.len()!=counts_r.len())||steps.as_ref().is_some_and(|v|v.len()!=counts_r.len()){return Err("group line mapping has wrong size".into());}
        Ok(Self{operation,domain,spanning,every,counts_r,counts_p,source_points,output_points,lines,steps,latest:HashMap::new(),tallies:BTreeMap::new(),arrived:HashMap::new(),closed:HashSet::new()})
    }
    fn place(&self,unit:Unit)->(usize,usize){match (&self.lines,&self.steps){(Some(lines),Some(steps))=>(lines[unit.0],steps[unit.0]),_=>(0,unit.0)}}
    fn followed(&self,unit:Unit)->bool{let(line,step)=self.place(unit);self.latest.get(&line).is_some_and(|n|step<*n)}
    fn expected(&self,unit:Unit)->Result<usize,String>{
        let (a,b)=if self.spanning{(self.every.checked_mul(self.counts_r[0]).ok_or("group size overflows")?,self.source_points)}
            else if self.domain==0{(self.counts_r[unit.0],self.source_points)}else{(self.counts_r[unit.0],self.counts_p[unit.1])};
        a.checked_mul(b).ok_or("group size overflows".into())
    }
    pub fn feed(&mut self,feed:GroupFeed)->Result<Vec<GroupChunk>,String>{
        let GroupFeed{data,flags,sigma,rows:out_r,points:out_p,plans:cell_plans,ordinal}=feed;
        data.validate()?;if data.shape.len()<3{return Err("group data requires three-domain shape".into());}
        if out_r.len()!=data.shape[0]||out_p.len()!=data.shape[1]||out_r.iter().any(|n|*n>=self.counts_r.len())||out_p.iter().any(|n|*n>=self.counts_p.len()){
            return Err("group input coordinate mapping is invalid".into());}
        let unique_r=out_r.iter().copied().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        let unique_p=out_p.iter().copied().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        let local_r=out_r.iter().map(|n|unique_r.binary_search(n).unwrap()).collect::<Vec<_>>();
        let local_p=out_p.iter().map(|n|unique_p.binary_search(n).unwrap()).collect::<Vec<_>>();
        let mut hit_r=vec![0usize;unique_r.len()];for n in &local_r{hit_r[*n]+=1;}
        let mut hit_p=vec![0usize;unique_p.len()];for n in &local_p{hit_p[*n]+=1;}
        let mut plans=vec![Step::Mapped{axis:0,codes:local_r,size:unique_r.len()},Step::Mapped{axis:1,codes:local_p,size:unique_p.len()}];
        plans.extend(cell_plans);
        let part=reduce::measure(&data,&flags,sigma.as_ref(),&plans,&self.operation)?;
        let cell_shape=&part["n"].shape[2..];let cells=product(cell_shape)?;
        let mut touched=Vec::new();
        for (i,row) in unique_r.iter().copied().enumerate(){
            let row=if self.spanning{ordinal/self.every}else{row};
            let columns=if self.domain==0{vec![(0,0)]}else{unique_p.iter().copied().enumerate().map(|(j,p)|(p,j)).collect()};
            for (column,j) in columns{
                let unit=(row,column);
                if self.closed.contains(&unit)||(self.domain==0&&self.followed(unit)&&!self.tallies.contains_key(&unit)){return Err("a sample arrived after its group closed; repeats run in order".into());}
                let mut shape=vec![1,if self.domain==0{self.output_points}else{1}];shape.extend_from_slice(cell_shape);
                if !self.tallies.contains_key(&unit){self.tallies.insert(unit,Tally::new(shape,&part,&self.operation)?);}
                let tally=self.tallies.get_mut(&unit).unwrap();
                if self.domain==0{for (j,p) in unique_p.iter().enumerate(){for cell in 0..cells{tally.add(p*cells+cell,(i*unique_p.len()+j)*cells+cell,&part,&self.operation)?;}}}
                else{for cell in 0..cells{tally.add(cell,(i*unique_p.len()+j)*cells+cell,&part,&self.operation)?;}}
                let additions=hit_r[i].checked_mul(if self.domain==0{data.shape[1]}else{hit_p[j]}).ok_or("group arrivals overflow")?;
                let count=self.arrived.entry(unit).or_default();*count=count.checked_add(additions).ok_or("group arrivals overflow")?;
                touched.push(unit);
            }
        }
        if self.domain==0{for unit in &touched{let(line,step)=self.place(*unit);let latest=self.latest.entry(line).or_insert(step);*latest=(*latest).max(step);}}
        let mut due=BTreeSet::new();for unit in touched{if self.arrived[&unit]>=self.expected(unit)?{due.insert(unit);}}
        if self.domain==0{due.extend(self.tallies.keys().filter(|u|self.followed(**u)).copied());}
        self.close(due.into_iter().collect())
    }
    pub fn finish(&mut self)->Result<Vec<GroupChunk>,String>{self.close(self.tallies.keys().copied().collect())}
    fn close(&mut self,units:Vec<Unit>)->Result<Vec<GroupChunk>,String>{
        let mut results=BTreeMap::new();
        for unit in &units{results.insert(*unit,self.tallies.remove(unit).ok_or("group is not open")?.finish(&self.operation)?);self.arrived.remove(unit);self.closed.insert(*unit);}
        if self.domain==0{let kept=self.closed.iter().copied().filter(|u|!self.followed(*u)).collect();self.closed=kept;}
        let mut chunks=Vec::new();
        for (rows,columns) in rectangles(&units,self.spanning){
            let first=&results[&(rows[0],columns[0])];let mut chunk=Planes::new();
            for (name,plane) in first{
                let parts=rows.iter().flat_map(|r|columns.iter().map(|c|&results[&(*r,*c)][name])).collect::<Vec<_>>();
                let mut shape=plane.shape.clone();shape[0]=rows.len();if self.domain==1{shape[1]=columns.len();}
                chunk.insert(name.clone(),Tensor::join(&parts,shape)?);
            }
            chunks.push(GroupChunk{rows,columns,planes:chunk});
        }
        Ok(chunks)
    }
}

fn rectangles(units:&[Unit],spanning:bool)->Vec<(Vec<usize>,Vec<usize>)>{
    let mut sorted=units.to_vec();sorted.sort_unstable();let mut runs:BTreeMap<usize,Vec<Vec<usize>>>=BTreeMap::new();
    for (r,c) in sorted{let runs=runs.entry(r).or_default();if let Some(last)=runs.last_mut(){if last.last().is_some_and(|n|n+1==c){last.push(c);continue;}}runs.push(vec![c]);}
    let mut blocks:Vec<(Vec<usize>,Vec<usize>)>=Vec::new();let mut open:HashMap<Vec<usize>,usize>=HashMap::new();
    for(row,runs)in runs{let mut next=HashMap::new();for cols in runs{let index=match open.get(&cols){Some(i)if !spanning&&blocks[*i].0.last().is_some_and(|r|r+1==row)=>{blocks[*i].0.push(row);*i},_=>{let i=blocks.len();blocks.push((vec![row],cols.clone()));i}};next.insert(cols,index);}open=next;}
    blocks
}
