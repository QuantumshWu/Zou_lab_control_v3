//! Deterministic current-Model unsafe forest and component alignment.
//! This changes only integer coordinates. Every original coefficient and
//! original channel interference survives in GaugedSource::prepare_internal.
use std::collections::{BTreeMap,BTreeSet};
use super::{Charge,Complex64 as C,FourierMatrices,ModelView,PhaseBasis,Sum,plus,difference,nonzero,matrix,forest};
use crate::lapack::Lapack;

type Charges=BTreeSet<Charge>;
#[derive(Clone)]struct Candidate{score:f64,i:usize,j:usize,q:Charge}
#[derive(Clone,Debug)]pub struct GaugeDiagnostics{
    pub candidate_edges:usize,pub unsafe_edges:usize,pub safe_edges:usize,pub tree_edges:usize,
    pub components:usize,pub maximum_sweeps:usize,pub completed_sweeps:usize,pub accepted_moves:usize,
    pub evaluated_pattern_costs:usize,pub initial_cost:Option<[u64;6]>,pub final_cost:Option<[u64;6]>,
    pub legacy_cost:Option<[u64;6]>,pub selected_legacy_compatible_start:bool,
    pub unsafe_component_differences_preserved:bool,pub physical_terms_removed:bool,
}
pub struct GaugeSelection{pub gauges:Vec<Charge>,pub diagnostics:GaugeDiagnostics}
fn sums(a:&Charges,b:&Charges)->Result<Charges,String>{a.iter().flat_map(|q|b.iter().map(move|p|plus(q,p))).collect()}
fn add_set(target:&mut Charges,value:&Charges){target.extend(value.iter().cloned());}
fn union3(a:&Charges,b:&Charges,c:&Charges)->Charges{a.iter().chain(b).chain(c).cloned().collect()}
#[derive(Clone)]struct Entry{q:Charge,i:usize,j:usize,safe:bool}
struct Pattern{
    g:usize,e:usize,ground:Vec<bool>,entries:Vec<Entry>,collapse:Vec<Vec<(usize,usize)>>,
    gg_rank:u64,zero:Charge,cache:BTreeMap<Vec<Charge>,[u64;6]>,
}
impl Pattern{
    fn primary(&self,gauges:&[Charge])->Result<([u64;2],Charges),String>{
        let mut pairs=0_u64;let mut recycling=Charges::new();
        for positions in &self.collapse{
            let charges=positions.iter().map(|&(i,j)|difference(&gauges[j],&gauges[i])).collect::<Result<Charges,_>>()?;
            pairs+=(charges.len()*charges.len())as u64;
            for a in &charges{for b in &charges{recycling.insert(difference(a,b)?);}}
        }
        let nonzero=recycling.iter().filter(|q|q.iter().any(|&k|k!=0)).count()as u64;
        Ok(([pairs,nonzero],recycling))
    }
    fn cost(&mut self,gauges:&[Charge])->Result<[u64;6],String>{
        if let Some(value)=self.cache.get(gauges){return Ok(*value);}
        let(prefix,r)=self.primary(gauges)?;let mut h=Charges::new();let mut ge=Charges::new();let mut eg=Charges::new();
        for entry in &self.entries{
            let charge=plus(&entry.q,&difference(&gauges[entry.j],&gauges[entry.i])?)?;
            h.insert(charge.clone());
            if charge.iter().all(|&k|k==0)||!entry.safe||self.ground[entry.i]==self.ground[entry.j]{continue;}
            if self.ground[entry.i]{ge.insert(charge);}else{eg.insert(charge);}
        }
        // Four sectors GG,GE,EG,EE, with exact optical commutator orientation.
        let mut ad:BTreeMap<(usize,usize),Charges>=BTreeMap::new();
        for block in[(0,2),(1,0),(1,3),(3,2)]{ad.insert(block,ge.clone());}
        for block in[(0,1),(2,0),(2,3),(3,1)]{ad.insert(block,eg.clone());}
        let mut j2:BTreeMap<(usize,usize),Charges>=(0..4).map(|i|((i,i),Charges::from([self.zero.clone()]))).collect();
        for(&key,values)in &ad{add_set(j2.entry(key).or_default(),values);}
        for(&(a,b),left)in &ad{for(&(c,d),right)in &ad{if b==c{add_set(j2.entry((a,d)).or_default(),&sums(left,right)?);}}}
        let k:Charges=r.into_iter().filter(|q|q.iter().any(|&v|v!=0)).collect();
        let dimensions=[self.g*self.g,self.g*self.e,self.e*self.g,self.e*self.e];
        let mut z=Charges::new();let mut weighted_z=0_u64;
        for j in 0..4{let values=sums(&k,j2.get(&(3,j)).unwrap_or(&Charges::new()))?;
            weighted_z+=self.gg_rank*dimensions[j]as u64*values.len()as u64;add_set(&mut z,&values);}
        let mut all_j=Charges::new();let mut weighted_j=0_u64;
        for(&(a,b),values)in &j2{weighted_j+=dimensions[a]as u64*dimensions[b]as u64*values.len()as u64;add_set(&mut all_j,values);}
        let all=union3(&h,&all_j,&z);
        let lengths:Vec<u64>=all.iter().map(|q|q.iter().map(|&v|u64::from(v.unsigned_abs())).sum()).collect();
        let cost=[prefix[0],prefix[1],weighted_z,weighted_j,lengths.iter().copied().max().unwrap_or(0),lengths.iter().sum()];
        // The Python selector uses ONLY these six structural terms. Its last
        // pilot-return MAC estimate is diagnostic and breaks no ties; do not
        // reproduce that otherwise unnecessary assembly here.
        self.cache.insert(gauges.to_vec(),cost);Ok(cost)
    }
}

pub fn select(lap:&Lapack,model:&ModelView<'_>,original:&FourierMatrices,phases:&PhaseBasis,align:bool)->Result<GaugeSelection,String>{
    let n=model.dimension;let axes=phases.axes.len();let zero=vec![0;axes];let h0=&original[&zero];
    let max_energy=(0..n).map(|i|h0[i*n+i].re.abs()).fold(0_f64,f64::max);
    let mut unsafe_edges=Vec::new();let mut safe_edges=Vec::new();let mut flags=BTreeMap::new();let mut all=0;
    for(q,h)in original{let mut off=h.clone();for i in 0..n{off[i*n+i]=C::default();}
        let source_norm=super::primitives::spectral_norm(lap,&off,n)?;let frequency=phases.frequency(q)?;
        let floor=(4.*source_norm).max(256.*f64::EPSILON*(2.*max_energy+(frequency.hi+frequency.lo).abs()));
        for i in 0..n{for j in i+1..n{let amplitude=h[i*n+j].re.hypot(h[i*n+j].im);if amplitude==0.{continue;}
            let mut delta=Sum::default();for x in[h0[i*n+i].re,-h0[j*n+j].re,frequency.hi,frequency.lo]{delta.add(x);}
            let detuning=delta.value().abs();let safe=detuning>floor;let score=amplitude/(1.+detuning);
            if !score.is_finite(){return Err("Unsafe coordinate edge score overflow".into());}
            flags.insert((q.clone(),i,j),safe);flags.insert((super::opposite(q)?,j,i),safe);all+=1;
            let edge=Candidate{score,i,j,q:q.clone()};if safe{safe_edges.push(edge);}else{unsafe_edges.push(edge);}
        }}
    }
    unsafe_edges.sort_by(|a,b|b.score.total_cmp(&a.score).then_with(||a.i.cmp(&b.i)).then_with(||a.j.cmp(&b.j)).then_with(||a.q.cmp(&b.q)));
    let mut groups:Vec<_>=(0..n).collect();let mut tree=Vec::new();
    for edge in &unsafe_edges{let(left,right)=(groups[edge.i],groups[edge.j]);if left==right{continue;}
        for value in &mut groups{if *value==right{*value=left;}}tree.push(edge.clone());}
    let mut groups:Vec<_>=(0..n).collect();let mut base=vec![zero.clone();n];
    for edge in &tree{let(left,right)=(groups[edge.i],groups[edge.j]);let shift=difference(&difference(&base[edge.i],&base[edge.j])?,&edge.q)?;
        for i in 0..n{if groups[i]==right{base[i]=plus(&base[i],&shift)?;groups[i]=left;}}}
    let mut members:BTreeMap<usize,Vec<usize>>=BTreeMap::new();for i in 0..n{members.entry(groups[i]).or_default().push(i);}
    let mut components:Vec<_>=members.keys().copied().collect();components.sort_by_key(|c|members[c][0]);
    for c in &components{let list=&members[c];let anchor=if list.contains(&model.initial_index){model.initial_index}else{list[0]};
        let shift=base[anchor].clone();for &i in list{base[i]=difference(&base[i],&shift)?;}}
    let mut diagnostics=GaugeDiagnostics{candidate_edges:all,unsafe_edges:unsafe_edges.len(),safe_edges:safe_edges.len(),tree_edges:tree.len(),
        components:components.len(),maximum_sweeps:3,completed_sweeps:0,accepted_moves:0,evaluated_pattern_costs:0,
        initial_cost:None,final_cost:None,legacy_cost:None,selected_legacy_compatible_start:false,
        unsafe_component_differences_preserved:true,physical_terms_removed:false};
    if !align{return Ok(GaugeSelection{gauges:base,diagnostics});}
    let anchor=groups[model.initial_index];let(legacy,_,_)=forest(original,phases,n,model.initial_index)?;
    let mut entries=Vec::new();for(q,h)in original{for i in 0..n{for j in 0..n{if nonzero(&h[i*n+j]){
        entries.push(Entry{q:q.clone(),i,j,safe:*flags.get(&(q.clone(),i,j)).unwrap_or(&false)});
    }}}}
    let mut collapse=Vec::new();let mut gg_rows=BTreeSet::new();
    for &channel in model.collapse{let c=matrix(channel,n,"alignment original C")?;let positions:Vec<_>=(0..n*n).filter(|&i|nonzero(&c[i])).map(|i|(i/n,i%n)).collect();
        let outputs:BTreeSet<_>=positions.iter().map(|p|p.0).collect();for &i in &outputs{for &j in &outputs{gg_rows.insert((i,j));}}collapse.push(positions);}
    let mut pattern=Pattern{g:model.ground.iter().filter(|&&g|g).count(),e:model.ground.iter().filter(|&&g|!g).count(),ground:model.ground.to_vec(),entries,collapse,gg_rank:gg_rows.len()as u64,zero:zero.clone(),cache:BTreeMap::new()};
    let mut legacy_start=base.clone();
    for c in &components{let offsets=members[c].iter().map(|&i|difference(&legacy[i],&base[i])).collect::<Result<Charges,_>>()?;
        if offsets.len()==1{let offset=if *c==anchor{zero.clone()}else{offsets.into_iter().next().unwrap()};for &i in &members[c]{legacy_start[i]=plus(&base[i],&offset)?;}}}
    let base_cost=pattern.cost(&base)?;let legacy_start_cost=pattern.cost(&legacy_start)?;
    let(mut current,mut current_cost)=if legacy_start_cost<=base_cost{diagnostics.selected_legacy_compatible_start=true;(legacy_start,legacy_start_cost)}else{(base.clone(),base_cost)};
    diagnostics.initial_cost=Some(current_cost);
    for sweep in 0..3{diagnostics.completed_sweeps=sweep+1;let mut changed=false;
        for component in &components{if *component==anchor{continue;}let list=&members[component];
            let mut offsets=Charges::from([zero.clone(),difference(&current[list[0]],&base[list[0]])?]);
            for edge in &safe_edges{let(ci,cj)=(groups[edge.i],groups[edge.j]);if ci==cj{continue;}
                if ci==*component{offsets.insert(difference(&plus(&edge.q,&current[edge.j])?,&base[edge.i])?);}
                else if cj==*component{offsets.insert(difference(&difference(&current[edge.i],&edge.q)?,&base[edge.j])?);}}
            let mut best=current.clone();let mut best_cost=current_cost;
            for offset in offsets{let mut proposal=current.clone();for &i in list{proposal[i]=plus(&base[i],&offset)?;}
                let(primary,_)=pattern.primary(&proposal)?;if primary>[current_cost[0],current_cost[1]]{continue;}
                let cost=pattern.cost(&proposal)?;if cost<best_cost{best_cost=cost;best=proposal;}}
            if best_cost<current_cost{current=best;current_cost=best_cost;diagnostics.accepted_moves+=1;changed=true;}
        }if !changed{break;}
    }
    for list in members.values(){let offsets=list.iter().map(|&i|difference(&current[i],&base[i])).collect::<Result<Charges,_>>()?;
        if offsets.len()!=1{return Err("Component alignment changed unsafe internal gauge differences".into());}}
    for &i in &members[&anchor]{if current[i]!=base[i]{return Err("Component alignment changed initial component anchor".into());}}
    diagnostics.final_cost=Some(current_cost);diagnostics.legacy_cost=Some(pattern.cost(&legacy)?);diagnostics.evaluated_pattern_costs=pattern.cache.len();
    Ok(GaugeSelection{gauges:current,diagnostics})
}

#[cfg(test)]
mod tests{
    use super::*;use crate::preparation::{MatrixView,DriveView,GaugedSource,GaugePolicy};
    #[test]
    fn preparation_aligned_selector_preserves_full_original_channel_and_explicit_gauge_action(){
        // The -1 drive charge has an exact unsafe 1<->2 resonance:
        // (10-3)-7=0. Other edges remain safe and allow component shifts.
        let n=4;let mut hr=vec![0.;16];let zeros=vec![0.;16];for(i,e)in[0.,10.,3.,4.].iter().enumerate(){hr[i*n+i]=*e;}
        hr[2]=0.03;hr[8]=0.03;
        let mut cosine=vec![0.;16];cosine[6]=0.2;cosine[9]=0.2;cosine[3]=1e-100;cosine[12]=1e-100;
        let mut collapse=vec![0.;16];collapse[2]=0.7;collapse[6]=0.2;collapse[7]=0.8;
        let jumps=[MatrixView{real:&collapse,imag:&zeros}];let drives=[DriveView{omega:7.,cosine:MatrixView{real:&cosine,imag:&zeros},sine:MatrixView{real:&zeros,imag:&zeros}}];
        let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&hr,imag:&zeros},collapse:&jumps,drives:&drives,ground:&[true,true,false,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[7.],None,&[vec![1]]).unwrap();let lap=Lapack::load().unwrap();
        assert!(GaugedSource::prepare(&model,phases.clone(),GaugePolicy::UnsafeComponentAligned).is_err());
        let selected=GaugedSource::prepare_with_lapack(&model,phases.clone(),GaugePolicy::UnsafeComponentAligned,&lap).unwrap();
        let explicit=GaugedSource::prepare(&model,phases,GaugePolicy::Explicit(&selected.gauges)).unwrap();
        assert_eq!(selected.collapse.len(),1);assert_eq!(selected.hamiltonian.len(),explicit.hamiltonian.len());
        for(q,a)in &selected.hamiltonian{for(x,y)in a.iter().zip(&explicit.hamiltonian[q]){assert_eq!(x.re.to_bits(),y.re.to_bits());assert_eq!(x.im.to_bits(),y.im.to_bits());}}
        let diagnostics=selected.diagnostics.coordinate_selection.as_ref().unwrap();assert!(diagnostics.unsafe_component_differences_preserved);assert!(!diagnostics.physical_terms_removed);
        assert!(diagnostics.unsafe_edges>0&&diagnostics.safe_edges>0);assert!(diagnostics.components<n);
        assert!(diagnostics.final_cost.unwrap()<=diagnostics.initial_cost.unwrap());
        assert_eq!(selected.gauges[0],vec![0]);assert!(selected.hamiltonian.values().flatten().any(|z|z.re!=0.&&z.re.abs()<1e-90));
        assert_eq!(selected.radiation().unwrap().coefficient_pairs,explicit.radiation().unwrap().coefficient_pairs);
    }
}
