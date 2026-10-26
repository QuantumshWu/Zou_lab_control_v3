//! Fixed-manifold full-density population engine and complete E/F kernels.
//! No physical model is rebuilt; all coefficient entries and mode edges are
//! copied once from the current prepared plan. No trajectory is retained.
#![allow(dead_code)]
use std::{ffi::c_void,mem,panic::{catch_unwind,AssertUnwindSafe},slice,sync::Arc,time::Instant};
#[path="../../src/lapack.rs"]
mod lapack;
use lapack::{Complex64 as C,Lapack};
mod parallel;
use parallel::{Execution,ParallelStats};
#[cfg(feature="preparation")]
pub mod preparation;
#[cfg(feature="preparation")]
pub mod owned_prepared;
// Reuse the production Pade implementation verbatim. This is not another
// exponential engine or a copy that can diverge from the mature solver.
#[cfg(feature="preparation")]
#[path="../../src/dense_expm.rs"]
mod dense_expm;

#[repr(C)]
pub struct AwbCoefficient { rows:u32,cols:u32,nnz:u64,row_ptr:*const u64,column:*const u32,values:*const C }
#[repr(C)]
pub struct AwbLinks { stage:u32,coefficient:u32,edges:u64,source:*const u32,target:*const u32 }
#[repr(C)]
#[derive(Clone,Copy)]
pub struct AwbEmbedding { count:u64,source:*const u32,target:*const u32 }
#[repr(C)]
pub struct AwbPlan {
    n:u32,g:u32,e:u32,rank:u32,
    input_modes:u32,output_modes:u32,d_modes:u32,j_modes:u32,z_modes:u32,
    coefficient_count:u32,link_count:u32,optical_count:u32,max_rhs:u32,kernel_policy:u32,
    coefficients:*const AwbCoefficient,links:*const AwbLinks,
    input_to_j:AwbEmbedding,first_to_j:AwbEmbedding,j_to_output:AwbEmbedding,z_to_output:AwbEmbedding,
    gg_indices:*const u32,optical_indices:*const u32,output_frequency:*const f64,
}
#[repr(C)]
pub struct AwbComplexBuffer {data:*mut C,length:u64,row_stride:u64,column_stride:u64}
#[repr(C)]
pub struct AwbRealBuffer {data:*mut f64,length:u64,row_stride:u64,column_stride:u64}
#[repr(C)]
#[derive(Clone,Copy,Default)]
pub struct AwbStats {
    apply_calls:u64,absolute_calls:u64,columns:u64,prepared_bytes:u64,workspace_bytes:u64,
    csr_complex_or_real_macs:u64,blas_calls:u64,preparation_ns:u64,apply_ns:u64,absolute_ns:u64,
}
struct Coefficient {rows:usize,cols:usize,row:Vec<usize>,col:Vec<usize>,value:Vec<C>,absolute:Vec<f64>,dense:Vec<C>,dense_abs:Vec<f64>}
struct Link {coefficient:usize,source:Vec<usize>,target:Vec<usize>}
struct Map {source:Vec<usize>,target:Vec<usize>}
struct Plan {
    n:usize,g:usize,e:usize,d:usize,rank:usize,ni:usize,no:usize,nd:usize,nj:usize,nz:usize,max_rhs:usize,
    coefficients:Arc<Vec<Coefficient>>,stages:[Vec<Link>;8],maps:[Map;4],gg:Vec<usize>,optical:Vec<usize>,frequency:Vec<f64>,lapack:Option<Arc<Lapack>>,execution:Execution,
}
struct Workspace<T> {x:Vec<T>,dx:Vec<T>,j:Vec<T>,ee:Vec<T>,z:Vec<T>,mass:Vec<T>,pencil:Vec<T>,packed:Vec<T>,product:Vec<T>}
impl<T> Default for Workspace<T> {fn default()->Self {Self{x:vec![],dx:vec![],j:vec![],ee:vec![],z:vec![],mass:vec![],pencil:vec![],packed:vec![],product:vec![]}}}
struct Handle {plan:Plan,complex:Workspace<C>,real:Workspace<f64>,stats:AwbStats}

fn size(value:u64)->Result<usize,String>{usize::try_from(value).map_err(|_|"Length overflows native usize".into())}
fn product(values:&[usize])->Result<usize,String>{values.iter().try_fold(1usize,|a,&b|a.checked_mul(b).ok_or_else(||"Shape multiplication overflow".into()))}
unsafe fn view<'a,T>(ptr:*const T,len:usize,label:&str)->Result<&'a[T],String>{
    if len==0{return Ok(&[]);}
    if ptr.is_null()||(ptr as usize)%mem::align_of::<T>()!=0||len.checked_mul(mem::size_of::<T>()).filter(|&v|v<=isize::MAX as usize).is_none(){return Err(format!("Invalid {label} pointer/length/alignment"));}
    Ok(slice::from_raw_parts(ptr,len))
}
unsafe fn indices(ptr:*const u32,len:usize,limit:usize,label:&str)->Result<Vec<usize>,String>{
    let out:Vec<_>=view(ptr,len,label)?.iter().map(|&v|v as usize).collect();
    if out.iter().any(|&v|v>=limit){return Err(format!("{label} index outside its declared shape"));} Ok(out)
}
fn unique(values:&[usize])->bool{let mut a=values.to_vec();a.sort_unstable();a.windows(2).all(|w|w[0]!=w[1])}
unsafe fn embedding(raw:AwbEmbedding,src:usize,dst:usize)->Result<Map,String>{
    let count=size(raw.count)?;let source=indices(raw.source,count,src,"embedding source")?;let target=indices(raw.target,count,dst,"embedding target")?;
    if !unique(&source)||!unique(&target){return Err("Embedding contains repeated source or target modes".into());}Ok(Map{source,target})
}
impl Plan {
    unsafe fn prepare(raw:&AwbPlan)->Result<Self,String>{
        let(n,g,e,rank)=(raw.n as usize,raw.g as usize,raw.e as usize,raw.rank as usize);
        if n==0||g==0||e==0||g.checked_add(e)!=Some(n)||rank>product(&[g,g])?||raw.max_rhs==0||raw.kernel_policy>1{return Err("Invalid primitive plan dimensions/kernel policy".into());}
        let d=product(&[n,n])?;let(ni,no,nd,nj,nz)=(raw.input_modes as usize,raw.output_modes as usize,raw.d_modes as usize,raw.j_modes as usize,raw.z_modes as usize);
        if ni==0||no==0||nj==0{return Err("Input/output/J domains must be nonempty".into());}
        let mut coefficients=Vec::new();
        for a in view(raw.coefficients,raw.coefficient_count as usize,"coefficient descriptors")? {
            let(rows,cols,nnz)=(a.rows as usize,a.cols as usize,size(a.nnz)?);
            let row:Vec<_>=view(a.row_ptr,rows.checked_add(1).ok_or("CSR row overflow")?,"CSR row pointers")?.iter().map(|&v|size(v)).collect::<Result<_,_>>()?;
            if row[0]!=0||row[rows]!=nnz||row.windows(2).any(|w|w[0]>w[1]){return Err("Malformed complete CSR row pointers".into());}
            let col=indices(a.column,nnz,cols,"CSR column")?;
            let value=view(a.values,nnz,"CSR values")?.to_vec();
            if value.iter().any(|a|!a.re.is_finite()||!a.im.is_finite()){return Err("Nonfinite primitive coefficient".into());}
            let absolute:Vec<_>=value.iter().map(|v|v.re.hypot(v.im)).collect();
            let length=product(&[rows,cols])?;
            let mut dense=Vec::new();let mut dense_abs=Vec::new();
            if raw.kernel_policy==1&&rows.min(cols)>=8&&(nnz as f64)>=0.15*(length as f64){
                dense=vec![C::default();length];dense_abs=vec![0.;length];
                for r in 0..rows{for p in row[r]..row[r+1]{let at=r+col[p]*rows;dense[at].re+=value[p].re;dense[at].im+=value[p].im;dense_abs[at]+=absolute[p];}}
            }
            coefficients.push(Coefficient{rows,cols,row,col,value,absolute,dense,dense_abs});
        }
        let dims=[(ni,nd,n,n),(nd,nj,n,n),(nj,no,n,n),(nj,no,n,n),(nj,nz,rank,e*e),(nj,no,rank,e*e),(nz,no,rank,rank),(nz,no,raw.optical_count as usize,rank)];
        let mut stages:[Vec<Link>;8]=std::array::from_fn(|_|Vec::new());
        for l in view(raw.links,raw.link_count as usize,"link descriptors")?{
            let stage=l.stage as usize;let id=l.coefficient as usize;
            if stage>=8||id>=coefficients.len(){return Err("Unknown primitive stage/coefficient id".into());}
            let(src,dst,rows,cols)=dims[stage];let c=&coefficients[id];
            if(c.rows,c.cols)!=(rows,cols){return Err(format!("Coefficient shape differs for stage {stage}"));}
            let edges=size(l.edges)?;
            stages[stage].push(Link{coefficient:id,source:indices(l.source,edges,src,"mode source")?,target:indices(l.target,edges,dst,"mode target")?});
        }
        let gg=indices(raw.gg_indices,rank,d,"GG output row")?;
        let optical=indices(raw.optical_indices,raw.optical_count as usize,d,"optical output row")?;
        if !unique(&gg)||!unique(&optical)||gg.iter().any(|&v|v/n>=g||v%n>=g)||optical.iter().any(|&v|(v/n<g)==(v%n<g)){return Err("Invalid or repeated complete GG/optical output indices".into());}
        let frequency=view(raw.output_frequency,no,"output frequencies")?.to_vec();
        if frequency.iter().any(|v|!v.is_finite()){return Err("Nonfinite output transport frequency".into());}
        let maps=[embedding(raw.input_to_j,ni,nj)?,embedding(raw.first_to_j,nd,nj)?,embedding(raw.j_to_output,nj,no)?,embedding(raw.z_to_output,nz,no)?];
        let lapack=if raw.kernel_policy==1{Some(Arc::new(Lapack::load()?))}else{None};
        let execution=Execution::new()?;
        Ok(Self{n,g,e,d,rank,ni,no,nd,nj,nz,max_rhs:raw.max_rhs as usize,coefficients:Arc::new(coefficients),stages,maps,gg,optical,frequency,lapack,execution})
    }
    fn bytes(&self)->usize{
        self.coefficients.iter().map(|a|a.row.len()*mem::size_of::<usize>()+a.col.len()*mem::size_of::<usize>()+a.value.len()*16+a.absolute.len()*8+a.dense.len()*16+a.dense_abs.len()*8).sum::<usize>()
        +self.stages.iter().flatten().map(|l|(l.source.len()+l.target.len())*mem::size_of::<usize>()).sum::<usize>()
        +self.maps.iter().map(|m|(m.source.len()+m.target.len())*mem::size_of::<usize>()).sum::<usize>()+(self.gg.len()+self.optical.len())*mem::size_of::<usize>()+self.frequency.len()*8
    }
}

trait Number:Copy+Default+Send+Sync {
    fn from_complex(x:C)->Self;
    fn coefficient(c:&Coefficient,p:usize)->Self;
    fn dense(c:&Coefficient)->&[Self];
    fn add(out:&mut Self,value:Self);
    fn madd(out:&mut Self,a:Self,b:Self,scale:f64);
    fn transport(out:&mut Self,omega:f64,value:Self);
    fn finite(self)->bool;
    fn gemm(lap:&Lapack,a:&[Self],b:&[Self],out:&mut[Self],m:usize,n:usize,k:usize)->Result<(),String>;
}
impl Number for C {
    #[inline]fn from_complex(x:C)->Self{x}
    #[inline]fn coefficient(c:&Coefficient,p:usize)->Self{c.value[p]}
    fn dense(c:&Coefficient)->&[Self]{&c.dense}
    #[inline]fn add(out:&mut Self,v:Self){out.re+=v.re;out.im+=v.im;}
    #[inline]fn madd(out:&mut Self,a:Self,b:Self,s:f64){out.re+=s*(a.re*b.re-a.im*b.im);out.im+=s*(a.re*b.im+a.im*b.re);}
    #[inline]fn transport(out:&mut Self,w:f64,v:Self){out.re+=w*v.im;out.im-=w*v.re;}
    #[inline]fn finite(self)->bool{self.re.is_finite()&&self.im.is_finite()}
    fn gemm(lap:&Lapack,a:&[Self],b:&[Self],out:&mut[Self],m:usize,n:usize,k:usize)->Result<(),String>{lap.multiply_complex_into(a,b,out,m,n,k)}
}
impl Number for f64 {
    #[inline]fn from_complex(x:C)->Self{x.re.hypot(x.im)}
    #[inline]fn coefficient(c:&Coefficient,p:usize)->Self{c.absolute[p]}
    fn dense(c:&Coefficient)->&[Self]{&c.dense_abs}
    #[inline]fn add(out:&mut Self,v:Self){*out+=v;}
    #[inline]fn madd(out:&mut Self,a:Self,b:Self,s:f64){*out+=s.abs()*a*b;}
    #[inline]fn transport(out:&mut Self,w:f64,v:Self){*out+=w.abs()*v;}
    #[inline]fn finite(self)->bool{self.is_finite()}
    fn gemm(lap:&Lapack,a:&[Self],b:&[Self],out:&mut[Self],m:usize,n:usize,k:usize)->Result<(),String>{lap.multiply_into(a,b,out,m,n,k)}
}
fn resize<T:Number>(a:&mut Vec<T>,len:usize,zero:bool)->Result<(),String>{
    if len>a.capacity(){a.try_reserve(len-a.len()).map_err(|_|"Workspace allocation failed")?;}
    a.resize(len,T::default());if zero{a.fill(T::default());}Ok(())
}
impl<T:Number> Workspace<T>{
    fn prepare(&mut self,p:&Plan,r:usize)->Result<(),String>{
        resize(&mut self.x,product(&[p.ni,p.d,r])?,false)?;
        resize(&mut self.dx,product(&[p.nd,p.d,r])?,true)?;
        resize(&mut self.j,product(&[p.nj,p.d,r])?,true)?;
        resize(&mut self.ee,product(&[p.nj,p.e,p.e,r])?,false)?;
        resize(&mut self.z,product(&[p.nz,p.rank,r])?,true)?;
        resize(&mut self.mass,product(&[p.no,p.d,r])?,true)?;
        resize(&mut self.pencil,product(&[p.no,p.d,r])?,true)
    }
    fn bytes(&self)->usize{[&self.x,&self.dx,&self.j,&self.ee,&self.z,&self.mass,&self.pencil,&self.packed,&self.product].iter().map(|v|v.capacity()*mem::size_of::<T>()).sum()}
}
fn embed<T:Number>(map:&Map,input:&[T],out:&mut[T],width:usize,rhs:usize){
    for(&s,&t)in map.source.iter().zip(&map.target){for k in 0..width*rhs{T::add(&mut out[t*width*rhs+k],input[s*width*rhs+k]);}}
}
fn hilbert_range<T:Number>(coefficients:&[Coefficient],links:&[Link],n:usize,input:&[T],out:&mut[T],start_mode:usize,rhs:usize,scale:f64,commutator:bool,right:bool){
    let d=n*n;let stop_mode=start_mode+out.len()/(d*rhs);
    // Each output sees precisely the original link/edge/CSR-entry order.
    // Filtering nonowned targets changes no accumulation for owned targets.
    for link in links{let c=&coefficients[link.coefficient];
        for(&source,&target)in link.source.iter().zip(&link.target){
            if target<start_mode||target>=stop_mode{continue;}
            let ib=source*d*rhs;let ob=(target-start_mode)*d*rhs;
            for row in 0..n{for at in c.row[row]..c.row[row+1]{let col=c.col[at];let a=T::coefficient(c,at);
                if !right||commutator{for j in 0..n{let i0=ib+(col*n+j)*rhs;let o0=ob+(row*n+j)*rhs;for b in 0..rhs{T::madd(&mut out[o0+b],a,input[i0+b],scale);}}}
                if right||commutator{let sign=if commutator{-scale}else{scale};for i in 0..n{let i0=ib+(i*n+row)*rhs;let o0=ob+(i*n+col)*rhs;for b in 0..rhs{T::madd(&mut out[o0+b],a,input[i0+b],sign);}}}
            }}
        }
    }
}
fn hilbert<T:Number>(p:&Plan,stage:usize,input:&[T],out:&mut[T],rhs:usize,scale:f64,commutator:bool,right:bool,stats:&mut AwbStats){
    let n=p.n;let links=&p.stages[stage];let coefficients=&p.coefficients;
    let macs=links.iter().map(|link|coefficients[link.coefficient].value.len()*link.source.len()*n*rhs*if commutator{2}else{1}).sum::<usize>();
    p.execution.output_chunks(out,p.d*rhs,macs,|start,chunk|hilbert_range(coefficients,links,n,input,chunk,start,rhs,scale,commutator,right));
    stats.csr_complex_or_real_macs=stats.csr_complex_or_real_macs.saturating_add(macs as u64);
}
fn thin_csr_range<T:Number>(coefficients:&[Coefficient],links:&[Link],input:&[T],out:&mut[T],start_mode:usize,input_width:usize,output_width:usize,output_rows:Option<&[usize]>,rhs:usize){
    if output_width==0{return;}
    let stop_mode=start_mode+out.len()/(output_width*rhs);
    for link in links{let c=&coefficients[link.coefficient];
        for(&source,&target)in link.source.iter().zip(&link.target){
            if target<start_mode||target>=stop_mode{continue;}
            for row in 0..c.rows{let rr=output_rows.map_or(row,|rows|rows[row]);let output=((target-start_mode)*output_width+rr)*rhs;
                for at in c.row[row]..c.row[row+1]{let input0=(source*input_width+c.col[at])*rhs;let a=T::coefficient(c,at);for b in 0..rhs{T::madd(&mut out[output+b],a,input[input0+b],1.);}}
            }
        }
    }
}
fn thin<T:Number>(p:&Plan,stage:usize,input:&[T],out:&mut[T],input_width:usize,output_width:usize,output_rows:Option<&[usize]>,rhs:usize,packed:&mut Vec<T>,result:&mut Vec<T>,stats:&mut AwbStats)->Result<(),String>{
    let links=&p.stages[stage];let coefficients=&p.coefficients;
    let uses_blas=p.lapack.is_some()&&links.iter().any(|link|!T::dense(&coefficients[link.coefficient]).is_empty()&&link.source.len()*rhs>=8);
    if !uses_blas{
        let macs=links.iter().map(|link|coefficients[link.coefficient].value.len()*link.source.len()*rhs).sum::<usize>();
        p.execution.output_chunks(out,output_width*rhs,macs,|start,chunk|thin_csr_range(coefficients,links,input,chunk,start,input_width,output_width,output_rows,rhs));
        stats.csr_complex_or_real_macs=stats.csr_complex_or_real_macs.saturating_add(macs as u64);
        return Ok(());
    }
    // Preserve the existing serial BLAS batching/reduction exactly. Native
    // output workers are used only for CSR stages, avoiding nested BLAS
    // threads, repeated packing, and a thread-count-dependent arithmetic.
    p.execution.mark_serial();
    for link in &p.stages[stage]{let c=&p.coefficients[link.coefficient];let columns=product(&[link.source.len(),rhs])?;
        if let Some(lap)=p.lapack.as_ref().filter(|_|!T::dense(c).is_empty()&&columns>=8){
            resize(packed,product(&[c.cols,columns])?,false)?;resize(result,product(&[c.rows,columns])?,false)?;
            for(edge,&source)in link.source.iter().enumerate(){for b in 0..rhs{for col in 0..c.cols{packed[col+(edge*rhs+b)*c.cols]=input[(source*input_width+col)*rhs+b];}}}
            T::gemm(lap,T::dense(c),packed,result,c.rows,columns,c.cols)?;
            for(edge,&target)in link.target.iter().enumerate(){for b in 0..rhs{for row in 0..c.rows{let rr=output_rows.map_or(row,|rows|rows[row]);T::add(&mut out[(target*output_width+rr)*rhs+b],result[row+(edge*rhs+b)*c.rows]);}}}
            stats.blas_calls+=1;
        }else{
            for(&source,&target)in link.source.iter().zip(&link.target){for row in 0..c.rows{let rr=output_rows.map_or(row,|rows|rows[row]);let output=(target*output_width+rr)*rhs;
                for at in c.row[row]..c.row[row+1]{let input0=(source*input_width+c.col[at])*rhs;let a=T::coefficient(c,at);for b in 0..rhs{T::madd(&mut out[output+b],a,input[input0+b],1.);}}
            }}
            stats.csr_complex_or_real_macs=stats.csr_complex_or_real_macs.saturating_add((c.value.len()*link.source.len()*rhs)as u64);
        }
    }Ok(())
}
fn execute<T:Number>(p:&Plan,w:&mut Workspace<T>,rhs:usize,stats:&mut AwbStats)->Result<(),String>{
    hilbert(p,0,&w.x,&mut w.dx,rhs,1.,true,false,stats);
    embed(&p.maps[0],&w.x,&mut w.j,p.d,rhs);embed(&p.maps[1],&w.dx,&mut w.j,p.d,rhs);
    hilbert(p,1,&w.dx,&mut w.j,rhs,0.5,true,false,stats);
    for m in 0..p.nj{for i in 0..p.e{for j in 0..p.e{for b in 0..rhs{w.ee[((m*p.e+i)*p.e+j)*rhs+b]=w.j[(m*p.d+(p.g+i)*p.n+p.g+j)*rhs+b];}}}}
    thin(p,4,&w.ee,&mut w.z,p.e*p.e,p.rank,None,rhs,&mut w.packed,&mut w.product,stats)?;
    embed(&p.maps[2],&w.j,&mut w.mass,p.d,rhs);
    for(&s,&t)in p.maps[3].source.iter().zip(&p.maps[3].target){for(row,&rr)in p.gg.iter().enumerate(){for b in 0..rhs{T::add(&mut w.mass[(t*p.d+rr)*rhs+b],w.z[(s*p.rank+row)*rhs+b]);}}}
    hilbert(p,2,&w.j,&mut w.pencil,rhs,1.,false,false,stats);hilbert(p,3,&w.j,&mut w.pencil,rhs,1.,false,true,stats);
    thin(p,5,&w.ee,&mut w.pencil,p.e*p.e,p.d,Some(&p.gg),rhs,&mut w.packed,&mut w.product,stats)?;
    thin(p,6,&w.z,&mut w.pencil,p.rank,p.d,Some(&p.gg),rhs,&mut w.packed,&mut w.product,stats)?;
    thin(p,7,&w.z,&mut w.pencil,p.rank,p.d,Some(&p.optical),rhs,&mut w.packed,&mut w.product,stats)?;
    for m in 0..p.no{for i in 0..p.d*rhs{let at=m*p.d*rhs+i;T::transport(&mut w.pencil[at],p.frequency[m],w.mass[at]);}}
    if w.mass.iter().chain(&w.pencil).any(|x|!x.finite()){return Err("Nonfinite complete native E/F result".into());}Ok(())
}

struct Layout {rows:usize,cols:usize,rs:usize,cs:usize,span:usize,begin:usize,end:usize}
fn gcd(mut a:usize,mut b:usize)->usize{while b!=0{let c=a%b;a=b;b=c;}a}
fn buffer<T>(ptr:*mut T,len:u64,rs:u64,cs:u64,rows:usize,cols:usize)->Result<Layout,String>{
    let(rs,cs,len)=(size(rs)?,size(cs)?,size(len)?);
    if rows==0||cols==0||rs==0||cs==0||ptr.is_null()||(ptr as usize)%mem::align_of::<T>()!=0{return Err("Invalid strided buffer descriptor".into());}
    let divisor=gcd(rs,cs);
    if cs/divisor<rows&&rs/divisor<cols{return Err("Strided buffer aliases its own logical entries".into());}
    let span=(rows-1).checked_mul(rs).and_then(|a|(cols-1).checked_mul(cs).and_then(|b|a.checked_add(b))).and_then(|v|v.checked_add(1)).ok_or("Buffer span overflow")?;
    let bytes=span.checked_mul(mem::size_of::<T>()).filter(|&v|v<=isize::MAX as usize).ok_or("Buffer byte span overflow")?;
    if span>len{return Err("Buffer allocation length is smaller than its logical strided span".into());}
    let begin=ptr as usize;let end=begin.checked_add(bytes).ok_or("Pointer span overflow")?;
    Ok(Layout{rows,cols,rs,cs,span,begin,end})
}
fn overlap(a:&Layout,b:&Layout)->bool{a.begin<b.end&&b.begin<a.end}
unsafe fn load<T:Number>(input:&AwbComplexBuffer,layout:&Layout,w:&mut Workspace<T>)->Result<(),String>{
    // A valid positive-stride NumPy view can span uninitialized padding in
    // its parent allocation. Do not create a typed slice over those gaps:
    // buffer() has checked the span, and only logical entries are readable.
    for row in 0..layout.rows{for col in 0..layout.cols{
        let x=input.data.add(row*layout.rs+col*layout.cs).read();
        if !x.re.is_finite()||!x.im.is_finite(){return Err("Nonfinite original input field".into());}
        w.x[row*layout.cols+col]=T::from_complex(x);
    }}
    Ok(())
}
unsafe fn store_output<T:Copy>(data:*mut T,layout:&Layout,value:&[T]){
    // Output allocations and padding may be uninitialized. Write only the
    // logical chunk, without forming a slice/reference over parent gaps.
    for row in 0..layout.rows{for col in 0..layout.cols{data.add(row*layout.rs+col*layout.cs).write(value[row*layout.cols+col]);}}
}
fn ns(start:Instant)->u64{start.elapsed().as_nanos().min(u64::MAX as u128)as u64}
unsafe fn error_text(ptr:*mut u8,capacity:u64,text:&str){if ptr.is_null()||capacity==0{return;}let count=(capacity.saturating_sub(1)as usize).min(text.len());std::ptr::copy_nonoverlapping(text.as_ptr(),ptr,count);*ptr.add(count)=0;}
fn guarded(error:*mut u8,capacity:u64,action:impl FnOnce()->Result<(),String>)->i32{
    match catch_unwind(AssertUnwindSafe(action)){
        Ok(Ok(()))=>{unsafe{error_text(error,capacity,"")};0},
        Ok(Err(message))=>{unsafe{error_text(error,capacity,&message)};1},
        Err(_)=>{unsafe{error_text(error,capacity,"Native pencil panic caught at C ABI boundary")};2},
    }
}
#[no_mangle]pub extern "C" fn awb_pencil_abi_version()->u32{1}
#[no_mangle]pub extern "C" fn awb_pencil_extension_version()->u32{2}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_create(raw:*const AwbPlan,output:*mut*mut c_void,error:*mut u8,capacity:u64)->i32{
    guarded(error,capacity,||{if raw.is_null()||output.is_null(){return Err("Null plan/handle output pointer".into());}*output=std::ptr::null_mut();let start=Instant::now();let plan=Plan::prepare(&*raw)?;
        let stats=AwbStats{prepared_bytes:plan.bytes()as u64,preparation_ns:ns(start),..AwbStats::default()};
        *output=Box::into_raw(Box::new(Handle{plan,complex:Workspace::default(),real:Workspace::default(),stats})).cast();Ok(())})
}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_apply(handle:*mut c_void,input:*const AwbComplexBuffer,nrhs:u32,mass:*const AwbComplexBuffer,pencil:*const AwbComplexBuffer,error:*mut u8,capacity:u64)->i32{
    guarded(error,capacity,||{if handle.is_null()||input.is_null()||mass.is_null()||pencil.is_null(){return Err("Null action descriptor".into());}
        let h=&mut*handle.cast::<Handle>();let rhs=nrhs as usize;if rhs==0||rhs>h.plan.max_rhs{return Err("RHS batch exceeds prepared max_rhs; chunk without truncating columns".into());}
        let(input,mass,pencil)=(&*input,&*mass,&*pencil);let p=&h.plan;
        let il=buffer(input.data,input.length,input.row_stride,input.column_stride,p.ni*p.d,rhs)?;
        let el=buffer(mass.data,mass.length,mass.row_stride,mass.column_stride,p.no*p.d,rhs)?;
        let fl=buffer(pencil.data,pencil.length,pencil.row_stride,pencil.column_stride,p.no*p.d,rhs)?;
        if overlap(&il,&el)||overlap(&il,&fl)||overlap(&el,&fl){return Err("Input/E/F buffers must not overlap".into());}
        let start=Instant::now();h.complex.prepare(p,rhs)?;load(input,&il,&mut h.complex)?;execute(p,&mut h.complex,rhs,&mut h.stats)?;
        store_output(mass.data,&el,&h.complex.mass);store_output(pencil.data,&fl,&h.complex.pencil);
        h.stats.apply_calls+=1;h.stats.columns+=rhs as u64;h.stats.apply_ns+=ns(start);h.stats.workspace_bytes=(h.complex.bytes()+h.real.bytes())as u64;Ok(())})
}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_apply_abs(handle:*mut c_void,input:*const AwbComplexBuffer,nrhs:u32,mass:*const AwbRealBuffer,pencil:*const AwbRealBuffer,error:*mut u8,capacity:u64)->i32{
    guarded(error,capacity,||{if handle.is_null()||input.is_null()||mass.is_null()||pencil.is_null(){return Err("Null envelope descriptor".into());}
        let h=&mut*handle.cast::<Handle>();let rhs=nrhs as usize;if rhs==0||rhs>h.plan.max_rhs{return Err("Envelope batch exceeds prepared max_rhs".into());}
        let(input,mass,pencil)=(&*input,&*mass,&*pencil);let p=&h.plan;
        let il=buffer(input.data,input.length,input.row_stride,input.column_stride,p.ni*p.d,rhs)?;
        let el=buffer(mass.data,mass.length,mass.row_stride,mass.column_stride,p.no*p.d,rhs)?;
        let fl=buffer(pencil.data,pencil.length,pencil.row_stride,pencil.column_stride,p.no*p.d,rhs)?;
        if overlap(&il,&el)||overlap(&il,&fl)||overlap(&el,&fl){return Err("Input and envelope buffers must not overlap".into());}
        let start=Instant::now();h.real.prepare(p,rhs)?;load(input,&il,&mut h.real)?;execute(p,&mut h.real,rhs,&mut h.stats)?;
        store_output(mass.data,&el,&h.real.mass);store_output(pencil.data,&fl,&h.real.pencil);
        h.stats.absolute_calls+=1;h.stats.absolute_ns+=ns(start);h.stats.workspace_bytes=(h.complex.bytes()+h.real.bytes())as u64;Ok(())})
}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_apply_m(handle:*mut c_void,input:*const AwbComplexBuffer,nrhs:u32,shift:C,output:*const AwbComplexBuffer,error:*mut u8,capacity:u64)->i32{
    guarded(error,capacity,||{
        if handle.is_null()||input.is_null()||output.is_null(){return Err("Null complete-M action descriptor".into());}
        if !shift.re.is_finite()||!shift.im.is_finite(){return Err("Complete-M shift must be finite".into());}
        let h=&mut*handle.cast::<Handle>();let rhs=nrhs as usize;
        if rhs==0||rhs>h.plan.max_rhs{return Err("Complete-M RHS batch exceeds prepared max_rhs".into());}
        let(input,output)=(&*input,&*output);let p=&h.plan;
        let il=buffer(input.data,input.length,input.row_stride,input.column_stride,p.ni*p.d,rhs)?;
        let ol=buffer(output.data,output.length,output.row_stride,output.column_stride,p.no*p.d,rhs)?;
        if overlap(&il,&ol){return Err("Complete-M input/output buffers must not overlap".into());}
        let start=Instant::now();h.complex.prepare(p,rhs)?;
        load(input,&il,&mut h.complex)?;
        // Keep exactly the original full E/F arithmetic. This extension
        // only fuses their final linear combination and single boundary
        // write; it does not fold, truncate, or substitute an inner solve.
        execute(p,&mut h.complex,rhs,&mut h.stats)?;
        for (f,e) in h.complex.pencil.iter_mut().zip(&h.complex.mass){
            let combined=C{re:shift.re*e.re-shift.im*e.im-f.re,
                           im:shift.re*e.im+shift.im*e.re-f.im};
            if !combined.finite(){return Err("Nonfinite complete complex shifted-pencil result".into());}
            *f=combined;
        }
        store_output(output.data,&ol,&h.complex.pencil);
        h.stats.apply_calls+=1;h.stats.columns+=rhs as u64;h.stats.apply_ns+=ns(start);
        h.stats.workspace_bytes=(h.complex.bytes()+h.real.bytes())as u64;Ok(())
    })
}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_stats(handle:*mut c_void,out:*mut AwbStats)->i32{if handle.is_null()||out.is_null(){1}else{*out=(*handle.cast::<Handle>()).stats;0}}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_parallel_stats(handle:*mut c_void,out:*mut ParallelStats)->i32{if handle.is_null()||out.is_null(){1}else{*out=(*handle.cast::<Handle>()).plan.execution.stats();0}}
#[no_mangle]pub unsafe extern "C" fn awb_pencil_destroy(handle:*mut c_void){if !handle.is_null(){let _=catch_unwind(AssertUnwindSafe(||drop(Box::from_raw(handle.cast::<Handle>()))));}}

#[cfg(all(test,feature="parallel"))]
mod output_partition_tests {
    use super::*;
    fn coefficients()->Vec<Coefficient>{
        let values=vec![C{re:0.3,im:-0.8},C{re:1e-100,im:2e-101},C{re:-1.7,im:0.2}];
        vec![Coefficient{rows:2,cols:2,row:vec![0,2,3],col:vec![0,1,0],
            absolute:values.iter().map(|v|v.re.hypot(v.im)).collect(),value:values,dense:vec![],dense_abs:vec![]}]
    }
    fn links()->Vec<Link>{vec![Link{coefficient:0,source:vec![0,2,3,1,4,0,2,1,3,4],target:vec![0,0,2,3,4,5,6,7,2,5]},
        Link{coefficient:0,source:vec![1,4,2,0],target:vec![7,0,2,4]}]}
    fn assert_bits(a:&[C],b:&[C]){assert_eq!(a.len(),b.len());for(x,y)in a.iter().zip(b){assert_eq!(x.re.to_bits(),y.re.to_bits());assert_eq!(x.im.to_bits(),y.im.to_bits());}}
    #[test]
    fn output_partition_hilbert_preserves_per_output_accumulation_bits(){
        let c=coefficients();let l=links();let rhs=3;let n=2;
        let input:Vec<_>=(0..5*n*n*rhs).map(|i|C{re:((i*17+5)%37)as f64/11.,im:((i*7+3)%23)as f64/-13.}).collect();
        let executor=Execution::with_threads(4).unwrap();
        for(comm,right)in[(true,false),(false,false),(false,true)]{
            let mut expected=vec![C{re:0.013,im:-0.007};8*n*n*rhs];let mut actual=expected.clone();
            hilbert_range(&c,&l,n,&input,&mut expected,0,rhs,0.5,comm,right);
            executor.output_chunks(&mut actual,n*n*rhs,4*262_144,|start,out|hilbert_range(&c,&l,n,&input,out,start,rhs,0.5,comm,right));
            assert_bits(&expected,&actual);
        }
        assert_eq!(executor.stats().parallel_stages,3);
    }
    #[test]
    fn output_partition_thin_and_absolute_keep_all_rows_and_tiny_entries(){
        let c=coefficients();let l=links();let rhs=2;let mapped=[1,3];
        let input:Vec<_>=(0..5*2*rhs).map(|i|((i*11+3)%19)as f64/17.).collect();
        let executor=Execution::with_threads(4).unwrap();
        let mut expected=vec![0.125;8*4*rhs];let mut actual=expected.clone();
        thin_csr_range(&c,&l,&input,&mut expected,0,2,4,Some(&mapped),rhs);
        executor.output_chunks(&mut actual,4*rhs,4*262_144,|start,out|thin_csr_range(&c,&l,&input,out,start,2,4,Some(&mapped),rhs));
        assert_eq!(expected,actual);
        for mode in 0..8{for row in [0,2]{for b in 0..rhs{assert_eq!(actual[(mode*4+row)*rhs+b],0.125);}}}
        let mut small=vec![0.;8];executor.output_chunks(&mut small,1,8,|start,out|{for(i,v)in out.iter_mut().enumerate(){*v=(start+i)as f64;}});
        assert_eq!(small,(0..8).map(|i|i as f64).collect::<Vec<_>>());
        assert_eq!(executor.stats().parallel_stages,1);assert_eq!(executor.stats().serial_stages,1);
    }
}
