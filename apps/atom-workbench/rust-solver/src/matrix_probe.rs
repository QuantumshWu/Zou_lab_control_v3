//! Explicit opt-in export for an external linear-solver experiment. No file,
//! operator snapshot, or solved result is retained when the path is absent.
use serde::Serialize;
use std::{fs::{File,OpenOptions},io::{BufWriter,Write},path::{Path,PathBuf},sync::{atomic::{AtomicU64,Ordering},Mutex,OnceLock}};
thread_local!{static ROLE:std::cell::Cell<&'static str>=const{std::cell::Cell::new("unscoped")};}

/// The !Send marker keeps restoration on the thread that entered the role.
pub(super) struct RoleGuard{previous:&'static str,_local:std::marker::PhantomData<std::rc::Rc<()>>}
pub(super) fn enter_role(role:&'static str)->RoleGuard{
    RoleGuard{previous:ROLE.with(|current|current.replace(role)),_local:std::marker::PhantomData}
}
impl Drop for RoleGuard{fn drop(&mut self){ROLE.with(|current|current.set(self.previous));}}

static NEXT_ID:AtomicU64=AtomicU64::new(1);
static SESSION:OnceLock<Mutex<Option<Session>>>=OnceLock::new();
pub(super) fn new_identity()->u64{NEXT_ID.fetch_add(1,Ordering::Relaxed)}

#[derive(Serialize)]
#[serde(rename_all="camelCase")]
struct Header<'a>{
    kind:&'static str,schema:&'static str,prepared_id:u64,dimension:usize,
    storage:&'static str,columns:&'a[Vec<(usize,f64)>],scales:&'a[f64],scaling:&'static str,orientation:&'static str,role:&'static str,
}
#[derive(Serialize)]
#[serde(rename_all="camelCase")]
struct Sample<'a>{kind:&'static str,prepared_id:u64,index:usize,rhs:&'a[f64],unscaled_rhs:&'a[f64],solution:&'a[f64]}
struct Session{identity:u64,dimension:usize,count:usize,writer:BufWriter<File>,exit_after_three:bool}
impl Session{
    fn open(path:&Path,identity:u64,columns:&[Vec<(usize,f64)>],scales:&[f64],role:&'static str,exit_after_three:bool)->Result<Self,String>{
        let n=columns.len();
        if n!=scales.len()||scales.iter().any(|s|!s.is_finite()||*s<=0.)||columns.iter().flatten().any(|&(i,v)|i>=n||!v.is_finite()){
            return Err("Matrix probe received invalid balanced matrix metadata".into());
        }
        if let Some(parent)=path.parent().filter(|p|!p.as_os_str().is_empty()){
            std::fs::create_dir_all(parent).map_err(|e|format!("Matrix probe directory: {e}"))?;
        }
        let file=OpenOptions::new().write(true).create_new(true).open(path)
            .map_err(|e|format!("Matrix probe refuses to overwrite/open {}: {e}",path.display()))?;
        let mut session=Self{identity,dimension:n,count:0,writer:BufWriter::new(file),exit_after_three};
        let header=Header{kind:"matrix",schema:"atom-balanced-system/1",prepared_id:identity,dimension:n,
            storage:"column-entries",columns,scales,scaling:"A_balanced = diag(1/scales) * A_original",orientation:"forward",role};
        session.line(&header)?;Ok(session)
    }
    fn line<T:Serialize>(&mut self,value:&T)->Result<(),String>{
        serde_json::to_writer(&mut self.writer,value).map_err(|e|format!("Matrix probe JSON: {e}"))?;
        self.writer.write_all(b"\n").and_then(|_|self.writer.flush()).map_err(|e|format!("Matrix probe write: {e}"))
    }
    fn sample(&mut self,rhs:&[f64],unscaled_rhs:&[f64],solution:&[f64])->Result<bool,String>{
        if self.count>=3{return Ok(false);}
        if [rhs,unscaled_rhs,solution].iter().any(|v|v.len()!=self.dimension||v.iter().any(|x|!x.is_finite())){
            return Err("Matrix probe received invalid RHS/solution".into());
        }
        let sample=Sample{kind:"rhs",prepared_id:self.identity,index:self.count+1,rhs,unscaled_rhs,solution};
        self.line(&sample)?;self.count+=1;Ok(self.count==3&&self.exit_after_three)
    }
}

/// Select the FIRST matching Prepared instance, before its forward solve.
/// A stable identity is used instead of an address that could move or be reused.
pub(super) fn begin(identity:u64,columns:&[Vec<(usize,f64)>],scales:&[f64])->Result<bool,String>{
    let Some(path)=std::env::var_os("ATOM_MATRIX_EXPORT_PATH")else{return Ok(false);};
    let role=ROLE.with(std::cell::Cell::get);
    if std::env::var("ATOM_MATRIX_EXPORT_ROLE").is_ok_and(|requested|requested!=role){return Ok(false);}
    let dimension=std::env::var("ATOM_MATRIX_EXPORT_DIM").unwrap_or_else(|_|"9216".into()).parse::<usize>()
        .map_err(|_|"Invalid ATOM_MATRIX_EXPORT_DIM")?;
    if columns.len()!=dimension{return Ok(false);}
    let mut state=SESSION.get_or_init(||Mutex::new(None)).lock().map_err(|_|"Matrix probe lock was poisoned")?;
    if state.is_none(){
        let exit=std::env::var("ATOM_MATRIX_EXPORT_EXIT").is_ok_and(|value|value=="1");
        *state=Some(Session::open(&PathBuf::from(path),identity,columns,scales,role,exit)?);
    }
    Ok(state.as_ref().is_some_and(|s|s.identity==identity&&s.count<3))
}

pub(super) fn record(identity:u64,rhs:&[f64],unscaled_rhs:&[f64],solution:&[f64])->Result<(),String>{
    let Some(state)=SESSION.get()else{return Ok(());};
    let exit={
        let mut state=state.lock().map_err(|_|"Matrix probe lock was poisoned")?;
        match state.as_mut(){Some(session)if session.identity==identity=>session.sample(rhs,unscaled_rhs,solution)?,_=>false}
    };
    if exit{eprintln!("Matrix probe flushed three forward RHS/solutions; explicit export exit86.");std::process::exit(86);}
    Ok(())
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn explicit_probe_preserves_balanced_columns_and_three_samples_without_overwrite(){
        let path=std::env::temp_dir().join(format!("atom-matrix-probe-{}-{}.jsonl",std::process::id(),new_identity()));
        let columns=vec![vec![(0,1.),(1,-0.2)],vec![(0,0.1),(1,1.)]];let scales=[2.,5.];
        let mut session=Session::open(&path,17,&columns,&scales,"test-probe",false).unwrap();
        assert!(Session::open(&path,18,&columns,&scales,"test-probe",false).is_err());
        for _ in 0..4{assert!(!session.sample(&[1.,2.],&[2.,10.],&[0.8,2.16]).unwrap());}
        drop(session);let text=std::fs::read_to_string(&path).unwrap();std::fs::remove_file(&path).unwrap();
        let rows:Vec<serde_json::Value>=text.lines().map(|line|serde_json::from_str(line).unwrap()).collect();
        assert_eq!(rows.len(),4);assert_eq!(rows[0]["role"],"test-probe");assert_eq!(rows[0]["columns"],serde_json::json!(columns));assert_eq!(rows[0]["scales"],serde_json::json!(scales));
        for(i,row)in rows.iter().skip(1).enumerate(){assert_eq!(row["preparedId"],17);assert_eq!(row["index"],i+1);assert_eq!(row["rhs"],serde_json::json!([1.,2.]));assert_eq!(row["unscaledRhs"],serde_json::json!([2.,10.]));}
    }
    #[test]
    fn nested_roles_restore_on_the_same_thread(){
        let previous=ROLE.with(std::cell::Cell::get);
        {let _outer=enter_role("trajectory-resolvent");assert_eq!(ROLE.with(std::cell::Cell::get),"trajectory-resolvent");
            {let _inner=enter_role("nested");assert_eq!(ROLE.with(std::cell::Cell::get),"nested");}
            assert_eq!(ROLE.with(std::cell::Cell::get),"trajectory-resolvent");}
        assert_eq!(ROLE.with(std::cell::Cell::get),previous);
    }
}
