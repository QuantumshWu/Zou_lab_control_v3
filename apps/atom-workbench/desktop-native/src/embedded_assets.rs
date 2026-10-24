//! Immutable UI/notices served directly from this executable's memory.
//! This module never creates files or directories and never loads a DLL.
use std::{collections::BTreeMap,fs::File,io::{Read,Seek,SeekFrom},path::Path,sync::Arc};
use sha2::{Sha256,Digest};
use serde_json::{Value,json};
const MAGIC:&[u8;16]=b"ATOM-MEMORY-V001";
const FOOTER:u64=96;
pub struct Assets{files:BTreeMap<String,Arc<[u8]>>,identity:String}
impl Assets{
    pub fn get(&self,name:&str)->Option<&[u8]>{self.files.get(name).map(|value|value.as_ref())}
    pub fn identity(&self)->&str{&self.identity}
    pub fn license_text(&self)->String{let mut text=String::from("Atom Workbench — third-party notices\n");for(name,bytes)in &self.files{if name.starts_with("licenses/"){text.push_str(&format!("\n\n===== {name} =====\n"));text.push_str(&String::from_utf8_lossy(bytes));}}text}
    pub fn report(&self)->Value{json!({"storage":"embedded-memory","payloadIdentity":self.identity,"files":self.files.len(),"resourceDirectoriesCreated":false,"numericalResultCache":false})}
    pub fn load(executable:&Path)->Result<Option<Self>,String>{
        let mut file=File::open(executable).map_err(|e|e.to_string())?;let size=file.metadata().map_err(|e|e.to_string())?.len();
        if size<FOOTER{return Ok(None);}
        file.seek(SeekFrom::End(-(FOOTER as i64))).map_err(|e|e.to_string())?;let mut footer=[0u8;96];file.read_exact(&mut footer).map_err(|e|e.to_string())?;
        if &footer[80..]!=MAGIC{return Ok(None);}
        let compressed=u64::from_le_bytes(footer[..8].try_into().unwrap());let index_len=u64::from_le_bytes(footer[8..16].try_into().unwrap());
        let length=compressed.checked_add(index_len).and_then(|x|x.checked_add(FOOTER)).ok_or("Embedded assets length overflow")?;
        if compressed==0||compressed>32*1024*1024||index_len==0||index_len>2*1024*1024||length>size{return Err("Invalid embedded asset bounds".into());}
        let offset=size-length;file.seek(SeekFrom::Start(offset)).map_err(|e|e.to_string())?;
        let mut packed=vec![0;compressed as usize];file.read_exact(&mut packed).map_err(|e|e.to_string())?;
        let mut index=vec![0;index_len as usize];file.read_exact(&mut index).map_err(|e|e.to_string())?;
        if Sha256::digest(&packed).as_slice()!=&footer[16..48]||Sha256::digest(&index).as_slice()!=&footer[48..80]{return Err("Embedded asset checksum failed".into());}
        let manifest:Value=serde_json::from_slice(&index).map_err(|e|e.to_string())?;
        if manifest["schema"]!="atom-memory-exe/1"{return Err("Unsupported in-memory resource schema".into());}
        let entries=manifest["files"].as_array().ok_or("Embedded asset list is missing")?;
        if entries.is_empty()||entries.len()>10000{return Err("Invalid embedded asset count".into());}
        let mut decoder=flate2::read::DeflateDecoder::new(packed.as_slice());let mut total=0u64;let mut files=BTreeMap::new();
        for entry in entries{
            let name=entry["path"].as_str().ok_or("Missing embedded asset path")?;let bytes=entry["bytes"].as_u64().ok_or("Missing embedded asset size")?;
            if name.is_empty()||name.contains('\\')||name.split('/').any(|part|part.is_empty()||part=="."||part=="..")||files.contains_key(name){return Err("Invalid or duplicated embedded asset path".into());}
            total=total.checked_add(bytes).ok_or("Embedded asset size overflow")?;if total>64*1024*1024{return Err("In-memory assets exceed their explicit size budget".into());}
            let mut data=vec![0;bytes as usize];decoder.read_exact(&mut data).map_err(|e|e.to_string())?;
            if entry["sha256"].as_str()!=Some(format!("{:x}",Sha256::digest(&data)).as_str()){return Err(format!("Embedded asset checksum failed: {name}"));}
            files.insert(name.to_owned(),Arc::from(data));
        }
        let mut extra=[0u8;1];if decoder.read(&mut extra).map_err(|e|e.to_string())?!=0{return Err("Unexpected trailing embedded assets".into());}
        for required in ["build.json","app/dist/index.html","app/dist/app.bundle.js","app/dist/desktop-bridge.js","app/dist/desktop-model-worker.js"]{if !files.contains_key(required){return Err(format!("Missing embedded application asset: {required}"));}}
        Ok(Some(Self{files,identity:format!("{:x}",Sha256::digest(&footer[16..80]))}))
    }
}
