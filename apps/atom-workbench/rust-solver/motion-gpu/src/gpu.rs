//! The engine's GPU: the device it runs on, its context, the loaded kernels and a stream.
use crate::driver::{self,Context,DeviceInfo,Function,Module,Stream};
use std::cell::RefCell;
use std::collections::HashMap;

/// The oldest GPUs the PTX targets: compute capability 7.5 (Turing).
pub const MINIMUM_CAPABILITY:(i32,i32)=(7,5);

/// Why there is no GPU to run on, for the operator (Chinese, as the app shows it).
pub fn unavailable(reason:&str)->String{
    format!("2D 径向运动需要 NVIDIA 显卡（Turing 架构 / 计算能力 7.5 或更新）与 CUDA 12 驱动（Windows 527.41 或更新）：{reason}")
}

pub struct Gpu{
    pub stream:Stream,
    functions:RefCell<HashMap<String,Function>>,
    _module:Module,
    pub context:Context,
}
impl Gpu{
    /// The GPU of the most multiprocessors among those new enough, its kernels loaded.
    pub fn new()->Result<Gpu,String>{
        let devices=driver::devices().map_err(|e|unavailable(&e))?;
        let device=devices.iter().filter(|d|d.compute_capability>=MINIMUM_CAPABILITY).max_by_key(|d|d.multiprocessors)
            .ok_or_else(||unavailable(&if devices.is_empty(){"没有找到 CUDA 设备".to_string()}else{
                format!("{} 太旧",devices.iter().map(|d|format!("{}（计算能力 {}.{}）",d.name,d.compute_capability.0,d.compute_capability.1)).collect::<Vec<_>>().join("、"))}))?;
        Gpu::on(device)
    }
    pub fn on(device:&DeviceInfo)->Result<Gpu,String>{
        let context=Context::new(device)?;
        let module=context.load(crate::ptx::PTX).map_err(|e|unavailable(&format!("驱动无法编译引擎的内核（驱动可能太旧）：{e}")))?;
        Ok(Gpu{stream:Stream::new()?,functions:RefCell::new(HashMap::new()),_module:module,context})
    }
    /// A kernel by name, looked up once.
    pub fn function(&self,name:&str)->Result<Function,String>{
        if let Some(f)=self.functions.borrow().get(name){return Ok(f.clone());}
        let f=self._module.function(name)?;self.functions.borrow_mut().insert(name.to_string(),f.clone());Ok(f)
    }
}
