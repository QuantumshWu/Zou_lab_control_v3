use std::{borrow::Cow,collections::BTreeMap,fs,io::{BufRead,BufReader,Write},os::windows::{io::AsRawHandle,process::CommandExt},path::{Path,PathBuf},process::{Child,Command,Stdio},sync::{mpsc,Arc},time::Instant};
use serde_json::{Value,json};
use sha2::{Sha256,Digest};
use tao::{dpi::LogicalSize,event::{Event,WindowEvent},event_loop::{ControlFlow,EventLoopBuilder,EventLoopProxy},window::WindowBuilder};
use windows_sys::Win32::{Foundation::{CloseHandle,HANDLE},System::JobObjects::{AssignProcessToJobObject,CreateJobObjectW,SetInformationJobObject,JobObjectExtendedLimitInformation,JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,JOBOBJECT_EXTENDED_LIMIT_INFORMATION},UI::WindowsAndMessaging::{MessageBoxW,MB_ICONERROR,MB_OK}};
use wry::{WebViewBuilder,WebContext,WebViewBuilderExtWindows};
use tao::platform::windows::WindowBuilderExtWindows;

#[derive(Clone)]enum UiEvent{Native(Value),Caption(Value),CaptionReady,ChromeProbe,Title(String),Report(Value)}
enum HostMessage{Ipc(Value),Line{lane:String,instance:u64,line:String},Eof{lane:String,instance:u64,error:Option<String>},Stop}
#[derive(Default)]struct Sessions{current:Option<String>,next_instance:u64}
impl Sessions{
    fn activate(&mut self,session:&str)->bool{let changed=self.current.as_deref()!=Some(session);self.current=Some(session.into());changed}
    fn accepts(&self,session:&str)->bool{self.current.as_deref()==Some(session)}
    fn next(&mut self)->Result<u64,String>{self.next_instance=self.next_instance.checked_add(1).ok_or("Native instance counter exhausted")?;Ok(self.next_instance)}
    fn reader_matches(&self,session:&str,current_instance:u64,event_instance:u64)->bool{self.accepts(session)&&current_instance==event_instance}
}
struct Job(HANDLE);
impl Drop for Job{fn drop(&mut self){unsafe{CloseHandle(self.0);}}}
impl Job{
    fn contain(child:&Child)->Result<Self,String>{
        let handle=unsafe{CreateJobObjectW(std::ptr::null(),std::ptr::null())};if handle.is_null(){return Err(std::io::Error::last_os_error().to_string());}
        let job=Self(handle);let mut info:JOBOBJECT_EXTENDED_LIMIT_INFORMATION=unsafe{std::mem::zeroed()};info.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if unsafe{SetInformationJobObject(handle,JobObjectExtendedLimitInformation,(&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),std::mem::size_of_val(&info)as u32)}==0||unsafe{AssignProcessToJobObject(handle,child.as_raw_handle()as HANDLE)}==0{return Err(std::io::Error::last_os_error().to_string());}Ok(job)
    }
}
struct NativeChild{generation:u64,instance:u64,session:String,child:Child,job:Option<Job>,spawned:Instant,spawn_ms:f64,blas_threads:String,capability_sent:Option<Instant>,warm_sent:Option<Instant>}
impl NativeChild{fn stop(&mut self){self.child.stdin.take();let _=self.child.kill();self.job.take();let _=self.child.wait();}}
impl Drop for NativeChild{fn drop(&mut self){self.stop();}}
fn hash(path:&Path)->Result<String,String>{Ok(format!("{:x}",Sha256::digest(fs::read(path).map_err(|e|e.to_string())?)))}
fn emit(proxy:&EventLoopProxy<UiEvent>,session:&str,mut value:Value){value["session"]=json!(session);let _=proxy.send_event(UiEvent::Native(value));}
fn close_lane(lanes:&mut BTreeMap<String,NativeChild>,lane:&str,proxy:&EventLoopProxy<UiEvent>){
    if let Some(mut native)=lanes.remove(lane){native.stop();emit(proxy,&native.session,json!({"kind":"closed","lane":lane,"generation":native.generation}));}
}
fn close_all(lanes:&mut BTreeMap<String,NativeChild>,proxy:&EventLoopProxy<UiEvent>){let names:Vec<_>=lanes.keys().cloned().collect();for name in names{close_lane(lanes,&name,proxy);}}
fn configure_blas_environment(command:&mut Command,defaults:&BTreeMap<String,String>)->String{
    let raw=defaults.get("ATOM_BLAS_THREADS").filter(|v|!v.is_empty()).map(String::as_str).unwrap_or("1");
    let threads=raw.parse::<u32>().ok().filter(|t|(1..=32).contains(t)).unwrap_or(1).to_string();
    // Private child only. OpenBLAS sizes its pool when it initializes, before
    // Rust's lapack::load applies ATOM_BLAS_THREADS; the same count avoids an
    // idle pool. An invalid ATOM_BLAS_THREADS reaches the worker, which rejects it.
    command.env("ATOM_BLAS_THREADS",raw).env("OPENBLAS_NUM_THREADS",&threads);threads
}

struct Backend{executable:PathBuf,working:PathBuf,manifest:Value,build_hash:String,embedded:bool}
fn supervisor(backend:Backend,data:PathBuf,test_mode:bool,proxy:EventLoopProxy<UiEvent>,receiver:mpsc::Receiver<HostMessage>,sender:mpsc::Sender<HostMessage>){
    let exe=backend.executable;
    let manifest=backend.manifest;
    let identity=hash(&exe);let startup_error=identity.as_ref().err().cloned();let binary_hash=identity.unwrap_or_default();
    let build_hash=backend.build_hash;
    let mut defaults:BTreeMap<String,String>=manifest["solverDefaults"].as_object().into_iter().flat_map(|m|m.iter()).filter_map(|(k,v)|v.as_str().map(|v|(k.clone(),v.to_owned()))).collect();
    for (key,value)in std::env::vars(){if key.starts_with("ATOM_"){defaults.insert(key,value);}}
    if backend.embedded{defaults.remove("ATOM_OPENBLAS_DLL");}
    else{defaults.insert("ATOM_OPENBLAS_DLL".into(),backend.working.join("libscipy_openblas.dll").to_string_lossy().into());}
    let mut lanes=BTreeMap::<String,NativeChild>::new();let mut sessions=Sessions::default();let mut process_ids=Vec::new();let mut captures=Vec::new();
    while let Ok(command)=receiver.recv(){match command{
        HostMessage::Stop=>break,
        HostMessage::Eof{lane,instance,error}=>{
            if lanes.get(&lane).is_some_and(|w|sessions.reader_matches(&w.session,w.instance,instance)){
                if let Some(mut native)=lanes.remove(&lane){native.stop();emit(&proxy,&native.session,json!({"kind":"closed","lane":lane,"generation":native.generation,"error":error}));}
            }
        },
        HostMessage::Line{lane,instance,line}=>{
            let Some(native)=lanes.get(&lane).filter(|w|sessions.reader_matches(&w.session,w.instance,instance))else{continue};
            let session=native.session.clone();let generation=native.generation;
            match serde_json::from_str::<Value>(&line){Ok(message)=>{
                let timing=match message["id"].as_i64(){Some(0)=>native.capability_sent.map(|start|json!({"stage":"capabilities","nativeIoMs":start.elapsed().as_secs_f64()*1000.,"sinceSpawnMs":native.spawned.elapsed().as_secs_f64()*1000.})),Some(-1)=>native.warm_sent.map(|start|json!({"stage":"numerical-ready","nativeIoMs":start.elapsed().as_secs_f64()*1000.,"sinceSpawnMs":native.spawned.elapsed().as_secs_f64()*1000.})),_=>None};
                emit(&proxy,&session,json!({"kind":"message","lane":lane,"generation":generation,"message":message,"transport":timing}));
            },Err(error)=>{emit(&proxy,&session,json!({"kind":"error","lane":lane,"generation":generation,"error":format!("Invalid native JSON: {error}")}));close_lane(&mut lanes,&lane,&proxy);}}
        },
        HostMessage::Ipc(value)=>{
            let op=value["op"].as_str().unwrap_or("");let session=value["session"].as_str().unwrap_or("").to_owned();
            if session.is_empty()||session.len()>96||!session.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'-'){continue;}
            if op=="hello"{
                if sessions.activate(&session){close_all(&mut lanes,&proxy);}
                if let Some(error)=&startup_error{emit(&proxy,&session,json!({"kind":"fatal","error":error}));}else{emit(&proxy,&session,json!({"kind":"configuration","binarySha256":binary_hash,"buildSha256":build_hash,"defaults":defaults}));}continue;
            }
            if !sessions.accepts(&session){continue;}
            if op=="close-all"{close_all(&mut lanes,&proxy);continue;}
            if op=="self-test"&&test_mode{close_all(&mut lanes,&proxy);let mut result=value["result"].clone();result["nativeProcesses"]=json!(process_ids);result["capturedRequests"]=json!(captures);result["nodeRuntime"]=json!(false);result["binarySha256"]=json!(binary_hash);let _=proxy.send_event(UiEvent::Report(result));continue;}
            let lane=value["lane"].as_str().unwrap_or("").to_owned();let generation=value["generation"].as_u64().unwrap_or(0);
            if !["steady","trajectory"].contains(&lane.as_str())||generation==0{continue;}
            if op=="close"{if lanes.get(&lane).is_some_and(|w|w.generation==generation){close_lane(&mut lanes,&lane,&proxy);}else{emit(&proxy,&session,json!({"kind":"closed","lane":lane,"generation":generation}));}continue;}
            if op=="open"{
                if lanes.get(&lane).is_some_and(|w|w.generation>=generation){continue;}
                close_lane(&mut lanes,&lane,&proxy);
                let spawn=(||->Result<NativeChild,String>{
                    let started=Instant::now();
                    if hash(&exe)?!=binary_hash{return Err("Native executable changed after configuration".into());}
                    let instance=sessions.next()?;
                    let mut command=Command::new(&exe);command.current_dir(&backend.working).envs(&defaults).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).creation_flags(0x08000000);
                    if backend.embedded{command.arg("--worker").env_remove("ATOM_OPENBLAS_DLL");}
                    let blas_threads=configure_blas_environment(&mut command,&defaults);
                    let mut child=command.spawn().map_err(|e|e.to_string())?;let spawned=Instant::now();let job=match Job::contain(&child){Ok(j)=>j,Err(e)=>{let _=child.kill();let _=child.wait();return Err(e);}};
                    let stdout=child.stdout.take().unwrap();let stderr=child.stderr.take().unwrap();let tx=sender.clone();let name=lane.clone();
                    std::thread::spawn(move||{let mut reader=BufReader::new(stdout);let mut error=None;loop{let mut line=String::new();match reader.read_line(&mut line){Ok(0)=>break,Ok(_)=>{if line.len()>64*1024*1024{error=Some("Native response exceeded 64 MiB".into());break;}if tx.send(HostMessage::Line{lane:name.clone(),instance,line}).is_err(){return;}},Err(e)=>{error=Some(e.to_string());break;}}}let _=tx.send(HostMessage::Eof{lane:name,instance,error});});
                    let log=data.join(format!("{lane}-native.log"));std::thread::spawn(move||{let mut output=fs::OpenOptions::new().create(true).append(true).open(log).ok();for(index,line)in BufReader::new(stderr).lines().enumerate(){if let Ok(line)=line{if index<10000{if let Some(file)=output.as_mut(){let _=writeln!(file,"{}",line.chars().take(4000).collect::<String>());}}}}});
                    Ok(NativeChild{generation,instance,session:session.clone(),child,job:Some(job),spawned,spawn_ms:started.elapsed().as_secs_f64()*1000.,blas_threads,capability_sent:None,warm_sent:None})
                })();
                match spawn{Ok(native)=>{let pid=native.child.id();process_ids.push(pid);let opened=json!({"kind":"opened","lane":lane,"generation":generation,"instanceId":native.instance,"pid":pid,"binarySha256":binary_hash,"hostSpawnMs":native.spawn_ms,"openblasThreads":native.blas_threads});lanes.insert(lane.clone(),native);emit(&proxy,&session,opened);},Err(error)=>emit(&proxy,&session,json!({"kind":"error","lane":lane,"generation":generation,"error":error}))}
            }else if op=="write"{
                let result=(||->Result<(),String>{let native=lanes.get_mut(&lane).filter(|w|w.generation==generation).ok_or("Native generation is unavailable")?;
                    let message=&value["message"];let mut bytes=serde_json::to_vec(message).map_err(|e|e.to_string())?;if bytes.len()>64*1024*1024{return Err("Native request exceeds 64 MiB".into());}
                    if test_mode&&message["model"]["dimension"]==32{
                        let folder=data.join("native-inputs");fs::create_dir_all(&folder).map_err(|e|e.to_string())?;
                        let file=folder.join(format!("{}-{}-{}.json",lane,native.instance,message["id"]));fs::write(&file,&bytes).map_err(|e|e.to_string())?;
                        captures.push(json!({"lane":lane,"file":file,"sha256":format!("{:x}",Sha256::digest(&bytes))}));
                    }
                    if message["id"]==0{native.capability_sent=Some(Instant::now());}else if message["id"]== -1{native.warm_sent=Some(Instant::now());}
                    bytes.push(b'\n');native.child.stdin.as_mut().ok_or("Native input closed")?.write_all(&bytes).map_err(|e|e.to_string())})();
                if let Err(error)=result{emit(&proxy,&session,json!({"kind":"error","lane":lane,"generation":generation,"error":error}));}
            }
        }
    }}close_all(&mut lanes,&proxy);
}

fn serve(root:&Path,request:&wry::http::Request<Vec<u8>>)->wry::http::Response<Cow<'static,[u8]>>{
    let path=percent_encoding::percent_decode_str(request.uri().path()).decode_utf8_lossy();let relative=if path=="/"{"index.html"}else{path.trim_start_matches('/')};
    let found=root.join(relative).canonicalize().ok().filter(|p|p.starts_with(root)&&p.is_file()).and_then(|p|fs::read(&p).ok().map(|data|(p,data)));
    match found{Some((path,data))=>{let mime=match path.extension().and_then(|s|s.to_str()).unwrap_or(""){"mjs"|"js"=>"text/javascript; charset=utf-8","html"=>"text/html; charset=utf-8","json"=>"application/json","css"=>"text/css; charset=utf-8","svg"=>"image/svg+xml","ico"=>"image/x-icon","woff2"=>"font/woff2","woff"=>"font/woff","ttf"=>"font/ttf",_=>"application/octet-stream"};wry::http::Response::builder().header("Content-Type",mime).header("Cache-Control","no-store").body(data.into()).unwrap()},None=>wry::http::Response::builder().status(404).body(Vec::new().into()).unwrap()}
}
fn serve_memory(assets:&crate::embedded_assets::Assets,request:&wry::http::Request<Vec<u8>>)->wry::http::Response<Cow<'static,[u8]>>{
    let path=percent_encoding::percent_decode_str(request.uri().path()).decode_utf8_lossy();let relative=if path=="/"{"index.html"}else{path.trim_start_matches('/')};
    let name=format!("app/dist/{relative}");
    match assets.get(&name){Some(data)=>{let mime=match Path::new(relative).extension().and_then(|s|s.to_str()).unwrap_or(""){"mjs"|"js"=>"text/javascript; charset=utf-8","html"=>"text/html; charset=utf-8","json"=>"application/json","css"=>"text/css; charset=utf-8","svg"=>"image/svg+xml","ico"=>"image/x-icon","woff2"=>"font/woff2","woff"=>"font/woff","ttf"=>"font/ttf",_=>"application/octet-stream"};wry::http::Response::builder().header("Content-Type",mime).header("Cache-Control","no-store").body(Cow::Owned(data.to_vec())).unwrap()},None=>wry::http::Response::builder().status(404).body(Vec::new().into()).unwrap()}
}
fn runtime(root:Option<&Path>)->Result<(bool,Option<String>),String>{
    if let Some(root)=root{let fixed=root.join("runtime/webview2");if fixed.join("msedgewebview2.exe").is_file(){std::env::set_var("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER",fixed);return Ok((true,None));}}
    for key in ["WEBVIEW2_BROWSER_EXECUTABLE_FOLDER","WEBVIEW2_RELEASE_CHANNEL_PREFERENCE","WEBVIEW2_RELEASE_CHANNELS"]{std::env::remove_var(key);}
    let help="Microsoft Edge WebView2 Evergreen Runtime is required.\nInstall it once from https://developer.microsoft.com/microsoft-edge/webview2/\nThen reopen Atom Workbench. Nothing is downloaded or installed automatically.";
    let version=wry::webview_version().map_err(|_|help.to_owned())?;if version.is_empty()||!version.chars().all(|c|c.is_ascii_digit()||c=='.'){return Err(help.into());}Ok((false,Some(version)))
}
fn run()->Result<(),String>{
    let args:Vec<_>=std::env::args().collect();let argument=|name:&str|args.iter().position(|a|a==name).and_then(|i|args.get(i+1));
    let chrome_test=argument("--chrome-test").is_some();
    let report=argument("--self-test").or_else(||argument("--chrome-test")).map(PathBuf::from);let test_mode=report.is_some();
    if test_mode&&argument("--data-dir").is_none(){return Err("Headless validation requires an isolated --data-dir; saved user scenes will not be modified.".into());}
    let data=argument("--data-dir").map(PathBuf::from).or_else(||std::env::var_os("LOCALAPPDATA").map(|p|PathBuf::from(p).join("AtomWorkbench"))).ok_or("LOCALAPPDATA is missing")?;fs::create_dir_all(&data).map_err(|e|e.to_string())?;
    let executable=std::env::current_exe().map_err(|e|e.to_string())?;
    let embedded=if argument("--root").is_none(){crate::embedded_assets::Assets::load(&executable)?.map(Arc::new)}else{None};
    if let Some(path)=argument("--export-licenses"){let assets=embedded.as_ref().ok_or("Third-party notices are embedded in the single-file distribution")?;fs::write(path,assets.license_text()).map_err(|e|e.to_string())?;return Ok(());}
    let root=if embedded.is_some(){None}else{Some(argument("--root").map(PathBuf::from).unwrap_or(executable.parent().unwrap().to_owned()).canonicalize().map_err(|e|e.to_string())?)};
    if embedded.is_some()&&!cfg!(feature="embedded-solver"){return Err("In-memory distribution requires the statically linked solver.".into());}
    if let Some(path)=argument("--resource-check").or_else(||argument("--bundle-check")){
        let value=json!({"ok":true,"embedded":embedded.as_ref().map(|bundle|bundle.report()),"staticSolver":cfg!(feature="embedded-solver"),"iconResourceAvailable":crate::app_icon::icon().is_some()});
        fs::write(path,serde_json::to_vec_pretty(&value).unwrap()).map_err(|e|e.to_string())?;return Ok(());
    }
    let read_resource=|name:&str|->Result<Vec<u8>,String>{if let Some(assets)=&embedded{assets.get(name).map(|bytes|bytes.to_vec()).ok_or_else(||format!("Missing memory resource: {name}"))}else{fs::read(root.as_ref().unwrap().join(name)).map_err(|e|e.to_string())}};
    let manifest_bytes=read_resource("build.json")?;let manifest:Value=serde_json::from_slice(&manifest_bytes).map_err(|e|format!("Invalid package manifest: {e}"))?;
    if !manifest["solverDefaults"].is_object(){return Err("Application solver settings are missing.".into());}
    if manifest["webview"]=="bundled-fixed"&&!root.as_ref().is_some_and(|path|path.join("runtime/webview2/msedgewebview2.exe").is_file()){return Err("The bundled rendering runtime is missing.".into());}
    let(bundled,version)=runtime(root.as_deref())?;
    let mut init=String::new();
    if chrome_test{
        init="window.__atomChromeTest=true;localStorage.clear();window.requestAnimationFrame=callback=>setTimeout(()=>callback(performance.now()),16);window.cancelAnimationFrame=clearTimeout;".into();
    }else if test_mode{
        let request=argument("--test-request").ok_or("Headless full-app validation requires --test-request <JSON>")?;
        let value:Value=serde_json::from_slice(&fs::read(request).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        let config=json!({"request":value,"runtimeVersion":version,"bundledWebView":bundled,"hiddenRafShim":true});
        init=format!("window.__atomSelfTest={config};localStorage.clear();const testScene=structuredClone(window.__atomSelfTest.request.scene);testScene.dynamics.autoRun=false;localStorage.setItem('rb87-workbench-demo-v4',JSON.stringify(testScene));window.requestAnimationFrame=callback=>setTimeout(()=>callback(performance.now()),16);window.cancelAnimationFrame=clearTimeout;");
    }
    let backend=if cfg!(feature="embedded-solver"){Backend{executable:executable.clone(),working:executable.parent().unwrap().to_owned(),manifest,build_hash:format!("{:x}",Sha256::digest(&manifest_bytes)),embedded:true}}
        else{let folder=root.as_ref().unwrap().join("solver");Backend{executable:folder.join("atom-lindblad.exe"),working:folder,manifest,build_hash:format!("{:x}",Sha256::digest(&manifest_bytes)),embedded:false}};
    let event_loop=EventLoopBuilder::<UiEvent>::with_user_event().build();let proxy=event_loop.create_proxy();let(tx,rx)=mpsc::channel();let actor_sender=tx.clone();let actor_data=data.clone();
    let mut actor=Some(std::thread::spawn(move||supervisor(backend,actor_data,test_mode,proxy,rx,actor_sender)));
    let window=WindowBuilder::new().with_title("Rb87 实验工作台 · Atom Workbench").with_window_icon(crate::chrome::icon()).with_taskbar_icon(crate::chrome::icon()).with_inner_size(LogicalSize::new(1600.,1000.)).with_visible(false).build(&event_loop).map_err(|e|e.to_string())?;
    let caption_proxy=event_loop.create_proxy();
    crate::frame::install(&window,move|caption|{let _=caption_proxy.send_event(UiEvent::Caption(caption));})?;
    // Preserve Atom Workbench's large-screen startup policy. The shared frame
    // handles the caption/edges; it does not rescale the application content.
    if !test_mode{window.set_maximized(true);}
    let caption=crate::frame::caption(&window).unwrap_or_else(||json!({"hover":null,"pressed":null,"maximized":false}));
    init.push_str(&format!("window.__atomWindowChrome=true;window.__atomCaption={caption};window.__atomCaptionReceive=value=>{{window.__atomCaption=value;window.dispatchEvent(new CustomEvent('atom-caption',{{detail:value}}));}};"));
    let native_icons=crate::app_icon::window_icons_available(&window);let embedded_report=embedded.as_ref().map(|bundle|bundle.report());
    let mut context=WebContext::new(Some(data.join("WebView2")));let files=root.as_ref().map(|root|root.join("app/dist").canonicalize()).transpose().map_err(|e|e.to_string())?;let ipc=tx.clone();
    let chrome_proxy=event_loop.create_proxy();let title_proxy=event_loop.create_proxy();
    let webview=WebViewBuilder::new_with_web_context(&mut context).with_custom_protocol("atom".into(),move|_,request|if let Some(assets)=&embedded{serve_memory(assets,&request)}else{serve(files.as_ref().unwrap(),&request)})
        .with_ipc_handler(move|request|{if request.body().len()<=64*1024*1024{if let Ok(value)=serde_json::from_str::<Value>(request.body()){
            if value["op"]=="caption-ready"{let _=chrome_proxy.send_event(UiEvent::CaptionReady);}
            else if chrome_test&&value["op"]=="chrome-probe"{let _=chrome_proxy.send_event(UiEvent::ChromeProbe);}
            else{let _=ipc.send(HostMessage::Ipc(value));}
        }}})
        .with_document_title_changed_handler(move|title|{let _=title_proxy.send_event(UiEvent::Title(title));})
        .with_navigation_handler(|url|url.starts_with("atom://localhost/")||url.starts_with("http://atom.localhost/")||url.starts_with("blob:http://atom.localhost/"))
        .with_new_window_req_handler(|_,_|wry::NewWindowResponse::Deny)
        .with_additional_browser_args("--force-color-profile=srgb --disable-features=msWebOOUI,msPdfOOUI,DeferRendererTasksAfterInput")
        .with_browser_accelerator_keys(false).with_devtools(false).with_clipboard(true)
        .with_initialization_script(&init).with_url("atom://localhost/index.html").build(&window).map_err(|e|e.to_string())?;
    crate::frame::raise(&window);window.set_visible(!test_mode);
    if test_mode{let timeout=event_loop.create_proxy();std::thread::spawn(move||{std::thread::sleep(std::time::Duration::from_secs(45));let _=timeout.send_event(UiEvent::Report(json!({"ok":false,"error":"Hidden full-app test timed out"})));});}
    event_loop.run(move|event,_,flow|{*flow=ControlFlow::Wait;match event{
        Event::UserEvent(UiEvent::Native(value))=>{let _=webview.evaluate_script(&format!("window.__atomNativeReceive&&window.__atomNativeReceive({});",value));},
        Event::UserEvent(UiEvent::Caption(value))=>{let _=webview.evaluate_script(&format!("window.__atomCaptionReceive&&window.__atomCaptionReceive({value});"));},
        Event::UserEvent(UiEvent::CaptionReady)=>{crate::frame::raise(&window);if let Some(value)=crate::frame::caption(&window){let _=webview.evaluate_script(&format!("window.__atomCaptionReceive&&window.__atomCaptionReceive({value});"));}},
        Event::UserEvent(UiEvent::ChromeProbe)=>{let value=crate::chrome_probe::inspect(&window);let _=webview.evaluate_script(&format!("window.__atomChromeProbeReceive&&window.__atomChromeProbeReceive({value});"));},
        Event::UserEvent(UiEvent::Title(title))=>window.set_title(&title),
        Event::UserEvent(UiEvent::Report(mut value))=>{let _=tx.send(HostMessage::Stop);if let Some(actor)=actor.take(){let _=actor.join();}value["nativeIcons"]=json!({"large":native_icons.0,"small":native_icons.1});value["embeddedResources"]=json!(embedded_report);if let Some(path)=&report{let _=fs::write(path,serde_json::to_vec_pretty(&value).unwrap());}*flow=ControlFlow::Exit;},
        Event::WindowEvent{event:WindowEvent::CloseRequested,..}=>{let _=tx.send(HostMessage::Stop);if let Some(actor)=actor.take(){let _=actor.join();}*flow=ControlFlow::Exit;},
        Event::LoopDestroyed=>{let _=tx.send(HostMessage::Stop);if let Some(actor)=actor.take(){let _=actor.join();}},_=>{}
    }});
}
pub fn main(){if let Err(error)=run(){let args:Vec<_>=std::env::args().collect();if let Some(i)=args.iter().position(|a|a=="--self-test"||a=="--chrome-test"||a=="--bundle-check"||a=="--resource-check"){if let Some(file)=args.get(i+1){let _=fs::write(file,json!({"ok":false,"error":error}).to_string());}}else{let text:Vec<_>=error.encode_utf16().chain(Some(0)).collect();let title:Vec<_>="Atom Workbench".encode_utf16().chain(Some(0)).collect();unsafe{MessageBoxW(std::ptr::null_mut(),text.as_ptr(),title.as_ptr(),MB_OK|MB_ICONERROR);}}std::process::exit(1);}}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn page_reload_and_late_reader_events_cannot_target_a_new_native_instance(){
        let mut sessions=Sessions::default();assert!(sessions.activate("page-a"));let old=sessions.next().unwrap();assert!(sessions.reader_matches("page-a",old,old));
        assert!(sessions.activate("page-b"));let new=sessions.next().unwrap();assert_ne!(old,new);
        assert!(!sessions.accepts("page-a")); // old close-all/open/write rejected
        assert!(!sessions.reader_matches("page-b",new,old)); // generation/PID may be reused; instance is not
        assert!(!sessions.reader_matches("page-a",old,old));
        assert!(sessions.reader_matches("page-b",new,new));assert!(!sessions.activate("page-b"));
    }
}
