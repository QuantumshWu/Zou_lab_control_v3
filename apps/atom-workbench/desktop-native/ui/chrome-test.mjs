// Only enabled by the native --chrome-test switch in an isolated hidden window.
export function installChromeTest({postNative}){
  const original=fetch.bind(globalThis);let intercepted=0;
  globalThis.fetch=(input,options)=>String(input).includes('/api/')?(intercepted++,Promise.resolve(new Response('{"error":"Window chrome validation does not run the numerical solver"}',{status:503,headers:{'content-type':'application/json'}}))):original(input,options);
  const assert=(condition,message)=>{if(!condition)throw Error(message);};
  window.__atomChromeProbeReceive=async native=>{
    let result;
    try{
      const bar=document.querySelector('.title-bar'),app=document.querySelector('#app'),rect=bar?.getBoundingClientRect();
      assert(bar&&rect,'Shared title bar was not mounted');
      assert(!native.visible&&native.hitWindows===4,'Native probe was not hidden or lacks the shared hit windows');
      assert(native.captionStyle&&native.resizeStyle,'Native caption/system/resize styles were removed');
      for(const [name,code]of Object.entries({caption:2,minimize:8,maximize:9,close:20,topLeft:13,bottom:15}))assert(native.hits[name]===code,'Incorrect native hit test: '+name);
      assert(Math.abs(rect.height*devicePixelRatio-native.barDeviceHeight)<1,'Native and CSS title bar heights differ');
      assert(Math.abs(app.getBoundingClientRect().top-rect.bottom)<1,'Main content does not begin below the title bar');
      assert(Math.abs(app.getBoundingClientRect().bottom-innerHeight)<1,'Main content extends below the client area');
      window.__atomCaptionReceive(native.hover);assert(bar.querySelector('.maximize').classList.contains('hover'),'Native hover was not painted');
      window.__atomCaptionReceive(native.pressed);assert(bar.querySelector('.maximize').classList.contains('pressed'),'Native press was not painted');
      window.__atomCaptionReceive({...native.clear,maximized:true});assert(bar.classList.contains('maximized'),'Shared restore glyph state missing');
      window.__atomCaptionReceive(native.clear);
      const mark=bar.querySelector('img');await mark.decode();assert(mark.naturalWidth>0,'v3 icon failed to load');
      result={ok:true,native,dom:{barHeight:rect.height,contentTop:app.getBoundingClientRect().top,title:bar.querySelector('.title-bar-title').textContent,buttons:bar.querySelectorAll('.title-bar-button').length,dpr:devicePixelRatio},interceptedSolveRequests:intercepted};
    }catch(error){result={ok:false,error:String(error.stack??error),native};}
    postNative({op:'self-test',result});
  };
  const begin=()=>requestAnimationFrame(()=>requestAnimationFrame(()=>postNative({op:'chrome-probe'})));
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',begin,{once:true});else begin();
}
