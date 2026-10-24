import {installNativeFetch} from './native-fetch.mjs';
import {installSelfTest} from './self-test.mjs';
import {installChromeTest} from './chrome-test.mjs';

const session=crypto.randomUUID();
const postNative=command=>window.ipc.postMessage(JSON.stringify({...command,session}));
if(window.__atomChromeTest){installChromeTest({postNative});}
else{
  const worker=new Worker(new URL('./desktop-model-worker.js',document.baseURI),{type:'module'});
  const adapter=installNativeFetch({worker,postNative,originalFetch:fetch.bind(globalThis)});
  window.__atomNativeReceive=event=>{if(event.session===session)adapter.receiveNative(event);};
  globalThis.fetch=window.__atomSelfTest?installSelfTest({adapter,postNative}):adapter.fetch;
  window.addEventListener('pagehide',()=>void adapter.dispose(),{once:true});
}
postNative({op:'hello'});
