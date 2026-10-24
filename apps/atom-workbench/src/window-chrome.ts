import {TitleBar,type CaptionState} from '../../desktop/ui/src/title_bar.js';

type NativeWindow=Window & {__atomWindowChrome?:boolean;__atomCaption?:CaptionState;ipc?:{postMessage(message:string):void}};

/** The shared v3 title bar only paints native frame state. All hit tests,
 * dragging, resizing and system commands remain in the same v3 frame.rs. */
export function installWindowChrome():void{
  const host=window as NativeWindow;if(!host.__atomWindowChrome)return;
  const bar=new TitleBar();document.body.classList.add('atom-native-window');document.body.prepend(bar.root);
  if(host.__atomCaption)bar.accept(host.__atomCaption);
  window.addEventListener('atom-caption',event=>bar.accept((event as CustomEvent<CaptionState>).detail));
  host.ipc?.postMessage(JSON.stringify({op:'caption-ready'}));
}
