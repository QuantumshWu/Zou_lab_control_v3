// The wait for the driver's first compile of the GPU kernels (ui-design.md §6.24): the whole window
// frozen behind one modal card -- its keyboard, its pointer and its focus -- with a moving bar (the
// driver reports no progress, so there is no percentage) and the time so far. Only the native title
// bar stays live: the window still moves and closes.
import {Dialog} from '../../desktop/ui/src/fluent/dialog.js';
import {element,setText} from '../../desktop/ui/src/fluent/dom.js';
import {clockTime} from './format.js';

/** A start that answers sooner shows nothing. From here the kernels' answer from the driver's cache
 * takes about 1 s: the model worker's load, the native worker's start and self-check, the kernels'
 * 0.25 s. Three times that keeps it from flashing on a loaded machine, and costs a real compile only
 * this long unexplained. */
const HOLD_DELAY_MS=3000;
/** How far "通常 1～2 分钟" reaches (measured here: 53-73 s at light load, 109 s at half load, about
 * 200 s at heavy load): past it the card says that this one takes longer. */
const USUAL_COMPILE_MS=120_000;

const TITLE='正在编译 GPU 内核，请稍候';
const WORDS='本版本首次运行或更新显卡驱动后需要编译一次，通常 1～2 分钟，视显卡和机器负载而定。';
const LONGER='比通常久。驱动不报告编译进度，可以继续等待；关闭窗口会中断编译，下次启动重新开始。';

/** Freezes the window from HOLD_DELAY_MS on, until the function it returns is called; that function
 * says whether the window froze. */
export function holdWhileCompiling():()=>boolean{
  const since=performance.now();
  let dialog:Dialog|undefined,tick=0;
  // The modal dialog keeps the page from the keys and the pointer. What is left of their defaults goes
  // too: Escape would close it, Ctrl+A select its words, a right click open the browser's menu, whose
  // Reload would end the compile. The window's own keys (Alt+F4, the system menu) stay.
  const keys=(event:KeyboardEvent):void=>{if(!event.altKey&&!event.metaKey)event.preventDefault();};
  const menu=(event:Event):void=>event.preventDefault();
  const show=():void=>{
    dialog=new Dialog(TITLE);dialog.root.classList.add('compile-hold');
    // A press on the card or the veil moves nothing: Dialog drags by its body.
    dialog.root.addEventListener('pointerdown',event=>event.stopImmediatePropagation(),true);
    const bar=element('div','bar-track');bar.setAttribute('role','progressbar');bar.setAttribute('aria-label',TITLE);bar.append(element('span','bar-fill compile-sweep'));
    const elapsed=element('div','note compile-elapsed'),longer=element('div','dialog-text compile-longer',LONGER);
    dialog.content.append(element('div','dialog-text',WORDS),bar,elapsed,longer);
    // The time so far, each second as it turns.
    const count=():void=>{const ms=performance.now()-since;setText(elapsed,`已用时 ${clockTime(ms)}`);longer.hidden=ms<USUAL_COMPILE_MS;tick=window.setTimeout(count,1000-ms%1000);};
    count();addEventListener('keydown',keys,true);addEventListener('contextmenu',menu,true);void dialog.open();
  };
  const waiting=window.setTimeout(show,HOLD_DELAY_MS);
  return ()=>{clearTimeout(waiting);clearTimeout(tick);removeEventListener('keydown',keys,true);removeEventListener('contextmenu',menu,true);dialog?.close('');return !!dialog;};
}
