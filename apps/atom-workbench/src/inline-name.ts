// A name edited where it stands: a click (or Enter, F2) on it turns it into a v3 line edit of
// the same box; Enter or leaving keeps the new name, Escape keeps the old one. The owner
// accepts a name or answers why not.
import {element} from '../../desktop/ui/src/fluent/dom.js';
import {setMiddle} from '../../desktop/ui/src/fluent/elide.js';

export class InlineName {
  readonly root:HTMLSpanElement;
  private readonly text=element('span','inline-name-text');
  private readonly edit=element('input','inline-name-edit');
  private name='';
  /** `rename` answers a problem for a refused name, nothing when it is taken. */
  constructor(private readonly rename:(name:string)=>string|void,private readonly what:string,className=''){
    this.root=element('span',`inline-name ${className}`.trim());
    this.text.tabIndex=0;this.text.setAttribute('role','button');
    this.edit.type='text';this.edit.spellcheck=false;this.edit.hidden=true;this.edit.setAttribute('aria-label',what);
    this.root.append(this.text,this.edit);
    this.text.addEventListener('click',()=>this.begin());
    this.text.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key==='F2'){event.preventDefault();this.begin();}});
    this.edit.addEventListener('keydown',event=>{
      if(event.key==='Enter'){event.preventDefault();this.finish(true);}
      if(event.key==='Escape'){event.preventDefault();event.stopPropagation();this.finish(false);}
    });
    this.edit.addEventListener('blur',()=>this.finish(true));
    this.edit.addEventListener('input',()=>{this.edit.removeAttribute('aria-invalid');this.edit.title='';});
  }
  get editing():boolean{return !this.edit.hidden;}
  set value(name:string){
    this.name=name;this.text.title=`${name}（点击重命名）`;this.text.setAttribute('aria-label',`${this.what}：${name}，点击重命名`);
    setMiddle(this.text,name,'cut');
    if(!this.editing)this.edit.value=name;
  }
  private begin():void{
    if(this.editing)return;
    // The edit takes the name's box: at least as wide as the name, never wider than its slot.
    this.edit.style.width=`${Math.max(this.text.getBoundingClientRect().width,this.root.parentElement?Math.min(this.root.parentElement.clientWidth*.6,160):160)}px`;
    this.edit.value=this.name;this.text.hidden=true;this.edit.hidden=false;this.edit.focus();this.edit.select();
  }
  private finish(keep:boolean):void{
    if(!this.editing)return;
    const name=this.edit.value.trim();
    if(keep&&name!==this.name){
      const problem=this.rename(name);
      if(problem){this.edit.setAttribute('aria-invalid','true');this.edit.title=problem;if(document.activeElement!==this.edit)this.close();return;}
      this.value=name;
    }
    this.close();
  }
  private close():void{this.edit.hidden=true;this.text.hidden=false;this.edit.removeAttribute('aria-invalid');this.edit.title='';this.edit.value=this.name;}
}
