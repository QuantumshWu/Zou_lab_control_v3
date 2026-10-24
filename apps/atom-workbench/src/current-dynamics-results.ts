/** Display state for one current physical input, never a history lookup. */
export class CurrentDynamicsResults<T>{
  private key?:string;private epoch=0;
  private mean?:T;private trajectory?:T;private progress?:T;
  activate(key:string):number{
    if(this.key!==key){this.clear();this.key=key;}
    return this.epoch;
  }
  clear():void{this.epoch++;this.key=undefined;this.mean=this.trajectory=this.progress=undefined;}
  set(key:string,epoch:number,kind:'mean'|'trajectory'|'progress',value:T):boolean{
    if(key!==this.key||epoch!==this.epoch)return false;
    this[kind]=value;if(kind==='trajectory')this.progress=undefined;return true;
  }
  get(key:string,kind:'mean'|'trajectory'|'progress'):T|undefined{return key===this.key?this[kind]:undefined;}
}
