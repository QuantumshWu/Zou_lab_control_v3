/** Results for one current input key, never a history lookup: other inputs drop every slot, and a
 * late result for inputs left and come back to cannot return (its epoch is gone). The app keeps
 * two: the trajectory of the trajectory inputs, and the means of the mean inputs, one slot per
 * time T. It holds final results only: a streamed checkpoint is stored only when 停止 (or a
 * failure) makes it the result. */
export class CurrentResults<S,T>{
  private key?:string;private epoch=0;private readonly slots=new Map<S,T>();
  activate(key:string):number{
    if(this.key!==key){this.clear();this.key=key;}
    return this.epoch;
  }
  clear():void{this.epoch++;this.key=undefined;this.slots.clear();}
  set(key:string,epoch:number,slot:S,value:T):boolean{
    if(key!==this.key||epoch!==this.epoch)return false;
    this.slots.set(slot,value);return true;
  }
  get(key:string,slot:S):T|undefined{return key===this.key?this.slots.get(slot):undefined;}
}
