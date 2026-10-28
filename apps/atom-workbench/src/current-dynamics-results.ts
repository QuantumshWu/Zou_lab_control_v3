/** Display state for one current physical input, never a history lookup. It holds final results
 * only: a streamed checkpoint is drawn only when 停止 (or a failure) makes it the result. */
export class CurrentDynamicsResults<T>{
  private key?:string;private epoch=0;
  private mean?:T;private trajectory?:T;
  activate(key:string):number{
    if(this.key!==key){this.clear();this.key=key;}
    return this.epoch;
  }
  clear():void{this.epoch++;this.key=undefined;this.mean=this.trajectory=undefined;}
  set(key:string,epoch:number,kind:'mean'|'trajectory',value:T):boolean{
    if(key!==this.key||epoch!==this.epoch)return false;
    this[kind]=value;return true;
  }
  get(key:string,kind:'mean'|'trajectory'):T|undefined{return key===this.key?this[kind]:undefined;}
}

/** What the last checkpoint of a stopped (or failed) solve stands as: the trajectory when it is one,
 * and the mean when it computed one and no final mean is here. */
export function keptCheckpoint(checkpoint:{timesUs:readonly number[];steady:{converged:boolean}},steadyOnly:boolean,finalMean:boolean):('trajectory'|'mean')[]{
  return [...(!steadyOnly&&checkpoint.timesUs.length>1?['trajectory' as const]:[]),...(checkpoint.steady.converged&&!finalMean?['mean' as const]:[])];
}
