import {mkdir,readFile,writeFile,rename} from 'node:fs/promises';
import {dirname} from 'node:path';

/** Bounded, serialized, atomic local records; independent of the solve queue. */
export function failureRecorder(path){
 let queue=Promise.resolve();
 return record=>{const snapshot=structuredClone(record);queue=queue.catch(()=>{}).then(async()=>{
  let previous=[];try{const value=JSON.parse(await readFile(path,'utf8'));if(Array.isArray(value))previous=value.slice(-4);}catch{}
  await mkdir(dirname(path),{recursive:true});await writeFile(path+'.tmp',JSON.stringify([...previous,snapshot],null,2));await rename(path+'.tmp',path);
 });return queue;};
}
