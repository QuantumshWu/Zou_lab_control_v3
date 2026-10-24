import {createHash} from 'node:crypto';
import {inflateRawSync} from 'node:zlib';
const sha=data=>createHash('sha256').update(data).digest();

// Recognize OUR verified generated EXE before replacing a root-level file.
export function isGeneratedAtomExe(bytes){
  try{
    if(bytes.length<160||bytes.subarray(0,2).toString()!=='MZ'||bytes.subarray(-16).toString('ascii')!=='ATOM-MEMORY-V001')return false;
    const footer=bytes.subarray(-96),payloadLength=Number(footer.readBigUInt64LE(0)),indexLength=Number(footer.readBigUInt64LE(8));
    if(!Number.isSafeInteger(payloadLength)||!Number.isSafeInteger(indexLength)||payloadLength<=0||indexLength<=0||payloadLength+indexLength+96>bytes.length-64)return false;
    const start=bytes.length-96-payloadLength-indexLength,payload=bytes.subarray(start,start+payloadLength),indexBytes=bytes.subarray(start+payloadLength,-96);
    if(!sha(payload).equals(footer.subarray(16,48))||!sha(indexBytes).equals(footer.subarray(48,80)))return false;
    const index=JSON.parse(indexBytes.toString());if(index.schema!=='atom-memory-exe/1'||!Array.isArray(index.files)||index.files.length>8192)return false;
    const resources=inflateRawSync(payload,{maxOutputLength:64*1024*1024});let offset=0,build=null;
    for(const entry of index.files){
      if(!Number.isSafeInteger(entry.bytes)||entry.bytes<0||offset+entry.bytes>resources.length)return false;
      const data=resources.subarray(offset,offset+entry.bytes);offset+=entry.bytes;
      if(entry.path==='build.json'){
        if(build||sha(data).toString('hex')!==entry.sha256)return false;build=JSON.parse(data.toString());
      }
    }
    return offset===resources.length&&build?.name==='Atom Workbench'&&build.distribution==='single-exe-memory'&&build.numericBackend==='same-executable-static';
  }catch{return false;}
}
