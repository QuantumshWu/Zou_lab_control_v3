import {spawnSync} from 'node:child_process';

// v3 supports ordinary Windows PowerShell; PS7 is optional, never implicit.
export function powershellExecutable(){
  for(const name of ['pwsh','powershell.exe','powershell']){
    const probe=spawnSync(name,['-NoProfile','-Command','$PSVersionTable.PSVersion.Major'],{encoding:'utf8',windowsHide:true,timeout:10000});
    if(probe.status===0&&Number(probe.stdout.trim())>=5)return name;
  }
  throw Error('Windows PowerShell 5 or PowerShell 7 is required for the native Windows build.');
}
