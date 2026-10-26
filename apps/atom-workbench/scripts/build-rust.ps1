param([ValidateSet('build','test')][string]$Action='build',[string]$TestFilter='', [switch]$IncludeIgnored,[ValidateSet('release','iterate')][string]$Profile='release',[switch]$CvodeResearch,[switch]$FilonResearch,[switch]$EeResearch)
$ErrorActionPreference='Stop'
$taskApp=Split-Path -Parent $PSScriptRoot
$taskRepository=Split-Path -Parent (Split-Path -Parent $taskApp)
$taskCommon=(& git -C $taskRepository rev-parse --path-format=absolute --git-common-dir).Trim()
$taskTools=if($env:ZLC_V3_TOOLS){$env:ZLC_V3_TOOLS}else{Join-Path (Split-Path -Parent $taskCommon) '.tools'}
$taskRust=Join-Path $taskTools 'rust-1.99.1-x86_64-pc-windows-gnu'
$taskNames=@('PATH','RUSTC','CARGO_HOME','CARGO_TARGET_DIR','CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER','LIB','ATOM_OPENBLAS_DLL')
$taskSaved=@{};foreach($taskName in $taskNames){$taskSaved[$taskName]=[Environment]::GetEnvironmentVariable($taskName,'Process')}
try {
  $env:PATH="$taskRust\bin;$taskRust\lib\rustlib\x86_64-pc-windows-gnu\bin;$env:PATH"
  $env:RUSTC=Join-Path $taskRust 'bin\rustc.exe'
  $env:CARGO_HOME=Join-Path $taskTools 'cargo-home'
  $env:CARGO_TARGET_DIR=if($env:ATOM_CARGO_TARGET_DIR){[IO.Path]::GetFullPath($env:ATOM_CARGO_TARGET_DIR)}else{Join-Path $taskApp 'rust-solver\target'}
  $taskDll=Join-Path $env:CARGO_TARGET_DIR 'x86_64-pc-windows-msvc\release\libscipy_openblas.dll'
  if(-not $env:ATOM_OPENBLAS_DLL -and (Test-Path -LiteralPath $taskDll)){$env:ATOM_OPENBLAS_DLL=$taskDll}
  $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=Join-Path $taskTools 'lld-link-1.99.1\lld-link.exe'
  $env:LIB=(@('crt\lib\x86_64','sdk\lib\um\x86_64','sdk\lib\ucrt\x86_64') | ForEach-Object {Join-Path (Join-Path $taskTools 'msvc-sdk') $_}) -join ';'
  $taskArgs=@($Action,'--manifest-path',(Join-Path $taskApp 'rust-solver\Cargo.toml'),'--target','x86_64-pc-windows-msvc','--profile',$Profile,'--offline')
  if($CvodeResearch){$taskArgs+=@('--features','cvode-research')}
  if($FilonResearch){$taskArgs+=@('--features','filon-research')}
  if($EeResearch){$taskArgs+=@('--features','ee-research')}
  if($Action -eq 'test' -and $TestFilter){$taskArgs+=$TestFilter}
  if($Action -eq 'test' -and $IncludeIgnored){$taskArgs+=@('--','--ignored','--nocapture')}
  & (Join-Path $taskRust 'bin\cargo.exe') @taskArgs
  if($LASTEXITCODE -ne 0){throw "Rust $Action failed: $LASTEXITCODE"}
}
finally{foreach($taskName in $taskNames){[Environment]::SetEnvironmentVariable($taskName,$taskSaved[$taskName],'Process')}}
