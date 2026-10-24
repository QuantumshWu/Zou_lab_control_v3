param([ValidateSet('build','check','test')][string]$Action='build',[string]$TestFilter='',[switch]$EmbeddedSolver,[switch]$RestoreDependencies)
$ErrorActionPreference='Stop'
$taskApp=Split-Path -Parent $PSScriptRoot
$taskRepository=Split-Path -Parent (Split-Path -Parent $taskApp)
$taskCommon=(& git -C $taskRepository rev-parse --path-format=absolute --git-common-dir).Trim()
$taskTools=if($env:ZLC_V3_TOOLS){$env:ZLC_V3_TOOLS}else{Join-Path (Split-Path -Parent $taskCommon) '.tools'}
$taskRust=Join-Path $taskTools 'rust-1.99.1-x86_64-pc-windows-gnu'
$taskNames=@('PATH','RUSTC','RUSTFLAGS','CARGO_HOME','CARGO_TARGET_DIR','CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER','LIB')
$taskSaved=@{};foreach($taskName in $taskNames){$taskSaved[$taskName]=[Environment]::GetEnvironmentVariable($taskName,'Process')}
try {
  $env:PATH="$taskRust\bin;$taskRust\lib\rustlib\x86_64-pc-windows-gnu\bin;$env:PATH"
  $env:RUSTC=Join-Path $taskRust 'bin\rustc.exe'
  $env:CARGO_HOME=Join-Path $taskTools 'cargo-home'
  $env:CARGO_TARGET_DIR=if($env:ATOM_NATIVE_TARGET_DIR){$env:ATOM_NATIVE_TARGET_DIR}else{Join-Path $taskApp 'desktop-native\target'}
  $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=Join-Path $taskTools 'lld-link-1.99.1\lld-link.exe'
  $env:LIB=(@('crt\lib\x86_64','sdk\lib\um\x86_64','sdk\lib\ucrt\x86_64') | ForEach-Object {Join-Path (Join-Path $taskTools 'msvc-sdk') $_}) -join ';'
  # The launcher does not require a separately installed MSVC redistributable.
  $env:RUSTFLAGS='-C target-feature=+crt-static'
  $taskCargo=Join-Path $taskRust 'bin\cargo.exe'
  if($RestoreDependencies){
    $taskFetch=@('fetch','--manifest-path',(Join-Path $taskApp 'desktop-native\Cargo.toml'),'--target','x86_64-pc-windows-msvc','--locked')
    # Cargo reuses its checksum-verified registry cache; --locked prevents an
    # implicit version update and missing sources are restored on a new PC.
    & $taskCargo @taskFetch
    if($LASTEXITCODE -ne 0){throw 'Pinned Cargo dependency restore failed'}
  }
  $taskArguments=@($Action,'--manifest-path',(Join-Path $taskApp 'desktop-native\Cargo.toml'),'--target','x86_64-pc-windows-msvc','--release','--locked','--offline')
  if($EmbeddedSolver){$taskArguments+=@('--features','embedded-solver')}
  if($Action -eq 'test' -and $TestFilter){$taskArguments+=$TestFilter}
  & $taskCargo @taskArguments
  if($LASTEXITCODE -ne 0){throw "Desktop $Action failed: $LASTEXITCODE"}
}finally{foreach($taskName in $taskNames){[Environment]::SetEnvironmentVariable($taskName,$taskSaved[$taskName],'Process')}}
