param([ValidateSet('configure','build')][string]$Action='build',[ValidateRange(1,8)][int]$Jobs=2)
$ErrorActionPreference='Stop'
$taskApp=Split-Path -Parent $PSScriptRoot
$taskPathsFile=Join-Path $taskApp '.cache/native/paths.json'
$taskLock=Join-Path $taskApp 'native/dependencies.lock.json'
$taskLockHash=(Get-FileHash -LiteralPath $taskLock -Algorithm SHA256).Hash.ToLowerInvariant()
$taskLockedInputs=Get-Content -LiteralPath $taskLock -Raw | ConvertFrom-Json
$taskPaths=if(Test-Path -LiteralPath $taskPathsFile){Get-Content -LiteralPath $taskPathsFile -Raw | ConvertFrom-Json}else{$null}
if(-not $taskPaths -or $taskPaths.lockSha256 -ne $taskLockHash -or [IO.Path]::GetFullPath($taskPaths.appRoot) -ne [IO.Path]::GetFullPath($taskApp)){
 & node (Join-Path $PSScriptRoot 'prepare-native-deps.mjs')
 if($LASTEXITCODE){throw 'Pinned native dependency preparation failed'}
 $taskPaths=Get-Content -LiteralPath $taskPathsFile -Raw | ConvertFrom-Json
}
$taskRoot=$taskPaths.buildRoot
$taskLibRoot=$taskPaths.libDir
$taskZig=$taskPaths.tools.zig
$taskCmake=$taskPaths.tools.cmake
$taskNinja=$taskPaths.tools.ninja
$taskSource=$taskPaths.openblasSource
$taskRepository=Split-Path -Parent (Split-Path -Parent $taskApp)
$taskCommon=(& git -C $taskRepository rev-parse --path-format=absolute --git-common-dir).Trim()
$taskTools=if($env:ZLC_V3_TOOLS){$env:ZLC_V3_TOOLS}else{Join-Path (Split-Path -Parent $taskCommon) '.tools'}
$taskSdk=Join-Path $taskTools 'msvc-sdk'
foreach($taskFile in @($taskZig,$taskCmake,$taskNinja,(Join-Path $taskSource 'CMakeLists.txt'))){if(-not(Test-Path -LiteralPath $taskFile)){throw "Required pinned build input is missing: $taskFile"}}
New-Item -ItemType Directory -Force $taskRoot,$taskLibRoot | Out-Null
$taskLibc=Join-Path $taskRoot 'msvc-libc.txt'
Set-Content -LiteralPath $taskLibc -Encoding ascii -Value @("include_dir=$taskSdk\sdk\include\ucrt","sys_include_dir=$taskSdk\crt\include","crt_dir=$taskSdk\sdk\lib\ucrt\x86_64","msvc_lib_dir=$taskSdk\crt\lib\x86_64","kernel32_lib_dir=$taskSdk\sdk\lib\um\x86_64",'gcc_dir=','darwin_sdk_dir=')
# Native x64 MSVC ABI and static CRT. The source tree also contains portable C
# LAPACK, so no Fortran runtime, Python, Node or sidecar DLL is required at run.
$taskCompiler=Join-Path $taskRoot 'cc.cmd'
Set-Content -LiteralPath $taskCompiler -Encoding ascii -Value @('@echo off',('"'+$taskZig+'" cc -target x86_64-windows-msvc -fms-runtime-lib=static %*'))
$taskAr=Join-Path $taskRoot 'ar.cmd'
Set-Content -LiteralPath $taskAr -Encoding ascii -Value @('@echo off',('"'+$taskZig+'" ar %*'))
$taskRanlib=Join-Path $taskRoot 'ranlib.cmd'
Set-Content -LiteralPath $taskRanlib -Encoding ascii -Value @('@echo off',('"'+$taskZig+'" ranlib %*'))
$taskRc=Join-Path $taskRoot 'rc.cmd'
Set-Content -LiteralPath $taskRc -Encoding ascii -Value @('@echo off',('"'+$taskZig+'" rc %*'))
$taskToolchain=Join-Path $taskRoot 'toolchain.cmake'
$taskForward={param($value) $value.Replace('\','/')}
Set-Content -LiteralPath $taskToolchain -Encoding utf8 -Value @(
 'set(CMAKE_SYSTEM_NAME Windows)','set(CMAKE_SYSTEM_PROCESSOR AMD64)',
 ('set(CMAKE_C_COMPILER "'+(& $taskForward $taskCompiler)+'")'),
 ('set(CMAKE_ASM_COMPILER "'+(& $taskForward $taskCompiler)+'")'),
 ('set(CMAKE_AR "'+(& $taskForward $taskAr)+'")'),
 ('set(CMAKE_RANLIB "'+(& $taskForward $taskRanlib)+'")'),
 ('set(CMAKE_RC_COMPILER "'+(& $taskForward $taskRc)+'")'),
 'set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)',
 'set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <OBJECTS>")',
 'set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")')
$taskSaved=@{};foreach($taskName in @('ZIG_LIBC','ZIG_GLOBAL_CACHE_DIR','LIB')){$taskSaved[$taskName]=[Environment]::GetEnvironmentVariable($taskName,'Process')}
try{
 $env:ZIG_LIBC=$taskLibc;$env:ZIG_GLOBAL_CACHE_DIR=Join-Path $taskRoot 'zig-cache'
 $env:LIB=(@('crt/lib/x86_64','sdk/lib/ucrt/x86_64','sdk/lib/um/x86_64')|ForEach-Object {Join-Path $taskSdk $_})-join ';'
 & $taskCmake -S $taskSource -B (Join-Path $taskRoot 'openblas-build') -G Ninja "-DCMAKE_MAKE_PROGRAM=$taskNinja" "-DCMAKE_TOOLCHAIN_FILE=$taskToolchain" @($taskLockedInputs.build.openblasCmakeOptions)
 if($LASTEXITCODE){throw 'Native static OpenBLAS configuration failed'}
 if($Action -eq 'configure'){return}
 & $taskCmake --build (Join-Path $taskRoot 'openblas-build') --target openblas_static --parallel $Jobs
 if($LASTEXITCODE){throw 'Native static OpenBLAS build failed'}
 $taskLibraries=@(Get-ChildItem -LiteralPath (Join-Path $taskRoot 'openblas-build/lib') -File | Where-Object {$_.Name -match 'openblas.*\.(a|lib)$'})
 if($taskLibraries.Count -ne 1){throw 'Expected exactly one native static OpenBLAS archive'}
 Copy-Item -LiteralPath $taskLibraries[0].FullName -Destination (Join-Path $taskLibRoot 'openblas.lib')
 $taskProof=@{schema='atom-native-static-build/1';dependencyLockSha256=$taskLockHash;openblasVersion='0.3.33';cpuKernels=@('PRESCOTT','HASWELL','ZEN');threads=$true;staticCrt=$true;
  openblasSha256=(Get-FileHash -LiteralPath (Join-Path $taskLibRoot 'openblas.lib') -Algorithm SHA256).Hash.ToLowerInvariant()}
 $taskProof | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskLibRoot 'build-provenance.json') -Encoding utf8
 Get-ChildItem -LiteralPath $taskLibRoot -File | Select-Object Name,Length
}finally{foreach($taskName in $taskSaved.Keys){[Environment]::SetEnvironmentVariable($taskName,$taskSaved[$taskName],'Process')}}
