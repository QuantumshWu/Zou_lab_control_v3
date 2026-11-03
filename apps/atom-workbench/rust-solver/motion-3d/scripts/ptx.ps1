# Regenerates kernels/motion3d.ptx from kernels/motion3d.cu with the pinned NVRTC (src/gpu/ptx.rs
# NVRTC_VERSION): takes the nvidia-cuda-nvrtc-cu12 wheel from apps/atom-workbench/.cache (fetching it
# from PyPI once when absent), checks its SHA-256, and runs examples/compile_ptx.rs. Run it after
# editing the kernels; the test gpu::ptx::tests::the_ptx_is_compiled_from_the_source fails until then.
$ErrorActionPreference='Stop'
$crate=Split-Path -Parent $PSScriptRoot
$app=Split-Path -Parent (Split-Path -Parent $crate)
$version='12.0.140'
$wheelName="nvidia_cuda_nvrtc_cu12-$version-py3-none-win_amd64.whl"
$url="https://files.pythonhosted.org/packages/e3/5d/a96c6ac367017cdd2e5853ed12db845e0265666fd12257f9f9324418ba45/$wheelName"
$sha256='5fb35744e31365b684731ca5b52d7fbb429f7dbfa7d0cceeeeeb5e04c73a36fa'
$cache=Join-Path $app ".cache\nvrtc\$version"
$wheel=Join-Path $cache $wheelName
$bin=Join-Path $cache 'nvidia\cuda_nvrtc\bin'
New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $bin 'nvrtc64_120_0.dll'))) {
    if (-not (Test-Path -LiteralPath $wheel)) { Invoke-WebRequest -Uri $url -OutFile $wheel -UseBasicParsing }
    $actual=(Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $sha256) { Remove-Item -LiteralPath $wheel; throw "The NVRTC wheel's SHA-256 is $actual, not the pinned $sha256" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip=[IO.Compression.ZipFile]::OpenRead($wheel)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -like 'nvidia/cuda_nvrtc/bin/*.dll') {
                $target=Join-Path $cache ($entry.FullName -replace '/','\')
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$target,$true)
            }
        }
    } finally { $zip.Dispose() }
}
# NVRTC loads its builtins library by name: from PATH.
$env:PATH="$bin;$env:PATH"
$tools=if($env:ZLC_V3_TOOLS){$env:ZLC_V3_TOOLS}else{Join-Path (Split-Path -Parent (& git -C $app rev-parse --path-format=absolute --git-common-dir).Trim()) '.tools'}
$rust=Join-Path $tools 'rust-1.99.1-x86_64-pc-windows-gnu'
$env:PATH="$rust\bin;$rust\lib\rustlib\x86_64-pc-windows-gnu\bin;$env:PATH"
$env:RUSTC=Join-Path $rust 'bin\rustc.exe'
$env:CARGO_HOME=Join-Path $tools 'cargo-home'
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=Join-Path $tools 'lld-link-1.99.1\lld-link.exe'
$env:LIB=(@('crt\lib\x86_64','sdk\lib\um\x86_64','sdk\lib\ucrt\x86_64') | ForEach-Object {Join-Path (Join-Path $tools 'msvc-sdk') $_}) -join ';'
& (Join-Path $rust 'bin\cargo.exe') run --manifest-path (Join-Path $crate 'Cargo.toml') --target x86_64-pc-windows-msvc --offline --profile iterate --example compile_ptx -- (Join-Path $bin 'nvrtc64_120_0.dll')
if ($LASTEXITCODE -ne 0) { throw "compile_ptx failed: $LASTEXITCODE" }
