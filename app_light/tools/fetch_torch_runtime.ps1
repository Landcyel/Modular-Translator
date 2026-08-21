# fetch_torch_runtime.ps1
# Build pluggable torch runtimes under dependencies/runtime/ (recommended scheme).
#
# Usage (PowerShell 7):
#   pwsh tools/fetch_torch_runtime.ps1 -Kind cpu      # CPU baseline
#   pwsh tools/fetch_torch_runtime.ps1 -Kind cuda     # CUDA add-on
#   pwsh tools/fetch_torch_runtime.ps1 -Kind all      # both
#
# Output layout (standard site-packages layout via pip --target):
#   dependencies/runtime/torch-cpu/   torch/ torchaudio/ *.dist-info
#   dependencies/runtime/torch-cuda/  torch/ torchaudio/ *.dist-info
#
# The app (app/torch_runtime.py) auto-selects:
#   torch-cuda (GPU detected via cudart64_*.dll) > torch-cpu > system torch.
# Delete torch-cuda/ to fall back to CPU; no config change is needed.

param(
    [ValidateSet("cpu", "cuda", "all")]
    [string]$Kind = "cpu",
    [string]$Python = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $Root "dependencies\runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

if (-not $Python) {
    $Cand = Join-Path $Root ".venv\Scripts\python.exe"
    $Python = if (Test-Path $Cand) { $Cand } else { "python" }
}
Write-Host "[fetch_torch_runtime] python: $Python"
Write-Host "[fetch_torch_runtime] runtime dir: $RuntimeDir"

function Install-Torch {
    param([string]$Flavor, [string]$Target)
    $TargetPath = Join-Path $RuntimeDir $Target
    if ($Force -and (Test-Path $TargetPath)) {
        Remove-Item -Recurse -Force $TargetPath
        Write-Host "[fetch_torch_runtime] removed existing: $TargetPath"
    }
    New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null

    if ($Flavor -eq "cuda") {
        $Index = "https://download.pytorch.org/whl/cu128"
        $Torch = "torch==2.11.0+cu128"
        $Torchaudio = "torchaudio==2.11.0+cu128"
    } else {
        $Index = "https://download.pytorch.org/whl/cpu"
        $Torch = "torch==2.11.0+cpu"
        $Torchaudio = "torchaudio==2.11.0+cpu"
    }

    & $Python -m pip install $Torch $Torchaudio `
        --index-url $Index `
        --extra-index-url https://pypi.org/simple `
        --target $TargetPath
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed for $Flavor (exit $LASTEXITCODE)"
    }
    Write-Host "[fetch_torch_runtime] $Flavor runtime ready: $TargetPath"
}

if ($Kind -in @("cpu", "all")) { Install-Torch "cpu" "torch-cpu" }
if ($Kind -in @("cuda", "all")) { Install-Torch "cuda" "torch-cuda" }

Write-Host "[fetch_torch_runtime] done. App auto-selects CUDA only when torch-cuda/ is present and a GPU is visible."
