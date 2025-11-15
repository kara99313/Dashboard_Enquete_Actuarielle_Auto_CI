# ============================================================
# AfriAI - START SCRIPT (PS1) - PowerShell robust (ASCII only)
# - Detect/activate venv if needed
# - Upgrade pip
# - Pin/fix numpy, scipy, statsmodels (stable combo)
# - Install requirements.txt (current dir or parent)
# - Run Streamlit (dashboard.py)
# ============================================================

$ErrorActionPreference = 'Stop'

# 0) Move to script directory
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
Write-Host "DIR: $here"

# 1) venv
if ($env:VIRTUAL_ENV) {
    Write-Host "venv already active: $env:VIRTUAL_ENV"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating venv (.venv)..."
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating venv (.venv)..."
    python -m venv .venv
    . .\.venv\Scripts\Activate.ps1
}

# 2) Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# 3) Verify/fix critical versions (avoid scipy/statsmodels mismatch)
$wantNumpy = "1.26.4"
$wantScipy = "1.11.4"
$wantSM    = "0.14.3"

function Get-PyVer([string]$pkg) {
    $out = python -c "import sys; from importlib.util import find_spec; import importlib.metadata as md; print(md.version(sys.argv[1]) if find_spec(sys.argv[1]) else '')" $pkg
    if ($LASTEXITCODE -ne 0) { return "" }
    return ($out.Trim())
}

$curNumpy = Get-PyVer "numpy"
$curScipy = Get-PyVer "scipy"
$curSM    = Get-PyVer "statsmodels"

Write-Host ("Current: numpy={0} | scipy={1} | statsmodels={2}" -f $curNumpy, $curScipy, $curSM)

if (($curNumpy -ne $wantNumpy) -or ($curScipy -ne $wantScipy) -or ($curSM -ne $wantSM)) {
    Write-Host "Fixing versions to stable combo..."
    pip install --no-cache-dir "numpy==$wantNumpy" "scipy==$wantScipy" "statsmodels==$wantSM"
}

# 4) Install requirements (current dir or parent)
$reqPath = ""
if (Test-Path ".\requirements.txt") {
    $reqPath = ".\requirements.txt"
} elseif (Test-Path "..\requirements.txt") {
    $reqPath = "..\requirements.txt"
}

if ($reqPath -ne "") {
    Write-Host "Installing $reqPath ..."
    pip install --no-cache-dir -r $reqPath
} else {
    Write-Warning "requirements.txt not found in current or parent folder. Skipping."
}

# 5) Sanity check imports (single-line python -c)
python -c "import numpy, scipy, statsmodels; print('OK:', numpy.__version__, scipy.__version__, statsmodels.__version__)" | Write-Host
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python sanity check failed."
    exit 1
}

# 6) Run Streamlit
if (-not (Test-Path ".\dashboard.py")) {
    Write-Error "dashboard.py not found in: $here"
    exit 1
}

$port = 8501
Write-Host "Starting Streamlit on port $port ..."
streamlit run dashboard.py --server.port $port
