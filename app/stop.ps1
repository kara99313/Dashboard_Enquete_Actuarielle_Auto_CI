# ============================================================
# AfriAI - STOP SCRIPT (PS1) - PowerShell robust (ASCII only)
# - Stop Streamlit by port, range, or all Streamlit processes
# - Optional: clear Streamlit cache
# Usage:
#   .\stop.ps1 -Port 8501
#   .\stop.ps1 -RangeStart 8501 -RangeEnd 8510
#   .\stop.ps1 -All
#   .\stop.ps1 -All -Force -ClearCache
# ============================================================

[CmdletBinding()]
param(
    [int]$Port = 0,
    [int]$RangeStart = 0,
    [int]$RangeEnd = 0,
    [switch]$All,
    [switch]$Force,
    [switch]$ClearCache
)

$ErrorActionPreference = 'Stop'

function Get-Pids-ByPort {
    param([int]$p)
    $ids = @()
    try {
        $conns = Get-NetTCPConnection -State Listen,Established -LocalPort $p -ErrorAction Stop
        $ids += ($conns | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
    } catch {
        # Fallback with netstat
        $lines = netstat -ano | Select-String ":$p\s"
        foreach ($l in $lines) {
            $parts = ($l.ToString() -split '\s+') | Where-Object { $_ -ne '' }
            if ($parts.Length -ge 5) {
                $id = $parts[-1]
                if ($id -match '^\d+$') { $ids += [int]$id }
            }
        }
        $ids = $ids | Sort-Object -Unique
    }
    return $ids
}

function Get-StreamlitPids {
    $ids = @()
    try {
        $procs = Get-CimInstance Win32_Process -Filter 'Name="python.exe" OR Name="pythonw.exe" OR Name="streamlit.exe"'
        foreach ($p in $procs) {
            $cmd = ($p.CommandLine) -as [string]
            if ($null -ne $cmd -and $cmd.ToLower().Contains("streamlit")) {
                $ids += [int]$p.ProcessId
            }
        }
    } catch {
        $procs = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match 'python|streamlit' }
        foreach ($p in $procs) { $ids += [int]$p.Id }
    }
    return ($ids | Sort-Object -Unique)
}

function Stop-Pids {
    param([int[]]$PidList, [switch]$ForceKill)
    $killed = @()
    foreach ($ProcId in ($PidList | Sort-Object -Unique)) {
        try {
            if ($ForceKill) {
                Stop-Process -Id $ProcId -Force -ErrorAction Stop
            } else {
                Stop-Process -Id $ProcId -ErrorAction Stop
            }
            Write-Host ("Stopped PID {0}" -f $ProcId)
            $killed += $ProcId
        } catch {
            Write-Warning ("Could not stop PID {0}: {1}" -f ${ProcId}, $_.Exception.Message)
        }
    }
    return $killed
}

function Clear-StreamlitCache {
    Write-Host "Clearing Streamlit cache..."
    $cleared = $false
    try {
        streamlit cache clear --yes 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Streamlit CLI cache clear done."
            $cleared = $true
        }
    } catch { }

    if (-not $cleared) {
        $paths = @(
            "$env:USERPROFILE\.streamlit\cache",
            "$env:LOCALAPPDATA\Temp\streamlit"
        )
        foreach ($pp in $paths) {
            if (Test-Path $pp) {
                try {
                    Remove-Item -LiteralPath $pp -Recurse -Force -ErrorAction Stop
                    Write-Host ("Removed cache folder: {0}" -f $pp)
                    $cleared = $true
                } catch {
                    Write-Warning ("Could not remove cache folder {0}: {1}" -f ${pp}, $_.Exception.Message)
                }
            }
        }
        if (-not $cleared) { Write-Host "No cache folder found to remove." }
    }
}

# ------------------ Main logic ------------------

$allKilled = @()

if ($All) {
    Write-Host "Stopping all Streamlit-related processes..."
    $ids = Get-StreamlitPids
    if ($ids.Count -eq 0) {
        Write-Host "No Streamlit-related process found."
    } else {
        $allKilled += Stop-Pids -PidList $ids -ForceKill:$Force
    }
}
elseif ($Port -gt 0) {
    Write-Host ("Stopping process(es) on port {0} ..." -f $Port)
    $ids = Get-Pids-ByPort -p $Port
    if ($ids.Count -eq 0) {
        Write-Host ("No process found on port {0}." -f $Port)
    } else {
        $allKilled += Stop-Pids -PidList $ids -ForceKill:$Force
    }
}
elseif ($RangeStart -gt 0 -and $RangeEnd -ge $RangeStart) {
    Write-Host ("Scanning ports {0} to {1} ..." -f $RangeStart, $RangeEnd)
    $rangeIds = @()
    for ($pp = $RangeStart; $pp -le $RangeEnd; $pp++) {
        $rangeIds += Get-Pids-ByPort -p $pp
    }
    $rangeIds = $rangeIds | Sort-Object -Unique
    if ($rangeIds.Count -eq 0) {
        Write-Host "No process found in range."
    } else {
        $allKilled += Stop-Pids -PidList $rangeIds -ForceKill:$Force
    }
}
else {
    Write-Host "Nothing to do. Provide -Port <n>, or -RangeStart/-RangeEnd, or -All."
}

if ($ClearCache) {
    Clear-StreamlitCache
}

if ($allKilled.Count -gt 0) {
    Write-Host ("Done. Killed PIDs: {0}" -f ($allKilled -join ", "))
} else {
    Write-Host "Done."
}
