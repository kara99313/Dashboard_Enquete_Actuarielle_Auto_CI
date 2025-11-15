Write-Host "===== RESET STREAMLIT / PYTHON ====="

# 1) Kill Python / Streamlit / Node processes
Write-Host "`nKilling processes python.exe, streamlit.exe, node.exe..."
$procNames = @("python.exe", "streamlit.exe", "node.exe")

foreach ($name in $procNames) {
    Get-Process -Name $name -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

# 2) Free ports (classic Streamlit ports + 8890)
Write-Host "`nChecking ports 8501-8505 and 8890..."
$ports = @(8501, 8502, 8503, 8504, 8505, 8890)

foreach ($port in $ports) {
    Write-Host "`nScanning port $port..."
    $connection = netstat -ano | Select-String (":$port ")
    if ($connection) {
        $pid = ($connection -split "\s+")[-1]
        Write-Host " Port $port used by PID $pid. Killing..."
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Write-Host " Port $port freed."
    }
    else {
        Write-Host " Port $port already free."
    }
}

Write-Host "`n===== RESET DONE ====="
