<# 
    KV7 / AfriAI - Script de démarrage Streamlit
    - Active l'environnement virtuel .venv
    - Réinitialise les processus Python / Streamlit et les ports
    - Lance le dashboard actuariel (dashboard.py)
#>

param(
    [string]$Port = "8890"
)

Write-Host "===== KV7 / AfriAI - START DASHBOARD =====" -ForegroundColor Cyan

# 1) Activer l'environnement virtuel
$venvActivate = ".\.venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "`nActivation de l'environnement virtuel (.venv)..." -ForegroundColor Yellow
    & $venvActivate
    Write-Host "Environnement virtuel actif." -ForegroundColor Green
} else {
    Write-Host "`nATTENTION : .venv introuvable. Vérifie le dossier ou recrée l'environnement virtuel." -ForegroundColor Red
}

# 2) Exécuter le script de reset (processus + ports)
$resetScript = ".\reset_streamlit.ps1"
if (Test-Path $resetScript) {
    Write-Host "`nExécution de reset_streamlit.ps1..." -ForegroundColor Yellow
    # On "source" le script pour qu'il s'exécute dans la même session
    . $resetScript
} else {
    Write-Host "`nreset_streamlit.ps1 introuvable, on continue sans reset." -ForegroundColor DarkYellow
}

# 3) Fixer le port Streamlit (en cohérence avec config.toml ou override)
$env:STREAMLIT_SERVER_PORT = $Port
Write-Host "`nPort cible pour Streamlit : $Port" -ForegroundColor Cyan

# 4) Lancer le dashboard
Write-Host "`nLancement du dashboard Streamlit (dashboard.py)..." -ForegroundColor Yellow
streamlit run dashboard.py

Write-Host "`n===== COMMANDE streamlit terminée (ou interrompue) =====" -ForegroundColor Cyan
