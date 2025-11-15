# Dashboard Actuariel Automobile — Côte d’Ivoire  
Projet GROUPE KV7 / AfriAI — Niveau international

---

## 1. Objectif du projet

Dashboard connecté à *KoboToolbox* permettant :

- Suivi du portefeuille automobile  
- Calcul automatique des métriques actuarielles :
  - Exposition  
  - Fréquence  
  - Sévérité  
  - Prime pure  
- Visualisations :
  - KPIs  
  - Histogrammes, Boxplots  
  - Série temporelle  
  - Cartographie GPS  
- Contrôle de la qualité des données

---

## 2. Fichiers du projet

```
Projet_Final/
│── dashboard.py
│── reset_streamlit.ps1
│── start_dashboard.ps1
│── requirements.txt
│── .streamlit/
│     ├── config.toml
│     └── secrets.toml
└── .venv/
```

---

## 3. Installation initiale

### 3.1. Aller dans le dossier du projet

```powershell
cd "C:\Users\hp\Documents\FORMATION_PYTHON_2025_AfriAI\Projet_Final"
```

### 3.2. Créer l’environnement virtuel

```powershell
python -m venv .venv
```

### 3.3. Activer l’environnement

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 3.4. Installer les dépendances

```powershell
pip install --no-cache-dir -r requirements.txt
```

---

## 4. Configurations Streamlit & KoboToolbox

### 4.1 — `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8890
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 4.2 — `.streamlit/secrets.toml`

```toml
KOBO_BASE = "https://kf.kobotoolbox.org"
KOBO_TOKEN = "TON_TOKEN_ICI"
ASSET_UID = "TON_ASSET_UID"
```

---

## 5. Lancement officiel (méthode recommandée)

### 5.1 Démarrage simple

```powershell
.\start_dashboard.ps1
```

➡️ Ouvrir ensuite :  
**http://localhost:8890**

---

## 6. En cas de PROBLÈME (port bloqué, application blanche, crash)

### 6.1 Réinitialisation complète

```powershell
.
eset_streamlit.ps1
.\start_dashboard.ps1
```

### 6.2 Nettoyage manuel du cache Streamlit

```powershell
streamlit cache clear
```

---

## 7. Méthodes manuelles (optionnelles)

### Activer l’environnement

```powershell
.\.venv\Scripts\Activate.ps1
```

### Démarrer Streamlit à la main

```powershell
streamlit run dashboard.py --server.port 8890
```

---

## 8. Résumé ultra-rapide

### Pour lancer :

```powershell
.\start_dashboard.ps1
```

### Si ça bug :

```powershell
.\reset_streamlit.ps1
.\start_dashboard.ps1
```

---

## 9. Script `reset_streamlit.ps1` (référence)

Ce script :

- Tue tous les processus python.exe, streamlit.exe, node.exe  
- Libère les ports 8501–8505 + 8890  
- Nettoie l’environnement

---

## 10. Script `start_dashboard.ps1` (référence)

Ce script :

- Active `.venv`
- Laisse toujours le port libre
- Lance `dashboard.py` proprement

---

## 11. Conclusion

Ce README contient **100% des commandes**, scripts, configurations, solutions et procédures pour exécuter le dashboard de manière **infaillible, stable et durable**.

