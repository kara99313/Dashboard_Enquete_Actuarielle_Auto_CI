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
- Analyse IA (Groq) + Recherche web (Tavily)

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

### 🔐 Secrets (Streamlit Cloud uniquement)

Les secrets doivent être définis via :

➡️ **Streamlit Cloud → Settings → Secrets**

```toml
KOBO_BASE = "https://kf.kobotoolbox.org"
KOBO_TOKEN = "VOTRE_TOKEN"
ASSET_UID = "VOTRE_ASSET_UID"

GROQ_API_KEY = "gsk_xxx"
TAVILY_API_KEY = "tvly_xxx"
```

⚠️ Ne jamais stocker un secret dans GitHub.  
⚠️ Le fichier `secrets.toml` local n'est plus utilisé.

---

## 🛡️ Gestion robuste des secrets dans le code

Le fichier `dashboard.py` utilise désormais :

```python
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except KeyError:
        return default
```

---

## 🤖 Agent IA intégré (Groq + Tavily)

- Analyse assistée par LLM Groq  
- Recherche web en temps réel via Tavily  
- Génération de résumés et d’explications  
- Sous-onglets : *Mode d'emploi* & *Conversation IA*  
- Nettoyage de l'historique

---

## 🧭 Structure du dashboard

- **Vue globale** : KPIs / Résumé / Tendances  
- **Analyse actuarielle** : exposition, fréquence, sévérité, prime pure, graphiques  
- **Résumé exécutif automatique** (IA)  
- **Agent IA**  
- **Gestion du thème (clair/sombre)**  


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
.\reset_streamlit.ps1
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

| Action | Commande |
|--------|------------|
| Lancer | `start_dashboard.ps1` |
| Réparer | `reset_streamlit.ps1` puis `start_dashboard.ps1` |

---

## 9. Scripts PowerShell inclus  
- Gestion des ports  
- Redémarrage propre  
- Activation venv  

---

## 10. Script `reset_streamlit.ps1` (référence)

Ce script :

- Tue tous les processus python.exe, streamlit.exe, node.exe  
- Libère les ports 8501–8505 + 8890  
- Nettoie l’environnement

---

## 11. Script `start_dashboard.ps1` (référence)

Ce script :

- Active `.venv`
- Laisse toujours le port libre
- Lance `dashboard.py` proprement

---

## 12. Section Générale — Documentation pour Tout Projet

### Structure professionnelle recommandée :
```
mon_projet/
│── src/
│── docs/
│── data/
│── notebooks/
│── tests/
│── scripts/
│── requirements.txt
│── README.md
│── .gitignore
└── .env
```

### Règle : Séparer *local* vs *déployé*
- Données lourdes → local uniquement  
- Notebooks → local uniquement  
- Secrets → jamais envoyés  
- Cache/logs → ignorés  
- Code source → GitHub  
- Fichiers essentiels → GitHub  

### Cycle Git standard :
```
git pull
git status
git add .
git commit -m "update"
git push origin main
```

### Sécurité :
- `.env` et `secrets.toml` → toujours ignorés  
- Ne jamais publier un token  
- Vérifier GitHub après chaque push  

---

## 13. Conclusion

Ce README contient **100% des commandes**, scripts, configurations, solutions et procédures pour exécuter le dashboard de manière **infaillible, stable et durable**.

---

## 14. Workflow Git & commandes utiles (pour ce dépôt et les futurs projets)

Cette section résume **toutes les commandes Git importantes** que tu utilises déjà, avec des explications simples pour n’importe quel projet (pas seulement ce dashboard).

> 🔐 Rappel : Git gère **l’historique du projet**. GitHub est juste **le serveur distant** qui stocke tes dépôts.

---

### 14.1. Vérifier l’état du dépôt

```bash
git status
```

- Montre :
  - les fichiers modifiés  
  - ceux prêts à être commit (`staged`)  
  - la branche actuelle (ex : `main`)  
- À utiliser **tout le temps** avant d’agir.

---

### 14.2. Vérifier le dépôt distant (GitHub)

```bash
git remote -v
```

- Affiche les dépôts distants configurés (souvent `origin`).  
- Exemple de sortie :
  - `origin  https://github.com/ton-compte/ton-depot.git (fetch)`  
  - `origin  https://github.com/ton-compte/ton-depot.git (push)`  
- Si tu vois bien ton dépôt GitHub → connexion OK.

---

### 14.3. Récupérer la dernière version depuis GitHub

```bash
git pull origin main
```

- Récupère les **derniers commits** présents sur GitHub vers ton PC.  
- À faire **avant de commencer une nouvelle session de travail**, surtout si tu travailles à plusieurs ou sur plusieurs machines.

> 💡 Si ta branche par défaut s’appelle `master`, la commande devient :  
> `git pull origin master`

---

### 14.4. Ajouter, valider et envoyer tes changements

👉 **Étape 1 : voir ce qui a changé**

```bash
git status
```

👉 **Étape 2 : ajouter les fichiers à suivre**

Ajouter un fichier précis :

```bash
git add dashboard.py
```

Ajouter tous les fichiers modifiés :

```bash
git add .
```

👉 **Étape 3 : créer un commit avec un message clair**

```bash
git commit -m "Message clair expliquant les changements"
```

Exemples de bons messages :

- `"Fix: correction bug scoring prime_pure"`  
- `"Feat: ajout stress tests + PDF report"`  
- `"Refactor: nettoyage code modèle GLM"`  

👉 **Étape 4 : envoyer vers GitHub**

```bash
git push origin main
```

- Envoie tes commits de la branche locale `main` vers la branche distante `main` sur GitHub.  
- C’est seulement **après `git push`** que :
  - GitHub voit tes changements  
  - Streamlit Cloud ou Render peuvent redeployer ta nouvelle version

> 🧠 Résumé :  
> `git add` → je prépare les fichiers  
> `git commit` → je valide une étape dans l’histoire du projet  
> `git push` → j’envoie cette étape sur GitHub

---

### 14.5. Cloner le projet sur une nouvelle machine

Pour récupérer tout le projet sur un autre PC :

```bash
git clone https://github.com/kara99313/Dashboard_Enquete_Actuarielle_Auto_CI.git
```

Puis :

```bash
cd Dashboard_Enquete_Actuarielle_Auto_CI
```

Ensuite, tu peux :

- Créer / activer un `.venv`  
- Installer `requirements.txt`  
- Lancer le dashboard comme expliqué plus haut

---

### 14.6. Voir l’historique des commits

Historique simple :

```bash
git log
```

Historique compact :

```bash
git log --oneline
```

Historique graphique (utile pour les branches) :

```bash
git log --oneline --graph --decorate --all
```

---

### 14.7. Voir les différences entre ta version et le dernier commit

Avant de committer :

```bash
git diff
```

- Affiche **ligne par ligne** ce qui a été changé.  
- Très utile pour vérifier qu’on ne commit pas quelque chose par erreur (par ex. un mot de passe, un test, un print, etc.).

---

### 14.8. Tableau récapitulatif (workflow standard)

| Étape | Commande                            | Rôle principal                                         |
|-------|-------------------------------------|--------------------------------------------------------|
| 1     | `git status`                        | Voir les fichiers modifiés                             |
| 2     | `git remote -v`                     | Vérifier la connexion au dépôt distant                 |
| 3     | `git pull origin main`              | Mettre à jour le projet depuis GitHub                  |
| 4     | `git add .`                         | Préparer tous les fichiers modifiés                    |
| 5     | `git commit -m "message"`           | Valider une étape dans l’historique                    |
| 6     | `git push origin main`              | Envoyer les commits vers GitHub                        |
| 7     | `git log --oneline`                 | Voir l’historique des commits                          |
| 8     | `git diff`                          | Vérifier les modifications avant commit                |
| 9     | `git clone <url>`                   | Récupérer un projet existant                           |

> 🏁 Avec ce tableau + les sections précédentes, tu as une **mini-doc Git réutilisable** pour tous tes futurs dépôts (crédit scoring, actuariel, suivi-évaluation, etc.).

## 🔗 Accès au Dashboard Déployé

👉 **Lien Streamlit :** https://dashboardenqueteactuarielleautoci-j82qjrkwcnptsrrupmtzxg.streamlit.app/

---



