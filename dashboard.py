# ==========================================================
# Dashboard Actuariel Automobile — Côte d’Ivoire
# AfriAI / GROUPE KV7 — Version PRO + Agent IA (Groq + Web)
# Source data : KoboToolbox (API v2)
# ==========================================================

import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

from groq import Groq  # Client Groq pour l’agent IA
from tavily import TavilyClient  # Recherche web temps réel

# --- Modèles ML / GLM (scikit-learn) ---
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# CONFIG GÉNÉRALE — Gestion robuste des secrets
# ==========================================================
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except KeyError:
        return default

KOBO_BASE = get_secret("KOBO_BASE", "https://kf.kobotoolbox.org")
KOBO_TOKEN = get_secret("KOBO_TOKEN")
ASSET_UID = get_secret("ASSET_UID")

GROQ_API_KEY = get_secret("GROQ_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

API_DATA_URL = f"{KOBO_BASE}/api/v2/assets/{ASSET_UID}/data/?format=json&group_sep=/"
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}

# ==========================================================
# CONFIG UI
# ==========================================================
st.set_page_config(
    page_title="Dashboard Actuariel Auto – Côte d’Ivoire",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# OPTIONS GLOBALES : LANGUE
# ==========================================================

st.sidebar.title("⚙️ Dashboard settings")
lang = st.sidebar.radio(
    "Langue / Language",
    ["FR", "EN"],
    index=0,
    help="Choisir la langue principale de l’interface / Choose the main UI language.",
)


def tr(fr: str, en: str) -> str:
    return fr if lang == "FR" else en


# ==========================================================
# THEME COULEURS
# ==========================================================

primary = "#B22222"  # Rouge 'actuariel'
accent = "#FFD700"  # Or
bg = "#FFFDF5"
card_bg = "#FFFFFF"
text_color = "#111827"

# ========== CSS GLOBAL ==========
st.markdown(
    f"""
<style>
:root {{
    --primary-color: {primary};
    --accent-color: {accent};
    --bg-color: {bg};
    --card-bg-color: {card_bg};
    --text-color: {text_color};
}}

html, body, [class*="css"] {{
    font-family: "Georgia", "Times New Roman", serif;
    background-color: var(--bg-color);
    color: var(--text-color);
}}

/* Sidebar premium */
section[data-testid="stSidebar"] > div {{
    background: radial-gradient(circle at top, #ffffff10, #00000060);
    border-right: 1px solid #ffffff22;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: var(--primary-color);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.9rem;
}}

section[data-testid="stSidebar"] label {{
    font-size: 0.85rem;
}}

/* Titre principal */
.page-title-animated {{
    text-align: center;
    font-weight: 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    padding-top: 0.4rem;
    padding-bottom: 0.15rem;
    font-size: 1.9rem;
    background: linear-gradient(120deg, #FF0000, #B22222, #800000);
    -webkit-background-clip: text;
    color: transparent;
    animation: title-glow-move 4.5s ease-in-out infinite alternate;
    font-family: "Times New Roman", serif;
}}

/* Sous-titre KoboToolbox — KV7 / AfriAI : couleur plus lisible */
.page-subtitle {{
    text-align: center;
    font-size: 1.05rem;
    margin-bottom: 0.4rem;
    color: {primary};  /* Rouge foncé lisible sur fond clair */
    font-style: italic;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    text-shadow: 0 0 6px rgba(255, 215, 0, 0.6);
    animation: subtitle-glow 5s ease-in-out infinite alternate;
    font-family: "Times New Roman", serif;
}}

/* Cartes KPI */
.kpi-card {{
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid {accent}55;
    background-color: var(--card-bg-color);
    box-shadow: 0 1px 6px rgba(0,0,0,0.14);
    margin-bottom: 8px;
}}
.kpi-title {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
}}
.kpi-value {{
    font-size: 1.4rem;
    font-weight: bold;
    color: var(--primary-color);
}}

/* Bulles chat IA */
.chat-bubble-user {{
    background-color: rgba(255, 240, 224, 0.95);
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 4px;
    border: 1px solid #FFCC99;
}}
.chat-bubble-assistant {{
    background-color: rgba(245, 247, 255, 0.97);
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 4px;
    border: 1px solid #CCCCFF;
}}

/* Pastilles statut agent IA */
.status-pill {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-bottom: 4px;
    border: 1px solid #DDD;
    background-color: rgba(255,255,255,0.05);
}}

/* Sous-titres du Résumé exécutif */
.exec-subtitle {{
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.9rem;
    color: var(--primary-color);
    border-left: 4px solid var(--accent-color);
    padding-left: 10px;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
}}

/* Animations */
@keyframes title-glow-move {{
    0% {{
        text-shadow: 0 0 0 rgba(0,0,0,0.0);
        transform: translateX(-6px);
    }}
    50% {{
        text-shadow: 0 0 16px rgba(255,255,255,0.5);
        transform: translateX(0px);
    }}
    100% {{
        text-shadow: 0 0 10px rgba(0,0,0,0.4);
        transform: translateX(6px);
    }}
}}
@keyframes subtitle-glow {{
    0% {{
        opacity: 0.8;
        text-shadow: 0 0 0 rgba(0,0,0,0);
    }}
    50% {{
        opacity: 1;
        text-shadow: 0 0 12px rgba(255,215,0,0.9);
    }}
    100% {{
        opacity: 0.95;
        text-shadow: 0 0 4px rgba(0,0,0,0.5);
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.info(
    tr(
        "🛠 Version dashboard : v1.8 — IA + Actuariat (bilingue, vue Accueil enrichie, 3 modèles)",
        "🛠 Dashboard version: v1.8 — AI + Actuarial (bilingual, enriched Home, 3 models)",
    )
)

# ==========================================================
# CONSTANTES
# ==========================================================

TARGET_VARS = {
    "exposition",
    "puissance",
    "agevehicule",
    "ageconducteur",
    "bonus",
    "marque",
    "carburant",
    "densite",
    "region",
    "nbre",
    "garantie",
    "cout",
    "frequence",
    "severite",
    "prime_pure",
    "latitude",
    "longitude",
    "departement",
    "district",
    "date_enquete",
    "date_souscription",
    "date_sinistre",
    "precision_gps",
}

RENAME_SOFT = {
    "age_vehicule": "agevehicule",
    "age_conducteur": "ageconducteur",
    "puissance_cv": "puissance",
    "bonus_malus": "bonus",
    "densite_zone": "densite",
    "nbre_sinistres": "nbre",
    "garantie_principale": "garantie",
    "cout_sinistres": "cout",
    "prime_pure_calculee": "prime_pure",
    "frequence_calculee": "frequence",
    "severite_calculee": "severite",
    "coordonnees_gps": "geopoint",
    "coordonnees_gps/geopoint": "geopoint",
    "date_de_lenquete": "date_enquete",
    "date_de_souscription": "date_souscription",
    "date_dernier_sinistre": "date_sinistre",
    "precision": "precision_gps",
    "gps_precision": "precision_gps",
}

FEATURE_NUM_COLS = [
    "exposition",
    "agevehicule",
    "ageconducteur",
    "bonus",
    "puissance",
    "densite",
    "nbre",
]
FEATURE_CAT_COLS = ["region", "garantie", "carburant", "marque"]
TARGET_COL = "prime_pure"

# ==========================================================
# FONCTIONS DATA
# ==========================================================


@st.cache_data(ttl=300)
def fetch_kobo_json() -> pd.DataFrame:
    r = requests.get(API_DATA_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    js = r.json()
    return pd.DataFrame(js.get("results", []))


def split_geopoint(
    df: pd.DataFrame,
    candidates=("geopoint", "coordonnees_gps", "gps"),
) -> pd.DataFrame:
    col = next((c for c in candidates if c in df.columns), None)
    if not col:
        return df

    parts = df[col].astype(str).str.split(" ", expand=True)
    if parts.shape[1] >= 2:
        df["latitude"] = pd.to_numeric(parts[0], errors="coerce")
        df["longitude"] = pd.to_numeric(parts[1], errors="coerce")
    if parts.shape[1] >= 3:
        df["altitude_m"] = pd.to_numeric(parts[2], errors="coerce")
    if parts.shape[1] >= 4:
        df["precision_gps"] = pd.to_numeric(parts[3], errors="coerce")
    return df


def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Renommage souple
    for src, dst in RENAME_SOFT.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    df = split_geopoint(df)

    decimal_cols = [
        "exposition",
        "bonus",
        "densite",
        "cout",
        "frequence",
        "severite",
        "prime_pure",
    ]
    for c in decimal_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.replace(",", ".", regex=False)

    float_cols = [
        "exposition",
        "bonus",
        "densite",
        "cout",
        "frequence",
        "severite",
        "prime_pure",
        "latitude",
        "longitude",
        "precision_gps",
    ]
    int_cols = ["puissance", "agevehicule", "ageconducteur", "nbre"]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    for c in ["date_enquete", "date_souscription", "date_sinistre", "_submission_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    cat_cols = ["marque", "carburant", "region", "departement", "district", "garantie"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Recalculs actuariels
    if {"nbre", "exposition"}.issubset(df.columns) and "frequence" not in df.columns:
        df["frequence"] = 0.0
        mask_expo = df["exposition"].notna() & (df["exposition"] > 0)
        df.loc[mask_expo, "frequence"] = (
            df.loc[mask_expo, "nbre"].astype(float) / df.loc[mask_expo, "exposition"]
        )

    if {"cout", "nbre"}.issubset(df.columns) and "severite" not in df.columns:
        df["severite"] = 0.0
        mask_nbre = df["nbre"].notna() & (df["nbre"] > 0)
        df.loc[mask_nbre, "severite"] = (
            df.loc[mask_nbre, "cout"].astype(float) / df.loc[mask_nbre, "nbre"]
        )

    if "prime_pure" not in df.columns and {"frequence", "severite"}.issubset(df.columns):
        df["prime_pure"] = df["frequence"] * df["severite"]

    def clip(df_, col, lo, hi):
        if col in df_.columns:
            df_[col] = df_[col].clip(lower=lo, upper=hi)

    clip(df, "exposition", 0.01, 1.20)
    clip(df, "bonus", 0.50, 3.50)
    clip(df, "agevehicule", 0, 40)
    clip(df, "ageconducteur", 16, 85)
    clip(df, "puissance", 2, 25)

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    k = {"nb_obs": len(df)}
    for col in ["exposition", "frequence", "severite", "prime_pure", "cout"]:
        if col in df.columns and df[col].notna().any():
            k[col] = {
                "mean": float(df[col].mean()),
                "p50": float(df[col].median()),
                "p95": float(df[col].quantile(0.95)),
            }
        else:
            k[col] = None
    return k


def compute_deciles_table(df: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    if "prime_pure" not in df.columns:
        return pd.DataFrame()

    cols = [
        c
        for c in ["exposition", "frequence", "severite", "prime_pure", "cout"]
        if c in df.columns
    ]
    if not cols:
        return pd.DataFrame()

    df2 = df[cols].dropna(subset=["prime_pure"]).copy()
    if df2.empty:
        return pd.DataFrame()

    n_unique = df2["prime_pure"].nunique()
    if n_unique < 2:
        return pd.DataFrame()

    q = min(n_deciles, n_unique)
    df2["decile"] = (
        pd.qcut(
            df2["prime_pure"],
            q=q,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    agg_dict = {c: "mean" for c in cols}
    res = df2.groupby("decile", observed=False).agg(agg_dict).reset_index()
    return res


# ==========================================================
# FONCTIONS MODELES / SCORING
# ==========================================================


def train_models_on_df(df: pd.DataFrame):
    """
    Entraîne 3 modèles cohérents sur la même base :
    - GLM Tweedie (log-link)
    - Régression linéaire
    - RandomForest
    """

    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET_COL}' absente des données.")

    df_train = df.dropna(subset=[TARGET_COL]).copy()
    if df_train.empty:
        raise ValueError("Aucune donnée non nulle pour entraîner les modèles.")

    num_cols = [c for c in FEATURE_NUM_COLS if c in df_train.columns]
    cat_cols = [c for c in FEATURE_CAT_COLS if c in df_train.columns]

    if not num_cols and not cat_cols:
        raise ValueError("Aucune variable explicative disponible pour le modèle.")

    X = df_train[num_cols + cat_cols]
    y = df_train[TARGET_COL].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    glm_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                TweedieRegressor(
                    power=1.5,
                    link="log",
                    alpha=0.0,
                    max_iter=5000,
                    tol=1e-5,
                ),
            ),
        ]
    )

    lin_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {}
    metrics = {}

    # GLM
    glm_model.fit(X, y)
    y_pred_glm = glm_model.predict(X)
    metrics["GLM_Tweedie"] = {
        "MAE": float(mean_absolute_error(y, y_pred_glm)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred_glm))),
        "R2": float(r2_score(y, y_pred_glm)),
        "n": int(len(y)),
    }
    models["GLM_Tweedie"] = glm_model

    # Régression linéaire
    lin_model.fit(X, y)
    y_pred_lin = lin_model.predict(X)
    y_pred_lin = np.maximum(y_pred_lin, 0.0)
    metrics["RegLin"] = {
        "MAE": float(mean_absolute_error(y, y_pred_lin)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred_lin))),
        "R2": float(r2_score(y, y_pred_lin)),
        "n": int(len(y)),
    }
    models["RegLin"] = lin_model

    # RandomForest
    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
    y_pred_rf = np.maximum(y_pred_rf, 0.0)
    metrics["RandomForest"] = {
        "MAE": float(mean_absolute_error(y, y_pred_rf)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred_rf))),
        "R2": float(r2_score(y, y_pred_rf)),
        "n": int(len(y)),
    }
    models["RandomForest"] = rf_model

    metrics["_features"] = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    return models, metrics


def build_single_risk_row(
    df_ref: pd.DataFrame,
    features_info: dict,
    user_inputs: dict,
) -> pd.DataFrame:
    num_cols = features_info.get("num_cols", [])
    cat_cols = features_info.get("cat_cols", [])

    row = {}

    for col in num_cols:
        if col in user_inputs and user_inputs[col] is not None:
            row[col] = float(user_inputs[col])
        else:
            if col in df_ref.columns and df_ref[col].notna().any():
                row[col] = float(df_ref[col].median())
            else:
                row[col] = 0.0

    for col in cat_cols:
        if col in user_inputs and user_inputs[col] not in (None, "", "__DEFAULT__"):
            row[col] = user_inputs[col]
        else:
            if col in df_ref.columns and df_ref[col].notna().any():
                row[col] = df_ref[col].mode().iloc[0]
            else:
                row[col] = "INCONNU"

    return pd.DataFrame([row])


# ==========================================================
# GENERATION DE RAPPORT PDF
# ==========================================================


def generate_pdf_report(
    kpis: dict,
    model_metrics: dict | None,
    stress_results: dict | None,
) -> bytes:
    """Rapport global de l’activité."""

    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except ImportError:
        st.error(
            tr(
                "Le module reportlab n'est pas installé. "
                "Installe-le avec : pip install reportlab pour activer l'export PDF.",
                "The 'reportlab' module is not installed. "
                "Install it with: pip install reportlab to enable PDF export.",
            )
        )
        return b""

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def line(txt: str, step: int = 14, bold: bool = False):
        nonlocal y
        if y < 80:
            c.showPage()
            y = height - 50
        if bold:
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(40, y, txt)
        y -= step

    c.setFont("Helvetica-Bold", 16)
    c.drawString(
        40,
        y,
        "Rapport global d’activité — Portefeuille Auto Côte d'Ivoire",
    )
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(
        40,
        y,
        f"Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    y -= 30

    # 1. Contexte
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "1. Contexte et périmètre")
    y -= 18
    line(
        "- Portefeuille d'assurance automobile analysé à partir des données collectées via KoboToolbox "
        "dans le cadre du projet GROUPE KV7 / AfriAI (Côte d’Ivoire)."
    )
    line(
        "- Les résultats reflètent le périmètre filtré dans le dashboard (dates, régions, garanties, etc.)."
    )
    line(
        "- Le rapport est conçu pour plusieurs niveaux de lecture : clients, Direction, Conseil "
        "d’Administration, régulateur."
    )
    y -= 6

    # 2. KPI
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "2. Indicateurs clés du portefeuille (KPI)")
    y -= 18

    if kpis:
        nb_obs = kpis.get("nb_obs", "NA")
        line(f"- Nombre de lignes de risque / contrats : {nb_obs}")
        prime_info = kpis.get("prime_pure")
        freq_info = kpis.get("frequence")
        sev_info = kpis.get("severite")

        if prime_info is not None:
            line(
                f"- Prime pure moyenne : {prime_info['mean']:.2f} FCFA "
                f"(médiane = {prime_info['p50']:.2f}, p95 = {prime_info['p95']:.2f})."
            )
        if freq_info is not None:
            line(
                f"- Fréquence moyenne : {freq_info['mean']:.4f} sinistre(s) par unité d'exposition."
            )
        if sev_info is not None:
            line(
                f"- Sévérité moyenne : {sev_info['mean']:.2f} FCFA par sinistre déclaré."
            )
    else:
        line("- Aucun KPI disponible (données insuffisantes).")
    y -= 6

    # 3. Lecture technique
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "3. Lecture technique & segmentation du risque")
    y -= 18
    line(
        "- La prime pure résulte de la combinaison fréquence × sévérité. Elle sert de base à la tarification "
        "et au pilotage technique."
    )
    line(
        "- Les déciles de risque et la segmentation par région / garantie permettent d’identifier les zones "
        "à forte sinistralité."
    )
    y -= 6

    # 4. Modèles
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "4. Modèles GLM / Régression / Machine Learning")
    y -= 18
    if model_metrics:
        for name, m in model_metrics.items():
            if name.startswith("_"):
                continue
            line(
                f"- {name} : R² = {m['R2']:.3f} | RMSE = {m['RMSE']:.2f} | "
                f"MAE = {m['MAE']:.2f} (n = {m['n']})"
            )
        y -= 6
        line(
            "Lecture Direction / CA : la comparaison entre GLM, Régression linéaire et RandomForest "
            "permet de concilier interprétabilité, robustesse et richesse des interactions. "
            "Les modèles restent des outils d’aide à la décision."
        )
    else:
        line("- Aucun modèle n’a été estimé sur le périmètre courant.")
    y -= 6

    # 5. Stress tests
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "5. Stress tests fréquence / sévérité")
    y -= 18
    if stress_results:
        for label_s, vals in stress_results.items():
            line(f"- Scénario : {label_s}", bold=True)
            line(f"  Prime pure moyenne de base : {vals['base_prime']:.2f} FCFA")
            line(
                f"  Prime pure moyenne stressée : {vals['stress_prime']:.2f} FCFA "
                f"(delta = {vals['delta_prime']:.2f} FCFA)"
            )
            y -= 4
        y -= 4
        line(
            "Les stress tests permettent d’apprécier la sensibilité du portefeuille à des dérives "
            "de fréquence ou de sévérité et d’anticiper les impacts sur le résultat technique."
        )
    else:
        line("- Aucun scénario de stress enregistré pendant la session.")
    y -= 6

    # 6. Recommandations
    c.setFont("Helvetica-Bold", 12)
    c.drawString(
        40,
        y,
        "6. Recommandations (Clients, Direction, CA, Régulateur)",
    )
    y -= 18
    line("6.1 Pour les clients :", bold=True)
    line(
        "- Expliquer de manière transparente les évolutions tarifaires : lien avec la sinistralité, "
        "les charges techniques et les exigences prudentielles."
    )
    line(
        "- Mettre en avant les leviers de maîtrise de la prime : prévention, choix des garanties, franchises."
    )
    y -= 6
    line("6.2 Pour la Direction opérationnelle :", bold=True)
    line(
        "- Suivre mensuellement les KPI par région, garantie et segment de clientèle, "
        "en mettant sous surveillance les déciles les plus risqués."
    )
    line(
        "- Renforcer la qualité des données : complétude des expositions, exactitude des coûts de sinistre, "
        "fiabilité des dates et coordonnées GPS."
    )
    y -= 6
    line("6.3 Pour le Conseil d’Administration :", bold=True)
    line(
        "- Utiliser ce tableau de bord comme support aux comités techniques / risques pour arbitrer : "
        "politique tarifaire, tolérance au risque, objectifs de rentabilité."
    )
    line(
        "- S’assurer de l’existence d’un dispositif de gouvernance des modèles : validation indépendante, "
        "documentation, revue périodique, backtesting."
    )
    y -= 6
    line("6.4 Pour le régulateur (CIMA / CRC) :", bold=True)
    line(
        "- Disposer d’une vision consolidée de la sinistralité et des marges techniques, "
        "cohérente avec la politique de provisionnement et les exigences de solvabilité."
    )
    line(
        "- S’inscrire dans une logique de transparence : traçabilité des données, documentation des hypothèses, "
        "capacité à expliquer les résultats des modèles aux autorités."
    )

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ==========================================================
# FONCTIONS AGENT IA
# ==========================================================


def build_dashboard_context_text(df_f: pd.DataFrame) -> str:
    if df_f.empty:
        return (
            "**1. Profil global du portefeuille filtré**\n"
            "- Aucune observation disponible sur le périmètre sélectionné."
        )

    k = compute_kpis(df_f)
    n = k.get("nb_obs", 0)
    lines: list[str] = []
    lines.append("**1. Profil global du portefeuille filtré**")

    if n:
        lines.append(
            f"- Portefeuille de {n} ligne(s) de risque sur le périmètre (dates, régions, garanties) "
            "actuellement sélectionné dans le dashboard."
        )

    prime_info = k.get("prime_pure")
    freq_info = k.get("frequence")
    sev_info = k.get("severite")

    if prime_info is not None:
        p_mean = prime_info["mean"]
        p_med = prime_info["p50"]
        p95 = prime_info["p95"]
        ratio_tail = p95 / max(p_med, 1e-6)

        if ratio_tail < 3:
            dispersion_txt = (
                "distribution relativement concentrée, queue de sinistralité modérée."
            )
        elif ratio_tail < 8:
            dispersion_txt = (
                "dispersion marquée, avec une queue de distribution significative."
            )
        else:
            dispersion_txt = (
                "queue de distribution très lourde : une minorité de dossiers tire fortement "
                "la prime pure vers le haut."
            )

        lines.append(
            "- Niveau de prime pure : environ "
            f"{p_mean:,.0f} FCFA en moyenne, médiane {p_med:,.0f} FCFA, p95 {p95:,.0f} FCFA ; "
            f"{dispersion_txt}"
        )

    if freq_info is not None and sev_info is not None:
        lines.append(
            "- Structure du risque : fréquence moyenne ~"
            f"{freq_info['mean']:.3f} sinistre(s) par unité d’exposition et sévérité moyenne "
            f"~{sev_info['mean']:,.0f} FCFA par sinistre déclaré."
        )

    lines.append("")
    lines.append("**2. Concentration géographique et par garanties**")

    if {"region", "prime_pure"}.issubset(df_f.columns):
        reg_stats = (
            df_f.dropna(subset=["region", "prime_pure"])
            .groupby("region", observed=False)
            .agg(
                prime_moy=("prime_pure", "mean"),
                nb=("prime_pure", "size"),
            )
            .sort_values("prime_moy", ascending=False)
        )
        if not reg_stats.empty:
            top3 = reg_stats.head(3)
            n_tot = reg_stats["nb"].sum()
            reg_parts = []
            for idx, row in top3.iterrows():
                part = (row["nb"] / n_tot * 100.0) if n_tot > 0 else 0.0
                reg_parts.append(
                    f"{idx} (prime pure moy. {row['prime_moy']:,.0f} FCFA ; ~{part:.1f}% du volume)"
                )
            lines.append(
                "- Régions les plus chargées : " + "; ".join(reg_parts) + "."
            )

    if {"garantie", "prime_pure"}.issubset(df_f.columns):
        gar_stats = (
            df_f.dropna(subset=["garantie", "prime_pure"])
            .groupby("garantie", observed=False)
            .agg(
                prime_moy=("prime_pure", "mean"),
                nb=("prime_pure", "size"),
            )
            .sort_values("prime_moy", ascending=False)
        )
        if not gar_stats.empty:
            top3g = gar_stats.head(3)
            n_totg = gar_stats["nb"].sum()
            gar_parts = []
            for idx, row in top3g.iterrows():
                part = (row["nb"] / n_totg * 100.0) if n_totg > 0 else 0.0
                gar_parts.append(
                    f"{idx} (prime pure moy. {row['prime_moy']:,.0f} FCFA ; ~{part:.1f}% du volume)"
                )
            lines.append(
                "- Garanties les plus exposées : " + "; ".join(gar_parts) + "."
            )

    lines.append("")
    lines.append("**3. Déciles de risque et concentration**")

    dec = compute_deciles_table(df_f)
    if not dec.empty:
        d_low = dec.iloc[0]
        d_high = dec.iloc[-1]
        ratio = d_high["prime_pure"] / max(d_low["prime_pure"], 1e-6)
        lines.append(
            "- Gradient de risque significatif : la prime pure moyenne passe d’environ "
            f"{d_low['prime_pure']:,.0f} FCFA (décile le plus bas) "
            f"à {d_high['prime_pure']:,.0f} FCFA (décile le plus élevé), "
            f"soit un facteur ~{ratio:,.1f}."
        )

    if "prime_pure" in df_f.columns and df_f["prime_pure"].notna().sum() > 0:
        try:
            q90 = df_f["prime_pure"].quantile(0.9)
            high = df_f[df_f["prime_pure"] >= q90]
            part_obs = (len(high) / len(df_f) * 100.0) if len(df_f) > 0 else 0.0
            if "cout" in df_f.columns and df_f["cout"].notna().any():
                tot_cost = df_f["cout"].sum()
                high_cost = high["cout"].sum()
                part_cost = (high_cost / tot_cost * 100.0) if tot_cost > 0 else 0.0
                lines.append(
                    f"- Concentration extrême : ~{part_obs:.1f}% des contrats (top 10% en prime pure) "
                    f"représentent ~{part_cost:.1f}% du coût total."
                )
        except Exception:
            pass

    lines.append("")
    lines.append("**4. Messages clés pour la Direction**")
    lines.append(
        "- Les derniers déciles de risque concentrent une part majeure du coût technique : "
        "ils doivent être pilotés (tarification, souscription, prévention)."
    )
    lines.append(
        "- Les régions / garanties les plus chargées doivent faire l’objet d’un suivi dédié "
        "(revue tarifaire, calibrage des franchises, plan de prévention)."
    )
    lines.append(
        "- Les décisions doivent tenir compte du périmètre actuel (filtres appliqués dans le dashboard)."
    )

    return "\n".join(lines)


def web_search_summary(query: str, max_results: int = 4) -> str:
    if not TAVILY_API_KEY:
        return ""
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        res = client.search(query=query, max_results=max_results)
    except Exception:
        return ""

    parts = []
    answer = res.get("answer")
    if answer:
        parts.append(f"Résumé global de la recherche : {answer}")
    results = res.get("results", [])
    for r in results[:max_results]:
        title = r.get("title", "Sans titre")
        url = r.get("url", "")
        content = r.get("content", "")
        short = content[:400].replace("\n", " ")
        parts.append(f"- {title} ({url}) : {short}...")
    return "\n".join(parts)


def call_groq_chat(
    messages: list[dict],
    model: str = "llama-3.3-70b-versatile",
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY manquant dans .streamlit/secrets.toml")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
        top_p=1.0,
        stream=False,
    )
    return completion.choices[0].message.content


# ==========================================================
# EN-TÊTE PRINCIPAL
# ==========================================================

st.markdown(
    f"""
<h1 class="page-title-animated">
    {tr("Dashboard Actuariel Automobile — Côte d'Ivoire", "Automobile Actuarial Dashboard — Côte d'Ivoire")}
</h1>
<p class="page-subtitle">
    Data collected & curated via KoboToolbox — GROUPE KV7 / AfriAI actuarial project (Côte d’Ivoire)
</p>
<hr style="border:1px solid {accent}; margin-top:0; margin-bottom:10px;" />
""",
    unsafe_allow_html=True,
)

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

with st.spinner(
    tr(
        "Connexion à KoboToolbox et préparation des données…",
        "Connecting to KoboToolbox and preparing data…",
    )
):
    df_raw = fetch_kobo_json()

if df_raw.empty:
    st.warning(
        tr(
            "Aucune donnée disponible pour le moment.",
            "No data available at the moment.",
        )
    )
    st.stop()

df = prepare_dataframe(df_raw)

# ==========================================================
# FILTRES SIDEBAR
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.subheader(tr("🎛️ Filtres d'analyse", "🎛️ Analysis filters"))

date_col = (
    "_submission_time"
    if "_submission_time" in df.columns
    else ("date_enquete" if "date_enquete" in df.columns else None)
)

if date_col and pd.notna(df[date_col]).any():
    tmin = df[date_col].min()
    tmax = df[date_col].max()
    d1, d2 = st.sidebar.date_input(
        tr("Période d'analyse", "Analysis period"),
        [tmin.date(), tmax.date()],
        help=tr(
            "Filtre sur la date d'enquête (ou date de soumission).",
            "Filter on survey (or submission) date.",
        ),
    )
else:
    d1 = d2 = None

regions = (
    sorted(df["region"].dropna().unique()) if "region" in df.columns else []
)
region_sel = st.sidebar.multiselect(
    tr("Région", "Region"),
    regions,
    default=regions,
)

deps = (
    sorted(df["departement"].dropna().unique())
    if "departement" in df.columns
    else []
)
depart_sel = st.sidebar.multiselect(
    tr("Département", "Department"),
    deps,
    default=deps,
)

garanties = (
    sorted(df["garantie"].dropna().unique())
    if "garantie" in df.columns
    else []
)
gar_sel = st.sidebar.multiselect(
    tr("Garantie", "Coverage"),
    garanties,
    default=garanties,
)

mask = pd.Series(True, index=df.index)

if date_col and d1:
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask &= df[date_col].between(d1, d2)

if region_sel and "region" in df.columns:
    mask &= df["region"].isin(region_sel)

if depart_sel and "departement" in df.columns:
    mask &= df["departement"].isin(depart_sel)

if gar_sel and "garantie" in df.columns:
    mask &= df["garantie"].isin(gar_sel)

df_f = df.loc[mask].copy()
kpis = compute_kpis(df_f)
ctx_text = build_dashboard_context_text(df_f)

with st.sidebar.expander(
    tr("🔎 Schéma & Qualité des variables", "🔎 Schema & data quality")
):
    have = set(df.columns)
    missing = sorted(list(TARGET_VARS - have))
    extra = sorted(list(have - TARGET_VARS))

    st.write(
        tr("**Variables cibles manquantes** :", "**Missing target variables** :"),
        missing if missing else tr("Aucune ✅", "None ✅"),
    )
    st.write(
        tr("**Variables supplémentaires** :", "**Extra variables** :"),
        extra if extra else tr("Aucune", "None"),
    )
    st.write(tr("**Types de données (échantillon)**", "**Data types (sample)**"))
    st.write(df.dtypes.astype(str))

if st.sidebar.button(
    tr("🔄 Recharger les données Kobo", "🔄 Reload Kobo data")
):
    fetch_kobo_json.clear()
    st.rerun()


def render_kpi_card(title: str, value: str):
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kpi-title">{title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="kpi-value">{value}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# TABS PRINCIPAUX
# ==========================================================

tab_home, tab_analysis, tab_map, tab_quality, tab_models, tab_exec, tab_agent = st.tabs(
    [
        tr("🏠 Accueil", "🏠 Home"),
        tr("📈 Analyse du portefeuille", "📈 Portfolio analysis"),
        tr("🗺️ Cartographie", "🗺️ Mapping"),
        tr("🧪 Qualité & Données", "🧪 Data quality"),
        tr("🤖 Modèles & Scoring", "🤖 Models & Scoring"),
        tr("📊 Résumé exécutif", "📊 Executive summary"),
        tr("🧠 Agent IA", "🧠 AI Agent"),
    ]
)

# ==========================================================
# TAB 1 — ACCUEIL (SANS SOUS-ONGLETS)
# ==========================================================

with tab_home:
    st.markdown(
        tr(
            "### Bienvenue sur le Dashboard Actuariel Automobile",
            "### Welcome to the Automobile Actuarial Dashboard",
        )
    )
    st.markdown(
        tr(
            "Ce tableau de bord fournit une **vision consolidée** du portefeuille auto, "
            "alimentée en continu par les données **KoboToolbox** sur le terrain.",
            "This dashboard provides a **consolidated view** of the motor portfolio, "
            "continuously fed by **KoboToolbox** field data.",
        )
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            tr("Lignes de risque", "Risk lines"),
            f"{kpis['nb_obs']}",
        )

    nb_regions = (
        int(df_f["region"].nunique())
        if ("region" in df_f.columns and not df_f.empty)
        else 0
    )
    nb_gar = (
        int(df_f["garantie"].nunique())
        if ("garantie" in df_f.columns and not df_f.empty)
        else 0
    )

    with c2:
        render_kpi_card(
            tr("Régions couvertes", "Regions covered"),
            f"{nb_regions}",
        )
    with c3:
        render_kpi_card(
            tr("Garanties couvertes", "Coverages"),
            f"{nb_gar}",
        )
    with c4:
        if date_col and df_f[date_col].notna().any():
            dmin = df_f[date_col].min().strftime("%d/%m/%Y")
            dmax = df_f[date_col].max().strftime("%d/%m/%Y")
            render_kpi_card(
                tr("Période des données", "Data period"),
                f"{dmin} → {dmax}",
            )
        else:
            render_kpi_card(
                tr("Période des données", "Data period"),
                tr("Non disponible", "Not available"),
            )

    st.markdown(
        tr(
            "#### Dernières données KoboToolbox (suivi temps quasi réel)",
            "#### Latest KoboToolbox data (near real-time monitoring)",
        )
    )
    cols_to_show = []
    if date_col and date_col in df_f.columns:
        cols_to_show.append(date_col)
    for c in ["region", "garantie", "prime_pure", "nbre", "cout"]:
        if c in df_f.columns and c not in cols_to_show:
            cols_to_show.append(c)

    if not df_f.empty and cols_to_show:
        df_last = df_f.sort_values(
            date_col if date_col in df_f.columns else df_f.index,
            ascending=False,
        )[cols_to_show].head(10)
        st.dataframe(df_last, use_container_width=True)
    else:
        st.info(
            tr(
                "Aucune observation filtrée pour le moment.",
                "No filtered observation at the moment.",
            )
        )

    st.markdown(
        tr(
            "##### Vue management",
            "##### Management view",
        )
    )
    st.markdown(
        tr(
            """
- **Utilisation quotidienne** : vérifier que les entrées terrain arrivent correctement (dates, régions, garanties).
- **Pilotage global** : suivre le nombre de lignes de risque et l’extension géographique du portefeuille.
- **Orientation** : les onglets suivants donnent une vision plus détaillée (analyse, cartographie, modèles, résumé exécutif, IA).
""",
            """
- **Daily use**: ensure that field entries come in correctly (dates, regions, coverages).
- **Global steering**: monitor the number of risk lines and geographical coverage.
- **Navigation**: use the next tabs for deeper analysis (portfolio, mapping, models, executive summary, AI agent).
""",
        )
    )

    # Bloc "À propos" sans sous-onglets
    with st.expander(
        tr("ℹ️ À propos du projet", "ℹ️ About this project"),
        expanded=False,
    ):
        if lang == "FR":
            st.markdown("### À propos du projet")
            st.markdown(
                """
Ce dashboard a été conçu dans le cadre du projet **AfriAI / GROUPE KV7** pour :

- Mettre en place une **chaîne analytique complète** pour l’assurance automobile en Côte d’Ivoire.
- Connecter la collecte terrain (**KoboToolbox**) aux indicateurs actuariels clés :
  - Exposition
  - Fréquence
  - Sévérité
  - Prime pure
- Offrir une **interface professionnelle** pour :
  - La Direction technique
  - La Direction générale
  - Le Conseil d’Administration
  - Les autorités de supervision (CIMA / BCEAO, etc.)

L’application est :
- **Bilingue** (français / anglais),
- Connectée aux **données réelles** de terrain,
- Compatible avec une **intégration IA avancée** (agent Groq + recherche web Tavily).
"""
            )
            st.markdown("### Architecture fonctionnelle (vue simplifiée)")
            st.markdown(
                """
1. **Collecte des données** : KoboToolbox (agents terrain, GPS, dates, garanties, sinistres).
2. **Ingestion & préparation** : ce dashboard prépare les variables actuarielles (fréquence, sévérité, prime pure).
3. **Analyse & visualisation** : onglets d’analyse détaillée, cartographie, qualité des données.
4. **Modélisation** : GLM Tweedie, Régression linéaire, RandomForest, stress tests.
5. **Pilotage & IA** : Résumé exécutif, rapports PDF, Agent IA (analyste / régulateur virtuel).
"""
            )
        else:
            st.markdown("### About this project")
            st.markdown(
                """
This dashboard has been designed within the **AfriAI / GROUPE KV7** project to:

- Implement a **complete analytical chain** for motor insurance in Côte d’Ivoire.
- Bridge field data (**KoboToolbox**) with key actuarial metrics:
  - Exposure
  - Frequency
  - Severity
  - Pure premium
- Offer a **professional interface** for:
  - Technical management
  - Executive management
  - Board of Directors
  - Supervisory authorities (CIMA / BCEAO, etc.)

The app is:
- **Bilingual** (French / English),
- Connected to **real field data**,
- Compatible with **advanced AI integration** (Groq agent + Tavily web search).
"""
            )
            st.markdown("### Functional architecture (simplified view)")
            st.markdown(
                """
1. **Data collection**: KoboToolbox (field agents, GPS, dates, coverages, claims).
2. **Ingestion & preparation**: this dashboard prepares actuarial variables (frequency, severity, pure premium).
3. **Analysis & visualization**: detailed analysis tabs, mapping, data quality.
4. **Modeling**: GLM Tweedie, Linear Regression, RandomForest, stress tests.
5. **Steering & AI**: Executive summary, PDF reports, AI Agent (virtual analyst / supervisor).
"""
            )

# ==========================================================
# TAB 2 — ANALYSE DU PORTEFEUILLE
# ==========================================================

with tab_analysis:
    st.subheader(
        tr(
            "Analyse du portefeuille filtré — KPIs & distributions",
            "Filtered portfolio analysis — KPIs & distributions",
        )
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card(
            tr("Nombre d'observations", "Number of observations"),
            f"{kpis['nb_obs']}",
        )
    with c2:
        expo_info = kpis.get("exposition")
        expo_val = f"{expo_info['mean']:.2f}" if expo_info is not None else "—"
        render_kpi_card(
            tr("Exposition moyenne", "Average exposure"),
            expo_val,
        )
    with c3:
        freq_info = kpis.get("frequence")
        freq_val = f"{freq_info['mean']:.4f}" if freq_info is not None else "—"
        render_kpi_card(
            tr("Fréquence moyenne", "Average frequency"),
            freq_val,
        )
    with c4:
        sev_info = kpis.get("severite")
        sev_val = f"{sev_info['mean']:.2f}" if sev_info is not None else "—"
        render_kpi_card(
            tr("Sévérité moyenne", "Average severity"),
            sev_val,
        )
    with c5:
        prime_info = kpis.get("prime_pure")
        prime_val = f"{prime_info['mean']:.2f}" if prime_info is not None else "—"
        render_kpi_card(
            tr("Prime pure moyenne", "Average pure premium"),
            prime_val,
        )

    st.markdown(
        tr(
            "### Structure du portefeuille (par région et par garantie)",
            "### Portfolio structure (by region and coverage)",
        )
    )

    c6, c7 = st.columns(2)

    if {"region", "prime_pure"}.issubset(df_f.columns):
        with c6:
            pivot = (
                df_f.dropna(subset=["region", "prime_pure"])
                .groupby("region", dropna=True, observed=False)["prime_pure"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig_reg = px.bar(
                pivot,
                x="region",
                y="prime_pure",
                title=tr(
                    "Prime pure moyenne par région",
                    "Average pure premium by region",
                ),
            )
            st.plotly_chart(fig_reg, use_container_width=True)
    else:
        c6.write(
            tr(
                "Pas assez de données pour l'analyse par région.",
                "Not enough data for analysis by region.",
            )
        )

    if {"garantie", "prime_pure"}.issubset(df_f.columns):
        with c7:
            pivot_gar = (
                df_f.dropna(subset=["garantie", "prime_pure"])
                .groupby("garantie", dropna=True, observed=False)["prime_pure"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig_gar = px.bar(
                pivot_gar,
                x="garantie",
                y="prime_pure",
                title=tr(
                    "Prime pure moyenne par garantie",
                    "Average pure premium by coverage",
                ),
            )
            st.plotly_chart(fig_gar, use_container_width=True)
    else:
        c7.write(
            tr(
                "Pas assez de données pour l'analyse par garantie.",
                "Not enough data for analysis by coverage.",
            )
        )

    st.markdown(
        tr(
            "### Distributions actuarielles",
            "### Actuarial distributions",
        )
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            tr(
                "**Distribution de la prime pure**",
                "**Pure premium distribution**",
            )
        )
        if "prime_pure" in df_f.columns:
            fig = px.histogram(
                df_f,
                x="prime_pure",
                nbins=30,
                title=tr(
                    "Distribution de la prime pure",
                    "Pure premium distribution",
                ),
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                tr(
                    "Colonne 'prime_pure' manquante.",
                    "'prime_pure' column missing.",
                )
            )
    with colB:
        st.markdown(
            tr(
                "**Prime pure par région**",
                "**Pure premium by region**",
            )
        )
        if {"prime_pure", "region"}.issubset(df_f.columns):
            fig = px.box(
                df_f.dropna(subset=["prime_pure", "region"]),
                x="region",
                y="prime_pure",
                points="outliers",
                title=tr(
                    "Prime pure par région",
                    "Pure premium by region",
                ),
            )
            fig.update_layout(xaxis_title="Région", yaxis_title="Prime pure")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                tr(
                    "Colonnes 'prime_pure' ou 'region' manquantes.",
                    "'prime_pure' or 'region' columns missing.",
                )
            )

    st.markdown(
        tr(
            "### Fréquence & Sévérité — distributions",
            "### Frequency & severity — distributions",
        )
    )

    colC, colD = st.columns(2)
    with colC:
        st.markdown(
            tr(
                "**Fréquence (nbre / exposition)**",
                "**Frequency (claims / exposure)**",
            )
        )
        if "frequence" in df_f.columns:
            fig = px.histogram(
                df_f,
                x="frequence",
                nbins=30,
                title=tr(
                    "Distribution de la fréquence",
                    "Frequency distribution",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                tr(
                    "Colonne 'frequence' manquante.",
                    "'frequence' column missing.",
                )
            )
    with colD:
        st.markdown(
            tr(
                "**Sévérité (cout / nbre)**",
                "**Severity (cost / claims)**",
            )
        )
        if "severite" in df_f.columns:
            fig = px.histogram(
                df_f,
                x="severite",
                nbins=30,
                title=tr(
                    "Distribution de la sévérité",
                    "Severity distribution",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                tr(
                    "Colonne 'severite' manquante.",
                    "'severite' column missing.",
                )
            )

    st.markdown(
        tr(
            "### Déciles de risque (prime pure)",
            "### Risk deciles (pure premium)",
        )
    )
    deciles = compute_deciles_table(df_f)
    if not deciles.empty:
        st.dataframe(deciles, use_container_width=True)
    else:
        st.info(
            tr(
                "Déciles non disponibles (données insuffisantes).",
                "Deciles not available (insufficient data).",
            )
        )

    st.markdown(
        tr(
            "### Évolution temporelle de la prime pure",
            "### Time evolution of pure premium",
        )
    )
    if "prime_pure" in df_f.columns and date_col and date_col in df_f.columns:
        df_temp = df_f[[date_col, "prime_pure"]].dropna().sort_values(date_col)
        if not df_temp.empty:
            fig = px.line(
                df_temp,
                x=date_col,
                y="prime_pure",
                title=tr(
                    "Prime pure dans le temps",
                    "Pure premium over time",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                tr(
                    "Pas de données temporelles suffisantes.",
                    "Not enough time-series data.",
                )
            )
    else:
        st.info(
            tr(
                "Colonnes temporelles / prime pure manquantes pour la série temporelle.",
                "Missing date / pure premium columns for time-series analysis.",
            )
        )

    st.markdown(
        tr(
            "### Table des observations filtrées",
            "### Filtered observations table",
        )
    )
    st.dataframe(df_f, use_container_width=True)
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        tr("⬇️ Télécharger le CSV filtré", "⬇️ Download filtered CSV"),
        csv,
        "actuariel_filtre.csv",
        "text/csv",
    )

# ==========================================================
# TAB 3 — CARTOGRAPHIE
# ==========================================================

with tab_map:
    st.subheader(tr("Cartographie des observations", "Mapping of observations"))

    if {"latitude", "longitude"}.issubset(df_f.columns):
        st.markdown(
            tr(
                "**Vue simple (scatter géographique)**",
                "**Simple view (geographical scatter)**",
            )
        )
        st.map(df_f[["latitude", "longitude"]].dropna())

        st.markdown(
            tr(
                "**Synthèse par région (si GPS et région disponibles)**",
                "**Regional summary (if GPS and region available)**",
            )
        )
        if "region" in df_f.columns:
            df_geo_reg = (
                df_f.dropna(subset=["latitude", "longitude", "region"])
                .groupby("region", dropna=True, observed=False)
                .agg(
                    lat=("latitude", "mean"),
                    lon=("longitude", "mean"),
                    nb=("region", "size"),
                    prime_moy=("prime_pure", "mean"),
                )
                .reset_index()
            )
            st.dataframe(df_geo_reg, use_container_width=True)
        else:
            st.info(
                tr(
                    "Région non disponible pour la synthèse par région.",
                    "Region not available for regional synthesis.",
                )
            )
    else:
        st.info(
            tr(
                "Pas de coordonnées GPS disponibles. Vérifie que le champ geopoint est bien rempli dans Kobo.",
                "No GPS coordinates available. Check that the geopoint field is correctly filled in Kobo.",
            )
        )

# ==========================================================
# TAB 4 — QUALITÉ & DONNÉES
# ==========================================================

with tab_quality:
    st.subheader(
        tr(
            "Qualité des données & structure",
            "Data quality & structure",
        )
    )

    st.markdown(
        tr(
            "### Aperçu des données brutes Kobo (10 premières lignes)",
            "### Raw Kobo data preview (first 10 rows)",
        )
    )
    with st.expander(
        tr("Données brutes KoboToolbox", "Raw KoboToolbox data")
    ):
        st.write(
            tr("Nb de lignes brutes :", "Number of raw rows:"),
            len(df_raw),
        )
        st.write(
            tr("Colonnes brutes :", "Raw columns:"),
            list(df_raw.columns),
        )
        st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown(
        tr(
            "### Données après préparation (types & manquants)",
            "### Data after preparation (types & missing values)",
        )
    )

    colQ1, colQ2 = st.columns(2)
    with colQ1:
        st.markdown(tr("**Types de données**", "**Data types**"))
        st.write(df.dtypes.astype(str))
    with colQ2:
        st.markdown(
            tr(
                "**Taux de valeurs manquantes (%)**",
                "**Missing values rate (%)**",
            )
        )
        miss = df.isna().mean().sort_values(ascending=False) * 100
        st.write(miss.round(1))

    st.markdown(
        tr(
            "### Vérification par rapport au schéma cible",
            "### Check against target schema",
        )
    )

    have = set(df.columns)
    missing = sorted(list(TARGET_VARS - have))
    extra = sorted(list(have - TARGET_VARS))

    cM1, cM2 = st.columns(2)
    with cM1:
        st.markdown(
            tr(
                "**Variables cibles manquantes**",
                "**Missing target variables**",
            )
        )
        if missing:
            st.write(missing)
        else:
            st.success(
                tr(
                    "Aucune — schéma cible complet 🎯",
                    "None — target schema complete 🎯",
                )
            )
    with cM2:
        st.markdown(
            tr(
                "**Variables en plus (non prévues dans le schéma)**",
                "**Extra variables (not in target schema)**",
            )
        )
        if extra:
            st.write(extra)
        else:
            st.info(
                tr(
                    "Aucune variable supplémentaire.",
                    "No extra variable.",
                )
            )

# ==========================================================
# TAB 5 — MODELES, SCORING, STRESS & PDF
# ==========================================================

with tab_models:
    st.subheader(
        tr(
            "Modèles GLM / Régression / Machine Learning, Scoring & Stress tests",
            "GLM / Regression / Machine Learning models, scoring & stress tests",
        )
    )

    if df_f.empty:
        st.warning(
            tr(
                "Aucune donnée filtrée : impossible d'entraîner les modèles.",
                "No filtered data: cannot train models.",
            )
        )
    else:
        st.markdown(
            tr(
                "#### 1. Entraînement des modèles sur les données filtrées",
                "#### 1. Model training on filtered data",
            )
        )

        if "models" not in st.session_state:
            st.session_state["models"] = None
            st.session_state["model_metrics"] = None
        if "stress_results" not in st.session_state:
            st.session_state["stress_results"] = {}

        if st.button(
            tr(
                "⚙️ (Re)entraîner GLM / Régression linéaire / RandomForest",
                "⚙️ (Re)train GLM / Linear Regression / RandomForest",
            )
        ):
            with st.spinner(
                tr(
                    "Entraînement des modèles (GLM, Régression linéaire, RandomForest)…",
                    "Training models (GLM, Linear Regression, RandomForest)…",
                )
            ):
                try:
                    models, metrics = train_models_on_df(df_f)
                    st.session_state["models"] = models
                    st.session_state["model_metrics"] = metrics
                    st.success(
                        tr(
                            "Modèles entraînés avec succès sur les données filtrées. ✅",
                            "Models successfully trained on filtered data. ✅",
                        )
                    )
                except Exception as e:
                    st.error(
                        tr(
                            f"Erreur lors de l'entraînement des modèles : {e}",
                            f"Error while training models: {e}",
                        )
                    )

        metrics = st.session_state.get("model_metrics", None)
        models = st.session_state.get("models", None)

        if metrics:
            st.markdown(
                tr(
                    "##### Performance des modèles (in-sample)",
                    "##### Model performance (in-sample)",
                )
            )
            met_df = []
            for name, m in metrics.items():
                if name.startswith("_"):
                    continue
                met_df.append(
                    {
                        tr("Modèle", "Model"): name,
                        "R2": m["R2"],
                        "RMSE": m["RMSE"],
                        "MAE": m["MAE"],
                        "n": m["n"],
                    }
                )
            st.dataframe(pd.DataFrame(met_df), use_container_width=True)

        st.markdown("---")
        st.markdown(
            tr(
                "#### 2. Scoring individuel & simulation tarifaire",
                "#### 2. Individual scoring & pricing simulation",
            )
        )

        if not models or not metrics:
            st.info(
                tr(
                    "Entraîne d'abord les modèles pour activer le scoring.",
                    "Train the models first to enable scoring.",
                )
            )
        else:
            model_name = st.selectbox(
                tr(
                    "Choix du modèle pour le scoring",
                    "Choose model for scoring",
                ),
                options=list(models.keys()),
                index=0,
            )
            features_info = metrics.get("_features", {"num_cols": [], "cat_cols": []})

            with st.form("scoring_form"):
                st.markdown(
                    tr(
                        "##### Paramètres du risque à scorer",
                        "##### Risk parameters to score",
                    )
                )

                cc1, cc2 = st.columns(2)

                if (
                    "exposition" in df_f.columns
                    and df_f["exposition"].notna().any()
                ):
                    expo_med = float(df_f["exposition"].median())
                else:
                    expo_med = 1.0
                expo_default = float(np.clip(expo_med, 0.01, 1.20))

                if (
                    "ageconducteur" in df_f.columns
                    and df_f["ageconducteur"].notna().any()
                ):
                    agecond_med = int(df_f["ageconducteur"].median())
                else:
                    agecond_med = 35
                agecond_default = int(np.clip(agecond_med, 16, 90))

                if (
                    "agevehicule" in df_f.columns
                    and df_f["agevehicule"].notna().any()
                ):
                    ageveh_med = int(df_f["agevehicule"].median())
                else:
                    ageveh_med = 5
                ageveh_default = int(np.clip(ageveh_med, 0, 40))

                if "bonus" in df_f.columns and df_f["bonus"].notna().any():
                    bonus_med = float(df_f["bonus"].median())
                else:
                    bonus_med = 1.0
                bonus_default = float(np.clip(bonus_med, 0.5, 3.5))

                if "puissance" in df_f.columns and df_f["puissance"].notna().any():
                    puiss_med = int(df_f["puissance"].median())
                else:
                    puiss_med = 8
                puiss_default = int(np.clip(puiss_med, 2, 40))

                if "densite" in df_f.columns and df_f["densite"].notna().any():
                    dens_med = float(df_f["densite"].median())
                else:
                    dens_med = 100.0
                dens_default = float(np.clip(dens_med, 0.0, 1000.0))

                with cc1:
                    exposition_in = st.number_input(
                        tr("Exposition (en années)", "Exposure (years)"),
                        min_value=0.01,
                        max_value=1.20,
                        value=expo_default,
                        step=0.01,
                    )
                    agecond_in = st.number_input(
                        tr("Âge conducteur", "Driver age"),
                        min_value=16,
                        max_value=90,
                        value=agecond_default,
                    )
                    ageveh_in = st.number_input(
                        tr("Âge véhicule", "Vehicle age"),
                        min_value=0,
                        max_value=40,
                        value=ageveh_default,
                    )
                with cc2:
                    bonus_in = st.number_input(
                        tr(
                            "Bonus-malus (coefficient)",
                            "Bonus-malus (coefficient)",
                        ),
                        min_value=0.5,
                        max_value=3.5,
                        value=bonus_default,
                        step=0.05,
                    )
                    puiss_in = st.number_input(
                        tr(
                            "Puissance fiscale (CV)",
                            "Fiscal power (HP)",
                        ),
                        min_value=2,
                        max_value=40,
                        value=puiss_default,
                    )
                    dens_in = st.number_input(
                        tr(
                            "Densité zone (proxy risque)",
                            "Zone density (risk proxy)",
                        ),
                        min_value=0.0,
                        max_value=1000.0,
                        value=dens_default,
                        step=1.0,
                    )

                col_cat1, col_cat2 = st.columns(2)
                with col_cat1:
                    region_in = st.selectbox(
                        tr("Région", "Region"),
                        options=["__DEFAULT__"] + (regions if regions else []),
                        format_func=lambda x: tr(
                            "Moyenne portefeuille",
                            "Portfolio average",
                        )
                        if x == "__DEFAULT__"
                        else x,
                    )
                    gar_in = st.selectbox(
                        tr("Garantie principale", "Main coverage"),
                        options=["__DEFAULT__"] + (garanties if garanties else []),
                        format_func=lambda x: tr(
                            "Moyenne portefeuille",
                            "Portfolio average",
                        )
                        if x == "__DEFAULT__"
                        else x,
                    )
                with col_cat2:
                    carburant_in = st.selectbox(
                        tr("Carburant", "Fuel"),
                        options=["__DEFAULT__"]
                        + (
                            sorted(df_f["carburant"].dropna().unique())
                            if "carburant" in df_f.columns
                            else []
                        ),
                        format_func=lambda x: tr(
                            "Moyenne portefeuille",
                            "Portfolio average",
                        )
                        if x == "__DEFAULT__"
                        else x,
                    )

                submitted = st.form_submit_button(
                    tr(
                        "📌 Calculer la prime pure prédit(e)",
                        "📌 Compute predicted pure premium",
                    )
                )

                if submitted:
                    user_inputs = {
                        "exposition": exposition_in,
                        "ageconducteur": agecond_in,
                        "agevehicule": ageveh_default if ageveh_in is None else ageveh_in,
                        "bonus": bonus_in,
                        "puissance": puiss_in,
                        "densite": dens_in,
                        "nbre": 1.0,
                        "region": region_in,
                        "garantie": gar_in,
                        "carburant": carburant_in,
                    }
                    try:
                        x_scoring = build_single_risk_row(
                            df_f,
                            features_info,
                            user_inputs,
                        )
                        model = models[model_name]
                        y_pred = model.predict(x_scoring)[0]
                        prime_pure_pred = float(max(y_pred, 0.0))

                        st.success(
                            tr(
                                f"Prime pure prédite (modèle {model_name}) : **{prime_pure_pred:,.0f} FCFA**",
                                f"Predicted pure premium ({model_name}) : **{prime_pure_pred:,.0f} FCFA**",
                            )
                        )

                        st.markdown(
                            tr(
                                "##### Simulation tarifaire (avec chargement et marge)",
                                "##### Pricing simulation (with loadings and margin)",
                            )
                        )

                        col_t1, col_t2, col_t3 = st.columns(3)
                        with col_t1:
                            chargement_in = st.number_input(
                                tr(
                                    "Chargements (frais, taxes, sécurité) en %",
                                    "Loadings (fees, taxes, safety) in %",
                                ),
                                min_value=0.0,
                                max_value=200.0,
                                value=30.0,
                                step=1.0,
                            )
                        with col_t2:
                            marge_in = st.number_input(
                                tr(
                                    "Marge technique / commerciale en %",
                                    "Technical / commercial margin in %",
                                ),
                                min_value=0.0,
                                max_value=200.0,
                                value=10.0,
                                step=1.0,
                            )
                        with col_t3:
                            coeff_com_in = st.number_input(
                                tr(
                                    "Coefficient de prudence (stress interne)",
                                    "Prudence coefficient (internal stress)",
                                ),
                                min_value=0.5,
                                max_value=3.0,
                                value=1.0,
                                step=0.05,
                            )

                        prime_comm = (
                            prime_pure_pred
                            * (1 + chargement_in / 100.0)
                            * (1 + marge_in / 100.0)
                            * coeff_com_in
                        )

                        st.info(
                            tr(
                                f"**Prime commerciale simulée** : **{prime_comm:,.0f} FCFA**",
                                f"**Simulated commercial premium**: **{prime_comm:,.0f} FCFA**",
                            )
                        )

                    except Exception as e:
                        st.error(
                            tr(
                                f"Erreur lors du scoring : {e}",
                                f"Error during scoring: {e}",
                            )
                        )

        st.markdown("---")
        st.markdown(
            tr(
                "#### 3. Stress tests sur le portefeuille filtré",
                "#### 3. Stress tests on the filtered portfolio",
            )
        )

        if (
            "prime_pure" not in df_f.columns
            or "frequence" not in df_f.columns
            or "severite" not in df_f.columns
        ):
            st.info(
                tr(
                    "Les colonnes 'prime_pure', 'frequence' et 'severite' sont nécessaires pour les stress tests.",
                    "'prime_pure', 'frequence' and 'severite' columns are required for stress tests.",
                )
            )
        else:
            base_kpis = kpis
            prime_info = base_kpis.get("prime_pure")
            base_prime_mean = (
                prime_info["mean"] if prime_info is not None else np.nan
            )

            st.write(
                tr(
                    f"Prime pure moyenne de base (portefeuille filtré) : **{base_prime_mean:,.2f}**",
                    f"Base pure premium mean (filtered portfolio): **{base_prime_mean:,.2f}**",
                )
            )

            if lang == "FR":
                scenarios_labels = [
                    "Fréquence +20%",
                    "Sévérité +30%",
                    "Fréquence +15% & Sévérité +15%",
                ]
            else:
                scenarios_labels = [
                    "Frequency +20%",
                    "Severity +30%",
                    "Frequency +15% & Severity +15%",
                ]

            scenario = st.selectbox(
                tr("Choisir un scénario de stress", "Choose a stress scenario"),
                options=scenarios_labels,
            )

            if st.button(
                tr("🚨 Appliquer le scénario de stress", "🚨 Apply stress scenario")
            ):
                df_stress = df_f.copy()
                if ("Fréquence +20%" in scenario) or ("Frequency +20%" in scenario):
                    df_stress["frequence"] = df_stress["frequence"] * 1.20
                elif ("Sévérité +30%" in scenario) or ("Severity +30%" in scenario):
                    df_stress["severite"] = df_stress["severite"] * 1.30
                else:
                    df_stress["frequence"] = df_stress["frequence"] * 1.15
                    df_stress["severite"] = df_stress["severite"] * 1.15

                df_stress["prime_pure"] = (
                    df_stress["frequence"] * df_stress["severite"]
                )
                stress_kpis = compute_kpis(df_stress)
                stress_prime_info = stress_kpis.get("prime_pure")
                stress_prime_mean = (
                    stress_prime_info["mean"]
                    if stress_prime_info is not None
                    else np.nan
                )
                delta = stress_prime_mean - base_prime_mean

                st.session_state["stress_results"][scenario] = {
                    "base_prime": base_prime_mean,
                    "stress_prime": stress_prime_mean,
                    "delta_prime": delta,
                }

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(tr("**Avant stress**", "**Before stress**"))
                    st.metric(
                        label=tr(
                            "Prime pure moyenne",
                            "Average pure premium",
                        ),
                        value=f"{base_prime_mean:,.2f}",
                    )
                with col_s2:
                    st.markdown(tr("**Après stress**", "**After stress**"))
                    st.metric(
                        label=tr(
                            "Prime pure moyenne stressée",
                            "Stressed average pure premium",
                        ),
                        value=f"{stress_prime_mean:,.2f}",
                        delta=f"{delta:,.2f}",
                    )

        st.markdown("---")
        st.markdown(
            tr(
                "#### 4. Export automatique de rapport PDF (niveau CA / Direction / clients / régulateur)",
                "#### 4. Automatic PDF export (Board / Management / clients / supervisor level)",
            )
        )

        if st.button(
            tr(
                "📄 Générer un rapport PDF détaillé (modèles, stress tests, recommandations)",
                "📄 Generate detailed PDF report (models, stress tests, recommendations)",
            )
        ):
            kpis_for_pdf = kpis
            metrics_for_pdf = st.session_state.get("model_metrics", None)
            stress_for_pdf = st.session_state.get("stress_results") or None
            pdf_bytes = generate_pdf_report(
                kpis_for_pdf,
                metrics_for_pdf,
                stress_for_pdf,
            )
            if pdf_bytes:
                st.download_button(
                    label=tr(
                        "⬇️ Télécharger le rapport PDF complet",
                        "⬇️ Download full PDF report",
                    ),
                    data=pdf_bytes,
                    file_name="rapport_actuariel_auto_CI.pdf",
                    mime="application/pdf",
                )

# ==========================================================
# TAB 6 — RÉSUMÉ EXÉCUTIF
# ==========================================================

with tab_exec:
    st.subheader(
        tr(
            "Résumé exécutif automatique — vue Direction & Top Management",
            "Automatic executive summary — ExCo & Top Management view",
        )
    )

    if df_f.empty:
        st.info(
            tr(
                "Aucune donnée filtrée : impossible de construire un résumé exécutif.",
                "No filtered data: cannot build executive summary.",
            )
        )
    else:
        sub_synth, sub_segments, sub_recos = st.tabs(
            [
                tr("🧾 Synthèse narrative", "🧾 Narrative synthesis"),
                tr("📌 Segments & concentration", "📌 Segments & concentration"),
                tr("🎯 Recommandations Direction", "🎯 Management recommendations"),
            ]
        )

        with sub_synth:
            st.markdown(
                '<h3 class="exec-subtitle">1. SYNTHÈSE NARRATIVE STRUCTURÉE</h3>',
                unsafe_allow_html=True,
            )
            if lang == "FR":
                st.markdown(
                    "Bloc directement exploitable dans un **rapport Word / PDF** ou une **note à la Direction / CA**."
                )
                st.markdown(ctx_text)
            else:
                st.info(
                    "The detailed narrative summary is currently available in **French** "
                    "for regulatory / actuarial wording quality. Switch to FR to see it."
                )

        with sub_segments:
            st.markdown(
                '<h3 class="exec-subtitle">2. SEGMENTS CLÉS ET CONCENTRATION</h3>',
                unsafe_allow_html=True,
            )

            col_seg1, col_seg2 = st.columns(2)
            with col_seg1:
                st.markdown(
                    tr(
                        "**Top régions par prime pure moyenne**",
                        "**Top regions by average pure premium**",
                    )
                )
                if {"region", "prime_pure"}.issubset(df_f.columns):
                    reg_stats = (
                        df_f.dropna(subset=["region", "prime_pure"])
                        .groupby("region", observed=False)
                        .agg(
                            Nb=("prime_pure", "size"),
                            Prime_moy=("prime_pure", "mean"),
                        )
                        .sort_values("Prime_moy", ascending=False)
                    )
                    if not reg_stats.empty:
                        reg_top = reg_stats.head(10).round(2)
                        st.dataframe(reg_top, use_container_width=True)
                        fig_exec_reg = px.bar(
                            reg_top.reset_index(),
                            x="region",
                            y="Prime_moy",
                            title=tr(
                                "Top régions — prime pure moyenne",
                                "Top regions — average pure premium",
                            ),
                        )
                        st.plotly_chart(fig_exec_reg, use_container_width=True)
                    else:
                        st.info(
                            tr(
                                "Aucune donnée exploitable pour les régions.",
                                "No exploitable data for regions.",
                            )
                        )
                else:
                    st.info(
                        tr(
                            "Variables 'region' ou 'prime_pure' manquantes.",
                            "'region' or 'prime_pure' variables missing.",
                        )
                    )

            with col_seg2:
                st.markdown(
                    tr(
                        "**Top garanties par prime pure moyenne**",
                        "**Top coverages by average pure premium**",
                    )
                )
                if {"garantie", "prime_pure"}.issubset(df_f.columns):
                    gar_stats = (
                        df_f.dropna(subset=["garantie", "prime_pure"])
                        .groupby("garantie", observed=False)
                        .agg(
                            Nb=("prime_pure", "size"),
                            Prime_moy=("prime_pure", "mean"),
                        )
                        .sort_values("Prime_moy", ascending=False)
                    )
                    if not gar_stats.empty:
                        gar_top = gar_stats.head(10).round(2)
                        st.dataframe(gar_top, use_container_width=True)
                        fig_exec_gar = px.bar(
                            gar_top.reset_index(),
                            x="garantie",
                            y="Prime_moy",
                            title=tr(
                                "Top garanties — prime pure moyenne",
                                "Top coverages — average pure premium",
                            ),
                        )
                        st.plotly_chart(fig_exec_gar, use_container_width=True)
                    else:
                        st.info(
                            tr(
                                "Aucune donnée exploitable pour les garanties.",
                                "No exploitable data for coverages.",
                            )
                        )
                else:
                    st.info(
                        tr(
                            "Variables 'garantie' ou 'prime_pure' manquantes.",
                            "'garantie' or 'prime_pure' variables missing.",
                        )
                    )

            st.markdown(
                tr(
                    "### Déciles et gradient de risque",
                    "### Deciles and risk gradient",
                )
            )
            dec = compute_deciles_table(df_f)
            if not dec.empty:
                st.dataframe(dec.round(2), use_container_width=True)
                d_low = dec.iloc[0]
                d_high = dec.iloc[-1]
                ratio = d_high["prime_pure"] / max(d_low["prime_pure"], 1e-6)
                fig_dec = px.bar(
                    dec,
                    x="decile",
                    y="prime_pure",
                    title=tr(
                        "Prime pure moyenne par décile de risque",
                        "Average pure premium by risk decile",
                    ),
                )
                st.plotly_chart(fig_dec, use_container_width=True)
                st.markdown(
                    tr(
                        f"- Le décile le plus risqué présente une prime pure moyenne d’environ "
                        f"**{d_high['prime_pure']:,.0f} FCFA**, contre **{d_low['prime_pure']:,.0f} FCFA** "
                        f"pour le décile le plus bas, soit un facteur d’environ **{ratio:,.1f}**.",
                        f"- The riskiest decile has an average pure premium of about "
                        f"**{d_high['prime_pure']:,.0f} FCFA**, versus **{d_low['prime_pure']:,.0f} FCFA** "
                        f"for the safest decile, i.e. a factor of around **{ratio:,.1f}**.",
                    )
                )
            else:
                st.info(
                    tr(
                        "Déciles non disponibles (données insuffisantes ou prime pure quasi constante).",
                        "Deciles not available (insufficient data or almost constant pure premium).",
                    )
                )

        with sub_recos:
            st.markdown(
                '<h3 class="exec-subtitle">3. RECOMMANDATIONS OPÉRATIONNELLES & STRATÉGIQUES</h3>',
                unsafe_allow_html=True,
            )
            if lang == "FR":
                st.markdown(
                    """
**3.1. Court terme — Ajustements rapides**

- Vérifier la cohérence des expositions et des sinistres sur les **régions / garanties les plus chargées**.
- Renforcer le **contrôle de qualité des données** sur les champs critiques : exposition, coût de sinistre, dates, GPS, type de garantie.
- Mettre en place un **suivi mensuel** des KPI : fréquence, sévérité, prime pure, segmentés par région et garantie.
- Documenter les hypothèses utilisées pour le **recalcul des métriques** (fréquence, sévérité, prime pure).

**3.2. Moyen terme — Pilotage technique**

- Utiliser les déciles de risque pour :
  - Identifier les **segments à forte sinistralité** (derniers déciles),
  - Définir des actions ciblées : ajustement tarifaire, renforcement des critères de souscription, actions de prévention, calibration des franchises.
- Mettre en place une **gouvernance des modèles** :
  - versionning des modèles (GLM, Régression, RandomForest),
  - procédures de recalibrage périodique,
  - backtesting et suivi de performance (R², RMSE, dérive).
- Développer un **tableau de bord de stress tests** (fréquence / sévérité).

**3.3. Long terme — Stratégie & Régulateur**

- Positionner ce dispositif comme un **outil central de tarification et de pilotage technique** pour l’assurance auto.
- Renforcer la compatibilité avec les attentes du **régulateur (CIMA / CRC)** : traçabilité des données, documentation des modèles, validation indépendante, archivage.
- Intégrer progressivement des dimensions supplémentaires :
  - score de risque client,
  - analyse géospatiale avancée,
  - scénarios macro-économiques / climatiques.
"""
                )
            else:
                st.markdown(
                    """
**3.1. Short term — Quick adjustments**

- Check consistency of exposures and claims on **most loaded regions / coverages**.
- Strengthen **data quality controls** on critical fields.
- Implement **monthly KPI monitoring** (frequency, severity, pure premium) by region and coverage.
- Document assumptions used for **metric recomputation**.

**3.2. Medium term — Technical steering**

- Use risk deciles to:
  - Identify **high-claims segments** (top deciles),
  - Define targeted actions: tariff adjustments, underwriting tightening, prevention, deductibles.
- Set up a **model governance framework** (versioning, periodic recalibration, backtesting).
- Build a **stress-testing dashboard** (frequency / severity).

**3.3. Long term — Strategy & Supervisor**

- Position this system as a **central pricing and technical steering tool**.
- Align with expectations of **CIMA / regional supervisor** (traceability, documentation, independent validation, archiving).
- Gradually integrate:
  - customer risk scoring,
  - advanced geospatial analysis,
  - macro / climate scenarios.
"""
                )

        st.markdown("---")
        st.markdown(
            tr(
                "#### 📄 Exporter un rapport global de l’activité (client, Direction, CA, régulateur)",
                "#### 📄 Export a global activity report (client, Management, Board, supervisor)",
            )
        )
        if st.button(
            tr(
                "📥 Générer le rapport PDF global (Résumé exécutif)",
                "📥 Generate global PDF report (Executive summary)",
            )
        ):
            kpis_for_pdf = kpis
            metrics_for_pdf = st.session_state.get("model_metrics", None)
            stress_for_pdf = st.session_state.get("stress_results") or None
            pdf_bytes = generate_pdf_report(
                kpis_for_pdf,
                metrics_for_pdf,
                stress_for_pdf,
            )
            if pdf_bytes:
                st.download_button(
                    label=tr(
                        "⬇️ Télécharger le rapport global (PDF)",
                        "⬇️ Download global report (PDF)",
                    ),
                    data=pdf_bytes,
                    file_name="rapport_global_activite_auto_CI.pdf",
                    mime="application/pdf",
                )

# ==========================================================
# TAB 7 — AGENT IA
# ==========================================================

with tab_agent:
    st.subheader(
        tr(
            "🧠 Agent IA — Analyste Actuariel, Régulateur & Assistant Général",
            "🧠 AI Agent — Actuarial Analyst, Regulator & General Assistant",
        )
    )

    data_ready = not df_f.empty
    models_ready = st.session_state.get("models") is not None
    groq_ready = GROQ_API_KEY is not None
    web_ready = TAVILY_API_KEY is not None

    st.markdown(
        f"""
<div>
    <span class="status-pill">
        {'🟢' if data_ready else '🟡'}
        {tr("Données filtrées :", "Filtered data:")}
        {'OK' if data_ready else tr('À vérifier', 'To be checked')}
    </span>
    <span class="status-pill">
        {'🟢' if models_ready else '🟡'}
        {tr("Modèles ML :", "ML models:")}
        {tr('entraînés', 'trained') if models_ready else tr('non entraînés', 'not trained')}
    </span>
    <span class="status-pill">
        {'🟢' if groq_ready else '🔴'}
        Groq API :
        {tr('clé détectée', 'key detected') if groq_ready else tr('clé manquante', 'missing key')}
    </span>
    <span class="status-pill">
        {'🟢' if web_ready else '🟡'}
        {tr('Web temps réel :', 'Real-time web:')}
        {tr('activé (Tavily)', 'enabled (Tavily)') if web_ready else tr('clé Tavily manquante', 'missing Tavily key')}
    </span>
    <span class="status-pill">
        🤖 LLM : llama-3.3-70b-versatile
    </span>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_info, tab_chat = st.tabs(
        [
            tr("ℹ️ Mode d'emploi", "ℹ️ How it works"),
            tr("💬 Conversation IA", "💬 AI Conversation"),
        ]
    )

    # --- MODE D'EMPLOI ---
    with tab_info:
        if lang == "FR":
            st.markdown(
                '<h3 style="color:#B22222;">1. Rôle de l’agent IA</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
- **Actuaire senior virtuel** : interprète les KPI, déciles, régions, garanties.
- **Assistant de rédaction** : produit des notes, rapports, synthèses exécutives.
- **Mode Audit / Régulateur** : se place dans la posture d’un contrôleur (CIMA / BCEAO).
- **Assistant général** : peut aussi répondre à des questions plus larges (assurance, actuariat, data science).
"""
            )
            st.markdown(
                '<h3 style="color:#B22222;">2. Sources d’information</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
1. Le **portefeuille filtré** (filtres à gauche).
2. Le **résumé exécutif actuariel** calculé automatiquement.
3. Les **modèles GLM / Régression / RandomForest** (si entraînés).
4. La **recherche web temps réel Tavily** (si activée).
"""
            )
            st.markdown(
                '<h3 style="color:#B22222;">3. Bonnes pratiques d’utilisation</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
- Sois précis dans ta question (périmètre, indicateurs, période, segment).
- Indique si tu veux un **rapport complet**, une **réponse courte** ou un **avis d’audit régulateur**.
- L’agent donne une **lecture de bonnes pratiques**, pas un avis juridique opposable.
"""
            )
        else:
            st.markdown(
                '<h3 style="color:#B22222;">1. Role of the AI agent</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
- **Virtual senior actuary**: interprets KPIs, deciles, regions, coverages.
- **Writing assistant**: produces notes, reports, executive summaries.
- **Audit / Supervisor mode**: adopts CIMA / BCEAO posture.
- **General assistant**: can answer broader questions (insurance, actuarial, data science).
"""
            )
            st.markdown(
                '<h3 style="color:#B22222;">2. Information sources</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
1. The **filtered portfolio** (sidebar filters).
2. The automatically computed **actuarial executive summary**.
3. The **GLM / Linear Regression / RandomForest models** (if trained).
4. **Real-time web search (Tavily)** (if enabled).
"""
            )
            st.markdown(
                '<h3 style="color:#B22222;">3. Best practices</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
- Be specific in your question (scope, indicators, period, segment).
- Specify whether you want a **full report**, a **short analytical answer** or a **regulator-style audit**.
- The agent provides **best-practice guidance**, not binding legal advice.
"""
            )

    # --- CONVERSATION IA ---
    with tab_chat:
        st.markdown(
            tr(
                "Utilise cet onglet pour dialoguer avec l’agent IA, configurer le mode de réponse "
                "et la recherche web.",
                "Use this tab to interact with the AI agent, configure response mode and web search.",
            )
        )

        col_mode1, col_mode2 = st.columns(2)
        if lang == "FR":
            answer_modes = {
                "Réponse analytique normale": "normal",
                "Rapport complet structuré": "report",
                "Mode Audit / Régulateur (BCEAO / CIMA)": "audit",
            }
        else:
            answer_modes = {
                "Analytical answer": "normal",
                "Full structured report": "report",
                "Audit / Regulator mode (BCEAO / CIMA)": "audit",
            }

        with col_mode1:
            answer_mode_label = st.radio(
                tr("Type de réponse souhaitée", "Desired answer type"),
                options=list(answer_modes.keys()),
                index=0,
            )
            answer_mode = answer_modes[answer_mode_label]

        with col_mode2:
            use_web = st.checkbox(
                tr(
                    "🌐 Autoriser la recherche web temps réel pour cette question",
                    "🌐 Allow real-time web search for this question",
                ),
                value=False,
                help=tr(
                    "Si coché et Tavily configuré, l’agent enrichit sa réponse avec une recherche web.",
                    "If checked and Tavily is configured, the agent enriches its answer with a web search.",
                ),
            )

        if "agent_messages" not in st.session_state:
            st.session_state["agent_messages"] = []
        if "agent_last_answer" not in st.session_state:
            st.session_state["agent_last_answer"] = ""

        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            history_scope = st.selectbox(
                tr("Historique affiché", "Displayed history"),
                options=[
                    tr("Derniers échanges (recommandé)", "Last turns (recommended)"),
                    tr("Historique complet", "Full history"),
                ],
                index=0,
            )
        with col_hist2:
            if st.button(
                tr(
                    "🗑️ Effacer l'historique",
                    "🗑️ Clear history",
                )
            ):
                st.session_state["agent_messages"] = []
                st.session_state["agent_last_answer"] = ""
                st.rerun()

        if history_scope.startswith("Derniers") or history_scope.startswith("Last"):
            history_to_show = st.session_state["agent_messages"][-8:]
        else:
            history_to_show = st.session_state["agent_messages"]

        for msg in history_to_show:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user"><b>{tr("Vous", "You")} :</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-bubble-assistant"><b>{tr("Agent IA", "AI Agent")} :</b> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Téléchargement de la dernière réponse
        if st.session_state["agent_last_answer"]:
            st.markdown(
                tr(
                    "##### 📄 Export de la dernière réponse de l’agent IA",
                    "##### 📄 Export of the last AI agent answer",
                )
            )
            st.download_button(
                label=tr(
                    "⬇️ Télécharger la réponse au format texte",
                    "⬇️ Download answer as text",
                ),
                data=st.session_state["agent_last_answer"].encode("utf-8"),
                file_name="rapport_agent_ia.txt",
                mime="text/plain",
            )

        # === Zone de saisie IA ===
        # IMPORTANT : ta version de Streamlit n'accepte PAS 'label' sur chat_input,
        # seulement 'placeholder' (en positionnel) + 'key'.
        user_prompt = st.chat_input(
            tr(
                "Pose ta question (portefeuille, modèles, assurance, audit BCEAO/CIMA, notions générales, rapport complet)…",
                "Ask your question (portfolio, models, insurance, BCEAO/CIMA audit, general concepts, full report)…",
            ),
            key="agent_chat_input",
        )

        if user_prompt:
            st.session_state["agent_messages"].append(
                {"role": "user", "content": user_prompt}
            )

            dashboard_context = ctx_text
            web_context = ""
            if use_web and web_ready:
                web_context = web_search_summary(user_prompt, max_results=4)

            if lang == "FR":
                base_instructions = f"""
Tu es un actuaire senior et expert en assurance automobile et tarification, travaillant pour un projet en Côte d'Ivoire (AfriAI / GROUPE KV7).

Tu disposes du résumé suivant du portefeuille filtré par l'utilisateur (en français) :
{dashboard_context}

{("Tu disposes également d'un résumé de recherche web temps réel (Tavily) sur la question de l'utilisateur :\n\n" + web_context) if web_context else ""}

CONSIGNES GÉNÉRALES :
- Réponds toujours en **français clair, structuré, pédagogique et professionnel**.
- Quand la question concerne les KPI, régions, garanties, déciles, explique en te basant sur le résumé du portefeuille ci-dessus.
- Quand la question est plus générale (finance, actuariat, assurance, data science), réponds comme un expert international.
- Si une information dépend d'internet en temps réel, précise que tu donnes une réponse générale et pas une information juridiquement opposable.
"""
                if answer_mode == "report":
                    mode_instructions = """
MODE RAPPORT COMPLET :
- Rédige un rapport structuré de type professionnel (2 à 4 pages) avec :
  1. Contexte et périmètre de l'étude
  2. Qualité des données et limites
  3. Analyse du portefeuille (KPI, régions, garanties, déciles, évolution temporelle)
  4. Lecture des stress tests / niveau de risque
  5. Recommandations techniques (tarification, souscription, prévention)
  6. Pistes stratégiques pour la Direction Technique / Direction Générale
- Utilise des titres, sous-titres numérotés, phrases courtes et percutantes.
- Garde un ton formel et orienté décision (comité technique, comité des risques, régulateur).
"""
                elif answer_mode == "audit":
                    mode_instructions = """
MODE AUDIT / RÉGULATEUR (BCEAO / CIMA) :
- Tu te places dans la posture d'un inspecteur ou contrôleur.
- Ta réponse doit :
  1. Identifier les forces et faiblesses du dispositif (données, gouvernance, modèles, stress tests).
  2. Mettre en avant les enjeux de conformité : bonnes pratiques de tarification, gouvernance des modèles, documentation, validation indépendante.
  3. Souligner les risques potentiels : sous-tarification, sur-tarification, discrimination, insuffisance des provisions techniques, risque de solvabilité, risque opérationnel.
  4. Formuler des recommandations structurées (court, moyen, long terme).
  5. Rester aligné avec l'esprit BCEAO/CIMA (prudence, stabilité financière, protection des assurés).
"""
                else:
                    mode_instructions = """
MODE RÉPONSE ANALYTIQUE NORMALE :
- Réponds de manière concise mais précise.
- Tu peux faire des listes à puces et de petits sous-titres si nécessaire.
"""
            else:
                base_instructions = f"""
You are a senior actuary and motor insurance pricing expert, working on a project in Côte d'Ivoire (AfriAI / GROUPE KV7).

You have the following **French** summary of the filtered portfolio:
{dashboard_context}

{("You also have a real-time web search (Tavily) summary on the user's question:\n\n" + web_context) if web_context else ""}

GENERAL GUIDELINES:
- Always answer in **clear, structured, professional English**.
- Use the portfolio summary when relevant.
- For general questions, answer as an international expert.
- If something requires real-time legal / regulatory info, mention that your answer is general guidance.
"""
                if answer_mode == "report":
                    mode_instructions = """
FULL REPORT MODE:
- Draft a structured professional report (2–4 pages) with sections, headings and clear recommendations.
"""
                elif answer_mode == "audit":
                    mode_instructions = """
AUDIT / REGULATOR MODE (BCEAO / CIMA):
- Adopt the posture of a supervisor or inspector and provide structured findings and recommendations.
"""
                else:
                    mode_instructions = """
ANALYTICAL ANSWER MODE:
- Answer concisely but precisely, with bullet points when useful.
"""

            system_content = base_instructions + "\n" + mode_instructions
            history = st.session_state["agent_messages"][-10:]
            messages_for_api = [{"role": "system", "content": system_content}] + history

            try:
                with st.spinner(
                    tr(
                        "L'agent IA réfléchit…",
                        "The AI agent is thinking…",
                    )
                ):
                    answer = call_groq_chat(messages_for_api)
                    st.session_state["agent_messages"].append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state["agent_last_answer"] = answer
                    st.rerun()
            except Exception as e:
                st.error(
                    tr(
                        f"Erreur lors de l'appel à l'API Groq : {e}",
                        f"Error when calling Groq API: {e}",
                    )
                )
