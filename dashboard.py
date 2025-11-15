# ==========================================================
# Dashboard Actuariel Automobile — Côte d’Ivoire
# AfriAI / GROUPE KV7 — Version clean (sans freq_in / sev_in, sans variable 'zone')
# Source data : KoboToolbox (API v2)
# ==========================================================

import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh  # ← Auto-refresh

# --- Modèles ML / GLM (scikit-learn) ---
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# CONFIG GÉNÉRALE
# ==========================================================

KOBO_BASE = st.secrets.get("KOBO_BASE", "https://kf.kobotoolbox.org")
KOBO_TOKEN = st.secrets["KOBO_TOKEN"]
ASSET_UID = st.secrets["ASSET_UID"]

API_DATA_URL = f"{KOBO_BASE}/api/v2/assets/{ASSET_UID}/data/?format=json&group_sep=/"
HEADERS = {"Authorization": f"Token {KOBO_TOKEN}"}

st.set_page_config(
    page_title="Dashboard Actuariel Auto — Côte d’Ivoire",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🔁 Auto-refresh toutes les 5 minutes (300 000 ms)
# En cohérence avec ttl=300s du cache de fetch_kobo_json
st_autorefresh(interval=300_000, key="kobo_autorefresh")

# ====== Style global (HTML/CSS léger) ======
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: "Georgia", "Times New Roman", serif;
    }
    h1, h2, h3 {
        color: #B22222;
    }
    .kpi-card {
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid #FFD70055;
        background-color: #FFFDF5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 8px;
    }
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #555;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #B22222;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.info("🛠 Version dashboard : v1.1 — Clean (sans freq_in / sev_in, sans 'zone')")

# ==========================================================
# CONSTANTES : VARIABLES CIBLES / FEATURES
# ==========================================================

TARGET_VARS = {
    "exposition", "puissance", "agevehicule", "ageconducteur", "bonus",
    "marque", "carburant", "densite", "region", "nbre", "garantie", "cout",
    "frequence", "severite", "prime_pure", "latitude", "longitude",
    "departement", "district", "date_enquete", "date_souscription",
    "date_sinistre", "precision_gps"
}

# Mapping souple : noms Kobo potentiels → noms cibles
RENAME_SOFT = {
    # "zone_geographique": "zone",  # supprimé : cette variable n'existe pas dans le questionnaire
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

# Facteurs de risque utilisés pour la modélisation de la prime pure
FEATURE_NUM_COLS = [
    "exposition", "agevehicule", "ageconducteur",
    "bonus", "puissance", "densite", "nbre"
]
FEATURE_CAT_COLS = ["region", "garantie", "carburant", "marque"]

TARGET_COL = "prime_pure"

# ==========================================================
# FONCTIONS UTILITAIRES
# ==========================================================

@st.cache_data(ttl=300)  # 300s = 5 min, cohérent avec l’auto-refresh
def fetch_kobo_json() -> pd.DataFrame:
    """Récupère les données Kobo brutes (JSON v2)."""
    r = requests.get(API_DATA_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    js = r.json()
    return pd.DataFrame(js.get("results", []))


def split_geopoint(df: pd.DataFrame, candidates=("geopoint", "coordonnees_gps", "gps")) -> pd.DataFrame:
    """Découpe un champ geopoint 'lat lon alt acc' en colonnes séparées."""
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
    """Nettoyage complet + recalcul fréquence / sévérité / prime pure."""
    df = df_raw.copy()

    # 1) Renommage souple
    for src, dst in RENAME_SOFT.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    # 2) Geopoint
    df = split_geopoint(df)

    # 3) Gestion virgules décimales
    decimal_cols = [
        "exposition", "bonus", "densite", "cout",
        "frequence", "severite", "prime_pure"
    ]
    for c in decimal_cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.replace(",", ".", regex=False)

    # 4) Conversion types
    float_cols = [
        "exposition", "bonus", "densite", "cout", "frequence",
        "severite", "prime_pure", "latitude", "longitude", "precision_gps"
    ]
    int_cols = ["puissance", "agevehicule", "ageconducteur", "nbre"]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    # 5) Dates
    for c in ["date_enquete", "date_souscription", "date_sinistre", "_submission_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 6) Catégorielles
    cat_cols = ["marque", "carburant", "region", "departement", "district", "garantie"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # 7) Recalculs actuariels
    if "frequence" not in df.columns and {"nbre", "exposition"}.issubset(df.columns):
        df["frequence"] = df.apply(
            lambda r: (r["nbre"] / r["exposition"]) if pd.notna(r["exposition"]) and r["exposition"] > 0 else 0,
            axis=1
        )
    if "severite" not in df.columns and {"cout", "nbre"}.issubset(df.columns):
        df["severite"] = df.apply(
            lambda r: (r["cout"] / r["nbre"]) if pd.notna(r["nbre"]) and r["nbre"] > 0 else 0,
            axis=1
        )
    if "prime_pure" not in df.columns and {"frequence", "severite"}.issubset(df.columns):
        df["prime_pure"] = df["frequence"] * df["severite"]

    # 8) Clipping bornes raisonnables
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
    """KPIs actuariels simples sur le portefeuille filtré."""
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

# ==========================================================
# FONCTIONS MODELES (GLM / ML) + SCORING
# ==========================================================

def train_models_on_df(df: pd.DataFrame):
    """
    Entraîne un GLM Tweedie et un RandomForest sur la prime pure
    à partir de facteurs de risque (sans fréquence/sévérité comme features).
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
            ("model", TweedieRegressor(power=1.5, link="log", alpha=0.0, max_iter=1000)),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    models = {}
    metrics = {}

    glm_model.fit(X, y)
    y_pred_glm = glm_model.predict(X)
    metrics["GLM_Tweedie"] = {
        "MAE": float(mean_absolute_error(y, y_pred_glm)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred_glm))),
        "R2": float(r2_score(y, y_pred_glm)),
        "n": int(len(y)),
    }
    models["GLM_Tweedie"] = glm_model

    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
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


def build_single_risk_row(df_ref: pd.DataFrame, features_info: dict, user_inputs: dict) -> pd.DataFrame:
    """Construit une ligne unique pour le scoring à partir des inputs utilisateur."""
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


def generate_pdf_report(kpis: dict, model_metrics: dict | None, stress_results: dict | None) -> bytes:
    """Génère un PDF simple (KPIs, modèles, stress tests)."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
    except ImportError:
        st.error(
            "Le module `reportlab` n'est pas installé. "
            "Installe-le avec : `pip install reportlab` pour activer l'export PDF."
        )
        return b""

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Rapport Actuariel — Auto Côte d'Ivoire")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # 1. KPIs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "1. KPIs portefeuille (données filtrées)")
    y -= 18
    c.setFont("Helvetica", 10)

    if kpis:
        c.drawString(50, y, f"- Nombre d'observations : {kpis.get('nb_obs', 'NA')}")
        y -= 14
        if kpis.get("prime_pure"):
            c.drawString(
                50, y,
                f"- Prime pure moyenne : {kpis['prime_pure']['mean']:.2f} "
                f"(p50={kpis['prime_pure']['p50']:.2f}, p95={kpis['prime_pure']['p95']:.2f})",
            )
            y -= 14
        if kpis.get("frequence"):
            c.drawString(
                50, y,
                f"- Fréquence moyenne : {kpis['frequence']['mean']:.4f}",
            )
            y -= 14
        if kpis.get("severite"):
            c.drawString(
                50, y,
                f"- Sévérité moyenne : {kpis['severite']['mean']:.2f}",
            )
            y -= 14
    else:
        c.drawString(50, y, "Aucun KPI disponible.")
        y -= 14

    y -= 10

    # 2. Modèles
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "2. Modèles GLM / Machine Learning")
    y -= 18
    c.setFont("Helvetica", 10)

    if model_metrics:
        for name, m in model_metrics.items():
            if name.startswith("_"):
                continue
            c.drawString(
                50, y,
                f"- {name} : R2={m['R2']:.3f} | RMSE={m['RMSE']:.2f} | "
                f"MAE={m['MAE']:.2f} (n={m['n']})"
            )
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
    else:
        c.drawString(50, y, "Aucun modèle entraîné dans cette session.")
        y -= 14

    y -= 10

    # 3. Stress tests
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "3. Stress tests (scénarios)")
    y -= 18
    c.setFont("Helvetica", 10)

    if stress_results:
        for label, vals in stress_results.items():
            c.drawString(50, y, f"- {label}")
            y -= 14
            c.drawString(
                70, y,
                f"Prime pure moyenne : base={vals['base_prime']:.2f} | "
                f"stress={vals['stress_prime']:.2f} | delta={vals['delta_prime']:.2f}"
            )
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
    else:
        c.drawString(50, y, "Aucun scénario de stress testé.")
        y -= 14

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

st.markdown(
    """
    <h1 style="text-align:center;"><b>Dashboard Actuariel Automobile — Côte d'Ivoire</b></h1>
    <p style="text-align:center; color:#555;">
    Données collectées via KoboToolbox — Projet GROUPE KV7 / AfriAI
    </p>
    <hr style="border:1px solid #FFD700; margin-top:0; margin-bottom:10px;" />
    """,
    unsafe_allow_html=True,
)

with st.spinner("Connexion à KoboToolbox et préparation des données…"):
    df_raw = fetch_kobo_json()

if df_raw.empty:
    st.warning("Aucune donnée disponible pour le moment.")
    st.stop()

df = prepare_dataframe(df_raw)

# ==========================================================
# BARRE LATÉRALE — FILTRES & INFO SCHÉMA
# ==========================================================

st.sidebar.title("⚙️ Filtres & Options")

date_col = "_submission_time" if "_submission_time" in df.columns else (
    "date_enquete" if "date_enquete" in df.columns else None
)

if date_col and pd.notna(df[date_col]).any():
    tmin = df[date_col].min()
    tmax = df[date_col].max()
    d1, d2 = st.sidebar.date_input(
        "Période d'analyse",
        [tmin.date(), tmax.date()],
        help="Filtre sur la date d'enquête (ou date de soumission)."
    )
else:
    d1 = d2 = None

regions = sorted(df["region"].dropna().unique()) if "region" in df.columns else []
region_sel = st.sidebar.multiselect("Région", regions, default=regions)

deps = sorted(df["departement"].dropna().unique()) if "departement" in df.columns else []
depart_sel = st.sidebar.multiselect("Département", deps, default=deps)

garanties = sorted(df["garantie"].dropna().unique()) if "garantie" in df.columns else []
gar_sel = st.sidebar.multiselect("Garantie", garanties, default=garanties)

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

with st.sidebar.expander("🔎 Schéma & Qualité des variables"):
    have = set(df.columns)
    missing = sorted(list(TARGET_VARS - have))
    extra = sorted(list(have - TARGET_VARS))
    st.write("**Variables cibles manquantes** :", missing if missing else "Aucune ✅")
    st.write("**Variables supplémentaires** :", extra if extra else "Aucune")
    st.write("**Types de données (échantillon)**")
    st.write(df.dtypes.astype(str))

# 🔘 Bouton manuel de rechargement Kobo (vide le cache + rerun)
if st.sidebar.button("🔄 Recharger les données Kobo"):
    fetch_kobo_json.clear()
    st.experimental_rerun()  # st.rerun() si tu es en version récente

# ==========================================================
# TABS PRINCIPAUX
# ==========================================================

tab_global, tab_tech, tab_map, tab_quality, tab_models = st.tabs(
    ["🏠 Vue globale", "📈 Analyse technique", "🗺️ Cartographie", "🧪 Qualité & Données", "🤖 Modèles & Scoring"]
)

# ==========================================================
# TAB 1 — VUE GLOBALE
# ==========================================================

with tab_global:
    st.subheader("Vue globale — Portefeuille filtré")

    kpis = compute_kpis(df_f)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Nombre d\'observations</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{kpis["nb_obs"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Exposition moyenne</div>', unsafe_allow_html=True)
        expo_info = kpis.get("exposition")
        expo_val = f"{expo_info['mean']:.2f}" if expo_info is not None else "—"
        st.markdown(f'<div class="kpi-value">{expo_val}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Fréquence moyenne</div>', unsafe_allow_html=True)
        if kpis["frequence"]:
            val = kpis["frequence"]["mean"]
            st.markdown(f'<div class="kpi-value">{val:.4f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-value">—</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Sévérité moyenne</div>', unsafe_allow_html=True)
        if kpis["severite"]:
            val = kpis["severite"]["mean"]
            st.markdown(f'<div class="kpi-value">{val:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-value">—</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Prime pure moyenne</div>', unsafe_allow_html=True)
        if kpis["prime_pure"]:
            val = kpis["prime_pure"]["mean"]
            st.markdown(f'<div class="kpi-value">{val:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-value">—</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Structure du portefeuille (par région et garantie)")
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
                title="Prime pure moyenne par région",
            )
            st.plotly_chart(fig_reg, use_container_width=True)
    else:
        c6.write("Pas assez de données pour l'analyse par région.")

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
                title="Prime pure moyenne par garantie",
            )
            st.plotly_chart(fig_gar, use_container_width=True)
    else:
        c7.write("Pas assez de données pour l'analyse par garantie.")

    st.markdown("### Table des observations filtrées")
    st.dataframe(df_f, use_container_width=True)

    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger le CSV filtré", csv, "actuariel_filtre.csv", "text/csv")

# ==========================================================
# TAB 2 — ANALYSE TECHNIQUE
# ==========================================================

with tab_tech:
    st.subheader("Analyse technique — fréquence, sévérité, prime pure")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Distribution de la prime pure**")
        if "prime_pure" in df_f.columns:
            fig = px.histogram(
                df_f,
                x="prime_pure",
                nbins=30,
                title="Distribution de la prime pure",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'prime_pure' manquante.")

    with colB:
        st.markdown("**Prime pure par région**")
        if {"prime_pure", "region"}.issubset(df_f.columns):
            fig = px.box(
                df_f.dropna(subset=["prime_pure", "region"]),
                x="region",
                y="prime_pure",
                points="outliers",
                title="Prime pure par région",
            )
            fig.update_layout(xaxis_title="Région", yaxis_title="Prime pure")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonnes 'prime_pure' ou 'region' manquantes.")

    st.markdown("### Fréquence & Sévérité — distributions")
    colC, colD = st.columns(2)

    with colC:
        st.markdown("**Fréquence (nbre / exposition)**")
        if "frequence" in df_f.columns:
            fig = px.histogram(df_f, x="frequence", nbins=30, title="Distribution de la fréquence")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'frequence' manquante.")

    with colD:
        st.markdown("**Sévérité (cout / nbre)**")
        if "severite" in df_f.columns:
            fig = px.histogram(df_f, x="severite", nbins=30, title="Distribution de la sévérité")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'severite' manquante.")

    st.markdown("### Évolution temporelle de la prime pure")
    if "prime_pure" in df_f.columns and date_col and date_col in df_f.columns:
        df_temp = df_f[[date_col, "prime_pure"]].dropna().sort_values(date_col)
        if not df_temp.empty:
            fig = px.line(
                df_temp,
                x=date_col,
                y="prime_pure",
                title="Prime pure dans le temps",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas de données temporelles suffisantes.")
    else:
        st.info("Colonnes temporelles / prime pure manquantes pour la série temporelle.")

# ==========================================================
# TAB 3 — CARTOGRAPHIE
# ==========================================================

with tab_map:
    st.subheader("Cartographie des observations")

    if {"latitude", "longitude"}.issubset(df_f.columns):
        st.markdown("**Vue simple (scatter géographique)**")
        st.map(df_f[["latitude", "longitude"]].dropna())

        st.markdown("**Synthèse par région (si GPS et région disponibles)**")
        if "region" in df_f.columns:
            df_geo_reg = (
                df_f.dropna(subset=["latitude", "longitude", "region"])
                .groupby("region", dropna=True, observed=False)
                .agg(
                    lat=("latitude", "mean"),
                    lon=("longitude", "mean"),
                    nb=("region", "size"),
                    prime_moy=("prime_pure", "mean")
                )
                .reset_index()
            )
            st.dataframe(df_geo_reg, use_container_width=True)
        else:
            st.info("Région non disponible pour la synthèse par région.")
    else:
        st.info("Pas de coordonnées GPS disponibles. Vérifie que le champ geopoint est bien rempli dans Kobo.")

# ==========================================================
# TAB 4 — QUALITÉ & DONNÉES
# ==========================================================

with tab_quality:
    st.subheader("Qualité des données & structure")

    st.markdown("### Aperçu des données brutes Kobo (10 premières lignes)")
    with st.expander("Données brutes KoboToolbox"):
        st.write("Nb de lignes brutes :", len(df_raw))
        st.write("Colonnes brutes :", list(df_raw.columns))
        st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("### Données après préparation (types & manquants)")

    colQ1, colQ2 = st.columns(2)

    with colQ1:
        st.markdown("**Types de données**")
        st.write(df.dtypes.astype(str))

    with colQ2:
        st.markdown("**Taux de valeurs manquantes (%)**")
        miss = df.isna().mean().sort_values(ascending=False) * 100
        st.write(miss.round(1))

    st.markdown("### Vérification par rapport au schéma cible")
    have = set(df.columns)
    missing = sorted(list(TARGET_VARS - have))
    extra = sorted(list(have - TARGET_VARS))

    cM1, cM2 = st.columns(2)
    with cM1:
        st.markdown("**Variables cibles manquantes**")
        if missing:
            st.write(missing)
        else:
            st.success("Aucune — schéma cible complet 🎯")

    with cM2:
        st.markdown("**Variables en plus (non prévues dans le schéma)**")
        if extra:
            st.write(extra)
        else:
            st.info("Aucune variable supplémentaire.")

    st.markdown("### Commentaires ingénierie des données")
    st.write(
        """
        - Les métriques actuarielles (fréquence, sévérité, prime pure) sont recalculées
          automatiquement si elles ne sont pas présentes dans les données Kobo.
        - Les variables numériques sont converties (gestion des virgules décimales) et
          contraintes dans des bornes raisonnables (exposition, bonus, âges, puissance).
        - La précision GPS permet de filtrer, si besoin, les points collectés avec un
          positionnement trop incertain (> 25–30 m).
        """
    )

# ==========================================================
# TAB 5 — MODELES, SCORING, STRESS & PDF
# ==========================================================

with tab_models:
    st.subheader("Modèles GLM / Machine Learning, Scoring & Stress tests")

    if df_f.empty:
        st.warning("Aucune donnée filtrée : impossible d'entraîner les modèles.")
    else:
        st.markdown("#### 1. Entraînement des modèles sur les données filtrées")

        if "models" not in st.session_state:
            st.session_state["models"] = None
            st.session_state["model_metrics"] = None

        if st.button("⚙️ (Re)entraîner GLM & RandomForest sur le portefeuille filtré"):
            with st.spinner("Entraînement des modèles (GLM & RandomForest)…"):
                try:
                    models, metrics = train_models_on_df(df_f)
                    st.session_state["models"] = models
                    st.session_state["model_metrics"] = metrics
                    st.success("Modèles entraînés avec succès sur les données filtrées. ✅")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement des modèles : {e}")

        metrics = st.session_state.get("model_metrics", None)
        models = st.session_state.get("models", None)

        if metrics:
            st.markdown("##### Performance des modèles (in-sample)")
            met_df = []
            for name, m in metrics.items():
                if name.startswith("_"):
                    continue
                met_df.append({
                    "Modèle": name,
                    "R2": m["R2"],
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "n": m["n"],
                })
            st.dataframe(pd.DataFrame(met_df), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 2. Scoring individuel & simulation tarifaire")

        if not models or not metrics:
            st.info("Entraîne d'abord les modèles pour activer le scoring.")
        else:
            model_name = st.selectbox(
                "Choix du modèle pour le scoring",
                options=list(models.keys()),
                index=0,
            )
            features_info = metrics.get("_features", {"num_cols": [], "cat_cols": []})

            with st.form("scoring_form"):
                st.markdown("##### Paramètres du risque à scorer")

                cc1, cc2 = st.columns(2)

                # Valeurs par défaut réalistes (clampées pour éviter tout conflit min/max)
                if "exposition" in df_f.columns and df_f["exposition"].notna().any():
                    expo_med = float(df_f["exposition"].median())
                else:
                    expo_med = 1.0
                expo_default = float(np.clip(expo_med, 0.01, 1.20))

                if "ageconducteur" in df_f.columns and df_f["ageconducteur"].notna().any():
                    agecond_med = int(df_f["ageconducteur"].median())
                else:
                    agecond_med = 35
                agecond_default = int(np.clip(agecond_med, 16, 90))

                if "agevehicule" in df_f.columns and df_f["agevehicule"].notna().any():
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
                        "Exposition (en années)",
                        min_value=0.01,
                        max_value=1.20,
                        value=expo_default,
                        step=0.01,
                    )
                    agecond_in = st.number_input(
                        "Âge conducteur",
                        min_value=16,
                        max_value=90,
                        value=agecond_default,
                    )
                    ageveh_in = st.number_input(
                        "Âge véhicule",
                        min_value=0,
                        max_value=40,
                        value=ageveh_default,
                    )

                with cc2:
                    bonus_in = st.number_input(
                        "Bonus-malus (coefficient)",
                        min_value=0.5,
                        max_value=3.5,
                        value=bonus_default,
                        step=0.05,
                    )
                    puiss_in = st.number_input(
                        "Puissance fiscale (CV)",
                        min_value=2,
                        max_value=40,
                        value=puiss_default,
                    )
                    dens_in = st.number_input(
                        "Densité zone (proxy risque)",
                        min_value=0.0,
                        max_value=1000.0,
                        value=dens_default,
                        step=1.0,
                    )

                col_cat1, col_cat2 = st.columns(2)

                with col_cat1:
                    region_in = st.selectbox(
                        "Région",
                        options=["__DEFAULT__"] + (regions if regions else []),
                        format_func=lambda x: "Moyenne portefeuille" if x == "__DEFAULT__" else x,
                    )
                    gar_in = st.selectbox(
                        "Garantie principale",
                        options=["__DEFAULT__"] + (garanties if garanties else []),
                        format_func=lambda x: "Moyenne portefeuille" if x == "__DEFAULT__" else x,
                    )

                with col_cat2:
                    carburant_in = st.selectbox(
                        "Carburant",
                        options=["__DEFAULT__"] + (
                            sorted(df_f["carburant"].dropna().unique())
                            if "carburant" in df_f.columns else []
                        ),
                        format_func=lambda x: "Moyenne portefeuille" if x == "__DEFAULT__" else x,
                    )

                submitted = st.form_submit_button("📌 Calculer la prime pure prédit(e)")

                if submitted:
                    user_inputs = {
                        "exposition": exposition_in,
                        "ageconducteur": agecond_in,
                        "agevehicule": ageveh_in,
                        "bonus": bonus_in,
                        "puissance": puiss_in,
                        "densite": dens_in,
                        "nbre": 1.0,  # par défaut 1 sinistre potentiel
                        "region": region_in,
                        "garantie": gar_in,
                        "carburant": carburant_in,
                    }

                    try:
                        x_scoring = build_single_risk_row(df_f, features_info, user_inputs)
                        model = models[model_name]
                        y_pred = model.predict(x_scoring)[0]
                        prime_pure_pred = float(max(y_pred, 0.0))

                        st.success(
                            f"Prime pure prédite (modèle {model_name}) : **{prime_pure_pred:,.0f} FCFA**"
                        )

                        st.markdown("##### Simulation tarifaire (avec chargement et marge)")

                        col_t1, col_t2, col_t3 = st.columns(3)
                        with col_t1:
                            chargement_in = st.number_input(
                                "Chargements (frais, taxes, sécurité) en %",
                                min_value=0.0,
                                max_value=200.0,
                                value=30.0,
                                step=1.0,
                            )
                        with col_t2:
                            marge_in = st.number_input(
                                "Marge technique / commerciale en %",
                                min_value=0.0,
                                max_value=200.0,
                                value=10.0,
                                step=1.0,
                            )
                        with col_t3:
                            coeff_com_in = st.number_input(
                                "Coefficient de prudence (stress interne)",
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
                            f"**Prime commerciale simulée** (après chargements, marge et prudence) : "
                            f"**{prime_comm:,.0f} FCFA**"
                        )

                    except Exception as e:
                        st.error(f"Erreur lors du scoring : {e}")

        st.markdown("---")
        st.markdown("#### 3. Stress tests sur le portefeuille filtré")

        stress_results = {}

        if "prime_pure" not in df_f.columns or "frequence" not in df_f.columns or "severite" not in df_f.columns:
            st.info("Les colonnes 'prime_pure', 'frequence' et 'severite' sont nécessaires pour les stress tests.")
        else:
            base_kpis = compute_kpis(df_f)
            base_prime_mean = base_kpis["prime_pure"]["mean"] if base_kpis["prime_pure"] else np.nan

            st.write(f"Prime pure moyenne de base (portefeuille filtré) : **{base_prime_mean:,.2f}**")

            scenario = st.selectbox(
                "Choisir un scénario de stress",
                options=[
                    "Fréquence +20%",
                    "Sévérité +30%",
                    "Fréquence +15% & Sévérité +15%",
                ],
            )

            if st.button("🚨 Appliquer le scénario de stress"):
                df_stress = df_f.copy()

                if scenario == "Fréquence +20%":
                    df_stress["frequence"] = df_stress["frequence"] * 1.20
                elif scenario == "Sévérité +30%":
                    df_stress["severite"] = df_stress["severite"] * 1.30
                elif scenario == "Fréquence +15% & Sévérité +15%":
                    df_stress["frequence"] = df_stress["frequence"] * 1.15
                    df_stress["severite"] = df_stress["severite"] * 1.15

                df_stress["prime_pure"] = df_stress["frequence"] * df_stress["severite"]
                stress_kpis = compute_kpis(df_stress)
                stress_prime_mean = stress_kpis["prime_pure"]["mean"] if stress_kpis["prime_pure"] else np.nan
                delta = stress_prime_mean - base_prime_mean

                stress_results[scenario] = {
                    "base_prime": base_prime_mean,
                    "stress_prime": stress_prime_mean,
                    "delta_prime": delta,
                }

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("**Avant stress**")
                    st.metric(
                        label="Prime pure moyenne",
                        value=f"{base_prime_mean:,.2f}",
                    )

                with col_s2:
                    st.markdown("**Après stress**")
                    st.metric(
                        label="Prime pure moyenne stressée",
                        value=f"{stress_prime_mean:,.2f}",
                        delta=f"{delta:,.2f}",
                    )

        st.markdown("---")
        st.markdown("#### 4. Export automatique de rapport PDF")

        if st.button("📄 Générer un rapport PDF (KPIs, modèles, stress tests)"):
            kpis_for_pdf = compute_kpis(df_f)
            metrics_for_pdf = st.session_state.get("model_metrics", None)

            pdf_bytes = generate_pdf_report(kpis_for_pdf, metrics_for_pdf, stress_results or None)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Télécharger le rapport PDF",
                    data=pdf_bytes,
                    file_name="rapport_actuariel_auto_CI.pdf",
                    mime="application/pdf",
                )
