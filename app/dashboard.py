# streamlit_app.py
# =====================================================================
# AfriAI — Dashboard Actuariel (Auto)
# - Menu horizontal (7 onglets)
# - Filtres UNIQUEMENT sur variables catégorielles (zone, carburant, garantie)
# - GLM Tweedie en direct (prime_pure ~ bonus + agevehicule + ageconducteur + puissance + densite)
# - HTML/CSS premium + Switch Thème (Sombre/Clair)
# - Export PDF (kaleido + reportlab)
# =====================================================================

import os
import io
import hashlib
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_percentage_error, mean_tweedie_deviance
)
from sklearn.linear_model import TweedieRegressor
from streamlit_option_menu import option_menu

# Optionnels (pour PDF). On gère l'absence proprement plus bas.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# -------------------------------
# Config générale
# -------------------------------
st.set_page_config(
    page_title="AfriAI — Actuariat Auto (GLM Tweedie)",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH = "Actuar0.csv"          # à la racine du projet
ASSETS_DIR = "assets"
CSS_DARK = os.path.join(ASSETS_DIR, "style_dark.css")
CSS_LIGHT = os.path.join(ASSETS_DIR, "style_light.css")

# Modélisation
FEATURES = ["bonus", "agevehicule", "ageconducteur", "puissance", "densite"]
TARGET = "prime_pure"
P_GRID = [round(x, 1) for x in np.arange(1.1, 2.0, 0.1)]  # 1.1→1.9
ALPHA_GRID = [0.0, 0.001, 0.01, 0.1, 1.0]
KFOLDS = 5

# ✅ CATEGORIAL FILTERS (marque & region retirés)
CATEGORICAL_CANDIDATES = ["zone", "carburant", "garantie"]

# -------------------------------
# CSS premium (deux thèmes)
# -------------------------------
DARK_CSS = """
/* === DARK THEME === */
.stApp, body { background:#0B1221 !important; color:#EAEFF7; font-family:'Poppins',sans-serif; }
a { color:#F5B301; text-decoration:none; }
div.stButton > button:first-child {
  background:linear-gradient(135deg,#F5B301,#E4A100); color:#0B1221; border:none; border-radius:10px;
  padding:0.6rem 1rem; font-weight:700; transition:all .25s ease-in-out;
}
div.stButton > button:first-child:hover { filter:brightness(1.1); transform:translateY(-1px); }

/* HEADER */
.header { position:sticky; top:0; z-index:10; background:#111A2E; border-radius:12px; padding:0.8rem 1rem;
          margin-bottom:1rem; box-shadow:0 6px 16px rgba(0,0,0,.35); }
.header h1 { color:#F5B301; margin:0; letter-spacing:.03em; text-transform:uppercase; font-size:1.6rem; }
.header p { margin:.2rem 0 0; color:#A9B8D1; }

/* TITLES */
h1, h2, h3, h4 {
  color:#F5B301; letter-spacing:.03em; text-transform:uppercase;
  margin-top:0 !important; margin-bottom:.5rem !important; line-height:1.3;
}

/* CARDS */
.card { background:#111A2E; border-radius:18px; padding:1.4rem; margin:1rem 0; box-shadow:0 4px 12px rgba(0,0,0,.35); }
.card:hover { transform:translateY(-2px); transition:transform .2s ease; }
.card h2, .card h3 { margin-top:0 !important; padding-top:0 !important; }

/* KPI */
.kpi { background:linear-gradient(145deg,#0E1830,#162040); border-left:5px solid #F5B301; border-radius:12px; padding:1rem;
       text-align:center; font-weight:700; }
.kpi .value { font-size:1.6rem; color:#FFFFFF; }
.kpi .label { color:#BFD1EA; font-size:.9rem; text-transform:uppercase; letter-spacing:.06em; }

/* TABLES */
table { background:#111A2E; color:#EAEFF7; border-collapse:collapse; border-radius:10px; overflow:hidden; width:100%; }
th { background:#F5B301; color:#0B1221; text-transform:uppercase; font-weight:700; }
td, th { padding:10px; border-bottom:1px solid #1C2742; }
tr:hover td { background:rgba(245,179,1,.05); }

/* FOOTER */
.footer { text-align:center; font-size:.9rem; color:#A9B8D1; margin-top:2rem; padding:1rem 0; border-top:1px solid #1C2742; }

/* ALIGN TITLES / BLOCKS */
main.block-container, section[data-testid="stVerticalBlock"] { padding-top:0 !important; margin-top:0 !important; }
[data-testid="column"] { align-items:flex-start !important; justify-content:flex-start !important; }

/* NAV WRAPPER */
.nav-wrapper { background:#111A2E; border-radius:12px; padding:.4rem .8rem; margin-bottom:1rem;
               box-shadow:0 6px 16px rgba(0,0,0,.35); }
"""

LIGHT_CSS = """
/* === LIGHT THEME === */
.stApp, body { background:#F7F9FC !important; color:#14213D; font-family:'Poppins',sans-serif; }
a { color:#B7791F; text-decoration:none; }
div.stButton > button:first-child {
  background:linear-gradient(135deg,#F59E0B,#D97706); color:#FFFFFF; border:none; border-radius:10px;
  padding:0.6rem 1rem; font-weight:700; transition:all .25s ease-in-out;
}
div.stButton > button:first-child:hover { filter:brightness(1.05); transform:translateY(-1px); }

/* HEADER */
.header { position:sticky; top:0; z-index:10; background:#FFFFFF; border-radius:12px; padding:0.8rem 1rem;
          margin-bottom:1rem; box-shadow:0 6px 16px rgba(0,0,0,.08); border:1px solid #E5E7EB; }
.header h1 { color:#111827; margin:0; letter-spacing:.02em; text-transform:uppercase; font-size:1.6rem; }
.header p { margin:.2rem 0 0; color:#6B7280; }

/* TITLES */
h1, h2, h3, h4 {
  color:#111827; letter-spacing:.02em; text-transform:uppercase;
  margin-top:0 !important; margin-bottom:.5rem !important; line-height:1.3;
}

/* CARDS */
.card { background:#FFFFFF; border-radius:16px; padding:1.2rem; margin:1rem 0;
        box-shadow:0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.03); border:1px solid #E5E7EB; }
.card:hover { transform:translateY(-2px); transition:transform .2s ease; }
.card h2, .card h3 { margin-top:0 !important; padding-top:0 !important; }

/* KPI */
.kpi { background:linear-gradient(145deg,#FFFFFF,#F3F4F6); border-left:5px solid #F59E0B; border-radius:12px; padding:1rem;
       text-align:center; font-weight:700; }
.kpi .value { font-size:1.6rem; color:#111827; }
.kpi .label { color:#6B7280; font-size:.9rem; text-transform:uppercase; letter-spacing:.06em; }

/* TABLES */
table { background:#FFFFFF; color:#111827; border-collapse:collapse; border-radius:10px; overflow:hidden; width:100%;
        border:1px solid #E5E7EB; }
th { background:#F59E0B; color:#FFFFFF; text-transform:uppercase; font-weight:700; }
td, th { padding:10px; border-bottom:1px solid #E5E7EB; }
tr:hover td { background:#FFF7ED; }

/* FOOTER */
.footer { text-align:center; font-size:.9rem; color:#6B7280; margin-top:2rem; padding:1rem 0; border-top:1px solid #E5E7EB; }

/* ALIGN TITLES / BLOCKS */
main.block-container, section[data-testid="stVerticalBlock"] { padding-top:0 !important; margin-top:0 !important; }
[data-testid="column"] { align-items:flex-start !important; justify-content:flex-start !important; }

/* NAV WRAPPER */
.nav-wrapper { background:#FFFFFF; border-radius:12px; padding:.4rem .8rem; margin-bottom:1rem; border:1px solid #E5E7EB; }
"""

def ensure_css_files():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if not os.path.exists(CSS_DARK):
        with open(CSS_DARK, "w", encoding="utf-8") as f:
            f.write(DARK_CSS)
    if not os.path.exists(CSS_LIGHT):
        with open(CSS_LIGHT, "w", encoding="utf-8") as f:
            f.write(LIGHT_CSS)

def inject_css(theme: str):
    ensure_css_files()
    css_path = CSS_DARK if theme == "Sombre" else CSS_LIGHT
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------
# Utils
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # Force sep=";" (dataset initial à points-virgules)
    return pd.read_csv(path, sep=";")

def cast_and_flag_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numériques pour features & cible, cast 'category' pour variables catégorielles connues si présentes."""
    # Cast chiffres (features + cible)
    cast_map = {
        "bonus": "Int64",
        "agevehicule": "Int64",
        "ageconducteur": "Int64",
        "puissance": "Int64",
        "densite": "Int64",
        TARGET: "float64",
    }
    for col, tp in cast_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(tp)

    # Cast catégorielles candidates -> category (si présentes)
    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Supprimer valeurs négatives sur colonnes censées être ≥0
    nonneg_cols = set(FEATURES + [TARGET]) & set(df.columns)
    for c in nonneg_cols:
        df = df[df[c].isna() | (df[c] >= 0)]
    # Drop NA sur features et cible
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

def get_categorical_filters(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Retourne un dict {col_cat: [valeurs sélectionnées]} pour les colonnes catégorielles existantes (zone, carburant, garantie)."""
    selections = {}
    with st.sidebar:
        st.subheader("Apparence")
        theme = st.radio("Thème", ["Sombre", "Clair"], horizontal=True, index=0, key="theme_radio")
        st.session_state["__theme__"] = theme

        st.subheader("Filtres catégoriels")
        cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
        if not cat_cols:
            st.info("Aucune variable catégorielle détectée parmi : zone, carburant, garantie.")
        for c in cat_cols:
            if pd.api.types.is_categorical_dtype(df[c]):
                cats = [str(x) for x in df[c].cat.categories]
            else:
                cats = sorted(map(str, df[c].dropna().unique().tolist()))
            default = cats  # tout sélectionné par défaut
            selections[c] = st.multiselect(f"{c}", cats, default=default, key=f"f_{c}")
    return selections

def apply_categorical_filters(df: pd.DataFrame, selections: Dict[str, List[str]]) -> pd.DataFrame:
    """Filtre le DataFrame uniquement sur les colonnes catégorielles selon selections (si vide => ne filtre pas)."""
    if not selections:
        return df
    mask = pd.Series(True, index=df.index)
    for col, chosen in selections.items():
        if col in df.columns and len(chosen) > 0:
            mask &= df[col].astype(str).isin(set(map(str, chosen)))
    return df.loc[mask].copy()

def hash_dataframe(df: pd.DataFrame) -> str:
    with io.BytesIO() as buffer:
        df.to_parquet(buffer, index=False)
        return hashlib.md5(buffer.getvalue()).hexdigest()

def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    y = df[TARGET].astype(float)
    return {
        "N": len(df),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "p10": float(y.quantile(0.10)),
        "p90": float(y.quantile(0.90)),
        "zeros_share": float((y == 0).mean()),
    }

def make_deciles_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    q = pd.qcut(y_pred, q=10, duplicates="drop")
    dfc = pd.DataFrame({"bin": q, "y": y_true, "yp": y_pred})
    out = dfc.groupby("bin", observed=True).agg(
        real=("y", "mean"), pred=("yp", "mean"), n=("y", "size")
    ).reset_index()
    out["decile"] = np.arange(1, len(out) + 1)
    return out

def metrics_report(y_true, y_pred, power: float) -> Dict[str, float]:
    dev = mean_tweedie_deviance(y_true, y_pred, power=power)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred + 1e-12))
    r2 = float(r2_score(y_true, y_pred))
    return {"deviance": dev, "rmse": rmse, "mape": mape, "r2": r2}

def fit_cv_tweedie(
    X: pd.DataFrame, y: pd.Series,
    p_grid, alpha_grid, n_splits: int = 5, random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float], TweedieRegressor]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for p in p_grid:
        for alpha in alpha_grid:
            preds_all, truth_all = [], []
            for tr_idx, te_idx in kf.split(X):
                Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
                ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

                model = TweedieRegressor(power=p, link='log', alpha=alpha)
                model.fit(Xtr, ytr)
                yhat = np.maximum(model.predict(Xte), 1e-12)
                preds_all.append(yhat)
                truth_all.append(yte.values)

            y_pred = np.concatenate(preds_all)
            y_true = np.concatenate(truth_all)
            m = metrics_report(y_true, y_pred, power=p)
            rows.append({"power": p, "alpha": alpha, **m})

    cv_df = pd.DataFrame(rows).sort_values(["deviance", "rmse"], ascending=True)
    best_row = cv_df.iloc[0].to_dict()
    best_model = TweedieRegressor(power=best_row["power"], link='log', alpha=best_row["alpha"])
    best_model.fit(X, y)
    return cv_df, best_row, best_model

def statsmodels_inference(X: pd.DataFrame, y: pd.Series, power: float) -> pd.DataFrame:
    X_sm = sm.add_constant(X, has_constant="add")
    fam = sm.families.Tweedie(var_power=power, link=sm.families.links.log())
    glm = sm.GLM(y, X_sm, family=fam)
    res = glm.fit()
    summ = res.summary2().tables[1].reset_index().rename(columns={"index": "param"})
    summ = summ.rename(columns={
        "Coef.": "coef",
        "Std.Err.": "std_err",
        "z": "z",
        "P>|z|": "p_value",
        "[0.025": "ci_low",
        "0.975]": "ci_high",
    })
    return summ[["param", "coef", "std_err", "z", "p_value", "ci_low", "ci_high"]]

# ---------- Export PDF helpers ----------
def _save_plotly_image(fig, path_png: str) -> bool:
    try:
        fig.write_image(path_png)  # nécessite kaleido
        return True
    except Exception as e:
        st.warning(f"Export image Plotly indisponible (kaleido manquant ?) : {e}")
        return False

def export_pdf_report(pdf_path: str, meta: Dict, png_paths: List[str]) -> bool:
    if not REPORTLAB_OK:
        st.error("reportlab n'est pas installé. Installe :  pip install --no-cache-dir reportlab")
        return False
    try:
        c = canvas.Canvas(pdf_path, pagesize=A4)
        W, H = A4
        y = H - 40

        # En-tête
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Rapport — GLM Tweedie (Prime Pure)")
        y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Date: {meta.get('date','')}")
        y -= 14
        c.drawString(40, y, f"Dataset hash: {meta.get('hash','')}")
        y -= 14
        c.drawString(40, y, f"Filtres catégoriels: {meta.get('filters','')}")
        y -= 18
        if meta.get("model"):
            c.drawString(40, y, f"Modèle: power={meta['model'].get('power')} | alpha={meta['model'].get('alpha')}")
            y -= 14
            c.drawString(40, y, f"Scores: Dev={meta['model'].get('deviance'):.4f} | RMSE={meta['model'].get('rmse'):.4f} | "
                                 f"MAPE={meta['model'].get('mape')*100:.2f}% | R²={meta['model'].get('r2'):.4f}")
            y -= 20

        # Images
        for p in png_paths:
            try:
                img = ImageReader(p)
                # largeur max 520 px environ -> 520/72 ~ 7.2 inches ; A4 landscape ~ ok ; ici on reste en portrait
                iw, ih = img.getSize()
                max_w = W - 80
                scale = min(max_w / iw, 300 / ih)  # 300 px de haut environ
                w, h = iw * scale, ih * scale
                if y - h < 40:
                    c.showPage()
                    y = H - 40
                c.drawImage(img, 40, y - h, width=w, height=h)
                y -= (h + 18)
            except Exception as e:
                st.warning(f"Impossible d'insérer {p} : {e}")

        c.showPage()
        c.save()
        return True
    except Exception as e:
        st.error(f"Erreur export PDF : {e}")
        return False

# -------------------------------
# Header + Thème + Données
# -------------------------------
# Thème (appliqué dès maintenant pour styliser tout)
st.sidebar.markdown("### Apparence")
theme_choice = st.sidebar.radio("Thème", ["Sombre", "Clair"], horizontal=True, index=0, key="__theme__")
inject_css(theme_choice)

st.markdown("""
<div class='header'>
  <h1>🚗 Tableau de bord Actuariel — GLM Tweedie</h1>
  <p>Module de modélisation en direct — AfriAI</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Chargement des données..."):
    try:
        df_raw = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {DATA_PATH}. Place le CSV à la racine du projet.")
        st.stop()

df = df_raw.copy()
df = cast_and_flag_categoricals(df)
df = basic_clean(df)

required = set(FEATURES + [TARGET])
if not required.issubset(df.columns):
    st.error(
        "Colonnes requises manquantes. Requises : "
        f"{sorted(required)}. Présentes : {sorted(df.columns.tolist())}"
    )
    st.stop()

# -------------------------------
# Filtres CATEGORIELS (sidebar)
# -------------------------------
cat_selections = {}
st.sidebar.markdown("### Filtres catégoriels")
for c in [x for x in CATEGORICAL_CANDIDATES if x in df.columns]:
    vals = sorted(map(str, df[c].dropna().unique().tolist()))
    sel = st.sidebar.multiselect(c, vals, default=vals, key=f"f_{c}")
    cat_selections[c] = sel

def apply_categorical_filters(df: pd.DataFrame, selections: Dict[str, List[str]]) -> pd.DataFrame:
    if not selections:
        return df
    mask = pd.Series(True, index=df.index)
    for col, chosen in selections.items():
        if col in df.columns and len(chosen) > 0:
            mask &= df[col].astype(str).isin(set(map(str, chosen)))
    return df.loc[mask].copy()

df_f = apply_categorical_filters(df, cat_selections)
dataset_hash = hash_dataframe(df_f)

# -------------------------------
# Menu horizontal (7 onglets)
# -------------------------------
with st.container():
    st.markdown("<div class='nav-wrapper'>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=[
            "Accueil & Synthèse", "Analyse du Portefeuille", "Modélisation GLM Tweedie",
            "Simulateur Tarifaire", "Performance & Sinistralité", "Qualité & Gouvernance",
            "Rapport & Export"
        ],
        icons=[
            "house", "bar-chart-line", "calculator",
            "sliders", "speedometer", "shield-check",
            "file-earmark-arrow-down"
        ],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#F59E0B" if theme_choice == "Clair" else "#F5B301", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px", "text-align": "center", "margin": "0px",
                "color": "#111827" if theme_choice == "Clair" else "#EAEFF7", "padding": "6px 12px"
            },
            "nav-link-selected": {"background-color": "#F3F4F6" if theme_choice == "Clair" else "#1A2440"},
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================== PAGES =================================

def page_accueil(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>🏁 Accueil & Synthèse</h2></div>", unsafe_allow_html=True)
    k = compute_kpis(dfv)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    items = [
        ("N", k["N"]),
        ("Moy. prime_pure", f"{k['mean']:.2f}"),
        ("Médiane", f"{k['median']:.2f}"),
        ("P10", f"{k['p10']:.2f}"),
        ("P90", f"{k['p90']:.2f}"),
        ("% Zéros", f"{100*k['zeros_share']:.1f}%"),
    ]
    for c, (lab, val) in zip([c1, c2, c3, c4, c5, c6], items):
        c.markdown(f"<div class='kpi'><div class='value'>{val}</div><div class='label'>{lab}</div></div>",
                   unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("<div class='card'><h3>Distribution — prime_pure</h3></div>", unsafe_allow_html=True)
        st.plotly_chart(px.histogram(dfv, x=TARGET, nbins=50), use_container_width=True)
    with g2:
        st.markdown("<div class='card'><h3>Répartition — Variables catégorielles</h3></div>", unsafe_allow_html=True)
        cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in dfv.columns]
        if cat_cols:
            col = st.selectbox("Catégorie", cat_cols, index=0)
            st.plotly_chart(px.histogram(dfv, x=col), use_container_width=True)
        else:
            st.info("Aucune variable catégorielle disponible (zone, carburant, garantie).")

def page_portefeuille(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>📊 Analyse du Portefeuille</h2></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if {"zone", "carburant"}.issubset(dfv.columns):
            st.markdown("<div class='card'><h3>Heatmap prime_pure — zone × carburant</h3></div>", unsafe_allow_html=True)
            piv = dfv.pivot_table(index="zone", columns="carburant", values=TARGET, aggfunc="mean")
            st.plotly_chart(px.imshow(piv, text_auto=".2f", aspect="auto"), use_container_width=True)
        else:
            st.info("Ajoute 'zone' et 'carburant' pour la heatmap.")
    with c2:
        if {"zone", "garantie"}.issubset(dfv.columns):
            st.markdown("<div class='card'><h3>Prime moyenne — zone × garantie</h3></div>", unsafe_allow_html=True)
            piv2 = dfv.pivot_table(index="zone", columns="garantie", values=TARGET, aggfunc="mean")
            st.plotly_chart(px.imshow(piv2, text_auto=".2f", aspect="auto"), use_container_width=True)
        else:
            st.info("Ajoute 'zone' et 'garantie' pour la vue croisée.")

    st.markdown("<div class='card'><h3>Vue tabulaire (échantillon)</h3></div>", unsafe_allow_html=True)
    st.dataframe(dfv.head(100), use_container_width=True)

def page_modelisation(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>🧮 Modélisation GLM Tweedie (direct)</h2></div>", unsafe_allow_html=True)
    X = dfv[FEATURES].astype(float)
    y = dfv[TARGET].astype(float)

    colp, cola = st.columns(2)
    with colp:
        p_values = st.multiselect("Puissance Tweedie (p)", P_GRID, default=P_GRID)
    with cola:
        alpha_values = st.multiselect("Ridge (alpha)", ALPHA_GRID, default=ALPHA_GRID)

    if st.button("🔁 Entraîner GLM Tweedie"):
        if len(dfv) < KFOLDS:
            st.error(f"Échantillon insuffisant pour KFold={KFOLDS}. N={len(dfv)}")
            st.stop()
        with st.spinner("Validation croisée + fit final..."):
            cv_df, best_row, best_model = fit_cv_tweedie(X, y, p_values, alpha_values, n_splits=KFOLDS)

        st.success(
            f"Meilleur modèle → power={best_row['power']} | alpha={best_row['alpha']} | "
            f"Deviance={best_row['deviance']:.4f} | RMSE={best_row['rmse']:.4f}"
        )
        st.session_state["best_model"] = best_model
        st.session_state["best_power"] = best_row["power"]

        st.markdown("<div class='card'><h3>Résultats Cross-Validation</h3></div>", unsafe_allow_html=True)
        st.dataframe(cv_df.reset_index(drop=True), use_container_width=True)

        piv = cv_df.pivot_table(index="power", columns="alpha", values="deviance")
        st.plotly_chart(px.imshow(piv, text_auto=".2f", aspect="auto",
                                  title="Deviance moyenne (plus bas est meilleur)"),
                        use_container_width=True)

        yhat = np.maximum(best_model.predict(X), 1e-12)
        mrep = metrics_report(y.values, yhat, power=best_row["power"])
        st.markdown(
            f"<div class='card'><h3>Scores (fit final)</h3>"
            f"<p>Deviance={mrep['deviance']:.4f} • RMSE={mrep['rmse']:.4f} • "
            f"MAPE={100*mrep['mape']:.2f}% • R²={mrep['r2']:.4f}</p></div>",
            unsafe_allow_html=True
        )

        cal = make_deciles_frame(y.values, yhat)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=cal["decile"], y=cal["real"], mode="lines+markers", name="Réel"))
        fig_cal.add_trace(go.Scatter(x=cal["decile"], y=cal["pred"], mode="lines+markers", name="Prédit"))
        fig_cal.update_layout(title="Calibration par déciles", xaxis_title="Déciles", yaxis_title="Prime pure moyenne")
        st.plotly_chart(fig_cal, use_container_width=True)

        fig_sc = px.scatter(x=y, y=yhat, labels={'x': 'Réel', 'y': 'Prédit'}, title="Réel vs Prédit")
        fig_sc.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", name="Idéal"))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.session_state["fig_cal"] = fig_cal
        st.session_state["fig_scatter"] = fig_sc
        st.session_state["last_scores"] = mrep

        st.markdown("<div class='card'><h3>Inférence (Statsmodels GLM Tweedie)</h3></div>", unsafe_allow_html=True)
        inf_df = statsmodels_inference(X, y, power=best_row["power"])
        st.dataframe(inf_df, use_container_width=True)

        out_preds = dfv[FEATURES + [TARGET]].copy()
        out_preds["yhat"] = yhat
        out_preds["dataset_hash"] = hash_dataframe(dfv)
        buf = io.StringIO()
        out_preds.to_csv(buf, index=False)
        st.download_button("⬇️ Télécharger prédictions (CSV)", buf.getvalue().encode("utf-8"),
                           file_name="predictions_tweedie.csv", mime="text/csv")

def page_simulateur(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>🧮 Simulateur Tarifaire</h2></div>", unsafe_allow_html=True)
    if "best_model" not in st.session_state:
        st.info("Entraîne d’abord le modèle dans l’onglet « Modélisation GLM Tweedie ».") 
        return

    best_model = st.session_state["best_model"]

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1:
        s_bonus = st.number_input("bonus", int(dfv["bonus"].min()), int(dfv["bonus"].max()), int(dfv["bonus"].median()))
    with cc2:
        s_agev = st.number_input("agevehicule", int(dfv["agevehicule"].min()), int(dfv["agevehicule"].max()),
                                 int(dfv["agevehicule"].median()))
    with cc3:
        s_agec = st.number_input("ageconducteur", int(dfv["ageconducteur"].min()), int(dfv["ageconducteur"].max()),
                                 int(dfv["ageconducteur"].median()))
    with cc4:
        s_puis = st.number_input("puissance", int(dfv["puissance"].min()), int(dfv["puissance"].max()),
                                 int(dfv["puissance"].median()))
    with cc5:
        s_dens = st.number_input("densite", int(dfv["densite"].min()), int(dfv["densite"].max()),
                                 int(dfv["densite"].median()))

    X0 = pd.DataFrame([{
        "bonus": s_bonus, "agevehicule": s_agev, "ageconducteur": s_agec,
        "puissance": s_puis, "densite": s_dens
    }]).astype(float)

    pred0 = float(np.maximum(best_model.predict(X0)[0], 1e-12))
    st.markdown(f"<div class='kpi'><div class='value'>{pred0:.4f}</div><div class='label'>Prime prédite</div></div>",
                unsafe_allow_html=True)

    st.markdown("**What-if — variations**")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1: dv_b = st.slider("Δ bonus", -10, 10, 0)
    with w2: dv_av = st.slider("Δ agevehicule", -5, 5, 0)
    with w3: dv_ac = st.slider("Δ ageconducteur", -5, 5, 0)
    with w4: dv_pu = st.slider("Δ puissance", -10, 10, 0)
    with w5: dv_de = st.slider("Δ densite", -10, 10, 0)

    X1 = X0.copy()
    X1.loc[0, "bonus"] += dv_b
    X1.loc[0, "agevehicule"] += dv_av
    X1.loc[0, "ageconducteur"] += dv_ac
    X1.loc[0, "puissance"] += dv_pu
    X1.loc[0, "densite"] += dv_de

    pred1 = float(np.maximum(best_model.predict(X1)[0], 1e-12))
    comp = pd.DataFrame({"Scenario": ["Initial", "What-if"], "prime_predite": [pred0, pred1]})
    st.plotly_chart(px.bar(comp, x="Scenario", y="prime_predite", text_auto=".4f",
                           title="Comparaison avant / après"), use_container_width=True)

def page_performance(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>📈 Performance & Sinistralité</h2></div>", unsafe_allow_html=True)
    if {"zone", "garantie"}.issubset(dfv.columns):
        piv = dfv.pivot_table(index="zone", columns="garantie", values=TARGET, aggfunc="mean")
        st.plotly_chart(px.imshow(piv, text_auto=".2f", aspect="auto", title="Prime moyenne — zone × garantie"),
                        use_container_width=True)
    if {"zone"}.issubset(dfv.columns):
        st.plotly_chart(px.box(dfv, x="zone", y=TARGET, points="outliers", title="Prime pure par zone"),
                        use_container_width=True)
    if {"carburant"}.issubset(dfv.columns):
        st.plotly_chart(px.box(dfv, x="carburant", y=TARGET, points="outliers", title="Prime pure par carburant"),
                        use_container_width=True)

    # Top segments (sur catégorielles restantes)
    group_cols = [c for c in CATEGORICAL_CANDIDATES if c in dfv.columns]
    if group_cols:
        st.dataframe(
            dfv.groupby(group_cols).agg(n=("prime_pure", "size"), prime_moy=("prime_pure", "mean"))
               .reset_index().sort_values("prime_moy", ascending=False).head(50),
            use_container_width=True
        )

def page_qualite(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>🛡️ Qualité & Gouvernance</h2></div>", unsafe_allow_html=True)
    missing = dfv.isna().mean().sort_values(ascending=False)
    st.markdown("<div class='card'><h3>Valeurs manquantes (%)</h3></div>", unsafe_allow_html=True)
    st.plotly_chart(px.bar(missing[missing > 0]*100, title="Taux de valeurs manquantes (%)"),
                    use_container_width=True)

    # Cardinalité des catégorielles restantes
    cats = [c for c in CATEGORICAL_CANDIDATES if c in dfv.columns]
    if cats:
        card = {c: int(dfv[c].nunique()) for c in cats}
        st.markdown("<div class='card'><h3>Cardinalité (catégorielles)</h3></div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"colonne": list(card.keys()), "cardinalite": list(card.values())}),
                     use_container_width=True)

    st.markdown("<div class='card'><h3>Traçabilité</h3></div>", unsafe_allow_html=True)
    st.write({
        "dataset_hash": hash_dataframe(dfv),
        "N": len(dfv),
        "colonnes": list(dfv.columns),
        "filtres_categorielles": {k: v for k, v in cat_selections.items() if v},
    })

def page_rapport(dfv: pd.DataFrame):
    st.markdown("<div class='card'><h2>📘 Rapport & Export</h2></div>", unsafe_allow_html=True)
    st.markdown("""
    - **Méthodo** : GLM Tweedie (lien log), grille p∈[1.1,1.9], alpha∈{0,0.001,0.01,0.1,1.0}, KFold=5.  
    - **Critère** : Deviance moyenne (plus bas = meilleur), tie-break RMSE.  
    - **Inférence** : GLM Statsmodels pour coefficients, p-values, IC95%.  
    - **Features** : bonus, agevehicule, ageconducteur, puissance, densite (uniquement).  
    """)
    # Export CSV (jeu filtré)
    buf = io.StringIO()
    dfv.to_csv(buf, index=False)
    st.download_button("⬇️ Exporter le portefeuille filtré (CSV)",
                       buf.getvalue().encode("utf-8"),
                       file_name="portefeuille_filtre.csv",
                       mime="text/csv")

    st.markdown("### Export PDF")
    need = []
    try:
        import kaleido  # type: ignore
        _KALEIDO = True
    except Exception:
        _KALEIDO = False
        need.append("kaleido==0.2.1")

    if not REPORTLAB_OK:
        need.append("reportlab")

    if need:
        st.warning(
            "Pour l’export PDF automatique, installe :\n\n"
            + "\n".join([f"- pip install --no-cache-dir {pkg}" for pkg in need])
        )

    # Construire quelques figures pour le PDF (selon ce qu’on a)
    figs = []
    # 1) Hist prime_pure
    figs.append(px.histogram(dfv, x=TARGET, nbins=50, title="Distribution — prime_pure"))
    # 2) Cat distribution si dispo
    for c in [x for x in CATEGORICAL_CANDIDATES if x in dfv.columns][:1]:
        figs.append(px.histogram(dfv, x=c, title=f"Répartition — {c}"))

    # 3) Si on a entraîné, ajouter les figures de calibration / réel vs prédit
    if "fig_cal" in st.session_state:
        figs.append(st.session_state["fig_cal"])
    if "fig_scatter" in st.session_state:
        figs.append(st.session_state["fig_scatter"])

    # Enregistrer les PNG si possible
    png_paths = []
    if _KALEIDO:
        os.makedirs("exports", exist_ok=True)
        for i, f in enumerate(figs, start=1):
            path = os.path.join("exports", f"figure_{i}.png")
            if _save_plotly_image(f, path):
                png_paths.append(path)

    # Bouton PDF
    if st.button("🖨️ Générer le PDF"):
        meta = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": hash_dataframe(dfv),
            "filters": {k: v for k, v in cat_selections.items() if v},
            "model": None
        }
        if "last_scores" in st.session_state and "best_power" in st.session_state:
            sc = st.session_state["last_scores"]
            meta["model"] = {
                "power": st.session_state.get("best_power"),
                "alpha": getattr(st.session_state.get("best_model"), "alpha", None),
                "deviance": sc["deviance"],
                "rmse": sc["rmse"],
                "mape": sc["mape"],
                "r2": sc["r2"],
            }

        pdf_path = os.path.join("exports", "rapport_glm_tweedie.pdf")
        ok = export_pdf_report(pdf_path, meta, png_paths)
        if ok:
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Télécharger le PDF", f, file_name="rapport_glm_tweedie.pdf", mime="application/pdf")

    st.markdown("<div class='footer'>© 2025 AfriAI • Module Actuariel GLM Tweedie — Designed by Idriss</div>",
                unsafe_allow_html=True)

# -------------------------------
# Router de pages
# -------------------------------
if selected == "Accueil & Synthèse":
    page_accueil(df_f)
elif selected == "Analyse du Portefeuille":
    page_portefeuille(df_f)
elif selected == "Modélisation GLM Tweedie":
    page_modelisation(df_f)
elif selected == "Simulateur Tarifaire":
    page_simulateur(df_f)
elif selected == "Performance & Sinistralité":
    page_performance(df_f)
elif selected == "Qualité & Gouvernance":
    page_qualite(df_f)
elif selected == "Rapport & Export":
    page_rapport(df_f)
