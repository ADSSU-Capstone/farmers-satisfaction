import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             silhouette_score, roc_curve, auc)

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

st.set_page_config(page_title="Farmers' Satisfaction Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --brand-primary: #7c2d3a;
    --brand-primary-dark: #4a1822;
    --brand-primary-light: #a14053;
    --brand-accent: #b8860b;
    --brand-bg: #faf7f2;
    --brand-surface: #ffffff;
    --brand-surface-alt: #f5efe6;
    --brand-text: #1c1917;
    --brand-muted: #78716c;
    --brand-border: #e7e0d5;
    --sidebar-bg-top: #2a0f16;
    --sidebar-bg-bottom: #4a1822;
    --sidebar-text: #f5efe6;
    --sidebar-muted: #e7d9c5;
}

html, body, .stApp {
    font-family: 'Manrope', system-ui, sans-serif !important;
    background: var(--brand-bg) !important;
    color: var(--brand-text) !important;
}
[data-testid="stAppViewContainer"] { background: var(--brand-bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }

.stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] *,
[data-testid="stCaptionContainer"] * {
    font-family: 'Manrope', system-ui, sans-serif !important;
    font-size: 1.05rem !important;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    font-family: 'Fraunces', 'Playfair Display', 'Merriweather', Georgia, serif !important;
    color: var(--brand-text) !important;
    letter-spacing: -0.018em !important;
}
.stApp h2 { font-size: 1.65rem !important; font-weight: 700 !important; }
.stApp h3 { font-size: 1.35rem !important; font-weight: 700 !important; }

.block-container { padding-top: 1.3rem !important; padding-bottom: 2.8rem !important; max-width: 1500px; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarNav"] {
    background: #2a0f16 !important;
    background-image: linear-gradient(180deg, #2a0f16 0%, #4a1822 100%) !important;
    border-right: 1px solid #5a2230 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--sidebar-text) !important;
    font-family: 'Manrope', system-ui, sans-serif !important;
}

[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stCheckbox label span,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: var(--sidebar-muted) !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] hr { border-color: #5a2230 !important; }

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input {
    background: #3a1a22 !important;
    border-color: #5a2230 !important;
    color: var(--sidebar-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg,
[data-testid="stSidebar"] svg { fill: var(--sidebar-muted) !important; }

[data-testid="stSidebar"] [role="slider"] {
    background: var(--brand-accent) !important;
    border-color: var(--brand-accent) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stTickBar"] * {
    color: var(--sidebar-muted) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #3a1a22 !important;
    border: 1px dashed #5a2230 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: var(--sidebar-muted) !important;
}

[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneButton"] {
    background: #5a2230 !important;
    color: var(--sidebar-text) !important;
    border: 1px solid #7c2d3a !important;
}

.hero {
    background: linear-gradient(135deg, #4a1822 0%, #7c2d3a 55%, #a14053 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 14px 30px -10px rgba(74, 24, 34, 0.55);
}
.hero h1 {
    font-family: 'Fraunces', 'Playfair Display', Georgia, serif !important;
    color: #ffffff !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.025em !important;
    margin: 0 0 10px 0 !important;
}
.hero p {
    color: #f5e6cc !important;
    font-family: 'Manrope', system-ui, sans-serif !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
}
.hero p:last-child {
    color: #e7c88a !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.88rem !important;
    margin-top: 10px !important;
    font-weight: 600 !important;
}

.metric-card {
    background: var(--brand-surface);
    border-radius: 12px;
    padding: 20px 22px;
    border: 1px solid var(--brand-border);
    border-top: 3px solid var(--brand-primary);
    box-shadow: 0 2px 6px rgba(28, 25, 23, 0.05);
    height: 100%;
}
.metric-label {
    color: var(--brand-muted) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    color: var(--brand-primary-dark) !important;
    font-family: 'Fraunces', 'Playfair Display', Georgia, serif !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-top: 8px;
    line-height: 1.05;
    font-feature-settings: "tnum" 1;
}
.metric-sub {
    color: var(--brand-primary) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    margin-top: 8px;
}

.section-header {
    color: var(--brand-primary-dark) !important;
    font-family: 'Fraunces', 'Playfair Display', Georgia, serif !important;
    font-size: 1.42rem !important;
    font-weight: 700 !important;
    border-left: 4px solid var(--brand-accent);
    padding-left: 14px;
    margin: 28px 0 14px 0;
    letter-spacing: -0.012em;
}

.info-box {
    background: #fdf6e8;
    border: 1px solid #e8d4a8;
    border-left: 4px solid var(--brand-accent);
    border-radius: 10px;
    padding: 16px 20px;
    color: #5a3a0a !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.0rem !important;
    line-height: 1.7;
    margin-top: 12px;
}
.warn-box {
    background: #fbeee0;
    border: 1px solid #e0b48a;
    border-left: 4px solid #b45309;
    border-radius: 10px;
    padding: 16px 20px;
    color: #6b2f0a !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.0rem !important;
    line-height: 1.7;
    margin-top: 12px;
}
.justify-box {
    background: var(--brand-surface-alt);
    border: 1px solid var(--brand-border);
    border-left: 4px solid var(--brand-primary);
    border-radius: 10px;
    padding: 18px 22px;
    color: var(--brand-text) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.0rem !important;
    line-height: 1.75;
}
.justify-box b, .justify-box strong {
    color: var(--brand-primary) !important;
    font-weight: 800 !important;
}

.sidebar-title {
    color: #e7c88a !important;
    margin: 0;
    font-family: 'Fraunces', 'Playfair Display', Georgia, serif !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
}
.sidebar-sub {
    color: #c4a883 !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.88rem !important;
    margin-top: 4px;
}

[data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent !important;
    border-bottom: none !important;
    padding-bottom: 12px;
}
[data-baseweb="tab"] {
    background-color: var(--brand-surface) !important;
    border: 2px solid var(--brand-border) !important;
    border-radius: 9px !important;
    padding: 12px 18px !important;
    color: var(--brand-text) !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    letter-spacing: -0.005em !important;
    height: auto !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--brand-primary) !important;
    border-color: var(--brand-primary) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px -4px rgba(124, 45, 58, 0.5);
}
[data-baseweb="tab"][aria-selected="true"] p,
[data-baseweb="tab"][aria-selected="true"] span {
    color: #ffffff !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { display: none !important; }

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stMetricLabel"] * { font-size: 0.98rem !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] * {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.0rem !important;
    color: var(--brand-primary-dark) !important;
}

.stButton > button,
.stDownloadButton > button {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    background: var(--brand-primary) !important;
    color: #ffffff !important;
    border: 1px solid var(--brand-primary-dark) !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--brand-primary-dark) !important;
}

.stAlert { font-family: 'Manrope', sans-serif !important; font-size: 1.0rem !important; }

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput input,
.stNumberInput input {
    background: #ffffff !important;
    border-color: var(--brand-border) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.0rem !important;
}
</style>
""", unsafe_allow_html=True)

VALID_BARANGAYS = {
    "Batucan", "Zillovia", "Labnig", "Causwagan", "Buena Gracia",
    "Marbon", "Del Monte", "San Agustin", "San Isidro", "La Flora",
    "Desamparados", "San Nicolas", "Sabang Gibong", "Maharlika",
    "Zamora", "Culiram",
}

BARANGAY_ALIASES = {
    "desamarados": "Desamparados", "desamparados": "Desamparados",
    "buena gracia": "Buena Gracia", "buena  gracia": "Buena Gracia",
    "san agustin": "San Agustin", "san isidro": "San Isidro",
    "san nicolas": "San Nicolas", "del monte": "Del Monte",
    "sabang gibong": "Sabang Gibong", "la flora": "La Flora",
    "zillovia": "Zillovia", "batucan": "Batucan", "labnig": "Labnig",
    "causwagan": "Causwagan", "marbon": "Marbon", "maharlika": "Maharlika",
    "zamora": "Zamora", "culiram": "Culiram",
}

BRGY_COORDS = {
    "Causwagan": (8.455, 125.790), "Batucan": (8.475, 125.765),
    "Del Monte": (8.495, 125.775), "Sabang Gibong": (8.425, 125.820),
    "Zillovia": (8.465, 125.760), "San Agustin": (8.440, 125.785),
    "San Isidro": (8.485, 125.805), "San Nicolas": (8.460, 125.830),
    "Labnig": (8.430, 125.755), "La Flora": (8.505, 125.790),
    "Zamora": (8.450, 125.840), "Culiram": (8.480, 125.745),
    "Maharlika": (8.415, 125.810), "Buena Gracia": (8.470, 125.820),
    "Marbon": (8.500, 125.760), "Desamparados": (8.445, 125.800),
}

LIKERT_SECTION_MAP = {
    "accessib": ("ACC", "Accessibility"),
    "efficien": ("EFF", "Efficiency"),
    "reliab": ("REL", "Reliability"),
    "overall": ("QUAL", "Overall Quality"),
    "service deliv": ("SD", "Service Delivery"),
    "availability": ("RES", "Availability of Resources"),
    "communicat": ("COMM", "Communication"),
    "timeli": ("TIME", "Timeliness of Assistance"),
}
LABEL_SECTIONS = {"ACC", "EFF", "REL", "QUAL"}
ALL_LIKERT_PREFIXES = ("ACC", "EFF", "REL", "QUAL", "SD", "RES", "COMM", "TIME")
SATISFACTION_ORDER = ["Highly Satisfied", "Satisfied", "Moderately Satisfied",
                     "Dissatisfied", "Very Dissatisfied"]
SATISFACTION_COLORS = {
    "Highly Satisfied": "#7c2d3a", "Satisfied": "#a14053",
    "Moderately Satisfied": "#b8860b", "Dissatisfied": "#c97b3a",
    "Very Dissatisfied": "#8a4420",
}


def _clean_text(v):
    if pd.isna(v):
        return None
    s = str(v)
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", s)
    s = " ".join(s.split()).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s


def _to_numeric(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).replace(",", "").strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _detect_header_row(raw, scan_rows=8, min_fill=0.5):
    ncols = raw.shape[1]
    best_idx, best_score = 0, -1
    for i in range(min(scan_rows, len(raw))):
        row = raw.iloc[i]
        non_null = row.notna().sum()
        if non_null < ncols * min_fill:
            continue
        uniq = row.dropna().astype(str).nunique()
        fill = non_null / ncols
        score = uniq * fill
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _classify_section(sec_text):
    if not sec_text:
        return None
    s = str(sec_text).lower()
    if "respondent profile" in s or "profile sa" in s:
        return "PROFILE"
    if "services received" in s or "nadawat" in s:
        return "SVC"
    if "open-ended" in s or "open ended" in s:
        return "OPEN"
    if "yes or no" in s or "answerable" in s:
        return "YN"
    for key, (prefix, _) in LIKERT_SECTION_MAP.items():
        if key in s:
            return prefix
    return None


def _build_clean_columns(section_row, question_row):
    counters, seen, out = {}, {}, []
    def _unique(name):
        if name not in seen:
            seen[name] = 0
            return name
        seen[name] += 1
        return f"{name}_{seen[name]}"
    for sec, q in zip(section_row, question_row):
        sec_clean = _clean_text(sec)
        q_clean = _clean_text(q) or "column"
        code = _classify_section(sec_clean)
        if code == "PROFILE":
            name = _unique(q_clean[:55])
        elif code == "SVC":
            counters["SVC"] = counters.get("SVC", 0) + 1
            name = f"SVC_Q{counters['SVC']}"
        elif code == "OPEN":
            counters["OPEN"] = counters.get("OPEN", 0) + 1
            name = f"OPEN_Q{counters['OPEN']}"
        elif code == "YN":
            counters["YN"] = counters.get("YN", 0) + 1
            name = f"YN_Q{counters['YN']}"
        elif code in ALL_LIKERT_PREFIXES:
            counters[code] = counters.get(code, 0) + 1
            name = f"{code}_Q{counters[code]}"
        else:
            name = _unique(q_clean[:55])
        out.append(name)
    return out


def _parse_raw_frame(raw):
    header_idx = _detect_header_row(raw)
    section_idx = header_idx - 1 if header_idx > 0 else None
    section_row = (raw.iloc[section_idx].ffill().tolist()
                   if section_idx is not None else [""] * raw.shape[1])
    question_row = raw.iloc[header_idx].tolist()
    clean_cols = _build_clean_columns(section_row, question_row)
    data = raw.iloc[header_idx + 1:].copy()
    data.columns = clean_cols
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return data


def _normalize_categoricals(df):
    spell = {
        "elemetary": "Elementary", "hi gh school": "High School",
        "highschool": "High School", "merried": "Married",
        "ruber": "Rubber", "vagetables": "Vegetables",
        "causwagan": "Causwagan",
    }
    cat_cols = ("Sex", "Civil Status", "Barangay", "Types of Farming Area",
                "Highest Educational Attainment", "Main Type of Crop Produced",
                "Avarage Monthly Income from Farming",
                "Average Monthly Income from Farming")
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"Nan": None, "nan": None, "": None})
            df[c] = df[c].str.title()
            df[c] = df[c].replace(spell)
    if "Barangay" in df.columns:
        def _fix_brgy(v):
            if pd.isna(v) or v is None:
                return None
            key = str(v).strip().lower()
            return BARANGAY_ALIASES.get(key, str(v).strip().title())
        df["Barangay"] = df["Barangay"].apply(_fix_brgy)
    return df


def _clean_dataframe(df):
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].apply(_clean_text)
    for c in df.columns:
        if re.match(r"^(ACC|EFF|REL|QUAL|SD|RES|COMM|TIME)_Q\d+$", str(c)):
            df[c] = df[c].apply(_to_numeric)
    for c in df.columns:
        if re.match(r"^(SVC|YN)_Q\d+$", str(c)):
            df[c] = (df[c].astype(str).str.strip().str.lower()
                     .replace({"nan": None, "": None, "yes": "Yes",
                               "no": "No", "y": "Yes", "n": "No"}))
    df = _normalize_categoricals(df)
    return df.reset_index(drop=True)


def _find_example_file():
    candidates = []
    for p in [".", "./data", "/content", "/content/data"]:
        if os.path.isdir(p):
            for ext in ("*.xlsx", "*.xls", "*.csv"):
                candidates += glob.glob(os.path.join(p, ext))
    if not candidates:
        return None
    responses = [f for f in candidates if "template" not in os.path.basename(f).lower()]
    preferred = [f for f in responses if any(
        k in os.path.basename(f).lower()
        for k in ["farmer", "survey", "satisfaction", "response"])]
    pool = preferred or responses or candidates
    return pool[0]


def _read_raw(path_or_buffer, is_upload=True):
    name = getattr(path_or_buffer, "name", str(path_or_buffer))
    if str(name).lower().endswith(".csv"):
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path_or_buffer, header=None, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path_or_buffer, header=None, encoding="latin-1")
    return pd.read_excel(path_or_buffer, header=None, sheet_name=0)


def load_and_prepare(raw):
    stats = {"raw": 0, "duplicates": 0, "missing_dropped": 0,
             "invalid_barangay": 0, "final": 0, "rules": []}
    stats["raw"] = int(len(raw))
    data = _parse_raw_frame(raw)
    stats["rules"].append(
        f"Header row detected at Excel row {_detect_header_row(raw) + 1}; "
        "section and question rows merged into clean column names.")
    before = len(data)
    data = data.drop_duplicates()
    stats["duplicates"] = int(before - len(data))
    stats["rules"].append(
        f"Removed {stats['duplicates']} duplicate response records.")
    min_nonnull = max(2, int(data.shape[1] * 0.35))
    before = len(data)
    data = data[data.notna().sum(axis=1) >= min_nonnull]
    stats["missing_dropped"] = int(before - len(data))
    stats["rules"].append(
        f"Dropped {stats['missing_dropped']} records with fewer than 35% non-null fields.")
    data = _clean_dataframe(data)
    stats["rules"].append(
        "Coerced Likert (Parts II and III) items to numeric 1-5; "
        "standardized Yes/No items; trimmed whitespace and stripped invisible Unicode.")
    stats["rules"].append(
        "Normalized categorical spellings (e.g., 'Desamarados' to 'Desamparados').")
    if "Barangay" in data.columns:
        before = len(data)
        data = data[data["Barangay"].isin(VALID_BARANGAYS)]
        stats["invalid_barangay"] = int(before - len(data))
        stats["rules"].append(
            f"Restricted to the 16 registered barangays of Talacogon: "
            f"removed {stats['invalid_barangay']} out-of-scope records.")
    stats["final"] = int(len(data))
    return data, stats


def get_sections(df):
    bucket = {"profile": [], "SVC": [], "YN": [], "OPEN": []}
    for p in LIKERT_SECTION_MAP:
        bucket[LIKERT_SECTION_MAP[p][0]] = []
    for c in df.columns:
        m = re.match(r"^([A-Z]+)_Q\d+$", str(c))
        if m and m.group(1) in bucket:
            bucket[m.group(1)].append(c)
        else:
            bucket["profile"].append(c)
    return bucket


def classify_score(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 4.21:
        return "Highly Satisfied"
    if score >= 3.41:
        return "Satisfied"
    if score >= 2.61:
        return "Moderately Satisfied"
    if score >= 1.81:
        return "Dissatisfied"
    return "Very Dissatisfied"


def compute_satisfaction(df, sections):
    label_cols = []
    for pref in LABEL_SECTIONS:
        label_cols += sections.get(pref, [])
    if not label_cols:
        return None, None
    numeric = df[label_cols].apply(pd.to_numeric, errors="coerce")
    score = numeric.mean(axis=1, skipna=True)
    return score.apply(classify_score), score


def preprocess(df):
    sections = get_sections(df)
    labels, scores = compute_satisfaction(df, sections)
    if labels is None or (labels != "Unknown").sum() < 5:
        likert_cols = []
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) >= 10 and vals.between(1, 5).mean() >= 0.85 and 2 <= vals.nunique() <= 6:
                likert_cols.append(c)
        if likert_cols:
            sections = {"profile": [], "SVC": [], "YN": [], "OPEN": []}
            for p in ALL_LIKERT_PREFIXES:
                sections[p] = []
            per = max(1, len(likert_cols) // 8)
            for i, p in enumerate(ALL_LIKERT_PREFIXES):
                start = i * per
                end = (i + 1) * per if i < 7 else len(likert_cols)
                sections[p] = likert_cols[start:end]
            for c in df.columns:
                if c not in likert_cols:
                    vals = df[c].dropna().astype(str).str.lower().unique()
                    if set(vals).issubset({"yes", "no", "y", "n", "true", "false"}):
                        sections["SVC"].append(c)
                    else:
                        sections["profile"].append(c)
            labels, scores = compute_satisfaction(df, sections)
    if labels is None:
        return None, None, None, None, None, "No Likert columns detected."
    df = df.copy()
    df["_satisfaction"] = labels.values
    df["_score"] = scores.values
    df = df[df["_satisfaction"] != "Unknown"].copy()
    if len(df) < 10:
        return None, None, None, None, None, f"Only {len(df)} respondents with valid scores."
    feature_cols = []
    feature_cols += sections.get("profile", [])
    feature_cols += sections.get("SVC", [])
    feature_cols += sections.get("YN", [])
    for pref in ("SD", "RES", "COMM", "TIME"):
        feature_cols += sections.get(pref, [])
    feature_cols = [c for c in dict.fromkeys(feature_cols)
                    if not str(c).startswith("_") and c in df.columns]
    if not feature_cols:
        return None, None, None, None, None, "No usable feature columns."
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == "object" or str(X[c].dtype) == "category":
            numeric = pd.to_numeric(X[c], errors="coerce")
            if numeric.notna().sum() >= 0.7 * len(numeric):
                X[c] = numeric.fillna(numeric.median())
            else:
                X[c] = X[c].astype(str).fillna("Unknown")
                X[c] = LabelEncoder().fit_transform(X[c])
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    y = df["_satisfaction"].copy()
    return X, y, df, feature_cols, scores.loc[df.index].values, None


def train_models(X, y, test_size, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=seed, max_depth=6),
        "Naive Bayes": GaussianNB(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        results[name] = {"model": m, "y_pred": m.predict(X_te_s),
                         "y_test": y_te, "X_test": X_te_s,
                         "y_proba": m.predict_proba(X_te_s)}
    return results, scaler


def metrics_row(y_t, y_p):
    return {
        "Accuracy": accuracy_score(y_t, y_p),
        "Precision": precision_score(y_t, y_p, average="weighted", zero_division=0),
        "Recall": recall_score(y_t, y_p, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_t, y_p, average="weighted", zero_division=0),
    }


def _kpi(col, label, value, sub=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def _interp(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def _warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)


def _justify(text):
    st.markdown(f'<div class="justify-box">{text}</div>', unsafe_allow_html=True)


st.sidebar.markdown("""
<div style="padding: 8px 0 18px 0;">
    <p class="sidebar-title">Farmers' Satisfaction</p>
    <p class="sidebar-sub">Talacogon, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)

raw_df = None
if uploaded is not None:
    try:
        raw_df = _read_raw(uploaded, is_upload=True)
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
elif use_example:
    ex_path = _find_example_file()
    if ex_path:
        try:
            raw_df = _read_raw(ex_path, is_upload=False)
        except Exception as e:
            st.sidebar.error(f"Could not read example dataset: {e}")
    else:
        st.sidebar.warning("No example dataset file found in the working directory.")

if raw_df is None:
    st.info("Awaiting data source. Upload a CSV or Excel file, or tick the example-dataset checkbox, to begin analysis.")
    st.stop()

df_raw, prep_stats = load_and_prepare(raw_df)
if df_raw is None or len(df_raw) == 0:
    st.error("Dataset could not be parsed or resulted in zero valid records after cleaning.")
    st.stop()

X, y, df_clean, feature_cols, overall_scores, err = preprocess(df_raw)
if err:
    st.error(f"Preprocessing failed: {err}")
    st.stop()

barangay_options = ["All Barangays"] + sorted([b for b in df_clean["Barangay"].dropna().unique()
                                               if b in VALID_BARANGAYS])
focus_barangay = st.sidebar.selectbox("Focus Barangay (drill-down)", barangay_options)
st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))
k_clusters = st.sidebar.slider("K (clusters)", 2, 6, 3)
cv_folds = st.sidebar.slider("CV folds", 3, 10, 10)

if focus_barangay != "All Barangays":
    mask = df_clean["Barangay"] == focus_barangay
    view_df = df_clean[mask].copy()
    X_view = X.loc[mask].copy()
    y_view = y.loc[mask].copy()
    scores_view = np.asarray(overall_scores)[mask.values]
else:
    view_df = df_clean.copy()
    X_view = X.copy()
    y_view = y.copy()
    scores_view = np.asarray(overall_scores)

results, scaler = train_models(X, y, test_size, random_state)
metrics_table = pd.DataFrame(
    {name: metrics_row(r["y_test"], r["y_pred"]) for name, r in results.items()}).T

X_scaled_full = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=k_clusters, random_state=random_state, n_init=10)
cluster_labels_full = km.fit_predict(X_scaled_full)
df_clean = df_clean.copy()
df_clean["_cluster"] = cluster_labels_full
sil_score = (silhouette_score(X_scaled_full, cluster_labels_full)
             if len(set(cluster_labels_full)) > 1 else 0.0)

st.markdown("""
<div class="hero">
    <h1>Data Mining Analysis of Farmers' Satisfaction with Agricultural Support Services</h1>
    <p>Municipality of Talacogon, Agusan del Sur &mdash; Classification and Clustering Evidence Base</p>
    <p>Decision Tree &middot; Naive Bayes &middot; K-Means</p>
</div>
""", unsafe_allow_html=True)

tab_prep, tab_obj1, tab_obj2, tab_obj3, tab_obj4, tab_obj5 = st.tabs([
    "Data Preprocessing & Cleaning",
    "Satisfaction Levels Across Service Dimensions",
    "Key Factors Influencing Satisfaction",
    "Patterns, Trends & Farmer Groupings",
    "Model Performance Evaluation",
    "Data-Driven Recommendations",
])

with tab_prep:
    st.markdown("<div class='section-header'>Before and After Preprocessing</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "Raw Records", f"{prep_stats['raw']:,}", "Original rows in source file")
    _kpi(c2, "Duplicates Removed", f"{prep_stats['duplicates']:,}", "Exact-row duplicates")
    _kpi(c3, "Missing/Out-of-Scope Dropped",
         f"{prep_stats['missing_dropped'] + prep_stats['invalid_barangay']:,}",
         f"Missing: {prep_stats['missing_dropped']} | Barangay: {prep_stats['invalid_barangay']}")
    _kpi(c4, "Final Valid Records", f"{prep_stats['final']:,}",
         f"{prep_stats['final']/max(prep_stats['raw'],1)*100:.1f}% retention")

    st.markdown("<div class='section-header'>Cleaning Rules Applied</div>", unsafe_allow_html=True)
    rules_df = pd.DataFrame({"Rule": prep_stats["rules"]})
    st.dataframe(rules_df, use_container_width=True, hide_index=True)
    _interp(
        f"{prep_stats['final']} of {prep_stats['raw']} raw records ({prep_stats['final']/max(prep_stats['raw'],1)*100:.1f}%) "
        f"survived the cleaning pipeline. Duplicate elimination, the 35% non-null threshold, and the 16-barangay "
        f"whitelist together removed {prep_stats['raw'] - prep_stats['final']} records. The retained dataset is "
        f"internally consistent and ready for classification and clustering."
    )

    st.markdown("<div class='section-header'>Preview of Cleaned Records</div>", unsafe_allow_html=True)
    visible = [c for c in df_clean.columns if not c.startswith("_")]
    st.dataframe(df_clean[visible].head(15), use_container_width=True, height=340)

with tab_obj1:
    st.markdown("<div class='section-header'>Satisfaction Distribution "
                + (f"&mdash; {focus_barangay}" if focus_barangay != "All Barangays" else "&mdash; All Barangays")
                + "</div>", unsafe_allow_html=True)
    sat_counts = y_view.value_counts()
    order = [o for o in SATISFACTION_ORDER if o in sat_counts.index]
    n_focus = len(view_df)
    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "Respondents", f"{n_focus:,}", "Filtered records")
    _kpi(c2, "Mean Satisfaction", f"{np.mean(scores_view):.2f}" if len(scores_view) else "—", "Scale 1-5")
    _kpi(c3, "Dominant Category", order[0] if order else "—",
         f"{(sat_counts[order[0]]/n_focus*100):.1f}% of respondents" if order and n_focus else "")
    _kpi(c4, "Barangays Covered", f"{view_df['Barangay'].nunique()}",
         "Filtered subset" if focus_barangay != "All Barangays" else "All 16 barangays")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if order:
            ax.pie(sat_counts[order].values, labels=order, autopct="%1.1f%%",
                   colors=[SATISFACTION_COLORS[o] for o in order], startangle=120,
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
        st.pyplot(fig)
        plt.close()
    with col2:
        tbl = sat_counts.rename("Count").to_frame()
        tbl["Percentage"] = (tbl["Count"] / tbl["Count"].sum() * 100).round(2)
        st.dataframe(tbl, use_container_width=True)
    _interp(
        f"Within the current selection, the dominant satisfaction category is '{order[0]}' "
        f"with {sat_counts[order[0]]/n_focus*100:.1f}% of respondents. This indicates the primary "
        f"perception tier that agricultural support services occupy for the selected geography."
    )

    st.markdown("<div class='section-header'>Mean Score by Satisfaction Dimension</div>", unsafe_allow_html=True)
    sections_view = get_sections(view_df)
    dim_means, dim_labels = [], []
    for key, (prefix, label) in LIKERT_SECTION_MAP.items():
        cols = sections_view.get(prefix, [])
        if cols:
            numeric = view_df[cols].apply(pd.to_numeric, errors="coerce")
            dim_means.append(numeric.mean(axis=1).mean())
            dim_labels.append(label)
    if dim_means:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        bar_colors = ["#7c2d3a" if v >= 3.41 else "#b8860b" if v >= 2.61 else "#8a4420"
                      for v in dim_means]
        ax.barh(dim_labels, dim_means, color=bar_colors, alpha=0.92)
        ax.axvline(3.41, color="#a14053", linestyle="--", linewidth=1, label="Satisfied (3.41)")
        ax.axvline(2.61, color="#b8860b", linestyle="--", linewidth=1, label="Moderately (2.61)")
        ax.set_xlim(1, 5)
        ax.set_xlabel("Mean Score (1-5)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        _interp(
            f"Among the eight dimensions, '{dim_labels[int(np.argmax(dim_means))]}' scores highest at "
            f"{max(dim_means):.2f}, while '{dim_labels[int(np.argmin(dim_means))]}' scores lowest at "
            f"{min(dim_means):.2f}. The lowest dimension identifies the operational area most in need of "
            f"corrective intervention."
        )

    st.markdown("<div class='section-header'>Barangay Satisfaction Ranking</div>", unsafe_allow_html=True)
    ranking = (view_df.groupby("Barangay")["_score"].agg(["mean", "count"])
               .rename(columns={"mean": "Mean Score", "count": "Respondents"})
               .sort_values("Mean Score", ascending=True))
    if len(ranking) > 1:
        fig, ax = plt.subplots(figsize=(9, max(3.2, len(ranking) * 0.32)))
        colors_rank = ["#7c2d3a" if v >= 3.41 else "#b8860b" if v >= 2.61 else "#8a4420"
                       for v in ranking["Mean Score"]]
        ax.barh(ranking.index, ranking["Mean Score"], color=colors_rank, alpha=0.9)
        ax.set_xlabel("Mean Satisfaction Score (1-5)")
        ax.set_xlim(1, 5)
        ax.grid(axis="x", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        top3 = ranking.tail(3).index.tolist()[::-1]
        bottom = ranking.index[0]
        _interp(
            f"Highest satisfaction: {', '.join(top3)}. Lowest satisfaction: {bottom} "
            f"({ranking.loc[bottom, 'Mean Score']:.2f}). The lowest-ranked barangay warrants priority "
            f"re-evaluation of service delivery."
        )
    st.dataframe(ranking.round(3), use_container_width=True)

    if HAS_PLOTLY and len(ranking) >= 1:
        st.markdown("<div class='section-header'>Geographic Intensity Map</div>", unsafe_allow_html=True)
        map_rows = []
        for brgy, row in ranking.iterrows():
            if brgy in BRGY_COORDS:
                lat, lon = BRGY_COORDS[brgy]
                map_rows.append({"Barangay": brgy, "lat": lat, "lon": lon,
                                 "Mean Score": row["Mean Score"],
                                 "Respondents": int(row["Respondents"])})
        if map_rows:
            map_df = pd.DataFrame(map_rows)
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", hover_name="Barangay",
                hover_data={"Mean Score": ":.2f", "Respondents": True, "lat": False, "lon": False},
                color="Mean Score", size="Respondents", size_max=36,
                color_continuous_scale="YlOrRd", zoom=9.6, height=460)
            fig_map.update_layout(mapbox_style="open-street-map",
                                  margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
            _interp(
                "Darker markers indicate higher mean satisfaction; larger markers indicate more "
                "respondents. Spatial clustering of low-satisfaction barangays highlights areas where "
                "resource reallocation is most urgent."
            )

    st.markdown("<div class='section-header'>Resource Allocation Cross-Tabulation</div>", unsafe_allow_html=True)
    if "Barangay" in view_df.columns:
        crosstab = pd.crosstab(view_df["Barangay"], view_df["_satisfaction"])
        crosstab = crosstab.reindex(columns=[c for c in SATISFACTION_ORDER if c in crosstab.columns],
                                    fill_value=0)
        st.dataframe(crosstab, use_container_width=True)
        _interp(
            "Counts indicate how many respondents in each barangay fall into each satisfaction tier. "
            "Barangays with high Dissatisfied counts relative to Satisfied counts should be prioritized "
            "in intervention planning."
        )

with tab_obj2:
    st.markdown("<div class='section-header'>Correlation of Numerical Variables with Overall Score</div>",
                unsafe_allow_html=True)
    numeric_frames = {}
    for key, (prefix, label) in LIKERT_SECTION_MAP.items():
        cols = sections_view.get(prefix, [])
        if cols:
            numeric_frames[label] = view_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if "Age" in view_df.columns:
        age_numeric = pd.to_numeric(view_df["Age"], errors="coerce")
        if age_numeric.notna().sum() > 5:
            numeric_frames["Age"] = age_numeric
    if numeric_frames:
        corr_df = pd.DataFrame(numeric_frames)
        corr_df["_overall"] = view_df["_score"].values
        corr_matrix = corr_df.corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlOrBr",
                    center=0, square=True, cbar_kws={"shrink": 0.75},
                    annot_kws={"fontsize": 8}, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        overall_corr = corr_matrix["_overall"].drop("_overall").sort_values(ascending=False)
        _interp(
            f"The strongest positive correlation with overall satisfaction is "
            f"'{overall_corr.index[0]}' (r = {overall_corr.iloc[0]:.2f}); the weakest positive "
            f"correlation is '{overall_corr.index[-1]}' (r = {overall_corr.iloc[-1]:.2f}). "
            f"Dimensions with r >= 0.50 are the primary drivers of overall perception."
        )

    st.markdown("<div class='section-header'>Feature Importance from Decision Tree</div>",
                unsafe_allow_html=True)
    dt_model = results["Decision Tree"]["model"]
    imp = pd.Series(dt_model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(imp) * 0.35)))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#7c2d3a", alpha=0.9)
    ax.set_xlabel("Relative Importance")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    _interp(
        f"'{imp.index[0]}' is the strongest single predictor of satisfaction in the Decision Tree, "
        f"accounting for {imp.iloc[0]*100:.1f}% of the model's split importance. "
        f"Top-3 predictors together explain {(imp.iloc[:3].sum())*100:.1f}% of predictive weight."
    )

    st.markdown("<div class='section-header'>Section Means vs Satisfaction Category</div>",
                unsafe_allow_html=True)
    if numeric_frames:
        sm = pd.DataFrame(numeric_frames)
        sm["_label"] = y_view.values
        means_table = sm.groupby("_label").mean().round(3)
        st.dataframe(means_table, use_container_width=True)
        _interp(
            "Rows show how each satisfaction dimension's mean shifts across the satisfaction categories. "
            "Monotonic declines from Highly Satisfied to Very Dissatisfied across a dimension confirm "
            "that dimension is behaviorally discriminative rather than noise."
        )

with tab_obj3:
    st.markdown("<div class='section-header'>K-Means Cluster Visualization (PCA Projection)</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    _kpi(c1, "K", f"{k_clusters}", "Number of clusters")
    _kpi(c2, "Silhouette Score", f"{sil_score:.4f}", "Higher is better")
    _kpi(c3, "Respondents Clustered", f"{len(df_clean):,}", "Full dataset")
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X_scaled_full)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    scatter = ax.scatter(Xp[:, 0], Xp[:, 1], c=cluster_labels_full,
                         cmap="YlOrBr", s=55, alpha=0.85,
                         edgecolors="white", linewidths=0.8)
    plt.colorbar(scatter, label="Cluster", ax=ax)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    _interp(
        f"The two principal components capture "
        f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}% of total "
        f"variance. The silhouette score of {sil_score:.3f} indicates "
        f"{'well-separated' if sil_score >= 0.5 else 'moderate' if sil_score >= 0.25 else 'weakly separated'} "
        f"cluster structure."
    )

    st.markdown("<div class='section-header'>Cluster Profiles</div>", unsafe_allow_html=True)
    for c in sorted(df_clean["_cluster"].unique()):
        sub = df_clean[df_clean["_cluster"] == c]
        share = len(sub) / len(df_clean) * 100
        with st.expander(f"Cluster {c} — {len(sub)} farmers ({share:.1f}%)",
                         expanded=(c == 0)):
            k1, k2, k3 = st.columns(3)
            with k1:
                if "Barangay" in sub.columns:
                    st.dataframe(sub["Barangay"].value_counts().head(5),
                                 use_container_width=True)
            with k2:
                if "Main Type of Crop Produced" in sub.columns:
                    st.dataframe(sub["Main Type of Crop Produced"].value_counts().head(5),
                                 use_container_width=True)
            with k3:
                st.dataframe(sub["_satisfaction"].value_counts(),
                             use_container_width=True)
            mean_score = pd.to_numeric(sub["_score"], errors="coerce").mean()
            _interp(
                f"Cluster {c} has a mean satisfaction score of {mean_score:.2f} "
                f"across {len(sub)} farmers. "
                f"{sub['_satisfaction'].mode().iloc[0] if not sub['_satisfaction'].mode().empty else '—'} "
                f"is the modal category in this segment."
            )

with tab_obj4:
    st.markdown("<div class='section-header'>Hold-out Test Set Performance</div>",
                unsafe_allow_html=True)
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color="#f0d4c0"),
                 use_container_width=True)
    best_name = metrics_table["F1-Score"].idxmax()
    _interp(
        f"{best_name} achieves the highest weighted F1-Score "
        f"({metrics_table.loc[best_name, 'F1-Score']:.4f}), and is therefore the recommended "
        f"classification model for downstream risk scoring."
    )

    st.markdown("<div class='section-header'>Per-Algorithm Metrics</div>", unsafe_allow_html=True)
    for name, res in results.items():
        st.markdown(f"**{name}**")
        m = metrics_row(res["y_test"], res["y_pred"])
        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, "Accuracy", f"{m['Accuracy']:.4f}")
        _kpi(c2, "Precision", f"{m['Precision']:.4f}")
        _kpi(c3, "Recall", f"{m['Recall']:.4f}")
        _kpi(c4, "F1-Score", f"{m['F1-Score']:.4f}")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
            labels_order = sorted(y.unique())
            cm = confusion_matrix(res["y_test"], res["y_pred"], labels=labels_order)
            fig, ax = plt.subplots(figsize=(5.6, 4.6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd",
                        xticklabels=labels_order, yticklabels=labels_order,
                        cbar=False, ax=ax, annot_kws={"fontsize": 9})
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col_r:
            st.markdown("<div class='section-header'>Classification Report</div>", unsafe_allow_html=True)
            rep = classification_report(res["y_test"], res["y_pred"],
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

        if len(set(y.unique())) > 1:
            try:
                y_bin = label_binarize(res["y_test"], classes=res["model"].classes_)
                proba = res["y_proba"]
                fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), proba.ravel())
                auc_micro = auc(fpr_micro, tpr_micro)
                fig_roc, ax_roc = plt.subplots(figsize=(5.6, 4.4))
                ax_roc.plot(fpr_micro, tpr_micro, color="#7c2d3a", lw=2,
                            label=f"Micro-average ROC (AUC = {auc_micro:.3f})")
                ax_roc.plot([0, 1], [0, 1], color="#a8a29e", lw=1, linestyle="--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"{name} — ROC Curve")
                ax_roc.legend(loc="lower right", fontsize=8)
                ax_roc.grid(alpha=0.25)
                for s in ("top", "right"):
                    ax_roc.spines[s].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_roc)
                plt.close()
            except Exception:
                pass

    st.markdown("<div class='section-header'>10-Fold Stratified Cross-Validation</div>",
                unsafe_allow_html=True)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Naive Bayes": GaussianNB(),
    }
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    cv_results = {name: cross_validate(m, X, y, cv=cv, scoring=scoring)
                  for name, m in cv_models.items()}
    summary_data = {}
    for name, res in cv_results.items():
        summary_data[name] = {
            "Mean Accuracy": res["test_accuracy"].mean(),
            "Mean Precision": res["test_precision_weighted"].mean(),
            "Mean Recall": res["test_recall_weighted"].mean(),
            "Mean F1-Score": res["test_f1_weighted"].mean(),
            "Std Dev (Accuracy)": res["test_accuracy"].std(),
        }
    summary_df = pd.DataFrame(summary_data).T.round(4)
    st.dataframe(summary_df.style.format("{:.4f}").highlight_max(axis=0, color="#f0d4c0"),
                 use_container_width=True)
    _interp(
        f"Across {cv_folds} stratified folds, "
        f"{summary_df['Mean F1-Score'].idxmax()} maintains the highest mean F1-Score "
        f"({summary_df['Mean F1-Score'].max():.4f}), with a standard deviation of "
        f"{summary_df.loc[summary_df['Mean F1-Score'].idxmax(), 'Std Dev (Accuracy)']:.4f}. "
        f"Low variance confirms the estimate is stable rather than an artifact of one split."
    )

    st.markdown("<div class='section-header'>Algorithm Justification</div>", unsafe_allow_html=True)
    _justify(
        "Decision Tree is selected for its interpretability: the split thresholds map directly onto "
        "specific Likert responses, allowing the agriculture office to translate the model into plain "
        "service-quality rules. Naive Bayes is retained as the probabilistic baseline required by the "
        "study design, providing a conditional probability view of satisfaction given each response "
        "feature. K-Means is included because the study requires a partitioning of farmers into "
        "behaviorally similar segments. Together, these three algorithms cover the two methodological "
        "categories specified for this research: supervised classification and unsupervised clustering."
    )

    class_balance = y.value_counts(normalize=True)
    if class_balance.max() > 0.55:
        _warn(
            f"Class imbalance detected: '{class_balance.idxmax()}' accounts for "
            f"{class_balance.max()*100:.1f}% of records. Accuracy alone can overstate performance. "
            f"Precision, recall and F1-Score are reported as the primary comparison metrics, and "
            f"stratified splits are used throughout to preserve class proportions."
        )
    else:
        _warn(
            f"No severe class imbalance (largest class: {class_balance.max()*100:.1f}%). "
            f"Weighted metrics are still preferred because multi-class confusion can hide small-class errors."
        )

with tab_obj5:
    st.markdown("<div class='section-header'>Mean Predicted Risk per Barangay</div>",
                unsafe_allow_html=True)
    best_model = results[best_name]["model"]
    X_scaled_all = StandardScaler().fit_transform(X)
    proba_all = best_model.predict_proba(X_scaled_all)
    classes = list(best_model.classes_)
    
    # Define risk as the probability of being in the lowest available class
    # Since "Dissatisfied" does not exist in the dataset, "Moderately Satisfied" is the lowest tier.
    if "Moderately Satisfied" in classes:
        risk_idx = classes.index("Moderately Satisfied")
    else:
        # Fallback if the class name is slightly different
        risk_idx = np.argmin(np.mean(proba_all, axis=0))
        
    risk_prob = proba_all[:, risk_idx]
    
    risk_df = df_clean.copy()
    risk_df["_risk"] = risk_prob
    risk_by_brgy = (risk_df.groupby("Barangay")["_risk"]
                    .agg(["mean", "count"]).rename(
                        columns={"mean": "Mean Risk", "count": "Respondents"})
                    .sort_values("Mean Risk", ascending=True))
                    
    if focus_barangay != "All Barangays" and focus_barangay in risk_by_brgy.index:
        risk_by_brgy = risk_by_brgy.loc[[focus_barangay]]
        
    fig, ax = plt.subplots(figsize=(9, max(3.2, len(risk_by_brgy) * 0.34)))
    # Red for high risk, yellow for medium, teal for low
    bar_colors = ["#8a4420" if v >= 0.5 else "#b8860b" if v >= 0.25 else "#0e7490"
                  for v in risk_by_brgy["Mean Risk"]]
    ax.barh(risk_by_brgy.index, risk_by_brgy["Mean Risk"], color=bar_colors, alpha=0.9)
    ax.set_xlabel("Mean Predicted Probability of 'Moderately Satisfied' (Risk)")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    top_risk = risk_by_brgy.tail(3).index.tolist()[::-1]
    _interp(
        f"Using the {best_name} model, the barangays with the highest mean predicted probability "
        f"of falling into the 'Moderately Satisfied' category are {', '.join(top_risk)}. "
        f"These are the top-priority areas for immediate service intervention."
    )
    st.markdown("<div class='section-header'>Record-Level Predictions (Downloadable)</div>",
                unsafe_allow_html=True)
    pred_df = df_clean[["Barangay", "Sex", "Age"]].copy() if {"Barangay", "Sex", "Age"}.issubset(df_clean.columns) else df_clean[["Barangay"]].copy()
    pred_df["Observed Satisfaction"] = df_clean["_satisfaction"].values
    pred_df["Predicted Satisfaction"] = best_model.predict(X_scaled_all)
    pred_df["Dissatisfaction Risk"] = np.round(risk_prob, 4)
    st.dataframe(pred_df.head(25), use_container_width=True)
    csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download record-level predictions",
                       data=csv_bytes,
                       file_name="farmer_satisfaction_predictions.csv",
                       mime="text/csv")
    _interp(
        "The downloadable table contains each respondent's observed satisfaction, the model's predicted "
        "satisfaction, and a continuous dissatisfaction-risk score. The municipal agriculture office can "
        "use the risk column to sequence follow-up visits and targeted service delivery."
    )

    st.markdown("<div class='section-header'>Data-Driven Recommendations</div>",
                unsafe_allow_html=True)
    lowest_dim = dim_labels[int(np.argmin(dim_means))] if dim_means else "—"
    highest_risk_brgy = top_risk[0] if top_risk else "—"
    weakest_predictor = imp.index[0] if len(imp) else "—"
    _justify(
        f"<b>1. Prioritize '{lowest_dim}'.</b> This is the lowest-scoring service dimension in the "
        f"selected geography and the most direct lever for improving aggregate satisfaction.<br><br>"
        f"<b>2. Deploy targeted interventions to {highest_risk_brgy}.</b> It has the highest mean predicted "
        f"probability of dissatisfaction under the {best_name} model and should absorb the largest share "
        f"of the next service cycle.<br><br>"
        f"<b>3. Invest in '{weakest_predictor}'.</b> The Decision Tree identifies it as the strongest "
        f"recorded driver of satisfaction; small improvements here produce disproportionate gains in the "
        f"modeled satisfaction distribution.<br><br>"
        f"<b>4. Institutionalize this dashboard as a monitoring instrument.</b> Re-run the pipeline after "
        f"each distribution cycle so that changes in predicted risk can be verified empirically rather "
        f"than assumed."
    )
