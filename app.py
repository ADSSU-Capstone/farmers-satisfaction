import streamlit as st
import pandas as pd
import numpy as np
import os, glob, re, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             silhouette_score)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Farmer Satisfaction Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# UI THEME  (terracotta / amber / espresso - warm palette)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #faf6f0; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1917 0%, #292524 100%);
    border-right: 1px solid #44403c;
}
[data-testid="stSidebar"] * { color: #fef3c7 !important; }
[data-testid="stSidebar"] .stRadio > label {
    color: #fcd34d !important; font-size: 0.72rem;
    letter-spacing: 0.08em; font-weight: 600;
}
[data-testid="stSidebar"] hr { border-color: #44403c; }

h1, h2, h3, h4 { color: #1c1917; font-weight: 700; }

.hero {
    background: linear-gradient(135deg, #c2410c 0%, #d97706 100%);
    border-radius: 16px;
    padding: 30px 34px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 12px 24px -8px rgba(194, 65, 12, 0.4);
}
.hero h1 { color: #ffffff; margin: 0 0 6px 0; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p  { color: #fde68a; margin: 0; font-size: 0.92rem; }

.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #e7e5e4;
    border-top: 3px solid #c2410c;
    box-shadow: 0 1px 3px rgba(28, 25, 23, 0.04);
    height: 100%;
}
.metric-label {
    color: #78716c; font-size: 0.70rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.09em;
}
.metric-value {
    color: #1c1917; font-size: 1.85rem; font-weight: 800;
    margin-top: 6px; line-height: 1.1;
}
.metric-sub { color: #b45309; font-size: 0.76rem; font-weight: 500; margin-top: 6px; }

.section-header {
    color: #1c1917; font-size: 1.1rem; font-weight: 700;
    border-left: 4px solid #c2410c;
    padding-left: 12px;
    margin: 26px 0 14px 0;
}

.info-box {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-radius: 10px; padding: 15px 18px;
    color: #7c2d12; font-size: 0.88rem; line-height: 1.6;
}
.warn-box {
    background: #fefce8; border: 1px solid #fde68a;
    border-radius: 10px; padding: 15px 18px;
    color: #713f12; font-size: 0.88rem; line-height: 1.6;
}
.success-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 10px; padding: 15px 18px;
    color: #14532d; font-size: 0.88rem; line-height: 1.6;
}
.sidebar-title { color: #fcd34d; margin: 0; font-size: 1.1rem; font-weight: 800; }
.sidebar-sub   { color: #a8a29e; font-size: 0.76rem; margin-top: 4px; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #e7e5e4; }
.stTabs [data-baseweb="tab"] {
    background-color: transparent; border-radius: 8px 8px 0 0;
    padding: 12px 20px; font-weight: 600; color: #78716c;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important; color: #c2410c !important;
    border-bottom: 2px solid #c2410c;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
LIKERT_SECTION_MAP = {
    'accessibility':             ('ACC',  'Accessibility'),
    'efficiency':                ('EFF',  'Efficiency'),
    'reliability':               ('REL',  'Reliability'),
    'overall service quality':   ('QUAL', 'Overall Quality'),
    'service delivery':          ('SD',   'Service Delivery'),
    'availability of resources': ('RES',  'Availability of Resources'),
    'communication':             ('COMM', 'Communication'),
    'timeliness':                ('TIME', 'Timeliness'),
}

LABEL_SECTIONS      = {'ACC', 'EFF', 'REL', 'QUAL'}
ALL_LIKERT_PREFIXES = ('ACC', 'EFF', 'REL', 'QUAL', 'SD', 'RES', 'COMM', 'TIME')

# ============================================================
# HELPERS - CLEANING
# ============================================================
def _clean_text(v):
    """Strip, collapse internal whitespace, normalize junk to None."""
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = " ".join(s.split())
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s


def _to_numeric(v):
    """Best-effort numeric conversion; returns NaN for non-numeric."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).replace(',', '').strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _detect_header_row(raw, scan_rows=8, min_fill=0.5):
    """
    Pick the row that looks like the real column-name row.

    The score favours rows that are:
      - well-populated (>= min_fill of cells are non-null)
      - highly unique (each cell contains a distinct value -> a real
        column name, not a merged section banner repeated across columns)
    """
    ncols = raw.shape[1]
    best_idx   = 0
    best_score = -1

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
            best_idx   = i

    return best_idx


def _build_clean_columns(section_row, question_row):
    """Combine a section header row + a question-text row into unique names."""
    counters, seen, out = {}, {}, []

    def _unique(name):
        if name not in seen:
            seen[name] = 0
            return name
        seen[name] += 1
        return f"{name}_{seen[name]}"

    for sec, q in zip(section_row, question_row):
        sec_s = _clean_text(sec) or ''
        q_s   = _clean_text(q)   or 'column'
        sec_l = sec_s.lower()

        if 'respondent profile' in sec_l:
            name = _unique(q_s[:55])
        elif 'services received' in sec_l:
            counters['SVC'] = counters.get('SVC', 0) + 1
            name = f"SVC_Q{counters['SVC']}"
        else:
            name = None
            for key, (prefix, _) in LIKERT_SECTION_MAP.items():
                if key in sec_l:
                    counters[prefix] = counters.get(prefix, 0) + 1
                    name = f"{prefix}_Q{counters[prefix]}"
                    break
            if name is None:
                if 'yes or no' in sec_l or 'answerable by' in sec_l:
                    counters['YN'] = counters.get('YN', 0) + 1
                    name = f"YN_Q{counters['YN']}"
                elif 'open-ended' in sec_l or 'open ended' in sec_l:
                    counters['OPEN'] = counters.get('OPEN', 0) + 1
                    name = f"OPEN_Q{counters['OPEN']}"
                else:
                    name = _unique(q_s[:55])
        out.append(name)
    return out


def _clean_dataframe(df):
    """Apply cleaning rules to the raw sliced table."""
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].apply(_clean_text)

    min_nonnull = max(2, int(df.shape[1] * 0.35))
    df = df[df.notna().sum(axis=1) >= min_nonnull]

    for c in df.columns:
        if re.match(r'^(ACC|EFF|REL|QUAL|SD|RES|COMM|TIME)_Q\d+$', str(c)):
            df[c] = df[c].apply(_to_numeric)

    for c in df.columns:
        if re.match(r'^(SVC|YN)_Q\d+$', str(c)):
            df[c] = (df[c]
                     .astype(str)
                     .str.strip()
                     .str.lower()
                     .replace({'nan': None, '': None,
                               'yes': 'Yes', 'no': 'No',
                               'y': 'Yes', 'n': 'No'}))

    for c in ('Sex', 'Civil Status', 'Barangay',
              'Types of Farming Area', 'Highest Educational Attainment'):
        if c in df.columns:
            df[c] = (df[c].astype(str)
                          .str.strip()
                          .str.title()
                          .replace({'Nan': None, '': None}))

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_and_clean():
    """Find, read, header-repair, and clean the survey file."""
    search_paths = ['.', './data', '/content', '/content/data']
    files = []
    for p in search_paths:
        if os.path.isdir(p):
            for ext in ('*.xlsx', '*.xls', '*.csv'):
                files += glob.glob(os.path.join(p, ext))

    if not files:
        return None, None, None

    responses = [f for f in files if 'template' not in os.path.basename(f).lower()]
    preferred = [f for f in responses
                 if any(k in os.path.basename(f).lower()
                        for k in ['farmer', 'survey', 'satisfaction', 'response'])]
    target = (preferred or responses or files)[0]
    base_name = os.path.basename(target)

    try:
        if target.lower().endswith('.csv'):
            raw = None
            for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
                try:
                    raw = pd.read_csv(target, header=None, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            raw = pd.read_excel(target, header=None, sheet_name=0)
    except Exception as e:
        st.error(f"Could not read {target}: {e}")
        return None, None, None

    if raw is None or raw.empty or raw.shape[1] < 2:
        return None, None, None

    header_idx  = _detect_header_row(raw)
    section_idx = header_idx - 1 if header_idx > 0 else None

    section_row  = raw.iloc[section_idx].ffill().tolist() if section_idx is not None \
                   else [''] * raw.shape[1]
    question_row = raw.iloc[header_idx].tolist()

    clean_cols = _build_clean_columns(section_row, question_row)

    data = raw.iloc[header_idx + 1:].copy()
    data.columns = clean_cols
    data = data.dropna(axis=0, how='all').dropna(axis=1, how='all')
    data = _clean_dataframe(data)

    meta = {
        'header_row_excel': header_idx + 1,
        'section_row_excel': (section_idx + 1) if section_idx is not None else None,
    }
    return data, base_name, meta


# ============================================================
# SECTION UTILITIES
# ============================================================
def get_sections(df):
    """Return dict of section prefix -> column list."""
    bucket = {'profile': [], 'SVC': [], 'YN': [], 'OPEN': []}
    for p in LIKERT_SECTION_MAP:
        bucket[LIKERT_SECTION_MAP[p][0]] = []

    for c in df.columns:
        matched = False
        m = re.match(r'^([A-Z]+)_Q\d+$', str(c))
        if m and m.group(1) in bucket:
            bucket[m.group(1)].append(c)
            matched = True
        if not matched:
            bucket['profile'].append(c)
    return bucket


def all_likert_cols(sections):
    out = []
    for p in ALL_LIKERT_PREFIXES:
        out += sections.get(p, [])
    return out


# ============================================================
# SATISFACTION LABEL
# ============================================================
def classify_score(score):
    if pd.isna(score):      return 'Unknown'
    if score >= 4.21:       return 'Highly Satisfied'
    if score >= 3.41:       return 'Satisfied'
    if score >= 2.61:       return 'Moderately Satisfied'
    if score >= 1.81:       return 'Dissatisfied'
    return 'Very Dissatisfied'


def compute_satisfaction(df, sections):
    label_cols = []
    for pref in LABEL_SECTIONS:
        label_cols += sections.get(pref, [])
    if not label_cols:
        return None, None
    numeric = df[label_cols].apply(pd.to_numeric, errors='coerce')
    score = numeric.mean(axis=1, skipna=True)
    return score.apply(classify_score), score


# ============================================================
# PREPROCESS
# ============================================================
def preprocess(df):
    sections = get_sections(df)

    labels, scores = compute_satisfaction(df, sections)
    if labels is None:
        return None, None, None, None, None, 'No Likert columns detected.'

    df = df.copy()
    df['_satisfaction'] = labels.values
    df['_score']        = scores.values
    df = df[df['_satisfaction'] != 'Unknown'].copy()

    if len(df) < 10:
        return None, None, None, None, None, \
               f'Only {len(df)} respondents with valid Likert scores.'

    feature_cols = []
    feature_cols += sections.get('profile', [])
    feature_cols += sections.get('SVC', [])
    feature_cols += sections.get('YN', [])
    for pref in ('SD', 'RES', 'COMM', 'TIME'):
        feature_cols += sections.get(pref, [])

    feature_cols = [c for c in dict.fromkeys(feature_cols)
                    if not str(c).startswith('_') and c in df.columns]

    if not feature_cols:
        return None, None, None, None, None, 'No usable feature columns.'

    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == 'object' or str(X[c].dtype) == 'category':
            numeric = pd.to_numeric(X[c], errors='coerce')
            if numeric.notna().sum() >= 0.7 * len(numeric):
                X[c] = numeric.fillna(numeric.median())
            else:
                X[c] = X[c].astype(str).fillna('Unknown')
                X[c] = LabelEncoder().fit_transform(X[c])
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

    y = df['_satisfaction'].copy()
    return X, y, df, feature_cols, scores.loc[df.index].values, None


# ============================================================
# MODELLING
# ============================================================
def train_models(X, y, test_size, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=seed, max_depth=6),
        'Naive Bayes':   GaussianNB(),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        results[name] = {
            'model': m, 'y_pred': m.predict(X_te_s),
            'y_test': y_te, 'X_test': X_te_s,
        }
    return results, scaler, X_tr, X_te, y_tr, y_te


def metrics_row(y_t, y_p):
    return {
        'Accuracy':  accuracy_score(y_t, y_p),
        'Precision': precision_score(y_t, y_p, average='weighted', zero_division=0),
        'Recall':    recall_score(y_t, y_p, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_t, y_p, average='weighted', zero_division=0),
    }


def run_kmeans(X_scaled, k, seed):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    lbl = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lbl) if len(set(lbl)) > 1 else 0.0
    return km, lbl, sil


def elbow(X_scaled, max_k):
    ks, wcss = list(range(1, max_k + 1)), []
    for k in ks:
        wcss.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_)
    return ks, wcss


# ============================================================
# LOAD + PREPROCESS
# ============================================================
df_raw, filename, meta = load_and_clean()

st.markdown("""
<div class="hero">
    <h1>Data Mining Analysis of Farmers' Satisfaction</h1>
    <p>Agricultural Support Services &mdash; Talacogon, Agusan del Sur</p>
    <p style="margin-top:6px; color:#fed7aa;">Decision Tree &middot; Naive Bayes &middot; K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Data Loading Report", expanded=False):
    if df_raw is None:
        st.error("No dataset found. Place the .xlsx / .xls / .csv file in the same folder as app.py.")
    else:
        st.success(f"Loaded: {filename}")
        if meta:
            st.caption(
                f"Header detected on Excel row {meta['header_row_excel']}"
                + (f"; section labels from row {meta['section_row_excel']}"
                   if meta.get('section_row_excel') else "")
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Respondents", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        brgy_col = next((c for c in df_raw.columns
                         if 'barangay' in str(c).lower()), None)
        c3.metric("Barangays",
                  df_raw[brgy_col].nunique() if brgy_col else "N/A")
        sections = get_sections(df_raw)
        c4.metric("Likert Items", len(all_likert_cols(sections)))
        st.write("**Detected columns:**")
        st.code(list(df_raw.columns), language=None)
        with st.expander("First 5 rows"):
            st.dataframe(df_raw.head(5), use_container_width=True)

if df_raw is None:
    st.stop()

X, y, df_clean, feature_cols, overall_scores, err = preprocess(df_raw)

if err:
    st.error(f"Preprocessing failed: {err}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="padding: 8px 0 20px 0;">
    <p class="sidebar-title">Farmer Satisfaction</p>
    <p class="sidebar-sub">Talacogon, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview",
     "Data Explorer",
     "Patterns & Trends",
     "Classification",
     "Clustering",
     "Key Factors",
     "Cross-Validation"],
    label_visibility="visible",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.info(f"Respondents: {len(X)}")
st.sidebar.info(f"Features: {len(feature_cols)}")

test_size    = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))
k_clusters   = st.sidebar.slider("K (clusters)", 2, 6, 3)
cv_folds     = st.sidebar.slider("CV folds", 3, 10, 10)

# ============================================================
# TRAIN MODELS
# ============================================================
results, scaler, X_tr, X_te, y_tr, y_te = train_models(
    X, y, test_size, random_state
)
metrics_table = pd.DataFrame({
    name: metrics_row(r['y_test'], r['y_pred'])
    for name, r in results.items()
}).T

X_scaled_clust = StandardScaler().fit_transform(X)
km, cluster_labels, sil_score = run_kmeans(X_scaled_clust, k_clusters, random_state)
df_clean = df_clean.copy()
df_clean['_cluster'] = cluster_labels


# ============================================================
# PAGE: OVERVIEW  (CF Output 1 - Satisfaction Classification)
# ============================================================
if page == "Overview":
    st.markdown("## Satisfaction Classification")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Overall distribution of respondents across the five satisfaction levels "
        "derived from the ACC, EFF, REL, and QUAL Likert sections.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    brgy_count = df_clean['Barangay'].nunique() if 'Barangay' in df_clean.columns else 0

    for col, lab, val, sub in [
        (c1, "Respondents", f"{len(X):,}",                       "After cleaning"),
        (c2, "Barangays",   f"{brgy_count}",                     "Covered areas"),
        (c3, "Mean Score",  f"{np.mean(overall_scores):.2f}",    "Scale of 1-5"),
        (c4, "Silhouette",  f"{sil_score:.3f}",                  f"K = {k_clusters}"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lab}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Distribution of Satisfaction Levels</div>",
                unsafe_allow_html=True)

    sat_counts = y.value_counts()
    col1, col2 = st.columns([1.2, 1])

    order = ['Highly Satisfied', 'Satisfied', 'Moderately Satisfied',
             'Dissatisfied', 'Very Dissatisfied']
    order = [o for o in order if o in sat_counts.index]
    colors = {'Highly Satisfied': '#15803d', 'Satisfied': '#65a30d',
              'Moderately Satisfied': '#d97706',
              'Dissatisfied': '#c2410c', 'Very Dissatisfied': '#7c2d12'}

    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.pie(sat_counts[order].values,
               labels=order,
               autopct='%1.1f%%',
               colors=[colors[o] for o in order],
               startangle=120,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title("Respondents by Satisfaction Level",
                     fontsize=11, fontweight='bold', color='#1c1917')
        st.pyplot(fig); plt.close()

    with col2:
        tbl = sat_counts.rename('Count').to_frame()
        tbl['Percentage'] = (tbl['Count'] / tbl['Count'].sum() * 100).round(2)
        st.dataframe(tbl, use_container_width=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:12px;">
        The mean Likert score across all
        <b>{len(all_likert_cols(get_sections(df_clean)))}</b>
        satisfaction and factor items is <b>{np.mean(overall_scores):.2f}</b>.
        Scores at or above <b>3.41</b> are classified as Satisfied or above,
        following the standard five-point Likert interpretation scale.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Model Performance (Hold-out Test Set)</div>",
                unsafe_allow_html=True)
    st.dataframe(
        metrics_table.style
            .format("{:.4f}")
            .highlight_max(axis=0, color='#fed7aa'),
        use_container_width=True,
    )


# ============================================================
# PAGE: DATA EXPLORER  (§3.5 Preprocessing evidence)
# ============================================================
elif page == "Data Explorer":
    st.markdown("## Data Explorer")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Cleaned dataset after header repair, whitespace collapsing, blank-row "
        "removal, Likert numeric coercion, and Yes/No normalization.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("Search across all columns", "")
    with c2:
        show_n = st.number_input("Rows to show", 5, 300, 25, 5)

    view = df_clean.copy()
    if search.strip():
        mask = view.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = view[mask]

    visible_cols = [c for c in view.columns if not c.startswith('_')]
    st.dataframe(view[visible_cols].head(int(show_n)),
                 use_container_width=True, height=520)

    st.markdown("<div class='section-header'>Preprocessing Summary</div>",
                unsafe_allow_html=True)
    sections = get_sections(df_clean)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profile Columns",   len(sections['profile']))
    c2.metric("Services Received", len(sections['SVC']))
    c3.metric("Yes / No Items",    len(sections['YN']))
    c4.metric("Likert Items",      len(all_likert_cols(sections)))

    st.download_button(
        "Download cleaned dataset (CSV)",
        data=df_clean[visible_cols].to_csv(index=False).encode('utf-8'),
        file_name="cleaned_farmer_satisfaction.csv",
        mime="text/csv",
    )


# ============================================================
# PAGE: PATTERNS & TRENDS  (CF Output 2 - Part 1)
# ============================================================
elif page == "Patterns & Trends":
    st.markdown("## Patterns and Trends")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Descriptive analysis of respondent profile, service access, and "
        "per-item Likert means to reveal recurring patterns and service-related issues.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Respondent Profile</div>",
                unsafe_allow_html=True)
    profile_candidates = ['Sex', 'Civil Status', 'Barangay',
                          'Types of Farming Area', 'Highest Educational Attainment',
                          'Years in Farming', 'Farm Size',
                          'Main Type of Crop Produced',
                          'Average Monthly Income from Farming']
    profile_shown = [c for c in profile_candidates if c in df_clean.columns]

    for i in range(0, len(profile_shown), 3):
        cols = st.columns(3)
        for j, col_name in enumerate(profile_shown[i:i+3]):
            with cols[j]:
                counts = df_clean[col_name].value_counts().head(6)
                fig, ax = plt.subplots(figsize=(4.6, 3.4))
                ax.barh(counts.index.astype(str), counts.values,
                        color='#c2410c', alpha=0.85)
                ax.set_title(col_name, fontsize=10, fontweight='bold',
                             color='#1c1917')
                ax.tick_params(labelsize=8)
                ax.grid(axis='x', alpha=0.25)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Mean Score per Likert Item</div>",
                unsafe_allow_html=True)
    sections = get_sections(df_clean)
    likert_cols = all_likert_cols(sections)

    if likert_cols:
        means = df_clean[likert_cols].apply(pd.to_numeric, errors='coerce').mean()
        fig, ax = plt.subplots(figsize=(13, 4.5))
        colors_bar = ['#15803d' if v >= 3.41 else '#d97706' if v >= 2.61
                      else '#c2410c' for v in means.values]
        ax.bar(range(len(means)), means.values, color=colors_bar, alpha=0.9)
        ax.axhline(3.41, color='#16a34a', linestyle='--', linewidth=1,
                   label='Satisfied threshold (3.41)')
        ax.axhline(2.61, color='#f59e0b', linestyle='--', linewidth=1,
                   label='Moderately threshold (2.61)')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=60, ha='right', fontsize=7.5)
        ax.set_ylabel("Mean Score (1-5)")
        ax.set_ylim(0, 5)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("<div class='section-header'>Section-Level Mean Scores</div>",
                    unsafe_allow_html=True)
        section_means = {}
        for key, (prefix, label) in LIKERT_SECTION_MAP.items():
            cols = sections.get(prefix, [])
            if cols:
                section_means[label] = (df_clean[cols]
                                        .apply(pd.to_numeric, errors='coerce')
                                        .mean().mean())
        sec_df = pd.DataFrame({
            'Section':    list(section_means.keys()),
            'Mean Score': [round(v, 3) for v in section_means.values()],
        }).sort_values('Mean Score', ascending=False)
        sec_df['Interpretation'] = sec_df['Mean Score'].apply(classify_score)
        st.dataframe(sec_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Service Access Patterns</div>",
                unsafe_allow_html=True)
    svc_cols = sections.get('SVC', [])
    if svc_cols:
        rates = (df_clean[svc_cols]
                 .apply(lambda s: (s == 'Yes').mean() * 100)
                 .sort_values(ascending=True))
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(rates) * 0.35)))
        ax.barh(rates.index, rates.values, color='#d97706', alpha=0.85)
        ax.set_xlabel("Percent of respondents who received service")
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ============================================================
# PAGE: CLASSIFICATION  (§3.7 - DT + NB)
# ============================================================
elif page == "Classification":
    st.markdown("## Classification Analysis")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Decision Tree and Naive Bayes models predicting satisfaction level "
        "from profile, service access, and the factor Likert sections "
        "(SD, RES, COMM, TIME).</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Hold-out Test Set Performance</div>",
                unsafe_allow_html=True)
    st.dataframe(
        metrics_table.style
            .format("{:.4f}")
            .highlight_max(axis=0, color='#fed7aa'),
        use_container_width=True,
    )

    algo = st.selectbox("Select algorithm:", list(results.keys()))
    res  = results[algo]
    m    = metrics_row(res['y_test'], res['y_pred'])

    c1, c2, c3, c4 = st.columns(4)
    for col, k in zip((c1, c2, c3, c4),
                      ('Accuracy', 'Precision', 'Recall', 'F1-Score')):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{k}</div>
            <div class="metric-value">{m[k]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-header'>Confusion Matrix</div>",
                    unsafe_allow_html=True)
        labels_order = sorted(y.unique())
        cm = confusion_matrix(res['y_test'], res['y_pred'], labels=labels_order)
        fig, ax = plt.subplots(figsize=(6.2, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=labels_order, yticklabels=labels_order,
                    cbar=False, ax=ax, annot_kws={'fontsize': 10})
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("<div class='section-header'>Classification Report</div>",
                    unsafe_allow_html=True)
        rep = classification_report(res['y_test'], res['y_pred'],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    if algo == 'Decision Tree':
        st.markdown("<div class='section-header'>Decision Rules (Top 3 Levels)</div>",
                    unsafe_allow_html=True)
        st.code(export_text(res['model'], feature_names=feature_cols, max_depth=3),
                language='text')

        st.markdown("<div class='section-header'>Feature Importance</div>",
                    unsafe_allow_html=True)
        imp = (pd.Series(res['model'].feature_importances_, index=feature_cols)
                 .sort_values().tail(15))
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(imp) * 0.35)))
        ax.barh(imp.index, imp.values, color='#c2410c', alpha=0.85)
        ax.set_xlabel("Importance")
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ============================================================
# PAGE: CLUSTERING  (§3.7 - K-Means)
# ============================================================
elif page == "Clustering":
    st.markdown("## K-Means Clustering")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Unsupervised grouping of respondents based on similarity in profile, "
        "support received, and factor Likert scores.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, lab, val in [(c1, "K", k_clusters),
                          (c2, "Silhouette Score", f"{sil_score:.4f}"),
                          (c3, "Respondents", len(df_clean))]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lab}</div>
            <div class="metric-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Elbow Method</div>",
                unsafe_allow_html=True)
    ks, wcss = elbow(X_scaled_clust, min(10, len(df_clean) - 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, wcss, marker='o', color='#c2410c', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax.axvline(k_clusters, color='#7c2d12', linestyle='--',
               label=f'Selected K = {k_clusters}')
    ax.set_xlabel("K"); ax.set_ylabel("Within-Cluster Sum of Squares")
    ax.legend(); ax.grid(alpha=0.25)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>PCA Projection</div>",
                unsafe_allow_html=True)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X_scaled_clust)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scatter = ax.scatter(Xp[:, 0], Xp[:, 1], c=cluster_labels,
                         cmap='YlOrBr', s=55, alpha=0.8,
                         edgecolors='white', linewidths=0.8)
    plt.colorbar(scatter, label='Cluster', ax=ax)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.grid(alpha=0.2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Cluster Profiles</div>",
                unsafe_allow_html=True)
    for c in sorted(df_clean['_cluster'].unique()):
        sub = df_clean[df_clean['_cluster'] == c]
        share = len(sub) / len(df_clean) * 100
        with st.expander(f"Cluster {c} — {len(sub)} farmers ({share:.1f}%)",
                         expanded=(c == 0)):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Barangays**")
                if 'Barangay' in sub.columns:
                    st.dataframe(sub['Barangay'].value_counts().head(4),
                                 use_container_width=True)
            with col2:
                st.markdown("**Main Crop**")
                if 'Main Type of Crop Produced' in sub.columns:
                    st.dataframe(sub['Main Type of Crop Produced']
                                 .value_counts().head(4),
                                 use_container_width=True)
            with col3:
                st.markdown("**Satisfaction Levels**")
                st.dataframe(sub['_satisfaction'].value_counts(),
                             use_container_width=True)

            sub_mean = pd.to_numeric(sub['_score'], errors='coerce').mean()
            st.markdown(f"**Mean satisfaction score:** {sub_mean:.2f}")


# ============================================================
# PAGE: KEY FACTORS  (CF Output 3)
# ============================================================
elif page == "Key Factors":
    st.markdown("## Key Factors Affecting Satisfaction")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Factors derived from Decision Tree feature importance, section-level "
        "score comparison, and correlation analysis.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Top Decision-Tree Feature Importances</div>",
                unsafe_allow_html=True)
    dt = results['Decision Tree']['model']
    imp = (pd.Series(dt.feature_importances_, index=feature_cols)
             .sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp.index[::-1], imp.values[::-1], color='#c2410c', alpha=0.85)
    ax.set_xlabel("Feature importance")
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.dataframe(imp.rename('Importance').to_frame().round(4),
                 use_container_width=True)

    st.markdown("<div class='section-header'>Section Means vs. Satisfaction</div>",
                unsafe_allow_html=True)
    sections = get_sections(df_clean)
    section_means = {}
    for key, (prefix, label) in LIKERT_SECTION_MAP.items():
        cols = sections.get(prefix, [])
        if not cols:
            continue
        numeric = df_clean[cols].apply(pd.to_numeric, errors='coerce')
        section_means[label] = numeric.mean(axis=1)
    if section_means:
        sm = pd.DataFrame(section_means)
        sm['_label'] = y.values
        group = sm.groupby('_label').mean().round(3)
        st.dataframe(group, use_container_width=True)
        st.markdown("""
        <div class="info-box" style="margin-top:12px;">
        Rows are the satisfaction levels, columns are the section mean scores.
        A well-separated table (each row distinct) suggests that section-level
        scores are strong discriminators of overall satisfaction.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Correlation of Factor Likert Items with Overall Score</div>",
                unsafe_allow_html=True)
    factor_prefixes = ('SD', 'RES', 'COMM', 'TIME')
    factor_cols = []
    for p in factor_prefixes:
        factor_cols += sections.get(p, [])
    if factor_cols:
        corr_df = df_clean[factor_cols].apply(pd.to_numeric, errors='coerce')
        corr_df['_score'] = df_clean['_score']
        corr = (corr_df.corr()['_score']
                       .drop('_score')
                       .sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(corr) * 0.3)))
        colors_corr = ['#15803d' if v > 0 else '#c2410c' for v in corr.values]
        ax.barh(corr.index[::-1], corr.values[::-1],
                color=colors_corr[::-1], alpha=0.85)
        ax.set_xlabel("Pearson correlation with overall satisfaction score")
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ============================================================
# PAGE: CROSS-VALIDATION  (§3.8 - 10-fold CV)
# ============================================================
elif page == "Cross-Validation":
    st.markdown(f"## {cv_folds}-Fold Cross-Validation")
    st.markdown(
        "<p style='color:#78716c; margin-top:-8px;'>"
        "Each algorithm is trained on K-1 folds and validated on the remaining "
        "fold. The procedure is repeated until every fold has served as the "
        "validation set once (Chapter 3.8).</p>",
        unsafe_allow_html=True,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Naive Bayes':   GaussianNB(),
    }
    cv_scores = {name: cross_val_score(m, X, y, cv=cv, scoring='accuracy')
                 for name, m in cv_models.items()}
    cv_df = pd.DataFrame(cv_scores)
    cv_df.index = [f"Fold {i+1}" for i in range(cv_folds)]

    summary = pd.DataFrame({
        'Mean Accuracy': cv_df.mean(),
        'Std Dev':       cv_df.std(),
        'Min':           cv_df.min(),
        'Max':           cv_df.max(),
    }).round(4)
    st.dataframe(summary, use_container_width=True)

    st.markdown("<div class='section-header'>Per-Fold Accuracy</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cv_df.plot(kind='bar', ax=ax, color=['#c2410c', '#d97706'],
               edgecolor='white', width=0.75)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    plt.xticks(rotation=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.dataframe(cv_df.round(4), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Data Mining Analysis of Farmers' Satisfaction with Agricultural Support "
    "Services — Decision Tree, Naive Bayes, and K-Means Clustering — "
    "Talacogon, Agusan del Sur"
)
