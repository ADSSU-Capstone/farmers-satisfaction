import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             silhouette_score)

st.set_page_config(page_title="Farmer Satisfaction", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #faf6f0; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1c1917 0%, #292524 100%); border-right: 1px solid #44403c; }
[data-testid="stSidebar"] * { color: #fef3c7 !important; }
[data-testid="stSidebar"] .stRadio > label { color: #fcd34d !important; font-size: 0.72rem; letter-spacing: 0.08em; font-weight: 600; }
[data-testid="stSidebar"] hr { border-color: #44403c; }
h1, h2, h3, h4 { color: #1c1917; font-weight: 700; }
.hero { background: linear-gradient(135deg, #c2410c 0%, #d97706 100%); border-radius: 16px; padding: 30px 34px; color: #ffffff; margin-bottom: 24px; box-shadow: 0 12px 24px -8px rgba(194, 65, 12, 0.4); }
.hero h1 { color: #ffffff; margin: 0 0 6px 0; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p  { color: #fde68a; margin: 0; font-size: 0.92rem; }
.metric-card { background: #ffffff; border-radius: 12px; padding: 18px 20px; border: 1px solid #e7e5e4; border-top: 3px solid #c2410c; box-shadow: 0 1px 3px rgba(28, 25, 23, 0.04); height: 100%; }
.metric-label { color: #78716c; font-size: 0.70rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; }
.metric-value { color: #1c1917; font-size: 1.85rem; font-weight: 800; margin-top: 6px; line-height: 1.1; }
.metric-sub { color: #b45309; font-size: 0.76rem; font-weight: 500; margin-top: 6px; }
.section-header { color: #1c1917; font-size: 1.1rem; font-weight: 700; border-left: 4px solid #c2410c; padding-left: 12px; margin: 26px 0 14px 0; }
.info-box { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 10px; padding: 15px 18px; color: #7c2d12; font-size: 0.88rem; line-height: 1.6; }
.sidebar-title { color: #fcd34d; margin: 0; font-size: 1.1rem; font-weight: 800; }
.sidebar-sub   { color: #a8a29e; font-size: 0.76rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

LIKERT_SECTION_MAP = {
    'accessib':      ('ACC',  'Accessibility'),
    'efficien':      ('EFF',  'Efficiency'),
    'reliab':        ('REL',  'Reliability'),
    'overall':       ('QUAL', 'Overall Quality'),
    'service deliv': ('SD',   'Service Delivery'),
    'availability':  ('RES',  'Availability of Resources'),
    'communicat':    ('COMM', 'Communication'),
    'timeli':        ('TIME', 'Timeliness'),
}
LABEL_SECTIONS      = {'ACC', 'EFF', 'REL', 'QUAL'}
ALL_LIKERT_PREFIXES = ('ACC', 'EFF', 'REL', 'QUAL', 'SD', 'RES', 'COMM', 'TIME')

def _clean_text(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    s = " ".join(s.split())
    if s == "" or s.lower() in {"nan", "none", "n/a", "/", "-"}:
        return None
    return s

def _to_numeric(v):
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
    if 'respondent profile' in s or 'profile sa' in s:
        return 'PROFILE'
    if 'services received' in s or 'nadawat' in s:
        return 'SVC'
    if 'open-ended' in s or 'open ended' in s:
        return 'OPEN'
    if 'yes or no' in s or 'answerable' in s:
        return 'YN'
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
        q_clean   = _clean_text(q) or 'column'
        code      = _classify_section(sec_clean)
        if code == 'PROFILE':
            name = _unique(q_clean[:55])
        elif code == 'SVC':
            counters['SVC'] = counters.get('SVC', 0) + 1
            name = f"SVC_Q{counters['SVC']}"
        elif code == 'OPEN':
            counters['OPEN'] = counters.get('OPEN', 0) + 1
            name = f"OPEN_Q{counters['OPEN']}"
        elif code == 'YN':
            counters['YN'] = counters.get('YN', 0) + 1
            name = f"YN_Q{counters['YN']}"
        elif code in ALL_LIKERT_PREFIXES:
            counters[code] = counters.get(code, 0) + 1
            name = f"{code}_Q{counters[code]}"
        else:
            name = _unique(q_clean[:55])
        out.append(name)
    return out

def _clean_dataframe(df):
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
            df[c] = (df[c].astype(str).str.strip().str.lower()
                     .replace({'nan': None, '': None, 'yes': 'Yes', 'no': 'No', 'y': 'Yes', 'n': 'No'}))

    spelling_map = {
        'elemetary': 'Elementary', 'highschool': 'High School', 'hi gh school': 'High School',
        'merried': 'Married', 'ruber': 'Rubber', 'vagetables': 'Vegetables', 
        'causwagan': 'Causwagan', 'bun.brook': 'Bunawan Brook'
    }

    for c in ('Sex', 'Civil Status', 'Barangay', 'Types of Farming Area', 
              'Highest Educational Attainment', 'Main Type of Crop Produced'):
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip().str.title()
                     .replace({'Nan': None, '': None}))
            df[c] = df[c].replace(spelling_map)

    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_and_clean():
    files = []
    for p in ['.', './data', '/content', '/content/data']:
        if os.path.isdir(p):
            for ext in ('*.xlsx', '*.xls', '*.csv'):
                files += glob.glob(os.path.join(p, ext))
    if not files:
        return None, None, None, None
    responses = [f for f in files if 'template' not in os.path.basename(f).lower()]
    preferred = [f for f in responses if any(k in os.path.basename(f).lower() for k in ['farmer', 'survey', 'satisfaction', 'response'])]
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
    except Exception:
        return None, None, None, None
    if raw is None or raw.empty or raw.shape[1] < 2:
        return None, None, None, None
    header_idx  = _detect_header_row(raw)
    section_idx = header_idx - 1 if header_idx > 0 else None
    section_row  = (raw.iloc[section_idx].ffill().tolist() if section_idx is not None else [''] * raw.shape[1])
    question_row = raw.iloc[header_idx].tolist()
    clean_cols = _build_clean_columns(section_row, question_row)
    data = raw.iloc[header_idx + 1:].copy()
    data.columns = clean_cols
    data = data.dropna(axis=0, how='all').dropna(axis=1, how='all')
    data = _clean_dataframe(data)
    return data, base_name

def get_sections(df):
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

def preprocess(df):
    sections = get_sections(df)
    labels, scores = compute_satisfaction(df, sections)
    
    if labels is None or (labels != 'Unknown').sum() < 5:
        likert_cols = []
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors='coerce').dropna()
            if len(vals) >= 10 and vals.between(1, 5).mean() >= 0.85 and 2 <= vals.nunique() <= 6:
                likert_cols.append(c)
        if likert_cols:
            sections = {'profile': [], 'SVC': [], 'YN': [], 'OPEN': []}
            for p in ALL_LIKERT_PREFIXES: sections[p] = []
            per = max(1, len(likert_cols) // 8)
            for i, p in enumerate(ALL_LIKERT_PREFIXES):
                start = i * per
                end = (i + 1) * per if i < 7 else len(likert_cols)
                sections[p] = likert_cols[start:end]
            for c in df.columns:
                if c not in likert_cols:
                    vals = df[c].dropna().astype(str).str.lower().unique()
                    if set(vals).issubset({'yes', 'no', 'y', 'n', 'true', 'false'}):
                        sections['SVC'].append(c)
                    else:
                        sections['profile'].append(c)
            labels, scores = compute_satisfaction(df, sections)

    if labels is None:
        return None, None, None, None, None, 'No Likert columns detected.'

    df = df.copy()
    df['_satisfaction'] = labels.values
    df['_score']        = scores.values
    df = df[df['_satisfaction'] != 'Unknown'].copy()

    if len(df) < 10:
        return None, None, None, None, None, f'Only {len(df)} respondents with valid scores.'

    feature_cols = []
    feature_cols += sections.get('profile', [])
    feature_cols += sections.get('SVC', [])
    feature_cols += sections.get('YN', [])
    for pref in ('SD', 'RES', 'COMM', 'TIME'):
        feature_cols += sections.get(pref, [])
    feature_cols = [c for c in dict.fromkeys(feature_cols) if not str(c).startswith('_') and c in df.columns]

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

def train_models(X, y, test_size, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
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
        results[name] = {'model': m, 'y_pred': m.predict(X_te_s), 'y_test': y_te, 'X_test': X_te_s}
    return results, scaler, X_tr, X_te, y_tr, y_te

def metrics_row(y_t, y_p):
    return {
        'Accuracy':  accuracy_score(y_t, y_p),
        'Precision': precision_score(y_t, y_p, average='weighted', zero_division=0),
        'Recall':    recall_score(y_t, y_p, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_t, y_p, average='weighted', zero_division=0),
    }

df_raw, filename = load_and_clean()

st.markdown("""
<div class="hero">
    <h1>Data Mining Analysis of Farmers' Satisfaction</h1>
    <p>Agricultural Support Services &mdash; Talacogon, Agusan del Sur</p>
    <p style="margin-top:6px; color:#fed7aa;">Decision Tree &middot; Naive Bayes &middot; K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.error("No dataset found. Place the dataset file in the same folder as app.py.")
    st.stop()

X, y, df_clean, feature_cols, overall_scores, err = preprocess(df_raw)
if err:
    st.error(f"Preprocessing failed: {err}")
    st.stop()

st.sidebar.markdown("""
<div style="padding: 8px 0 20px 0;">
    <p class="sidebar-title">Farmer Satisfaction</p>
    <p class="sidebar-sub">Talacogon, Agusan del Sur</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Data Explorer", "Patterns & Trends", "Classification", "Clustering", "Key Factors", "Cross-Validation"],
)

st.sidebar.markdown("---")
test_size    = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)
random_state = int(st.sidebar.number_input("Random state", value=42, step=1))
k_clusters   = st.sidebar.slider("K (clusters)", 2, 6, 3)
cv_folds     = st.sidebar.slider("CV folds", 3, 10, 10)

results, scaler, X_tr, X_te, y_tr, y_te = train_models(X, y, test_size, random_state)
metrics_table = pd.DataFrame({name: metrics_row(r['y_test'], r['y_pred']) for name, r in results.items()}).T

X_scaled_clust = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=k_clusters, random_state=random_state, n_init=10)
cluster_labels = km.fit_predict(X_scaled_clust)
sil_score = silhouette_score(X_scaled_clust, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
df_clean = df_clean.copy()
df_clean['_cluster'] = cluster_labels

if page == "Overview":
    st.markdown("## Satisfaction Classification")
    c1, c2, c3, c4 = st.columns(4)
    brgy_count = df_clean['Barangay'].nunique() if 'Barangay' in df_clean.columns else 0

    for col, lab, val, sub in [
        (c1, "Respondents", f"{len(X):,}", "Total valid"),
        (c2, "Barangays", f"{brgy_count}", "Covered"),
        (c3, "Mean Score", f"{np.mean(overall_scores):.2f}", "Scale of 1-5"),
        (c4, "Silhouette", f"{sil_score:.3f}", f"K = {k_clusters}"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lab}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Distribution of Satisfaction Levels</div>", unsafe_allow_html=True)
    sat_counts = y.value_counts()
    order = ['Highly Satisfied', 'Satisfied', 'Moderately Satisfied', 'Dissatisfied', 'Very Dissatisfied']
    order = [o for o in order if o in sat_counts.index]
    colors = {'Highly Satisfied': '#15803d', 'Satisfied': '#65a30d', 'Moderately Satisfied': '#d97706', 'Dissatisfied': '#c2410c', 'Very Dissatisfied': '#7c2d12'}

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.pie(sat_counts[order].values, labels=order, autopct='%1.1f%%', colors=[colors[o] for o in order], startangle=120, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        st.pyplot(fig); plt.close()
    with col2:
        tbl = sat_counts.rename('Count').to_frame()
        tbl['Percentage'] = (tbl['Count'] / tbl['Count'].sum() * 100).round(2)
        st.dataframe(tbl, use_container_width=True)

    st.markdown("<div class='section-header'>Model Performance (Hold-out Test Set)</div>", unsafe_allow_html=True)
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#fed7aa'), use_container_width=True)

elif page == "Data Explorer":
    st.markdown("## Data Explorer")
    c1, c2 = st.columns([2, 1])
    with c1: search = st.text_input("Search across all columns", "")
    with c2: show_n = st.number_input("Rows to show", 5, 300, 25, 5)

    view = df_clean.copy()
    if search.strip():
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]

    visible_cols = [c for c in view.columns if not c.startswith('_')]
    st.dataframe(view[visible_cols].head(int(show_n)), use_container_width=True, height=520)

elif page == "Patterns & Trends":
    st.markdown("## Patterns and Trends")
    st.markdown("<div class='section-header'>Respondent Profile</div>", unsafe_allow_html=True)
    profile_candidates = ['Sex', 'Civil Status', 'Barangay', 'Types of Farming Area', 'Highest Educational Attainment', 'Years in Farming', 'Farm Size', 'Main Type of Crop Produced', 'Average Monthly Income from Farming']
    profile_shown = [c for c in profile_candidates if c in df_clean.columns]

    for i in range(0, len(profile_shown), 3):
        cols = st.columns(3)
        for j, col_name in enumerate(profile_shown[i:i+3]):
            with cols[j]:
                counts = df_clean[col_name].value_counts().head(6)
                fig, ax = plt.subplots(figsize=(4.6, 3.4))
                ax.barh(counts.index.astype(str), counts.values, color='#c2410c', alpha=0.85)
                ax.set_title(col_name, fontsize=10, fontweight='bold', color='#1c1917')
                ax.tick_params(labelsize=8); ax.grid(axis='x', alpha=0.25)
                for s in ('top', 'right'): ax.spines[s].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Mean Score per Likert Item</div>", unsafe_allow_html=True)
    sections = get_sections(df_clean)
    likert_cols = all_likert_cols(sections)
    if likert_cols:
        means = df_clean[likert_cols].apply(pd.to_numeric, errors='coerce').mean()
        fig, ax = plt.subplots(figsize=(13, 4.5))
        colors_bar = ['#15803d' if v >= 3.41 else '#d97706' if v >= 2.61 else '#c2410c' for v in means.values]
        ax.bar(range(len(means)), means.values, color=colors_bar, alpha=0.9)
        ax.axhline(3.41, color='#16a34a', linestyle='--', linewidth=1, label='Satisfied (3.41)')
        ax.axhline(2.61, color='#f59e0b', linestyle='--', linewidth=1, label='Moderately (2.61)')
        ax.set_xticks(range(len(means))); ax.set_xticklabels(means.index, rotation=60, ha='right', fontsize=7.5)
        ax.set_ylabel("Mean Score (1-5)"); ax.set_ylim(0, 5); ax.legend(loc='lower right', fontsize=8); ax.grid(axis='y', alpha=0.25)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Service Access Patterns</div>", unsafe_allow_html=True)
    svc_cols = sections.get('SVC', [])
    if svc_cols:
        rates = (df_clean[svc_cols].apply(lambda s: (s == 'Yes').mean() * 100).sort_values(ascending=True))
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(rates) * 0.35)))
        ax.barh(rates.index, rates.values, color='#d97706', alpha=0.85)
        ax.set_xlabel("Percent of respondents who received service"); ax.set_xlim(0, 100); ax.grid(axis='x', alpha=0.25)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "Classification":
    st.markdown("## Classification Analysis")
    st.markdown("<div class='section-header'>Hold-out Test Set Performance</div>", unsafe_allow_html=True)
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#fed7aa'), use_container_width=True)

    algo = st.selectbox("Select algorithm:", list(results.keys()))
    res  = results[algo]
    m    = metrics_row(res['y_test'], res['y_pred'])

    c1, c2, c3, c4 = st.columns(4)
    for col, k in zip((c1, c2, c3, c4), ('Accuracy', 'Precision', 'Recall', 'F1-Score')):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{k}</div>
            <div class="metric-value">{m[k]:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        labels_order = sorted(y.unique())
        cm = confusion_matrix(res['y_test'], res['y_pred'], labels=labels_order)
        fig, ax = plt.subplots(figsize=(6.2, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_order, yticklabels=labels_order, cbar=False, ax=ax, annot_kws={'fontsize': 10})
        plt.xticks(rotation=30, ha='right', fontsize=8); plt.yticks(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("<div class='section-header'>Classification Report</div>", unsafe_allow_html=True)
        rep = classification_report(res['y_test'], res['y_pred'], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    if algo == 'Decision Tree':
        st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
        imp = (pd.Series(res['model'].feature_importances_, index=feature_cols).sort_values().tail(15))
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(imp) * 0.35)))
        ax.barh(imp.index, imp.values, color='#c2410c', alpha=0.85); ax.set_xlabel("Importance")
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "Clustering":
    st.markdown("## K-Means Clustering")
    c1, c2, c3 = st.columns(3)
    for col, lab, val in [(c1, "K", k_clusters), (c2, "Silhouette Score", f"{sil_score:.4f}"), (c3, "Respondents", len(df_clean))]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lab}</div>
            <div class="metric-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>PCA Projection</div>", unsafe_allow_html=True)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X_scaled_clust)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scatter = ax.scatter(Xp[:, 0], Xp[:, 1], c=cluster_labels, cmap='YlOrBr', s=55, alpha=0.8, edgecolors='white', linewidths=0.8)
    plt.colorbar(scatter, label='Cluster', ax=ax)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.2)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Cluster Profiles</div>", unsafe_allow_html=True)
    for c in sorted(df_clean['_cluster'].unique()):
        sub = df_clean[df_clean['_cluster'] == c]
        share = len(sub) / len(df_clean) * 100
        with st.expander(f"Cluster {c} - {len(sub)} farmers ({share:.1f}%)", expanded=(c == 0)):
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'Barangay' in sub.columns: st.dataframe(sub['Barangay'].value_counts().head(4), use_container_width=True)
            with col2:
                if 'Main Type of Crop Produced' in sub.columns: st.dataframe(sub['Main Type of Crop Produced'].value_counts().head(4), use_container_width=True)
            with col3:
                st.dataframe(sub['_satisfaction'].value_counts(), use_container_width=True)
            sub_mean = pd.to_numeric(sub['_score'], errors='coerce').mean()
            st.markdown(f"**Mean score:** {sub_mean:.2f}")

elif page == "Key Factors":
    st.markdown("## Key Factors Affecting Satisfaction")
    st.markdown("<div class='section-header'>Top Feature Importances</div>", unsafe_allow_html=True)
    dt = results['Decision Tree']['model']
    imp = (pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp.index[::-1], imp.values[::-1], color='#c2410c', alpha=0.85); ax.set_xlabel("Importance")
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.dataframe(imp.rename('Importance').to_frame().round(4), use_container_width=True)

    st.markdown("<div class='section-header'>Section Means vs Satisfaction</div>", unsafe_allow_html=True)
    sections = get_sections(df_clean)
    section_means = {}
    for key, (prefix, label) in LIKERT_SECTION_MAP.items():
        cols = sections.get(prefix, [])
        if cols:
            numeric = df_clean[cols].apply(pd.to_numeric, errors='coerce')
            section_means[label] = numeric.mean(axis=1)
    if section_means:
        sm = pd.DataFrame(section_means); sm['_label'] = y.values
        st.dataframe(sm.groupby('_label').mean().round(3), use_container_width=True)

    st.markdown("<div class='section-header'>Correlation with Overall Score</div>", unsafe_allow_html=True)
    factor_cols = []
    for p in ('SD', 'RES', 'COMM', 'TIME'): factor_cols += sections.get(p, [])
    if factor_cols:
        corr_df = df_clean[factor_cols].apply(pd.to_numeric, errors='coerce')
        corr_df['_score'] = df_clean['_score']
        corr = (corr_df.corr()['_score'].drop('_score').sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(corr) * 0.3)))
        ax.barh(corr.index[::-1], corr.values[::-1], color='#15803d', alpha=0.85)
        ax.set_xlabel("Correlation"); ax.axvline(0, color='black', linewidth=0.8); ax.grid(axis='x', alpha=0.25)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "Cross-Validation":
    st.markdown(f"## {cv_folds}-Fold Cross-Validation")
    st.markdown("<p style='color:#78716c; margin-top:-8px;'>Each algorithm is trained on K-1 folds and validated on the remaining fold. The results are averaged across all folds using Accuracy, Precision, Recall, and F1-Score.</p>", unsafe_allow_html=True)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_models = {'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6), 'Naive Bayes': GaussianNB()}
    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    cv_results = {name: cross_validate(m, X, y, cv=cv, scoring=scoring_metrics) for name, m in cv_models.items()}

    summary_data = {}
    for name, res in cv_results.items():
        summary_data[name] = {
            'Mean Accuracy':  res['test_accuracy'].mean(),
            'Mean Precision': res['test_precision_weighted'].mean(),
            'Mean Recall':    res['test_recall_weighted'].mean(),
            'Mean F1-Score':  res['test_f1_weighted'].mean(),
            'Std Dev (Acc)':  res['test_accuracy'].std(),
        }
    summary_df = pd.DataFrame(summary_data).T.round(4)
    
    st.markdown("<div class='section-header'>Average Metrics Across All Folds</div>", unsafe_allow_html=True)
    st.dataframe(summary_df.style.format("{:.4f}").highlight_max(axis=0, color='#fed7aa'), use_container_width=True)

    st.markdown("<div class='section-header'>Overall Performance Comparison</div>", unsafe_allow_html=True)
    plot_df = summary_df[['Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1-Score']].T
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df.plot(kind='bar', ax=ax, color=['#c2410c', '#d97706'], edgecolor='white', width=0.75)
    ax.set_ylabel("Score (0 to 1)"); ax.set_ylim(0, 1.05); ax.legend(loc='lower right', fontsize=9); ax.grid(axis='y', alpha=0.25)
    plt.xticks(rotation=0, fontweight='bold')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()
