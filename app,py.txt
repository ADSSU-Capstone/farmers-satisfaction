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
    page_title="Farmer Satisfaction Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem; font-weight: bold; color: #1f4e79;
        text-align: center; padding: 1rem 0;
        border-bottom: 3px solid #1f4e79;
    }
    .sub-header {
        font-size: 1.05rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #16a085 0%, #27ae60 100%);
        padding: 1.3rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.9rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AUTO-DETECT COLUMN TYPES
# ============================================================
def find_likert_columns(df):
    """Auto-detect Likert 1-5 columns by pattern AND by value range."""
    candidates = []
    patterns = ['ACC_Q', 'REL_Q', 'QUAL_Q', 'SD_Q', 'RES_Q', 'COMM_Q', 'TIME_Q',
                'Accessibility', 'Reliability', 'Quality', 'Service Delivery',
                'Resources', 'Communication', 'Timeliness']
    for col in df.columns:
        col_str = str(col)
        if any(p.lower() in col_str.lower() for p in patterns):
            candidates.append(col)

    if not candidates:
        for col in df.columns:
            try:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 5 and vals.between(1, 5).all():
                    candidates.append(col)
            except Exception:
                continue
    return candidates


def find_binary_columns(df):
    """Auto-detect Yes/No or 0/1 columns."""
    candidates = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.lower().str.strip().unique()
        vals_set = set(vals)
        if vals_set.issubset({'yes', 'no', '0', '1', 'y', 'n', 'true', 'false'}):
            if len(vals) > 0:
                candidates.append(col)
    return candidates


def find_profile_columns(df):
    """Auto-detect profile columns."""
    keywords = ['age', 'sex', 'gender', 'civil', 'barangay', 'area',
                'education', 'years', 'farm_size', 'crop', 'income',
                'support', 'respondent']
    matches = []
    for col in df.columns:
        cl = str(col).lower()
        if any(k in cl for k in keywords):
            matches.append(col)
    return matches


# ============================================================
# SATISFACTION LABEL
# ============================================================
def classify_score(score):
    if pd.isna(score): return 'Unknown'
    if score >= 4.21:  return 'Highly Satisfied'
    elif score >= 3.41: return 'Satisfied'
    elif score >= 2.61: return 'Moderately Satisfied'
    elif score >= 1.81: return 'Dissatisfied'
    else: return 'Very Dissatisfied'


def compute_satisfaction_label(df, likert_cols):
    """Derive satisfaction from any available Likert columns."""
    sat_cols = [c for c in likert_cols
                if any(p.lower() in str(c).lower()
                       for p in ['ACC_', 'REL_', 'QUAL_', 'Accessibility',
                                 'Reliability', 'Quality'])]

    if not sat_cols:
        sat_cols = likert_cols

    if not sat_cols:
        return None, None

    df_likert = df[sat_cols].apply(pd.to_numeric, errors='coerce')
    overall_score = df_likert.mean(axis=1)
    labels = overall_score.apply(classify_score)
    return labels, overall_score


# ============================================================
# DATA LOADING (works on Streamlit Cloud AND Colab)
# ============================================================
@st.cache_data
def load_data():
    search_paths = ['.', '/content/data']
    files = []
    for path in search_paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, '*.xlsx'))
            files += glob.glob(os.path.join(path, '*.xls'))
            files += glob.glob(os.path.join(path, '*.csv'))

    if not files:
        return None, None

    responses = [f for f in files if 'template' not in os.path.basename(f).lower()]
    preferred = [f for f in responses if any(k in os.path.basename(f).lower()
                 for k in ['response', 'farmer', 'survey'])]
    if preferred:
        target_file = preferred[0]
    elif responses:
        target_file = responses[0]
    else:
        target_file = files[0]

    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file, encoding='utf-8-sig')
        else:
            df = pd.read_excel(target_file, header=0)
    except Exception as e:
        st.error(f"Error reading {target_file}: {e}")
        return None, None

    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    df = df.dropna(how='all')
    return df, os.path.basename(target_file)


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_data(df):
    df = df.copy()

    likert_cols = find_likert_columns(df)
    binary_cols = find_binary_columns(df)
    profile_cols = find_profile_columns(df)

    st.session_state['likert_cols'] = likert_cols
    st.session_state['binary_cols'] = binary_cols
    st.session_state['profile_cols'] = profile_cols

    if not likert_cols:
        return None, None, None, None, None, 'No Likert 1-5 columns detected'

    y_labels, overall_score = compute_satisfaction_label(df, likert_cols)

    if y_labels is None:
        return None, None, None, None, None, 'Could not compute satisfaction score'

    df['_satisfaction'] = y_labels.values
    df['_overall_score'] = overall_score.values

    valid_mask = df['_satisfaction'] != 'Unknown'
    df = df[valid_mask].copy()

    if len(df) < 5:
        return None, None, None, None, None, f'Only {len(df)} valid rows after filtering'

    feature_cols = []
    for c in df.columns:
        if c.startswith('_'):
            continue
        if c in likert_cols and any(p.lower() in str(c).lower()
                                     for p in ['ACC_', 'REL_', 'QUAL_',
                                               'Accessibility', 'Reliability', 'Quality']):
            continue
        if c in profile_cols or c in binary_cols or c in likert_cols:
            feature_cols.append(c)

    if not feature_cols:
        return None, None, None, None, None, 'No usable feature columns found'

    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == 'object':
            numeric = pd.to_numeric(X[col], errors='coerce')
            if numeric.notna().sum() >= 0.6 * len(numeric):
                X[col] = numeric.fillna(numeric.median())
            else:
                X[col] = X[col].astype(str).fillna('Unknown')
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    y = df['_satisfaction'].copy()

    return X, y, df, feature_cols, overall_score.values, None


# ============================================================
# CLASSIFICATION
# ============================================================
def train_classifiers(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=6),
        'Naïve Bayes': GaussianNB()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model, 'y_pred': y_pred, 'y_test': y_test,
            'X_test': X_test_scaled
        }
    return results, scaler, X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


# ============================================================
# CLUSTERING
# ============================================================
def run_kmeans(X_scaled, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if k > 1 and len(set(labels)) > 1 else 0
    return km, labels, sil


def elbow_curve(X_scaled, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return list(range(1, max_k + 1)), wcss


# ============================================================
# LOAD DATA
# ============================================================
df_raw, filename = load_data()

st.markdown('<div class="main-header">🌾 Farmer Satisfaction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Mining Analysis of Farmers\' Satisfaction with Agricultural Support Services<br>'
            '<i>Talacogon, Agusan del Sur • Decision Tree • Naïve Bayes • K-Means</i></div>',
            unsafe_allow_html=True)

# ---- DEBUG PANEL ----
with st.expander("🔍 Debug: Detected Data", expanded=True):
    if df_raw is None:
        st.error("❌ No dataset found. Upload your Excel/CSV file to the repo root.")
    else:
        st.success(f"✅ File loaded: `{filename}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Barangays", df_raw['Barangay'].nunique() if 'Barangay' in df_raw.columns else 'N/A')

        st.write("**Detected columns:**")
        st.code(list(df_raw.columns))

        st.write("**Auto-detection results:**")
        detected_likert = find_likert_columns(df_raw)
        detected_binary = find_binary_columns(df_raw)
        detected_profile = find_profile_columns(df_raw)
        st.write(f"- 🎯 Likert (1–5) columns: **{len(detected_likert)}** → `{detected_likert[:8]}`" if len(detected_likert) > 8 else f"- 🎯 Likert columns: **{len(detected_likert)}** → `{detected_likert}`")
        st.write(f"- 🔘 Binary (Yes/No) columns: **{len(detected_binary)}**")
        st.write(f"- 👤 Profile columns: **{len(detected_profile)}**")

        st.write("**First 3 rows:**")
        st.dataframe(df_raw.head(3), use_container_width=True)

if df_raw is None:
    st.stop()

# Preprocess
X, y, df_clean, feature_cols, overall_scores, error = preprocess_data(df_raw)

if error:
    st.error(f"⚠️ Could not build a valid dataset: **{error}**")
    st.markdown("""
    ### 🔧 How to fix this:

    **Checklist — your file must have:**

    1. **At least one Likert 1–5 column** for satisfaction, ideally named:
       - `ACC_Q1` … `ACC_Q5` (Accessibility)
       - `REL_Q11` … `REL_Q15` (Reliability)
       - `QUAL_Q16` … `QUAL_Q20` (Overall Quality)

    2. **Profile columns** (any of these):
       - `Age`, `Sex`, `Civil_Status`, `Barangay`, `Farming_Area_Type`,
         `Educational_Attainment`, `Years_in_Farming`, `Farm_Size`,
         `Main_Crop`, `Avg_Monthly_Income`

    3. **At least 5 complete rows** (one per respondent).
    """)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Configuration")
st.sidebar.info(f"👥 Respondents: **{len(X)}**")
st.sidebar.info(f"🎯 Features: **{len(feature_cols)}**")

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
k_clusters = st.sidebar.slider("K (clusters)", 2, 6, 3)
cv_folds = st.sidebar.slider("CV folds", 3, 10, 10)

# ============================================================
# TRAIN
# ============================================================
results, scaler, X_train, X_test, y_train, y_test = train_classifiers(
    X, y, test_size=test_size, random_state=int(random_state)
)

metrics_table = pd.DataFrame({
    name: compute_metrics(r['y_test'], r['y_pred'])
    for name, r in results.items()
}).T

X_scaled_clust = StandardScaler().fit_transform(X)
km, cluster_labels, sil_score = run_kmeans(X_scaled_clust, k_clusters, int(random_state))
df_clean['_cluster'] = cluster_labels

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 EDA",
    "🌳 Classification", "🔵 Clustering",
    "📋 Cross-Validation", "💡 Recommendations"
])


# ---- TAB 1: OVERVIEW ----
with tab1:
    st.header("📊 Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Respondents</p><h2>{len(X):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{df_clean["Barangay"].nunique() if "Barangay" in df_clean.columns else 0}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Mean Score</p><h2>{np.mean(overall_scores):.2f}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Silhouette</p><h2>{sil_score:.3f}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Satisfaction Level Distribution")
    sat_counts = y.value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_map = {'Highly Satisfied':'#27ae60','Satisfied':'#2ecc71',
                      'Moderately Satisfied':'#f39c12','Dissatisfied':'#e74c3c',
                      'Very Dissatisfied':'#c0392b'}
        colors = [colors_map.get(x, '#95a5a6') for x in sat_counts.index]
        ax.pie(sat_counts.values, labels=sat_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title("Farmer Satisfaction Levels")
        st.pyplot(fig); plt.close()
    with col2:
        st.dataframe(sat_counts.rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
        ), use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Model Performance")
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#c6efce'),
                 use_container_width=True)


# ---- TAB 2: EDA ----
with tab2:
    st.header("📈 Exploratory Data Analysis")
    st.subheader("👥 Respondent Profile")
    profile_cols_show = [c for c in ['Sex','Civil_Status','Barangay',
                                     'Educational_Attainment','Main_Crop',
                                     'Avg_Monthly_Income','Years_in_Farming','Farm_Size']
                         if c in df_clean.columns]
    for i in range(0, len(profile_cols_show), 3):
        cols = st.columns(3)
        for j, col_name in enumerate(profile_cols_show[i:i+3]):
            with cols[j]:
                counts = df_clean[col_name].value_counts().head(6)
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                ax.barh(counts.index.astype(str), counts.values, color='#3498db')
                ax.set_title(col_name, fontsize=10); ax.tick_params(labelsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("⭐ Likert Means")
    likert_cols = st.session_state.get('likert_cols', [])
    if likert_cols:
        means = df_clean[likert_cols].apply(pd.to_numeric, errors='coerce').mean()
        fig, ax = plt.subplots(figsize=(12, 4))
        means.plot(kind='bar', color='#9b59b6', ax=ax)
        ax.axhline(3.41, color='green', linestyle='--', label='Satisfied threshold')
        ax.axhline(2.61, color='orange', linestyle='--', label='Moderately threshold')
        ax.set_ylabel("Mean Score (1-5)")
        ax.legend(); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        st.pyplot(fig); plt.close()


# ---- TAB 3: CLASSIFICATION ----
with tab3:
    st.header("🌳 Classification Analysis")
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#c6efce'),
                 use_container_width=True)

    st.markdown("---")
    selected_alg = st.selectbox("Choose algorithm:", list(results.keys()))
    result = results[selected_alg]
    m = compute_metrics(result['y_test'], result['y_pred'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{m['Accuracy']:.4f}")
    c2.metric("Precision", f"{m['Precision']:.4f}")
    c3.metric("Recall",    f"{m['Recall']:.4f}")
    c4.metric("F1-Score",  f"{m['F1-Score']:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        labels_order = sorted(y.unique())
        cm = confusion_matrix(result['y_test'], result['y_pred'], labels=labels_order)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=labels_order, yticklabels=labels_order, ax=ax)
        plt.xticks(rotation=30, ha='right'); plt.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.subheader("Classification Report")
        rep = classification_report(result['y_test'], result['y_pred'],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    if selected_alg == 'Decision Tree':
        st.markdown("---")
        st.subheader("📜 Decision Rules (top 3 levels)")
        tree_rules = export_text(result['model'], feature_names=feature_cols, max_depth=3)
        st.code(tree_rules, language='text')
        st.subheader("Feature Importances")
        imp = pd.Series(result['model'].feature_importances_,
                        index=feature_cols).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(10, max(4, len(imp)*0.35)))
        imp.plot(kind='barh', color='#27ae60', ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ---- TAB 4: CLUSTERING ----
with tab4:
    st.header("🔵 K-Means Clustering")
    c1, c2, c3 = st.columns(3)
    c1.metric("K", k_clusters)
    c2.metric("Silhouette", f"{sil_score:.4f}")
    c3.metric("Respondents", len(df_clean))

    st.markdown("---")
    st.subheader("Elbow Method")
    ks, wcss = elbow_curve(X_scaled_clust, max_k=min(10, len(df_clean)-1))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks, wcss, marker='o', color='#3498db', linewidth=2)
    ax.axvline(k_clusters, color='red', linestyle='--', label=f'K = {k_clusters}')
    ax.set_xlabel("K"); ax.set_ylabel("WCSS"); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("PCA Visualization")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_clust)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels,
                    cmap='viridis', s=60, alpha=0.7, edgecolors='k')
    plt.colorbar(sc, label='Cluster'); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🏘️ Cluster Profiles")
    for c in sorted(df_clean['_cluster'].unique()):
        sub = df_clean[df_clean['_cluster'] == c]
        st.markdown(f"### 🔹 Cluster {c} — {len(sub)} farmers ({len(sub)/len(df_clean)*100:.1f}%)")
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'Barangay' in sub.columns:
                st.dataframe(sub['Barangay'].value_counts().head(4).rename('Count'),
                             use_container_width=True)
        with col2:
            if 'Main_Crop' in sub.columns:
                st.dataframe(sub['Main_Crop'].value_counts().head(4).rename('Count'),
                             use_container_width=True)
        with col3:
            st.dataframe(sub['_satisfaction'].value_counts().rename('Count'),
                         use_container_width=True)
        sub_score = pd.to_numeric(sub['_overall_score'], errors='coerce').mean()
        st.markdown(f"**Mean Satisfaction Score:** `{sub_score:.2f}`")
        st.markdown("---")


# ---- TAB 5: CV ----
with tab5:
    st.header(f"📋 {cv_folds}-Fold Cross-Validation")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Naïve Bayes': GaussianNB()
    }
    cv_scores = {name: cross_val_score(m, X, y, cv=cv, scoring='accuracy')
                 for name, m in cv_models.items()}
    cv_df = pd.DataFrame(cv_scores)
    cv_df.index = [f"Fold {i+1}" for i in range(cv_folds)]
    cv_df.loc['Mean'] = cv_df.mean(); cv_df.loc['Std'] = cv_df.iloc[:-2].std()
    st.dataframe(cv_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cv_df.iloc[:-2].plot(kind='bar', ax=ax, color=['#27ae60','#3498db'])
    ax.set_ylabel("Accuracy"); ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig); plt.close()


# ---- TAB 6: RECOMMENDATIONS ----
with tab6:
    st.header("💡 Recommendations")
    best_model = metrics_table['F1-Score'].idxmax()
    st.success(f"### 🏆 Best Model: **{best_model}** (F1 = {metrics_table['F1-Score'].max():.4f})")

    sat_counts = y.value_counts(); total = len(y)
    highly = sat_counts.get('Highly Satisfied', 0) + sat_counts.get('Satisfied', 0)
    moderate = sat_counts.get('Moderately Satisfied', 0)
    low = sat_counts.get('Dissatisfied', 0) + sat_counts.get('Very Dissatisfied', 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Satisfied+", f"{highly} ({highly/total*100:.1f}%)")
    c2.metric("⚠️ Moderate",   f"{moderate} ({moderate/total*100:.1f}%)")
    c3.metric("❌ Dissatisfied+", f"{low} ({low/total*100:.1f}%)")

    st.markdown("---")
    st.subheader("🔑 Top Factors (from Decision Tree)")
    dt_model = results['Decision Tree']['model']
    imp = pd.Series(dt_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(imp.index[::-1], imp.values[::-1], color='#e67e22')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.dataframe(imp.rename('Importance').to_frame().round(4), use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Policy Recommendations")
    st.markdown(f"""
    Based on **{total}** respondents:

    1. **For LGU of Talacogon:**
       - Target clusters with the lowest mean satisfaction scores.
       - Prioritize barangays with high dissatisfaction rates.

    2. **For Agricultural Extension Workers:**
       - Improve communication and timeliness (top Decision Tree factors).
       - Ensure fair distribution of inputs.

    3. **For Policy Makers:**
       - Use Decision Tree rules to design targeted interventions.
       - Align with PPAN 2023–2028 and DA programs.

    4. **For Future Researchers:**
       - Extend with Random Forest, XGBoost, SVM.
       - Apply SMOTE for class imbalance.
       - Add SHAP-based interpretability.

    5. **Limitations (Chapter 3):**
       - Findings specific to Talacogon; require re-validation.
       - Only selected service quality indicators analyzed.
       - Data quality depends on respondent honesty.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")
    st.download_button("📊 Metrics CSV",
                       metrics_table.to_csv().encode('utf-8'),
                       "metrics.csv", "text/csv")
    preds = pd.DataFrame({
        'Actual': y_test.reset_index(drop=True),
        'DT_Predicted': pd.Series(results['Decision Tree']['y_pred']),
        'NB_Predicted': pd.Series(results['Naïve Bayes']['y_pred'])
    })
    st.download_button("📄 Predictions CSV",
                       preds.to_csv(index=False).encode('utf-8'),
                       "predictions.csv", "text/csv")


st.markdown("---")
st.caption("🎓 Capstone Dashboard • Data Mining Analysis of Farmers' Satisfaction • "
           "Mercado, M. & Lequin, K.J.S. • ADSSU • 2026")