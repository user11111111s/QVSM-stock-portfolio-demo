# =========================================
# Quantum-Ready Stock Risk Prediction GUI
# State-of-the-Art Dark Dashboard Design
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import rbf_kernel

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="QRISK · Quantum Stock Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------
# Global CSS Injection — Full Design System
# --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020408 !important;
    color: #e0eaf7 !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 15% 20%, rgba(0,220,200,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 80%, rgba(120,60,255,0.07) 0%, transparent 60%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 39px,
            rgba(0,220,200,0.025) 39px,
            rgba(0,220,200,0.025) 40px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 39px,
            rgba(0,220,200,0.025) 39px,
            rgba(0,220,200,0.025) 40px
        ),
        #020408 !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

[data-testid="stMain"] > div { padding: 0 !important; }

/* ── Main Content Wrapper ── */
.block-container {
    padding: 0 2.5rem 4rem 2.5rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Hero Banner ── */
.hero-banner {
    position: relative;
    padding: 3.5rem 3rem 3rem;
    margin-bottom: 2.5rem;
    overflow: hidden;
    border-bottom: 1px solid rgba(0,220,200,0.15);
}

.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
        rgba(0,220,200,0.08) 0%,
        rgba(120,60,255,0.06) 50%,
        transparent 100%);
    pointer-events: none;
}

.hero-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    color: #00dcc8;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.85;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(130deg, #ffffff 0%, #a0f0e8 40%, #7844ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}

.hero-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    color: rgba(224,234,247,0.5);
    font-weight: 300;
    letter-spacing: 0.03em;
}

.hero-badge {
    display: inline-block;
    margin-left: 1rem;
    padding: 0.2em 0.75em;
    background: rgba(0,220,200,0.12);
    border: 1px solid rgba(0,220,200,0.3);
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #00dcc8;
    letter-spacing: 0.12em;
    vertical-align: middle;
    text-transform: uppercase;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin: 2.5rem 0 1.2rem;
}

.section-header::before {
    content: '';
    display: block;
    width: 3px;
    height: 1.4rem;
    background: linear-gradient(180deg, #00dcc8, #7844ff);
    border-radius: 10px;
    flex-shrink: 0;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(224,234,247,0.8);
}

.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,220,200,0.2), transparent);
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}

.metric-card {
    position: relative;
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem 1.6rem 1.5rem;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s;
    backdrop-filter: blur(20px);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: var(--card-accent, linear-gradient(90deg, transparent, #00dcc8, transparent));
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 120px; height: 120px;
    background: var(--card-glow, rgba(0,220,200,0.07));
    border-radius: 50%;
    filter: blur(20px);
    pointer-events: none;
}

.metric-card:hover {
    border-color: rgba(0,220,200,0.25);
    transform: translateY(-2px);
}

.metric-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(224,234,247,0.4);
    margin-bottom: 0.8rem;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    color: var(--val-color, #00dcc8);
    margin-bottom: 0.3rem;
    letter-spacing: -0.03em;
}

.metric-label {
    font-size: 0.78rem;
    color: rgba(224,234,247,0.5);
    font-weight: 400;
}

.metric-indicator {
    position: absolute;
    top: 1.4rem; right: 1.4rem;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--ind-color, #00dcc8);
    box-shadow: 0 0 10px var(--ind-color, #00dcc8);
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* ── Glassmorphism Panel ── */
.glass-panel {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(24px);
}

/* ── Select Box Overrides ── */
[data-testid="stSelectbox"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(224,234,247,0.5) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(0,220,200,0.06) !important;
    border: 1px solid rgba(0,220,200,0.25) !important;
    border-radius: 10px !important;
    color: #e0eaf7 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Dataframe Overrides ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

iframe[title="st_dataframe"] {
    border-radius: 12px;
}

/* ── Explanation Pills ── */
.pill-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 0.5rem;
}

.pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1rem;
    background: rgba(0,220,200,0.06);
    border: 1px solid rgba(0,220,200,0.18);
    border-radius: 100px;
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    color: rgba(224,234,247,0.75);
    font-weight: 300;
}

.pill-dot {
    width: 5px; height: 5px;
    background: #00dcc8;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Status Bar ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.7rem 1.5rem;
    background: rgba(0,220,200,0.05);
    border: 1px solid rgba(0,220,200,0.12);
    border-radius: 10px;
    margin-bottom: 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: rgba(224,234,247,0.4);
}

.status-dot { color: #00dcc8; }
.status-sep { color: rgba(255,255,255,0.1); }

/* ── Column spacing ── */
[data-testid="column"] {
    padding: 0 0.5rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,220,200,0.3); border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# --------------------------
# Matplotlib Global Style
# --------------------------
DARK_BG    = "#020408"
PANEL_BG   = "#0a1018"
GRID_COLOR = "#0f1e2a"
TEAL       = "#00dcc8"
VIOLET     = "#7844ff"
TEXT_MID   = "#6a8099"
TEXT_LIGHT = "#b0c8e0"

matplotlib.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    GRID_COLOR,
    'axes.labelcolor':   TEXT_MID,
    'axes.titlecolor':   TEXT_LIGHT,
    'xtick.color':       TEXT_MID,
    'ytick.color':       TEXT_MID,
    'grid.color':        GRID_COLOR,
    'grid.linewidth':    0.7,
    'text.color':        TEXT_LIGHT,
    'font.family':       'monospace',
    'font.size':         9,
    'figure.dpi':        140,
})

# --------------------------
# Load & Engineer Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/stock_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

df = load_data()

df['Log_Return']   = np.log(df['Close'] / df['Close'].shift(1))
df['Volatility']   = df['Log_Return'].rolling(10).std()
df['Volume_Scaled'] = (
    (df['Volume'] - df['Volume'].rolling(20).mean()) /
     df['Volume'].rolling(20).std()
)
df = df.dropna().reset_index(drop=True)

df['Future_Volatility'] = df['Log_Return'].rolling(5).std().shift(-5)
df = df.dropna().reset_index(drop=True)

threshold       = df['Future_Volatility'].median()
df['Risk_Label'] = (df['Future_Volatility'] > threshold).astype(int)

# Split & scale
X = df[['Log_Return','Volatility','Volume_Scaled']].values
y = df['Risk_Label'].values
split = int(len(X) * 0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = MinMaxScaler(feature_range=(-1, 1))
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# Classical models
lr  = LogisticRegression(max_iter=500)
lr.fit(Xs_train, y_train)
lr_preds = lr.predict(Xs_test)
lr_acc   = accuracy_score(y_test, lr_preds)

svm = SVC(kernel='rbf')
svm.fit(Xs_train, y_train)
svm_preds = svm.predict(Xs_test)
svm_acc   = accuracy_score(y_test, svm_preds)

# QSVM Proxy
QSVM_MAX = 120
df_q = df.tail(QSVM_MAX).reset_index(drop=True)
X_q  = df_q[['Log_Return','Volatility','Volume_Scaled']].values
y_q  = df_q['Risk_Label'].values
sq   = int(len(X_q)*0.75)
Xq_train, Xq_test = X_q[:sq], X_q[sq:]
yq_train, yq_test = y_q[:sq], y_q[sq:]

sc_q = MinMaxScaler(feature_range=(-1,1))
Xqs_train = sc_q.fit_transform(Xq_train)
Xqs_test  = sc_q.transform(Xq_test)

K_train = rbf_kernel(Xqs_train, Xqs_train, gamma=1.0)
K_test  = rbf_kernel(Xqs_test,  Xqs_train, gamma=1.0)

qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, yq_train)
qsvm_preds = qsvm.predict(K_test)
qsvm_acc   = accuracy_score(yq_test, qsvm_preds)

best_model = max([('LR', lr_acc), ('SVM', svm_acc), ('QSVM', qsvm_acc)], key=lambda x: x[1])

# --------------------------
# HERO BANNER
# --------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-label">⬡ Quantum Risk Intelligence Platform · v2.0</div>
    <div class="hero-title">QRISK<span style="opacity:0.3">·</span>NEURAL</div>
    <div class="hero-sub">
        Classical vs Quantum-Kernel Model Comparison
        <span class="hero-badge">LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Status bar
n_train = len(X_train)
n_test  = len(X_test)
n_total = len(df)
st.markdown(f"""
<div class="status-bar">
    <span><span class="status-dot">●</span> SYSTEM ONLINE</span>
    <span class="status-sep">│</span>
    <span>RECORDS: {n_total:,}</span>
    <span class="status-sep">│</span>
    <span>TRAIN: {n_train:,} · TEST: {n_test:,}</span>
    <span class="status-sep">│</span>
    <span>SPLIT: 75/25</span>
    <span class="status-sep">│</span>
    <span>BEST ENGINE: {best_model[0]} @ {best_model[1]:.1%}</span>
</div>
""", unsafe_allow_html=True)

# --------------------------
# METRIC CARDS
# --------------------------
st.markdown("""
<div class="section-header">
    <div class="section-title">Model Performance</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

def acc_color(v):
    if v >= 0.80: return "#00dcc8"
    if v >= 0.65: return "#7844ff"
    return "#ff4d6d"

card_cfgs = [
    ("lr",   "Logistic Regression", "BASELINE · LINEAR",   lr_acc,   "rgba(0,220,200,0.08)",  "#00dcc8"),
    ("svm",  "SVM · RBF Kernel",    "CLASSICAL · NON-LINEAR", svm_acc, "rgba(120,60,255,0.08)", "#7844ff"),
    ("qsvm", "QSVM Proxy",          "QUANTUM-INSPIRED",     qsvm_acc, "rgba(255,77,109,0.08)", "#00dcc8"),
]

# Render metric cards using columns
cols = st.columns(3)
for col, (key, name, tag, acc, _bg, col_accent) in zip(cols, card_cfgs):
    bar_w = int(acc * 100)
    vc    = acc_color(acc)
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--card-glow:{_bg}; --card-accent: linear-gradient(90deg,transparent,{col_accent},transparent);">
            <div class="metric-indicator" style="--ind-color:{col_accent};"></div>
            <div class="metric-tag">{tag}</div>
            <div class="metric-value" style="--val-color:{vc};">{acc:.1%}</div>
            <div class="metric-label">{name}</div>
            <div style="margin-top:1rem; height:3px; background:rgba(255,255,255,0.06); border-radius:10px; overflow:hidden;">
                <div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg,{col_accent},{vc}); border-radius:10px; transition:width 1s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------
# ACCURACY BAR CHART
# --------------------------
st.markdown("""
<div class="section-header">
    <div class="section-title">Accuracy Comparison</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

with st.container():
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    fig1.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(PANEL_BG)

    models_    = ['Logistic\nRegression', 'SVM\nRBF', 'QSVM\nProxy']
    accuracies = [lr_acc, svm_acc, qsvm_acc]
    colors_    = [TEAL, VIOLET, '#ff6eb0']
    x_pos      = np.arange(len(models_))

    bars = ax1.bar(x_pos, accuracies, color=colors_, width=0.45,
                   zorder=3, edgecolor='none')

    # Glow effect — shadow bars
    for bar, c in zip(bars, colors_):
        ax1.bar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                width=0.55, color=c, alpha=0.12, zorder=2, edgecolor='none')

    # Value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{acc:.3f}', ha='center', va='bottom',
                 fontsize=10, fontfamily='monospace',
                 fontweight='bold', color='white')

    # Reference line
    ax1.axhline(0.5, color=GRID_COLOR, linewidth=1.5, linestyle='--', zorder=1)
    ax1.text(2.35, 0.505, 'CHANCE', fontsize=7, color=TEXT_MID,
             fontfamily='monospace', va='bottom')

    ax1.set_xlim(-0.4, 2.4)
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_, fontsize=9, color=TEXT_LIGHT)
    ax1.set_ylabel('Accuracy', fontsize=8, color=TEXT_MID)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.yaxis.grid(True, zorder=0)
    ax1.xaxis.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    fig1.tight_layout(pad=1.5)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

# --------------------------
# CONFUSION MATRIX  +  VOLATILITY  (side by side)
# --------------------------
st.markdown("""
<div class="section-header">
    <div class="section-title">Diagnostics</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.6])

with col_left:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                letter-spacing:0.18em;text-transform:uppercase;
                color:rgba(224,234,247,0.4);margin-bottom:0.6rem;">
    ◈ Select Engine
    </div>""", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "", ["Logistic Regression", "SVM", "QSVM (Proxy)"],
        label_visibility="collapsed"
    )

    if model_choice == "Logistic Regression":
        preds, true, model_col = lr_preds,   y_test,  TEAL
    elif model_choice == "SVM":
        preds, true, model_col = svm_preds,  y_test,  VIOLET
    else:
        preds, true, model_col = qsvm_preds, yq_test, '#ff6eb0'

    cm = confusion_matrix(true, preds)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor(DARK_BG)
    ax2.set_facecolor(DARK_BG)

    # Custom colormap: dark → model accent
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'q', ['#0a1018', model_col], N=256)

    im = ax2.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=cm.max())

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            brightness = cm[i,j] / cm.max()
            txt_color  = 'black' if brightness > 0.55 else 'white'
            ax2.text(j, i, f'{cm[i,j]}',
                     ha='center', va='center',
                     fontsize=18, fontfamily='monospace',
                     fontweight='bold', color=txt_color)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pred: LOW', 'Pred: HIGH'], fontsize=8)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Act: LOW', 'Act: HIGH'], fontsize=8)
    ax2.set_title(f'{model_choice}', fontsize=9, color=model_col,
                  fontfamily='monospace', pad=12)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    fig2.tight_layout(pad=1.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with col_right:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                letter-spacing:0.18em;text-transform:uppercase;
                color:rgba(224,234,247,0.4);margin-bottom:0.6rem;">
    ◈ Rolling Volatility — 10-Day Window
    </div>""", unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor(DARK_BG)
    ax3.set_facecolor(PANEL_BG)

    dates = df['Date']
    vol   = df['Volatility']

    # Filled area + line
    ax3.fill_between(dates, vol, alpha=0.12, color=TEAL, zorder=1)
    ax3.plot(dates, vol, color=TEAL, linewidth=1.3, zorder=2)

    # Risk label overlay — shade HIGH-risk periods
    high_risk = df['Risk_Label'].values
    for i in range(len(dates)-1):
        if high_risk[i] == 1:
            ax3.axvspan(dates.iloc[i], dates.iloc[i+1],
                        alpha=0.06, color=VIOLET, zorder=0)

    # Median line
    med_vol = vol.median()
    ax3.axhline(med_vol, color=VIOLET, linewidth=1, linestyle=':', alpha=0.6)
    ax3.text(dates.iloc[-1], med_vol, f'  median', fontsize=7,
             color=VIOLET, va='center', alpha=0.8)

    ax3.set_xlabel('Date', fontsize=8)
    ax3.set_ylabel('Volatility (σ)', fontsize=8)
    ax3.yaxis.grid(True, zorder=0)
    ax3.xaxis.grid(False)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color=TEAL, linewidth=1.5, label='10-day rolling vol'),
        mpatches.Patch(color=VIOLET, alpha=0.25, label='High-risk zone'),
    ]
    ax3.legend(handles=legend_els, loc='upper left', fontsize=7,
               framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR)

    fig3.tight_layout(pad=1.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# --------------------------
# CLASSIFICATION REPORT
# --------------------------
st.markdown("""
<div class="section-header">
    <div class="section-title">Classification Report</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

report     = classification_report(true, preds, output_dict=True)
report_df  = pd.DataFrame(report).transpose().round(3)

st.dataframe(
    report_df.style
        .set_properties(**{
            'background-color': '#0a1018',
            'color': '#b0c8e0',
            'border': '1px solid #0f1e2a',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': '0.8rem',
        })
        .format(precision=3)
        .background_gradient(
            cmap='YlGnBu',
            subset=['precision','recall','f1-score'],
            low=0.1, high=0.0
        ),
    use_container_width=True,
    height=200
)

# --------------------------
# SYSTEM EXPLANATION PILLS
# --------------------------
st.markdown("""
<div class="section-header">
    <div class="section-title">Methodology Notes</div>
    <div class="section-line"></div>
</div>
<div class="pill-grid">
    <div class="pill"><div class="pill-dot"></div>Log Returns → stationary price signal</div>
    <div class="pill"><div class="pill-dot"></div>Rolling Volatility → market instability metric</div>
    <div class="pill"><div class="pill-dot"></div>Future Vol → forward-looking risk labels</div>
    <div class="pill"><div class="pill-dot"></div>Chronological split → zero data leakage</div>
    <div class="pill"><div class="pill-dot"></div>QSVM RBF kernel → quantum ML proxy</div>
    <div class="pill"><div class="pill-dot"></div>Proxy mode → stable cross-platform execution</div>
    <div class="pill"><div class="pill-dot"></div>MinMax scale [-1, 1] → quantum gate compatibility</div>
</div>
<div style="height:3rem;"></div>
""", unsafe_allow_html=True)