import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Genre Clustering",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark Netflix-style palette)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a0000 50%, #0d0d0d 100%);
        color: #f0f0f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
        border-right: 1px solid #E50914;
    }
    [data-testid="stSidebar"] * { color: #f0f0f0 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a1a, #2a0000);
        border: 1px solid #E50914;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(229,9,20,0.2);
    }
    [data-testid="metric-container"] label { color: #aaa !important; font-size: 0.85rem; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #E50914 !important; font-size: 1.8rem; font-weight: 800;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(90deg, #E50914 0%, #8B0000 50%, #E50914 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 32px;
        box-shadow: 0 8px 40px rgba(229,9,20,0.5);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        right: 40px;
        top: 18px;
        width: 110px;
        height: 110px;
        opacity: 0.14;
        background: white;
        clip-path: polygon(25% 12%, 25% 88%, 82% 50%);
    }
    .real-icon-play {
        display: inline-block;
        width: 54px;
        height: 54px;
        border-radius: 14px;
        background: linear-gradient(135deg, #E50914, #8B0000);
        position: relative;
        box-shadow: 0 4px 20px rgba(229,9,20,0.4);
    }
    .real-icon-play::after {
        content: "";
        position: absolute;
        left: 21px;
        top: 15px;
        width: 0;
        height: 0;
        border-top: 12px solid transparent;
        border-bottom: 12px solid transparent;
        border-left: 18px solid white;
    }
    .inline-real-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        border-radius: 5px;
        background: #E50914;
        margin-right: 8px;
        vertical-align: -3px;
        position: relative;
    }
    .inline-real-icon::after {
        content: "";
        position: absolute;
        left: 7px;
        top: 4px;
        width: 0;
        height: 0;
        border-top: 5px solid transparent;
        border-bottom: 5px solid transparent;
        border-left: 7px solid white;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: white;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.85);
        margin: 0;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(229,9,20,0.15), transparent);
        border-left: 4px solid #E50914;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h2 { color: #E50914; margin: 0; font-size: 1.3rem; font-weight: 700; }

    /* Card wrapper */
    .card {
        background: rgba(26,26,26,0.8);
        border: 1px solid rgba(229,9,20,0.25);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }

    /* Cluster badge */
    .cluster-badge {
        display: inline-block;
        background: linear-gradient(135deg, #E50914, #8B0000);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px 4px;
    }

    /* Genre pill */
    .genre-pill {
        display: inline-block;
        background: rgba(229,9,20,0.15);
        border: 1px solid rgba(229,9,20,0.4);
        color: #ff6b6b;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 2px 3px;
        font-weight: 500;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #E50914, #b00710);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff1a1a, #E50914);
        box-shadow: 0 4px 20px rgba(229,9,20,0.5);
        transform: translateY(-1px);
    }

    .stSelectbox > div > div, .stSlider {
        color: #f0f0f0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(26,26,26,0.8);
        border-radius: 10px 10px 0 0;
        border-bottom: 2px solid #E50914;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #aaa !important;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(229,9,20,0.2) !important;
        color: #E50914 !important;
        border-bottom: 2px solid #E50914;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(229,9,20,0.1) !important;
        border: 1px solid rgba(229,9,20,0.3) !important;
        border-radius: 8px !important;
        color: #E50914 !important;
        font-weight: 600;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #E50914; border-radius: 3px; }

    /* divider */
    hr { border-color: rgba(229,9,20,0.3) !important; }

    /* Info box */
    .info-box {
        background: rgba(229,9,20,0.08);
        border: 1px solid rgba(229,9,20,0.3);
        border-radius: 10px;
        padding: 16px 20px;
        color: #ddd;
        font-size: 0.9rem;
    }
    .info-box b { color: #E50914; }

    /* Table  */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY TEMPLATE  (dark Netflix theme)
# ─────────────────────────────────────────────
NETFLIX_COLORS = ["#E50914", "#FF6B6B", "#FF9F43", "#FECA57",
                  "#48DBFB", "#FF9FF3", "#54A0FF", "#5F27CD",
                  "#00D2D3", "#2ECC71"]

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f0f0", family="Inter"),
        colorway=NETFLIX_COLORS,
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_preprocess():
    df = pd.read_csv("netflix_titles.csv")
    df.dropna(subset=["listed_in"], inplace=True)
    df["genres"] = df["listed_in"].apply(lambda x: [g.strip() for g in x.split(",")])
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["country_clean"] = df["country"].fillna("Unknown").apply(lambda x: x.split(",")[0].strip())
    df["description"] = df["description"].fillna("")
    return df
    
@st.cache_data(show_spinner=False)
def run_kmeans(df, n_clusters):
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["genres"])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(genre_matrix)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(genre_matrix)

    sil = silhouette_score(genre_matrix, labels) if n_clusters > 1 else 0.0

    result_df = df.copy()
    result_df["cluster"] = labels
    result_df["pca_x"] = coords[:, 0]
    result_df["pca_y"] = coords[:, 1]

    # Top genres per cluster
    cluster_genres = {}
    for c in range(n_clusters):
        mask = labels == c
        genre_sums = genre_df[mask].sum().sort_values(ascending=False)
        cluster_genres[c] = genre_sums.head(5).index.tolist()

    # Elbow inertias (for elbow chart)
    inertias = []
    ks = range(2, 12)
    for k in ks:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(genre_matrix)
        inertias.append(km_tmp.inertia_)

    return result_df, cluster_genres, sil, list(ks), inertias, mlb.classes_


def genre_freq(df):
    from collections import Counter
    all_genres = [g for genres in df["genres"] for g in genres]
    return pd.DataFrame(Counter(all_genres).most_common(30), columns=["genre", "count"])


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <span class="real-icon-play"></span>
        <h2 style="color:#E50914; margin:4px 0; font-size:1.4rem; font-weight:800;">Netflix Analyzer</h2>
        <p style="color:#888; font-size:0.8rem; margin:0;">K-Means Genre Clustering</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Dataset")
    st.success("Dataset netflix_titles.csv dimuat otomatis dari repository GitHub.")

    st.divider()

    st.markdown("### Konfigurasi K-Means")
    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=5, step=1)
    content_filter = st.selectbox("Filter Tipe Konten", ["Semua", "Movie", "TV Show"])

    st.divider()

    st.markdown("""
    <div class="info-box">
        <b>Tentang Algoritma</b><br><br>
        <b>K-Means Clustering</b> mengelompokkan konten Netflix berdasarkan kesamaan genre menggunakan representasi vektor biner (MultiLabelBinarizer), kemudian divisualisasikan dengan <b>PCA 2D</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Jalankan Analisis", use_container_width=True)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">Analisis Genre Film & Serial<br>Terpopuler di Netflix</p>
    <p class="hero-sub">Menggunakan Algoritma K-Means Clustering • Visualisasi Interaktif Berbasis Streamlit</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
# Load data otomatis dari file netflix_titles.csv di repository
with st.spinner("Memuat dan memproses dataset..."):
    df_raw = load_and_preprocess()

# Apply content filter
if content_filter == "Movie":
    df_filtered = df_raw[df_raw["type"] == "Movie"].reset_index(drop=True)
elif content_filter == "TV Show":
    df_filtered = df_raw[df_raw["type"] == "TV Show"].reset_index(drop=True)
else:
    df_filtered = df_raw.reset_index(drop=True)


# ─── OVERVIEW METRICS ───────────────────────
st.markdown('<div class="section-header"><h2>Ringkasan Dataset</h2></div>', unsafe_allow_html=True)

all_genres = [g for gs in df_filtered["genres"] for g in gs]
unique_genres = len(set(all_genres))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Konten", f"{len(df_filtered):,}")
with col2:
    movies = len(df_filtered[df_filtered["type"] == "Movie"])
    st.metric("Film (Movie)", f"{movies:,}")
with col3:
    shows = len(df_filtered[df_filtered["type"] == "TV Show"])
    st.metric("Serial (TV Show)", f"{shows:,}")
with col4:
    st.metric("Jumlah Genre Unik", f"{unique_genres}")


# ─── RUN CLUSTERING ─────────────────────────
if run_btn or "clustered_df" not in st.session_state:
    with st.spinner("Menjalankan K-Means Clustering..."):
        result_df, cluster_genres, sil_score, ks, inertias, all_genre_names = run_kmeans(df_filtered, n_clusters)
    st.session_state["clustered_df"] = result_df
    st.session_state["cluster_genres"] = cluster_genres
    st.session_state["sil_score"] = sil_score
    st.session_state["ks"] = ks
    st.session_state["inertias"] = inertias
    st.session_state["all_genre_names"] = all_genre_names
    st.session_state["n_clusters_used"] = n_clusters
else:
    result_df = st.session_state["clustered_df"]
    cluster_genres = st.session_state["cluster_genres"]
    sil_score = st.session_state["sil_score"]
    ks = st.session_state["ks"]
    inertias = st.session_state["inertias"]
    all_genre_names = st.session_state["all_genre_names"]
    n_clusters_used = st.session_state.get("n_clusters_used", n_clusters)


# ─── TABS ────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Hasil Clustering",
    "Analisis Genre",
    "Evaluasi Model",
    "Distribusi Konten",
    "Eksplorasi Data"
])


# ══════════════════════════════════════════════
# TAB 1 — CLUSTERING RESULTS
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown('<div class="section-header"><h2>Visualisasi Cluster (PCA 2D)</h2></div>', unsafe_allow_html=True)

        result_df["cluster_label"] = result_df["cluster"].apply(lambda x: f"Cluster {x+1}")
        fig_scatter = px.scatter(
            result_df,
            x="pca_x", y="pca_y",
            color="cluster_label",
            hover_data={"title": True, "type": True, "listed_in": True, "pca_x": False, "pca_y": False},
            labels={"pca_x": "Komponen Utama 1", "pca_y": "Komponen Utama 2", "cluster_label": "Cluster"},
            color_discrete_sequence=NETFLIX_COLORS,
            opacity=0.75,
            height=500,
        )
        fig_scatter.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(0,0,0,0.4)")))
        fig_scatter.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Peta Cluster Genre Netflix — K-Means + PCA",
            showlegend=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header"><h2>Detail Cluster</h2></div>', unsafe_allow_html=True)
        for c_idx, genres_list in cluster_genres.items():
            count = len(result_df[result_df["cluster"] == c_idx])
            color = NETFLIX_COLORS[c_idx % len(NETFLIX_COLORS)]
            st.markdown(f"""
            <div class="card" style="border-color:{color}40; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-weight:700; color:{color}; font-size:1rem;">Cluster {c_idx+1}</span>
                    <span style="background:{color}22; color:{color}; padding:2px 10px; border-radius:12px; font-size:0.8rem;">{count} konten</span>
                </div>
                <div>{''.join([f'<span class="genre-pill">{g}</span>' for g in genres_list])}</div>
            </div>
            """, unsafe_allow_html=True)

    # Cluster size bar
    st.markdown('<div class="section-header"><h2>Distribusi Ukuran Cluster</h2></div>', unsafe_allow_html=True)
    cluster_counts = result_df.groupby("cluster_label").size().reset_index(name="Jumlah Konten")
    fig_bar = px.bar(
        cluster_counts,
        x="cluster_label", y="Jumlah Konten",
        color="cluster_label",
        color_discrete_sequence=NETFLIX_COLORS,
        text="Jumlah Konten",
        height=320,
    )
    fig_bar.update_traces(textposition="outside", textfont=dict(color="#fff"))
    fig_bar.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False,
                          title="Jumlah Konten per Cluster")
    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — GENRE ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    gf = genre_freq(df_filtered)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header"><h2>Top 15 Genre Terpopuler</h2></div>', unsafe_allow_html=True)
        top15 = gf.head(15).sort_values("count")
        fig_h = px.bar(
            top15, x="count", y="genre",
            orientation="h",
            color="count",
            color_continuous_scale=["#8B0000", "#E50914", "#FF6B6B"],
            text="count",
            height=460,
        )
        fig_h.update_traces(textposition="outside", textfont=dict(color="#fff"))
        fig_h.update_layout(**PLOTLY_TEMPLATE["layout"], title="Frekuensi Genre", showlegend=False,
                            coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header"><h2>Proporsi Genre (Pie)</h2></div>', unsafe_allow_html=True)
        top10 = gf.head(10)
        others = pd.DataFrame([{"genre": "Lainnya", "count": gf.iloc[10:]["count"].sum()}])
        pie_data = pd.concat([top10, others], ignore_index=True)
        fig_pie = px.pie(
            pie_data, names="genre", values="count",
            color_discrete_sequence=NETFLIX_COLORS,
            hole=0.4,
            height=460,
        )
        fig_pie.update_traces(textinfo="label+percent", textfont_size=11, pull=[0.05] + [0]*10)
        fig_pie.update_layout(**PLOTLY_TEMPLATE["layout"], title="Porsi Genre di Netflix",
                              showlegend=True, legend=dict(orientation="v", x=1.05))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Genre per cluster heatmap
    st.markdown('<div class="section-header"><h2>Heatmap Genre per Cluster</h2></div>', unsafe_allow_html=True)
    top_genres_list = gf.head(15)["genre"].tolist()
    heatmap_data = []
    for c_idx in range(n_clusters):
        sub = df_filtered[result_df["cluster"] == c_idx]
        for g in top_genres_list:
            count = sub["genres"].apply(lambda gs: g in gs).sum()
            heatmap_data.append({"Cluster": f"Cluster {c_idx+1}", "Genre": g, "Jumlah": count})
    hm_df = pd.DataFrame(heatmap_data).pivot(index="Cluster", columns="Genre", values="Jumlah")
    fig_hm = px.imshow(
        hm_df,
        color_continuous_scale=["#0d0d0d", "#8B0000", "#E50914", "#FF6B6B"],
        text_auto=True,
        aspect="auto",
        height=350,
    )
    fig_hm.update_layout(**PLOTLY_TEMPLATE["layout"], title="Distribusi Genre per Cluster")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Genre trend over years
    st.markdown('<div class="section-header"><h2>Tren Genre Sepanjang Tahun</h2></div>', unsafe_allow_html=True)
    selected_genres = st.multiselect(
        "Pilih Genre untuk Ditampilkan",
        options=gf["genre"].tolist(),
        default=gf.head(5)["genre"].tolist()
    )
    if selected_genres:
        trend_records = []
        for _, row in df_filtered.dropna(subset=["year_added"]).iterrows():
            for g in row["genres"]:
                if g in selected_genres:
                    trend_records.append({"year": int(row["year_added"]), "genre": g})
        if trend_records:
            trend_df = (pd.DataFrame(trend_records)
                        .groupby(["year", "genre"])
                        .size().reset_index(name="count")
                        .query("year >= 2015"))
            fig_line = px.line(
                trend_df, x="year", y="count", color="genre",
                markers=True,
                color_discrete_sequence=NETFLIX_COLORS,
                height=350,
            )
            fig_line.update_layout(**PLOTLY_TEMPLATE["layout"], title="Tren Penambahan Konten per Genre")
            st.plotly_chart(fig_line, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><h2>Elbow Method — Menentukan K Optimal</h2></div>', unsafe_allow_html=True)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(ks), y=inertias,
        mode="lines+markers",
        line=dict(color="#E50914", width=3),
        marker=dict(size=9, color="#FF6B6B", line=dict(width=2, color="#E50914")),
        name="Inertia",
        fill="tozeroy",
        fillcolor="rgba(229,9,20,0.08)"
    ))
    fig_elbow.add_vline(
        x=n_clusters,
        line_dash="dash", line_color="#FECA57",
        annotation_text=f"K={n_clusters} (dipilih)",
        annotation_font_color="#FECA57"
    )
    fig_elbow.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="Elbow Method — Inertia vs Jumlah Cluster",
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        height=380
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Silhouette Score", f"{sil_score:.4f}", help="Rentang: -1 s/d 1. Semakin mendekati 1 = cluster lebih baik.")
    with col_m2:
        st.metric("Jumlah Cluster (K)", f"{n_clusters}")
    with col_m3:
        st.metric("Total Data", f"{len(df_filtered):,}")

    st.divider()
    st.markdown('<div class="section-header"><h2>Interpretasi Hasil Clustering</h2></div>', unsafe_allow_html=True)

    for c_idx, genres_list in cluster_genres.items():
        count = len(result_df[result_df["cluster"] == c_idx])
        pct = count / len(result_df) * 100
        sub_types = result_df[result_df["cluster"] == c_idx]["type"].value_counts().to_dict()
        color = NETFLIX_COLORS[c_idx % len(NETFLIX_COLORS)]
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {color}; padding-left:20px;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                <span class="inline-real-icon" style="background:{color};"></span>
                <div>
                    <span style="font-size:1.1rem; font-weight:700; color:{color};">Cluster {c_idx+1}</span>
                    <span style="color:#888; margin-left:10px;">({count} konten · {pct:.1f}%)</span>
                </div>
            </div>
            <div style="color:#ccc; font-size:0.9rem; margin-bottom:8px;">
                <b style="color:#aaa;">Genre Dominan:</b> {'  '.join([f'<span class="genre-pill">{g}</span>' for g in genres_list])}
            </div>
            <div style="color:#aaa; font-size:0.85rem;">
                Movies: <b style="color:#f0f0f0;">{sub_types.get("Movie", 0)}</b> &nbsp;|&nbsp;
                TV Shows: <b style="color:#f0f0f0;">{sub_types.get("TV Show", 0)}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — CONTENT DISTRIBUTION
# ══════════════════════════════════════════════
with tab4:
    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown('<div class="section-header"><h2>Movie vs TV Show</h2></div>', unsafe_allow_html=True)
        type_counts = df_filtered["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig_donut = px.pie(
            type_counts, names="type", values="count",
            hole=0.55,
            color_discrete_sequence=["#E50914", "#48DBFB"],
            height=320
        )
        fig_donut.update_traces(textinfo="label+percent+value")
        fig_donut.update_layout(**PLOTLY_TEMPLATE["layout"], title="Rasio Movie vs TV Show")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_y:
        st.markdown('<div class="section-header"><h2>Distribusi Rating</h2></div>', unsafe_allow_html=True)
        rating_counts = df_filtered["rating"].value_counts().head(10).reset_index()
        rating_counts.columns = ["rating", "count"]
        fig_rating = px.bar(
            rating_counts, x="rating", y="count",
            color="count",
            color_continuous_scale=["#8B0000", "#E50914", "#FF9F43"],
            text="count",
            height=320
        )
        fig_rating.update_traces(textposition="outside", textfont=dict(color="#fff"))
        fig_rating.update_layout(**PLOTLY_TEMPLATE["layout"], title="Distribusi Rating Konten",
                                 coloraxis_showscale=False)
        st.plotly_chart(fig_rating, use_container_width=True)

    # Release year histogram
    st.markdown('<div class="section-header"><h2>Penambahan Konten per Tahun</h2></div>', unsafe_allow_html=True)
    year_data = df_filtered.dropna(subset=["year_added"])
    year_counts = year_data.groupby(["year_added", "type"]).size().reset_index(name="count")
    year_counts = year_counts[year_counts["year_added"] >= 2010]
    fig_year = px.bar(
        year_counts, x="year_added", y="count", color="type",
        color_discrete_sequence=["#E50914", "#48DBFB"],
        barmode="stack",
        text_auto=False,
        height=350
    )
    fig_year.update_layout(**PLOTLY_TEMPLATE["layout"],
                           title="Pertumbuhan Konten Netflix (2010–kini)",
                           xaxis_title="Tahun", yaxis_title="Jumlah Konten")
    st.plotly_chart(fig_year, use_container_width=True)

    # Top countries
    st.markdown('<div class="section-header"><h2>Top 15 Negara Produsen</h2></div>', unsafe_allow_html=True)
    country_counts = df_filtered["country_clean"].value_counts().head(15).reset_index()
    country_counts.columns = ["country", "count"]
    country_counts = country_counts[country_counts["country"] != "Unknown"]
    fig_country = px.bar(
        country_counts, x="count", y="country",
        orientation="h",
        color="count",
        color_continuous_scale=["#8B0000", "#E50914", "#FF9F43"],
        text="count",
        height=420
    )
    fig_country.update_traces(textposition="outside", textfont=dict(color="#fff"))
    fig_country.update_layout(**PLOTLY_TEMPLATE["layout"], title="Konten Berdasarkan Negara Asal",
                              coloraxis_showscale=False)
    st.plotly_chart(fig_country, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><h2>Eksplorasi Data Terkluster</h2></div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_cluster = st.selectbox("Filter Cluster", ["Semua"] + [f"Cluster {i+1}" for i in range(n_clusters)])
    with col_f2:
        selected_type = st.selectbox("Filter Tipe", ["Semua", "Movie", "TV Show"])
    with col_f3:
        search_term = st.text_input("Cari Judul", placeholder="Ketik nama film/serial...")

    display_df = result_df[["title", "type", "listed_in", "country_clean", "release_year", "rating", "cluster_label"]].copy()
    display_df.columns = ["Judul", "Tipe", "Genre", "Negara", "Tahun", "Rating", "Cluster"]

    if selected_cluster != "Semua":
        display_df = display_df[display_df["Cluster"] == selected_cluster]
    if selected_type != "Semua":
        display_df = display_df[display_df["Tipe"] == selected_type]
    if search_term:
        display_df = display_df[display_df["Judul"].str.contains(search_term, case=False, na=False)]

    st.info(f"Menampilkan **{len(display_df):,}** konten dari total **{len(result_df):,}**")
    st.dataframe(
        display_df.reset_index(drop=True),
        height=420,
        use_container_width=True,
        column_config={
            "Cluster": st.column_config.TextColumn("Cluster"),
            "Tipe": st.column_config.TextColumn("Tipe"),
            "Tahun": st.column_config.NumberColumn("Tahun", format="%d"),
        }
    )

    # Download
    csv_out = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Hasil Clustering (CSV)",
        data=csv_out,
        file_name="netflix_kmeans_result.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.82rem; padding:16px 0;">
    <span class="inline-real-icon"></span><b style="color:#E50914;">Netflix Genre Clustering App</b> — 
    Dibangun dengan Streamlit + Scikit-learn (K-Means) + Plotly<br>
    Analisis Genre Film dan Serial Terpopuler pada Netflix Menggunakan Algoritma K-Means Berbasis Streamlit
</div>
""", unsafe_allow_html=True)
