from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

METRIC_LABELS = {
    "Age": "Usia (Tahun)",
    "Sh/90": "Total Tembakan per 90'",
    "SoT/90": "Tembakan On Target per 90'",
    "SoT%": "Akurasi Tembakan On Target (%)",
    "G/Sh": "Efisiensi Gol per Tembakan",
    "Gls_per90": "Gol per 90'",
    "Ast_per90": "Assist per 90'",
    "Crs_per90": "Umpan Silang (Crosses) per 90'",
    "TklW_per90": "Tekel Sukses (Won) per 90'",
    "Int_per90": "Intersep / Potong Bola per 90'",
    "Fld_per90": "Dilanggar Lawan (Fouls Drawn) per 90'",
    "Fls_per90": "Pelanggaran Dilakukan (Fouls Committed) per 90'",
}

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Football Player Recommender", layout="wide", page_icon="⚽"
)
st.title("⚽ Sistem Rekomendasi Pemain Sepak Bola 2025/2026")
st.markdown(
    "Temukan pemain dengan profil statistik dan gaya bermain paling identik berbasis Machine Learning (Nearest Neighbors)."
)

# --- BASE DIRECTORY & PATH FILE ---
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "players_cleaned.parquet"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
MODEL_PATH = BASE_DIR / "models" / "nn_model.joblib"


# --- FUNGSI LOAD DATA & MODEL ---
@st.cache_data
def load_assets():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File data tidak ditemukan di: {DATA_PATH}")
    if not SCALER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model/scaler belum ada di folder models/")

    df = pd.read_parquet(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    nn_model = joblib.load(MODEL_PATH)

    # --- TAMBAHKAN BAGIAN INI (AUTO-FALLBACK FITUR PER90) ---
    metrics_to_normalize = ['Gls', 'Ast', 'Crs', 'TklW', 'Int', 'Fld', 'Fls']
    for col in metrics_to_normalize:
        col_per90 = f"{col}_per90"
        if col_per90 not in df.columns and col in df.columns:
            # Gunakan np.where untuk menghindari division by zero
            df[col_per90] = np.where(df['90s'] > 0, df[col] / df['90s'], 0.0)
    # --------------------------------------------------------

    return df, scaler, nn_model

try:
    df, scaler, nn_model = load_assets()
except Exception as e:
    st.error(f"Error memuat file: {e}")
    st.stop()

# --- FITUR MODEL ---
features = [
    "Age",
    "Sh/90",
    "SoT/90",
    "SoT%",
    "G/Sh",
    "Gls_per90",
    "Ast_per90",
    "Crs_per90",
    "TklW_per90",
    "Int_per90",
    "Fld_per90",
    "Fls_per90",
]


# --- HELPER FUNCTION: RADAR CHART ---
from scipy import stats


def generate_radar_chart(target_row, rec_row, feature_cols, full_df):
    radar_cols = [col for col in feature_cols if col != "Age"]

    target_vals = []
    rec_vals = []
    for col in radar_cols:
        col_series = full_df[col].fillna(0)
        t_rank = stats.percentileofscore(
            col_series,
            float(target_row[col]) if pd.notna(target_row[col]) else 0,
        )
        r_rank = stats.percentileofscore(
            col_series, float(rec_row[col]) if pd.notna(rec_row[col]) else 0
        )
        target_vals.append(t_rank)
        rec_vals.append(r_rank)

    # Ubah singkatan menjadi nama lengkap yang jelas
    display_names = [METRIC_LABELS.get(c, c) for c in radar_cols]

    # Tutup lingkaran radar
    categories = display_names + [display_names[0]]
    target_vals += [target_vals[0]]
    rec_vals += [rec_vals[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=target_vals,
            theta=categories,
            fill="toself",
            name=f"{target_row['Player']} (Target)",
            line=dict(color="#1f77b4", width=2),
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=rec_vals,
            theta=categories,
            fill="toself",
            name=f"{rec_row['Player']} (Rekomendasi)",
            line=dict(color="#ff7f0e", width=2),
            opacity=0.6,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=10),
            )
        ),
        showlegend=True,
        title=f"Perbandingan Profil Taktis (Persentil): {target_row['Player']} vs {rec_row['Player']}",
        margin=dict(l=60, r=60, t=50, b=50),
        height=550,
    )
    return fig

# --- SIDEBAR (FILTER) ---
st.sidebar.header("Filter Pencarian 🔍")
list_pemain = df["Player"].sort_values().unique()
target_player = st.sidebar.selectbox("Pilih Pemain Target:", list_pemain)

st.sidebar.markdown("---")
n_rekomendasi = st.sidebar.slider(
    "Jumlah Rekomendasi", min_value=3, max_value=10, value=5
)

# --- EKSEKUSI REKOMENDASI ---
if target_player:
    target_data = df[df["Player"] == target_player].iloc[0]

    # Banner profil pemain target
    st.subheader(f"Profil Target: {target_player}")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.markdown(f"**Klub:** {target_data['Squad']}")
    info_col2.markdown(f"**Liga:** {target_data['Comp']}")
    info_col3.markdown(f"**Posisi:** {target_data['Pos']}")
    info_col4.markdown(f"**Usia:** {target_data['Age']} tahun")

    # Ambil fitur & standarisasi (dengan penanganan fillna)
    target_features = target_data[features].fillna(0).values.reshape(1, -1)
    target_scaled = scaler.transform(target_features)

    # Prediksi tetangga terdekat
    distances, indices = nn_model.kneighbors(
        target_scaled, n_neighbors=n_rekomendasi + 1
    )

    rekomendasi_list = []
    top_distances = []
    top_maes = []

    # Loop hasil rekomendasi (mulai dari indeks 1 untuk skip diri sendiri)
    for i in range(1, len(distances[0])):
        idx_pemain = indices[0][i]
        cosine_dist = distances[0][i]
        sim_score = (1 - cosine_dist) * 100

        rec_player = df.iloc[idx_pemain]

        # Hitung Mean Absolute Error (MAE) statistik antar fitur
        stat_target = target_data[features].fillna(0).values
        stat_rec = rec_player[features].fillna(0).values
        mae_val = float(np.mean(np.abs(stat_target - stat_rec)))

        top_distances.append(cosine_dist)
        top_maes.append(mae_val)

        rekomendasi_list.append(
            {
                "Pemain": rec_player["Player"],
                "Klub": rec_player["Squad"],
                "Liga": rec_player["Comp"],
                "Posisi": rec_player["Pos"],
                "Usia": rec_player["Age"],
                "Kemiripan": f"{sim_score:.1f}%",
                "Cosine Distance": round(cosine_dist, 4),
                "Deviasi Rata-rata (MAE)": round(mae_val, 2),
            }
        )

    # --- RINGKASAN METRIK EVALUASI ---
    st.markdown("---")
    st.markdown("### 📊 Metrik Evaluasi Kemiripan")
    m1, m2, m3 = st.columns(3)

    avg_distance = float(np.mean(top_distances))
    avg_sim = (1 - avg_distance) * 100
    avg_mae = float(np.mean(top_maes))

    m1.metric("Rata-rata Kemiripan Top Match", f"{avg_sim:.2f}%")
    m2.metric(
        "Rata-rata Cosine Distance (Spatial Error)",
        f"{avg_distance:.4f}",
        help="Semakin mendekati 0.0, sudut vektor gaya bermain semakin identik.",
    )
    m3.metric(
        "Rata-rata Deviasi Fitur (MAE)",
        f"{avg_mae:.2f}",
        help="Rata-rata selisih nominal statistik per 90 menit.",
    )

    # --- TABEL HASIL REKOMENDASI ---
    st.markdown("---")
    st.markdown("### 📋 Daftar Pemain Rekomendasi Teratas")
    df_hasil = pd.DataFrame(rekomendasi_list)
    st.dataframe(df_hasil, use_container_width=True)

    # --- RADAR CHART COMPARISON ---
    st.markdown("---")
    st.markdown("### 🕸️ Analisis Radar Profil Pemain")
    selected_rec_name = st.selectbox(
        "Pilih pemain dari daftar di atas untuk dibandingkan langsung:",
        df_hasil["Pemain"].tolist(),
    )

    rec_selected_data = df[df["Player"] == selected_rec_name].iloc[0]
    fig_radar = generate_radar_chart(
        target_data, rec_selected_data, feature_cols=features, full_df=df
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- TABEL DETAIL STATISTIK PER 90 MENIT (HEAD-TO-HEAD) ---
st.markdown("---")
st.markdown(
    f"### 📊 Perbandingan Detail Statistik Per 90 Menit: {target_data['Player']} vs {rec_selected_data['Player']}"
)

# Kolom-kolom per 90 yang ingin dibandingkan
stat_per90_cols = [
    "Gls_per90",
    "Ast_per90",
    "Sh/90",
    "SoT/90",
    "SoT%",
    "G/Sh",
    "Crs_per90",
    "TklW_per90",
    "Int_per90",
    "Fld_per90",
    "Fls_per90",
]

comparison_rows = []

for col in stat_per90_cols:
    val_target = (
        float(target_data[col]) if pd.notna(target_data.get(col)) else 0.0
    )
    val_rec = (
        float(rec_selected_data[col])
        if pd.notna(rec_selected_data.get(col))
        else 0.0
    )
    diff = val_rec - val_target

    if col == "SoT%":
        t_str = f"{val_target:.1f}%"
        r_str = f"{val_rec:.1f}%"
        d_str = f"{'+' if diff > 0 else ''}{diff:.1f}%"
    else:
        t_str = f"{val_target:.2f}"
        r_str = f"{val_rec:.2f}"
        d_str = f"{'+' if diff > 0 else ''}{diff:.2f}"

    comparison_rows.append(
        {
            # Nama metrik sekarang jelas dan tidak disingkat
            "Metrik / Atribut": METRIC_LABELS.get(col, col),
            f"{target_data['Player']} (Target)": t_str,
            f"{rec_selected_data['Player']} (Rekomendasi)": r_str,
            "Selisih (Kandidat - Target)": d_str,
        }
    )

df_per90_comp = pd.DataFrame(comparison_rows)
st.dataframe(df_per90_comp, use_container_width=True, hide_index=True)