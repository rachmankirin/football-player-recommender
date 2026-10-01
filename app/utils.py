import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_radar_comparison(target_row, rec_row, feature_cols):
    """Membuat visualisasi Radar Chart interaktif membandingkan

    pemain target vs pemain rekomendasi menggunakan Plotly.
    """
    # Ambil nilai fitur numerik (ganti NaN dengan 0)
    target_values = target_row[feature_cols].fillna(0).values.tolist()
    rec_values = rec_row[feature_cols].fillna(0).values.tolist()

    # Tutup loop grafik radar (kembali ke titik awal)
    categories = list(feature_cols) + [feature_cols[0]]
    target_values += [target_values[0]]
    rec_values += [rec_values[0]]

    fig = go.Figure()

    # Traces untuk Pemain Target
    fig.add_trace(
        go.Scatterpolar(
            r=target_values,
            theta=categories,
            fill="toself",
            name=f"{target_row['Player']} (Target)",
            line=dict(color="#1f77b4", width=2),
            opacity=0.7,
        )
    )

    # Traces untuk Pemain Rekomendasi
    fig.add_trace(
        go.Scatterpolar(
            r=rec_values,
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
                showticklabels=True,
                ticks="outside",
            )
        ),
        showlegend=True,
        title=f"Perbandingan Gaya Main: {target_row['Player']} vs {rec_row['Player']}",
        margin=dict(l=40, r=40, t=50, b=40),
        height=500,
    )

    return fig


def calculate_feature_mae(target_row, rec_row, feature_cols):
    """Menghitung rata-rata selisih mutlak nilai statistik riil per 90 menit."""
    v_target = target_row[feature_cols].fillna(0).values
    v_rec = rec_row[feature_cols].fillna(0).values
    return float(np.mean(np.abs(v_target - v_rec)))