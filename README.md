# ⚽ Smart Football Scouting System

**Rekomendasi Pemain Pengganti Berbasis Unsupervised Machine Learning dan Dashboard Interaktif Streamlit**

Sistem ini membantu mencari pemain pengganti ketika seorang pemain pergi dari klub. Pertanyaan yang dijawab bukan *"siapa pemain terbaik?"*, melainkan **"siapa pemain lain yang gaya mainnya paling mirip?"**

Setiap pemain diubah menjadi vektor 12 fitur (*sidik jari gaya main*), lalu model `NearestNeighbors` dengan metrik cosine mencari pemain dengan profil paling dekat.

---

## 📌 Ringkasan

| | |
|---|---|
| **Data** | Statistik pemain musim 2025/2026 (`players_data_light-2025_2026.csv`), 2.839 baris × 53 kolom |
| **Data bersih** | 1.464 pemain lapangan × 60 kolom |
| **Pendekatan** | Unsupervised learning: `StandardScaler` + `NearestNeighbors` (cosine) |
| **Fitur model** | 12 fitur (umur, menembak, output serangan, bertahan, duel) |
| **Output** | Top-5 pemain dengan gaya main paling mirip |
| **Antarmuka** | Dashboard Streamlit *(dalam pengembangan)* |

---

## 🗂️ Struktur Proyek

```
.
├── data/
│   ├── raw/
│   │   └── players_data_light-2025_2026.csv
│   └── processed/
│       └── players_cleaned.parquet
├── models/
│   ├── scaler.joblib
│   └── nn_model.joblib
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_model_similarity.ipynb
└── README.md
```

---

## 🔄 Alur Kerja

### 1. EDA dan Pembersihan Data (`01_eda_and_cleaning.ipynb`)

| Langkah | Temuan | Keputusan |
|---|---|---|
| Cek *missing values* | Kolom khusus kiper (`Save%`, `CS%`, `GA90`) hampir seluruhnya kosong untuk pemain lapangan | Buang kiper (`Pos == 'GK'`) |
| Distribusi menit bermain | Banyak pemain dengan menit sangat sedikit menghasilkan angka yang menyesatkan | Buang pemain dengan **< 900 menit** (setara 10 laga penuh) |
| Normalisasi | Total kumulatif bias terhadap durasi bermain | Ubah 7 metrik ke format **per 90 menit** |

Hasilnya: **2.839 → 1.464 pemain**, disimpan ke `data/processed/players_cleaned.parquet`.

Tujuh metrik yang dinormalisasi: `Gls`, `Ast`, `Crs`, `TklW`, `Int`, `Fld`, `Fls`.

### 2. Pemodelan (`02_model_similarity.ipynb`)

1. **Pilih 12 fitur**

   | Sisi permainan | Fitur |
   |---|---|
   | Profil | `Age` |
   | Menembak | `Sh/90`, `SoT/90`, `SoT%`, `G/Sh` |
   | Output serangan | `Gls_per90`, `Ast_per90`, `Crs_per90` |
   | Bertahan | `TklW_per90`, `Int_per90` |
   | Duel fisik | `Fld_per90`, `Fls_per90` |

   Nilai kosong diisi 0 (terutama `SoT%` dan `G/Sh` pada pemain yang tidak pernah menembak).

2. **Standarisasi** dengan `StandardScaler` (rata-rata 0, variansi 1) agar fitur berskala besar seperti `Age` tidak mendominasi jarak.
3. **Latih `NearestNeighbors`** dengan `metric='cosine'`.
4. **Simpan** scaler dan model ke folder `models/` dengan `joblib`.

---

## 📊 Hasil Evaluasi

Dihitung dari top-5 tetangga terdekat untuk seluruh 1.464 pemain:

| Metrik | Nilai |
|---|---|
| Rata-rata cosine distance | 0,1368 |
| Rata-rata kemiripan (`1 - jarak`) | 86,32% |
| Jarak terdekat | 0,0042 |
| Jarak terjauh di top-5 | 0,4455 |

> ⚠️ **Angka 86,32% adalah ukuran kedekatan, bukan akurasi.** Model ini unsupervised dan tidak punya label kebenaran, jadi hasilnya menunjukkan bahwa rekomendasi *dekat secara statistik*, bukan bahwa rekomendasi itu pasti "benar".

---

## 🚀 Cara Menjalankan

**1. Clone dan siapkan environment**

```bash
git clone <url-repo-kamu>
cd <nama-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn joblib matplotlib seaborn pyarrow jupyter
```

**2. Letakkan data mentah** di `data/raw/players_data_light-2025_2026.csv`.

**3. Jalankan notebook secara berurutan** dengan *Restart & Run All*:

1. `notebooks/01_eda_and_cleaning.ipynb`
2. `notebooks/02_model_similarity.ipynb`

> Path data pada notebook 01 perlu disesuaikan dengan lokasi file di komputermu.

---

## 🔎 Contoh Penggunaan Model

```python
import joblib
import pandas as pd

df = pd.read_parquet("data/processed/players_cleaned.parquet")
scaler = joblib.load("models/scaler.joblib")
nn_model = joblib.load("models/nn_model.joblib")

features = [
    "Age", "Sh/90", "SoT/90", "SoT%", "G/Sh",
    "Gls_per90", "Ast_per90", "Crs_per90",
    "TklW_per90", "Int_per90", "Fld_per90", "Fls_per90",
]

X_scaled = scaler.transform(df[features].fillna(0))

# Cari 5 pemain paling mirip dengan pemain tertentu
idx = df.index[df["Player"] == "Brenden Aaronson"][0]
distances, indices = nn_model.kneighbors(X_scaled[[idx]], n_neighbors=6)

# Indeks 0 adalah pemain itu sendiri, jadi dibuang
result = df.iloc[indices[0][1:]][["Player", "Squad", "Pos"]].copy()
result["Similarity (%)"] = (1 - distances[0][1:]) * 100
print(result)
```

---

## ⚠️ Keterbatasan

- **Cosine membandingkan bentuk profil, bukan besar angkanya.** Karena data sudah distandardisasi, dua pemain dengan pola sama tetapi intensitas berbeda bisa tampak sangat mirip.
- **Posisi (`Pos`) tidak dipakai sebagai fitur.** Pemain dengan posisi berbeda bisa direkomendasikan jika profilnya mirip, sehingga hasil sebaiknya difilter per posisi.
- **Fitur condong ke ofensif.** 7 dari 12 fitur terkait serangan, sehingga rekomendasi untuk bek atau gelandang bertahan bisa kurang tajam.
- **Kiper tidak dicakup.** Metrik kiper berbeda total dan dikeluarkan dari analisis.
- **Hanya satu musim** (2025/2026), sehingga tren performa antar musim belum terlihat.
- **Evaluasi hanya berbasis jarak.** Belum ada validasi dari pemandu bakat atau data transfer nyata.

---

## 🗺️ Rencana Pengembangan

- [ ] Dashboard Streamlit: pilih pemain, tampilkan top-5 pengganti
- [ ] Filter rekomendasi berdasarkan posisi, liga, dan rentang umur
- [ ] Bandingkan metrik cosine dengan euclidean
- [ ] Tambah fitur bertahan dan distribusi bola agar lebih seimbang

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · joblib · matplotlib · seaborn · Streamlit

---

## 👤 Penulis

**Abdur Rachman**
