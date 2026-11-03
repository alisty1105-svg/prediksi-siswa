
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ==========================================
# 1. KONFIGURASI DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Dashboard Risiko Akademik",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Risiko Akademik")
st.subheader("SD Ainul Hayat - Siswa Kelas VI")
st.write(
    "Dashboard analisis akademik dan prediksi dini "
    "kebutuhan pendampingan belajar siswa."
)

# ==========================================
# 2. MEMBACA DATASET
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_excel(
        "nilai_siswa.xlsx",
        sheet_name="datasiswa",
        header=1
    )
    df.columns = df.columns.astype(str).str.strip()
    return df

df = load_data()

# ==========================================
# 3. MENENTUKAN VARIABEL
# ==========================================

mata_pelajaran = [
    "nilai_bahasa_indonesia",
    "nilai_matematika",
    "nilai_ipas",
    "nilai_pai",
    "nilai_bahasa_inggris",
    "nilai_pancasila",
    "nilai_pjok"
]

fitur = mata_pelajaran + [
    "kehadiran_persen",
    "ketuntasan_tugas",
    "keterlibatan_belajar"
]

nama_mapel = {
    "nilai_bahasa_indonesia": "Bahasa Indonesia",
    "nilai_matematika": "Matematika",
    "nilai_ipas": "IPAS",
    "nilai_pai": "PAI",
    "nilai_bahasa_inggris": "Bahasa Inggris",
    "nilai_pancasila": "Pancasila",
    "nilai_pjok": "PJOK"
}

# Konversi kolom numerik
kolom_numerik = fitur + ["angkatan", "risiko_akademik"]

for kolom in kolom_numerik:
    df[kolom] = pd.to_numeric(
        df[kolom], errors="coerce"
    )

# ==========================================
# 4. SIDEBAR FILTER
# ==========================================

st.sidebar.title("Pengaturan Dashboard")

daftar_angkatan = sorted(
    df["angkatan"].dropna().unique()
)

pilihan = st.sidebar.selectbox(
    "Pilih Angkatan",
    ["Semua Angkatan"] + daftar_angkatan
)

if pilihan == "Semua Angkatan":
    data = df.copy()
else:
    data = df[df["angkatan"] == pilihan].copy()

# ==========================================
# 5. RINGKASAN DATA
# ==========================================

st.header("Ringkasan Akademik")

jumlah_siswa = len(data)
rata_nilai = data[mata_pelajaran].mean().mean()
rata_hadir = data["kehadiran_persen"].mean()

data_berlabel = data[
    data["risiko_akademik"].notna()
]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Jumlah Siswa",
    jumlah_siswa
)

col2.metric(
    "Rata-rata Nilai",
    round(rata_nilai, 2)
)

col3.metric(
    "Rata-rata Kehadiran",
    f"{rata_hadir:.2f}%"
)

col4.metric(
    "Data Berlabel",
    len(data_berlabel)
)

st.divider()

# ==========================================
# 6. GRAFIK RATA-RATA NILAI
# ==========================================

st.header("Visualisasi Nilai Akademik")

rata_mapel = (
    data[mata_pelajaran]
    .mean()
    .rename(index=nama_mapel)
    .reset_index()
)

rata_mapel.columns = [
    "Mata Pelajaran",
    "Rata-rata Nilai"
]

fig1 = px.bar(
    rata_mapel,
    x="Mata Pelajaran",
    y="Rata-rata Nilai",
    text_auto=".2f",
    title="Rata-rata Nilai Setiap Mata Pelajaran"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ==========================================
# 7. GRAFIK KETERLIBATAN BELAJAR
# ==========================================

st.header("Analisis Keterlibatan Belajar")

engagement = pd.DataFrame({
    "Indikator": [
        "Kehadiran",
        "Ketuntasan Tugas",
        "Keterlibatan Belajar"
    ],
    "Rata-rata": [
        data["kehadiran_persen"].mean(),
        data["ketuntasan_tugas"].mean(),
        data["keterlibatan_belajar"].mean()
    ]
})

fig2 = px.bar(
    engagement,
    x="Indikator",
    y="Rata-rata",
    text_auto=".2f",
    title="Rata-rata Keterlibatan Belajar Siswa"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==========================================
# 8. ANALISIS RISIKO AKADEMIK
# ==========================================

st.header("Distribusi Risiko Akademik")

risiko = data[
    data["risiko_akademik"].notna()
].copy()

if not risiko.empty:

    distribusi = (
        risiko["risiko_akademik"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .reset_index()
    )

    distribusi.columns = [
        "Kategori",
        "Jumlah"
    ]

    distribusi["Kategori"] = (
        distribusi["Kategori"]
        .map({
            0: "Tidak Berisiko",
            1: "Berisiko"
        })
    )

    fig3 = px.pie(
        distribusi,
        names="Kategori",
        values="Jumlah",
        hole=0.4,
        title="Proporsi Risiko Akademik"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

else:
    st.info("Belum terdapat data risiko pada filter ini.")

# ==========================================
# 9. PEMODELAN RANDOM FOREST
# ==========================================

st.header("Prediksi Risiko Akademik")

st.info(
    "Model menggunakan data berlabel angkatan 1-4 "
    "untuk memperkirakan risiko akademik angkatan 5."
)

data_latih = df[
    df["angkatan"].isin([1, 2, 3, 4])
    & df["risiko_akademik"].notna()
].copy()

data_prediksi = df[
    df["angkatan"] == 5
].copy()

if len(data_prediksi) == 0:
    st.warning("Data angkatan 5 tidak ditemukan.")

elif data_latih["risiko_akademik"].nunique() != 2:
    st.error("Data latih harus memiliki label 0 dan 1.")

else:

    X_train = data_latih[fitur]

    y_train = data_latih[
        "risiko_akademik"
    ].astype(int)

    X_test = data_prediksi[fitur]

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2
            )
        )
    ])

    model.fit(X_train, y_train)

    probabilitas = model.predict_proba(X_test)

    kelas = list(
        model.named_steps["classifier"].classes_
    )

    indeks_risiko = kelas.index(1)

    probabilitas_risiko = probabilitas[
        :, indeks_risiko
    ]

    # ======================================
    # 10. HASIL PREDIKSI
    # ======================================

    ambang = st.slider(
        "Ambang probabilitas risiko",
        0.10,
        0.90,
        0.50,
        0.05
    )

    hasil = data_prediksi[
        ["id_siswa", "angkatan"]
    ].copy()

    hasil["Probabilitas"] = probabilitas_risiko

    hasil["Prediksi"] = (
        hasil["Probabilitas"] >= ambang
    ).astype(int)

    hasil["Kategori"] = np.where(
        hasil["Prediksi"] == 1,
        "Prioritas Peninjauan",
        "Pemantauan Rutin"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Jumlah Prediksi",
        len(hasil)
    )

    b.metric(
        "Prioritas Peninjauan",
        int((hasil["Prediksi"] == 1).sum())
    )

    c.metric(
        "Pemantauan Rutin",
        int((hasil["Prediksi"] == 0).sum())
    )

    st.subheader("Tabel Hasil Prediksi")

    hasil_tampil = hasil.copy()

    hasil_tampil["Probabilitas"] = (
        hasil_tampil["Probabilitas"] * 100
    ).round(2)

    st.dataframe(
        hasil_tampil,
        use_container_width=True
    )

    # Grafik probabilitas
    fig4 = px.histogram(
        hasil,
        x="Probabilitas",
        nbins=15,
        title="Distribusi Probabilitas Risiko"
    )

    fig4.add_vline(
        x=ambang,
        line_dash="dash",
        annotation_text="Ambang Risiko"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # ======================================
    # 11. FEATURE IMPORTANCE
    # ======================================

    st.subheader("Faktor yang Berkontribusi pada Model")

    importance = pd.DataFrame({
        "Variabel": fitur,
        "Importance": (
            model.named_steps[
                "classifier"
            ].feature_importances_
        )
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    fig5 = px.bar(
        importance.head(10),
        x="Importance",
        y="Variabel",
        orientation="h",
        title="10 Feature Importance Tertinggi"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # ======================================
    # 12. DOWNLOAD HASIL
    # ======================================

    st.download_button(
        label="Download Hasil Prediksi CSV",
        data=hasil.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )

# ==========================================
# 13. MENAMPILKAN DATASET
# ==========================================

st.divider()

st.header("Data Akademik Siswa")

with st.expander("Lihat seluruh data"):
    st.dataframe(
        data,
        use_container_width=True
    )

st.caption(
    "Dashboard Akademik SD Ainul Hayat | "
    "Random Forest sebagai alat bantu identifikasi dini."
)

