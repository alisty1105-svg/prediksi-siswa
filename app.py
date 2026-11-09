import io
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# KONFIGURASI DAN TEMA
# =========================================================
st.set_page_config(
    page_title="Dashboard Akademik SD Ainul Hayat",
    page_icon="logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root { --green-dark:#14532d; --green:#2f855a; --green-light:#e8f5e9; }
.stApp { background: linear-gradient(180deg,#f5fbf5 0%,#ffffff 55%); }
section[data-testid="stSidebar"] { background: #e5f3e7; }
section[data-testid="stSidebar"] * { color:#164b2c; }
h1,h2,h3 { color:#14532d !important; }
div[data-testid="stMetric"] {
  background:#e8f5e9; border:1px solid #c5e1c5;
  padding:14px 16px; border-radius:14px;
}
div.stButton > button, div.stDownloadButton > button {
  background:#17633a; color:white; border:0; border-radius:9px;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  background:#2f855a; color:white; border:0;
}
.hero {
  background:linear-gradient(105deg,#14532d,#3b8d58);
  color:white; padding:24px 28px; border-radius:18px; margin-bottom:18px;
}
.hero h1,.hero p { color:white !important; }
.small-note { color:#53715c; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

    # =========================================================
# KONSTANTA DATA
# =========================================================
MAPEL = {
    "nilai_bahasa_indonesia": "Bahasa Indonesia",
    "nilai_matematika": "Matematika",
    "nilai_ipas": "IPAS",
    "nilai_pai": "Pendidikan Agama Islam",
    "nilai_bahasa_inggris": "Bahasa Inggris",
    "nilai_pancasila": "Pendidikan Pancasila",
    "nilai_pjok": "PJOK",
}
FITUR_ENGAGEMENT = [
    "kehadiran_persen", "ketuntasan_tugas", "keterlibatan_belajar"
]
FITUR_DEFAULT = list(MAPEL.keys()) + FITUR_ENGAGEMENT
TARGET_DEFAULT = "risiko_akademik"

# =========================================================
# FUNGSI BANTUAN
# =========================================================
def read_dataset(uploaded_file):
    """Membaca CSV/Excel; Excel memakai sheet pertama secara default."""
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.session_state.get("sheet_choice", xls.sheet_names[0])
        if sheet not in xls.sheet_names:
            sheet = xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=sheet)
    raise ValueError("Format file harus CSV, XLSX, atau XLS.")

def clean_columns(frame):
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str).str.strip()
        .str.lower().str.replace(" ", "_", regex=False)
    )
    return frame

def standardize_data(frame):
    frame = clean_columns(frame)
    # Konversi kolom yang lazim numerik, tanpa memaksa kolom identitas menjadi angka.
    numeric_candidates = FITUR_DEFAULT + ["angkatan", TARGET_DEFAULT]
    for col in numeric_candidates:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame

def find_student_id(frame):
    for candidate in ["id_siswa", "nis", "nisn", "nomor_induk", "nama_siswa", "nama"]:
        if candidate in frame.columns:
            return candidate
    return None

def show_table(frame, height=420):
    st.dataframe(frame, use_container_width=True, height=height, hide_index=True)

def model_factory(name):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", min_samples_leaf=2
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=6
        )
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced"
        )
    return GradientBoostingClassifier(random_state=42)

def build_pipeline(X, algorithm):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler() if algorithm == "Logistic Regression" else "passthrough")
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model_factory(algorithm))])

def image_bytes(upload):
    return upload.getvalue() if upload is not None else None

# =========================================================
# INISIALISASI DATA
# =========================================================
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "model_result" not in st.session_state:
    st.session_state.model_result = None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.image("logo.jpg", width=120)
st.sidebar.markdown("## SD Ainul Hayat")
st.sidebar.caption("Sistem Informasi Akademik dan Prediksi Pendampingan Belajar Siswa")
menu = st.sidebar.radio(
    "MENU UTAMA",
    [
        "Dashboard Utama",
        "Data Siswa",
        "Data Angkatan",
        "Nilai Semua Komponen",
        "Analisis Akademik",
        "Pemodelan & Prediksi",
        "Input Dataset Baru",
        "Informasi Dataset"
    ]
)
st.sidebar.divider()
st.sidebar.markdown("### Identitas dan tampilan sekolah")
logo_upload = st.sidebar.file_uploader(
    "Unggah logo sekolah (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo"
)
foto_upload = st.sidebar.file_uploader(
    "Unggah foto/ilustrasi anak sekolah (PNG/JPG)", type=["png", "jpg", "jpeg"], key="foto"
)
st.sidebar.caption("Logo dan foto yang diunggah hanya ditampilkan pada sesi dashboard ini.")

# =========================================================
# MEMUAT DATASET
# =========================================================
st.markdown('<div class="hero"><h1>🌿 Dashboard Akademik SD Ainul Hayat</h1><p>Monitoring data siswa, nilai seluruh komponen, analisis akademik, dan prediksi dini kebutuhan pendampingan belajar siswa kelas VI.</p></div>', unsafe_allow_html=True)

if st.session_state.dataset is None:
    st.info("Silakan unggah dataset Excel/CSV pada menu **Input Dataset Baru**, atau letakkan file nilai_siswa.xlsx di folder yang sama dengan program.")
    default_path = "nilai_siswa.xlsx"
    try:
        raw = pd.read_excel(default_path, sheet_name="datasiswa", header=1)
        st.session_state.dataset = standardize_data(raw)
    except Exception:
        st.warning("Dataset belum dimuat. Dashboard akan menampilkan menu input dataset terlebih dahulu.")
else:
    st.session_state.dataset = standardize_data(st.session_state.dataset)

df = st.session_state.dataset

# Header sekolah: logo dan gambar hanya jika pengguna mengunggah file.
if logo_upload is not None or foto_upload is not None:
    left, right = st.columns([1, 3])
    with left:
        if logo_upload is not None:
            st.image(logo_upload, caption="Logo sekolah", width=135)
    with right:
        if foto_upload is not None:
            st.image(foto_upload, caption="Foto/ilustrasi siswa SD", use_container_width=True)

if df is None:
    if menu == "Input Dataset Baru":
        st.header("Input Dataset Baru")
        st.write("Unggah dataset Excel/CSV untuk mulai menggunakan dashboard.")
        first_upload = st.file_uploader("Pilih file Excel/CSV", type=["xlsx", "xls", "csv"], key="first_dataset_upload")
        if first_upload is not None:
            try:
                if first_upload.name.lower().endswith((".xlsx", ".xls")):
                    first_book = pd.ExcelFile(first_upload)
                    first_sheet = st.selectbox("Pilih sheet", first_book.sheet_names)
                    first_upload.seek(0)
                    first_data = pd.read_excel(first_upload, sheet_name=first_sheet)
                else:
                    first_data = pd.read_csv(first_upload)
                first_data = standardize_data(first_data)
                st.write(f"Data ditemukan: {len(first_data)} baris dan {len(first_data.columns)} kolom.")
                show_table(first_data.head(20))
                if st.button("Gunakan dataset sebagai data aktif", type="primary"):
                    st.session_state.dataset = first_data
                    st.rerun()
            except Exception as e:
                st.error(f"File tidak dapat dibaca: {e}")
    else:
        st.warning("Dataset belum tersedia. Buka menu **Input Dataset Baru** untuk mengunggah file.")
    st.stop()

# =========================================================
# FILTER GLOBAL
# =========================================================
if "angkatan" in df.columns:
    cohorts = sorted(df["angkatan"].dropna().unique().tolist())
else:
    cohorts = []
with st.sidebar:
    st.markdown("### Filter data")
    cohort_options = ["Semua Angkatan"] + cohorts
    selected_cohort = st.selectbox("Angkatan", cohort_options)
if selected_cohort == "Semua Angkatan" or "angkatan" not in df.columns:
    filtered = df.copy()
else:
    filtered = df[df["angkatan"] == selected_cohort].copy()

    # =========================================================
# MENU: DASHBOARD UTAMA
# =========================================================
if menu == "Dashboard Utama":
    st.header("Ringkasan Akademik")
    n_students = len(filtered)
    available_subjects = [c for c in MAPEL if c in filtered.columns]
    avg_score = filtered[available_subjects].mean().mean() if available_subjects else np.nan
    avg_attendance = filtered["kehadiran_persen"].mean() if "kehadiran_persen" in filtered else np.nan
    n_labeled = filtered[TARGET_DEFAULT].notna().sum() if TARGET_DEFAULT in filtered else 0

    a,b,c,d = st.columns(4)
    a.metric("Jumlah Siswa", f"{n_students:,}")
    b.metric("Rata-rata Nilai", "-" if pd.isna(avg_score) else f"{avg_score:.2f}")
    c.metric("Rata-rata Kehadiran", "-" if pd.isna(avg_attendance) else f"{avg_attendance:.1f}%")
    d.metric("Data dengan Label Risiko", f"{n_labeled:,}")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("Rata-rata nilai per mata pelajaran")
        if available_subjects:
            avg = filtered[available_subjects].mean().rename(index=MAPEL).rename_axis("Mata Pelajaran").reset_index(name="Rata-rata")
            fig = px.bar(avg, x="Mata Pelajaran", y="Rata-rata", text_auto=".1f",
                         color="Rata-rata", color_continuous_scale=["#b7e4c7", "#14532d"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom nilai mata pelajaran belum ditemukan pada dataset.")
    with col2:
        st.subheader("Komposisi angkatan")
        if "angkatan" in df.columns:
            count = df["angkatan"].value_counts().sort_index().rename_axis("Angkatan").reset_index(name="Jumlah Siswa")
            fig = px.pie(count, names="Angkatan", values="Jumlah Siswa", hole=.48,
                         color_discrete_sequence=["#14532d","#2f855a","#74c69d","#b7e4c7","#d8f3dc"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom angkatan belum tersedia.")
    st.subheader("Data siswa terbaru")
    show_table(filtered.head(12))
    st.caption("Prediksi merupakan alat bantu peninjauan. Keputusan pendampingan tetap dilakukan oleh guru berdasarkan kondisi siswa.")

    
# =========================================================
# MENU: DATA SISWA
# =========================================================
elif menu == "Data Siswa":
    st.header("Data Siswa")
    id_col = find_student_id(filtered)
    search = st.text_input("Cari nama / ID siswa", placeholder="Ketik nama atau nomor induk...")
    view = filtered.copy()
    if search:
        if id_col:
            view = view[view[id_col].astype(str).str.contains(search, case=False, na=False)]
        else:
            view = view[view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)]
    st.metric("Jumlah data siswa sesuai filter", len(view))
    show_table(view)
    st.download_button("Unduh data siswa (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                       "data_siswa.csv", "text/csv")

    # =========================================================
# MENU: DATA ANGKATAN
# =========================================================
elif menu == "Data Angkatan":
    st.header("Rekapitulasi Data Angkatan")
    if "angkatan" not in df.columns:
        st.error("Kolom 'angkatan' belum tersedia.")
    else:
        cohort_summary = df.groupby("angkatan", dropna=False).size().reset_index(name="Jumlah Siswa")
        score_cols = [c for c in MAPEL if c in df.columns]
        if score_cols:
            means = df.groupby("angkatan")[score_cols].mean().reset_index()
            cohort_summary = cohort_summary.merge(means, on="angkatan", how="left")
        st.dataframe(cohort_summary, use_container_width=True, hide_index=True)
        if not cohort_summary.empty:
            fig = px.bar(cohort_summary, x="angkatan", y="Jumlah Siswa", text_auto=True,
                         color="Jumlah Siswa", color_continuous_scale=["#b7e4c7","#14532d"])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        if "risiko_akademik" in df.columns:
            st.subheader("Distribusi label risiko per angkatan")
            labeled = df.dropna(subset=["risiko_akademik"])
            if not labeled.empty:
                risk_tab = labeled.groupby(["angkatan","risiko_akademik"]).size().reset_index(name="Jumlah")
                risk_tab["Kategori"] = risk_tab["risiko_akademik"].map({0:"Tidak berisiko",1:"Berisiko"})
                fig = px.bar(risk_tab, x="angkatan", y="Jumlah", color="Kategori", barmode="group",
                             color_discrete_map={"Tidak berisiko":"#74c69d","Berisiko":"#14532d"})
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MENU: NILAI SEMUA KOMPONEN
# =========================================================
elif menu == "Nilai Semua Komponen":
    st.header("Nilai Semua Komponen Akademik")
    score_cols = [c for c in MAPEL if c in filtered.columns]
    if not score_cols:
        st.warning("Kolom nilai belum ditemukan. Periksa nama kolom dataset.")
    else:
        labels = {c: MAPEL[c] for c in score_cols}
        selected_subjects = st.multiselect("Pilih mata pelajaran", score_cols,
                                           default=score_cols, format_func=lambda x: labels[x])
        if selected_subjects:
            st.subheader("Tabel nilai siswa")
            cols_to_show = [c for c in ["id_siswa","nama_siswa","nama","angkatan"] if c in filtered.columns] + selected_subjects
            show_table(filtered[cols_to_show])
            st.subheader("Statistik setiap komponen")
            stats = filtered[selected_subjects].describe().T.reset_index().rename(columns={"index":"Komponen"})
            stats["Komponen"] = stats["Komponen"].map(MAPEL).fillna(stats["Komponen"])
            show_table(stats)
            mean_df = filtered[selected_subjects].mean().rename(index=MAPEL).rename_axis("Mata Pelajaran").reset_index(name="Rata-rata")
            fig = px.bar(mean_df, x="Mata Pelajaran", y="Rata-rata", text_auto=".2f",
                         color="Rata-rata", color_continuous_scale=["#b7e4c7","#14532d"])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Unduh nilai komponen (CSV)", filtered[cols_to_show].to_csv(index=False).encode("utf-8-sig"),
                               "nilai_semua_komponen.csv", "text/csv")
    engagement_cols = [c for c in FITUR_ENGAGEMENT if c in filtered.columns]
    if engagement_cols:
        st.subheader("Komponen keterlibatan belajar")
        show_table(filtered[[c for c in ["id_siswa","nama_siswa","nama","angkatan"] if c in filtered.columns] + engagement_cols])

# =========================================================
# MENU: ANALISIS AKADEMIK
# =========================================================
elif menu == "Analisis Akademik":
    st.header("Analisis Akademik dan Keterlibatan Belajar")
    score_cols = [c for c in MAPEL if c in filtered.columns]
    engagement_cols = [c for c in FITUR_ENGAGEMENT if c in filtered.columns]
    if score_cols:
        st.subheader("Sebaran nilai")
        long_df = filtered[score_cols].melt(var_name="Mata Pelajaran", value_name="Nilai")
        long_df["Mata Pelajaran"] = long_df["Mata Pelajaran"].map(MAPEL)
        fig = px.box(long_df, x="Mata Pelajaran", y="Nilai", color="Mata Pelajaran",
                     color_discrete_sequence=["#14532d","#2f855a","#74c69d","#95d5b2"])
        st.plotly_chart(fig, use_container_width=True)
    if engagement_cols:
        st.subheader("Ringkasan keterlibatan belajar")
        eng = filtered[engagement_cols].mean().rename({
            "kehadiran_persen":"Kehadiran (%)",
            "ketuntasan_tugas":"Ketuntasan tugas",
            "keterlibatan_belajar":"Keterlibatan belajar"
        }).rename_axis("Indikator").reset_index(name="Rata-rata")
        st.dataframe(eng, use_container_width=True, hide_index=True)
        fig = px.bar(eng, x="Indikator", y="Rata-rata", text_auto=".2f",
                     color="Rata-rata", color_continuous_scale=["#b7e4c7","#14532d"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    if TARGET_DEFAULT in filtered.columns:
        st.subheader("Distribusi kebutuhan pendampingan (berdasarkan label dataset)")
        labels = filtered[TARGET_DEFAULT].dropna().map({0:"Tidak berisiko",1:"Berisiko"})
        if len(labels):
            counts = labels.value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
            fig = px.pie(counts, names="Kategori", values="Jumlah", hole=.45,
                         color="Kategori", color_discrete_map={"Tidak berisiko":"#74c69d","Berisiko":"#14532d"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Belum ada label risiko pada data terpilih.")

            # =========================================================
# MENU: PEMODELAN DAN PREDIKSI
# =========================================================
elif menu == "Pemodelan & Prediksi":
    st.header("Pemodelan Machine Learning dan Prediksi")
    st.write("Pilih algoritma, target, fitur, serta skema evaluasi. Hasil pengolahan akan ditampilkan pada halaman ini.")
    if TARGET_DEFAULT not in df.columns:
        st.error("Kolom target 'risiko_akademik' tidak ditemukan. Tambahkan label target pada dataset historis.")
        st.stop()
    target = st.selectbox("Variabel target", [TARGET_DEFAULT] + [c for c in df.columns if c != TARGET_DEFAULT])
    possible_features = [c for c in df.columns if c not in [target]]
    default_features = [c for c in FITUR_DEFAULT if c in possible_features]
    features = st.multiselect("Pilih variabel prediktor", possible_features, default=default_features)
    algorithm = st.selectbox("Algoritma klasifikasi", ["Random Forest", "Decision Tree", "Logistic Regression", "Gradient Boosting"])
    eval_method = st.radio("Metode evaluasi", ["Stratified random split", "Berdasarkan urutan angkatan"], horizontal=True)
    test_size = st.slider("Proporsi data uji", 0.10, 0.40, 0.20, 0.05)
    threshold = st.slider("Ambang probabilitas kelas positif / risiko", 0.10, 0.90, 0.50, 0.05)
    labeled = df.dropna(subset=[target]).copy()
    if st.button("Jalankan pelatihan dan evaluasi model", type="primary"):
        if len(features) == 0:
            st.error("Pilih minimal satu fitur prediktor.")
        elif labeled[target].nunique() < 2:
            st.error("Target harus memiliki minimal dua kelas.")
        else:
            X = labeled[features]
            y = labeled[target]
            try:
                if eval_method == "Berdasarkan urutan angkatan" and "angkatan" in labeled.columns:
                    cohort_values = sorted(labeled["angkatan"].dropna().unique())
                    if len(cohort_values) < 2:
                        raise ValueError("Diperlukan minimal dua angkatan berlabel untuk evaluasi temporal.")
                    test_cohort = cohort_values[-1]
                    train_idx = labeled["angkatan"] < test_cohort
                    test_idx = labeled["angkatan"] == test_cohort
                    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                if len(X_train) == 0 or len(X_test) == 0:
                    raise ValueError("Data latih atau data uji kosong.")
                if y_train.nunique() < 2:
                    raise ValueError("Data latih hanya memiliki satu kelas.")
                pipe = build_pipeline(X_train, algorithm)
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                classes = list(pipe.named_steps["model"].classes_)
                positive_class = 1 if 1 in classes else classes[-1]
                if hasattr(pipe, "predict_proba"):
                    pos_index = classes.index(positive_class)
                    prob = pipe.predict_proba(X_test)[:, pos_index]
                    pred_threshold = np.where(prob >= threshold, positive_class, classes[0])
                else:
                    prob = np.full(len(X_test), np.nan)
                    pred_threshold = pred
                # Threshold berlaku untuk klasifikasi biner dengan kelas positif 1.
                if positive_class == 1 and len(classes) == 2 and hasattr(pipe, "predict_proba"):
                    pred = pred_threshold
                metrics = {
                    "Akurasi": accuracy_score(y_test, pred),
                    "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1-score (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
                }
                result_df = X_test.copy()
                id_col = find_student_id(labeled)
                if id_col and id_col not in result_df.columns:
                    result_df.insert(0, id_col, labeled.loc[X_test.index, id_col])
                result_df["Aktual"] = y_test.values
                result_df["Prediksi"] = pred
                if not np.isnan(prob).all():
                    result_df["Probabilitas_kelas_positif"] = prob
                st.session_state.model_result = {
                    "pipeline": pipe, "features": features, "target": target,
                    "algorithm": algorithm, "metrics": metrics,
                    "results": result_df, "threshold": threshold,
                    "classes": classes
                }
            except Exception as e:
                st.error(f"Model belum dapat dijalankan: {e}")

    saved = st.session_state.get("model_result")
    if saved is not None:
        st.divider()
        st.subheader(f"Hasil Pengolahan: {saved['algorithm']}")
        st.caption(f"Target: {saved['target']} | Fitur: {', '.join(saved['features'])} | Ambang: {saved['threshold']:.2f}")
        m1,m2,m3,m4 = st.columns(4)
        vals = list(saved["metrics"].values())
        for col, (label, value) in zip([m1,m2,m3,m4], saved["metrics"].items()):
            col.metric(label, f"{value:.3f}")
        results = saved["results"]
        st.subheader("Perbandingan nilai aktual dan prediksi")
        show_table(results)
        st.download_button("Unduh hasil evaluasi model (CSV)", results.to_csv(index=False).encode("utf-8-sig"),
                           "hasil_evaluasi_model.csv", "text/csv")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(results["Aktual"], results["Prediksi"], labels=saved["classes"])
        cm_df = pd.DataFrame(cm, index=[f"Aktual {x}" for x in saved["classes"]],
                             columns=[f"Prediksi {x}" for x in saved["classes"]])
        st.dataframe(cm_df, use_container_width=True)
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=["#e8f5e9","#14532d"],
                        aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Laporan klasifikasi")
        report = classification_report(results["Aktual"], results["Prediksi"], zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
        # Feature importance untuk model berbasis pohon
        estimator = saved["pipeline"].named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            names = saved["pipeline"].named_steps["preprocess"].get_feature_names_out()
            importance = pd.DataFrame({"Fitur": names, "Importance": estimator.feature_importances_})
            importance = importance.sort_values("Importance", ascending=False).head(15)
            st.subheader("Feature Importance")
            fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Fitur",
                         orientation="h", color="Importance",
                         color_continuous_scale=["#b7e4c7","#14532d"])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Algoritma yang dipilih tidak menyediakan feature_importances_ bawaan.")
        st.info("Catatan: hasil prediksi adalah dukungan pengambilan keputusan. Feature importance bukan bukti sebab-akibat; guru perlu memeriksa konteks setiap siswa.")

 # Prediksi data baru/current cohort tanpa label target
        unlabeled = df[df[target].isna()] if target in df.columns else pd.DataFrame()
        if not unlabeled.empty:
            st.subheader("Prediksi untuk data yang belum memiliki label")
            missing_feature = [c for c in saved["features"] if c not in unlabeled.columns]
            if missing_feature:
                st.warning(f"Kolom fitur tidak ditemukan pada data tanpa label: {missing_feature}")
            else:
                pipe = saved["pipeline"]
                new_pred = pipe.predict(unlabeled[saved["features"]])
                prediction = unlabeled[[c for c in ["id_siswa","nama_siswa","nama","angkatan"] if c in unlabeled.columns]].copy()
                prediction["Prediksi"] = new_pred
                if hasattr(pipe, "predict_proba"):
                    classes = list(pipe.named_steps["model"].classes_)
                    if 1 in classes:
                        prediction["Probabilitas_risiko"] = pipe.predict_proba(unlabeled[saved["features"]])[:, classes.index(1)]
                show_table(prediction)
                st.download_button("Unduh prediksi data tanpa label", prediction.to_csv(index=False).encode("utf-8-sig"),
                                   "prediksi_data_baru.csv", "text/csv")

                # =========================================================
# MENU: INPUT DATASET BARU
# =========================================================
elif menu == "Input Dataset Baru":
    st.header("Input Dataset Baru")
    st.write("Unggah dataset akademik baru untuk mengganti atau menambahkan data ke dataset yang sedang digunakan.")
    uploaded = st.file_uploader("Pilih file Excel/CSV", type=["xlsx","xls","csv"], key="dataset_upload")
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith((".xlsx",".xls")):
                book = pd.ExcelFile(uploaded)
                sheet = st.selectbox("Pilih sheet Excel", book.sheet_names)
                uploaded.seek(0)
                incoming = pd.read_excel(uploaded, sheet_name=sheet)
            else:
                incoming = read_dataset(uploaded)
            incoming = standardize_data(incoming)
            st.subheader("Pratinjau dataset yang diunggah")
            st.write(f"Jumlah baris: **{len(incoming)}** | Jumlah kolom: **{len(incoming.columns)}**")
            show_table(incoming.head(20))
            mode = st.radio("Cara memasukkan data", ["Ganti dataset aktif", "Tambahkan ke dataset aktif"], horizontal=True)
            if st.button("Gunakan dataset ini", type="primary"):
                if mode == "Ganti dataset aktif" or st.session_state.dataset is None:
                    st.session_state.dataset = incoming
                else:
                    st.session_state.dataset = pd.concat([st.session_state.dataset, incoming], ignore_index=True)
                st.session_state.model_result = None
                st.success("Dataset aktif berhasil diperbarui untuk sesi dashboard ini.")
                st.rerun()
        except Exception as e:
            st.error(f"File tidak dapat dibaca: {e}")

    st.divider()
    st.subheader("Input satu data siswa secara manual")
    st.caption("Kolom akan disesuaikan dengan kolom dataset aktif. Data manual ditambahkan ke sesi saat ini, belum disimpan permanen ke file sumber.")
    with st.form("manual_student_form"):
        columns = st.columns(3)
        manual = {}
        known_fields = [c for c in ["id_siswa","nama_siswa","nama","angkatan"] + FITUR_DEFAULT + [TARGET_DEFAULT] if c in df.columns]
        for i, colname in enumerate(known_fields):
            with columns[i % 3]:
                if colname in FITUR_DEFAULT or colname in ["angkatan", TARGET_DEFAULT]:
                    val = st.number_input(colname.replace("_"," ").title(), value=0.0, key=f"manual_{colname}")
                else:
                    val = st.text_input(colname.replace("_"," ").title(), key=f"manual_{colname}")
                manual[colname] = val
        submitted = st.form_submit_button("Tambahkan data siswa")
        if submitted:
            new_row = pd.DataFrame([manual])
            for c in new_row.columns:
                if c in FITUR_DEFAULT + ["angkatan", TARGET_DEFAULT]:
                    new_row[c] = pd.to_numeric(new_row[c], errors="coerce")
            st.session_state.dataset = pd.concat([df, new_row], ignore_index=True)
            st.session_state.model_result = None
            st.success("Data siswa ditambahkan ke dataset aktif pada sesi ini.")
            st.rerun()

            # =========================================================
# MENU: INFORMASI DATASET
# =========================================================
elif menu == "Informasi Dataset":
    st.header("Informasi dan Kualitas Dataset")
    st.metric("Jumlah baris", len(df))
    st.metric("Jumlah kolom", len(df.columns))
    st.subheader("Daftar kolom dan tipe data")
    schema = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": [str(x) for x in df.dtypes],
        "Jumlah Data Terisi": [df[c].notna().sum() for c in df.columns],
        "Jumlah Kosong": [df[c].isna().sum() for c in df.columns],
        "Jumlah Unik": [df[c].nunique(dropna=True) for c in df.columns]
    })
    show_table(schema)
    st.subheader("Pemeriksaan data kosong")
    missing = df.isna().sum().rename_axis("Kolom").reset_index(name="Jumlah Kosong")
    missing = missing[missing["Jumlah Kosong"] > 0].sort_values("Jumlah Kosong", ascending=False)
    if missing.empty:
        st.success("Tidak ditemukan nilai kosong.")
    else:
        show_table(missing)
    st.subheader("Pratinjau seluruh dataset")
    show_table(df)

st.divider()
st.markdown('<p class="small-note">Dashboard Akademik SD Ainul Hayat | Model machine learning digunakan sebagai alat bantu identifikasi dini, bukan penentu tunggal keputusan pendampingan siswa.</p>', unsafe_allow_html=True)
