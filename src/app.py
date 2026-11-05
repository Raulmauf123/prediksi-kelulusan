"""Streamlit app untuk demo prediksi kelulusan mahasiswa."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import IQRClipper  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model_terbaik.joblib"

st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Prediksi Kelulusan Mahasiswa Tepat Waktu")
st.caption("Demo akademik project Machine Learning kelompok.")

if model is None:
    st.error("Model belum ditemukan. Jalankan `python src/main.py` terlebih dahulu.")
    st.stop()

st.subheader("Masukkan Data Mahasiswa")

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Usia", 18, 60, 21)
    program = st.selectbox("Program Studi", ["Informatika", "Sistem Informasi", "Teknik Industri"])
    method = st.selectbox("Metode Belajar", ["Tatap Muka", "Hybrid", "Online"])
    attendance = st.slider("Kehadiran (%)", 0, 100, 85)
    assignment = st.slider("Rata-rata Nilai Tugas", 0, 100, 75)

with c2:
    midterm = st.slider("Nilai UTS", 0, 100, 70)
    final = st.slider("Nilai UAS", 0, 100, 72)
    study = st.number_input("Jam Belajar per Minggu", 0.0, 50.0, 12.0, step=0.5)
    gpa = st.number_input("IPK Sebelumnya", 0.0, 4.0, 3.0, step=0.01)
    late = st.number_input("Jumlah Keterlambatan Pengumpulan", 0, 20, 2)

org = st.selectbox("Aktif Organisasi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

if st.button("🔮 Prediksi Kelulusan", type="primary", use_container_width=True):
    academic_score = assignment * 0.25 + midterm * 0.30 + final * 0.45
    engagement_score = attendance * 0.70 + study * 1.50

    input_data = pd.DataFrame([{
        "age": age,
        "program_study": program,
        "learning_method": method,
        "attendance_pct": attendance,
        "assignment_avg": assignment,
        "midterm_score": midterm,
        "final_score": final,
        "study_hours_week": study,
        "previous_gpa": gpa,
        "late_submissions": late,
        "organization_active": org,
        "academic_score": academic_score,
        "engagement_score": engagement_score,
    }])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])

    st.divider()
    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.success(f"✅ Berpeluang lulus tepat waktu — {probability:.1%}")
    else:
        st.warning(f"⚠️ Berisiko tidak lulus tepat waktu — {probability:.1%}")

    st.metric("Probabilitas lulus tepat waktu", f"{probability:.1%}")
    st.info("Hasil ini hanya untuk demonstrasi akademik dan bukan keputusan akademik nyata.")
