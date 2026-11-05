# Project Machine Learning — Prediksi Kelulusan Mahasiswa

## Tujuan
Project ini merupakan supervised learning dengan binary classification untuk memprediksi apakah mahasiswa berpeluang lulus tepat waktu.

## Dataset
- 1.500 data simulasi untuk pembelajaran.
- Target: `graduated_on_time` (0 = tidak tepat waktu, 1 = tepat waktu).
- Dataset bukan data mahasiswa nyata.

## Struktur Program Final
```text
src/
├── main.py
├── preprocessing.py
├── app.py
└── prediksi_kelulusan_mahasiswa.ipynb
```

### Peran file
- `main.py`: workflow training, preprocessing, tuning, evaluasi, dan penyimpanan model.
- `preprocessing.py`: class `IQRClipper` untuk penanganan outlier.
- `app.py`: aplikasi Streamlit untuk demo prediksi.
- `prediksi_kelulusan_mahasiswa.ipynb`: notebook pembelajaran.

## Kenapa `preprocessing.py` dipisah?
`IQRClipper` adalah custom transformer yang ikut tersimpan di model `joblib`. Class ini dipisah dari `main.py` supaya bisa di-import kembali ketika Streamlit membuka model. Ini mencegah error referensi `__main__.IQRClipper`.

## Cara menjalankan
1. Buat dan aktifkan virtual environment.
2. Install dependency:
```powershell
python -m pip install -r requirements.txt
```
3. Training:
```powershell
python src/main.py
```
4. Jalankan aplikasi:
```powershell
python -m streamlit run src/app.py
```

## Model
Algoritma yang dibandingkan:
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

Tuning menggunakan GridSearchCV dan Stratified K-Fold 5 fold. Pemilihan model berdasarkan F1-Score dan ROC-AUC pada data uji.

## Hasil final eksperimen
Model terbaik: **Random Forest**
- Accuracy: **0.7233**
- Precision: **0.6124**
- Recall: **0.7054**
- F1-Score: **0.6556**
- ROC-AUC: **0.7670**
- Best Params: `{'model__max_depth': 5, 'model__min_samples_leaf': 10, 'model__n_estimators': 150}`

## Catatan
Model dan dataset dibuat untuk demonstrasi akademik. Hasil tidak dimaksudkan sebagai dasar keputusan akademik nyata.
