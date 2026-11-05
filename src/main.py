"""Project Machine Learning: Prediksi Kelulusan Mahasiswa Tepat Waktu."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import IQRClipper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset_kelulusan_mahasiswa.csv"
MODEL_DIR = PROJECT_ROOT / "models"
DOC_DIR = PROJECT_ROOT / "dokumentasi"
MODEL_DIR.mkdir(exist_ok=True)
DOC_DIR.mkdir(exist_ok=True)

TARGET = "graduated_on_time"
NUMERIC = [
    "age", "attendance_pct", "assignment_avg", "midterm_score", "final_score",
    "study_hours_week", "previous_gpa", "late_submissions", "organization_active",
    "academic_score", "engagement_score",
]
CATEGORICAL = ["program_study", "learning_method"]


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def add_features(df):
    df = df.copy()
    df["academic_score"] = (
        df["assignment_avg"] * 0.25
        + df["midterm_score"] * 0.30
        + df["final_score"] * 0.45
    )
    df["engagement_score"] = (
        df["attendance_pct"] * 0.70
        + df["study_hours_week"] * 1.50
    )
    return df


def prepare_features(df):
    X = df.drop(columns=["student_id", TARGET]).copy()
    y = df[TARGET].astype(int).copy()
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL:
        X[col] = X[col].astype(object)
        X[col] = X[col].where(X[col].notna(), None)
    return X, y


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_clip", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
    }, pred, proba


def main():
    print("=" * 70)
    print("PROJECT MACHINE LEARNING - PREDIKSI KELULUSAN MAHASISWA")
    print("=" * 70)

    df = add_features(load_dataset())
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, random_state=42, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=5,
            random_state=42, class_weight="balanced"
        ),
    }

    baseline = []
    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        metrics, _, _ = evaluate(pipe, X_test, y_test)
        baseline.append({"Model": name, **metrics})
        print(f"\n{name} baseline:")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.4f}")

    pd.DataFrame(baseline).to_csv(DOC_DIR / "hasil_baseline.csv", index=False)

    grids = {
        "Decision Tree": {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 5, 7],
            "model__min_samples_leaf": [5, 10, 20],
        },
        "KNN": {
            "model__n_neighbors": [5, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [5, 8, None],
            "model__min_samples_leaf": [2, 5, 10],
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_rows = []
    best_models = {}

    for name, model in models.items():
        search = GridSearchCV(
            make_pipeline(model),
            grids[name], scoring="f1", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics, pred, proba = evaluate(search.best_estimator_, X_test, y_test)
        tuned_rows.append({
            "Model": name,
            "Best_Params": str(search.best_params_),
            **metrics,
        })
        best_models[name] = search.best_estimator_
        print(f"\n{name} tuned:")
        print("  Best params:", search.best_params_)
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.4f}")

    tuned = pd.DataFrame(tuned_rows).sort_values(
        ["F1", "ROC_AUC"], ascending=False
    ).reset_index(drop=True)
    tuned.to_csv(DOC_DIR / "hasil_tuning.csv", index=False)

    best_name = tuned.loc[0, "Model"]
    best_model = best_models[best_name]
    best_metrics, best_pred, best_proba = evaluate(best_model, X_test, y_test)
    report = classification_report(y_test, best_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, best_pred)

    print("\n" + "=" * 70)
    print("MODEL TERBAIK")
    print("=" * 70)
    print("Model:", best_name)
    print(report)
    print("Confusion Matrix:\n", cm)
    print("Metrik final:")
    for k, v in best_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(DOC_DIR / "confusion_matrix.csv")
    (DOC_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    (DOC_DIR / "best_model_info.txt").write_text(
        f"Model terbaik: {best_name}\n\n{tuned.loc[0].to_string()}",
        encoding="utf-8",
    )

    model_path = MODEL_DIR / "model_terbaik.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel tersimpan di: {model_path}")


if __name__ == "__main__":
    main()
