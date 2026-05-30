import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from mlflow.models.signature import infer_signature

# Tambahkan path automate preprocessing (path absolut)
PREPROCESSING_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Eksperimen_SML_Juanda Harefa', 'preprocessing')
)
sys.path.insert(0, PREPROCESSING_PATH)

# Import dengan nama file yang benar
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "automate_preprocessing",
    os.path.join(PREPROCESSING_PATH, "automate_Juanda-Harefa.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_preprocessing = _mod.run_preprocessing

# ============================================================
# SETUP DAGSHUB + MLFLOW
# ============================================================
dagshub.init(
    repo_owner="aadnauj",
    repo_name="Eksperimen_SML_Juanda-Harefa",
    mlflow=True
)

MLFLOW_TRACKING_URI = "https://dagshub.com/aadnauj/Eksperimen_SML_Juanda-Harefa.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "aadnauj"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c90af6e65e78f47beb8645616ea78a165e6ff3d2"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn_Modelling_Advanced_Tuning")

print("=" * 60)
print("  MODELLING_TUNING.PY - Advanced Level")
print("  Manual Logging + Hyperparameter Tuning + Extra Artifacts")
print("  MLflow → DagsHub")
print("=" * 60)

# ============================================================
# LOAD & PREPROCESSING DATA
# ============================================================
FILE_ID = "19IfOP0QmCHccMu8A6B2fCUpFqZwCxuzO"
X_train, X_test, y_train, y_test = run_preprocessing(
    file_id=FILE_ID,
    target_col='Exited',
    test_size=0.2,
    random_state=42
)

# ============================================================
# HYPERPARAMETER GRID
# ============================================================
param_grids = {
    "KNN_Tuned": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "DecisionTree_Tuned": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "RandomForest_Tuned": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
    },
    "SVM_Tuned": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    },
    "NaiveBayes_Tuned": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
}

# ============================================================
# FUNGSI: BUAT ARTEFAK CONFUSION MATRIX
# ============================================================
def save_confusion_matrix_artifact(cm, model_name, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return save_path

# ============================================================
# FUNGSI: BUAT ARTEFAK ROC CURVE
# ============================================================
def save_roc_curve_artifact(model, X_test, y_test, model_name, save_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return roc_auc, save_path

# ============================================================
# FUNGSI: BUAT ARTEFAK CLASSIFICATION REPORT (JSON)
# ============================================================
def save_classification_report_artifact(y_test, y_pred, model_name, save_path):
    report = classification_report(y_test, y_pred,
                                   target_names=['Not Churn', 'Churn'],
                                   output_dict=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    return save_path

# ============================================================
# TRAINING + MANUAL LOGGING MLflow
# ============================================================
os.makedirs("tmp_artifacts", exist_ok=True)
all_results = {}

for run_name, config in param_grids.items():

    print(f"\n🔍 Tuning {run_name}...")

    # GridSearchCV untuk hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Prediksi
    y_pred = best_model.predict(X_test)

    # Hitung semua metrik
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    y_prob    = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc   = auc(fpr, tpr)

    # Simpan artefak lokal sementara
    cm_path      = f"tmp_artifacts/{run_name}_confusion_matrix.png"
    roc_path     = f"tmp_artifacts/{run_name}_roc_curve.png"
    report_path  = f"tmp_artifacts/{run_name}_classification_report.json"
    feature_path = f"tmp_artifacts/{run_name}_feature_importance.png"

    save_confusion_matrix_artifact(cm, run_name, cm_path)
    save_roc_curve_artifact(best_model, X_test, y_test, run_name, roc_path)
    save_classification_report_artifact(y_test, y_pred, run_name, report_path)

    # Feature importance (hanya untuk tree-based)
    has_feature_importance = False
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_,
                      index=X_train.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        fi.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Feature Importance - {run_name}')
        ax.set_ylabel('Importance')
        plt.tight_layout()
        plt.savefig(feature_path, dpi=100, bbox_inches='tight')
        plt.close()
        has_feature_importance = True

    # ---- MANUAL LOGGING ke MLflow ----
    with mlflow.start_run(run_name=run_name):

        # Log hyperparameter terbaik
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")

        # Log metrik standar (sama dengan autolog)
        mlflow.log_metric("accuracy",         acc)
        mlflow.log_metric("precision",        precision)
        mlflow.log_metric("recall",           recall)
        mlflow.log_metric("f1_score",         f1)
        mlflow.log_metric("true_positive",    int(tp))
        mlflow.log_metric("false_positive",   int(fp))
        mlflow.log_metric("false_negative",   int(fn))
        mlflow.log_metric("true_negative",    int(tn))

        # Log metrik TAMBAHAN (tidak tercakup autolog)
        mlflow.log_metric("roc_auc",          roc_auc)
        mlflow.log_metric("best_cv_f1_score", best_cv_score)
        mlflow.log_metric("specificity",      tn / (tn + fp) if (tn + fp) > 0 else 0)
        mlflow.log_metric("balanced_accuracy",(recall + (tn/(tn+fp))) / 2)

        # Log tag
        mlflow.set_tag("model_type",   type(best_model).__name__)
        mlflow.set_tag("level",        "advanced")
        mlflow.set_tag("tuning",       "GridSearchCV")
        mlflow.set_tag("dataset",      "Bank_Customer_Churn")

        # Log model dengan signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=run_name
        )

        # ---- LOG ARTEFAK TAMBAHAN (minimal 2) ----
        # Artefak 1: Confusion Matrix
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Artefak 2: ROC Curve
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Artefak 3: Classification Report JSON
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Artefak 4: Feature Importance (jika ada)
        if has_feature_importance:
            mlflow.log_artifact(feature_path, artifact_path="plots")

        # Simpan hasil
        all_results[run_name] = {
            "accuracy":  acc,
            "precision": precision,
            "recall":    recall,
            "f1_score":  f1,
            "roc_auc":   roc_auc,
            "best_params": best_params
        }

        print(f"  ✅ Acc={acc:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f} | Best params: {best_params}")

# ============================================================
# RINGKASAN HASIL
# ============================================================
print("\n" + "=" * 60)
print("  RINGKASAN HASIL SEMUA MODEL")
print("=" * 60)

results_df = pd.DataFrame(all_results).T
results_df = results_df[["accuracy","precision","recall","f1_score","roc_auc"]]
results_df = results_df.sort_values("f1_score", ascending=False)
print(results_df.round(4).to_string())

print()
best_model_name = results_df.index[0]
print(f"🏆 Model terbaik: {best_model_name}")
print(f"   F1-Score : {results_df.loc[best_model_name, 'f1_score']:.4f}")
print(f"   ROC-AUC  : {results_df.loc[best_model_name, 'roc_auc']:.4f}")

print()
print("✅ modelling_tuning.py selesai! Cek DagsHub → Experiments")
print(f"   {MLFLOW_TRACKING_URI}")

# Hapus artefak temp
import shutil
shutil.rmtree("tmp_artifacts", ignore_errors=True)
