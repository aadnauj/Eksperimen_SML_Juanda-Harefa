import os
import sys
import argparse
import importlib.util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from mlflow.models.signature import infer_signature

# ============================================================
# ARGPARSE — Hyperparameter dari CLI / GitHub Actions
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",  type=int,   default=100)
parser.add_argument("--max_depth",     type=int,   default=10)
parser.add_argument("--random_state",  type=int,   default=42)
parser.add_argument("--test_size",     type=float, default=0.2)
args = parser.parse_args()

# ============================================================
# SETUP DAGSHUB + MLFLOW
# ============================================================
os.environ["MLFLOW_TRACKING_USERNAME"] = "aadnauj"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c90af6e65e78f47beb8645616ea78a165e6ff3d2"

dagshub.init(
    repo_owner="aadnauj",
    repo_name="Eksperimen_SML_Juanda-Harefa",
    mlflow=True
)

MLFLOW_TRACKING_URI = "https://dagshub.com/aadnauj/Eksperimen_SML_Juanda-Harefa.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn_Workflow_CI")

print("=" * 55)
print("  WORKFLOW CI - MLProject Modelling")
print(f"  n_estimators : {args.n_estimators}")
print(f"  max_depth    : {args.max_depth}")
print(f"  random_state : {args.random_state}")
print(f"  test_size    : {args.test_size}")
print("=" * 55)

# ============================================================
# LOAD PREPROCESSING - Gunakan automate_Juanda-Harefa.py
# ============================================================
# Cari file automate preprocessing
possible_paths = [
    os.path.join(os.path.dirname(__file__), "automate_Juanda-Harefa.py"),
    os.path.join(os.path.dirname(__file__), "..", "Eksperimen_SML_Juanda Harefa", "preprocessing", "automate_Juanda-Harefa.py"),
    os.path.join(os.getcwd(), "Eksperimen_SML_Juanda Harefa", "preprocessing", "automate_Juanda-Harefa.py"),
]

automate_path = None
for path in possible_paths:
    if os.path.exists(path):
        automate_path = path
        break

if automate_path:
    spec = importlib.util.spec_from_file_location("automate_preprocessing", automate_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_preprocessing = mod.run_preprocessing
    FILE_ID = "19IfOP0QmCHccMu8A6B2fCUpFqZwCxuzO"
    X_train, X_test, y_train, y_test = run_preprocessing(
        file_id=FILE_ID,
        target_col='Exited',
        test_size=args.test_size,
        random_state=args.random_state
    )
else:
    # Fallback: load dari CSV lokal jika ada
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
        from sklearn.model_selection import train_test_split
        data = pd.read_csv(csv_files[0])
        data = data.drop(columns=[c for c in ['RowNumber','CustomerId','Surname'] if c in data.columns])
        for col in ['Geography','Gender']:
            if col in data.columns:
                data[col] = LabelEncoder().fit_transform(data[col])
        X = data.drop(columns=['Exited'])
        y = data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        print("✅ Data dimuat dari CSV lokal")
    else:
        raise FileNotFoundError("Tidak ada data ditemukan!")

# ============================================================
# TRAINING + MANUAL LOGGING MLFLOW
# ============================================================
os.makedirs("tmp_artifacts", exist_ok=True)

with mlflow.start_run(run_name=f"RF_n{args.n_estimators}_d{args.max_depth}"):

    # Training
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrik
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc   = auc(fpr, tpr)

    # Log parameter
    mlflow.log_param("n_estimators",  args.n_estimators)
    mlflow.log_param("max_depth",     args.max_depth)
    mlflow.log_param("random_state",  args.random_state)
    mlflow.log_param("test_size",     args.test_size)

    # Log metrik
    mlflow.log_metric("accuracy",        acc)
    mlflow.log_metric("precision",       precision)
    mlflow.log_metric("recall",          recall)
    mlflow.log_metric("f1_score",        f1)
    mlflow.log_metric("roc_auc",         roc_auc)
    mlflow.log_metric("true_positive",   int(tp))
    mlflow.log_metric("false_positive",  int(fp))
    mlflow.log_metric("false_negative",  int(fn))
    mlflow.log_metric("true_negative",   int(tn))
    mlflow.log_metric("specificity",     tn / (tn + fp) if (tn + fp) > 0 else 0)

    # Log tag
    mlflow.set_tag("model",   "RandomForestClassifier")
    mlflow.set_tag("source",  "Workflow-CI")
    mlflow.set_tag("dataset", "Bank_Customer_Churn")

    # Artefak 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title('Confusion Matrix - RandomForest CI')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    cm_path = "tmp_artifacts/confusion_matrix.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # Artefak 2: ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title('ROC Curve - RandomForest CI')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    roc_path = "tmp_artifacts/roc_curve.png"
    plt.savefig(roc_path, dpi=100)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # Artefak 3: Classification Report
    report = classification_report(y_test, y_pred,
                                   target_names=['Not Churn', 'Churn'],
                                   output_dict=True)
    report_path = "tmp_artifacts/classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path, artifact_path="reports")

    # Artefak 4: Feature Importance
    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    fi.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Feature Importance - RandomForest CI')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    fi_path = "tmp_artifacts/feature_importance.png"
    plt.savefig(fi_path, dpi=100)
    plt.close()
    mlflow.log_artifact(fi_path, artifact_path="plots")

    # Log Model
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="RandomForest_WorkflowCI"
    )

    print(f"\n✅ Training selesai!")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"   ROC-AUC   : {roc_auc:.4f}")
    print(f"\n   Run ID: {mlflow.active_run().info.run_id}")

# Cleanup
import shutil
shutil.rmtree("tmp_artifacts", ignore_errors=True)

print("\n✅ Workflow CI selesai! Cek DagsHub → Experiments → Churn_Workflow_CI")
