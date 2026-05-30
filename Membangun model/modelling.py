import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

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
mlflow.set_experiment("Churn_Modelling_Basic")

print("=" * 55)
print("  MODELLING.PY - Basic Level")
print("  MLflow → DagsHub")
print("=" * 55)

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
# DEFINISI MODEL
# ============================================================
models = {
    "KNN_Basic":           KNeighborsClassifier(),
    "DecisionTree_Basic":  DecisionTreeClassifier(random_state=42),
    "RandomForest_Basic":  RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM_Basic":           SVC(probability=True, random_state=42),
    "NaiveBayes_Basic":    GaussianNB()
}

# ============================================================
# TRAINING + AUTOLOG MLflow
# ============================================================
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

for run_name, model in models.items():
    with mlflow.start_run(run_name=run_name):

        # Training
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Metrik
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        # Log metrik manual
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)

        # Log tag
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("level",      "basic")

        print(f"[{run_name}] Acc={acc:.4f} | Prec={precision:.4f} | Rec={recall:.4f} | F1={f1:.4f}")

print()
print("✅ modelling.py selesai! Cek DagsHub → Experiments")
print(f"   {MLFLOW_TRACKING_URI}")
