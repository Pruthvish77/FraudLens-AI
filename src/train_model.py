"""
Model Training and Serialization Pipeline for FinGuard AI.
Trains an enterprise-grade fraud classification model with class imbalance handling,
evaluates using precision-recall metrics (PR-AUC, ROC-AUC), and saves artifacts.
"""

import os
import json
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from data_loader import load_dataset, preprocess_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_fraud_model(dataset_path: str = None, use_xgboost: bool = True):
    """
    Executes end-to-end model training, evaluation, and serialization.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print(" [FraudLens AI] INITIATING TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    df = load_dataset(dataset_path)
    total_records = len(df)
    fraud_count = int(df["Class"].sum())
    fraud_pct = (fraud_count / total_records) * 100

    print(f"Total Transactions : {total_records:,}")
    print(f"Fraudulent Cases   : {fraud_count:,} ({fraud_pct:.3f}%)")

    # 2. Preprocess & Stratified Split
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_data(df)

    # Calculate class weight ratio for imbalance compensation
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"\nTraining Set: {len(X_train)} samples | Positive Class Weight: {scale_pos_weight:.2f}")

    # 3. Model Architecture Selection
    if use_xgboost and HAS_XGBOOST:
        print("[FraudLens AI] Training XGBoost Classifier...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="aucpr",
            random_state=42
        )
    else:
        print("[FraudLens AI] Training Balanced Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print(f"PR-AUC (Avg Prec): {pr_auc:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(cm))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # 5. Serialization
    model_path = os.path.join(MODELS_DIR, "fraud_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    metadata_path = os.path.join(MODELS_DIR, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metrics_payload = {
        "model_type": "XGBoost" if (use_xgboost and HAS_XGBOOST) else "RandomForest",
        "total_records": total_records,
        "fraud_cases": fraud_count,
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "feature_cols": feature_cols
    }

    with open(metadata_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print("\n[FraudLens AI] Artifacts persisted successfully:")
    print(f" -> Model   : {model_path}")
    print(f" -> Scaler  : {scaler_path}")
    print(f" -> Metrics : {metadata_path}")
    print("=" * 60)

    return model, scaler, metrics_payload


if __name__ == "__main__":
    train_fraud_model()
