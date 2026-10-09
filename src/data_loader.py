"""
Data Loader and Synthetic Data Generator for FinGuard AI.
Handles loading real Kaggle Credit Card Fraud datasets or generates
statistically authentic synthetic transaction datasets for rapid experimentation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DEFAULT_RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")


def generate_synthetic_data(n_samples: int = 6000, fraud_ratio: float = 0.015, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic credit card transaction data matching the standard PCA structure:
    - Time: Seconds elapsed
    - V1 to V28: PCA-transformed feature representations
    - Amount: Transaction amount in USD
    - Class: 0 for legitimate, 1 for fraudulent
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Normal transactions: standard Gaussian PCA distributions
    legit_v = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    legit_time = np.sort(np.random.uniform(0, 172800, size=n_legit)) # 48 hours in seconds
    legit_amount = np.random.exponential(scale=75.0, size=n_legit) + 1.0

    # Fraudulent transactions: shifted distributions on key discriminatory features (e.g., V4, V10, V11, V12, V14, V17)
    fraud_v = np.random.normal(loc=0.0, scale=1.5, size=(n_fraud, 28))
    fraud_v[:, 3] += np.random.normal(4.5, 1.0, size=n_fraud)    # V4 elevated
    fraud_v[:, 9] -= np.random.normal(5.0, 1.0, size=n_fraud)    # V10 depressed
    fraud_v[:, 10] += np.random.normal(4.0, 1.0, size=n_fraud)   # V11 elevated
    fraud_v[:, 11] -= np.random.normal(6.0, 1.2, size=n_fraud)   # V12 depressed
    fraud_v[:, 13] -= np.random.normal(7.0, 1.2, size=n_fraud)   # V14 strongly depressed
    fraud_v[:, 16] -= np.random.normal(4.0, 1.0, size=n_fraud)   # V17 depressed

    fraud_time = np.random.uniform(0, 172800, size=n_fraud)
    # Fraud amounts often exhibit high outliers or micro-testing amounts
    fraud_amount = np.random.choice([
        np.random.uniform(1.0, 5.0),       # Micro-card testing
        np.random.uniform(500.0, 2500.0)   # High-ticket draining
    ], size=n_fraud)

    # Combine into DataFrame
    legit_df = pd.DataFrame(legit_v, columns=[f"V{i}" for i in range(1, 29)])
    legit_df["Time"] = legit_time
    legit_df["Amount"] = np.round(legit_amount, 2)
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_v, columns=[f"V{i}" for i in range(1, 29)])
    fraud_df["Time"] = fraud_time
    fraud_df["Amount"] = np.round(fraud_amount, 2)
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Save to data directory for quick reuse
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df


def load_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Loads dataset from provided path, default Kaggle file, or generates synthetic data.
    """
    if filepath and os.path.exists(filepath):
        print(f"[FraudLens AI] Loading dataset from: {filepath}")
        return pd.read_csv(filepath)
    
    if os.path.exists(DEFAULT_RAW_PATH):
        print(f"[FraudLens AI] Loading raw dataset from: {DEFAULT_RAW_PATH}")
        return pd.read_csv(DEFAULT_RAW_PATH)
    
    if os.path.exists(SAMPLE_DATA_PATH):
        print(f"[FraudLens AI] Loading cached sample dataset from: {SAMPLE_DATA_PATH}")
        return pd.read_csv(SAMPLE_DATA_PATH)

    print("[FraudLens AI] Raw dataset not found. Generating high-fidelity synthetic benchmark dataset...")
    return generate_synthetic_data()


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Normalizes 'Time' and 'Amount' using RobustScaler (resilient to outliers),
    and performs a stratified split to preserve fraud ratio.
    """
    data = df.copy()

    # Robust scaling for Time and Amount
    scaler = RobustScaler()
    data["scaled_amount"] = scaler.fit_transform(data["Amount"].values.reshape(-1, 1))
    data["scaled_time"] = scaler.fit_transform(data["Time"].values.reshape(-1, 1))

    # Features: V1..V28 + scaled_amount + scaled_time
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    X = data[feature_cols]
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler, feature_cols
