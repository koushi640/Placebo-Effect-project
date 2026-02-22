"""
Quick Train Script
Run this from project root to train all models using data from the data folder.
Saves trained models and summary in the models folder.

Models (as per Minor Abstract Placebo):
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)

Usage (from project root):
    python models/quicktrain.py
"""

import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Paths relative to project root
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
X_PATH = os.path.join(DATA_DIR, "X_scaled.csv")
Y_PATH = os.path.join(DATA_DIR, "y_target.csv")


def quicktrain(test_size=0.2, random_state=42, output_prefix=""):
    """Load data, train all models, save to models folder."""
    print("=" * 60)
    print("QUICK TRAIN - Placebo Response Models")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {X_PATH}, {Y_PATH}")
    prefix_info = f"{output_prefix}" if output_prefix else ""
    print(f"Output: {MODELS_DIR}/ ({prefix_info})")
    print()

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("ERROR: Preprocessed data not found.")
        print("Run the full pipeline first: python main.py")
        print("Or run: python -c \"from src.preprocessing import load_and_prepare_dataset, preprocess_data; load_and_prepare_dataset(); preprocess_data()\"")
        return None

    # Load data
    print("1. Loading data...")
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).values.ravel()
    print(f"   X: {X.shape}, y: {y.shape}")

    # Split
    print("\n2. Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=random_state, kernel="rbf"),
    }

    results = []
    print("\n3. Training models...")
    print("-" * 60)

    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        pr = precision_score(y_test, y_pred, zero_division=0)
        re = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(pr, 4),
                "Recall": round(re, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(auc, 4),
            }
        )
        print(f"      ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        # Save each model
        safe_name = name.lower().replace(" ", "_")
        pkl_path = os.path.join(MODELS_DIR, f"{output_prefix}model_{safe_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"      Saved: {pkl_path}")

    # Summary DataFrame
    results_df = pd.DataFrame(results)
    best_idx = results_df["ROC_AUC"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = models[best_name]

    # Save best model
    best_path = os.path.join(MODELS_DIR, f"{output_prefix}best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n4. Best model: {best_name} -> {best_path}")

    # Save summary CSV in models folder
    summary_path = os.path.join(MODELS_DIR, f"{output_prefix}training_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"   Summary: {summary_path}")

    # Save a quick text summary in models folder
    txt_path = os.path.join(MODELS_DIR, f"{output_prefix}training_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Quick Train - Placebo Response Models\n")
        f.write("=" * 50 + "\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data: {X_PATH}, {Y_PATH}\n")
        f.write(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write(f"Best model: {best_name}\n")
    print(f"   Summary (txt): {txt_path}")

    # Also save to reports for compatibility with main pipeline
    os.makedirs(REPORTS_DIR, exist_ok=True)
    reports_csv = os.path.join(REPORTS_DIR, f"{output_prefix}model_results_summary.csv")
    results_df.to_csv(reports_csv, index=False)

    print("\n" + "=" * 60)
    print("QUICK TRAIN COMPLETE")
    print("=" * 60)
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick train placebo response models")
    parser.add_argument("--advanced", action="store_true", help="Use CV + hyperparameter tuning + SMOTE")
    args = parser.parse_args()
    if args.advanced:
        from src.train_model import train_models_advanced
        train_models_advanced(X_PATH, Y_PATH, use_smote=True)
    else:
        quicktrain()
