"""
Model Training Module
Trains multiple ML models and saves the best one.
Includes: 5-fold CV, hyperparameter tuning, class imbalance (class_weight, SMOTE).

Models as per project abstract (Minor Abstract Placebo):
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
)
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
    make_scorer,
)
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------------------------------------------------------------------------
# Default data paths
# ---------------------------------------------------------------------------
DEFAULT_X_PATH = "data/X_scaled.csv"
DEFAULT_Y_PATH = "data/y_target.csv"
CV_FOLDS = 5
RANDOM_STATE = 42


def _get_base_models(with_class_weight=False):
    """Return base models. If with_class_weight, use class_weight='balanced'."""
    cw = "balanced" if with_class_weight else None
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, class_weight=cw
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight=cw
        ),
        "SVM": SVC(
            probability=True, random_state=RANDOM_STATE, kernel="rbf", class_weight=cw
        ),
    }


def run_cross_validation(X, y, models=None, cv=CV_FOLDS):
    """
    5-fold stratified CV; report mean ± std for accuracy and ROC-AUC.
    Returns dict of model_name -> {accuracy_mean, accuracy_std, roc_auc_mean, roc_auc_std}.
    """
    if models is None:
        models = _get_base_models(with_class_weight=False)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        results[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
        }
    return results


def tune_hyperparameters(X_train, y_train, use_class_weight=True):
    """
    Tune Random Forest and SVM with RandomizedSearchCV.
    Logistic Regression: small GridSearchCV (C).
    Returns dict of model_name -> best_estimator.
    """
    cw = "balanced" if use_class_weight else None
    best_models = {}

    # Logistic Regression: grid
    param_grid_lr = {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [1000]}
    lr = LogisticRegression(random_state=RANDOM_STATE, class_weight=cw)
    search_lr = GridSearchCV(
        lr, param_grid_lr, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1, refit=True
    )
    search_lr.fit(X_train, y_train)
    best_models["Logistic Regression"] = search_lr.best_estimator_

    # Random Forest: randomized search
    param_dist_rf = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search_rf = RandomizedSearchCV(
        rf,
        param_dist_rf,
        n_iter=32,
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search_rf.fit(X_train, y_train)
    best_models["Random Forest"] = search_rf.best_estimator_

    # SVM: randomized search
    param_dist_svm = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "class_weight": [None, "balanced"],
    }
    svm = SVC(probability=True, random_state=RANDOM_STATE, kernel="rbf")
    search_svm = RandomizedSearchCV(
        svm,
        param_dist_svm,
        n_iter=24,
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search_svm.fit(X_train, y_train)
    best_models["SVM"] = search_svm.best_estimator_

    return best_models


def train_with_smote(X_train, y_train, base_models=None):
    """
    Apply SMOTE to training data, then train models.
    Returns dict of model_name -> fitted estimator.
    """
    if base_models is None:
        base_models = _get_base_models(with_class_weight=False)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5, n_jobs=-1)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"   SMOTE failed ({e}), using original train set.")
        X_resampled, y_resampled = X_train, y_train
    fitted = {}
    for name, model in base_models.items():
        m = clone(model)
        m.fit(X_resampled, y_resampled)
        fitted[name] = m
    return fitted


def train_models(
    X_path=DEFAULT_X_PATH,
    y_path=DEFAULT_Y_PATH,
    test_size=0.2,
    random_state=RANDOM_STATE,
):
    """
    Train multiple ML models and save them (original baseline flow).
    """
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    print(f"\n1. Loaded X: {X.shape}, y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    models = _get_base_models(with_class_weight=False)
    results = {}

    print("\n2. Training baseline models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_pred": y_pred,
            "y_pred_proba": y_proba,
            "y_test": y_test,
        }
        print(f"   {name}: ROC-AUC = {results[name]['roc_auc']:.4f}")

    results_summary = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [results[m]["accuracy"] for m in results],
            "Precision": [results[m]["precision"] for m in results],
            "Recall": [results[m]["recall"] for m in results],
            "F1_Score": [results[m]["f1"] for m in results],
            "ROC_AUC": [results[m]["roc_auc"] for m in results],
        }
    )
    results_summary.to_csv("reports/model_results_summary.csv", index=False)
    best_name = results_summary.loc[results_summary["ROC_AUC"].idxmax(), "Model"]
    best_model = results[best_name]["model"]
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n3. Best model (baseline): {best_name} -> models/best_model.pkl")
    print("=" * 60)
    return results


def train_models_advanced(
    X_path=DEFAULT_X_PATH,
    y_path=DEFAULT_Y_PATH,
    test_size=0.2,
    cv_folds=CV_FOLDS,
    use_smote=True,
):
    """
    Full advanced pipeline:
    1) 5-fold CV report (mean ± std) for baseline and class_weight='balanced'
    2) Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for LR, RF, SVM
    3) Class imbalance: train with class_weight='balanced' and optionally SMOTE
    4) Compare and save best model; write reports/cv_results.csv, reports/tuning_results.csv
    """
    print("=" * 60)
    print("ADVANCED TRAINING: CV + TUNING + IMBALANCE")
    print("=" * 60)

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"\n1. Data: X {X.shape}, y {y.shape}")
    print(f"   Class counts: 0={n_neg}, 1={n_pos} (imbalance ratio ~{max(n_neg,n_pos)/max(1,min(n_neg,n_pos)):.1f}:1)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # ----- 5-fold CV -----
    print("\n2. 5-fold Cross-Validation (baseline models)...")
    cv_baseline = run_cross_validation(X, y, _get_base_models(with_class_weight=False), cv=cv_folds)
    cv_balanced = run_cross_validation(X, y, _get_base_models(with_class_weight=True), cv=cv_folds)

    rows_cv = []
    for name in cv_baseline:
        rows_cv.append(
            {
                "Model": name,
                "Setting": "baseline",
                "Accuracy_mean": cv_baseline[name]["accuracy_mean"],
                "Accuracy_std": cv_baseline[name]["accuracy_std"],
                "ROC_AUC_mean": cv_baseline[name]["roc_auc_mean"],
                "ROC_AUC_std": cv_baseline[name]["roc_auc_std"],
            }
        )
        rows_cv.append(
            {
                "Model": name,
                "Setting": "class_weight=balanced",
                "Accuracy_mean": cv_balanced[name]["accuracy_mean"],
                "Accuracy_std": cv_balanced[name]["accuracy_std"],
                "ROC_AUC_mean": cv_balanced[name]["roc_auc_mean"],
                "ROC_AUC_std": cv_balanced[name]["roc_auc_std"],
            }
        )
    df_cv = pd.DataFrame(rows_cv)
    df_cv.to_csv("reports/cv_results.csv", index=False)
    print("   Saved: reports/cv_results.csv")
    for name in cv_baseline:
        b = cv_baseline[name]
        bl = cv_balanced[name]
        print(f"   {name}:")
        print(f"      baseline  -> Accuracy {b['accuracy_mean']:.4f} ± {b['accuracy_std']:.4f}, ROC-AUC {b['roc_auc_mean']:.4f} ± {b['roc_auc_std']:.4f}")
        print(f"      balanced  -> Accuracy {bl['accuracy_mean']:.4f} ± {bl['accuracy_std']:.4f}, ROC-AUC {bl['roc_auc_mean']:.4f} ± {bl['roc_auc_std']:.4f}")

    # ----- Hyperparameter tuning -----
    print("\n3. Hyperparameter tuning (RandomizedSearchCV RF/SVM, GridSearchCV LR)...")
    tuned_models = tune_hyperparameters(X_train, y_train, use_class_weight=True)
    print("   Tuning done.")

    # ----- Train final models: tuned + optional SMOTE -----
    print("\n4. Training final models (tuned + class_weight=balanced)...")
    results_final = {}
    for name, model in tuned_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results_final[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        print(f"   {name}: ROC-AUC = {results_final[name]['roc_auc']:.4f}")

    if use_smote:
        print("\n5. Training with SMOTE (resampled train only)...")
        smote_models = train_with_smote(X_train, y_train, _get_base_models(with_class_weight=False))
        for name, model in smote_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            results_final[f"{name} (SMOTE)"] = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            print(f"   {name} (SMOTE): ROC-AUC = {results_final[f'{name} (SMOTE)']['roc_auc']:.4f}")

    # ----- Summary and save best -----
    summary_rows = [
        {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1_Score": r["f1"],
            "ROC_AUC": r["roc_auc"],
        }
        for name, r in results_final.items()
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("reports/model_results_summary.csv", index=False)
    df_summary.to_csv("reports/tuning_results.csv", index=False)
    print("\n   Saved: reports/model_results_summary.csv, reports/tuning_results.csv")

    best_name = df_summary.loc[df_summary["ROC_AUC"].idxmax(), "Model"]
    best_model = results_final[best_name]["model"]
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n6. Best model: {best_name} (ROC-AUC {results_final[best_name]['roc_auc']:.4f})")
    print("   Saved: models/best_model.pkl")

    for name in tuned_models:
        pkl_name = f"models/model_{name.lower().replace(' ', '_')}.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(tuned_models[name], f)
    print("   Saved: models/model_*.pkl (tuned LR, RF, SVM)")

    print("\n" + "=" * 60)
    print("ADVANCED TRAINING COMPLETE")
    print("=" * 60)
    return results_final
