"""
Model Evaluation Module
Creates comprehensive evaluation visualizations and metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                            precision_recall_curve)

# Ensure directories exist
os.makedirs('reports/figures', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def evaluate_models(X_path='data/X_scaled.csv', y_path='data/y_target.csv',
                   test_size=0.2, random_state=42):
    """
    Evaluate trained models and create visualizations.
    
    Parameters:
    -----------
    X_path : str
        Path to scaled features CSV
    y_path : str
        Path to target CSV
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    """
    print("=" * 60)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    
    # Split to get test set (same random state as training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Load models
    print("\n2. Loading trained models...")
    models = {}
    model_names = ['Logistic Regression', 'Random Forest', 'SVM']
    
    for name in model_names:
        filename = f"models/model_{name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"   ✓ Loaded: {filename}")
    
    # Get predictions
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)[:, 1]
    
    # ROC Curves
    print("\n3. Creating ROC curves...")
    plt.figure(figsize=(10, 8))
    
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/roc_curves.png")
    plt.close()
    
    # Confusion Matrices
    print("\n4. Creating confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar_kws={"shrink": 0.8}, square=True, linewidths=1)
        axes[idx].set_title(f'{name}\nAccuracy: {cm.trace()/cm.sum():.3f}', 
                            fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['Not Improved', 'Improved'])
        axes[idx].set_yticklabels(['Not Improved', 'Improved'])
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/confusion_matrices.png")
    plt.close()
    
    # Model Comparison Bar Chart
    print("\n5. Creating model comparison chart...")
    results_df = pd.read_csv('reports/model_results_summary.csv')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    x = np.arange(len(model_names))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2) * width + width/2
        ax.bar(x + offset, results_df[metric], width, label=metric, 
               edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/model_comparison.png")
    plt.close()
    
    # Precision-Recall Curves
    print("\n6. Creating Precision-Recall curves...")
    plt.figure(figsize=(10, 8))
    
    for name in model_names:
        precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/precision_recall_curves.png")
    plt.close()
    
    # Feature Importance (for Random Forest)
    print("\n7. Creating feature importance plot...")
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', 
                palette='viridis', edgecolor='black')
    plt.title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: reports/figures/feature_importance.png")
    plt.close()
    
    print("\n   Top 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETE! All visualizations saved.")
    print("=" * 60)
