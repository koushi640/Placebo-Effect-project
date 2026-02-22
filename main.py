"""
Main Script: Complete ML Pipeline for Placebo Response Prediction
This script orchestrates the entire pipeline from data generation to prediction.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import load_and_prepare_dataset, preprocess_data
from src.eda import perform_eda
from src.train_model import train_models_advanced
from models.quicktrain import quicktrain as quicktrain_models
from src.evaluate_model import evaluate_models
from src.predict import example_predictions


def main():
    """Main execution function - runs complete pipeline."""
    print("\n" + "=" * 70)
    print("PLACEBO RESPONSE PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 70)
    
    try:
        # Step 1: Load and Prepare Dataset
        print("\n" + "=" * 70)
        print("STEP 1: LOADING AND PREPARING DATASET")
        print("=" * 70)
        data = load_and_prepare_dataset('data/Dataset.csv')
        
        # Step 2: Preprocess Data
        print("\n" + "=" * 70)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 70)
        X_scaled, y, scaler, le_gender, le_treatment = preprocess_data('data/dataset.csv')
        
        # Step 3: Exploratory Data Analysis
        print("\n" + "=" * 70)
        print("STEP 3: EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        perform_eda('data/dataset.csv')
        
        # Step 4: Quick Models (baseline)
        print("\n" + "=" * 70)
        print("STEP 4: QUICK MODEL TRAINING (BASELINE)")
        print("=" * 70)
        quicktrain_models(output_prefix="quick_")

        # Step 5: Train Models (with CV, hyperparameter tuning, class imbalance)
        print("\n" + "=" * 70)
        print("STEP 5: MODEL TRAINING (CV + TUNING + IMBALANCE)")
        print("=" * 70)
        results = train_models_advanced('data/X_scaled.csv', 'data/y_target.csv', use_smote=True)
        
        # Step 6: Evaluate Models
        print("\n" + "=" * 70)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 70)
        evaluate_models('data/X_scaled.csv', 'data/y_target.csv')
        
        # Step 7: Example Predictions
        print("\n" + "=" * 70)
        print("STEP 7: EXAMPLE PREDICTIONS")
        print("=" * 70)
        example_predictions()
        
        # Summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  ✓ data/dataset.csv")
        print("  ✓ data/X_scaled.csv, data/y_target.csv")
        print("  ✓ models/quick_*.pkl (quick baseline models)")
        print("  ✓ models/*.pkl (tuned models)")
        print("  ✓ reports/quick_model_results_summary.csv")
        print("  ✓ reports/model_results_summary.csv")
        print("  ✓ reports/cv_results.csv (5-fold CV mean ± std)")
        print("  ✓ reports/tuning_results.csv")
        print("  ✓ reports/figures/*.png (all visualizations)")
        print("\n🎉 All steps completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
