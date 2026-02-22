# Project Structure

## Complete Directory Tree

```
placebo effect/
│
├── data/                              # Data directory
│   ├── PLACEHOLDER.txt               # Ensures directory exists
│   ├── dataset.csv                    # Generated dataset (created when run)
│   ├── X_scaled.csv                   # Scaled features (created when run)
│   └── y_target.csv                   # Target labels (created when run)
│
├── models/                            # Trained models directory
│   ├── PLACEHOLDER.txt               # Ensures directory exists
│   ├── model_logistic_regression.pkl  # Logistic Regression model (created when run)
│   ├── model_random_forest.pkl        # Random Forest model (created when run)
│   ├── model_svm.pkl                  # SVM model (created when run)
│   ├── best_model.pkl                 # Best performing model (created when run)
│   ├── scaler.pkl                     # Feature scaler (created when run)
│   ├── label_encoder_gender.pkl       # Gender encoder (created when run)
│   └── label_encoder_treatment.pkl   # Treatment encoder (created when run)
│
├── reports/                           # Reports directory
│   ├── final_report.md                # Complete project report (Markdown)
│   ├── model_results_summary.csv      # Model performance metrics (created when run)
│   └── figures/                       # Visualizations directory
│       ├── PLACEHOLDER.txt           # Ensures directory exists
│       ├── correlation_heatmap.png    # Correlation matrix (created when run)
│       ├── trait_distributions.png    # Trait distributions (created when run)
│       ├── placebo_response_analysis.png # Response analysis (created when run)
│       ├── feature_importance_correlation.png # Feature correlations (created when run)
│       ├── age_distribution.png       # Age analysis (created when run)
│       ├── roc_curves.png             # ROC curves (created when run)
│       ├── confusion_matrices.png     # Confusion matrices (created when run)
│       ├── model_comparison.png        # Model comparison (created when run)
│       ├── precision_recall_curves.png # PR curves (created when run)
│       └── feature_importance.png     # Feature importance (created when run)
│
├── src/                               # Source code directory
│   ├── __init__.py                    # Package initialization
│   ├── preprocessing.py               # Data generation & preprocessing
│   ├── eda.py                         # Exploratory data analysis
│   ├── train_model.py                 # Model training
│   ├── evaluate_model.py              # Model evaluation
│   └── predict.py                     # Prediction interface
│
├── main.py                            # Main pipeline script
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── STRUCTURE.md                       # This file
└── .gitignore                         # Git ignore patterns
```

## File Descriptions

### Root Files
- **`main.py`**: Main entry point - runs the complete pipeline
- **`requirements.txt`**: Python package dependencies
- **`README.md`**: Comprehensive project documentation
- **`.gitignore`**: Git ignore patterns for version control

### Source Code (`src/`)
- **`__init__.py`**: Makes `src` a Python package
- **`preprocessing.py`**: 
  - `generate_dataset()`: Creates synthetic psychological dataset
  - `preprocess_data()`: Cleans, encodes, and scales data
- **`eda.py`**: 
  - `perform_eda()`: Creates all exploratory visualizations
- **`train_model.py`**: 
  - `train_models()`: Trains all ML models and saves them
- **`evaluate_model.py`**: 
  - `evaluate_models()`: Evaluates models and creates comparison visualizations
- **`predict.py`**: 
  - `predict_placebo_response()`: Makes predictions on new data
  - `example_predictions()`: Demonstrates prediction system

### Data Directory (`data/`)
- Contains all datasets (CSV files)
- Created automatically when pipeline runs

### Models Directory (`models/`)
- Contains all trained models and preprocessors (PKL files)
- Created automatically when pipeline runs

### Reports Directory (`reports/`)
- **`final_report.md`**: Complete project report (can be converted to DOCX)
- **`model_results_summary.csv`**: Performance metrics for all models
- **`figures/`**: All visualization PNG files

## Usage

### Run Complete Pipeline
```bash
python main.py
```

### Use Individual Modules
```python
from src.preprocessing import generate_dataset, preprocess_data
from src.eda import perform_eda
from src.train_model import train_models
from src.evaluate_model import evaluate_models
from src.predict import predict_placebo_response

# Generate dataset
data = generate_dataset(n_samples=1000)

# Preprocess
X, y, scaler, le_gender, le_treatment = preprocess_data('data/dataset.csv')

# EDA
perform_eda('data/dataset.csv')

# Train models
results = train_models('data/X_scaled.csv', 'data/y_target.csv')

# Evaluate
evaluate_models('data/X_scaled.csv', 'data/y_target.csv')

# Predict
result = predict_placebo_response(...)
```

## Notes

- All directories are created automatically when needed
- PLACEHOLDER.txt files ensure directories exist in version control
- Generated files (CSV, PKL, PNG) are ignored by git (see `.gitignore`)
- The structure follows best practices for ML projects
- Clean separation of concerns: data, models, reports, source code
