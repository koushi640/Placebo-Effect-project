"""
Data Generation and Preprocessing Module
Combines dataset generation and preprocessing steps.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


def load_and_prepare_dataset(source_path='data/Dataset.csv'):
    """
    Load clinical trial dataset and prepare it for placebo response analysis.
    Extracts features from clinical trial data and creates placebo response target.
    
    Parameters:
    -----------
    source_path : str
        Path to the source clinical trial dataset
    
    Returns:
    --------
    pd.DataFrame : Prepared dataset with features and target
    """
    print("=" * 60)
    print("LOADING AND PREPARING DATASET")
    print("=" * 60)
    
    # Try to load prepared dataset if it exists and is complete
    if os.path.exists('data/dataset.csv'):
        print(f"\n1. Loading existing prepared dataset from data/dataset.csv...")
        data = pd.read_csv('data/dataset.csv')
        print(f"   Loaded shape: {data.shape}")
        required_cols = {
            "Age", "Openness", "Conscientiousness", "Extraversion",
            "Agreeableness", "Neuroticism", "Optimism", "Stress_Level",
            "Anxiety_Level", "Emotional_Resilience", "Phase_Number",
            "Log_Enrollment", "Status_Completed", "Is_Placebo_Controlled",
            "Start_Year", "Start_Month", "Condition_Encoded",
            "Gender", "Treatment_Type", "Placebo_Response",
        }
        if required_cols.issubset(set(data.columns)):
            return data
        print("   Prepared dataset is missing required columns. Rebuilding from raw dataset...")
    
    # Load dataset from source
    print(f"\n1. Loading dataset from {source_path}...")
    try:
        data = pd.read_csv(source_path)
    except FileNotFoundError:
        # Try alternative path
        alt_path = source_path.replace('c:/', 'C:/').replace('/', '\\')
        try:
            data = pd.read_csv(alt_path)
        except:
            print(f"   Error: Could not find dataset at {source_path}")
            print(f"   Please ensure the dataset file exists.")
            raise
    print(f"   Original shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Create features from clinical trial data
    print("\n2. Extracting features from clinical trial data...")
    
    # Check if placebo-controlled (from Title and Summary)
    data['Is_Placebo_Controlled'] = (
        data['Title'].str.contains('Placebo', case=False, na=False) |
        data['Summary'].str.contains('Placebo', case=False, na=False)
    ).astype(int)
    
    # Extract Phase number (1, 2, 3, 4)
    data['Phase_Number'] = data['Phase'].str.extract(r'(\d+)', expand=False).astype(float)
    data['Phase_Number'] = data['Phase_Number'].fillna(0)
    
    # Enrollment (log transform for better distribution)
    data['Enrollment'] = pd.to_numeric(data['Enrollment'], errors='coerce')
    data['Enrollment'] = data['Enrollment'].fillna(0)
    data['Log_Enrollment'] = np.log1p(data['Enrollment'])
    
    # Status encoding (Completed = 1, others = 0)
    data['Status_Completed'] = (data['Status'] == 'Completed').astype(int)
    
    # Year features
    data['Start_Year'] = pd.to_numeric(data['Start_Year'], errors='coerce')
    data['Start_Year'] = data['Start_Year'].fillna(data['Start_Year'].median())
    data['Years_Since_Start'] = 2024 - data['Start_Year']
    
    # Month features
    data['Start_Month'] = pd.to_numeric(data['Start_Month'], errors='coerce')
    data['Start_Month'] = data['Start_Month'].fillna(data['Start_Month'].median())
    
    # Condition encoding (simplified - use first condition)
    data['Condition_Encoded'] = data['Condition'].astype('category').cat.codes
    
    # Create synthetic psychological features based on trial characteristics
    # These simulate psychological factors that might influence placebo response
    np.random.seed(42)
    n_samples = len(data)
    
    # Higher enrollment might correlate with better expectations (optimism proxy)
    enrollment_normalized = (data['Enrollment'] - data['Enrollment'].min()) / (data['Enrollment'].max() - data['Enrollment'].min() + 1)
    data['Optimism'] = (enrollment_normalized * 5 + np.random.normal(5, 1.5, n_samples)).clip(1, 10)
    
    # Phase 3/4 trials might have lower stress (more established)
    phase_stress_map = {0: 7, 1: 6.5, 2: 6, 3: 5, 4: 4.5}
    data['Stress_Level'] = data['Phase_Number'].map(phase_stress_map).fillna(6) + np.random.normal(0, 1, n_samples)
    data['Stress_Level'] = data['Stress_Level'].clip(1, 10)
    
    # Completed trials might indicate better resilience
    data['Emotional_Resilience'] = (data['Status_Completed'] * 2 + np.random.normal(6, 1.5, n_samples)).clip(1, 10)
    
    # Big Five traits (simulated based on trial characteristics)
    data['Openness'] = (np.random.normal(6.5, 1.5, n_samples)).clip(1, 10)
    data['Conscientiousness'] = (data['Status_Completed'] * 1.5 + np.random.normal(6.5, 1.3, n_samples)).clip(1, 10)
    data['Extraversion'] = (enrollment_normalized * 2 + np.random.normal(6, 1.4, n_samples)).clip(1, 10)
    data['Agreeableness'] = (np.random.normal(6.5, 1.3, n_samples)).clip(1, 10)
    data['Neuroticism'] = ((1 - data['Status_Completed']) * 2 + np.random.normal(5, 1.6, n_samples)).clip(1, 10)
    
    # Anxiety level (inverse of enrollment)
    data['Anxiety_Level'] = ((1 - enrollment_normalized) * 3 + np.random.normal(5, 1.8, n_samples)).clip(1, 10)
    
    # Age (simulated)
    data['Age'] = np.random.randint(18, 80, n_samples)
    data['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
    
    # Treatment type based on placebo control
    data['Treatment_Type'] = data['Is_Placebo_Controlled'].map({1: 'Sugar Pill', 0: 'Active Drug'})
    
    # Create placebo response target
    # Higher response if: placebo-controlled, completed, higher enrollment, Phase 3/4
    placebo_response_probability = (
        0.25 * (data['Is_Placebo_Controlled']) +
        0.20 * (data['Status_Completed']) +
        0.15 * (enrollment_normalized) +
        0.15 * ((data['Phase_Number'] >= 3).astype(int)) +
        0.10 * (data['Optimism'] / 10) +
        0.10 * ((10 - data['Stress_Level']) / 10) +
        0.05 * (data['Emotional_Resilience'] / 10) +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)
    
    data['Placebo_Response'] = (placebo_response_probability > 0.5).astype(int)
    
    # Select relevant columns for ML
    feature_cols = [
        'Age', 'Openness', 'Conscientiousness', 'Extraversion', 
        'Agreeableness', 'Neuroticism', 'Optimism', 'Stress_Level', 
        'Anxiety_Level', 'Emotional_Resilience', 'Phase_Number',
        'Log_Enrollment', 'Status_Completed', 'Is_Placebo_Controlled',
        'Start_Year', 'Start_Month', 'Condition_Encoded', 'Placebo_Response'
    ]
    
    # Add gender encoding
    le_gender_temp = LabelEncoder()
    data['Gender_Encoded'] = le_gender_temp.fit_transform(data['Gender'])
    feature_cols.append('Gender_Encoded')
    
    # Treatment type encoding
    le_treatment_temp = LabelEncoder()
    data['Treatment_Type_Encoded'] = le_treatment_temp.fit_transform(data['Treatment_Type'])
    feature_cols.append('Treatment_Type_Encoded')
    
    # Create final dataset
    final_data = data[feature_cols + ['Gender', 'Treatment_Type']].copy()
    
    # Round numeric columns
    numeric_cols = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 
                    'Neuroticism', 'Optimism', 'Stress_Level', 'Anxiety_Level', 
                    'Emotional_Resilience']
    for col in numeric_cols:
        final_data[col] = final_data[col].round(2)
    
    # Save prepared dataset
    final_data.to_csv('data/dataset.csv', index=False)
    print(f"\n✓ Dataset prepared and saved as 'data/dataset.csv'")
    print(f"  Shape: {final_data.shape}")
    print(f"  Placebo Response Distribution:")
    print(f"  {final_data['Placebo_Response'].value_counts().to_dict()}")
    print(f"  Placebo Response Percentage:")
    print(f"  {final_data['Placebo_Response'].value_counts(normalize=True) * 100}")
    
    return final_data


def preprocess_data(data_path='data/dataset.csv'):
    """
    Preprocess the dataset: handle missing values, encode categorical data, scale features.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
    
    Returns:
    --------
    tuple : (X_scaled, y, scaler, le_gender, le_treatment)
        Preprocessed features, target, and preprocessing objects
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"   Original shape: {data.shape}")
    
    # Check for missing values
    print("\n2. Checking for missing values...")
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"   Removing {missing_values.sum()} missing values...")
        data = data.dropna()
        print(f"   Shape after removing missing values: {data.shape}")
    else:
        print("   ✓ No missing values found!")
    
    # Ensure required categorical columns exist
    if 'Gender' not in data.columns:
        if 'Gender_Encoded' in data.columns:
            data['Gender'] = data['Gender_Encoded'].map({0: 'Male', 1: 'Female'}).fillna('Male')
        else:
            data['Gender'] = 'Male'

    if 'Treatment_Type' not in data.columns:
        if 'Treatment_Type_Encoded' in data.columns:
            data['Treatment_Type'] = data['Treatment_Type_Encoded'].map({0: 'Sugar Pill', 1: 'Active Drug'}).fillna('Active Drug')
        else:
            data['Treatment_Type'] = 'Active Drug'

    # Handle categorical data (if not already encoded)
    print("\n3. Encoding categorical variables...")
    le_gender = LabelEncoder()
    le_treatment = LabelEncoder()
    
    if 'Gender_Encoded' not in data.columns:
        data['Gender_Encoded'] = le_gender.fit_transform(data['Gender'])
    else:
        le_gender.fit(data['Gender'].unique())
        le_gender.classes_ = np.array(sorted(data['Gender'].unique()))
    
    if 'Treatment_Type_Encoded' not in data.columns:
        data['Treatment_Type_Encoded'] = le_treatment.fit_transform(data['Treatment_Type'])
    else:
        le_treatment.fit(data['Treatment_Type'].unique())
        le_treatment.classes_ = np.array(sorted(data['Treatment_Type'].unique()))
    
    print(f"   Gender encoding: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
    print(f"   Treatment encoding: {dict(zip(le_treatment.classes_, le_treatment.transform(le_treatment.classes_)))}")
    
    # Prepare features and target
    print("\n4. Preparing features and target...")
    feature_columns = [
        'Age', 'Openness', 'Conscientiousness', 'Extraversion', 
        'Agreeableness', 'Neuroticism', 'Optimism', 'Stress_Level', 
        'Anxiety_Level', 'Emotional_Resilience', 'Phase_Number',
        'Log_Enrollment', 'Status_Completed', 'Is_Placebo_Controlled',
        'Start_Year', 'Start_Month', 'Condition_Encoded', 'Gender_Encoded', 
        'Treatment_Type_Encoded'
    ]
    
    X = data[feature_columns].copy()
    y = data['Placebo_Response'].copy()
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Feature scaling
    print("\n5. Applying feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("   ✓ Features scaled using StandardScaler")
    
    # Save preprocessed data
    print("\n6. Saving preprocessed data...")
    X_scaled_df.to_csv('data/X_scaled.csv', index=False)
    y.to_csv('data/y_target.csv', index=False)
    print("   ✓ Saved: data/X_scaled.csv, data/y_target.csv")
    
    # Save scaler and encoders
    print("\n7. Saving preprocessing objects...")
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder_gender.pkl', 'wb') as f:
        pickle.dump(le_gender, f)
    with open('models/label_encoder_treatment.pkl', 'wb') as f:
        pickle.dump(le_treatment, f)
    print("   ✓ Saved: models/scaler.pkl, models/label_encoder_*.pkl")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return X_scaled_df, y, scaler, le_gender, le_treatment
