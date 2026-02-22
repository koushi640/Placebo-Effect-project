"""
Prediction Module
User-friendly interface for making placebo response predictions.
"""

import numpy as np
import pickle
import os


def load_models():
    """Load trained models and preprocessors."""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/label_encoder_gender.pkl', 'rb') as f:
            le_gender = pickle.load(f)
        
        with open('models/label_encoder_treatment.pkl', 'rb') as f:
            le_treatment = pickle.load(f)
        
        return best_model, scaler, le_gender, le_treatment
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model files not found: {e}\n"
            "Please run the training pipeline first (python main.py)"
        )


def predict_placebo_response(age, gender, openness, conscientiousness, extraversion,
                            agreeableness, neuroticism, optimism, stress_level,
                            anxiety_level, emotional_resilience, treatment_type,
                            phase_number=3, log_enrollment=5.0, status_completed=1,
                            is_placebo_controlled=1, start_year=2010, start_month=6,
                            condition_encoded=0):
    """
    Predict placebo response based on psychological factors and trial characteristics.
    
    Parameters:
    -----------
    age : int
        Age of the individual
    gender : str
        Gender ('Male' or 'Female')
    openness : float
        Openness score (1-10)
    conscientiousness : float
        Conscientiousness score (1-10)
    extraversion : float
        Extraversion score (1-10)
    agreeableness : float
        Agreeableness score (1-10)
    neuroticism : float
        Neuroticism score (1-10)
    optimism : float
        Optimism score (1-10)
    stress_level : float
        Stress level (1-10)
    anxiety_level : float
        Anxiety level (1-10)
    emotional_resilience : float
        Emotional resilience score (1-10)
    treatment_type : str
        Treatment type ('Sugar Pill' or 'Active Drug')
    phase_number : float
        Clinical trial phase (0-4)
    log_enrollment : float
        Log of enrollment number
    status_completed : int
        Whether trial completed (1) or not (0)
    is_placebo_controlled : int
        Whether trial is placebo-controlled (1) or not (0)
    start_year : int
        Year trial started
    start_month : int
        Month trial started (1-12)
    condition_encoded : int
        Encoded condition type
    
    Returns:
    --------
    dict : Prediction results with response, probability, and recommendation
    """
    # Load models
    best_model, scaler, le_gender, le_treatment = load_models()
    
    # Encode categorical variables
    gender_encoded = le_gender.transform([gender])[0]
    treatment_encoded = le_treatment.transform([treatment_type])[0]
    
    # Create feature array (matching the feature order from preprocessing)
    features = np.array([[
        age, openness, conscientiousness, extraversion, agreeableness,
        neuroticism, optimism, stress_level, anxiety_level,
        emotional_resilience, phase_number, log_enrollment, status_completed,
        is_placebo_controlled, start_year, start_month, condition_encoded,
        gender_encoded, treatment_encoded
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = best_model.predict(features_scaled)[0]
    probability = best_model.predict_proba(features_scaled)[0][1]
    
    # Generate recommendation
    if prediction == 1:
        response = "YES"
        recommendation = "Placebo (Sugar Pill) - High likelihood of positive response"
    else:
        response = "NO"
        recommendation = "Consider alternative treatment - Low likelihood of placebo response"
    
    return {
        'Placebo_Response': response,
        'Probability': probability,
        'Recommendation': recommendation
    }


def example_predictions():
    """Run example predictions to demonstrate the system."""
    print("=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    examples = [
        {
            'name': 'Example 1: High Optimism, Low Stress, Placebo-Controlled',
            'age': 35,
            'gender': 'Female',
            'openness': 7.5,
            'conscientiousness': 8.0,
            'extraversion': 7.0,
            'agreeableness': 7.5,
            'neuroticism': 3.0,
            'optimism': 8.5,
            'stress_level': 3.0,
            'anxiety_level': 2.5,
            'emotional_resilience': 8.0,
            'treatment_type': 'Sugar Pill',
            'phase_number': 3,
            'log_enrollment': 7.0,
            'status_completed': 1,
            'is_placebo_controlled': 1,
            'start_year': 2010,
            'start_month': 6,
            'condition_encoded': 0
        },
        {
            'name': 'Example 2: Low Optimism, High Stress, Non-Placebo',
            'age': 45,
            'gender': 'Male',
            'openness': 5.0,
            'conscientiousness': 6.0,
            'extraversion': 4.5,
            'agreeableness': 5.5,
            'neuroticism': 8.0,
            'optimism': 4.0,
            'stress_level': 8.5,
            'anxiety_level': 9.0,
            'emotional_resilience': 4.0,
            'treatment_type': 'Active Drug',
            'phase_number': 2,
            'log_enrollment': 4.0,
            'status_completed': 0,
            'is_placebo_controlled': 0,
            'start_year': 2015,
            'start_month': 3,
            'condition_encoded': 1
        },
        {
            'name': 'Example 3: Moderate Profile',
            'age': 28,
            'gender': 'Female',
            'openness': 6.5,
            'conscientiousness': 7.0,
            'extraversion': 6.0,
            'agreeableness': 6.5,
            'neuroticism': 5.0,
            'optimism': 6.5,
            'stress_level': 5.5,
            'anxiety_level': 5.0,
            'emotional_resilience': 6.5,
            'treatment_type': 'Sugar Pill',
            'phase_number': 3,
            'log_enrollment': 5.5,
            'status_completed': 1,
            'is_placebo_controlled': 1,
            'start_year': 2012,
            'start_month': 8,
            'condition_encoded': 0
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}")
        print("-" * 60)
        result = predict_placebo_response(
            example['age'], example['gender'], example['openness'],
            example['conscientiousness'], example['extraversion'],
            example['agreeableness'], example['neuroticism'],
            example['optimism'], example['stress_level'],
            example['anxiety_level'], example['emotional_resilience'],
            example['treatment_type'],
            example['phase_number'], example['log_enrollment'],
            example['status_completed'], example['is_placebo_controlled'],
            example['start_year'], example['start_month'],
            example['condition_encoded']
        )
        print(f"Input:")
        print(f"  Optimism: {example['optimism']}")
        print(f"  Stress Level: {example['stress_level']}")
        print(f"  Emotional Resilience: {example['emotional_resilience']}")
        print(f"\nOutput:")
        print(f"  Placebo Response: {result['Placebo_Response']}")
        print(f"  Probability: {result['Probability']*100:.2f}%")
        print(f"  Recommendation: {result['Recommendation']}")


if __name__ == "__main__":
    example_predictions()
