import numpy as np
import pandas as pd
from typing import Dict, Any

SLEEP_DURATION_MAP = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "More than 8 hours": 4
}

DIETARY_HABITS_MAP = {
    "Unhealthy": 1,
    "Moderate": 2,
    "Healthy": 3
}

GENDER_MAP = {
    "Male": 0,
    "Female": 1
}

def transform_profile_to_features(profile: Dict[str, Any], scaler, model) -> pd.DataFrame:
    """Transform profile to match EXACT model features."""
    
    # 1. Numeric features
    numeric_features = {
        'age': profile.get('age'),
        'cgpa': profile.get('cgpa'),
        'academic_pressure': profile.get('academic_pressure'),
        'study_satisfaction': profile.get('study_satisfaction'),
        'work/study_hours': profile.get('study_hours'),
        'financial_stress': profile.get('financial_stress')
    }
    
    # 2. Categorical features
    categorical_features = {
        'gender': GENDER_MAP.get(profile.get('gender'), 0), # type: ignore
        'sleep_duration': SLEEP_DURATION_MAP.get(profile.get('sleep_duration'), 0), # type: ignore
        'dietary_habits': DIETARY_HABITS_MAP.get(profile.get('dietary_habits'), 0), # type: ignore
        'suicidal_thoughts': 1 if profile.get('suicidal_thoughts') else 0,
        'family_mental_history': 1 if profile.get('family_history') else 0
    }
    
    # 3. ONE-HOT DEGREE
    degree_features = {}
    user_degree = profile.get('degree', 'BA')
    
    model_features = model.feature_names_in_
    degree_columns = [col for col in model_features if col.startswith('degree_')]
    
    for deg_col in degree_columns:
        degree_name = deg_col.replace('degree_', '')
        degree_features[deg_col] = 1 if user_degree == degree_name else 0
    
    # 4. Engineered features (BEFORE scaling)
    sleep_val = categorical_features['sleep_duration']
    acad_pressure_raw = numeric_features['academic_pressure']
    fin_stress_raw = numeric_features['financial_stress']
    
    engineered_pre = {
        'sleep_adequate': 1 if sleep_val >= 3 else 0,
        'high_academic_pressure': 1 if acad_pressure_raw and acad_pressure_raw > 3 else 0,
        'stress_interaction': (fin_stress_raw or 0) * (acad_pressure_raw or 0)
    }
    
    # 5. Combine
    all_features = {
        **numeric_features,
        **categorical_features,
        **degree_features,
        **engineered_pre
    }
    
    # 6. DataFrame
    df = pd.DataFrame([all_features])
    
    # 7. Scale numeric
    numeric_cols = ['age', 'cgpa', 'academic_pressure', 'study_satisfaction', 
                    'work/study_hours', 'financial_stress']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # 8. Recalculate engineered (AFTER scaling)
    df['stress_interaction'] = df['financial_stress'] * df['academic_pressure']
    df['high_academic_pressure'] = (df['academic_pressure'] > 0).astype(int)
    
    # 9. REORDER theo exact model features
    df = df[model.feature_names_in_]
    
    return df


def load_model_and_scaler(path: str):
    """Load model and scaler from given paths."""
    model_path = f"{path}/logreg.pkl"
    scaler_path = f"{path}/scaler.pkl"
    import joblib
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler