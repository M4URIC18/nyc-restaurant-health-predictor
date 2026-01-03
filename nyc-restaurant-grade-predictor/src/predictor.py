"""
Restaurant grade prediction module.

Handles model loading, feature encoding, and predictions.
"""

import json
import joblib
import numpy as np
import pandas as pd
import os
import streamlit as st

# -------------------------------------------------
# Configuration
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "restaurant_grade_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# Feature columns (must match trainer.py)
# NOTE: We use PREVIOUS inspection data to predict NEXT grade
FEATURE_COLUMNS = [
    'borough',
    'zipcode',
    'cuisine_description',
    'days_since_last_inspection',
    'inspection_frequency',
    'prev_grade_1',
    'prev_grade_2',
    'prev_score_1',      # Score from previous inspection
    'prev_score_2',      # Score from inspection before that
    'critical_violations_12mo',
    'total_violations_all_time',
    'avg_score_historical',
    'score_trend',
    'grade_stability',
    'cuisine_avg_score',
    'zipcode_avg_score',
    'violation_diversity'
]

CATEGORICAL_COLUMNS = ['borough', 'cuisine_description', 'prev_grade_1', 'prev_grade_2']

# Default values for new restaurants with no history
FEATURE_DEFAULTS = {
    'days_since_last_inspection': 0,
    'inspection_frequency': 1.0,
    'prev_grade_1': 'Unknown',
    'prev_grade_2': 'Unknown',
    'prev_score_1': 13.0,   # Default to borderline A score
    'prev_score_2': 13.0,
    'critical_violations_12mo': 0,
    'total_violations_all_time': 1,
    'avg_score_historical': 13.0,
    'score_trend': 0.0,
    'grade_stability': 1,
    'cuisine_avg_score': 15.0,
    'zipcode_avg_score': 15.0,
    'violation_diversity': 1
}


# -------------------------------------------------
# Model loading with caching support
# -------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model...")
def _cached_load_model():
    """
    Internal function to load model with Streamlit caching.

    Uses @st.cache_resource because:
    - sklearn model is not pickle-serializable for cache_data
    - Model should persist across reruns and sessions
    - Only needs to load once per server restart
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)

    return model, metadata


def load_model(force_reload: bool = False):
    """
    Load model and metadata.

    Args:
        force_reload: If True, clear cache and reload from disk.
                     Use after retraining the model.

    Returns:
        tuple: (model, metadata)
    """
    if force_reload:
        _cached_load_model.clear()

    return _cached_load_model()


def clear_model_cache():
    """Clear the cached model, forcing reload on next prediction."""
    _cached_load_model.clear()


def get_model_metadata():
    """Get current model metadata."""
    _, metadata = load_model()
    return metadata


def model_needs_retraining() -> bool:
    """
    Check if the model needs to be retrained.

    Returns True if:
    - metadata indicates retraining needed
    - model feature count doesn't match expected features
    """
    try:
        model, metadata = load_model()

        # Check explicit flag
        if metadata.get('requires_retraining', False):
            return True

        # Check if model was trained (has training metrics)
        if metadata.get('training_metrics') is None:
            return True

        # Check feature count matches
        expected_features = len(FEATURE_COLUMNS)
        if hasattr(model, 'n_features_in_'):
            if model.n_features_in_ != expected_features:
                return True

        return False
    except Exception:
        return True


# -------------------------------------------------
# Feature encoding
# -------------------------------------------------

def build_feature_vector(restaurant_data: dict) -> np.ndarray:
    """
    Transform restaurant data dict into model-ready feature array.

    Expected input (all features):
    {
        "borough": "Queens",
        "zipcode": "11372",
        "cuisine_description": "Latin American",
        "days_since_last_inspection": 45,
        "inspection_frequency": 1.5,
        "prev_grade_1": "A",
        "prev_grade_2": "B",
        "prev_score_1": 12,
        "prev_score_2": 15,
        "critical_violations_12mo": 2,
        "total_violations_all_time": 15,
        "avg_score_historical": 14.5,
        "score_trend": -0.02,
        "grade_stability": 1,
        "cuisine_avg_score": 18.3,
        "zipcode_avg_score": 16.2,
        "violation_diversity": 5
    }

    Missing features will be filled with defaults.

    Returns:
        numpy array with encoded features in correct order
    """
    _, metadata = load_model()
    encoders = metadata.get('encoders', {})

    features = []

    for col in FEATURE_COLUMNS:
        value = restaurant_data.get(col)

        # Use defaults for missing values
        if value is None:
            value = FEATURE_DEFAULTS.get(col, 0)

        # Encode categorical columns
        if col in CATEGORICAL_COLUMNS:
            str_val = str(value).strip().title()
            encoder = encoders.get(col, {})
            encoded = encoder.get(str_val, 0)
            features.append(encoded)

        # Handle zipcode (numeric string)
        elif col == 'zipcode':
            try:
                features.append(float(str(value).replace(',', '')))
            except (ValueError, TypeError):
                features.append(0.0)

        # All other numeric columns
        else:
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

class ModelNeedsRetrainingError(Exception):
    """Raised when the model needs to be retrained before predictions can be made."""
    pass


def predict_restaurant_grade(restaurant_data: dict) -> dict:
    """
    Predict health grade for a restaurant.

    Args:
        restaurant_data: Dict with restaurant features

    Returns:
        dict with:
            - grade: Predicted grade (A, B, or C)
            - probabilities: Dict of grade probabilities
            - raw_output: Raw probability array

    Raises:
        ModelNeedsRetrainingError: If model needs to be retrained
    """
    # Check if model is compatible
    if model_needs_retraining():
        raise ModelNeedsRetrainingError(
            "The model needs to be retrained with the new features. "
            "Please click 'Retrain Model' in the sidebar."
        )

    model, _ = load_model()

    X = build_feature_vector(restaurant_data)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    prob_dict = {label: float(p) for label, p in zip(model.classes_, probs)}

    return {
        "grade": pred,
        "probabilities": prob_dict,
        "raw_output": probs.tolist()
    }


def predict_batch(restaurant_data_list: list) -> list:
    """
    Predict grades for multiple restaurants.

    Args:
        restaurant_data_list: List of restaurant data dicts

    Returns:
        List of prediction dicts
    """
    return [predict_restaurant_grade(data) for data in restaurant_data_list]
