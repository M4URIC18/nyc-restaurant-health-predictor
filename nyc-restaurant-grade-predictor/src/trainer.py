"""
Model training pipeline for NYC restaurant health prediction.

Handles model training, evaluation, and saving.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "restaurant_grade_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# Feature configuration
# NOTE: We use CURRENT inspection data to predict NEXT grade
# - prev_score_1 = score from current inspection (used to predict next inspection)
# - prev_score_2 = score from previous inspection
# - target_grade = grade from the NEXT inspection (what we're predicting)
FEATURE_COLUMNS = [
    'borough',
    'zipcode',
    'cuisine_description',
    'days_since_last_inspection',
    'inspection_frequency',
    'prev_grade_1',
    'prev_grade_2',
    'prev_score_1',      # Score from current inspection
    'prev_score_2',      # Score from previous inspection
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
TARGET_COLUMN = 'target_grade'  # Output of compute_training_features()


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target vector y from DataFrame.

    Args:
        df: DataFrame with all features and target column

    Returns:
        tuple: (X, y, encoders) where encoders is a dict of LabelEncoders
    """
    # Filter to valid grades only
    df = df[df[TARGET_COLUMN].isin(['A', 'B', 'C'])].copy()

    if len(df) == 0:
        raise ValueError("No valid training samples with grades A, B, or C")

    # Initialize encoders
    encoders = {}

    # Encode categorical columns
    X = df[FEATURE_COLUMNS].copy()

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = X[col].astype(str).fillna('Unknown')
        X[col] = le.fit_transform(X[col])
        encoders[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}

    # Convert zipcode to numeric
    X['zipcode'] = pd.to_numeric(X['zipcode'], errors='coerce').fillna(0)

    # Fill any remaining NaN values
    X = X.fillna(0)

    # Target
    y = df[TARGET_COLUMN].values

    return X, y, encoders


def train_model(
    df: pd.DataFrame,
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Train a RandomForestClassifier on the provided data.

    Args:
        df: DataFrame with features and target
        n_estimators: Number of trees in the forest
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (model, metrics, feature_importances, encoders)
    """
    # Prepare data
    X, y, encoders = prepare_training_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1': float(f1_score(y_test, y_pred, average='weighted')),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    # Feature importances
    feature_importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))

    return model, metrics, feature_importances, encoders


def save_model(
    model,
    encoders: dict,
    metrics: dict,
    feature_importances: dict,
    model_path: str = MODEL_PATH,
    meta_path: str = META_PATH
):
    """
    Save trained model and metadata to disk.

    Args:
        model: Trained sklearn model
        encoders: Dict of encoders for categorical columns
        metrics: Dict of evaluation metrics
        feature_importances: Dict of feature importance scores
        model_path: Path to save model pickle
        meta_path: Path to save metadata JSON
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(model, model_path)

    # Build metadata
    metadata = {
        'feature_columns': FEATURE_COLUMNS,
        'categorical_columns': CATEGORICAL_COLUMNS,
        'target_column': TARGET_COLUMN,
        'output_classes': ['A', 'B', 'C'],
        'encoders': encoders,
        'training_date': datetime.now().isoformat(),
        'training_metrics': metrics,
        'feature_importances': feature_importances,
        'model_type': 'RandomForestClassifier',
        'model_params': {
            'n_estimators': model.n_estimators,
            'class_weight': 'balanced'
        }
    }

    # Save metadata
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(meta_path: str = META_PATH) -> dict:
    """
    Load model metadata from disk.

    Args:
        meta_path: Path to metadata JSON file

    Returns:
        dict: Model metadata
    """
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r') as f:
        return json.load(f)


def get_feature_importance_ranking(feature_importances: dict) -> list:
    """
    Get feature importances sorted by importance (descending).

    Args:
        feature_importances: Dict mapping feature names to importance scores

    Returns:
        list: List of (feature_name, importance) tuples sorted by importance
    """
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_features
