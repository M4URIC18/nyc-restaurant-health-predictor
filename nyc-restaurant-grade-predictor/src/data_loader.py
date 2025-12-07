"""
Data loading module for NYC restaurant health prediction app.

Provides functions to load raw data and feature-enriched data with Streamlit caching.
"""

import pandas as pd
import streamlit as st

from .data_pipeline import get_or_fetch_data, refresh_data, has_cached_data, get_cache_info
from .feature_engineering import compute_all_features


# -------------------------------------------------
# Raw data loading (multiple rows per restaurant)
# -------------------------------------------------

@st.cache_data
def load_raw_data():
    """
    Load raw inspection data with all historical records.

    Returns DataFrame with multiple rows per restaurant (one per inspection).
    Used for training the model.
    """
    df = get_or_fetch_data()

    # Rename 'boro' to 'borough' for consistency
    if 'boro' in df.columns and 'borough' not in df.columns:
        df = df.rename(columns={'boro': 'borough'})

    # Clean/standardize core fields
    df['borough'] = df['borough'].astype(str).str.strip().str.title()
    df['cuisine_description'] = df['cuisine_description'].astype(str).str.strip().str.title()

    # Clean zipcode
    if df['zipcode'].dtype == 'object':
        df['zipcode'] = df['zipcode'].str.strip().replace('', None)

    # Create critical_flag_bin
    if 'critical_flag_bin' not in df.columns and 'critical_flag' in df.columns:
        df['critical_flag_bin'] = (df['critical_flag'].str.lower() == 'critical').astype(int)

    # Ensure inspection_date is datetime
    if 'inspection_date' in df.columns and df['inspection_date'].dtype == 'object':
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')

    return df


# -------------------------------------------------
# Feature-enriched data (one row per restaurant)
# -------------------------------------------------

@st.cache_data
def load_restaurant_data():
    """
    Load feature-enriched restaurant data.

    Returns DataFrame with one row per restaurant, including all computed features.
    Used for display and predictions.
    """
    # Get raw data
    raw_df = load_raw_data()

    # Compute all features
    enriched_df = compute_all_features(raw_df)

    return enriched_df


# -------------------------------------------------
# Training data loader
# -------------------------------------------------

def load_training_data():
    """
    Load data prepared for model training.

    Returns raw data without Streamlit caching (for training pipeline).
    """
    df = get_or_fetch_data()

    # Rename 'boro' to 'borough' for consistency
    if 'boro' in df.columns and 'borough' not in df.columns:
        df = df.rename(columns={'boro': 'borough'})

    # Clean/standardize core fields
    df['borough'] = df['borough'].astype(str).str.strip().str.title()
    df['cuisine_description'] = df['cuisine_description'].astype(str).str.strip().str.title()

    # Clean zipcode
    if df['zipcode'].dtype == 'object':
        df['zipcode'] = df['zipcode'].str.strip().replace('', None)

    # Create critical_flag_bin
    if 'critical_flag_bin' not in df.columns and 'critical_flag' in df.columns:
        df['critical_flag_bin'] = (df['critical_flag'].str.lower() == 'critical').astype(int)

    # Ensure inspection_date is datetime
    if 'inspection_date' in df.columns and df['inspection_date'].dtype == 'object':
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')

    return df


# -------------------------------------------------
# Public functions
# -------------------------------------------------

def get_data():
    """Returns feature-enriched restaurant data (one row per restaurant)."""
    return load_restaurant_data()


def get_raw_data():
    """Returns raw inspection data (multiple rows per restaurant)."""
    return load_raw_data()


def clear_data_cache():
    """Clear all Streamlit data caches."""
    st.cache_data.clear()
