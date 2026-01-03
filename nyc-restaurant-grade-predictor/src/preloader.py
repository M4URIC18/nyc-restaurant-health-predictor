"""
Application preloader for CleanKitchen NYC.

Warms all data and model caches at application startup to eliminate
cold-start latency when navigating between pages.

Call preload_all() from app.py to initialize all resources.
"""

import streamlit as st
from typing import Optional
import pandas as pd


@st.cache_data(show_spinner=False)
def preload_raw_data() -> pd.DataFrame:
    """
    Preload raw inspection data (multiple rows per restaurant).

    Uses @st.cache_data because DataFrames are serializable.
    """
    from .data_loader import load_raw_data
    return load_raw_data()


@st.cache_data(show_spinner=False)
def preload_restaurant_data() -> pd.DataFrame:
    """
    Preload feature-enriched restaurant data (one row per restaurant).

    This is the main data used by the Filter page.
    """
    from .data_loader import load_restaurant_data
    return load_restaurant_data()


@st.cache_resource(show_spinner=False)
def preload_model():
    """
    Preload ML model and metadata.

    Uses @st.cache_resource because:
    - sklearn model is not pickle-serializable for cache_data
    - Model should persist across reruns and sessions
    - Only needs to load once per server restart

    Returns:
        tuple: (model, metadata)
    """
    from .predictor import load_model
    return load_model()


@st.cache_data(show_spinner=False)
def preload_model_metadata() -> Optional[dict]:
    """
    Preload model metadata for About page.
    """
    from .trainer import load_metadata
    return load_metadata()


def preload_all(show_progress: bool = False):
    """
    Preload all application data and models.

    Call this from app.py on startup to warm all caches.

    Args:
        show_progress: If True, show a spinner during loading
    """
    if show_progress:
        with st.spinner("Initializing application..."):
            _do_preload()
    else:
        _do_preload()


def _do_preload():
    """Execute all preloading operations."""
    # 1. Preload raw data (triggers CSV read)
    preload_raw_data()

    # 2. Preload restaurant data (triggers feature engineering)
    preload_restaurant_data()

    # 3. Preload model (triggers joblib load)
    try:
        preload_model()
    except FileNotFoundError:
        # Model not trained yet - this is OK
        pass

    # 4. Preload metadata (for About page)
    preload_model_metadata()
