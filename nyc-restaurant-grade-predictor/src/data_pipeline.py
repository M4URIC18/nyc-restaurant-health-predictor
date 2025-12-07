"""
Data pipeline for fetching and cleaning NYC restaurant inspection data.

This module handles:
- Fetching data from NYC Open Data API
- Cleaning and transforming the data
- Caching to avoid repeated API calls
"""

import pandas as pd
import numpy as np
import os
import time
import requests
from io import StringIO

# =============================================================================
# Configuration
# =============================================================================

API_ENDPOINT = "https://data.cityofnewyork.us/api/v3/views/43nn-pn8j/query.csv"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 300  # 5 min timeout for large CSV download

# Cache configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "..", "data")
CACHE_FILENAME = "cleaned_restaurant_inspections.csv"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILENAME)

# Columns to drop during cleaning
DROP_COLUMNS = [
    'phone', 'action', 'record_date', 'community_board',
    'council_district', 'census_tract', 'bin', 'bbl', 'nta', 'location'
]

# Text columns to trim whitespace
TEXT_COLUMNS = ['dba', 'street', 'building', 'cuisine_description', 'violation_description']


# =============================================================================
# Token Management
# =============================================================================

def get_app_token(token=None):
    """
    Get NYC Open Data API token.

    Priority order:
    1. Directly passed token parameter
    2. NYC_OPENDATA_APP_TOKEN environment variable
    3. Streamlit secrets (for deployment)
    4. .env file in data directory (for local testing)

    Returns:
        str: API token, or None if not found
    """
    # 1. Direct token
    if token:
        return token

    # 2. Environment variable
    env_token = os.environ.get('NYC_OPENDATA_APP_TOKEN')
    if env_token:
        return env_token

    # 3. Streamlit secrets (for deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'NYC_OPENDATA_APP_TOKEN' in st.secrets:
            return st.secrets['NYC_OPENDATA_APP_TOKEN']
    except (ImportError, Exception):
        pass

    # 4. Try .env file for local testing (in project root)
    env_file = os.path.join(BASE_DIR, '..', '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NYC_OPENDATA_APP_TOKEN='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")

    return None


# =============================================================================
# API Fetching
# =============================================================================

def fetch_from_api(app_token=None):
    """
    Fetch all restaurant inspection data from NYC Open Data API.

    Args:
        app_token: Optional API token. If None, will attempt to find one.

    Returns:
        pd.DataFrame: Raw restaurant inspection data

    Raises:
        ValueError: If no API token is available
        requests.RequestException: If API request fails after retries
    """
    token = get_app_token(app_token)
    if not token:
        raise ValueError(
            "NYC Open Data API token not found.\n\n"
            "Set it using one of these methods:\n"
            "  1. Environment variable: export NYC_OPENDATA_APP_TOKEN='your_token'\n"
            "  2. Streamlit secrets: Add NYC_OPENDATA_APP_TOKEN to .streamlit/secrets.toml\n"
            "  3. Create data/.env file with: NYC_OPENDATA_APP_TOKEN=your_token\n\n"
            "Get a free token at: https://data.cityofnewyork.us/profile/edit/developer_settings"
        )

    headers = {'X-App-Token': token}
    params = {'$limit': 500000}  # Get all records

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                API_ENDPOINT,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY_SECONDS * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise

    # Parse CSV response
    df = pd.read_csv(StringIO(response.text))
    return df


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_data(df):
    """
    Clean and transform raw restaurant inspection data.

    Cleaning steps:
    1. Lowercase column names, replace spaces with underscores
    2. Drop unnecessary columns
    3. Remove placeholder dates (01/01/1900)
    4. Filter to "Cycle Inspection" only (produces grades)
    5. Convert dates to datetime
    6. Convert ZIP to 5-digit string
    7. Convert CAMIS to string
    8. Trim whitespace from text columns
    9. Filter valid NYC coordinates
    10. Remove duplicates

    Args:
        df: Raw DataFrame from API

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()

    # 1. Lowercase column names, replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # 2. Drop unnecessary columns (only those that exist)
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Remove placeholder inspection dates
    if 'inspection_date' in df.columns:
        df = df[df['inspection_date'] != '01/01/1900']

    # 4. Keep only Cycle Inspections (the only ones that produce health grades)
    if 'inspection_type' in df.columns:
        df = df[df['inspection_type'].str.contains('Cycle Inspection', case=False, na=False)]

    # 5. Convert date columns to datetime
    if 'inspection_date' in df.columns:
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    if 'grade_date' in df.columns:
        df['grade_date'] = pd.to_datetime(df['grade_date'], errors='coerce')

    # 6. Convert ZIPCODE to 5-digit string
    if 'zipcode' in df.columns:
        df['zipcode'] = pd.to_numeric(df['zipcode'], errors='coerce')
        df['zipcode'] = df['zipcode'].apply(
            lambda x: str(int(x)).zfill(5) if pd.notna(x) else None
        )

    # 7. Convert CAMIS to string (it's an ID, not a number)
    if 'camis' in df.columns:
        df['camis'] = df['camis'].astype(str)

    # 8. Trim whitespace from text columns
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 9. Filter to valid NYC coordinates
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[
            (df['latitude'] > 40) & (df['latitude'] < 42) &
            (df['longitude'] < -73) & (df['longitude'] > -75)
        ]

    # 10. Remove duplicates
    df = df.drop_duplicates()

    return df


# =============================================================================
# Main Entry Points
# =============================================================================

def get_or_fetch_data(force_refresh=False, token=None):
    """
    Get restaurant inspection data, fetching from API if needed.

    Args:
        force_refresh: If True, fetch fresh data even if cache exists
        token: Optional API token

    Returns:
        pd.DataFrame: Cleaned restaurant inspection data

    Raises:
        FileNotFoundError: If no cache exists and no API token available
    """
    # Ensure cache directory exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Check for cached data
    if not force_refresh and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, low_memory=False, dtype={'zipcode': str})

    # Try to get API token
    app_token = get_app_token(token)

    # If no token but cache exists, use cache with warning
    if not app_token and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, low_memory=False, dtype={'zipcode': str})

    # If no token and no cache, raise error
    if not app_token:
        raise FileNotFoundError(
            "No cached data found and no API token available.\n\n"
            "Either:\n"
            "  1. Set NYC_OPENDATA_APP_TOKEN environment variable\n"
            "  2. Add token to .streamlit/secrets.toml\n"
            "  3. Create data/.env file with the token\n\n"
            "Get a free token at: https://data.cityofnewyork.us/profile/edit/developer_settings"
        )

    # Fetch from API
    raw_df = fetch_from_api(app_token)

    # Clean data
    cleaned_df = clean_data(raw_df)

    # Save to cache
    cleaned_df.to_csv(CACHE_PATH, index=False)

    return cleaned_df


def refresh_data(token=None):
    """
    Force refresh data from API.

    This is called when user clicks "Fetch Fresh Data" button.

    Args:
        token: Optional API token

    Returns:
        tuple: (success: bool, message: str, record_count: int or None)
    """
    try:
        df = get_or_fetch_data(force_refresh=True, token=token)
        return True, "Data refreshed successfully!", len(df)
    except ValueError as e:
        return False, str(e), None
    except requests.RequestException as e:
        return False, f"API request failed: {e}", None
    except Exception as e:
        return False, f"Error refreshing data: {e}", None


def has_cached_data():
    """Check if cached data exists."""
    return os.path.exists(CACHE_PATH)


def get_cache_info():
    """Get information about cached data."""
    if not os.path.exists(CACHE_PATH):
        return None

    stat = os.stat(CACHE_PATH)
    return {
        'path': CACHE_PATH,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime
    }
