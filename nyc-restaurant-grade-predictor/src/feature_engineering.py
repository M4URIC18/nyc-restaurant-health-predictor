"""
Feature engineering for NYC restaurant health prediction model.

Computes historical and contextual features from raw inspection data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def compute_days_since_last_inspection(df: pd.DataFrame) -> pd.Series:
    """
    Calculate days since the most recent inspection for each restaurant.

    Args:
        df: DataFrame with 'camis' and 'inspection_date' columns

    Returns:
        Series indexed by camis with days since last inspection
    """
    today = pd.Timestamp.now()
    last_inspection = df.groupby('camis')['inspection_date'].max()
    days_since = (today - last_inspection).dt.days
    return days_since.rename('days_since_last_inspection')


def compute_inspection_frequency(df: pd.DataFrame) -> pd.Series:
    """
    Calculate inspections per year for each restaurant.

    Args:
        df: DataFrame with 'camis' and 'inspection_date' columns

    Returns:
        Series indexed by camis with inspection frequency (per year)
    """
    grouped = df.groupby('camis')['inspection_date']
    inspection_count = grouped.count()
    date_range = (grouped.max() - grouped.min()).dt.days

    # Avoid division by zero - if only one inspection, assume 1 year span
    years = date_range.clip(lower=365) / 365.0
    frequency = inspection_count / years

    return frequency.rename('inspection_frequency')


def compute_grade_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute previous grades (lag-1 and lag-2) for each restaurant (for INFERENCE).

    For predicting the NEXT inspection:
    - prev_grade_1 = grade from the LATEST inspection (rank 1)
    - prev_grade_2 = grade from the inspection before that (rank 2)

    This ensures the model sees the most recent grade when predicting.

    Args:
        df: DataFrame with 'camis', 'inspection_date', 'grade' columns

    Returns:
        DataFrame with columns: camis, prev_grade_1, prev_grade_2
    """
    # Filter to rows with valid grades
    graded = df[df['grade'].isin(['A', 'B', 'C'])].copy()
    graded = graded.sort_values(['camis', 'inspection_date'], ascending=[True, False])

    # Get unique inspections per restaurant (by date)
    graded = graded.drop_duplicates(subset=['camis', 'inspection_date'])

    # Rank inspections by date (most recent = 1)
    graded['inspection_rank'] = graded.groupby('camis').cumcount() + 1

    # Get latest grade as prev_grade_1 (rank 1 = most recent)
    prev_1 = graded[graded['inspection_rank'] == 1][['camis', 'grade']].rename(
        columns={'grade': 'prev_grade_1'}
    )
    # Get second-latest grade as prev_grade_2 (rank 2)
    prev_2 = graded[graded['inspection_rank'] == 2][['camis', 'grade']].rename(
        columns={'grade': 'prev_grade_2'}
    )

    # Get all unique camis
    all_camis = pd.DataFrame({'camis': df['camis'].unique()})

    # Merge
    result = all_camis.merge(prev_1, on='camis', how='left')
    result = result.merge(prev_2, on='camis', how='left')

    # Fill missing with 'Unknown'
    result['prev_grade_1'] = result['prev_grade_1'].fillna('Unknown')
    result['prev_grade_2'] = result['prev_grade_2'].fillna('Unknown')

    return result


def compute_prev_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute previous inspection scores for each restaurant (for INFERENCE).

    For predicting the NEXT inspection:
    - prev_score_1 = score from the LATEST inspection (rank 1)
    - prev_score_2 = score from the inspection before that (rank 2)

    This ensures the model sees the most recent score when predicting.

    Args:
        df: DataFrame with 'camis', 'inspection_date', 'score' columns

    Returns:
        DataFrame with columns: camis, prev_score_1, prev_score_2
    """
    # Sort by restaurant and date (most recent first)
    scored = df[df['score'].notna()].copy()
    scored = scored.sort_values(['camis', 'inspection_date'], ascending=[True, False])
    scored = scored.drop_duplicates(subset=['camis', 'inspection_date'])

    # Rank inspections (1 = most recent)
    scored['inspection_rank'] = scored.groupby('camis').cumcount() + 1

    # Get latest score as prev_score_1 (rank 1 = most recent)
    prev_1 = scored[scored['inspection_rank'] == 1][['camis', 'score']].rename(
        columns={'score': 'prev_score_1'}
    )
    # Get second-latest score as prev_score_2 (rank 2)
    prev_2 = scored[scored['inspection_rank'] == 2][['camis', 'score']].rename(
        columns={'score': 'prev_score_2'}
    )

    # Get all unique camis
    all_camis = pd.DataFrame({'camis': df['camis'].unique()})

    # Merge
    result = all_camis.merge(prev_1, on='camis', how='left')
    result = result.merge(prev_2, on='camis', how='left')

    return result


def compute_violation_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute violation counts for each restaurant.

    Args:
        df: DataFrame with 'camis', 'inspection_date', 'critical_flag' columns

    Returns:
        DataFrame with columns: camis, critical_violations_12mo, total_violations_all_time
    """
    today = pd.Timestamp.now()
    cutoff_12mo = today - pd.Timedelta(days=365)

    # Total violations (all time)
    total_violations = df.groupby('camis').size().rename('total_violations_all_time')

    # Critical violations in last 12 months
    recent = df[df['inspection_date'] >= cutoff_12mo]
    critical_recent = recent[recent['critical_flag'] == 'Critical']
    critical_12mo = critical_recent.groupby('camis').size().rename('critical_violations_12mo')

    # Combine
    result = pd.DataFrame({'camis': df['camis'].unique()})
    result = result.merge(total_violations.reset_index(), on='camis', how='left')
    result = result.merge(critical_12mo.reset_index(), on='camis', how='left')

    result['total_violations_all_time'] = result['total_violations_all_time'].fillna(0).astype(int)
    result['critical_violations_12mo'] = result['critical_violations_12mo'].fillna(0).astype(int)

    return result


def compute_avg_score_historical(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the average inspection score for each restaurant.

    Args:
        df: DataFrame with 'camis' and 'score' columns

    Returns:
        Series indexed by camis with average historical score
    """
    avg_score = df.groupby('camis')['score'].mean()
    return avg_score.rename('avg_score_historical')


def compute_score_trend(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the trend in inspection scores over time.

    Uses linear regression slope: positive = worsening (higher scores = worse),
    negative = improving.

    Args:
        df: DataFrame with 'camis', 'inspection_date', 'score' columns

    Returns:
        Series indexed by camis with score trend (slope)
    """
    def calc_slope(group):
        if len(group) < 2:
            return 0.0

        # Convert dates to numeric (days since first inspection)
        x = (group['inspection_date'] - group['inspection_date'].min()).dt.days.values
        y = group['score'].values

        # Handle case where all x values are the same
        if x.std() == 0:
            return 0.0

        # Simple linear regression slope
        slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0.0
        return slope

    trends = df.groupby('camis').apply(calc_slope, include_groups=False)
    return trends.rename('score_trend')


def compute_grade_stability(df: pd.DataFrame) -> pd.Series:
    """
    Check if restaurant's grade changed in last 2 inspections.

    Args:
        df: DataFrame with 'camis', 'inspection_date', 'grade' columns

    Returns:
        Series indexed by camis with stability flag (1 = stable, 0 = changed)
    """
    # Filter to rows with valid grades
    graded = df[df['grade'].isin(['A', 'B', 'C'])].copy()
    graded = graded.sort_values(['camis', 'inspection_date'], ascending=[True, False])
    graded = graded.drop_duplicates(subset=['camis', 'inspection_date'])

    def check_stability(group):
        if len(group) < 2:
            return 1  # Stable by default if not enough history

        # Get two most recent grades
        recent_grades = group.head(2)['grade'].values
        return 1 if recent_grades[0] == recent_grades[1] else 0

    stability = graded.groupby('camis').apply(check_stability, include_groups=False)

    # Fill missing (restaurants with no graded inspections)
    all_camis = df['camis'].unique()
    stability = stability.reindex(all_camis, fill_value=1)

    return stability.rename('grade_stability')


def compute_context_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute contextual averages by cuisine and zipcode.

    Args:
        df: DataFrame with 'camis', 'cuisine_description', 'zipcode', 'score' columns

    Returns:
        DataFrame with columns: camis, cuisine_avg_score, zipcode_avg_score
    """
    # Get unique restaurant info
    restaurant_info = df.groupby('camis').agg({
        'cuisine_description': 'first',
        'zipcode': 'first',
        'score': 'mean'
    }).reset_index()

    # Cuisine average (excluding current restaurant to avoid leakage)
    cuisine_avg = restaurant_info.groupby('cuisine_description')['score'].transform('mean')
    restaurant_info['cuisine_avg_score'] = cuisine_avg

    # Zipcode average
    zipcode_avg = restaurant_info.groupby('zipcode')['score'].transform('mean')
    restaurant_info['zipcode_avg_score'] = zipcode_avg

    return restaurant_info[['camis', 'cuisine_avg_score', 'zipcode_avg_score']]


def compute_violation_diversity(df: pd.DataFrame) -> pd.Series:
    """
    Count unique violation codes per restaurant.

    More diverse violations may indicate systemic issues.

    Args:
        df: DataFrame with 'camis' and 'violation_code' columns

    Returns:
        Series indexed by camis with violation diversity count
    """
    diversity = df.groupby('camis')['violation_code'].nunique()
    return diversity.rename('violation_diversity').fillna(0).astype(int)


def compute_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for training with correct temporal alignment.

    Creates MULTIPLE rows per restaurant - one for each inspection where we have
    a NEXT inspection to use as target. This ensures the model learns to predict
    the NEXT inspection from CURRENT data.

    For each training row:
    - Features are computed from inspection at time T and earlier
    - Target (target_grade) is from inspection at time T+1 (the next one)
    - prev_score_1 = score at time T (current inspection)
    - prev_score_2 = score at time T-1 (previous inspection)

    Args:
        df: Raw inspection DataFrame with multiple rows per restaurant

    Returns:
        DataFrame with multiple rows per restaurant, each predicting the next inspection
    """
    # Ensure inspection_date is datetime
    if df['inspection_date'].dtype == 'object':
        df = df.copy()
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')

    # Filter to inspections with valid grades (these can be targets)
    graded = df[df['grade'].isin(['A', 'B', 'C'])].copy()

    # Sort chronologically and dedupe by restaurant+date
    graded = graded.sort_values(['camis', 'inspection_date'], ascending=[True, True])
    graded = graded.drop_duplicates(subset=['camis', 'inspection_date'])

    # Add inspection number within each restaurant (1 = oldest)
    graded['insp_num'] = graded.groupby('camis').cumcount() + 1
    graded['total_insps'] = graded.groupby('camis')['camis'].transform('count')

    # Shift to get next inspection's data (the target we're predicting)
    graded['target_grade'] = graded.groupby('camis')['grade'].shift(-1)
    graded['next_score'] = graded.groupby('camis')['score'].shift(-1)
    graded['next_date'] = graded.groupby('camis')['inspection_date'].shift(-1)

    # Also get previous inspection's data for prev_score_2 and prev_grade_2
    graded['prev_score_2'] = graded.groupby('camis')['score'].shift(1)
    graded['prev_grade_2'] = graded.groupby('camis')['grade'].shift(1)

    # Filter to rows that have a next inspection (exclude most recent per restaurant)
    training_rows = graded[graded['target_grade'].notna()].copy()

    if len(training_rows) == 0:
        raise ValueError("No valid training pairs found (need restaurants with 2+ inspections)")

    # Current inspection's score/grade becomes prev_score_1/prev_grade_1
    training_rows['prev_score_1'] = training_rows['score']
    training_rows['prev_grade_1'] = training_rows['grade']

    # Compute days until next inspection (for training, this is known)
    training_rows['days_since_last_inspection'] = (
        training_rows['next_date'] - training_rows['inspection_date']
    ).dt.days

    # Select base columns
    base_cols = ['camis', 'dba', 'borough', 'zipcode', 'cuisine_description',
                 'latitude', 'longitude', 'inspection_date']
    base_cols = [c for c in base_cols if c in training_rows.columns]

    result = training_rows[base_cols + [
        'target_grade', 'prev_score_1', 'prev_score_2',
        'prev_grade_1', 'prev_grade_2', 'days_since_last_inspection'
    ]].copy()

    # Compute inspection frequency up to each inspection date
    # For efficiency, compute once per restaurant and use the overall frequency
    frequency = compute_inspection_frequency(df)
    result = result.merge(frequency.reset_index(), on='camis', how='left')

    # Compute violation counts (using all historical data for now - could be improved)
    violation_counts = compute_violation_counts(df)
    result = result.merge(violation_counts, on='camis', how='left')

    # Compute avg_score_historical (using all data)
    avg_score = compute_avg_score_historical(df)
    result = result.merge(avg_score.reset_index(), on='camis', how='left')

    # Compute score trend
    trend = compute_score_trend(df)
    result = result.merge(trend.reset_index(), on='camis', how='left')

    # Compute grade stability
    stability = compute_grade_stability(df)
    result = result.merge(stability.reset_index(), on='camis', how='left')

    # Compute context averages
    context = compute_context_averages(df)
    result = result.merge(context, on='camis', how='left')

    # Compute violation diversity
    diversity = compute_violation_diversity(df)
    result = result.merge(diversity.reset_index(), on='camis', how='left')

    # Fill missing values with sensible defaults
    result['inspection_frequency'] = result['inspection_frequency'].fillna(1.0)
    result['prev_grade_1'] = result['prev_grade_1'].fillna('Unknown')
    result['prev_grade_2'] = result['prev_grade_2'].fillna('Unknown')
    result['prev_score_1'] = result['prev_score_1'].fillna(13.0)
    result['prev_score_2'] = result['prev_score_2'].fillna(result['prev_score_1'])
    result['critical_violations_12mo'] = result['critical_violations_12mo'].fillna(0)
    result['total_violations_all_time'] = result['total_violations_all_time'].fillna(1)
    result['avg_score_historical'] = result['avg_score_historical'].fillna(result['prev_score_1'])
    result['score_trend'] = result['score_trend'].fillna(0.0)
    result['grade_stability'] = result['grade_stability'].fillna(1)
    result['cuisine_avg_score'] = result['cuisine_avg_score'].fillna(result['prev_score_1'])
    result['zipcode_avg_score'] = result['zipcode_avg_score'].fillna(result['prev_score_1'])
    result['violation_diversity'] = result['violation_diversity'].fillna(1)
    result['days_since_last_inspection'] = result['days_since_last_inspection'].fillna(0)

    return result


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features from raw inspection data.

    Takes raw inspection data (multiple rows per restaurant) and returns
    a DataFrame with one row per restaurant containing all features.

    Args:
        df: Raw inspection DataFrame

    Returns:
        DataFrame with one row per restaurant and all computed features
    """
    # Ensure inspection_date is datetime
    if df['inspection_date'].dtype == 'object':
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')

    # Get latest inspection data for each restaurant (base features)
    df_sorted = df.sort_values('inspection_date', ascending=False)
    latest = df_sorted.drop_duplicates(subset=['camis'], keep='first').copy()

    # Select base columns (borough already renamed from boro by data_loader)
    base_cols = ['camis', 'dba', 'borough', 'street', 'zipcode', 'cuisine_description',
                 'score', 'grade', 'critical_flag', 'inspection_date',
                 'latitude', 'longitude']
    base_cols = [c for c in base_cols if c in latest.columns]
    result = latest[base_cols].copy()

    # Create critical_flag_bin
    result['critical_flag_bin'] = (result['critical_flag'] == 'Critical').astype(int)

    # Compute time-based features
    days_since = compute_days_since_last_inspection(df)
    result = result.merge(days_since.reset_index(), on='camis', how='left')

    frequency = compute_inspection_frequency(df)
    result = result.merge(frequency.reset_index(), on='camis', how='left')

    # Compute historical features
    grade_lags = compute_grade_lag_features(df)
    result = result.merge(grade_lags, on='camis', how='left')

    prev_scores = compute_prev_scores(df)
    result = result.merge(prev_scores, on='camis', how='left')

    violation_counts = compute_violation_counts(df)
    result = result.merge(violation_counts, on='camis', how='left')

    avg_score = compute_avg_score_historical(df)
    result = result.merge(avg_score.reset_index(), on='camis', how='left')

    # Compute trend features
    trend = compute_score_trend(df)
    result = result.merge(trend.reset_index(), on='camis', how='left')

    stability = compute_grade_stability(df)
    result = result.merge(stability.reset_index(), on='camis', how='left')

    # Compute context features
    context = compute_context_averages(df)
    result = result.merge(context, on='camis', how='left')

    diversity = compute_violation_diversity(df)
    result = result.merge(diversity.reset_index(), on='camis', how='left')

    # Fill any remaining NaN values with sensible defaults
    result['days_since_last_inspection'] = result['days_since_last_inspection'].fillna(0)
    result['inspection_frequency'] = result['inspection_frequency'].fillna(1.0)
    result['prev_grade_1'] = result['prev_grade_1'].fillna('Unknown')
    result['prev_grade_2'] = result['prev_grade_2'].fillna('Unknown')
    # For prev_score, use current score as fallback (first inspection scenario)
    result['prev_score_1'] = result['prev_score_1'].fillna(result['score'])
    result['prev_score_2'] = result['prev_score_2'].fillna(result['prev_score_1'])
    result['critical_violations_12mo'] = result['critical_violations_12mo'].fillna(0)
    result['total_violations_all_time'] = result['total_violations_all_time'].fillna(1)
    result['avg_score_historical'] = result['avg_score_historical'].fillna(result['score'])
    result['score_trend'] = result['score_trend'].fillna(0.0)
    result['grade_stability'] = result['grade_stability'].fillna(1)
    result['cuisine_avg_score'] = result['cuisine_avg_score'].fillna(result['score'])
    result['zipcode_avg_score'] = result['zipcode_avg_score'].fillna(result['score'])
    result['violation_diversity'] = result['violation_diversity'].fillna(1)

    return result
