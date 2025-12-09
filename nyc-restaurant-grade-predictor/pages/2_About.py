"""
About Page - Model documentation with dynamically loaded statistics.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.components import load_css, render_top_nav, render_header_divider
from src.trainer import load_metadata

# --- Page Configuration ---
st.set_page_config(
    page_title="About - CleanKitchen NYC",
    layout="wide",
    initial_sidebar_state="collapsed"
)
load_css()

# --- Load Model Metadata ---
metadata = load_metadata()

# --- Top Navigation ---
render_top_nav()

# --- Page Content ---
st.markdown('<div class="newspaper-body">', unsafe_allow_html=True)

# -------------------------------------------------
#  Project Goal
# -------------------------------------------------
st.markdown('<div class="column-clear"></div>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 1.75rem;">Project Goal</h2>', unsafe_allow_html=True)

st.markdown(
    """
    Each year, the **New York City Health Department** inspects roughly 24,000 restaurants
    and evaluates them on food handling, temperature control, hygiene, and vermin management.
    The city assigns each restaurant a letter grade (**A, B, or C**), with **A** being the
    best score. Restaurants must display this grade at their entrance.

    This project uses **machine learning** to predict a restaurant's next inspection grade
    based on its historical inspection data, violation patterns, and contextual factors.
    """
)

# -------------------------------------------------
#  Data Source
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Data Source</h2>', unsafe_allow_html=True)
st.markdown(
    """
    NYC publishes all restaurant inspection results through **NYC Open Data**. The dataset includes:

    * Restaurant name, address, borough, and ZIP code
    * Cuisine type
    * Inspection dates and scores
    * Violation codes and severity (critical vs. non-critical)
    * Assigned grades (A, B, C)

    Data is fetched via the NYC Open Data API and cached locally for performance.
    """
)

# -------------------------------------------------
#  Model Approach
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Model Approach</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Algorithm**: Random Forest Classifier

        Random Forest is an ensemble method that builds multiple decision trees and combines
        their predictions. It handles mixed feature types well and provides feature importance rankings.
        """
    )

with col2:
    st.markdown(
        """
        **Configuration**:
        * 100 decision trees
        * Balanced class weights (handles grade imbalance)
        * 80/20 train/test split (stratified)
        """
    )

st.markdown(
    """
    **Temporal Alignment**: The model is trained to predict a restaurant's *next* inspection grade
    based on data available at the time of the *current* inspection. This prevents data leakage
    and ensures realistic predictions.
    """
)

# -------------------------------------------------
#  Features
# -------------------------------------------------
st.markdown('<h2 style="font-size: 1.75rem;">Features (17 Total)</h2>', unsafe_allow_html=True)

st.markdown("The model uses 17 engineered features across five categories:")

feature_data = {
    "Category": [
        "Location", "Location",
        "Restaurant",
        "Inspection History", "Inspection History", "Inspection History",
        "Inspection History", "Inspection History", "Inspection History",
        "Violation History", "Violation History", "Violation History",
        "Trends & Context", "Trends & Context", "Trends & Context",
        "Trends & Context", "Trends & Context"
    ],
    "Feature": [
        "borough", "zipcode",
        "cuisine_description",
        "prev_grade_1", "prev_grade_2", "prev_score_1",
        "prev_score_2", "days_since_last_inspection", "inspection_frequency",
        "critical_violations_12mo", "total_violations_all_time", "violation_diversity",
        "avg_score_historical", "score_trend", "grade_stability",
        "cuisine_avg_score", "zipcode_avg_score"
    ],
    "Description": [
        "NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)",
        "5-digit ZIP code",
        "Type of cuisine served",
        "Grade from most recent previous inspection",
        "Grade from two inspections ago",
        "Score from most recent previous inspection",
        "Score from two inspections ago",
        "Days between inspections",
        "Average inspections per year",
        "Critical violations in the past 12 months",
        "Total violations across all inspections",
        "Count of unique violation types",
        "Average inspection score over restaurant's history",
        "Score trajectory over time (positive = worsening)",
        "Whether grade changed between last two inspections",
        "Average score for restaurants of this cuisine type",
        "Average score for restaurants in this ZIP code"
    ]
}

df_features = pd.DataFrame(feature_data)
st.dataframe(
    df_features,
    width='stretch',
    hide_index=True,
    column_config={
        "Category": st.column_config.TextColumn("Category", width="small"),
        "Feature": st.column_config.TextColumn("Feature", width="medium"),
        "Description": st.column_config.TextColumn("Description", width="large"),
    }
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
#  Model Performance (Dynamic)
# -------------------------------------------------
st.markdown("---")
st.markdown('<h2 style="font-size: 1.75rem;">Model Performance</h2>', unsafe_allow_html=True)

if metadata:
    metrics = metadata.get('training_metrics', {})

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")

    # Training info
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        train_samples = metrics.get('train_samples', 'N/A')
        test_samples = metrics.get('test_samples', 'N/A')
        st.markdown(f"**Training samples**: {train_samples:,}" if isinstance(train_samples, int) else f"**Training samples**: {train_samples}")
        st.markdown(f"**Test samples**: {test_samples:,}" if isinstance(test_samples, int) else f"**Test samples**: {test_samples}")

    with col_info2:
        training_date = metadata.get('training_date', 'Unknown')
        if training_date != 'Unknown':
            # Format the ISO date nicely
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(training_date)
                training_date = dt.strftime("%B %d, %Y at %I:%M %p")
            except:
                pass
        st.markdown(f"**Last trained**: {training_date}")
        st.markdown(f"**Model type**: {metadata.get('model_type', 'Unknown')}")

    # Feature Importance Chart
    st.markdown("---")
    st.subheader("What Drives Predictions?")

    importances = metadata.get('feature_importances', {})
    if importances:
        # Human-readable names and categories
        feature_info = {
            'grade_stability': ('Grade Consistency', 'History'),
            'avg_score_historical': ('Historical Avg Score', 'History'),
            'total_violations_all_time': ('Total Violations', 'Violations'),
            'inspection_frequency': ('Inspection Frequency', 'History'),
            'prev_score_1': ('Last Inspection Score', 'History'),
            'days_since_last_inspection': ('Days Since Inspection', 'History'),
            'score_trend': ('Score Trend', 'History'),
            'violation_diversity': ('Violation Types', 'Violations'),
            'critical_violations_12mo': ('Critical Violations (1yr)', 'Violations'),
            'prev_score_2': ('Prior Inspection Score', 'History'),
            'prev_grade_1': ('Last Grade', 'History'),
            'zipcode_avg_score': ('Neighborhood Avg', 'Location'),
            'zipcode': ('ZIP Code', 'Location'),
            'cuisine_avg_score': ('Cuisine Type Avg', 'Restaurant'),
            'cuisine_description': ('Cuisine Type', 'Restaurant'),
            'prev_grade_2': ('Prior Grade', 'History'),
            'borough': ('Borough', 'Location'),
        }

        # Category colors
        cat_colors = {'History': '#4A90D9', 'Violations': '#D94A4A', 'Location': '#6B8E23', 'Restaurant': '#9B59B6'}

        # Calculate percentages and sort
        total = sum(importances.values())
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        # Compact two-column layout for top 10
        col1, col2 = st.columns(2)
        for i, (feature, importance) in enumerate(sorted_items[:10]):
            name, cat = feature_info.get(feature, (feature, 'Other'))
            pct = (importance / total) * 100
            color = cat_colors.get(cat, '#888')

            with col1 if i % 2 == 0 else col2:
                st.markdown(
                    f"<div style='margin-bottom: 8px;'>"
                    f"<span style='font-weight: 600;'>{name}</span> "
                    f"<span style='color: {color}; font-size: 0.75em;'>({cat})</span> "
                    f"<span style='float: right; font-weight: 700;'>{pct:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.progress(pct / 100)

        # Key insight - compact
        st.caption(
            "The model relies most on track record (grade consistency, historical scores, violations) "
            "rather than location or cuisine type."
        )

else:
    st.info(
        """
        **No model has been trained yet.**

        Train a model on the **Filter** page to see performance metrics and feature importance rankings.

        Once trained, this section will display:
        * Accuracy, Precision, Recall, and F1 Score
        * Training and test sample counts
        * Feature importance visualization
        """
    )

# -------------------------------------------------
#  Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Data source: NYC Open Data - DOHMH New York City Restaurant Inspection Results")
