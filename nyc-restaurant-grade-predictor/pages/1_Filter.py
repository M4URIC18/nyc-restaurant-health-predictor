"""
Filter & Predictor Page - Main functionality for CleanKitchen NYC.

Includes:
- Restaurant filtering (borough, ZIP, cuisine)
- Interactive map visualization
- Grade prediction interface
- Model management
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.components import load_css, render_top_nav, render_header_divider
from src.data_loader import get_data, get_raw_data, refresh_data, load_training_data, clear_data_cache
from src.predictor import predict_restaurant_grade, clear_model_cache, get_model_metadata, model_needs_retraining, ModelNeedsRetrainingError
from src.feature_engineering import compute_training_features
from src.trainer import train_model, save_model, get_feature_importance_ranking
from src.utils import (
    get_grade_color,
    format_probabilities,
    row_to_model_input,
    display_value,
    prepare_map_dataframe,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Filter & Predict - CleanKitchen NYC",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

# --- Top Navigation ---
render_top_nav()
render_header_divider()

# --- Load Data ---
@st.cache_data(show_spinner="Loading NYC restaurant data...")
def load_app_data():
    """Load feature-enriched restaurant data (one row per restaurant)."""
    df = get_data()
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    df["cuisine_description"] = df["cuisine_description"].astype(str).str.strip().str.title()
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    return df

df = load_app_data()

if df.empty:
    st.error("No data loaded. Please check your CSV files in the data/ folder.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filter Restaurants")

# Borough filter
boroughs = ["All"] + sorted(df["borough"].dropna().unique().tolist())
borough_choice = st.sidebar.selectbox("Borough", boroughs, index=0)

# ZIP filter (depends on borough choice)
if borough_choice != "All":
    zip_candidates = df.loc[df["borough"] == borough_choice, "zipcode"].unique()
else:
    zip_candidates = df["zipcode"].unique()

zips = ["All"] + sorted([z for z in zip_candidates if pd.notna(z)])
zip_choice = st.sidebar.selectbox("ZIP code", zips, index=0)

# Cuisine filter
cuisine_list = sorted(df["cuisine_description"].dropna().unique().tolist())
cuisine_choice = st.sidebar.multiselect(
    "Cuisine type",
    options=cuisine_list,
    default=[]
)

# Apply filters
df_filtered = df.copy()

if borough_choice != "All":
    df_filtered = df_filtered[df_filtered["borough"] == borough_choice]

if zip_choice != "All":
    df_filtered = df_filtered[df_filtered["zipcode"] == zip_choice]

if cuisine_choice:
    df_filtered = df_filtered[df_filtered["cuisine_description"].isin(cuisine_choice)]

# Reset map selection when filters change
current_filters = (borough_choice, zip_choice, tuple(cuisine_choice))
if 'prev_filters' not in st.session_state:
    st.session_state.prev_filters = current_filters
elif st.session_state.prev_filters != current_filters:
    st.session_state.prev_filters = current_filters
    if 'selected_camis' in st.session_state:
        del st.session_state.selected_camis
    if 'restaurant_selector' in st.session_state:
        del st.session_state.restaurant_selector

st.sidebar.markdown(f"""
<div class="results-counter">
    <p>{len(df_filtered):,} restaurants found</p>
</div>
""", unsafe_allow_html=True)

# --- Data Management (sidebar) ---
st.sidebar.divider()
st.sidebar.caption("Data Management")
if st.sidebar.button("Fetch Fresh Data"):
    with st.spinner("Fetching latest data from NYC Open Data API... This may take 1-2 minutes."):
        success, message, count = refresh_data()
        if success:
            st.sidebar.success(f"{message} ({count:,} records)")
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(message)

# --- Model Management (sidebar) ---
st.sidebar.divider()
st.sidebar.caption("Model Management")

# Show current model info
try:
    metadata = get_model_metadata()
    training_date = metadata.get("training_date", "Unknown")
    if training_date != "Unknown":
        try:
            dt = datetime.fromisoformat(training_date)
            training_date = dt.strftime("%Y-%m-%d %H:%M")
        except:
            pass
    metrics = metadata.get("training_metrics", {})
    accuracy = metrics.get("accuracy")
    if accuracy:
        st.sidebar.markdown(f"**Last trained:** {training_date}")
        st.sidebar.markdown(f"**Accuracy:** {accuracy:.1%}")
    else:
        st.sidebar.markdown("**Model:** Not yet trained with new features")
except Exception:
    st.sidebar.markdown("**Model:** Ready for training")

# Training button
if st.sidebar.button("Retrain Model"):
    with st.sidebar:
        progress_bar = st.progress(0, text="Loading training data...")

        try:
            # Step 1: Load raw data
            progress_bar.progress(10, text="Loading raw inspection data...")
            raw_df = load_training_data()

            # Step 2: Compute features using training-specific function
            progress_bar.progress(30, text="Computing training features...")
            feature_df = compute_training_features(raw_df)

            # Step 3: Train model
            progress_bar.progress(50, text="Training model...")
            model, metrics, feature_importances, encoders = train_model(feature_df)

            # Step 4: Save model
            progress_bar.progress(80, text="Saving model...")
            save_model(model, encoders, metrics, feature_importances)

            # Step 5: Clear caches
            progress_bar.progress(90, text="Clearing caches...")
            clear_model_cache()
            clear_data_cache()

            progress_bar.progress(100, text="Complete!")

            # Show results
            st.success("Model trained successfully!")
            st.markdown(f"""
            **Training Results:**
            - Accuracy: {metrics['accuracy']:.1%}
            - Precision: {metrics['precision']:.1%}
            - Recall: {metrics['recall']:.1%}
            - F1 Score: {metrics['f1']:.1%}
            - Training samples: {metrics['train_samples']:,}
            """)

            # Show top feature importances
            st.markdown("**Top Features:**")
            top_features = get_feature_importance_ranking(feature_importances)[:5]
            for name, importance in top_features:
                st.markdown(f"- {name}: {importance:.3f}")

        except Exception as e:
            st.error(f"Training failed: {e}")

st.sidebar.caption("Training takes ~30 seconds")

# --- MAIN LAYOUT ---
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Map of Restaurants")

    if len(df_filtered) == 0:
        st.info("No restaurants match your filters. Try changing the filters.")
    else:
        # Prepare data for PyDeck
        map_df = prepare_map_dataframe(df_filtered)

        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        # Adaptive zoom based on data spread
        lat_range = map_df["latitude"].max() - map_df["latitude"].min()
        lon_range = map_df["longitude"].max() - map_df["longitude"].min()
        max_range = max(lat_range, lon_range)
        zoom = 15 if max_range < 0.01 else 13 if max_range < 0.05 else 12 if max_range < 0.1 else 11

        layer = pdk.Layer(
            "ScatterplotLayer",
            id="restaurants",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=8,
            radius_min_pixels=4,
            radius_max_pixels=8,
            radius_scale=1,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,
        )

        tooltip = {
            "html": "<b>{name}</b><br/>Cuisine: {cuisine_description}<br/>Borough: {borough}<br/>ZIP: {zipcode}<br/>Score: {score_display}<br/>Grade: <b>{grade_display}</b>",
            "style": {"backgroundColor": "#FFFFFF", "color": "#2C3E50", "fontSize": "14px", "padding": "12px", "borderRadius": "6px", "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"}
        }

        event = st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
            ),
            height=500,
            on_select="rerun",
            selection_mode="single-object",
            key="restaurant_map"
        )

        # Handle map click selection
        if event.selection and event.selection.get("objects", {}).get("restaurants"):
            selected_objects = event.selection["objects"]["restaurants"]
            if selected_objects:
                clicked_camis = selected_objects[0].get("camis")
                if clicked_camis and st.session_state.get('selected_camis') != clicked_camis:
                    st.session_state.selected_camis = clicked_camis
                    # Update the selectbox value directly (it uses CAMIS as its value)
                    st.session_state.restaurant_selector = clicked_camis
                    st.rerun()

        # Grade legend
        st.markdown("""
        <div style="display: flex; gap: 20px; font-size: 16px; margin-top: 10px; flex-wrap: wrap;">
            <span><span style="color: #7DB87D; font-size: 20px;">●</span> A</span>
            <span><span style="color: #E8C84A; font-size: 20px;">●</span> B</span>
            <span><span style="color: #8B3A3A; font-size: 20px;">●</span> C</span>
            <span><span style="color: #9BA8C4; font-size: 20px;">●</span> Pending</span>
            <span><span style="color: #D4956A; font-size: 20px;">●</span> N/A</span>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Showing all {len(map_df):,} restaurants on map.")

    st.subheader("Restaurant List")
    st.caption("Filtered view based on your selections in the sidebar.")

    # Show a simpler table
    cols_to_show = [
        c for c in ["DBA", "dba", "borough", "zipcode",
                    "cuisine_description", "score", "grade", "inspection_date"]
        if c in df_filtered.columns
    ]
    if cols_to_show:
        st.dataframe(
            df_filtered[cols_to_show].head(300),
        )
    else:
        st.dataframe(df_filtered.head(300))


with right_col:
    st.subheader("Restaurant Details")

    if len(df_filtered) == 0:
        st.info("Use the filters to select at least one restaurant.")
    else:
        # Let user pick a restaurant from a dropdown
        if "dba" in df_filtered.columns:
            name_col = "dba"
        elif "DBA" in df_filtered.columns:
            name_col = "DBA"
        else:
            name_col = df_filtered.columns[0]

        df_filtered = df_filtered.reset_index(drop=True)

        # Build labels dictionary keyed by CAMIS for stable identification
        def make_label(row):
            name = row[name_col]
            street = row.get('street')
            zipcode = row.get('zipcode')
            borough = row.get('borough')

            if pd.notna(street) and street not in ('nan', ''):
                return f"{name} ({street}, {zipcode})"
            else:
                return f"{name} ({borough}, {zipcode})"

        camis_list = df_filtered['camis'].tolist()
        labels_dict = df_filtered.set_index('camis').apply(make_label, axis=1).to_dict()

        # Initialize selectbox state if not set or if current value not in filtered list
        current_selection = st.session_state.get('restaurant_selector')
        if current_selection not in camis_list:
            # Use selected_camis from map click if valid, otherwise first in list
            if st.session_state.get('selected_camis') in camis_list:
                st.session_state.restaurant_selector = st.session_state.selected_camis
            elif camis_list:
                st.session_state.restaurant_selector = camis_list[0]

        selected_camis = st.selectbox(
            "Choose a restaurant to analyze:",
            options=camis_list,
            format_func=lambda c: labels_dict.get(c, str(c)),
            key="restaurant_selector"
        )

        selected_row = df_filtered[df_filtered['camis'] == selected_camis].iloc[0]

        # Get values
        restaurant_name = selected_row.get(name_col, 'N/A')
        borough = selected_row.get('borough', 'N/A')
        zipcode = selected_row.get('zipcode', 'N/A')
        cuisine = selected_row.get('cuisine_description', 'N/A')
        score = display_value(selected_row.get('score'), 'N/A')
        grade = display_value(selected_row.get('grade'), 'N/A')
        grade_color = get_grade_color(grade if grade != 'N/A' else 'Z')

        # Format inspection date nicely
        inspection_date = selected_row.get('inspection_date')
        if pd.notna(inspection_date):
            try:
                inspection_date_str = pd.to_datetime(inspection_date).strftime('%b %d, %Y')
            except:
                inspection_date_str = str(inspection_date)
        else:
            inspection_date_str = 'N/A'

        # Restaurant info card (compact)
        st.markdown(f"""
        <div class="info-card" style="padding: 12px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.1rem; color: #2C3E50;">{restaurant_name}</h3>
            <div style="font-size: 0.85rem; color: #6C757D;">{borough} &bull; {zipcode} &bull; {cuisine}</div>
        </div>
        """, unsafe_allow_html=True)

        # Get camis for this restaurant
        camis = selected_row.get('camis')

        # Check if model needs retraining
        if model_needs_retraining():
            st.warning(
                "**Model needs to be trained.** Please click **'Retrain Model'** in the sidebar."
            )

        # Predict button at top
        if st.button("Predict Next Inspection", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    model_input = row_to_model_input(selected_row)
                    result = predict_restaurant_grade(model_input)
                    # Calculate time estimate
                    days_since = selected_row.get('days_since_last_inspection', 0)
                    if pd.isna(days_since):
                        days_since = 0
                    median_interval = 124
                    if days_since > 365:
                        time_estimate = f"Last inspected {round(days_since / 365)}+ years ago"
                    elif days_since > median_interval:
                        time_estimate = "Overdue for inspection"
                    elif days_since > median_interval - 30:
                        time_estimate = "Due within ~1 month"
                    elif days_since > median_interval - 60:
                        time_estimate = "Expected in ~2 months"
                    else:
                        time_estimate = f"Expected in ~{round((median_interval - days_since) / 30)} months"
                    # Store in session state
                    st.session_state.prediction_result = {
                        'camis': camis,
                        'grade': result["grade"],
                        'probabilities': result["probabilities"],
                        'time_estimate': time_estimate
                    }
                except ModelNeedsRetrainingError:
                    st.error("**Model needs retraining.** Click 'Retrain Model' in sidebar.")
                    if 'prediction_result' in st.session_state:
                        del st.session_state.prediction_result
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    if 'prediction_result' in st.session_state:
                        del st.session_state.prediction_result

        # Fetch inspection history data
        history_items = []
        if camis:
            raw_df = get_raw_data()
            history = raw_df[raw_df['camis'] == camis].copy()
            history = history.sort_values('inspection_date', ascending=False)
            history = history.drop_duplicates(subset=['inspection_date']).head(5)

            for _, insp in history.iterrows():
                insp_date = insp.get('inspection_date')
                if pd.notna(insp_date):
                    try:
                        date_str = pd.to_datetime(insp_date).strftime('%b %Y')
                    except:
                        date_str = str(insp_date)
                else:
                    date_str = 'N/A'

                insp_grade = display_value(insp.get('grade'), '-')
                insp_score = display_value(insp.get('score'), '-')
                insp_color = get_grade_color(insp_grade if insp_grade != '-' else 'Z')

                history_items.append(
                    f'<div style="display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid #eee;">'
                    f'<div class="grade-badge" style="background: {insp_color}; width: 26px; height: 26px; font-size: 0.8rem;">{insp_grade}</div>'
                    f'<div style="flex: 1; font-size: 0.8rem;">'
                    f'<div style="font-weight: 500;">{date_str}</div>'
                    f'<div style="color: #6C757D;">Score: {insp_score}</div>'
                    f'</div></div>'
                )

        # Check if we have a valid prediction for this restaurant
        pred = st.session_state.get('prediction_result')
        show_prediction = pred and pred.get('camis') == camis

        # Side-by-side inspection cards
        insp_left, insp_right = st.columns(2)

        with insp_left:
            st.markdown(f"""
            <div class="info-card" style="padding: 10px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.85rem; color: #6C757D;">Latest Inspection</h4>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="text-align: center;">
                        <div class="grade-badge" style="background: {grade_color}; width: 42px; height: 42px; font-size: 1.2rem;">
                            {grade}
                        </div>
                    </div>
                    <div style="flex: 1; font-size: 0.85rem;">
                        <div style="margin-bottom: 4px;">
                            <span style="color: #6C757D;">Score:</span>
                            <span style="font-weight: 600; margin-left: 4px;">{score}</span>
                        </div>
                        <div>
                            <span style="color: #6C757D;">Date:</span>
                            <span style="font-weight: 500; margin-left: 4px;">{inspection_date_str}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Prediction result under Latest Inspection
            if show_prediction:
                pred_grade = pred['grade']
                pred_color = get_grade_color(pred_grade)
                st.markdown(f"""
                <div class="info-card" style="padding: 10px; border: 2px solid {pred_color}; text-align: center;">
                    <h4 style="margin: 0 0 4px 0; font-size: 0.85rem; color: #6C757D;">Predicted Next</h4>
                    <div style="font-size: 0.7rem; color: #888; margin-bottom: 6px;">{pred['time_estimate']}</div>
                    <div class="grade-badge" style="background: {pred_color}; width: 42px; height: 42px; font-size: 1.2rem; margin: 0 auto;">
                        {pred_grade}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with insp_right:
            if history_items:
                history_html = (
                    '<div class="info-card" style="padding: 10px;">'
                    '<h4 style="margin: 0 0 8px 0; font-size: 0.85rem; color: #6C757D;">History</h4>'
                    + ''.join(history_items) +
                    '</div>'
                )
                st.markdown(history_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card" style="padding: 10px;">
                    <h4 style="margin: 0 0 8px 0; font-size: 0.85rem; color: #6C757D;">History</h4>
                    <div style="color: #6C757D; font-size: 0.85rem;">No history available</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bars under History (compact horizontal)
            if show_prediction:
                formatted_probs = format_probabilities(pred['probabilities'])
                conf_items = ''.join([
                    f'<div style="flex: 1; text-align: center;">'
                    f'<div style="font-size: 0.75rem; color: #6C757D;">{g}</div>'
                    f'<div style="background: #E9ECEF; border-radius: 4px; height: 4px; margin: 4px 0;">'
                    f'<div style="background: {get_grade_color(g)}; width: {p}%; height: 100%; border-radius: 4px;"></div>'
                    f'</div>'
                    f'<div style="font-size: 0.7rem; font-weight: 500;">{p:.0f}%</div>'
                    f'</div>'
                    for g, p in formatted_probs
                ])
                st.markdown(f"""
                <div class="info-card" style="padding: 10px;">
                    <h4 style="margin: 0 0 8px 0; font-size: 0.85rem; color: #6C757D;">Confidence</h4>
                    <div style="display: flex; gap: 8px;">{conf_items}</div>
                </div>
                """, unsafe_allow_html=True)
