import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime

from src.data_loader import get_data, get_raw_data, refresh_data, load_training_data, clear_data_cache
from src.predictor import predict_restaurant_grade, clear_model_cache, get_model_metadata, model_needs_retraining, ModelNeedsRetrainingError
from src.feature_engineering import compute_all_features, compute_training_features
from src.trainer import train_model, save_model, get_feature_importance_ranking
from src.utils import (
    get_grade_color,
    format_probabilities,
    row_to_model_input,
    normalize_text,
    display_value,
    prepare_map_dataframe,
)


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="CleanKitchen NYC",
    page_icon="🍽️",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS for clean light theme
# -------------------------------------------------
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* Tighter header spacing */
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
    }

    /* Reduce main title margin */
    h1 {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: 600 !important;
    }

    h2, h3 {
        margin-top: 0.5rem !important;
    }

    /* Button styling */
    .stButton > button {
        background: #E87C4F;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #D4703F;
    }

    /* Form elements */
    .stSelectbox > div > div {
        border-radius: 6px;
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Grade badge */
    .grade-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        border-radius: 12px;
        font-size: 32px;
        font-weight: 700;
        color: white;
    }

    .grade-A { background: #7DB87D; }
    .grade-B { background: #E8C84A; }
    .grade-C { background: #8B3A3A; }
    .grade-P { background: #9BA8C4; }
    .grade-Z { background: #D4956A; }

    /* Card container */
    .info-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #E9ECEF;
        margin-bottom: 1rem;
    }

    /* Results counter */
    .results-counter {
        background: #F8F9FA;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        border-left: 3px solid #E87C4F;
    }

    .results-counter p {
        margin: 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.75rem; font-weight: 700;">CleanKitchen NYC</span>
    <span style="color: #6C757D; margin-left: 12px;">AI-powered restaurant grade predictions</span>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load data
# -------------------------------------------------
@st.cache_data
def load_app_data():
    """Load feature-enriched restaurant data (one row per restaurant)."""
    df = get_data()

    # Normalize text fields for filters (may already be done, but ensure consistency)
    df["borough"] = df["borough"].astype(str).str.strip().str.title()
    df["cuisine_description"] = df["cuisine_description"].astype(str).str.strip().str.title()

    # Ensure inspection_date is datetime
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")

    return df


df = load_app_data()

if df.empty:
    st.error("No data loaded. Please check your CSV files in the data/ folder.")
    st.stop()


# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filter Restaurants")

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

st.sidebar.markdown(f"""
<div class="results-counter">
    <p>{len(df_filtered):,} restaurants found</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Data Management (sidebar)
# -------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("Data Management")
if st.sidebar.button("🔄 Fetch Fresh Data"):
    with st.spinner("Fetching latest data from NYC Open Data API... This may take 1-2 minutes."):
        success, message, count = refresh_data()
        if success:
            st.sidebar.success(f"{message} ({count:,} records)")
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(message)


# -------------------------------------------------
# Model Management (sidebar)
# -------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("Model Management")

# Show current model info
try:
    metadata = get_model_metadata()
    training_date = metadata.get("training_date", "Unknown")
    if training_date != "Unknown":
        # Parse ISO format date
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
if st.sidebar.button("🧠 Retrain Model"):
    with st.sidebar:
        progress_bar = st.progress(0, text="Loading training data...")

        try:
            # Step 1: Load raw data
            progress_bar.progress(10, text="Loading raw inspection data...")
            raw_df = load_training_data()

            # Step 2: Compute features (using training-specific function for correct temporal alignment)
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


# -------------------------------------------------
# MAIN LAYOUT: Map (left) + Details/Prediction (right)
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Map of Restaurants")

    if len(df_filtered) == 0:
        st.info("No restaurants match your filters. Try changing the filters.")
    else:
        # Prepare data for PyDeck (no marker limit needed)
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

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
            ),
            height=500,
            width='stretch',
        )

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
            width='stretch'
        )
    else:
        st.dataframe(df_filtered.head(300), width='stretch')


with right_col:
    st.subheader("Restaurant Details")

    if len(df_filtered) == 0:
        st.info("Use the filters to select at least one restaurant.")
    else:
        # Let user pick a restaurant from a dropdown
        # Use name + zip as label
        if "dba" in df_filtered.columns:
            name_col = "dba"
        elif "DBA" in df_filtered.columns:
            name_col = "DBA"
        else:
            name_col = df_filtered.columns[0]  # fallback

        df_filtered = df_filtered.reset_index(drop=True)
        options = df_filtered.index.tolist()

        # Build labels with street name and ZIP for easy identification
        def make_label(i):
            name = df_filtered.loc[i, name_col]
            street = df_filtered.loc[i, 'street'] if 'street' in df_filtered.columns else None
            zipcode = df_filtered.loc[i, 'zipcode']

            if pd.notna(street) and street not in ('nan', ''):
                return f"{name} ({street}, {zipcode})"
            else:
                # Fall back to borough if no street
                borough = df_filtered.loc[i, 'borough']
                return f"{name} ({borough}, {zipcode})"

        labels = [make_label(i) for i in options]

        selected_idx = st.selectbox(
            "Choose a restaurant to analyze:",
            options=options,
            format_func=lambda i: labels[i]
        )

        selected_row = df_filtered.loc[selected_idx]

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

        # Restaurant info card
        st.markdown(f"""
        <div class="info-card">
            <h3 style="margin: 0 0 12px 0; font-size: 1.1rem; color: #2C3E50;">{restaurant_name}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9rem;">
                <div>
                    <span style="color: #6C757D;">Borough</span><br/>
                    <span style="font-weight: 500;">{borough}</span>
                </div>
                <div>
                    <span style="color: #6C757D;">ZIP</span><br/>
                    <span style="font-weight: 500;">{zipcode}</span>
                </div>
                <div style="grid-column: span 2;">
                    <span style="color: #6C757D;">Cuisine</span><br/>
                    <span style="font-weight: 500;">{cuisine}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Latest inspection card
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin: 0 0 12px 0; font-size: 0.95rem; color: #6C757D;">Latest Inspection</h4>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="text-align: center;">
                    <div class="grade-badge" style="background: {grade_color}; width: 48px; height: 48px; font-size: 1.3rem;">
                        {grade}
                    </div>
                    <div style="font-size: 0.75rem; color: #6C757D; margin-top: 4px;">Grade</div>
                </div>
                <div style="flex: 1; font-size: 0.9rem;">
                    <div style="margin-bottom: 6px;">
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

        # Inspection History (last 5 inspections)
        camis = selected_row.get('camis')
        if camis:
            raw_df = get_raw_data()
            history = raw_df[raw_df['camis'] == camis].copy()
            history = history.sort_values('inspection_date', ascending=False)
            # Get unique inspections by date
            history = history.drop_duplicates(subset=['inspection_date']).head(5)

            if len(history) > 0:
                history_items = []
                for _, insp in history.iterrows():
                    insp_date = insp.get('inspection_date')
                    if pd.notna(insp_date):
                        try:
                            date_str = pd.to_datetime(insp_date).strftime('%b %d, %Y')
                        except:
                            date_str = str(insp_date)
                    else:
                        date_str = 'N/A'

                    insp_grade = display_value(insp.get('grade'), '-')
                    insp_score = display_value(insp.get('score'), '-')
                    insp_color = get_grade_color(insp_grade if insp_grade != '-' else 'Z')

                    history_items.append(
                        f'<div style="display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #eee;">'
                        f'<div class="grade-badge" style="background: {insp_color}; width: 32px; height: 32px; font-size: 0.9rem;">{insp_grade}</div>'
                        f'<div style="flex: 1; font-size: 0.85rem;">'
                        f'<span style="font-weight: 500;">{date_str}</span>'
                        f'<span style="color: #6C757D; margin-left: 8px;">Score: {insp_score}</span>'
                        f'</div></div>'
                    )

                history_html = (
                    '<div class="info-card">'
                    '<h4 style="margin: 0 0 12px 0; font-size: 0.95rem; color: #6C757D;">Inspection History</h4>'
                    + ''.join(history_items) +
                    '</div>'
                )
                st.markdown(history_html, unsafe_allow_html=True)

        # Check if model needs retraining
        if model_needs_retraining():
            st.warning(
                "**Model needs to be trained.** The prediction model has been updated with new features. "
                "Please click **'Retrain Model'** in the sidebar to train the model before making predictions."
            )

        if st.button("Predict Next Inspection", width='stretch'):
            with st.spinner("Analyzing restaurant data..."):
                try:
                    # Build model input
                    model_input = row_to_model_input(selected_row)
                    result = predict_restaurant_grade(model_input)

                    predicted_grade = result["grade"]
                    probabilities = result["probabilities"]
                    formatted_probs = format_probabilities(probabilities)

                    # Get current inspection info
                    current_score = selected_row.get('score')
                    current_grade = selected_row.get('grade')

                    # Derive grade from score if official grade not available
                    if pd.isna(current_grade) or current_grade not in ['A', 'B', 'C']:
                        if pd.notna(current_score):
                            if current_score <= 13:
                                derived_grade = 'A'
                            elif current_score <= 27:
                                derived_grade = 'B'
                            else:
                                derived_grade = 'C'
                            grade_status = f"{derived_grade} (pending)"
                        else:
                            derived_grade = None
                            grade_status = "Pending"
                    else:
                        derived_grade = current_grade
                        grade_status = current_grade

                    # Calculate expected time to next inspection
                    days_since = selected_row.get('days_since_last_inspection', 0)
                    if pd.isna(days_since):
                        days_since = 0

                    median_interval = 124  # median days between inspections

                    # Handle stale data (no inspection in over a year)
                    if days_since > 365:
                        years_ago = round(days_since / 365, 1)
                        time_estimate = f"Last inspected {years_ago:.0f}+ years ago"
                    elif days_since > median_interval:
                        time_estimate = "Overdue for inspection"
                    elif days_since > median_interval - 30:
                        time_estimate = "Due within ~1 month"
                    elif days_since > median_interval - 60:
                        time_estimate = "Expected in ~2 months"
                    elif days_since > median_interval - 90:
                        time_estimate = "Expected in ~3 months"
                    else:
                        months = round((median_interval - days_since) / 30)
                        time_estimate = f"Expected in ~{months} months"

                    # Next inspection prediction card
                    pred_color = get_grade_color(predicted_grade)
                    st.markdown(f"""
                    <div class="info-card" style="text-align: center; border: 2px solid {pred_color};">
                        <p style="font-size: 0.75rem; margin-bottom: 0.5rem; color: #6C757D; text-transform: uppercase;">
                            Predicted Next Inspection
                        </p>
                        <p style="font-size: 0.7rem; color: #888; margin-bottom: 8px;">
                            {time_estimate}
                        </p>
                        <div class="grade-badge grade-{predicted_grade}" style="margin: 0 auto;">
                            {predicted_grade}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Confidence by Grade")
                    for g, p in formatted_probs:
                        grade_color = get_grade_color(g)
                        st.markdown(f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span>Grade {g}</span>
                                <span style="font-weight: 500;">{p:.1f}%</span>
                            </div>
                            <div style="background: #E9ECEF; border-radius: 4px; height: 6px; overflow: hidden;">
                                <div style="background: {grade_color}; width: {p}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Historical context
                    if derived_grade == 'C' or (pd.notna(current_score) and current_score >= 28):
                        st.markdown("""
                        <div style="background: #F8F9FA; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 0.8rem; color: #6C757D;">
                            <strong>Historical Pattern:</strong> 64% of restaurants that fail an inspection
                            pass their re-inspection within ~4 months after addressing violations.
                        </div>
                        """, unsafe_allow_html=True)

                except ModelNeedsRetrainingError:
                    st.error(
                        "**Model needs retraining.** Please click 'Retrain Model' in the sidebar first."
                    )
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
