import pandas as pd

# -------------------------------------------------
# 1. Grade → Color mapping (UI styling)
# -------------------------------------------------

GRADE_COLORS = {
    "A": "#7DB87D",   # green
    "B": "#E8C84A",   # yellow
    "C": "#8B3A3A",   # red
    "P": "#9BA8C4",   # blue (Pending)
    "Z": "#D4956A",   # orange (Grade Pending / Not Yet Graded)
}

GRADE_COLORS_RGB = {
    "A": [125, 184, 125, 200],   # green with alpha
    "B": [232, 200, 74, 200],    # yellow
    "C": [139, 58, 58, 200],     # red
    "P": [155, 168, 196, 200],   # blue (Pending)
    "Z": [212, 149, 106, 200],   # orange
}

def get_grade_color(grade: str):
    """Return the hex color associated with a grade letter."""
    return GRADE_COLORS.get(grade, "#D4956A")


def get_grade_color_rgb(grade: str):
    """Return RGBA list for PyDeck visualization."""
    if pd.isna(grade):
        return [212, 149, 106, 200]
    return GRADE_COLORS_RGB.get(str(grade).upper(), [212, 149, 106, 200])


# -------------------------------------------------
# 2. Format model prediction probabilities
# -------------------------------------------------

def format_probabilities(prob_dict: dict):
    """
    Convert the { 'A': 0.87, 'B': 0.09, ... } dictionary
    into a sorted, clean list for UI display.
    """
    formatted = []
    for grade, prob in prob_dict.items():
        formatted.append((grade, round(prob * 100, 2)))  # convert to %
    return sorted(formatted, key=lambda x: x[1], reverse=True)


# -------------------------------------------------
# 3. Convert a restaurant row into model input dictionary
# -------------------------------------------------

# All features expected by the model
# NOTE: We use PREVIOUS inspection data to predict NEXT grade
MODEL_FEATURES = [
    "borough",
    "zipcode",
    "cuisine_description",
    "days_since_last_inspection",
    "inspection_frequency",
    "prev_grade_1",
    "prev_grade_2",
    "prev_score_1",
    "prev_score_2",
    "critical_violations_12mo",
    "total_violations_all_time",
    "avg_score_historical",
    "score_trend",
    "grade_stability",
    "cuisine_avg_score",
    "zipcode_avg_score",
    "violation_diversity",
]

# Required fields that must be present (for basic restaurant info)
REQUIRED_FIELDS = [
    "borough",
    "zipcode",
    "cuisine_description",
]


def row_to_model_input(row: pd.Series):
    """
    Take a row from the restaurant dataset and extract
    all features expected by the model.

    Required: borough, zipcode, cuisine_description
    Optional: All historical/computed features (will use defaults if missing)
    """
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in row:
            raise KeyError(f"Missing required field: {field}. Available: {list(row.index)}")

    data = {}

    for field in MODEL_FEATURES:
        if field in row.index:
            value = row[field]
            # Handle NaN values
            if pd.isna(value):
                data[field] = None
            else:
                data[field] = value
        else:
            # Field not present - predictor will use defaults
            data[field] = None

    return data


# -------------------------------------------------
# 4. Quick helper: clean filter inputs (ZIP, Cuisine, etc.)
# -------------------------------------------------

def clean_zip(zip_value):
    """Ensure ZIP codes become integers or None."""
    try:
        return int(zip_value)
    except (ValueError, TypeError):
        return None


def normalize_text(text):
    """Simple helper for cleaning cuisine/borough search inputs."""
    if pd.isna(text):
        return ""
    return str(text).strip().title()


def display_value(value, fallback="Unavailable"):
    """Return fallback string if value is NaN, None, or empty."""
    if pd.isna(value) or value == "" or value is None:
        return fallback
    return value


# -------------------------------------------------
# 5. Map marker helper functions (for Streamlit-Folium or PyDeck)
# -------------------------------------------------

def restaurant_popup_html(row):
    """
    Builds the HTML used in popups on the map
    for folium / leaflet display.
    """
    name = row.get("dba") or row.get("DBA") or "Unknown Restaurant"
    cuisine = row.get("cuisine_description", "Unknown")
    borough = row.get("borough", "Unknown")
    zipcode = row.get("zipcode", "")
    score = display_value(row.get("score"), "N/A")
    grade = display_value(row.get("grade"), "N/A")

    color = get_grade_color(grade)

    html = f"""
    <div style="font-size:14px;">
        <b>{name}</b><br>
        <span>Cuisine: {cuisine}</span><br>
        <span>Borough: {borough}</span><br>
        <span>ZIP: {zipcode}</span><br>
        <span>Score: {score}</span><br>
        <span>Grade: <b style='color:{color};'>{grade}</b></span>
    </div>
    """
    return html


# -------------------------------------------------
# 6. PyDeck map data preparation
# -------------------------------------------------

def prepare_map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for PyDeck visualization with color column."""
    map_df = df[['latitude', 'longitude', 'dba', 'grade',
                 'cuisine_description', 'borough', 'zipcode', 'score']].copy()

    map_df['color'] = map_df['grade'].apply(get_grade_color_rgb)
    map_df['name'] = map_df['dba'].fillna('Unknown Restaurant')
    map_df['grade_display'] = map_df['grade'].fillna('N/A')
    map_df['score_display'] = map_df['score'].fillna('N/A').astype(str)
    map_df = map_df.dropna(subset=['latitude', 'longitude'])

    return map_df

