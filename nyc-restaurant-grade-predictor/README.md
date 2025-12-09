# CleanKitchen NYC

A multi-page web application to explore NYC restaurant health inspection grades using real inspection data and machine learning predictions.

## What This Project Does

* **Browse restaurants** across New York City with inspection grades
* **Predict next inspection grade** (A, B, or C) using a trained ML model
* **Filter** by borough, ZIP code, and cuisine type
* **Interactive map** powered by PyDeck with color-coded grade markers
* **View inspection history** showing the last 5 inspections per restaurant
* **Retrain the model** on-demand through the UI
* **Fetch fresh data** directly from the NYC Open Data API

## Machine Learning Model

* Trained on over 290K NYC inspection records
* Uses Random Forest classifier for grade prediction
* **17 engineered features** including:
  * Historical: previous grades, previous scores, grade stability
  * Temporal: days since last inspection, inspection frequency
  * Violations: critical violations (12 months), total violations, violation diversity
  * Contextual: cuisine average score, ZIP code average score, score trend
* Outputs predicted grade with probability distribution for each class
* Model can be retrained via the Filter page UI

## Project Structure

```text
nyc-restaurant-grade-predictor/
├── app.py                    # Landing page
├── requirements.txt          # Python dependencies
├── README.md
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── style.css             # Custom styling
├── pages/
│   ├── 1_Filter.py           # Main filtering & prediction page
│   ├── 2_About.py            # Project information
│   └── 3_Creators.py         # Team credits
├── src/
│   ├── data_loader.py        # Data loading with caching
│   ├── data_pipeline.py      # NYC Open Data API integration
│   ├── feature_engineering.py # Feature computation (17 features)
│   ├── predictor.py          # Model inference
│   ├── trainer.py            # Model training pipeline
│   ├── utils.py              # Helper functions
│   └── components.py         # Shared UI components
├── models/                   # Trained model storage (auto-generated)
└── data/                     # CSV data files
```

## How It Works

1. Open the app and navigate to the **Filter** page
2. Use sidebar filters to narrow down by borough, ZIP code, or cuisine
3. Browse restaurants on the interactive map (color-coded by grade)
4. Click a map marker to select a restaurant
5. View restaurant details and inspection history (last 5 inspections)
6. Click **Predict Grade** to get the ML prediction with probability breakdown

## Data Sources

* **NYC Open Data API**: Restaurant Inspection Results (updated regularly)
* Data can be refreshed on-demand via the "Fetch Fresh Data" button

## Tech Used

* **Python** (Pandas, NumPy, Scikit-Learn)
* **Streamlit** multi-page application
* **PyDeck** for interactive map visualization
* **NYC Open Data API** for live data fetching
* **Joblib** for model serialization

## Setup

Install packages:

```bash
pip install -r requirements.txt
```

Set your NYC Open Data API token (required for data fetching):

```bash
export NYC_OPENDATA_APP_TOKEN=your_token_here
```

Run the app:

```bash
streamlit run app.py
```

The ML model will be trained automatically when you first click "Retrain Model" on the Filter page.

## Future Work

* Improve prediction accuracy with additional features
* Add restaurant risk indicators
* Expand data visualizations

## Contact

For help or questions, reach out any time!
