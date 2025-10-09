# House Price Predictor (Streamlit)

This repository contains a small Streamlit app to predict house prices from a single set of user-provided inputs.

Files
- `app.py` — Streamlit application. It shows one widget per column: sliders for continuous (numeric) features and selectboxes/text inputs for categorical features. It can train a simple `LinearRegression` on `Housing.csv` if present and containing a `MEDV` target. It also supports saving/loading simple presets, an auto-predict toggle, and a feature contribution bar chart.
- `requirements.txt` — Python dependencies (streamlit, pandas, numpy, scikit-learn).
- `Housing.csv` — (optional) sample dataset; if present and containing `MEDV`, the app will train on it.

Requirements
- Python 3.8+
- Install dependencies:

  pip install -r requirements.txt

How to run

1. From the repository folder run:

```powershell
streamlit run app.py
```

2. The app UI
- Upload a CSV or check "Use sample data from Housing.csv" in the sidebar (if the file exists).
- For each numeric column a slider appears to enter a single continuous value.
- For each categorical column a selectbox or text input appears to choose one category.
- Use the Predict button to run a prediction. Turn on "Auto predict (live)" in the sidebar for immediate predictions when you change inputs.
- Save a preset by entering a name and clicking "Save preset". Load it from the presets list — loaded presets are shown as JSON (you'll need to adjust widgets to match loaded values because Streamlit does not programmatically change widget states).
- The app shows a simple bar chart indicating feature contributions (uses model coefficients if trained, otherwise a heuristic).

Notes and limitations
- The app is intentionally simple and offline: no model is persisted between runs unless you export/save the model manually.
- Presets are stored as JSON files in a local `presets/` folder.
- If `Housing.csv` is missing or doesn't have `MEDV`, a heuristic fallback prediction is used (sum of numeric inputs).

If you want, I can:
- Add a button to apply a loaded preset to the widgets (requires a more advanced pattern using session state).
- Persist the trained model to disk (`.pkl`) and add a download link.
- Improve UI layout and styling.

