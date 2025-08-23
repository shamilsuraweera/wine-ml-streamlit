# 🍷 Wine Quality Predictor (Streamlit + scikit-learn)

A simple project that trains a wine-quality model and serves it with a Streamlit app.  
This README keeps it short: **what I did, with the tech stack, step by step.**

---

## 🧰 Tech Stack (what I used)
- **Python 3**
- **pandas, numpy** – data handling
- **scikit-learn** – preprocessing + model + evaluation
- **matplotlib** – quick charts
- **joblib** – save/load the trained pipeline
- **Streamlit** – simple web UI (single + batch CSV predictions)

---

## ✅ What I did (step-by-step)
1. **Set up the project folders**
   - Created `app/`, `data/`, `models/`, `notebooks/`, `reports/`.
   - Placed the dataset at `data/WineQT.csv` (note the **e** in Wine).

2. **Explored the data (notebook)**
   - Loaded the CSV.
   - Checked shapes, types, quick stats, and histograms.

3. **Framed the task**
   - Treated it as **binary classification**: `quality >= 7` → **good (1)**, else **not good (0)**.

4. **Built a clean pipeline**
   - Inside scikit-learn **Pipeline**: `SimpleImputer(median)` → `StandardScaler()` → model.
   - Split data into **train/test**.

5. **Tried a few models & picked the best**
   - Logistic Regression, Random Forest, Gradient Boosting, SVC.
   - Cross-validated and selected the best performer.
   - Evaluated on the test set.

6. **Saved everything**
   - `models/model.pkl` – the full pipeline (preprocess + model).
   - `models/metrics.json` – main test metrics.
   - `models/feature_names.json` – feature order for the app.
   - `models/class_map.json` – labels (only for classification).

7. **Generated reports**
   - Saved to `reports/`:
     - `confusion_matrix.png`
     - `confusion_matrix_normalized.png`
     - `roc_curve.png`
     - `classification_report.csv/json`
   - (Optional) Computed **Best F1** / **Best F2** decision thresholds → `reports/thresholds.json`.

8. **Built the Streamlit app**
   - `app/app.py` with **three tabs**:
     - **Single prediction**: enter features → get label (+ score).
     - **Batch CSV**: upload CSV → download predictions.
     - **Reports**: shows the saved plots/tables from `reports/`.
   - Sidebar control for **decision threshold** (Default 0.50 / Best F1 / Best F2 / Custom).

---

## ▶️ How to run (super short)
```bash
# (optional) create venv
python3 -m venv .venv
source .venv/bin/activate

# install packages
pip install -r requirements.txt || pip install pandas numpy scikit-learn matplotlib joblib streamlit

# run the app from the project root
streamlit run app/app.py
