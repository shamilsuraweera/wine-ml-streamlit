# app/app.py
# Streamlit UI for the Wine Quality project (no User Guide tab)
# - Works for binary classification or regression pipelines saved to models/model.pkl
# - Reads optional artifacts: feature_names.json, metrics.json, class_map.json
# - Reads optional reports from reports/ (confusion matrices, ROC, classification report, thresholds.json)
# - Tabs: Single prediction | Batch CSV | Reports
# - For binary classifiers: decision threshold (Default/Best F1/Best F2/Custom)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ----------------- Streamlit page config -----------------
st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="wide")

# ----------------- Paths -----------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Prefer plural "reports/", but gracefully fall back to legacy "report/"
REPORTS_DIR = PROJECT_ROOT / "reports"
LEGACY_REPORT_DIR = PROJECT_ROOT / "report"
if not REPORTS_DIR.exists() and LEGACY_REPORT_DIR.exists():
    REPORTS_DIR = LEGACY_REPORT_DIR  # fallback so old repos still work

MODEL_PATH      = MODELS_DIR / "model.pkl"
FEATS_PATH      = MODELS_DIR / "feature_names.json"
METRICS_PATH    = MODELS_DIR / "metrics.json"
CLASS_MAP_PATH  = MODELS_DIR / "class_map.json"
THRESHOLDS_PATH = REPORTS_DIR / "thresholds.json"  # created by notebook Step 14

# ----------------- Helpers -----------------
def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def ensure_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in required_cols]
    return missing, extra

def get_estimator(pipeline):
    """Return the final estimator from a sklearn Pipeline, or the object itself."""
    try:
        return pipeline.named_steps.get("model", pipeline)
    except Exception:
        return pipeline

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model and optional JSON artifacts once (cached)."""
    warns = []

    if not MODEL_PATH.exists():
        return None, None, None, None, None, [f"Missing model file at {MODEL_PATH}"]

    # Robust load with clear error if versions differ
    try:
        pipe = load(MODEL_PATH)
    except Exception as e:
        return None, None, None, None, None, [f"Failed to load model.pkl: {e}"]

    # Feature names (preferred)
    feats_json = load_json(FEATS_PATH)
    if feats_json and "features" in feats_json:
        features = feats_json["features"]
    else:
        # Fallback to standard WineQT feature list (case-sensitive)
        features = [
            "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
            "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
        ]
        warns.append("feature_names.json not found — using a standard WineQT feature list.")

    metrics    = load_json(METRICS_PATH) or {}
    class_map  = load_json(CLASS_MAP_PATH) or {}
    thresholds = load_json(THRESHOLDS_PATH) or {}

    est = get_estimator(pipe)
    is_classifier = bool(hasattr(est, "predict_proba") or hasattr(est, "classes_"))

    return pipe, features, metrics, class_map, thresholds, warns

def get_positive_index(est, class_map, desired_pos=1, fallback_classes=None):
    """
    Determine positive label and its index in predict_proba output.
    Prefer label '1' if present; otherwise use max(classes).
    """
    classes = None
    if hasattr(est, "classes_"):
        classes = list(est.classes_)
    elif class_map and "classes" in class_map:
        classes = list(class_map["classes"])
    elif fallback_classes is not None:
        classes = list(sorted(set(fallback_classes)))
    else:
        classes = [0, 1]  # safest default

    pos_label = desired_pos if desired_pos in classes else max(classes)
    try:
        pos_idx = classes.index(pos_label)
    except ValueError:
        pos_idx = -1
    return pos_label, pos_idx, classes

def predict_single(pipe, features, row_dict, threshold=None, class_map=None):
    """Build one-row DataFrame in correct order and run the pipeline."""
    X = pd.DataFrame([{f: row_dict.get(f, 0.0) for f in features}], columns=features)
    est = get_estimator(pipe)

    # Base prediction
    y_hat = pipe.predict(X)
    result = {"prediction": y_hat[0]}

    # Positive-class score (classification)
    score = None
    if hasattr(est, "predict_proba") or hasattr(est, "decision_function"):
        pos_label, pos_idx, classes = get_positive_index(est, class_map)
        try:
            proba = pipe.predict_proba(X)
            if proba.ndim == 2 and pos_idx >= 0:
                score = float(proba[0, pos_idx])
            elif proba.ndim == 1:
                score = float(proba[0])
        except Exception:
            try:
                scores = pipe.decision_function(X)
                if np.ndim(scores) == 2 and pos_idx >= 0:
                    margin = scores[0, pos_idx]
                else:
                    margin = scores[0] if np.ndim(scores) == 1 else 0.0
                score = float(1 / (1 + np.exp(-margin)))  # display-only squash
            except Exception:
                score = None

        result["score_positive"] = score
        if threshold is not None and score is not None and len(set(classes)) == 2:
            result["threshold"] = float(threshold)
            result["positive_label"] = pos_label
            result["prediction_thresholded"] = int(score >= threshold)

    return result

def predict_batch(pipe, features, df_in, threshold=None, class_map=None):
    """Predict over a CSV; preserve extra columns; add score & thresholded label if available."""
    missing, extra = ensure_columns(df_in, features)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df_in[features].copy()

    preds = pipe.predict(X)
    out = df_in.copy()
    out["prediction"] = preds

    est = get_estimator(pipe)
    score = None
    if hasattr(est, "predict_proba") or hasattr(est, "decision_function"):
        pos_label, pos_idx, classes = get_positive_index(est, class_map)
        try:
            proba = pipe.predict_proba(X)
            if proba.ndim == 2 and pos_idx >= 0:
                score = proba[:, pos_idx]
        except Exception:
            try:
                scores = pipe.decision_function(X)
                if np.ndim(scores) == 2 and pos_idx >= 0:
                    margin = scores[:, pos_idx]
                    score = 1 / (1 + np.exp(-margin))
                elif np.ndim(scores) == 1:
                    score = 1 / (1 + np.exp(-scores))
            except Exception:
                score = None

        if score is not None:
            out["score_positive"] = score
            if threshold is not None and len(set(classes)) == 2:
                out["prediction_thresholded"] = (out["score_positive"] >= threshold).astype(int)

    return out

# ----------------- Load artifacts -----------------
pipe, features, metrics, class_map, thresholds, load_warnings = load_artifacts()

# ----------------- Header & warnings -----------------
st.title("🍷 Wine Quality Predictor")

if load_warnings:
    with st.expander("Warnings / Missing optional files", expanded=False):
        for msg in load_warnings:
            st.write("•", msg)

if pipe is None:
    st.error("`models/model.pkl` not found or failed to load. Train/save in the notebook and push to the repo.")
    st.stop()

est = get_estimator(pipe)
is_classifier = bool(hasattr(est, "predict_proba") or hasattr(est, "classes_"))

# ----------------- Sidebar controls -----------------
threshold = None
if is_classifier:
    st.sidebar.markdown("### Decision threshold (binary only)")
    use_opt = st.sidebar.selectbox(
        "Use preset from reports/thresholds.json?",
        ["Default (0.50)", "Best F1", "Best F2", "Custom"],
        index=0
    )
    if use_opt == "Best F1" and thresholds:
        threshold = float(thresholds.get("best_F1", {}).get("threshold", 0.50))
    elif use_opt == "Best F2" and thresholds:
        threshold = float(thresholds.get("best_F2", {}).get("threshold", 0.50))
    elif use_opt == "Custom":
        threshold = float(st.sidebar.slider("Threshold", 0.0, 1.0, 0.50, 0.01))
    else:
        threshold = 0.50

with st.sidebar:
    st.markdown("### Model metrics")
    if metrics:
        keys = ["accuracy","f1_weighted","roc_auc_ovr","r2","mae","rmse"]
        for k in keys:
            if k in metrics and metrics[k] is not None:
                try:
                    st.metric(k.replace("_"," ").upper(), f"{metrics[k]:.3f}")
                except Exception:
                    st.metric(k.replace("_"," ").upper(), str(metrics[k]))
    else:
        st.caption("metrics.json not found (optional).")

# ----------------- Tabs -----------------
tab_single, tab_batch, tab_reports = st.tabs(
    ["Single prediction", "Batch CSV", "Reports"]
)

# -------- Tab: Single prediction --------
with tab_single:
    st.subheader("Single prediction")

    # Example defaults (editable)
    example = {
        "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4
    }
    use_example = st.checkbox("Use example values", value=True)

    cols = st.columns(3)
    values = {}
    for i, f in enumerate(features):  # type: ignore
        with cols[i % 3]:
            values[f] = st.number_input(
                f,
                value=float(example.get(f, 0.0) if use_example else 0.0),
                step=0.01,
                format="%.5f"
            )

    if st.button("Predict", type="primary"):
        res = predict_single(
            pipe, features, values,
            threshold=threshold if is_classifier else None,
            class_map=class_map
        )
        if is_classifier:
            shown_label = res.get("prediction_thresholded", res["prediction"])
            st.success(f"Predicted label: {shown_label}")
            if res.get("score_positive") is not None:
                st.write(
                    f"Positive-class score: **{res['score_positive']:.3f}**"
                    f"  |  Threshold: **{res.get('threshold', 0.50):.2f}**"
                )
        else:
            st.success(f"Predicted quality (regression): {res['prediction']}")

# -------- Tab: Batch CSV --------
with tab_batch:
    st.subheader("Batch CSV prediction")
    st.caption("Upload a CSV with the exact feature columns (case-sensitive). Extra columns will be preserved.")

    # Downloadable template
    tmpl_blank = pd.DataFrame(columns=features)
    st.download_button(
        "Download CSV template (headers only)",
        tmpl_blank.to_csv(index=False).encode("utf-8"),
        file_name="wine_features_template.csv",
        mime="text/csv"
    )

    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_in = None

        if df_in is not None:
            st.write("Input preview:", df_in.head())
            try:
                df_out = predict_batch(
                    pipe, features, df_in,
                    threshold=threshold if is_classifier else None,
                    class_map=class_map
                )
                st.success("Predictions ready.")
                st.dataframe(df_out.head())
                st.download_button(
                    "Download predictions CSV",
                    df_out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Failed to predict: {e}")

# -------- Tab: Reports --------
with tab_reports:
    st.subheader("Reports (saved by notebook)")

    cm   = REPORTS_DIR / "confusion_matrix.png"
    cmn  = REPORTS_DIR / "confusion_matrix_normalized.png"
    roc  = REPORTS_DIR / "roc_curve.png"
    cr_c = REPORTS_DIR / "classification_report.csv"

    cols_r = st.columns(2)
    if cm.exists():
        cols_r[0].image(str(cm), caption="Confusion Matrix (Test)", use_container_width=True)
    if cmn.exists():
        cols_r[1].image(str(cmn), caption="Confusion Matrix (Normalized)", use_container_width=True)
    if roc.exists():
        st.image(str(roc), caption="ROC Curve", use_container_width=True)
    if cr_c.exists():
        try:
            cr_df = pd.read_csv(cr_c, index_col=0)
            st.dataframe(cr_df.style.format(precision=3))
        except Exception as e:
            st.error(f"Could not read classification_report.csv: {e}")

    if not any(p.exists() for p in [cm, cmn, roc, cr_c]):
        st.info("No files found under reports/. Generate them from the training notebook (Steps 12–14).")

# ----------------- Footer -----------------
st.caption(f"Models folder: {MODELS_DIR}  •  Reports folder: {REPORTS_DIR}")
