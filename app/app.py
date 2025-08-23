# app/app.py
# Streamlit UI for Wine Quality model (works for binary classification or regression)
# - Auto-detects project root from this file's location (no hard-coded paths)
# - Loads: models/model.pkl (required), feature_names.json (preferred),
#          metrics.json, class_map.json, thresholds.json, and report/*.png if present
# - Modes: Single prediction, Batch CSV prediction, Reports viewer
# - Threshold slider for binary classification (uses thresholds.json if available)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"

MODEL_PATH = MODELS_DIR / "model.pkl"
FEATS_PATH = MODELS_DIR / "feature_names.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
CLASS_MAP_PATH = MODELS_DIR / "class_map.json"
THRESHOLDS_PATH = REPORT_DIR / "thresholds.json"  # produced by your threshold-tuning step

# ---------- Helpers ----------
def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def ensure_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in required_cols]
    return missing, extra

def get_estimator(pipeline):
    # Our pipeline steps are named ("prep", "model")
    try:
        return pipeline.named_steps.get("model", pipeline)
    except Exception:
        return pipeline

@st.cache_resource(show_spinner=False)
def load_artifacts():
    errors = []
    if not MODEL_PATH.exists():
        errors.append(f"Missing model file at {MODEL_PATH}")
        return None, None, None, None, None, errors

    pipe = load(MODEL_PATH)

    # Feature names (for form ordering)
    feats_json = load_json(FEATS_PATH)
    if feats_json and "features" in feats_json:
        features = feats_json["features"]
    else:
        # Fallback to typical WineQT numeric features
        features = [
            "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
            "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
        ]
        errors.append("feature_names.json not found — using a standard WineQT feature list.")

    # Optional files
    metrics = load_json(METRICS_PATH) or {}
    class_map = load_json(CLASS_MAP_PATH) or {}
    thresholds = load_json(THRESHOLDS_PATH) or {}

    # Detect task type (classifier vs regressor)
    est = get_estimator(pipe)
    is_classifier = bool(
        hasattr(est, "predict_proba") or hasattr(est, "classes_")
    )

    return pipe, features, metrics, class_map, thresholds, errors


def get_positive_index(est, class_map, y_train_classes=None, desired_pos=1):
    """
    Determine index of the positive class in predict_proba output.
    """
    classes = None
    if hasattr(est, "classes_"):
        classes = list(est.classes_)
    elif class_map and "classes" in class_map:
        classes = list(class_map["classes"])
    elif y_train_classes is not None:
        classes = list(sorted(set(y_train_classes)))

    if classes is None:
        # assume binary with classes [0, 1]
        classes = [0, 1]
    pos_label = desired_pos if desired_pos in classes else max(classes)
    try:
        pos_idx = classes.index(pos_label)
    except ValueError:
        pos_idx = -1  # fallback (shouldn't happen)
    return pos_label, pos_idx, classes


def predict_single(pipe, features, row_dict, threshold=None, class_map=None):
    """
    Build a 1-row DataFrame from UI inputs in the correct order and predict.
    """
    X = pd.DataFrame([{f: row_dict.get(f, 0.0) for f in features}], columns=features)
    est = get_estimator(pipe)

    # Try probability first (for classification); otherwise do plain predict
    y_hat = pipe.predict(X)
    result = {"prediction": y_hat[0]}

    if hasattr(est, "predict_proba") or hasattr(est, "decision_function"):
        # compute score/proba for positive class if binary
        pos_label, pos_idx, classes = get_positive_index(est, class_map)
        try:
            proba = pipe.predict_proba(X)
            if proba.ndim == 2 and pos_idx >= 0:
                score = float(proba[0, pos_idx])
            else:
                score = float(proba[0]) if proba.ndim == 1 else None
        except Exception:
            # decision_function fallback
            try:
                scores = pipe.decision_function(X)
                if scores.ndim == 2 and pos_idx >= 0:
                    # Convert margin to [0,1] via logistic-ish squashing for display only
                    margin = scores[0, pos_idx]
                    score = float(1 / (1 + np.exp(-margin)))
                else:
                    margin = scores[0] if np.ndim(scores) == 1 else 0.0
                    score = float(1 / (1 + np.exp(-margin)))
            except Exception:
                score = None

        # If a threshold is requested and score is available, convert to label
        if threshold is not None and score is not None and len(set(classes)) == 2:
            result["threshold"] = threshold
            result["positive_label"] = pos_label
            result["score_positive"] = score
            result["prediction_thresholded"] = int(score >= threshold)
        else:
            result["score_positive"] = score

    return result


def predict_batch(pipe, features, df_in, threshold=None, class_map=None):
    # Ensure column order and subset to required columns
    missing, extra = ensure_columns(df_in, features)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df_in[features].copy()
    preds = pipe.predict(X)

    out = df_in.copy()
    out["prediction"] = preds

    # Add positive-class score + thresholded label if classification & available
    est = get_estimator(pipe)
    if hasattr(est, "predict_proba") or hasattr(est, "decision_function"):
        pos_label, pos_idx, classes = get_positive_index(est, class_map)
        score = None
        try:
            proba = pipe.predict_proba(X)
            score = proba[:, pos_idx] if proba.ndim == 2 and pos_idx >= 0 else None
        except Exception:
            try:
                scores = pipe.decision_function(X)
                if np.ndim(scores) == 2 and pos_idx >= 0:
                    margin = scores[:, pos_idx]
                    score = 1 / (1 + np.exp(-margin))
                elif np.ndim(scores) == 1:
                    margin = scores
                    score = 1 / (1 + np.exp(-margin))
            except Exception:
                score = None

        if score is not None:
            out["score_positive"] = score
            if threshold is not None and len(set(classes)) == 2:
                out["prediction_thresholded"] = (out["score_positive"] >= threshold).astype(int)

    return out


# ---------- UI ----------
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("🍷 Wine Quality Predictor")

pipe, features, metrics, class_map, thresholds, load_errors = load_artifacts()

if load_errors:
    with st.expander("Warnings / Missing files", expanded=False):
        for e in load_errors:
            st.write("•", e)

if pipe is None:
    st.error("Model pipeline not found. Train and save `models/model.pkl` first.")
    st.stop()

est = get_estimator(pipe)
is_classifier = bool(hasattr(est, "predict_proba") or hasattr(est, "classes_"))

# Sidebar: mode & threshold
mode = st.sidebar.radio("Mode", ["Single prediction", "Batch CSV", "Reports"], index=0)

threshold = None
if is_classifier:
    st.sidebar.markdown("### Decision threshold (binary only)")
    use_opt = st.sidebar.selectbox(
        "Use preset from thresholds.json?",
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

# Metrics panel (top-right)
with st.sidebar:
    st.markdown("### Model metrics")
    if metrics:
        for k in ["accuracy","f1_weighted","roc_auc_ovr","r2","mae","rmse"]:
            if k in metrics and metrics[k] is not None:
                st.metric(k.replace("_"," ").upper(), f"{metrics[k]:.3f}")
    else:
        st.caption("metrics.json not found.")

# ---------- Modes ----------
if mode == "Single prediction":
    st.subheader("Single prediction")

    # Demo/example defaults (feel free to adjust)
    example = {
        "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4
    }
    use_example = st.checkbox("Use example values", value=True)

    cols = st.columns(3)
    values = {}
    for i, f in enumerate(features):
        with cols[i % 3]:
            values[f] = st.number_input(
                f, value=float(example.get(f, 0.0) if use_example else 0.0), step=0.01, format="%.5f"
            )

    if st.button("Predict"):
        res = predict_single(pipe, features, values, threshold=threshold if is_classifier else None, class_map=class_map)
        if is_classifier:
            st.success(f"Predicted label: {res.get('prediction_thresholded', res['prediction'])}")
            if res.get("score_positive") is not None:
                st.write(f"Positive-class score: **{res['score_positive']:.3f}**  |  Threshold: **{res.get('threshold', 0.50):.2f}**")
        else:
            st.success(f"Predicted quality (regression): {res['prediction']}")

elif mode == "Batch CSV":
    st.subheader("Batch CSV prediction")
    st.caption("Upload a CSV containing the feature columns. Extra columns will be preserved.")

    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        df_in = pd.read_csv(file)
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

else:  # Reports
    st.subheader("Reports")
    cm = REPORT_DIR / "confusion_matrix.png"
    cmn = REPORT_DIR / "confusion_matrix_normalized.png"
    roc = REPORT_DIR / "roc_curve.png"
    cr_csv = REPORT_DIR / "classification_report.csv"

    if cm.exists():
        st.image(str(cm), caption="Confusion Matrix (Test)", use_column_width=True)
    if cmn.exists():
        st.image(str(cmn), caption="Confusion Matrix (Normalized)", use_column_width=True)
    if roc.exists():
        st.image(str(roc), caption="ROC Curve", use_column_width=True)
    if cr_csv.exists():
        cr_df = pd.read_csv(cr_csv, index_col=0)
        st.dataframe(cr_df.style.format(precision=3))
    if not any(p.exists() for p in [cm, cmn, roc, cr_csv]):
        st.info("No report files found in /report. Generate them from the notebook (steps 12–14).")

st.caption(f"Models folder: {MODELS_DIR}  •  Reports folder: {REPORT_DIR}")
