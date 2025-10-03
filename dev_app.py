import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time
from typing import List, Tuple, Optional

st.set_page_config(page_title="Ride Cancellation Predictor (Ensemble)", layout="wide")

PIPE_PATH = "artifacts/models/pipeline.pkl"

# Utilities
def _safe_time(x):
    if isinstance(x, time):
        return x
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(str(x), fmt).time()
        except Exception:
            pass
    return None

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either Date/Time or precomputed hour/dayofweek/is_weekend + categorical fields.
    Returns the minimal pre-booking feature frame expected by the pipeline.
    """
    if "Date" in df.columns and df["Date"].notna().any():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    if "Time" in df.columns and df["Time"].notna().any():
        df["Time"] = df["Time"].apply(_safe_time)
        df["hour"] = pd.to_datetime(df["Time"].astype(str), errors="coerce").dt.hour

    # Ensure required columns exist
    req = ["hour", "dayofweek", "is_weekend", "Vehicle Type", "Payment Method"]
    for c in req:
        if c not in df.columns:
            df[c] = None
    return df[req]

@st.cache_resource
def load_pipeline():
    try:
        pipe = joblib.load(PIPE_PATH)
        return pipe
    except Exception as e:
        st.error(
            "💡 Couldn’t find a trained pipeline at "
            f"`{PIPE_PATH}`.\n\n"
            "Run training first:\n\n"
            "```bash\npython scripts/train.py --csv data/ncr_ride_booking.csv\n```"
        )
        st.stop()

def _pre_and_estimator_from_pipeline(pipe):
    """
    Returns (preprocessor, estimator) from a saved pipeline.
    The estimator may be a single model, VotingClassifier, or StackingClassifier.
    """
    pre = None
    est = None
    if hasattr(pipe, "named_steps"):
        pre = pipe.named_steps.get("pre", None)
        est = pipe.named_steps.get("ens", None) or pipe.named_steps.get("clf", None)
    return pre, est

def _get_output_feature_names(pre) -> List[str]:
    """
    Build feature names after ColumnTransformer (numeric + OHE categories).
    """
    num_features = ["hour", "dayofweek", "is_weekend"]
    cat_features = ["Vehicle Type", "Payment Method"]
    names = list(num_features)
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(cat_features))
        names.extend(cat_names)
    except Exception:
        # Fallback if categories not available yet
        pass
    return names

def _single_model_importances(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Extract importances for a single estimator (tree-based 'feature_importances_' or linear 'coef_').
    Returns a dataframe or None if not available.
    """
    try:
        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
            k = min(len(imps), len(feature_names))
            return pd.DataFrame({"feature": feature_names[:k], "importance": imps[:k]})
        if hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            k = min(len(coef), len(feature_names))
            return pd.DataFrame({"feature": feature_names[:k], "importance": np.abs(coef[:k])})
    except Exception:
        pass
    return None

def compute_importances(pipe) -> Optional[pd.DataFrame]:
    """
    Try to compute global feature importances from the saved pipeline:
    - If final estimator is single model -> read directly
    - If VotingClassifier -> average available importances across members
    - If StackingClassifier -> try averaging base estimator importances
    Falls back to None if not available.
    """
    pre, est = _pre_and_estimator_from_pipeline(pipe)
    if pre is None or est is None:
        return None

    feat_names = _get_output_feature_names(pre)

    # Single model
    df_imp = _single_model_importances(est, feat_names)
    if df_imp is not None:
        return df_imp.sort_values("importance", ascending=False).head(20)

    try:
        from sklearn.ensemble import VotingClassifier, StackingClassifier  # noqa
        if est.__class__.__name__ == "VotingClassifier" and hasattr(est, "estimators_"):
            parts = []
            for m in est.estimators_:
                imp = _single_model_importances(m, feat_names)
                if imp is not None:
                    parts.append(imp.set_index("feature"))
            if parts:
                merged = pd.concat(parts, axis=1).fillna(0)
                merged["importance"] = merged.mean(axis=1)
                return merged[["importance"]].reset_index().sort_values("importance", ascending=False).head(20)
        # Stacking: average importances of base learners if available
        if est.__class__.__name__ == "StackingClassifier" and hasattr(est, "estimators_"):
            parts = []
            for m in est.estimators_:
                imp = _single_model_importances(m, feat_names)
                if imp is not None:
                    parts.append(imp.set_index("feature"))
            if parts:
                merged = pd.concat(parts, axis=1).fillna(0)
                merged["importance"] = merged.mean(axis=1)
                return merged[["importance"]].reset_index().sort_values("importance", ascending=False).head(20)
    except Exception:
        pass

    return None

def likely_reason(row: dict) -> str:
    hour = row.get("hour", None)
    vt = (row.get("Vehicle Type") or "").lower()
    pay = (row.get("Payment Method") or "").lower()
    reasons = []
    if hour is not None:
        if hour in [7, 8, 9, 18, 19, 20]:
            reasons.append("driver/unavailable (peak-hour)")
        if hour is not None and (hour >= 22 or hour <= 5):
            reasons.append("driver/unavailable (late-night)")
    if "ebike" in vt or "auto" in vt:
        reasons.append("vehicle/driver capacity")
    if "cash" in pay or "cod" in pay:
        reasons.append("customer/payment friction")
    return ", ".join(sorted(set(reasons))) if reasons else "uncertain (mix of customer/driver/system factors)"

# Frontend 
st.title("Ride Cancellation Predictor\n Ensemble Pipeline")
st.caption("Uses pre-booking features only.")

pipe = load_pipeline()

with st.sidebar:
    st.header("Single Prediction")
    col1, col2 = st.columns(2)
    with col1:
        date_in = st.date_input("Date", value=None, format="YYYY-MM-DD")
        hour_in = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=18)
    with col2:
        time_in = st.text_input("Time (HH:MM:SS)", value="18:00:00")
        is_weekend_in = st.checkbox("Is Weekend?", value=False)

    vehicle = st.selectbox(
        "Vehicle Type",
        ["Go Sedan", "Go", "SUV", "eBike", "Auto", "Mini", "Prime Sedan", "Prime SUV"],
        index=0
    )
    payment = st.selectbox(
        "Payment Method",
        ["UPI", "Cash", "Debit Card", "Credit Card", "Wallet"],
        index=0
    )

    if st.button("Predict"):
        row = {
            "Date": date_in.isoformat() if date_in else None,
            "Time": time_in,
            "hour": hour_in,
            "dayofweek": pd.to_datetime(date_in).dayofweek if date_in else None,
            "is_weekend": int(is_weekend_in),
            "Vehicle Type": vehicle,
            "Payment Method": payment
        }
        X = derive_features(pd.DataFrame([row]))
        proba = pipe.predict_proba(X)[:, 1][0]
        st.metric("Cancellation probability", f"{proba:.2%}")
        st.info(f"Likely reasons (heuristic): {likely_reason(X.iloc[0].to_dict())}")

st.markdown("---")
tab1, tab2 = st.tabs(["📈 Feature Importance", "📤 Batch Predictions"])

with tab1:
    st.subheader("Top features (when available)")
    imp_df = compute_importances(pipe)
    if imp_df is not None and not imp_df.empty:
        st.dataframe(imp_df.sort_values("importance", ascending=False).head(15), use_container_width=True)
        st.bar_chart(imp_df.set_index("feature")["importance"])
    else:
        st.write("Feature importances are not available for this ensemble. (Try a tree model or Voting over trees.)")

with tab2:
    st.subheader("Upload CSV for batch predictions")
    st.caption("Columns: Date, Time, Vehicle Type, Payment Method (or directly hour, dayofweek, is_weekend).")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df_up = pd.read_csv(file)
        Xb = derive_features(df_up.copy())
        probs = pipe.predict_proba(Xb)[:, 1]
        out = Xb.copy()
        out["cancel_probability"] = probs
        st.dataframe(out.head(50), use_container_width=True)
        st.download_button("Download predictions CSV", data=out.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
