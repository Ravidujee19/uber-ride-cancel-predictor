
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, time

st.set_page_config(page_title="Uber Ride Cancellation Predictor", layout="centered")

PIPE_PATH = "artifacts/models/pipeline.pkl"

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
    if "Date" in df.columns and df["Date"].notna().any():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    if "Time" in df.columns and df["Time"].notna().any():
        df["Time"] = df["Time"].apply(_safe_time)
        df["hour"] = pd.to_datetime(df["Time"].astype(str), errors="coerce").dt.hour
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
    except Exception:
        st.error("Could not find a trained pipeline. Please run training first.")
        st.stop()

st.title("Uber Ride Cancellation Predictor")
st.caption("Enter ride details below to predict cancellation probability.")

pipe = load_pipeline()


with st.form("prediction_form"):
    date_in = st.date_input("Date")
    time_in = st.text_input("Time (HH:MM:SS)", value="", placeholder="e.g. 18:00:00")
    vehicle = st.selectbox("Vehicle Type", ["Select vehicle type...", "Go Sedan", "Go", "SUV", "eBike", "Auto", "Mini", "Prime Sedan", "Prime SUV"], index=0)
    payment = st.selectbox("Payment Method", ["Select payment method...", "UPI", "Cash", "Debit Card", "Credit Card", "Wallet"], index=0)
    # Center the button using markdown
    st.markdown("<div style='text-align:center; margin-top:1.5rem;'>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)
    if submitted:
        errors = []
        if not time_in:
            errors.append("Please enter the time of your ride.")
        if vehicle == "Select vehicle type...":
            errors.append("Please select a vehicle type.")
        if payment == "Select payment method...":
            errors.append("Please select a payment method.")
        if errors:
            for err in errors:
                st.warning(err)
        else:
            row = {
                "Date": date_in.isoformat() if date_in else None,
                "Time": time_in,
                "Vehicle Type": vehicle,
                "Payment Method": payment
            }
            X = derive_features(pd.DataFrame([row]))
            proba = pipe.predict_proba(X)[:, 1][0]
            st.metric("Cancellation probability", f"{proba:.2%}")

            # Improved recommendations
            hour = X.iloc[0]["hour"]
            vt = (X.iloc[0]["Vehicle Type"] or "").lower()
            pay = (X.iloc[0]["Payment Method"] or "").lower()
            recs = []
            if proba < 0.5:
                recs.append("Your ride is likely to be accepted. No special actions needed!")
            else:
                if hour in [7,8,9,18,19,20]:
                    recs.append("Peak hours detected. Try booking outside these times for better availability.")
                if hour is not None and (hour >= 22 or hour <= 5):
                    recs.append("Late-night rides have higher cancellation risk. Consider booking earlier in the day.")
                if "ebike" in vt or "auto" in vt:
                    recs.append("Standard car types are less likely to be cancelled than eBike or Auto.")
                if "cash" in pay or "cod" in pay:
                    recs.append("Digital payment methods (UPI, card, wallet) are preferred by drivers.")
                if not recs:
                    recs.append("Consider changing your ride time or payment method for better chances.")
            st.markdown("<div style='margin-top:1.5rem; text-align:center;'><b>Recommendations:</b></div>", unsafe_allow_html=True)
            for r in recs:
                st.markdown(f"<div style='text-align:center; color:#60a5fa; margin-bottom:0.5rem;'>{r}</div>", unsafe_allow_html=True)