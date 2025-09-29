import argparse, json, sys
import pandas as pd
import joblib

REQ_COLS = ["hour","dayofweek","is_weekend","Vehicle Type","Payment Method"]

def load_pipeline(path: str):
    return joblib.load(path)

def to_dataframe(payload):
    if isinstance(payload, dict):
        data = [payload]
    elif isinstance(payload, list):
        data = payload
    else:
        raise ValueError("Payload must be a JSON object or list of objects")
    df = pd.DataFrame(data)

    # derive time features if Date/Time provided
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    if "Time" in df.columns:
        df["hour"] = pd.to_datetime(df["Time"].astype(str), errors="coerce").dt.hour

    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = None
    return df[REQ_COLS]

def main():
    ap = argparse.ArgumentParser(description="Predict cancellation probability using saved ensemble pipeline.")
    ap.add_argument("--json", type=str, required=True, help="Path to JSON file or inline JSON string")
    ap.add_argument("--pipeline", type=str, default="artifacts/models/pipeline.pkl", help="Path to saved pipeline.pkl")
    ap.add_argument("--proba", action="store_true", help="Output probabilities instead of class labels")
    args = ap.parse_args()

    # Read payload
    try:
        try:
            with open(args.json, "r") as f:
                payload = json.load(f)
        except FileNotFoundError:
            payload = json.loads(args.json)
    except Exception as e:
        print(f"Failed to read JSON: {e}", file=sys.stderr); sys.exit(2)

    df = to_dataframe(payload)
    pipe = load_pipeline(args.pipeline)

    if args.proba and hasattr(pipe, "predict_proba"):
        preds = pipe.predict_proba(df)[:,1].tolist()
    else:
        preds = pipe.predict(df).tolist()

    out = [{"input": row, "prediction": pred} for row, pred in zip(df.to_dict(orient="records"), preds)]
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
