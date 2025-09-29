import argparse, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
import joblib

SEED = 42
warnings.filterwarnings("ignore")

def build_preprocessor():
    numeric_features = ["hour","dayofweek","is_weekend"]
    categorical_features = ["Vehicle Type","Payment Method"]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

def derive_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    if "Time" in df.columns:
        df["hour"] = pd.to_datetime(df["Time"].astype(str), errors="coerce").dt.hour
    for col in ["hour","dayofweek","is_weekend","Vehicle Type","Payment Method"]:
        if col not in df.columns:
            df[col] = None
    return df[["hour","dayofweek","is_weekend","Vehicle Type","Payment Method"]]

def try_import_xgb_lgb():
    xgb, lgbm = None, None
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        xgb = None
    try:
        import lightgbm as lgbm  # type: ignore
    except Exception:
        lgbm = None
    return xgb, lgbm

def main():
    ap = argparse.ArgumentParser(description="Train ensemble cancellation model from CSV.")
    ap.add_argument("--csv", default="data/ncr_ride_booking.csv",
                    help="Path to dataset CSV (default: data/ncr_ride_booking.csv)")
    ap.add_argument("--outdir", default="artifacts/models", help="Where to save artifacts")
    ap.add_argument("--metrics", default="artifacts/models/metrics.json", help="Where to save metrics JSON")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(args.csv)

    cancel_map = {
        "Completed": 0,
        "Cancelled by Driver": 1,
        "Cancelled by Customer": 1,
        "No Driver Found": 1,
        "Incomplete": 0
    }
    if "Booking Status" not in df.columns:
        raise ValueError("CSV must contain 'Booking Status' column.")
    df["is_cancelled"] = df["Booking Status"].map(cancel_map).astype(int)

    X = derive_minimal_features(df.copy())
    y = df["is_cancelled"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    pre = build_preprocessor()

    # Plain classifiers
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    rf_clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)
    gb_clf = GradientBoostingClassifier(random_state=SEED)

    xgb, lgbm = try_import_xgb_lgb()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=SEED
    ) if xgb is not None else None

    lgb_clf = lgbm.LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED
    ) if lgbm is not None else None

    # For individual model metrics, wrap each with the SAME top-level preprocessor
    indiv_models = [("lr", lr_clf), ("rf", rf_clf), ("gb", gb_clf)]
    if xgb_clf is not None:
        indiv_models.append(("xgb", xgb_clf))
    if lgb_clf is not None:
        indiv_models.append(("lgb", lgb_clf))

    report = {}

    for name, clf in indiv_models:
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        report[name] = {
            "roc_auc": float(np.round(roc_auc_score(y_val, proba), 4)),
            "cls_report": classification_report(y_val, pred, output_dict=True)
        }

    # Soft Voting (shared top-level preprocessor)
    voting_estimators = [(n, c) for n, c in indiv_models]
    voting_pipe = Pipeline([
        ("pre", pre),
        ("ens", VotingClassifier(estimators=voting_estimators, voting="soft"))
    ])
    voting_pipe.fit(X_train, y_train)
    v_proba = voting_pipe.predict_proba(X_val)[:, 1]
    report["voting_soft"] = {
        "roc_auc": float(np.round(roc_auc_score(y_val, v_proba), 4)),
        "cls_report": classification_report(y_val, (v_proba >= 0.5).astype(int), output_dict=True)
    }

    # Stacking
    stack = Pipeline([
        ("pre", pre),
        ("ens", StackingClassifier(
            estimators=voting_estimators,
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
            stack_method="predict_proba",
            passthrough=False,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        ))
    ])
    stack.fit(X_train, y_train)
    s_proba = stack.predict_proba(X_val)[:, 1]
    report["stacking"] = {
        "roc_auc": float(np.round(roc_auc_score(y_val, s_proba), 4)),
        "cls_report": classification_report(y_val, (s_proba >= 0.5).astype(int), output_dict=True)
    }

    # Select best by ROC-AUC 
    candidates = [("voting_soft", voting_pipe), ("stacking", stack)]
    # also include single models (wrapped once) as fallbacks
    for name, clf in indiv_models:
        candidates.append((name, Pipeline([("pre", pre), ("clf", clf)])))

    best_name, best_auc, best_obj = None, -1.0, None
    for n, obj in candidates:
        auc = report[n]["roc_auc"]
        if auc > best_auc:
            best_name, best_auc, best_obj = n, auc, obj

    # Save one pipeline including preprocessor
    out_path = os.path.join(args.outdir, "pipeline.pkl")
    joblib.dump(best_obj, out_path)

    # Save metrics
    with open(args.metrics, "w") as f:
        json.dump({"best": best_name, "best_auc": best_auc, "models": report}, f, indent=2)

    # Save feature spec
    with open("artifacts/feature_spec.json","w") as f:
        json.dump({
            "numeric": ["hour","dayofweek","is_weekend"],
            "categorical": ["Vehicle Type","Payment Method"],
            "target": "is_cancelled",
            "note": "Minimal pre-booking features; ensembles (soft-voting & stacking)"
        }, f, indent=2)

    print(json.dumps({
        "status": "ok",
        "best_model": best_name,
        "roc_auc": best_auc,
        "artifact": out_path,
        "metrics": args.metrics
    }, indent=2))

if __name__ == "__main__":
    main()
