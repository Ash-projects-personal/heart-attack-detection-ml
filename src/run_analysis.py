"""Heart disease classification on the real UCI Cleveland data.

The target is the dataset's angiographic disease status, binarised:
num > 0 means disease present. This is not heart attack prediction
and these are not my patients. It is a 303 row clinical dataset
collected at the Cleveland Clinic and published by UCI.

Feature engineering is tested this time, not asserted: the Random
Forest is trained once on raw features and once on raw plus three
engineered features, same split, same seed, and both scores are
reported whatever they show.
"""
import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load():
    df = pd.read_csv(ROOT / "data" / "processed.cleveland.data", names=COLUMNS, na_values="?")
    rows_raw = len(df)
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["num"] > 0).astype(int)
    return df, rows_raw


def add_engineered(df):
    df = df.copy()
    df["age_chol_risk"] = df["age"] * df["chol"] / 1000.0
    df["pressure_rate_ratio"] = df["trestbps"] / df["thalach"]
    df["high_risk_profile"] = ((df["cp"] == 4) & (df["exang"] == 1)).astype(int)
    return df


def score(model, X_tr, y_tr, X_te, y_te, X_all, y_all):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    cv = cross_val_score(model, X_all, y_all,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="accuracy")
    return {
        "test_accuracy": float(accuracy_score(y_te, pred)),
        "test_roc_auc": float(roc_auc_score(y_te, prob)),
        "test_f1": float(f1_score(y_te, pred)),
        "cv_accuracy_mean": float(cv.mean()),
        "cv_accuracy_std": float(cv.std()),
    }


def main():
    OUT.mkdir(exist_ok=True)
    started = time.time()
    df, rows_raw = load()
    df_eng = add_engineered(df)
    raw_features = COLUMNS[:-1]
    eng_features = raw_features + ["age_chol_risk", "pressure_rate_ratio", "high_risk_profile"]

    y = df["target"]
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)

    def split(frame, cols):
        return (frame.loc[idx_train, cols], frame.loc[idx_test, cols],
                y.loc[idx_train], y.loc[idx_test], frame[cols], y)

    results = {}
    X_tr, X_te, y_tr, y_te, X_all, y_all = split(df, raw_features)
    results["random_forest_raw"] = score(
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2),
        X_tr, y_tr, X_te, y_te, X_all, y_all)
    X_tr, X_te, y_tr, y_te, X_all, y_all = split(df_eng, eng_features)
    rf_eng = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=2)
    results["random_forest_engineered"] = score(rf_eng, X_tr, y_tr, X_te, y_te, X_all, y_all)
    results["logistic_regression_engineered"] = score(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        X_tr, y_tr, X_te, y_te, X_all, y_all)
    results["mlp_engineered"] = score(
        make_pipeline(StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                    random_state=42, early_stopping=True)),
        X_tr, y_tr, X_te, y_te, X_all, y_all)

    rf_eng.fit(X_all, y_all)
    importances = sorted(zip(eng_features, rf_eng.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [n for n, _ in importances[:8]][::-1]
    vals = [v for _, v in importances[:8]][::-1]
    ax.barh(names, vals)
    ax.set_xlabel("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=110)
    plt.close(fig)

    metrics = {
        "dataset": {
            "name": "UCI Heart Disease, processed Cleveland file (dataset 45)",
            "source": "https://archive.ics.uci.edu/dataset/45/heart+disease",
            "license": "CC BY 4.0",
            "rows_raw": int(rows_raw),
            "rows_after_dropping_missing": int(len(df)),
            "features_raw": raw_features,
            "target": "num > 0 (angiographic heart disease present), binarised",
            "positive_rate": float(y.mean()),
            "note": "Real clinical data collected at the Cleveland Clinic, published by UCI. Small dataset: 303 rows before dropping missing values.",
        },
        "split": "stratified 80/20, seed 42, fixed before fitting; plus 5 fold stratified cross validation for every model",
        "engineered_features": {
            "age_chol_risk": "age * chol / 1000",
            "pressure_rate_ratio": "trestbps / thalach",
            "high_risk_profile": "1 if cp == 4 and exang == 1 else 0",
        },
        "results": results,
        "feature_importance_engineered_model": [
            {"feature": n, "importance": float(v)} for n, v in importances
        ],
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "run_log.txt").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("top features:", importances[:3])


if __name__ == "__main__":
    main()
