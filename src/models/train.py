"""
Training pipeline with MLflow experiment tracking.
Run this script to train and log all models in one shot.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .evaluate import evaluate_model, compare_models


EXPERIMENT_NAME = "geopolitical_credit_risk"
DATA_PATH = Path("data/processed/master_dataset.csv")


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path, parse_dates=["year"])
    return df


def get_feature_sets(df: pd.DataFrame):
    """
    Returns two feature sets:
      - financial_only : baseline (no geo features)
      - augmented      : + sGPR + macro transmission features
    """
    financial_cols = [
        "leverage_ratio", "current_ratio", "roa",
        "interest_coverage", "revenue_growth", "asset_turnover"
    ]
    geo_macro_cols = [
        c for c in df.columns
        if c.startswith("sgpr_") or c.startswith("macro_")
    ]
    return {
        "financial_only": financial_cols,
        "augmented": financial_cols + geo_macro_cols
    }


def train_and_log(df: pd.DataFrame, feature_set_name: str, features: list[str]):
    """Train logistic regression + XGBoost, log everything to MLflow."""
    X = df[features].fillna(df[features].median())
    y = df["default_label"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Logistic Regression ──────────────────────────────────
    with mlflow.start_run(run_name=f"logistic_{feature_set_name}"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        metrics = evaluate_model(y, y_prob, f"logistic_{feature_set_name}")
        mlflow.log_params({"model": "LogisticRegression", "features": feature_set_name})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        lr.fit(X_scaled, y)
        mlflow.sklearn.log_model(lr, "model")
        print(f"  LogReg {feature_set_name}: AUC={metrics['auc_roc']:.4f}, KS={metrics['ks_statistic']:.4f}")

    # ── XGBoost ─────────────────────────────────────────────
    with mlflow.start_run(run_name=f"xgboost_{feature_set_name}"):
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            eval_metric="auc", random_state=42, verbosity=0
        )
        y_prob = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba")[:, 1]
        metrics = evaluate_model(y, y_prob, f"xgboost_{feature_set_name}")
        mlflow.log_params({"model": "XGBoost", "features": feature_set_name,
                           "n_estimators": 300, "max_depth": 4})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        xgb.fit(X, y)
        mlflow.xgboost.log_model(xgb, "model")
        print(f"  XGBoost {feature_set_name}: AUC={metrics['auc_roc']:.4f}, KS={metrics['ks_statistic']:.4f}")

    return metrics


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    feature_sets = get_feature_sets(df)
    all_results = []
    for name, features in feature_sets.items():
        print(f"\nTraining on feature set: {name}")
        result = train_and_log(df, name, features)
        all_results.append(result)
    print("\n── Results Summary ──")
    print(compare_models(all_results).to_string())
