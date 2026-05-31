"""
Model evaluation utilities.
Consistent metrics used across all models in the project.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    roc_curve, classification_report
)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic — standard credit risk discriminator.
    Higher is better. Industry benchmark: >40% is good for corporate PD.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "") -> dict:
    """
    Run the full credit risk model evaluation suite.

    Returns a dict of metrics suitable for MLflow logging and
    comparison tables in the paper.
    """
    metrics = {
        "model": model_name,
        "auc_roc": roc_auc_score(y_true, y_prob),
        "ks_statistic": ks_statistic(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }
    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of evaluate_model dicts.
    Ready to export as the results table in the paper.
    """
    df = pd.DataFrame(results).set_index("model")
    df["auc_roc"] = df["auc_roc"].round(4)
    df["ks_statistic"] = df["ks_statistic"].round(4)
    df["brier_score"] = df["brier_score"].round(4)
    return df
