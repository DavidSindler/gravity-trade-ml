from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------
# Basic evaluation metrics
# ------------------------

def evaluate_regression(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "r2": r2}


def save_metrics_table(metrics_rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_rows).sort_values(by="rmse")
    df.to_csv(out_path, index=False)


# ------------------------
# Prediction diagnostics
# ------------------------

def save_pred_vs_actual_plot(
    y_true,
    y_pred,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_true, y_pred, s=4, alpha=0.6)
    plt.xlabel("Actual log(exports)")
    plt.ylabel("Predicted log(exports)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residual_diagnostics(
    y_true,
    y_pred,
    out_prefix: str | Path,
    title_prefix: str,
) -> None:
    """
    Save residual diagnostics:
    1) Residuals vs Fitted
    2) Histogram of residuals
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    # Residuals vs fitted
    plt.figure()
    plt.scatter(y_pred, residuals, s=4, alpha=0.6)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"{title_prefix} — Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residuals_vs_fitted.png", dpi=200)
    plt.close()

    # Histogram of residuals
    plt.figure()
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"{title_prefix} — Residual Distribution")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residuals_hist.png", dpi=200)
    plt.close()


# ------------------------
# Feature importance
# ------------------------

def get_feature_importance(model, feature_names):
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    top_k: int = 15,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = importance_df.head(top_k)

    plt.figure()
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
