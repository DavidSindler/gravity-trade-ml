from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_gravity_data(
    path: str | Path,
    start_year: int = 2000,
    end_year: int = 2020,
) -> pd.DataFrame:
    """
    Load gravity dataset and construct a clean modeling dataset for log(exports).
    """
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)

    # Keep years of interest
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    # Build target: exports (prefer BACI if available)
    if "tradeflow_baci" in df.columns:
        df["exports"] = df["tradeflow_baci"]
    elif "tradeflow_comtrade_d" in df.columns:
        df["exports"] = df["tradeflow_comtrade_d"]
    else:
        df["exports"] = df["tradeflow_comtrade_o"]

    keep = [
        "year",
        "iso3_o",
        "iso3_d",
        "dist",
        "contig",
        "comlang_off",
        "pop_o",
        "pop_d",
        "gdp_o",
        "gdp_d",
        "exports",
        "eu_o",
        "eu_d",
    ]
    df = df[keep].copy()

    # Coerce numeric where needed
    for c in ["dist", "pop_o", "pop_d", "gdp_o", "gdp_d", "exports", "contig", "comlang_off", "eu_o", "eu_d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop missing/invalid core vars
    df = df.dropna(subset=["exports", "dist", "gdp_o", "gdp_d", "pop_o", "pop_d"])
    df = df[df["exports"] > 0].copy()
    df = df[df["dist"] > 0].copy()
    df = df[(df["gdp_o"] > 0) & (df["gdp_d"] > 0)].copy()
    df = df[(df["pop_o"] > 0) & (df["pop_d"] > 0)].copy()

    # Logs
    df["log_exports"] = np.log(df["exports"])
    df["log_dist"] = np.log(df["dist"])
    df["log_gdp_o"] = np.log(df["gdp_o"])
    df["log_gdp_d"] = np.log(df["gdp_d"])
    df["log_pop_o"] = np.log(df["pop_o"])
    df["log_pop_d"] = np.log(df["pop_d"])

    # Interaction features (gravity-inspired)
    df["log_gdp_o_x_log_gdp_d"] = df["log_gdp_o"] * df["log_gdp_d"]
    df["log_dist_x_contig"] = df["log_dist"] * df["contig"]
    df["log_dist_x_comlang"] = df["log_dist"] * df["comlang_off"]

    # Final dataset
    df = df[
        [
            "year",
            "iso3_o",
            "iso3_d",
            "eu_o",
            "eu_d",
            "contig",
            "comlang_off",
            "log_exports",
            "log_dist",
            "log_gdp_o",
            "log_gdp_d",
            "log_pop_o",
            "log_pop_d",
            "log_gdp_o_x_log_gdp_d",
            "log_dist_x_contig",
            "log_dist_x_comlang",
        ]
    ].copy()

    return df


def temporal_split(
    df: pd.DataFrame,
    train_end_year: int = 2016,
    test_start_year: int = 2017,
    *,
    target: str = "log_exports",
    use_interactions: bool = True,
) -> SplitData:
    """
    Temporal split to avoid leakage.
    Drops rows with missing values in the selected feature set.
    """
    base_features = [
        "log_gdp_o",
        "log_gdp_d",
        "log_dist",
        "contig",
        "comlang_off",
        "log_pop_o",
        "log_pop_d",
    ]

    interaction_features = [
        "log_gdp_o_x_log_gdp_d",
        "log_dist_x_contig",
        "log_dist_x_comlang",
    ]

    feature_cols = base_features + (interaction_features if use_interactions else [])

    # Drop rows where any selected feature or target is missing
    needed = feature_cols + [target, "year"]
    clean = df.dropna(subset=needed).copy()

    train_df = clean[clean["year"] <= train_end_year].copy()
    test_df = clean[clean["year"] >= test_start_year].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Temporal split produced empty set. Train rows: {len(train_df)}, Test rows: {len(test_df)}."
        )

    return SplitData(
        X_train=train_df[feature_cols],
        y_train=train_df[target],
        X_test=test_df[feature_cols],
        y_test=test_df[target],
    )


def filter_eu_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only EU->EU pairs using eu_o and eu_d indicators.
    """
    # if eu_o/eu_d missing, they will be NaN -> filtered out
    return df[(df["eu_o"] == 1) & (df["eu_d"] == 1)].copy()
