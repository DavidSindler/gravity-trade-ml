from pathlib import Path
import pandas as pd

FILES = [
    "results/feature_importance_baseline.csv",
    "results/feature_importance_interactions.csv",
    "results/feature_importance_eu_only_baseline.csv",
    "results/feature_importance_eu_only_interactions.csv",
]

KEEP_MODELS = {"RandomForest", "XGBoost"}


def main():
    dfs = []

    for f in FILES:
        p = Path(f)
        if not p.exists():
            print(f"Missing: {f}")
            continue

        df = pd.read_csv(p)
        df = df[df["model"].isin(KEEP_MODELS)].copy()
        df["experiment"] = p.stem.replace("feature_importance_", "")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No feature-importance files found. Run main.py first.")

    all_df = pd.concat(dfs, ignore_index=True)

    # Top 10 features per (experiment, model)
    all_df = all_df.sort_values(
        ["experiment", "model", "importance"],
        ascending=[True, True, False],
    )
    top10 = all_df.groupby(["experiment", "model"]).head(10)

    # Save long format
    out_long = "results/feature_importance_summary_top10.csv"
    top10.to_csv(out_long, index=False)

    # Save wide (pivot) format
    pivot = top10.pivot_table(
        index="feature",
        columns=["experiment", "model"],
        values="importance",
        aggfunc="mean",
    ).fillna(0.0)

    out_wide = "results/feature_importance_summary_wide.csv"
    pivot.to_csv(out_wide)

    print(f"Saved: {out_long}")
    print(f"Saved: {out_wide}")


if __name__ == "__main__":
    main()
