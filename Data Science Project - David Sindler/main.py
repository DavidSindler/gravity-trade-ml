from pathlib import Path
import pandas as pd

from src.data_loader import load_gravity_data, temporal_split, filter_eu_pairs
from src.models import get_models
from src.evaluation import (
    evaluate_regression,
    save_metrics_table,
    save_pred_vs_actual_plot,
    get_feature_importance,
    save_feature_importance_plot,
    save_residual_diagnostics,
)


def run_experiment(
    df: pd.DataFrame,
    label: str,
    use_interactions: bool,
    train_end_year: int,
    test_start_year: int,
):
    split = temporal_split(
        df,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        use_interactions=use_interactions,
    )

    models = get_models(random_state=42)

    metrics = []
    best_name, best_rmse, best_pred = None, float("inf"), None
    feature_cols = list(split.X_train.columns)
    all_importances = []

    print(
        f"\n=== Experiment: {label} | "
        f"train<= {train_end_year}, test>= {test_start_year}, "
        f"interactions={use_interactions} ==="
    )

    for name, model in models.items():
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)

        m = evaluate_regression(split.y_test, y_pred)
        m["model"] = name
        metrics.append(m)

        print(f"{name:>16} | RMSE={m['rmse']:.4f} | R2={m['r2']:.4f}")

        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_name = name
            best_pred = y_pred

        if hasattr(model, "feature_importances_"):
            imp_df = get_feature_importance(model, feature_cols)
            imp_df["model"] = name
            all_importances.append(imp_df)

            save_feature_importance_plot(
                imp_df,
                out_path=f"results/feature_importance_{name}_{label}.png",
                title=f"Feature Importance — {name} ({label})",
                top_k=15,
            )

    metrics_path = f"results/metrics_{label}.csv"
    save_metrics_table(metrics, metrics_path)
    print(f"Saved: {metrics_path}")

    if best_pred is not None:
        save_pred_vs_actual_plot(
            split.y_test,
            best_pred,
            f"results/pred_vs_actual_{best_name}_{label}.png",
            title=f"Predicted vs Actual — {best_name} ({label})",
        )

        save_residual_diagnostics(
            y_true=split.y_test,
            y_pred=best_pred,
            out_prefix=f"results/residuals_{label}",
            title_prefix=f"{best_name} ({label})",
        )

    if all_importances:
        pd.concat(all_importances).to_csv(
            f"results/feature_importance_{label}.csv", index=False
        )


def main():
    data_path = Path("data/raw/gravity_trade.csv")
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = load_gravity_data(data_path, start_year=2000, end_year=2020)
    df_eu = filter_eu_pairs(df)

    
    # MAIN SPLIT (primary results)
   
    run_experiment(
        df,
        label="baseline",
        use_interactions=False,
        train_end_year=2016,
        test_start_year=2017,
    )

    run_experiment(
        df,
        label="interactions",
        use_interactions=True,
        train_end_year=2016,
        test_start_year=2017,
    )

    run_experiment(
        df_eu,
        label="eu_only_baseline",
        use_interactions=False,
        train_end_year=2016,
        test_start_year=2017,
    )

    run_experiment(
        df_eu,
        label="eu_only_interactions",
        use_interactions=True,
        train_end_year=2016,
        test_start_year=2017,
    )

    
    # ENHANCEMENT 5: Alternative temporal split (robustness)
   
    run_experiment(
        df,
        label="baseline_alt_split",
        use_interactions=False,
        train_end_year=2012,
        test_start_year=2013,
    )

    run_experiment(
        df_eu,
        label="eu_only_baseline_alt_split",
        use_interactions=False,
        train_end_year=2012,
        test_start_year=2013,
    )

    print("\n All experiments completed.")


if __name__ == "__main__":
    main()
