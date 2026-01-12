from __future__ import annotations

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_models(random_state: int = 42):
    """
    Return a dictionary of regression models to compare.
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
