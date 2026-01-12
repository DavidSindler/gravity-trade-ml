Predicting European Export Flows with Machine Learning
=====================================================

This project predicts bilateral export flows between European countries using
classical gravity-model variables and modern machine learning methods.

The goal is to compare traditional linear regression approaches with tree-based
ensemble models (Random Forest and XGBoost) and assess whether machine learning
methods improve out-of-sample predictive performance.


Project Structure
-----------------


├── main.py                         Entry point (run this)

├── requirements.txt                Python dependencies

├── README.md

├── PROPOSAL.md

├── src/

│   ├── data_loader.py              Data loading and preprocessing

│   ├── models.py                   Model definitions

│   ├── evaluation.py               Evaluation metrics and plots

│   └── feature_importance_summary.py

├── data/

│   └── raw/                        Raw gravity trade data

├── results/

│   ├── metrics_*.csv               Model comparison tables

│   ├── pred_vs_actual_*.png        Predicted vs. actual plots

│   ├── residuals_*_*.png           Residual diagnostics

│   ├── feature_importance_*.csv    Feature importance tables

│   └── feature_importance_*.png    Feature importance plots

└── notebooks/                      Optional exploratory analysis


How to Run
----------

1. Install dependencies

   pip install -r requirements.txt

2. Run the full pipeline

   python main.py

This script:
- loads and preprocesses the gravity trade dataset
- performs temporal train/test splits to avoid data leakage
- trains multiple regression and machine learning models
- evaluates models using RMSE and R²
- generates tables and figures in the `results/` folder


Models Implemented
------------------

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor


Outputs
-------

Running `python main.py` produces:

- results/metrics_*.csv  
  Out-of-sample model comparison tables

- results/pred_vs_actual_*.png  
  Predicted vs. actual plots for the best-performing models

- results/residuals_*_*.png  
  Residual diagnostics

- results/feature_importance_*.csv and .png  
  Feature importance analysis for tree-based models


Post-processing
---------------

To generate summary tables comparing feature importance across models
and samples, run:

python src/feature_importance_summary.py


Reproducibility
---------------

- Fixed random seeds are used throughout
- Temporal splitting avoids look-ahead bias
- All results can be reproduced by running `python main.py`
