Project Proposal
================

Title
-----

Predicting European Bilateral Export Flows Using Machine Learning


Motivation
----------

International trade flows are traditionally modeled using the gravity framework,
which relates bilateral trade to economic size, distance, and historical or
institutional ties. While gravity models perform well empirically, they rely on
strong functional-form assumptions and linear specifications.

Recent advances in machine learning offer flexible alternatives that may capture
non-linearities and interaction effects ignored by classical models. This project
aims to evaluate whether modern machine learning methods can improve the
out-of-sample predictive performance of gravity models of trade, while maintaining
economic interpretability.


Research Question
-----------------

Can machine learning models outperform traditional gravity regressions in predicting
bilateral export flows between European countries, particularly in out-of-sample
settings and under alternative sample definitions and temporal splits?


Data
----

The project uses a large panel gravity dataset covering bilateral trade flows across
countries and years. The dataset includes standard gravity variables such as:

- Bilateral distance
- Exporter and importer GDP
- Exporter and importer population
- Common language indicators
- Contiguity indicators
- Bilateral export flows

The analysis focuses on the period 2000–2020, with both full-sample and EU-only
subsamples considered. Temporal train/test splits are used to avoid data leakage.


Methodology
-----------

The empirical analysis proceeds in several steps:

1. Data preprocessing and log-transformations of trade flows and gravity variables.
2. Temporal train/test splits to ensure realistic forecasting evaluation.
3. Estimation of benchmark linear models:
   - Ordinary Least Squares
   - Ridge Regression
   - Lasso Regression
4. Estimation of non-linear machine learning models:
   - Random Forest Regressor
   - XGBoost Regressor
5. Comparison of predictive performance using RMSE and R².
6. Robustness checks using:
   - Interaction terms
   - EU-only subsamples
   - Alternative temporal split points
7. Model diagnostics including residual analysis and feature importance evaluation.


Expected Contributions
----------------------

This project contributes by:

- Providing a systematic comparison between classical gravity models and modern
  machine learning approaches.
- Evaluating model performance under multiple robustness scenarios.
- Demonstrating the trade-offs between predictive accuracy and interpretability.
- Offering reproducible and extensible code suitable for further research.


Expected Outputs
----------------

The project will produce:

- Model comparison tables with out-of-sample performance metrics.
- Predicted vs actual trade flow plots.
- Residual diagnostic figures.
- Feature importance analyses for tree-based models.
- A fully reproducible empirical pipeline.


Timeline
--------

- Data preparation and baseline models: completed
- Machine learning extensions and robustness checks: completed
- Interpretation and report writing: ongoing


Reproducibility
---------------

All results are fully reproducible using the provided codebase. Running

python main.py

recreates all tables and figures used in the analysis.


Conclusion
----------

By combining economic theory with machine learning tools, this project aims to
assess whether flexible algorithms can enhance predictive performance in a
well-established empirical setting, while retaining economic insight.
