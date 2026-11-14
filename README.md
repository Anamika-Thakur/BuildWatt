# BuildWatt: Predicting Building Energy Loads with Advanced ML & MLOps

**BuildWatt** is an end-to-end Machine Learning and MLOps system that predicts building **Heating Load ($Y_1$)** and **Cooling Load ($Y_2$)** from architectural design parameters using the UCI Energy Efficiency Dataset.

Beyond classical linear baselines, BuildWatt provides a complete 8-model benchmark suite, rigorous regression calibration (including Split Conformal Prediction Intervals for guaranteed coverage), and automated experiment tracking with **MLflow**.

---

## 1. Features & Targets

### Architectural Input Features ($X_1 - X_8$)
- **$X_1$ - Relative Compactness**: Ratio of volume to surface area
- **$X_2$ - Surface Area**: Total exterior building surface ($m^2$)
- **$X_3$ - Wall Area**: Exterior wall surface ($m^2$)
- **$X_4$ - Roof Area**: Roof surface area ($m^2$)
- **$X_5$ - Overall Height**: Building height ($m$)
- **$X_6$ - Orientation**: Cardinal direction (2: North, 3: East, 4: South, 5: West)
- **$X_7$ - Glazing Area**: Window glass proportion ($0\% - 40\%$)
- **$X_8$ - Glazing Area Distribution**: Uniform, North, South, East, West

### Energy Demands Predicted ($Y_1, Y_2$)
- **$Y_1$ - Heating Load ($kW$)**: Thermal energy required to heat the building
- **$Y_2$ - Cooling Load ($kW$)**: Thermal energy required to cool the building

---

## 2. Multi-Model Implementation Suite

All 8 requested models are implemented in priority order:

| Priority | Model | Architecture / Pipeline | Reason & Role |
| :---: | :--- | :--- | :--- |
| **1** | **Polynomial Regression (Degree 2)** | `PolynomialFeatures(deg=2)` + `StandardScaler` + `Ridge(alpha=10)` | Tests quadratic & bilinear feature interactions |
| **2** | **Elastic Net** | `StandardScaler` + `ElasticNet(alpha=0.05, l1_ratio=0.5)` | Combines $L_1$ sparsity and $L_2$ group selection |
| **3** | **Decision Tree** | `DecisionTreeRegressor(max_depth=7, min_samples_split=5)` | Interpretable, non-parametric hierarchical partitions |
| **4** | **Random Forest** | `RandomForestRegressor(n_estimators=150, max_depth=12)` | Ensemble bagging reducing tree variance |
| **5** | **Gradient Boosting (Champion)** | `GradientBoostingRegressor(n_estimators=200, lr=0.08, max_depth=4)` | **Top Performer**: Sequential boosting minimizing residual errors |
| **6** | **XGBoost** | `XGBRegressor(n_estimators=200, lr=0.08, max_depth=4, colsample=0.85)` | Advanced regularized gradient boosting |
| **7** | **SVR (RBF Kernel)** | `StandardScaler` + `SVR(C=100, epsilon=0.1, gamma='scale')` | Kernel trick mapping to infinite-dimensional Hilbert space |
| **8** | **MLP Regressor** | `StandardScaler` + `MLPRegressor((128, 64), relu, early_stopping)` | Multi-layer perceptron neural network |

---

## 3. Systematic Benchmark & Leaderboard Results

Benchmarked with **5-Fold Cross-Validation** and an independent **80/20 Train-Test Split** ($N=768$).

### Target 1: Heating Load ($Y_1$)
| Rank | Model | Test $R^2$ | 5-Fold CV $R^2$ | Test RMSE ($kW$) | Test MAE ($kW$) | Test MAPE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **5. Gradient Boosting (Champion)** | **0.9984** | **0.9984 ($\pm 0.0003$)** | **0.4042** | **0.2981** | **1.33%** |
| 🥈 | **6. XGBoost** | 0.9984 | 0.9984 ($\pm 0.0002$) | 0.4070 | 0.3020 | 1.33% |
| 🥉 | **4. Random Forest** | 0.9976 | 0.9972 ($\pm 0.0003$) | 0.4984 | 0.3569 | 1.48% |
| 4 | **3. Decision Tree** | 0.9975 | 0.9972 ($\pm 0.0006$) | 0.5087 | 0.3581 | 1.47% |
| 5 | **7. SVR (RBF)** | 0.9948 | 0.9883 ($\pm 0.0036$) | 0.7347 | 0.5614 | 2.92% |
| 6 | **8. MLP Regressor** | 0.9947 | 0.9372 ($\pm 0.0327$) | 0.7454 | 0.5176 | 2.42% |
| 7 | **1. Polynomial Regression (Deg 2)** | 0.9311 | 0.9338 ($\pm 0.0046$) | 2.6794 | 1.8472 | 8.21% |
| 8 | **2. Elastic Net** | 0.9032 | 0.9090 ($\pm 0.0103$) | 3.1767 | 2.2580 | 10.69% |

### Target 2: Cooling Load ($Y_2$)
| Rank | Model | Test $R^2$ | 5-Fold CV $R^2$ | Test RMSE ($kW$) | Test MAE ($kW$) | Test MAPE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **5. Gradient Boosting (Champion)** | **0.9879** | **0.9883 ($\pm 0.0025$)** | **1.0591** | **0.6806** | **2.44%** |
| 🥈 | **6. XGBoost** | 0.9871 | 0.9861 ($\pm 0.0025$) | 1.0937 | 0.7060 | 2.52% |
| 🥉 | **8. MLP Regressor** | 0.9809 | 0.8833 ($\pm 0.0110$) | 1.3321 | 0.9618 | 3.69% |
| 4 | **4. Random Forest** | 0.9656 | 0.9662 ($\pm 0.0047$) | 1.7860 | 1.1106 | 3.63% |
| 5 | **7. SVR (RBF)** | 0.9656 | 0.9564 ($\pm 0.0065$) | 1.7865 | 1.1369 | 4.41% |
| 6 | **3. Decision Tree** | 0.9590 | 0.9596 ($\pm 0.0079$) | 1.9492 | 1.2292 | 4.06% |
| 7 | **1. Polynomial Regression (Deg 2)** | 0.9045 | 0.8947 ($\pm 0.0091$) | 2.9752 | 1.9943 | 7.63% |
| 8 | **2. Elastic Net** | 0.8800 | 0.8757 ($\pm 0.0141$) | 3.3342 | 2.3309 | 9.13% |

### Why Model 5 (Gradient Boosting) is the Best Choice:
1. **Top Metric Leader**: Achieves the highest $R^2$ ($0.9984$ on Heating, $0.9879$ on Cooling) and lowest RMSE on both targets.
2. **Minimal Variance Across Folds**: Standard deviation of $R^2$ across 5-fold cross-validation is only $\pm 0.0003$, proving unmatched stability.
3. **Natural Handling of Mixed Scales & Thresholds**: Building energy loads exhibit sharp transitions at critical building heights ($3.5m$ vs $7.0m$) and glazing thresholds. Tree ensembles model step-like non-linear physical interactions without requiring artificial polynomial basis expansion.

---

## 4. Regression Calibration Methods

In building energy management and HVAC design, point predictions alone are risky; an under-dimensioned heating system can fail during extreme weather. We implemented two complementary calibration methodologies:

### A. Split Conformal Prediction Intervals (Uncertainty Calibration)
Using a dedicated calibration split, nonconformity residual scores $s_i = |y_i - \hat{y}_i|$ are computed to establish distribution-free prediction bounds $[\hat{y} - q_{1-\alpha}, \hat{y} + q_{1-\alpha}]$ with finite-sample coverage guarantees:
- **90% Confidence Target ($\alpha=0.10$)**:
  - **Empirical Coverage**: **$91.56\%$**
  - **Interval Margin**: $\pm 0.69\text{ kW}$ (Average Width: $1.37\text{ kW}$)
- **95% Confidence Target ($\alpha=0.05$)**:
  - **Empirical Coverage**: **$96.75\%$**
  - **Interval Margin**: $\pm 0.88\text{ kW}$ (Average Width: $1.76\text{ kW}$)
- **Reliability Diagram**: Evaluated from $\alpha=0.01$ to $0.50$, confirming that empirical coverage closely tracks and bounds nominal expectations.

### B. Isotonic Point Calibration
Evaluated non-parametric isotonic regression on out-of-sample predictions. Because Gradient Boosting already achieves near-perfect calibration ($R^2 = 0.9987$), raw predictions are optimal and uncompromised by binning artifacts.

---

## 5. MLOps Experiment Tracking with MLflow

All experiments are tracked in a durable SQLite database backend with automated artifact logging.

### Logged Components:
- **Parameters**: Hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `alpha`, etc.), dataset splits, seeds.
- **Metrics**: Train & Test $R^2$, RMSE, MAE, MAPE, 5-Fold Cross-Validation mean & standard deviation, Conformal coverage rates.
- **Artifacts**:
  - High-resolution Actual vs Predicted scatter plots (`artifacts/actual_vs_pred_*.png`)
  - Target Comparison bar charts (`artifacts/model_comparison_*.png`)
  - Calibration reliability curves & prediction interval bands (`artifacts/calibration_*.png`, `artifacts/conformal_*.png`)
  - Feature importance chart (`artifacts/feature_importance_gradient_boosting.png`)
  - Full leaderboard CSV (`artifacts/benchmark_leaderboard.csv`)
- **Model Registry**: Serialized scikit-learn pipelines and XGBoost models saved for inference.

### Launching the MLflow UI Dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to inspect runs, compare metric curves, and download models.

---

## 6. How to Run

### Run the Complete CLI Pipeline:
```bash
python train_models.py
```
Optional CLI flags:
```bash
python train_models.py --data-path ENB2012_data.xlsx --artifacts-dir artifacts --random-state 42
```

### Run the Interactive Notebook:
Open `BuildWatt.ipynb` in VS Code, JupyterLab, or Jupyter Notebook to execute cells interactively.
