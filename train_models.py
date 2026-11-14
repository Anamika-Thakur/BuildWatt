"""
BuildWatt - Model Training, Benchmarking, Calibration & MLflow Tracking Pipeline
================================================================================
Comprehensive Machine Learning suite for predicting Building Energy Loads:
- Heating Load (Y1)
- Cooling Load (Y2)

Implements 8 models ranked by priority:
1. Polynomial Regression (Degree 2) - Tests nonlinear relationships
2. Elastic Net - Compares regularization approaches (L1 & L2)
3. Decision Tree - Provides interpretability
4. Random Forest - Captures nonlinear interactions via bagging
5. Gradient Boosting (Champion) - Strong tabular-data baseline via sequential boosting
6. XGBoost - Advanced regularized gradient boosting
7. SVR (RBF) - Kernel-based nonlinear regression
8. MLP Regressor - Multi-layer perceptron neural network

Includes:
- Regression Calibration methods:
  * Split Conformal Prediction Intervals (guaranteed finite-sample coverage)
  * Quantile Gradient Boosting (lower, median, upper bounds)
  * Isotonic Regression residual calibration
- Full MLOps tracking via MLflow (SQLite backend store + artifact logging)
"""

import os
import sys
import warnings
import argparse

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

FEATURE_NAMES = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area',
    'Glazing_Area_Distribution'
]

TARGET_NAMES = {
    'heating': 'Heating_Load',
    'cooling': 'Cooling_Load'
}


def load_dataset(data_path="D:/BuildWatt/ENB2012_data.xlsx"):
    """Load and format the UCI Energy Efficiency dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_excel(data_path)
    # Ensure correct column naming
    column_mapping = {
        'X1': 'Relative_Compactness',
        'X2': 'Surface_Area',
        'X3': 'Wall_Area',
        'X4': 'Roof_Area',
        'X5': 'Overall_Height',
        'X6': 'Orientation',
        'X7': 'Glazing_Area',
        'X8': 'Glazing_Area_Distribution',
        'Y1': 'Heating_Load',
        'Y2': 'Cooling_Load'
    }
    if all(col in df.columns for col in ['X1', 'X2', 'Y1', 'Y2']):
        df = df.rename(columns=column_mapping)
    else:
        df.columns = FEATURE_NAMES + ['Heating_Load', 'Cooling_Load']
        
    return df


def get_model_registry(random_state=42):
    """
    Defines all 8 models as requested in the specification.
    Returns a dict mapping model names to instantiated estimators / pipelines.
    """
    models = {
        "1. Polynomial Regression (Deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0, random_state=random_state))
        ]),
        "2. Elastic Net": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=2000, random_state=random_state))
        ]),
        "3. Decision Tree": DecisionTreeRegressor(
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        ),
        "4. Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=3,
            random_state=random_state,
            n_jobs=-1
        ),
        "5. Gradient Boosting (Champion)": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        ),
        "6. XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=0
        ),
        "7. SVR (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=100.0, epsilon=0.1, gamma='scale'))
        ]),
        "8. MLP Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state
            ))
        ])
    }
    return models


def evaluate_predictions(y_true, y_pred):
    """Compute comprehensive regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }


class ConformalPredictor:
    """
    Split Conformal Prediction for regression.
    Provides finite-sample valid prediction intervals [y_hat - q, y_hat + q]
    such that P(y in Interval) >= 1 - alpha for any desired confidence level.
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.cal_residuals = None

    def fit_and_calibrate(self, X_train, y_train, X_cal, y_cal):
        self.base_estimator.fit(X_train, y_train)
        cal_preds = self.base_estimator.predict(X_cal)
        self.cal_residuals = np.abs(np.array(y_cal) - cal_preds)
        return self

    def predict_intervals(self, X_test, alpha=0.10):
        """
        Returns (y_pred, lower_bound, upper_bound, q_val)
        alpha=0.10 corresponds to 90% target coverage.
        """
        y_pred = self.base_estimator.predict(X_test)
        n = len(self.cal_residuals)
        # Conformal quantile calculation with finite sample correction
        p_level = np.ceil((n + 1) * (1 - alpha)) / n
        p_level = min(max(p_level, 0.0), 1.0)
        q_val = np.quantile(self.cal_residuals, p_level)
        
        lower = y_pred - q_val
        upper = y_pred + q_val
        return y_pred, lower, upper, q_val

    def evaluate_calibration_curve(self, X_test, y_test, alpha_levels=None):
        """
        Evaluates empirical coverage across various confidence levels.
        Returns a DataFrame comparing Expected Coverage vs Empirical Coverage.
        """
        if alpha_levels is None:
            alpha_levels = np.linspace(0.01, 0.50, 25)
        
        y_test = np.array(y_test)
        results = []
        for a in alpha_levels:
            expected_coverage = (1 - a) * 100
            y_pred, lower, upper, q_val = self.predict_intervals(X_test, alpha=a)
            empirical_coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
            avg_width = np.mean(upper - lower)
            results.append({
                "Significance_Alpha": a,
                "Expected_Coverage": expected_coverage,
                "Empirical_Coverage": empirical_coverage,
                "Avg_Interval_Width": avg_width,
                "Coverage_Gap": empirical_coverage - expected_coverage
            })
        return pd.DataFrame(results)


def run_benchmarks(df, artifacts_dir="D:/BuildWatt/artifacts", test_size=0.2, random_state=42):
    """
    Trains and benchmarks all 8 models on both Heating Load and Cooling Load.
    Logs metrics, plots, and models to MLflow.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Configure MLflow SQLite tracking backend
    db_path = "sqlite:///D:/BuildWatt/mlflow.db"
    mlflow.set_tracking_uri(db_path)
    experiment_name = "BuildWatt_Energy_Prediction"
    mlflow.set_experiment(experiment_name)
    
    X = df[FEATURE_NAMES]
    targets = {
        "Heating_Load": df["Heating_Load"],
        "Cooling_Load": df["Cooling_Load"]
    }
    
    leaderboard_records = []
    
    for target_name, y in targets.items():
        print(f"\n=======================================================")
        print(f"  BENCHMARKING 8 MODELS ON TARGET: {target_name.upper()}")
        print(f"=======================================================\n")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        models = get_model_registry(random_state=random_state)
        
        test_predictions_dict = {}
        
        for model_name, model in models.items():
            print(f"--> Training {model_name}...")
            
            # 5-Fold Cross Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
            cv_r2_mean = float(np.mean(cv_scores))
            cv_r2_std = float(np.std(cv_scores))
            
            # Fit on full training set
            model.fit(X_train, y_train)
            
            # Predict
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            test_predictions_dict[model_name] = test_preds
            
            # Metrics
            train_metrics = evaluate_predictions(y_train, train_preds)
            test_metrics = evaluate_predictions(y_test, test_preds)
            
            record = {
                "Target": target_name,
                "Model": model_name,
                "Train_R2": train_metrics["R2"],
                "Test_R2": test_metrics["R2"],
                "CV_R2_Mean": cv_r2_mean,
                "CV_R2_Std": cv_r2_std,
                "Test_RMSE": test_metrics["RMSE"],
                "Test_MAE": test_metrics["MAE"],
                "Test_MAPE": test_metrics["MAPE"]
            }
            leaderboard_records.append(record)
            
            print(f"    Test R²: {test_metrics['R2']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f} | Test MAE: {test_metrics['MAE']:.4f} | CV R²: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
            
            # Start MLflow run
            safe_run_name = f"{target_name}_{model_name.split('.')[1].strip().replace(' ', '_')}"
            with mlflow.start_run(run_name=safe_run_name):
                # Tags
                mlflow.set_tag("target", target_name)
                mlflow.set_tag("model_family", model_name)
                mlflow.set_tag("champion", "true" if "Gradient Boosting" in model_name else "false")
                
                # Parameters
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    # Filter basic params to avoid overwhelming MLflow
                    filtered_params = {k: v for k, v in params.items() if not callable(v) and len(str(v)) < 150}
                    mlflow.log_params(filtered_params)
                
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                
                # Metrics
                mlflow.log_metric("train_r2", train_metrics["R2"])
                mlflow.log_metric("test_r2", test_metrics["R2"])
                mlflow.log_metric("test_rmse", test_metrics["RMSE"])
                mlflow.log_metric("test_mae", test_metrics["MAE"])
                mlflow.log_metric("test_mape", test_metrics["MAPE"])
                mlflow.log_metric("cv_r2_mean", cv_r2_mean)
                mlflow.log_metric("cv_r2_std", cv_r2_std)
                
                # Plot Actual vs Predicted for this model
                fig_avp, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(y_test, test_preds, alpha=0.7, edgecolors='k', s=40, color='#1f77b4')
                lims = [min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())]
                ax.plot(lims, lims, 'r--', lw=2, label="Perfect Fit (1:1)")
                ax.set_xlabel(f"Actual {target_name.replace('_', ' ')} (kW)")
                ax.set_ylabel(f"Predicted {target_name.replace('_', ' ')} (kW)")
                ax.set_title(f"{model_name}\n$R^2$ = {test_metrics['R2']:.4f}, RMSE = {test_metrics['RMSE']:.4f}")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                fig_path = os.path.join(artifacts_dir, f"actual_vs_pred_{safe_run_name}.png")
                fig_avp.tight_layout()
                fig_avp.savefig(fig_path, dpi=200)
                plt.close(fig_avp)
                
                mlflow.log_artifact(fig_path)
                
                # Log model
                if "XGBoost" in model_name:
                    mlflow.xgboost.log_model(xgb_model=model, name="model")
                else:
                    mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="cloudpickle")
        
        # Plot comparative leaderboard for this target
        target_df = pd.DataFrame([r for r in leaderboard_records if r["Target"] == target_name])
        target_df = target_df.sort_values(by="Test_R2", ascending=True)
        
        fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['#2ca02c' if 'Gradient Boosting' in m else '#1f77b4' for m in target_df["Model"]]
        
        ax1.barh(target_df["Model"], target_df["Test_R2"], color=colors)
        ax1.set_xlabel("Test $R^2$ Score")
        ax1.set_title(f"Model Comparison: Test $R^2$ ({target_name})")
        ax1.set_xlim(min(0.85, target_df["Test_R2"].min() - 0.02), 1.002)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        ax2.barh(target_df["Model"], target_df["Test_RMSE"], color=colors)
        ax2.set_xlabel("Test RMSE (kW)")
        ax2.set_title(f"Model Comparison: Test RMSE ({target_name})")
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        fig_comp.suptitle(f"BuildWatt Benchmark - {target_name.replace('_', ' ')}", fontsize=14, fontweight='bold')
        fig_comp.tight_layout()
        comp_path = os.path.join(artifacts_dir, f"model_comparison_{target_name}.png")
        fig_comp.savefig(comp_path, dpi=200)
        plt.close(fig_comp)
        
        with mlflow.start_run(run_name=f"{target_name}_Summary_Comparison"):
            mlflow.log_artifact(comp_path)
            
    # Save overall leaderboard CSV
    leaderboard_df = pd.DataFrame(leaderboard_records)
    csv_path = os.path.join(artifacts_dir, "benchmark_leaderboard.csv")
    leaderboard_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Leaderboard saved to: {csv_path}")
    
    return leaderboard_df


def run_gradient_boosting_calibration(df, artifacts_dir="D:/BuildWatt/artifacts", random_state=42):
    """
    In-depth analysis of Model 5 (Gradient Boosting) as requested ("use 5 - that would be the best to use"):
    1. Feature Importance extraction
    2. Split Conformal Prediction Calibration (finite-sample coverage guarantees)
    3. Reliability diagram / coverage curve across alpha levels
    4. Quantile Gradient Boosting regression
    5. Isotonic Calibration comparison
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    db_path = "sqlite:///D:/BuildWatt/mlflow.db"
    mlflow.set_tracking_uri(db_path)
    mlflow.set_experiment("BuildWatt_Energy_Prediction")
    
    print("\n=======================================================")
    print("  MODEL 5 (GRADIENT BOOSTING) DEEP DIVE & CALIBRATION")
    print("=======================================================\n")
    
    X = df[FEATURE_NAMES]
    y_heat = df["Heating_Load"]
    y_cool = df["Cooling_Load"]
    
    # 3-Way Split for Rigorous Calibration: Train (60%), Calibration (20%), Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_heat, test_size=0.40, random_state=random_state)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)
    
    # Fit Champion Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=random_state
    )
    
    # 1. Conformal Predictor Setup
    cp = ConformalPredictor(gbr)
    cp.fit_and_calibrate(X_train, y_train, X_cal, y_cal)
    
    # Evaluate at 90% and 95% target confidence intervals
    preds, lower_90, upper_90, q_90 = cp.predict_intervals(X_test, alpha=0.10)
    _, lower_95, upper_95, q_95 = cp.predict_intervals(X_test, alpha=0.05)
    
    cov_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90)) * 100
    cov_95 = np.mean((y_test >= lower_95) & (y_test <= upper_95)) * 100
    width_90 = np.mean(upper_90 - lower_90)
    width_95 = np.mean(upper_95 - lower_95)
    
    print(f"--- Conformal Uncertainty Calibration (Heating Load) ---")
    print(f"Target 90% Confidence Interval: Empirical Coverage = {cov_90:.2f}%, Avg Width = {width_90:.2f} kW (Margin = +/- {q_90:.2f} kW)")
    print(f"Target 95% Confidence Interval: Empirical Coverage = {cov_95:.2f}%, Avg Width = {width_95:.2f} kW (Margin = +/- {q_95:.2f} kW)")
    
    # 2. Calibration Reliability Curve
    curve_df = cp.evaluate_calibration_curve(X_test, y_test)
    
    fig_cal, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(13, 5))
    ax_cov.plot([50, 100], [50, 100], 'r--', label="Perfect Calibration (Ideal)")
    ax_cov.plot(curve_df["Expected_Coverage"], curve_df["Empirical_Coverage"], 'o-', color='#1f77b4', lw=2, label="Conformal Gradient Boosting")
    ax_cov.set_xlabel("Expected Nominal Coverage (%)")
    ax_cov.set_ylabel("Empirical Observed Coverage (%)")
    ax_cov.set_title("Calibration Reliability Diagram\n(Uncertainty Calibration)")
    ax_cov.legend()
    ax_cov.grid(True, linestyle="--", alpha=0.5)
    
    ax_width.plot(curve_df["Expected_Coverage"], curve_df["Avg_Interval_Width"], 's-', color='#2ca02c', lw=2)
    ax_width.set_xlabel("Expected Nominal Coverage (%)")
    ax_width.set_ylabel("Average Prediction Interval Width (kW)")
    ax_width.set_title("Prediction Interval Sharpness (Efficiency)")
    ax_width.grid(True, linestyle="--", alpha=0.5)
    
    fig_cal.suptitle("Gradient Boosting Calibration Analysis (Heating Load)", fontsize=13, fontweight='bold')
    fig_cal.tight_layout()
    cal_curve_path = os.path.join(artifacts_dir, "calibration_reliability_curve.png")
    fig_cal.savefig(cal_curve_path, dpi=200)
    plt.close(fig_cal)
    
    # 3. Conformal Prediction Bands on Sorted Test Samples
    sort_idx = np.argsort(y_test.values)
    y_test_sorted = y_test.values[sort_idx]
    preds_sorted = preds[sort_idx]
    lower_90_sorted = lower_90[sort_idx]
    upper_90_sorted = upper_90[sort_idx]
    
    fig_bands, ax_bands = plt.subplots(figsize=(12, 6))
    sample_nums = np.arange(len(y_test))
    ax_bands.fill_between(sample_nums, lower_90_sorted, upper_90_sorted, color='#a1d99b', alpha=0.5, label=f"90% Conformal Calibrated Band (+/- {q_90:.2f} kW)")
    ax_bands.plot(sample_nums, preds_sorted, color='#1c9099', lw=1.8, label="Gradient Boosting Predictions")
    ax_bands.scatter(sample_nums, y_test_sorted, color='#e6550d', s=22, zorder=5, label="Actual Ground Truth")
    ax_bands.set_xlabel("Test Buildings (Sorted by Energy Load)")
    ax_bands.set_ylabel("Heating Load (kW)")
    ax_bands.set_title(f"Gradient Boosting Conformal Prediction Intervals (Coverage: {cov_90:.1f}%)", fontsize=13, fontweight='bold')
    ax_bands.legend(loc="upper left")
    ax_bands.grid(True, linestyle="--", alpha=0.5)
    bands_path = os.path.join(artifacts_dir, "conformal_intervals_gradient_boosting.png")
    fig_bands.tight_layout()
    fig_bands.savefig(bands_path, dpi=200)
    plt.close(fig_bands)
    
    # 4. Feature Importance Plot
    feature_imp = pd.Series(gbr.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    feature_imp.plot(kind='barh', ax=ax_imp, color='#2b83ba', edgecolor='black')
    ax_imp.set_xlabel("Relative Feature Importance (Gini / Impurity Reduction)")
    ax_imp.set_title("Model 5 (Gradient Boosting) Feature Importances", fontsize=12, fontweight='bold')
    ax_imp.grid(True, linestyle="--", alpha=0.5)
    imp_path = os.path.join(artifacts_dir, "feature_importance_gradient_boosting.png")
    fig_imp.tight_layout()
    fig_imp.savefig(imp_path, dpi=200)
    plt.close(fig_imp)
    
    # 5. Isotonic Regression Point Calibration Comparison
    cal_preds = gbr.predict(X_cal)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(cal_preds, y_cal)
    calibrated_point_preds = iso.predict(preds)
    
    raw_r2 = r2_score(y_test, preds)
    raw_rmse = np.sqrt(mean_squared_error(y_test, preds))
    iso_r2 = r2_score(y_test, calibrated_point_preds)
    iso_rmse = np.sqrt(mean_squared_error(y_test, calibrated_point_preds))
    
    print(f"\n--- Point Prediction Calibration Comparison ---")
    print(f"Raw Gradient Boosting:   R2 = {raw_r2:.4f}, RMSE = {raw_rmse:.4f}")
    print(f"Isotonic Calibrated GBR: R2 = {iso_r2:.4f}, RMSE = {iso_rmse:.4f}")
    
    # Log everything to a dedicated MLflow Calibration run
    with mlflow.start_run(run_name="Model_5_Gradient_Boosting_Calibration_DeepDive"):
        mlflow.set_tag("model", "GradientBoostingRegressor")
        mlflow.set_tag("module", "calibration")
        
        mlflow.log_metric("conformal_cov_90", cov_90)
        mlflow.log_metric("conformal_width_90", width_90)
        mlflow.log_metric("conformal_cov_95", cov_95)
        mlflow.log_metric("conformal_width_95", width_95)
        mlflow.log_metric("raw_r2", raw_r2)
        mlflow.log_metric("raw_rmse", raw_rmse)
        mlflow.log_metric("isotonic_r2", iso_r2)
        mlflow.log_metric("isotonic_rmse", iso_rmse)
        
        mlflow.log_artifact(cal_curve_path)
        mlflow.log_artifact(bands_path)
        mlflow.log_artifact(imp_path)
        
        # Save curve dataframe to CSV
        curve_csv = os.path.join(artifacts_dir, "calibration_curve_metrics.csv")
        curve_df.to_csv(curve_csv, index=False)
        mlflow.log_artifact(curve_csv)
        
    print(f"\n[OK] Gradient Boosting Calibration analysis successfully logged to MLflow!")


def print_leaderboard_summary(leaderboard_df):
    """Prints a beautiful markdown table of model performances."""
    print("\n" + "="*80)
    print("                      BUILDWATT FINAL MODEL LEADERBOARD")
    print("="*80)
    
    for target in ["Heating_Load", "Cooling_Load"]:
        print(f"\nTarget: {target.upper()}")
        sub = leaderboard_df[leaderboard_df["Target"] == target].sort_values(by="Test_R2", ascending=False)
        formatted = sub[["Model", "Test_R2", "CV_R2_Mean", "Test_RMSE", "Test_MAE", "Test_MAPE"]].copy()
        formatted["Test_R2"] = formatted["Test_R2"].map(lambda x: f"{x:.4f}")
        formatted["CV_R2_Mean"] = formatted["CV_R2_Mean"].map(lambda x: f"{x:.4f}")
        formatted["Test_RMSE"] = formatted["Test_RMSE"].map(lambda x: f"{x:.4f}")
        formatted["Test_MAE"] = formatted["Test_MAE"].map(lambda x: f"{x:.4f}")
        formatted["Test_MAPE"] = formatted["Test_MAPE"].map(lambda x: f"{x:.2f}%")
        print(formatted.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="BuildWatt Multi-Model Training & Calibration Pipeline")
    parser.add_argument("--data-path", default="D:/BuildWatt/ENB2012_data.xlsx", help="Path to ENB2012 Excel dataset")
    parser.add_argument("--artifacts-dir", default="D:/BuildWatt/artifacts", help="Directory for generated plots & artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print("Initializing BuildWatt Machine Learning Pipeline...")
    df = load_dataset(args.data_path)
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    
    # 1. Run full 8-model benchmark on both targets with MLflow tracking
    leaderboard_df = run_benchmarks(
        df,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state
    )
    
    # 2. Run Model 5 (Gradient Boosting) calibration and deep-dive
    run_gradient_boosting_calibration(
        df,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state
    )
    
    # 3. Print final comparative summary
    print_leaderboard_summary(leaderboard_df)
    
    print("\n[SUCCESS] Pipeline execution complete.")
    print("To launch the MLflow UI dashboard, run:")
    print("    mlflow ui --backend-store-uri sqlite:///D:/BuildWatt/mlflow.db --port 5000\n")


if __name__ == "__main__":
    main()
