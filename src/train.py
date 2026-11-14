"""
Stage 2: Multi-Model Benchmark & MLflow Tracking
Trains 8 candidate models, logs metrics and artifacts to MLflow (DagsHub/Local),
and outputs benchmark metrics and comparison plots.
"""
import os
import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings('ignore')
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

FEATURE_NAMES = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
]


def configure_mlflow(repo_owner="Anamika-Thakur", repo_name="BuildWatt"):
    """
    Configures MLflow tracking URI.
    Checks for DagsHub remote or falls back to local SQLite store.
    """
    token = os.environ.get("DAGSHUB_USER_TOKEN") or os.environ.get("DAGSHUB_TOKEN")
    dagshub_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    
    # Check for DagsHub token in cache or environment
    try:
        import dagshub.auth.tokens as t
        cache = getattr(t._get_token_storage(), '_token_cache', {})
        if cache and "dagshub.com" in cache:
            token = cache["dagshub.com"].token
    except Exception:
        pass

    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(dagshub_uri)
        print(f"[MLflow] Logging to DagsHub remote: {dagshub_uri}")
    else:
        # Fallback to local SQLite backend
        local_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(local_uri)
        print(f"[MLflow] Logging to local SQLite backend: {local_uri}")
        
    mlflow.set_experiment("BuildWatt_Energy_Prediction")


def get_model_registry(random_state=42):
    return {
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
            max_depth=7, min_samples_split=5, min_samples_leaf=2, random_state=random_state
        ),
        "4. Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_split=3, random_state=random_state, n_jobs=-1
        ),
        "5. Gradient Boosting (Champion)": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=random_state
        ),
        "6. XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=4, subsample=0.85,
            colsample_bytree=0.85, random_state=random_state, verbosity=0
        ),
        "7. SVR (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=100.0, epsilon=0.1, gamma='scale'))
        ]),
        "8. MLP Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu', max_iter=1500,
                early_stopping=True, validation_fraction=0.1, random_state=random_state
            ))
        ])
    }


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    }


def train_benchmark(train_path="data/train.csv", test_path="data/test.csv",
                    artifacts_dir="artifacts", models_dir="models", random_state=42):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    configure_mlflow()
    
    targets = {
        "Heating_Load": ("Heating_Load", "heat"),
        "Cooling_Load": ("Cooling_Load", "cool")
    }
    
    leaderboard_records = []
    summary_metrics = {}
    
    for target_col, (target_name, short_tag) in targets.items():
        print(f"\n--- Training Models for: {target_name} ---")
        X_train = train_df[FEATURE_NAMES]
        y_train = train_df[target_col]
        X_test = test_df[FEATURE_NAMES]
        y_test = test_df[target_col]
        
        models = get_model_registry(random_state)
        
        for name, model in models.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            metrics = evaluate(y_test, test_preds)
            
            record = {
                "Target": target_name,
                "Model": name,
                "Test_R2": metrics["r2"],
                "CV_R2_Mean": cv_mean,
                "CV_R2_Std": cv_std,
                "Test_RMSE": metrics["rmse"],
                "Test_MAE": metrics["mae"],
                "Test_MAPE": metrics["mape"]
            }
            leaderboard_records.append(record)
            print(f"  {name:<30} -> R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | CV: {cv_mean:.4f}")
            
            # Save champion model
            if "Gradient Boosting" in name:
                model_file = os.path.join(models_dir, f"champion_gbr_{short_tag}.joblib")
                joblib.dump(model, model_file)
                summary_metrics[f"{short_tag}_champion_r2"] = metrics["r2"]
                summary_metrics[f"{short_tag}_champion_rmse"] = metrics["rmse"]
                summary_metrics[f"{short_tag}_champion_mae"] = metrics["mae"]
            
            # Log to MLflow
            safe_run_name = f"{target_name}_{name.split('.')[1].strip().replace(' ', '_')}"
            try:
                with mlflow.start_run(run_name=safe_run_name):
                    mlflow.set_tag("target", target_name)
                    mlflow.set_tag("model_family", name)
                    mlflow.log_metrics({
                        "test_r2": metrics["r2"],
                        "test_rmse": metrics["rmse"],
                        "test_mae": metrics["mae"],
                        "test_mape": metrics["mape"],
                        "cv_r2_mean": cv_mean,
                        "cv_r2_std": cv_std
                    })
                    if "XGBoost" in name:
                        mlflow.xgboost.log_model(xgb_model=model, name="model")
                    else:
                        mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="cloudpickle")
            except Exception as e:
                print(f"  [MLflow Warning] Could not log run {safe_run_name}: {e}")

        # Target comparison plot
        sub_df = pd.DataFrame([r for r in leaderboard_records if r["Target"] == target_name]).sort_values(by="Test_R2")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#2ca02c' if 'Gradient Boosting' in m else '#1f77b4' for m in sub_df["Model"]]
        ax1.barh(sub_df["Model"], sub_df["Test_R2"], color=colors)
        ax1.set_xlabel("Test $R^2$")
        ax1.set_title(f"Model Comparison: Test $R^2$ ({target_name})")
        ax1.set_xlim(0.85, 1.002)
        
        ax2.barh(sub_df["Model"], sub_df["Test_RMSE"], color=colors)
        ax2.set_xlabel("Test RMSE (kW)")
        ax2.set_title(f"Model Comparison: Test RMSE ({target_name})")
        
        plt.suptitle(f"BuildWatt Benchmark - {target_name}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(artifacts_dir, f"model_comparison_{target_name}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)

    # Save leaderboard CSV
    leaderboard_df = pd.DataFrame(leaderboard_records)
    csv_path = os.path.join(artifacts_dir, "benchmark_leaderboard.csv")
    leaderboard_df.to_csv(csv_path, index=False)
    
    # Save DVC metrics JSON
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
        
    print(f"\n[OK] Leaderboard saved to: {csv_path}")
    print(f"[OK] Summary metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    train_benchmark(
        train_path=args.train_path,
        test_path=args.test_path,
        artifacts_dir=args.artifacts_dir,
        models_dir=args.models_dir,
        random_state=args.random_state
    )
