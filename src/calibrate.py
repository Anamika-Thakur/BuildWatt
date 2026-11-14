"""
Stage 3: Champion Model Calibration & Uncertainty Quantification
Performs Split Conformal Prediction calibration on Model 5 (Gradient Boosting).
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURE_NAMES = [
    'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
    'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
]


def calibrate_and_quantify(cal_path="data/cal.csv", test_path="data/test.csv",
                           models_dir="models", artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    cal_df = pd.read_csv(cal_path)
    test_df = pd.read_csv(test_path)
    
    model_heat_path = os.path.join(models_dir, "champion_gbr_heat.joblib")
    if not os.path.exists(model_heat_path):
        raise FileNotFoundError(f"Champion model not found at {model_heat_path}. Run train stage first.")
        
    gbr = joblib.load(model_heat_path)
    
    X_cal = cal_df[FEATURE_NAMES]
    y_cal = cal_df["Heating_Load"]
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["Heating_Load"]
    
    # 1. Nonconformity residuals on calibration set
    cal_preds = gbr.predict(X_cal)
    cal_residuals = np.abs(np.array(y_cal) - cal_preds)
    n_cal = len(cal_residuals)
    
    test_preds = gbr.predict(X_test)
    
    def get_conformal_bounds(alpha):
        p_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        p_level = min(max(p_level, 0.0), 1.0)
        q = np.quantile(cal_residuals, p_level)
        lower = test_preds - q
        upper = test_preds + q
        cov = np.mean((y_test >= lower) & (y_test <= upper)) * 100
        width = np.mean(upper - lower)
        return lower, upper, q, cov, width
        
    l90, u90, q90, cov90, width90 = get_conformal_bounds(0.10)
    l95, u95, q95, cov95, width95 = get_conformal_bounds(0.05)
    
    print(f"--- Conformal Uncertainty Calibration (Heating Load) ---")
    print(f"Target 90%: Observed Coverage = {cov90:.2f}%, Margin = +/- {q90:.2f} kW (Width: {width90:.2f} kW)")
    print(f"Target 95%: Observed Coverage = {cov95:.2f}%, Margin = +/- {q95:.2f} kW (Width: {width95:.2f} kW)")
    
    # 2. Calibration curve across alpha levels
    alphas = np.linspace(0.01, 0.50, 25)
    curve_records = []
    for a in alphas:
        _, _, q, cov, w = get_conformal_bounds(a)
        curve_records.append({
            "Alpha": float(a),
            "Expected_Coverage": float((1 - a) * 100),
            "Empirical_Coverage": float(cov),
            "Interval_Width": float(w),
            "Margin": float(q)
        })
    curve_df = pd.DataFrame(curve_records)
    curve_csv = os.path.join(artifacts_dir, "calibration_curve_metrics.csv")
    curve_df.to_csv(curve_csv, index=False)
    
    # Plot Reliability Curve
    fig_cal, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot([50, 100], [50, 100], 'r--', lw=2, label="Perfect Calibration")
    ax1.plot(curve_df["Expected_Coverage"], curve_df["Empirical_Coverage"], 'o-', color='#1f77b4', lw=2, label="Conformal GBR")
    ax1.set_xlabel("Expected Nominal Coverage (%)")
    ax1.set_ylabel("Empirical Observed Coverage (%)")
    ax1.set_title("Calibration Reliability Diagram")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.plot(curve_df["Expected_Coverage"], curve_df["Interval_Width"], 's-', color='#2ca02c', lw=2)
    ax2.set_xlabel("Expected Nominal Coverage (%)")
    ax2.set_ylabel("Prediction Interval Width (kW)")
    ax2.set_title("Prediction Interval Sharpness")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Gradient Boosting Calibration Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    rel_path = os.path.join(artifacts_dir, "calibration_reliability_curve.png")
    plt.savefig(rel_path, dpi=200)
    plt.close(fig_cal)
    
    # 3. Plot Conformal Prediction Bands on Sorted Test Set
    sort_idx = np.argsort(y_test.values)
    y_sorted = y_test.values[sort_idx]
    preds_sorted = test_preds[sort_idx]
    l90_sorted = l90[sort_idx]
    u90_sorted = u90[sort_idx]
    
    fig_bands, ax_b = plt.subplots(figsize=(12, 5))
    x_nums = np.arange(len(y_test))
    ax_b.fill_between(x_nums, l90_sorted, u90_sorted, color='#a1d99b', alpha=0.5, label=f"90% Conformal Band (+/- {q90:.2f} kW)")
    ax_b.plot(x_nums, preds_sorted, color='#1c9099', lw=2, label="Gradient Boosting Predictions")
    ax_b.scatter(x_nums, y_sorted, color='#e6550d', s=22, zorder=5, label="Ground Truth Energy Load")
    ax_b.set_xlabel("Test Buildings (Sorted by Energy Load)")
    ax_b.set_ylabel("Heating Load (kW)")
    ax_b.set_title(f"Gradient Boosting Conformal Prediction Intervals (Coverage: {cov90:.1f}%)", fontsize=12, fontweight='bold')
    ax_b.legend(loc="upper left")
    ax_b.grid(True, linestyle="--", alpha=0.5)
    
    bands_path = os.path.join(artifacts_dir, "conformal_intervals_gradient_boosting.png")
    plt.tight_layout()
    plt.savefig(bands_path, dpi=200)
    plt.close(fig_bands)
    
    # 4. Feature Importance Plot
    feat_imp = pd.Series(gbr.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
    fig_imp, ax_i = plt.subplots(figsize=(8, 5))
    feat_imp.plot(kind='barh', ax=ax_i, color='#2b83ba', edgecolor='black')
    ax_i.set_xlabel("Relative Feature Importance")
    ax_i.set_title("Model 5 (Gradient Boosting) Feature Importances", fontsize=12, fontweight='bold')
    ax_i.grid(True, linestyle="--", alpha=0.5)
    
    imp_path = os.path.join(artifacts_dir, "feature_importance_gradient_boosting.png")
    plt.tight_layout()
    plt.savefig(imp_path, dpi=200)
    plt.close(fig_imp)
    
    # Update metrics.json
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    cur_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            cur_metrics = json.load(f)
            
    cur_metrics["conformal_coverage_90"] = float(cov90)
    cur_metrics["conformal_margin_90"] = float(q90)
    cur_metrics["conformal_width_90"] = float(width90)
    cur_metrics["conformal_coverage_95"] = float(cov95)
    cur_metrics["conformal_margin_95"] = float(q95)
    cur_metrics["conformal_width_95"] = float(width95)
    
    with open(metrics_path, "w") as f:
        json.dump(cur_metrics, f, indent=2)
        
    print(f"[OK] Calibration artifacts & metrics saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-path", default="data/cal.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    
    calibrate_and_quantify(
        cal_path=args.cal_path,
        test_path=args.test_path,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir
    )

