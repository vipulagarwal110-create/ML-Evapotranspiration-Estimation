"""
Phase 3: Machine Learning Model Training & Evaluation Engine.
Trains Linear Regression, SVR, and Random Forest across the 3 reduced input configurations
plus the full-input upper-bound benchmark.
Generates evaluation tables, scatter plots, time-series overlays, and feature importance charts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 12

DATA_PATH = os.path.join("data", "evapotranspiration_dataset_with_ET0.csv")
FIGURES_DIR = os.path.join("reports", "figures")
RESULTS_CSV = os.path.join("reports", "model_benchmark_results.csv")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 3: MACHINE LEARNING MODEL TRAINING & COMPARATIVE BENCHMARKING")
    print("=" * 80)
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Loaded dataset: {df.shape[0]} daily records from {df.index[0].date()} to {df.index[-1].date()}.")
    
    # 2. Chronological Train-Test Split (80% Train: 2010-2020, 20% Test: 2021-2023)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training observations: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"Testing observations:  {len(test_df)} days ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    # 3. Define Feature Configurations
    feature_configs = {
        'Config 1 (Temp Only)': ['max_temp', 'min_temp'],
        'Config 2 (Temp + Humidity)': ['max_temp', 'min_temp', 'relative_humidity'],
        'Config 3 (Temp + Humidity + Wind)': ['max_temp', 'min_temp', 'relative_humidity', 'wind_speed'],
        'Config 4 (Full Inputs - Upper Bound)': ['max_temp', 'min_temp', 'relative_humidity', 'wind_speed', 'solar_radiation']
    }
    
    # 4. Define Models
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regression (SVR)': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=10.0, epsilon=0.1, kernel='rbf'))
        ]),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    }
    
    y_train = train_df['ET0']
    y_test = test_df['ET0']
    
    results = []
    predictions = {}
    
    # 5. Model Training & Evaluation Loop
    for config_name, features in feature_configs.items():
        print(f"\nEvaluating: {config_name} (Features: {features})")
        X_train = train_df[features]
        X_test = test_df[features]
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            key = f"{model_name} | {config_name}"
            predictions[key] = y_pred
            
            results.append({
                'Configuration': config_name,
                'Model': model_name,
                'Input Features': ', '.join(features),
                'MAE (mm/day)': round(mae, 3),
                'RMSE (mm/day)': round(rmse, 3),
                'R2 Score': round(r2, 3)
            })
            print(f"  -> {model_name:32s} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[OUTPUT] Benchmark results saved to: {RESULTS_CSV}")
    
    # 6. Generate Figure 1: Model Comparison Metrics Bar Chart
    plt.figure(figsize=(14, 5.5))
    
    # Plot R2 comparison
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='Configuration', y='R2 Score', hue='Model', palette='Blues_d')
    plt.title('Model Comparison: R² Score across Feature Configurations', fontweight='bold', fontsize=11)
    plt.xlabel('Input Configuration', fontweight='bold')
    plt.ylabel('R² Score (Higher is better)', fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Model', loc='lower right')
    
    # Plot RMSE comparison
    plt.subplot(1, 2, 2)
    sns.barplot(data=results_df, x='Configuration', y='RMSE (mm/day)', hue='Model', palette='Reds_d')
    plt.title('Model Comparison: RMSE (mm/day) across Feature Configurations', fontweight='bold', fontsize=11)
    plt.xlabel('Input Configuration', fontweight='bold')
    plt.ylabel('RMSE in mm/day (Lower is better)', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Model', loc='upper right')
    
    plt.tight_layout()
    comp_fig_path = os.path.join(FIGURES_DIR, "model_comparison_metrics.png")
    plt.savefig(comp_fig_path, dpi=200)
    plt.close()
    print(f"[OUTPUT] Saved comparison chart: {comp_fig_path}")
    
    # 7. Generate Figure 2: Predicted vs Actual Scatter Plots (1:1 line) for Random Forest
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, sharex=True)
    configs_to_plot = [
        'Config 1 (Temp Only)',
        'Config 2 (Temp + Humidity)',
        'Config 3 (Temp + Humidity + Wind)'
    ]
    colors = ['#d95f02', '#7570b3', '#1b9e77']
    
    for idx, (cfg, color) in enumerate(zip(configs_to_plot, colors)):
        ax = axes[idx]
        y_pred = predictions[f"Random Forest | {cfg}"]
        
        ax.scatter(y_test, y_pred, alpha=0.35, s=15, color=color, label='Test Observations')
        
        # 1:1 perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.8, label='1:1 Perfect Fit')
        
        # Find R2 and RMSE
        row = results_df[(results_df['Model'] == 'Random Forest') & (results_df['Configuration'] == cfg)].iloc[0]
        ax.set_title(f"{cfg}\nR² = {row['R2 Score']:.3f} | RMSE = {row['RMSE (mm/day)']:.3f} mm/d", fontsize=10, fontweight='bold')
        ax.set_xlabel('Actual FAO-56 ET0 (mm/day)', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Predicted ET0 (mm/day)', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=9)
        
    plt.suptitle('Random Forest: Predicted vs. Actual ET0 across Reduced Input Sets', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    scatter_fig_path = os.path.join(FIGURES_DIR, "predicted_vs_actual_scatter.png")
    plt.savefig(scatter_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OUTPUT] Saved scatter plot: {scatter_fig_path}")
    
    # 8. Generate Figure 3: Time-Series Prediction Overlay for Test Year (2023)
    plt.figure(figsize=(15, 5))
    test_year_mask = (test_df.index >= '2023-01-01') & (test_df.index <= '2023-12-31')
    y_test_year = y_test[test_year_mask]
    
    plt.plot(y_test_year.index, y_test_year.values, color='black', lw=1.8, label='Actual FAO-56 ET0 (Ground Truth)', alpha=0.9)
    
    # Plot RF Config 1 and RF Config 3
    pred_c1 = predictions['Random Forest | Config 1 (Temp Only)'][test_year_mask]
    pred_c3 = predictions['Random Forest | Config 3 (Temp + Humidity + Wind)'][test_year_mask]
    
    plt.plot(y_test_year.index, pred_c1, color='#d95f02', lw=1.2, linestyle=':', label='RF Config 1 (Temp Only)', alpha=0.8)
    plt.plot(y_test_year.index, pred_c3, color='#1b9e77', lw=1.4, linestyle='-', label='RF Config 3 (Temp + RH + Wind)', alpha=0.85)
    
    plt.title('Daily ET0 Time-Series: Actual vs. Predicted over Unseen Year 2023 (Hisar, Haryana)', fontsize=12, fontweight='bold')
    plt.xlabel('Date (Year 2023)', fontweight='bold')
    plt.ylabel('Reference Evapotranspiration ET0 (mm/day)', fontweight='bold')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    ts_fig_path = os.path.join(FIGURES_DIR, "timeseries_prediction_overlay.png")
    plt.savefig(ts_fig_path, dpi=200)
    plt.close()
    print(f"[OUTPUT] Saved time-series overlay: {ts_fig_path}")
    
    # 9. Generate Figure 4: Random Forest Feature Importance
    rf_best = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_best.fit(train_df[feature_configs['Config 3 (Temp + Humidity + Wind)']], y_train)
    
    importances = rf_best.feature_importances_
    feat_names = ['Max Temperature', 'Min Temperature', 'Relative Humidity', 'Wind Speed']
    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(8, 4))
    bars = plt.barh(feat_df['Feature'], feat_df['Importance'], color='#1b9e77', edgecolor='black', alpha=0.85)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va='center', fontweight='bold', fontsize=9)
        
    plt.title('Random Forest Feature Importance (Configuration 3: Temp + RH + Wind)', fontsize=11, fontweight='bold')
    plt.xlabel('Relative Importance (Gini Impurity Reduction)', fontweight='bold')
    plt.xlim(0, max(importances) * 1.18)
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    plt.tight_layout()
    feat_fig_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(feat_fig_path, dpi=200)
    plt.close()
    print(f"[OUTPUT] Saved feature importance chart: {feat_fig_path}")
    
    print("\n" + "=" * 80)
    print("ALL PHASE 3 EXPERIMENTS AND FIGURES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
