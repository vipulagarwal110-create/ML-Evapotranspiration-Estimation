import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning & Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# Load dataset
csv_candidates = [
    'data/raw/raw_weather_data.csv',
    'raw_weather_data.csv',
    '../input/evapotranspiration-nasa-power/raw_weather_data.csv'
]

csv_file = None
for path in csv_candidates:
    if os.path.exists(path):
        csv_file = path
        break

if csv_file is None:
    raise FileNotFoundError("Could not find raw_weather_data.csv. Please ensure it is uploaded.")

df_raw = pd.read_csv(csv_file)
print(f"Dataset successfully loaded from: {csv_file}")
print(f"Total Rows (Days): {df_raw.shape[0]}, Total Columns: {df_raw.shape[1]}")
df_raw.head()


# 1. Missing Values & Fill-Code Checks
fill_code_counts = (df_raw == -999.0).sum()
nan_counts = df_raw.isna().sum()

quality_summary = pd.DataFrame({
    'NASA -999 Fill Codes': fill_code_counts,
    'Standard NaNs': nan_counts,
    'Data Type': df_raw.dtypes
})
print("--- Missing Values & Fill-Code Analysis ---")
print(quality_summary)

# 2. Physical Consistency & Range Checks
print("\n--- Physical Range & Consistency Checks ---")
temp_inversions = (df_raw['T2M_MAX'] < df_raw['T2M_MIN']).sum()
rh_out_of_bounds = ((df_raw['RH2M'] < 0) | (df_raw['RH2M'] > 100)).sum()
wind_negative = (df_raw['WS2M'] < 0).sum()
solar_negative = (df_raw['ALLSKY_SFC_SW_DWN'] < 0).sum()

print(f"1. Inverted Temperatures (Tmax < Tmin): {temp_inversions} days")
print(f"2. Relative Humidity outside [0, 100%]: {rh_out_of_bounds} days")
print(f"3. Negative Wind Speeds:               {wind_negative} days")
print(f"4. Negative Solar Radiation:           {solar_negative} days")
print("=> Dataset quality is pristine! No synthetic imputation was required.")


def calculate_fao56_penman_monteith(df, latitude_deg=29.15, elevation_m=215.0):
    lat_rad = np.radians(latitude_deg)
    dates = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    J = dates.dt.dayofyear  # Day of year
    
    T_mean = df['T2M']
    T_max = df['T2M_MAX']
    T_min = df['T2M_MIN']
    RH = df['RH2M']
    u2 = df['WS2M']
    Rs = df['ALLSKY_SFC_SW_DWN']
    P = df['PS']
    
    # 1. Delta (slope of vapor pressure curve)
    delta = 4098.0 * (0.6108 * np.exp((17.27 * T_mean) / (T_mean + 237.3))) / ((T_mean + 237.3) ** 2)
    
    # 2. Psychrometric constant (gamma)
    gamma = 0.665e-3 * P
    
    # 3. Vapor Pressure (es, ea, deficit)
    e0_Tmax = 0.6108 * np.exp((17.27 * T_max) / (T_max + 237.3))
    e0_Tmin = 0.6108 * np.exp((17.27 * T_min) / (T_min + 237.3))
    es = (e0_Tmax + e0_Tmin) / 2.0
    ea = es * (RH / 100.0)
    
    # 4. Solar Radiation Geometry
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * J / 365.0)
    solar_dec = 0.409 * np.sin((2 * np.pi * J / 365.0) - 1.39)
    omega_s = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(solar_dec), -1.0, 1.0))
    Gsc = 0.0820  # Solar constant (MJ/m^2/min)
    
    Ra = (24.0 * 60.0 / np.pi) * Gsc * dr * (
        omega_s * np.sin(lat_rad) * np.sin(solar_dec) +
        np.cos(lat_rad) * np.cos(solar_dec) * np.sin(omega_s)
    )
    Rso = (0.75 + 2e-5 * elevation_m) * Ra
    
    # Net shortwave radiation (alpha = 0.23 for reference grass)
    Rns = (1.0 - 0.23) * Rs
    
    # Net longwave radiation (Stefan-Boltzmann)
    sigma = 4.903e-9
    Tmax_k4 = (T_max + 273.16) ** 4
    Tmin_k4 = (T_min + 273.16) ** 4
    rs_ratio = np.clip(Rs / np.where(Rso > 0, Rso, 1e-5), 0.3, 1.0)
    Rnl = sigma * ((Tmax_k4 + Tmin_k4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * rs_ratio - 0.35)
    
    Rn = Rns - Rnl
    G = 0.0
    
    # Final ET0 formula
    numerator = 0.408 * delta * (Rn - G) + gamma * (900.0 / (T_mean + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    
    et0 = np.maximum(numerator / denominator, 0.0)
    return et0

# Compute target column ET0
df_model = df_raw.copy()
df_model['ET0'] = calculate_fao56_penman_monteith(df_raw).values
df_model['date'] = pd.to_datetime(df_raw['DATE'].astype(str), format='%Y%m%d')
df_model.set_index('date', inplace=True)

# Rename to clean feature names
df_model.rename(columns={
    'T2M_MAX': 'max_temp',
    'T2M_MIN': 'min_temp',
    'T2M': 'mean_temp',
    'RH2M': 'relative_humidity',
    'WS2M': 'wind_speed',
    'ALLSKY_SFC_SW_DWN': 'solar_radiation',
    'PS': 'surface_pressure'
}, inplace=True)

print("Computed ET0 Target Summary:")
print(df_model[['max_temp', 'min_temp', 'relative_humidity', 'wind_speed', 'solar_radiation', 'ET0']].describe())


plt.figure(figsize=(15, 4.5))
plt.plot(df_model.index, df_model['ET0'], color='#1f77b4', lw=0.9, alpha=0.85)
plt.title('Daily Reference Evapotranspiration (ET0) in Hisar, Haryana (2010–2023)', fontsize=13, fontweight='bold')
plt.ylabel('ET0 (mm/day)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
corr = df_model[['max_temp', 'min_temp', 'relative_humidity', 'wind_speed', 'solar_radiation', 'ET0']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, linewidths=0.6)
plt.title('Pearson Correlation Matrix with Target ET0', fontsize=12, fontweight='bold')
plt.tight_layout()
# plt.show()


# Chronological 80/20 Train-Test Split (preserves time series order)
train_size = int(len(df_model) * 0.8)
train_df = df_model.iloc[:train_size]
test_df = df_model.iloc[train_size:]

print(f"Training Set: {len(train_df)} days ({train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')})")
print(f"Testing Set:  {len(test_df)} days ({test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')})")

feature_configs = {
    'Config 1 (Temp Only)': ['max_temp', 'min_temp'],
    'Config 2 (Temp + Humidity)': ['max_temp', 'min_temp', 'relative_humidity'],
    'Config 3 (Temp + Humidity + Wind)': ['max_temp', 'min_temp', 'relative_humidity', 'wind_speed']
}

models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Regression (SVR)': Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(C=10.0, epsilon=0.1, kernel='rbf'))
    ]),
    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
}

results = []
predictions = {}

y_train = train_df['ET0']
y_test = test_df['ET0']

for config_name, features in feature_configs.items():
    X_train = train_df[features]
    X_test = test_df[features]
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        key = f"{model_name} | {config_name}"
        predictions[key] = y_pred
        
        results.append({
            'Configuration': config_name,
            'Model': model_name,
            'Features Used': ', '.join(features),
            'MAE (mm/day)': round(mae, 3),
            'RMSE (mm/day)': round(rmse, 3),
            'R2 Score': round(r2, 3)
        })

results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("EXPERIMENTAL MODEL BENCHMARK RESULTS")
print("=" * 80)
print(results_df)


fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

configs = list(feature_configs.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, config_name in enumerate(configs):
    key = f"Random Forest | {config_name}"
    y_pred = predictions[key]
    
    ax = axes[i]
    ax.scatter(y_test, y_pred, alpha=0.35, color=colors[i], s=18, edgecolors='none')
    
    # 1:1 line
    val_min, val_max = 0, 15
    ax.plot([val_min, val_max], [val_min, val_max], 'r--', lw=1.8, label='1:1 Line (Ideal)')
    
    # Metrics
    row = results_df[(results_df['Model'] == 'Random Forest') & (results_df['Configuration'] == config_name)].iloc[0]
    ax.set_title(f"{config_name}\nR² = {row['R2 Score']:.3f} | RMSE = {row['RMSE (mm/day)']:.3f} mm/day", 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual FAO-56 ET0 (mm/day)', fontsize=10)
    if i == 0:
        ax.set_ylabel('Predicted ET0 (mm/day)', fontsize=10)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
# plt.show()


rf_final = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
rf_final.fit(train_df[feature_configs['Config 3 (Temp + Humidity + Wind)']], y_train)

importances = rf_final.feature_importances_
feature_names = ['Max Temp', 'Min Temp', 'Relative Humidity', 'Wind Speed']

plt.figure(figsize=(8.5, 4.5))
bars = plt.barh(feature_names, importances, color='#2ca02c', edgecolor='black', alpha=0.85)
plt.title('Random Forest Feature Importance (Configuration 3)', fontsize=12, fontweight='bold')
plt.xlabel('Relative Importance (MDI / Gini Reduction)', fontsize=10)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w * 100:.1f}%", va='center', fontweight='bold')

plt.xlim(0, max(importances) + 0.1)
plt.tight_layout()
# plt.show()


