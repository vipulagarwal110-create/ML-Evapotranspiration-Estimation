"""
Script to generate the Kaggle notebook: evapotranspiration_fao56_calculation.ipynb
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True)
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(keepends=True)
    })

# Cell 1: Markdown Title & Intro
add_md(r"""# 🌾 Estimation of Reference Evapotranspiration ($ET_0$) via FAO-56 Penman–Monteith

### **Project 1: Machine Learning-Based Evapotranspiration Estimation for Irrigation Support**
* **Target Location:** Hisar, Haryana, Northern India ($29.15^\circ\text{ N}, 75.72^\circ\text{ E}$, Elevation: $215\,\text{m}$)
* **Meteorological Source:** NASA POWER Daily Agroclimatology (2010–2023, 14 years)
* **Base Paper Reference:** Shaloo et al., *"Modeling daily reference evapotranspiration and evaluating uncertainty analysis in machine learning under limited meteorological data conditions for Northern India."*

---

## 📋 Table of Contents
1. **Setup & Imports:** Import standard scientific Python libraries.
2. **Dataset Ingestion:** Robust loader for NASA POWER CSV files (handles local, Kaggle, and header metadata).
3. **Exploratory Data Analysis (EDA):** Summary statistics, distributions, and data hygiene.
4. **FAO-56 Penman–Monteith Formulation:** Physics and mathematical background.
5. **$ET_0$ Calculation:** Self-contained, vectorized computation of daily reference evapotranspiration.
6. **Seasonal Insights & Visualizations:** Multi-year time series, monthly distributions, and correlation heatmap.
7. **Exporting Modeling Dataset:** Save the completed dataset with the $ET_0$ ground-truth target ready for ML models.
""")

# Cell 2: Imports
add_md(r"""## 1. Setup & Environment Configuration""")
add_code(r"""import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("Libraries imported successfully.")
""")

# Cell 3: Data Loading
add_md(r"""## 2. Dataset Loading

This loader handles:
1. Files placed in Kaggle's `/kaggle/input/` directory
2. Files placed in a local working directory (`data/raw/` or current folder)
3. The NASA POWER metadata comment block (`-BEGIN HEADER-` ... `-END HEADER-`)
""")

add_code(r'''def load_weather_data(file_path=None):
    candidates = []
    if file_path and os.path.exists(file_path):
        candidates.append(file_path)
        
    # Check Kaggle input directory
    candidates.extend(glob.glob('/kaggle/input/**/POWER*.csv', recursive=True))
    candidates.extend(glob.glob('/kaggle/input/**/*.csv', recursive=True))
    
    # Check local workspace paths
    candidates.extend([
        'data/raw/raw_weather_data.csv',
        'raw_weather_data.csv'
    ])
    candidates.extend(glob.glob(os.path.expanduser('~/Downloads/POWER*.csv')))
    
    selected_file = None
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            selected_file = p
            break
            
    if not selected_file:
        raise FileNotFoundError(
            "Weather CSV file not found. Please upload or specify your NASA POWER CSV dataset."
        )
        
    print(f"Loading data from: {selected_file}")
    
    with open(selected_file, 'r', encoding='utf-8', errors='ignore') as f:
        sample_lines = [f.readline() for _ in range(35)]
        
    skip_rows = 0
    for idx, line in enumerate(sample_lines):
        if '-END HEADER-' in line:
            skip_rows = idx + 1
            break
            
    df = pd.read_csv(selected_file, skiprows=skip_rows)
    print(f"Loaded {df.shape[0]} daily records with columns: {list(df.columns)}")
    return df

df_raw = load_weather_data()
df_raw.head()
''')

# Cell 4: Cleaning & Preprocessing
add_md(r"""## 3. Data Cleaning & Feature Standardization

We standardize variables to standard meteorological names:
* `max_temp`: Daily maximum air temperature ($^\circ\text{C}$)
* `min_temp`: Daily minimum air temperature ($^\circ\text{C}$)
* `mean_temp`: Daily average air temperature ($^\circ\text{C}$)
* `relative_humidity`: Relative humidity ($\%$)
* `wind_speed`: Wind speed at 2m height ($\text{m/s}$)
* `solar_radiation`: All-sky downward solar irradiance ($\text{MJ/m}^2/\text{day}$)
* `surface_pressure`: Surface atmospheric pressure ($\text{kPa}$)
""")

add_code(r'''def preprocess_weather_data(df):
    df_clean = df.copy()
    
    # Construct Datetime index
    if 'YEAR' in df_clean.columns and 'DOY' in df_clean.columns:
        dates = pd.to_datetime(df_clean['YEAR'].astype(str) + df_clean['DOY'].astype(str), format='%Y%j')
        df_clean.index = dates
    elif 'DATE' in df_clean.columns:
        df_clean.index = pd.to_datetime(df_clean['DATE'].astype(str), format='%Y%m%d')
    else:
        df_clean.index = pd.date_range(start='2010-01-01', periods=len(df_clean), freq='D')
        
    df_clean.index.name = 'Date'
    
    # Rename columns
    mapping = {
        'T2M_MAX': 'max_temp',
        'T2M_MIN': 'min_temp',
        'T2M': 'mean_temp',
        'RH2M': 'relative_humidity',
        'WS2M': 'wind_speed',
        'ALLSKY_SFC_SW_DWN': 'solar_radiation',
        'PS': 'surface_pressure'
    }
    df_clean.rename(columns=mapping, inplace=True)
    
    if 'mean_temp' not in df_clean.columns and 'max_temp' in df_clean.columns and 'min_temp' in df_clean.columns:
        df_clean['mean_temp'] = (df_clean['max_temp'] + df_clean['min_temp']) / 2.0
        
    # Handle NASA fill value (-999.0)
    df_clean.replace(-999.0, np.nan, inplace=True)
    df_clean.interpolate(method='time', inplace=True)
    df_clean.bfill(inplace=True)
    df_clean.ffill(inplace=True)
    
    cols = ['max_temp', 'min_temp', 'mean_temp', 'relative_humidity', 'wind_speed', 'solar_radiation', 'surface_pressure']
    keep_cols = [c for c in cols if c in df_clean.columns]
    
    return df_clean[keep_cols]

df_weather = preprocess_weather_data(df_raw)
print("Cleaned Meteorological Dataset Preview:")
display(df_weather.head())
print("\nSummary Statistics:")
display(df_weather.describe().round(2))
''')

# Cell 5: FAO-56 Methodology
add_md(r"""## 4. The FAO-56 Penman–Monteith Methodology

Reference evapotranspiration ($ET_0$, in $\text{mm/day}$) represents the rate of evapotranspiration from an extensive surface of $0.12\,\text{m}$ tall green grass cover with an albedo of $0.23$, actively growing, well-watered, and shading the ground completely.

According to **FAO Irrigation and Drainage Paper No. 56** (*Allen et al., 1998*):

$$ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T + 273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$$

### Physical Variables:
1. **$T$ (Mean Air Temperature):** $T = \frac{T_{max} + T_{min}}{2}$ ($^\circ\text{C}$)
2. **$\Delta$ (Slope of Saturation Vapor Pressure Curve):**
   $$\Delta = \frac{4098 \left[ 0.6108 \exp\left(\frac{17.27 T}{T + 237.3}\right) \right]}{(T + 237.3)^2}$$
3. **$\gamma$ (Psychrometric Constant):**
   $$\gamma = 0.665 \times 10^{-3} P \quad (\text{kPa}/^\circ\text{C})$$
4. **$(e_s - e_a)$ (Vapor Pressure Deficit, VPD):**
   $$e_s = \frac{e^0(T_{max}) + e^0(T_{min})}{2}, \quad e_a = e_s \times \frac{RH}{100}$$
5. **$R_n$ (Net Radiation):** $R_n = R_{ns} - R_{nl}$
   * Shortwave: $R_{ns} = (1 - 0.23) R_s$
   * Longwave: $R_{nl}$ calculated via Stefan-Boltzmann law adjusted for cloudiness ($R_s / R_{so}$) and humidity.
6. **$G$ (Soil Heat Flux):** For daily time steps, $G \approx 0$.
""")

# Cell 6: Calculation Code
add_md(r"""## 5. Vectorized Calculation of Reference Evapotranspiration ($ET_0$)

This pure NumPy/Pandas calculation runs self-contained without needing any external hydrology package installations.
""")

add_code(r'''def calculate_fao56_penman_monteith(df, latitude_deg=29.15, elevation_m=215.0):
    lat_rad = np.deg2rad(latitude_deg)
    doy = df.index.dayofyear
    
    tmax = df['max_temp']
    tmin = df['min_temp']
    tmean = df['mean_temp']
    rh = df['relative_humidity']
    wind = df['wind_speed']
    rs = df['solar_radiation']
    
    # 1. Atmospheric Pressure & Psychrometric constant (gamma)
    if 'surface_pressure' in df.columns:
        pressure = df['surface_pressure']
    else:
        pressure = 101.3 * (((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26)
    gamma = 0.000665 * pressure
    
    # 2. Slope of Saturation Vapor Pressure Curve (delta)
    delta = 4098.0 * (0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    
    # 3. Vapor Pressure Deficit (es - ea)
    e0_tmax = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))
    e0_tmin = 0.6108 * np.exp((17.27 * tmin) / (tmin + 237.3))
    es = (e0_tmax + e0_tmin) / 2.0
    ea = es * (rh / 100.0)
    
    # 4. Extraterrestrial Radiation (Ra)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    solar_dec = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(solar_dec), -1.0, 1.0))
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(solar_dec) + np.cos(lat_rad) * np.cos(solar_dec) * np.sin(ws)
    )
    
    # 5. Clear-sky Radiation (Rso) & Net Radiation (Rn)
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rns = (1.0 - 0.23) * rs
    
    cloud_factor = np.clip(1.35 * (rs / rso) - 0.35, 0.05, 1.0)
    sigma = 4.903e-9
    tmax_k4 = (tmax + 273.16) ** 4
    tmin_k4 = (tmin + 273.16) ** 4
    rnl = sigma * ((tmax_k4 + tmin_k4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * cloud_factor
    rn = rns - rnl
    
    # 6. Penman-Monteith Equation
    numerator = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * wind * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind)
    et0 = np.clip(numerator / denominator, 0.0, None)
    
    return pd.Series(et0, index=df.index, name='ET0')

# Compute ET0 for Hisar (Latitude: 29.15, Elevation: 215m)
df_weather['ET0'] = calculate_fao56_penman_monteith(df_weather, latitude_deg=29.15, elevation_m=215.0)

print("FAO-56 Penman-Monteith ET0 successfully calculated!")
print("\nSummary Statistics of Target ET0 (mm/day):")
display(df_weather['ET0'].describe().round(3))
''')

# Cell 7: Visualizations
add_md(r"""## 6. Exploratory Visualizations & Seasonal Dynamics""")

add_code(r'''# 1. Multi-Year Daily Time Series of ET0
plt.figure(figsize=(15, 5))
plt.plot(df_weather.index, df_weather['ET0'], color='#1f77b4', lw=0.7, alpha=0.75, label='Daily ET0')
plt.plot(df_weather['ET0'].rolling(window=30, center=True).mean(), color='#d62728', lw=2.2, label='30-Day Rolling Average')
plt.title('Daily Reference Evapotranspiration (ET0) — Hisar, Haryana (2010–2023)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('ET0 (mm/day)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
''')

add_code(r'''# 2. Monthly Boxplot showing Seasonal Distribution
df_weather['Month'] = df_weather.index.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='ET0', data=df_weather, order=month_order, palette='YlOrRd')
plt.title('Monthly Seasonal Distribution of Daily ET0 in Northern India', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('ET0 (mm/day)')
plt.tight_layout()
plt.show()
''')

add_code(r'''# 3. Correlation Heatmap
feature_cols = ['max_temp', 'min_temp', 'mean_temp', 'relative_humidity', 'wind_speed', 'solar_radiation', 'ET0']
corr_matrix = df_weather[feature_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Pearson Correlation: Meteorological Drivers vs. Target ET0', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
''')

# Cell 8: Export
add_md(r"""## 7. Exporting the ML-Ready Target Dataset

The dataset is now enriched with the ground-truth target:
* **Inputs ($X$):** `max_temp`, `min_temp`, `relative_humidity`, `wind_speed`, `solar_radiation`
* **Target ($y$):** `ET0` (FAO-56 Penman-Monteith Reference Evapotranspiration in mm/day)

This file is now exported and ready for training:
* **Linear Regression** (simple baseline)
* **Support Vector Regression (SVR)** (non-linear baseline)
* **Random Forest Regressor** (main machine learning model)
""")

add_code(r'''from IPython.display import FileLink, display

# 1. Choose output filename
output_filename = 'evapotranspiration_dataset_with_ET0.csv'

# 2. Export the dataframe to CSV
df_export = df_weather.drop(columns=['Month'], errors='ignore')
df_export.to_csv(output_filename, index=True)

print(f"==================================================")
print(f"Dataset successfully exported to: {output_filename}")
print(f"Total Rows: {df_export.shape[0]} | Total Columns: {df_export.shape[1]}")
print(f"Columns included: {list(df_export.columns)}")
print(f"==================================================")

# 3. Preview first 5 rows
print("\nFirst 5 Records:")
display(df_export.head())

# 4. Direct 1-Click Download Link in Kaggle / Jupyter
print("\n>>> Click the link below to download the final CSV to your computer:")
display(FileLink(output_filename))
''')

output_path = "evapotranspiration_fao56_calculation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook generated successfully: {output_path}")
