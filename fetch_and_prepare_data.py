"""
Execution script to fetch, compute FAO-56 ET0, clean, and export the dataset.
Target Location: Hisar, Haryana, India (Latitude: 29.15, Longitude: 75.72, Elevation: 215m)
Period: 2010-01-01 to 2023-12-31 (14 Years of Daily Observations)
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from src.data_fetcher import fetch_nasa_power_daily
from src.et0_calculator import calculate_fao56_penman_monteith
from src.preprocessor import clean_and_preprocess_dataset

# Location Parameters (Hisar, Haryana, Northern India)
LATITUDE = 29.15
LONGITUDE = 75.72
ELEVATION = 215.0
START_DATE = "20100101"
END_DATE = "20231231"

RAW_DATA_PATH = os.path.join("data", "raw", "nasa_power_hisar_2010_2023.json")
PROCESSED_CSV_PATH = os.path.join("data", "processed", "evapotranspiration_dataset.csv")
METADATA_PATH = os.path.join("data", "processed", "dataset_metadata.json")

def main():
    print("=" * 70)
    print("STEP 1: Fetching Meteorological Data from NASA POWER API")
    print(f"Location: Lat {LATITUDE}, Lon {LONGITUDE}, Elevation: {ELEVATION}m")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
    
    # 1. Fetch raw data
    raw_df = fetch_nasa_power_daily(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=START_DATE,
        end_date=END_DATE,
        save_raw_path=RAW_DATA_PATH
    )
    
    # 2. Compute FAO-56 Penman-Monteith ET0
    print("\n" + "=" * 70)
    print("STEP 2: Computing Reference Evapotranspiration (FAO-56 Penman-Monteith)")
    print("=" * 70)
    raw_df["ET0"] = calculate_fao56_penman_monteith(
        df=raw_df,
        latitude_deg=LATITUDE,
        elevation_m=ELEVATION
    )
    print("FAO-56 ET0 computation complete.")
    print(f"ET0 Statistics:\n{raw_df['ET0'].describe()}")
    
    # 3. Clean and preprocess
    print("\n" + "=" * 70)
    print("STEP 3: Cleaning & Preprocessing Dataset")
    print("=" * 70)
    cleaned_df, report = clean_and_preprocess_dataset(raw_df)
    
    # 4. Save processed dataset
    cleaned_df.to_csv(PROCESSED_CSV_PATH, index=True)
    print(f"\n[OUTPUT] Cleaned dataset saved to: {PROCESSED_CSV_PATH}")
    print(f"[OUTPUT] Dataset shape: {cleaned_df.shape} (Rows: {cleaned_df.shape[0]}, Columns: {cleaned_df.shape[1]})")
    
    # Save metadata
    metadata = {
        "location": {
            "name": "Hisar, Haryana, India",
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "elevation_meters": ELEVATION
        },
        "period": {
            "start": START_DATE,
            "end": END_DATE,
            "total_days": len(cleaned_df)
        },
        "quality_report": report,
        "summary_statistics": cleaned_df.describe().to_dict()
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[OUTPUT] Metadata saved to: {METADATA_PATH}")
    
    # 5. Quick validation plot
    plot_path = os.path.join("data", "processed", "data_overview.png")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    features = ["max_temp", "min_temp", "relative_humidity", "wind_speed", "solar_radiation", "ET0"]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#e6ab02", "#1f78b4"]
    units = ["°C", "°C", "%", "m/s", "MJ/m²/d", "mm/day"]
    
    for ax, feat, col, u in zip(axes, features, colors, units):
        ax.plot(cleaned_df.index, cleaned_df[feat], color=col, alpha=0.7, lw=0.8)
        ax.set_title(f"{feat.replace('_', ' ').title()} ({u})", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel(u)
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OUTPUT] Overview plot saved to: {plot_path}")
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
