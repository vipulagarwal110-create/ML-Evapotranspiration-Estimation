"""
Data Cleaning and Preprocessing Module.
Cleans raw NASA POWER data, replaces invalid fill codes, and formats features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def clean_and_preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw meteorological data:
    1. Replaces NASA POWER missing value fill code (-999.0) with NaN.
    2. Performs linear interpolation for isolated missing values.
    3. Validates physical consistency bounds.
    4. Computes calendar features (day of year, month, year).
    
    Args:
        df: Raw DataFrame from NASA POWER API.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and quality report dictionary.
    """
    df_clean = df.copy()
    
    # 1. Replace fill values (-999.0)
    fill_counts = (df_clean == -999.0).sum().to_dict()
    df_clean.replace(-999.0, np.nan, inplace=True)
    
    initial_nans = df_clean.isna().sum().to_dict()
    
    # 2. Interpolate if any NaNs exist (time-series linear interpolation)
    if df_clean.isna().sum().sum() > 0:
        df_clean.interpolate(method="time", inplace=True)
        df_clean.bfill(inplace=True)
        df_clean.ffill(inplace=True)
        
    # 3. Physical consistency checks
    if "T2M_MAX" in df_clean.columns and "T2M_MIN" in df_clean.columns:
        # Swap if T2M_MAX < T2M_MIN due to any edge anomaly
        inverted = df_clean["T2M_MAX"] < df_clean["T2M_MIN"]
        if inverted.any():
            t_max_orig = df_clean.loc[inverted, "T2M_MAX"]
            df_clean.loc[inverted, "T2M_MAX"] = df_clean.loc[inverted, "T2M_MIN"]
            df_clean.loc[inverted, "T2M_MIN"] = t_max_orig
            
    if "RH2M" in df_clean.columns:
        df_clean["RH2M"] = df_clean["RH2M"].clip(0.0, 100.0)
        
    if "WS2M" in df_clean.columns:
        df_clean["WS2M"] = df_clean["WS2M"].clip(lower=0.0)
        
    if "ALLSKY_SFC_SW_DWN" in df_clean.columns:
        df_clean["ALLSKY_SFC_SW_DWN"] = df_clean["ALLSKY_SFC_SW_DWN"].clip(lower=0.0)
        
    # Standardize column names
    column_mapping = {
        "T2M_MAX": "max_temp",
        "T2M_MIN": "min_temp",
        "T2M": "mean_temp",
        "RH2M": "relative_humidity",
        "WS2M": "wind_speed",
        "ALLSKY_SFC_SW_DWN": "solar_radiation",
        "PS": "surface_pressure"
    }
    df_clean.rename(columns={k: v for k, v in column_mapping.items() if k in df_clean.columns}, inplace=True)
    
    # 4. Add temporal features useful for analysis
    df_clean["day_of_year"] = df_clean.index.dayofyear
    df_clean["month"] = df_clean.index.month
    df_clean["year"] = df_clean.index.year
    
    report = {
        "total_records": len(df_clean),
        "start_date": df_clean.index[0].strftime("%Y-%m-%d"),
        "end_date": df_clean.index[-1].strftime("%Y-%m-%d"),
        "fill_values_detected": fill_counts,
        "nans_interpolated": initial_nans,
        "features": list(df_clean.columns)
    }
    
    return df_clean, report
