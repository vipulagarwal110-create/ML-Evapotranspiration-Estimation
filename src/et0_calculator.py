"""
Reference Evapotranspiration (ET0) Calculator.
Implements the standardized FAO-56 Penman-Monteith method (Allen et al., 1998).
"""

import numpy as np
import pandas as pd
import pyet

def calculate_fao56_penman_monteith(
    df: pd.DataFrame,
    latitude_deg: float,
    elevation_m: float,
    col_tmax: str = "T2M_MAX",
    col_tmin: str = "T2M_MIN",
    col_tmean: str = "T2M",
    col_rh: str = "RH2M",
    col_wind: str = "WS2M",
    col_rs: str = "ALLSKY_SFC_SW_DWN",
    col_ps: str = "PS"
) -> pd.Series:
    """
    Calculate daily FAO-56 Penman-Monteith Reference Evapotranspiration (ET0) in mm/day.
    
    Equations follow FAO Irrigation and Drainage Paper No. 56:
        ET0 = [0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)] /
              [delta + gamma * (1 + 0.34 * u2)]
    
    Args:
        df: DataFrame containing meteorological time series with DatetimeIndex.
        latitude_deg: Latitude of the station in decimal degrees.
        elevation_m: Altitude / elevation of station above sea level (meters).
        col_tmax: Column name for Maximum Temperature (°C).
        col_tmin: Column name for Minimum Temperature (°C).
        col_tmean: Column name for Mean Temperature (°C).
        col_rh: Column name for Mean Relative Humidity (%).
        col_wind: Column name for Wind Speed at 2m (m/s).
        col_rs: Column name for Solar Radiation (MJ/m^2/day).
        col_ps: Column name for Surface Atmospheric Pressure (kPa).
        
    Returns:
        pd.Series: Daily ET0 in mm/day.
    """
    lat_rad = np.deg2rad(latitude_deg)
    
    pressure_series = df[col_ps] if col_ps in df.columns else None
    
    # Calculate using verified pyet FAO-56 implementation
    et0_series = pyet.pm_fao56(
        tmean=df[col_tmean],
        wind=df[col_wind],
        rs=df[col_rs],
        tmax=df[col_tmax],
        tmin=df[col_tmin],
        rh=df[col_rh],
        pressure=pressure_series,
        elevation=elevation_m,
        lat=lat_rad,
        clip_zero=True
    )
    
    et0_series.name = "ET0"
    return et0_series
