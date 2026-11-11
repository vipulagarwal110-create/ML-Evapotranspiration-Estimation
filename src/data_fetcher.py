"""
Data Fetcher Module for NASA POWER Meteorological Data.
Fetches daily agroclimatology data for a specified geographical location and date range.
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, Any, Optional

NASA_POWER_POINT_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

DEFAULT_PARAMS = [
    "T2M_MAX",            # Maximum Temperature (°C)
    "T2M_MIN",            # Minimum Temperature (°C)
    "T2M",                # Mean Temperature (°C)
    "RH2M",               # Relative Humidity (%)
    "WS2M",               # Wind Speed at 2 Meters (m/s)
    "ALLSKY_SFC_SW_DWN",  # All Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)
    "PS",                 # Surface Pressure (kPa)
]

def fetch_nasa_power_daily(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: Optional[list] = None,
    community: str = "AG",
    save_raw_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch daily meteorological data from NASA POWER API.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        start_date: Start date string formatted as YYYYMMDD
        end_date: End date string formatted as YYYYMMDD
        parameters: List of NASA POWER parameter names
        community: Data community ('AG' for agroclimatology)
        save_raw_path: Optional file path to save raw JSON response
        
    Returns:
        pd.DataFrame indexed by Datetime with parameter columns.
    """
    if parameters is None:
        parameters = DEFAULT_PARAMS

    payload = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    print(f"[NASA POWER] Querying API for Lat: {latitude}, Lon: {longitude} from {start_date} to {end_date}...")
    response = requests.get(NASA_POWER_POINT_API, params=payload, timeout=90)
    response.raise_for_status()
    data = response.json()

    if save_raw_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_raw_path)), exist_ok=True)
        with open(save_raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[NASA POWER] Raw response saved to {save_raw_path}")

    param_dict = data.get("properties", {}).get("parameter", {})
    if not param_dict:
        raise ValueError("No parameter data found in NASA POWER response.")

    df = pd.DataFrame(param_dict)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df.sort_index(inplace=True)

    print(f"[NASA POWER] Successfully downloaded {len(df)} daily records ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}).")
    return df
