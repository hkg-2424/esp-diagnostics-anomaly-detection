"""
ESP Telemetry Data Loader Module
Extracts, calibrates, and filters real-world downhole sensor streams for ESP-lifted wells
from the Equinor Volve Field dataset (Wells 15/9-F-12 and 15/9-F-14).
"""

import os
import pandas as pd
import numpy as np

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
LOCAL_VOLVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "dca", "datasets", "volve_production_data.csv")
REMOTE_VOLVE_URL = "https://raw.githubusercontent.com/jamiubadmusng/oil-and-gas/main/data/processed/production_data_processed.csv"


def load_esp_well_telemetry(well_name: str = "15/9-F-12") -> pd.DataFrame:
    """
    Loads daily sensor telemetry for an ESP well in the Volve field.
    Standardizes units:
    - Pressures: Converted from bar to psi (1 bar = 14.5038 psi)
    - Temperatures: In Celsius
    - Rates: Converted to STB/D and Liquid BPD
    - Identifies operational status: Normal flowing, Choked/Throttled, or Shut-in.
    """
    # Check local datasets directory first, then fallback to sibling or remote
    cached_file = os.path.join(DATA_CACHE_DIR, "volve_production_data.csv")
    if os.path.exists(cached_file):
        df = pd.read_csv(cached_file, parse_dates=["DATEPRD"])
    elif os.path.exists(LOCAL_VOLVE_PATH):
        df = pd.read_csv(LOCAL_VOLVE_PATH, parse_dates=["DATEPRD"])
    else:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        if not os.path.exists(cached_file):
            print(f"Downloading Volve production data from {REMOTE_VOLVE_URL} ...")
            import urllib.request
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(REMOTE_VOLVE_URL, headers=headers)
            with urllib.request.urlopen(req) as resp, open(cached_file, "wb") as f:
                f.write(resp.read())
        df = pd.read_csv(cached_file, parse_dates=["DATEPRD"])

    well_df = df[df["NPD_WELL_BORE_NAME"] == well_name].copy()
    if well_df.empty:
        raise ValueError(f"Well '{well_name}' not found in Volve dataset.")

    well_df = well_df.sort_values("DATEPRD").reset_index(drop=True)

    # Unit Conversions (bar to psi)
    BAR_TO_PSI = 14.50377
    SM3_TO_BBL = 6.28981

    well_df["INTAKE_PRESSURE_PSI"] = well_df["AVG_DOWNHOLE_PRESSURE"] * BAR_TO_PSI
    well_df["DOWNHOLE_TEMP_C"] = well_df["AVG_DOWNHOLE_TEMPERATURE"]
    well_df["DP_TUBING_PSI"] = well_df["AVG_DP_TUBING"] * BAR_TO_PSI
    well_df["WHP_PSI"] = well_df["AVG_WHP_P"] * BAR_TO_PSI
    well_df["WHT_C"] = well_df["AVG_WHT_P"]
    well_df["CHOKE_PCT"] = well_df["AVG_CHOKE_SIZE_P"]

    # Production rates in BPD
    well_df["OIL_RATE_BPD"] = well_df["BORE_OIL_VOL"] * SM3_TO_BBL
    well_df["WATER_RATE_BPD"] = well_df["BORE_WAT_VOL"] * SM3_TO_BBL
    well_df["GAS_RATE_MSCFD"] = (well_df["BORE_GAS_VOL"] * 35.3147) / 1000.0
    well_df["LIQUID_RATE_BPD"] = well_df["OIL_RATE_BPD"] + well_df["WATER_RATE_BPD"]

    # Calculate discharge pressure: P_dis = P_intake + DP_tubing
    well_df["DISCHARGE_PRESSURE_PSI"] = well_df["INTAKE_PRESSURE_PSI"] + well_df["DP_TUBING_PSI"]
    well_df["PUMP_DELTA_P_PSI"] = well_df["DP_TUBING_PSI"]

    # Elapsed days
    start_date = well_df["DATEPRD"].min()
    well_df["DAYS"] = (well_df["DATEPRD"] - start_date).dt.days

    # Ground-truth operational status labeling
    # Operational Trip / Anomaly: Liquid rate drops abruptly > 70% or zero onstream hrs after active flow
    liquid = well_df["LIQUID_RATE_BPD"].values
    onstream = well_df["ON_STREAM_HRS"].values
    
    anomalies = np.zeros(len(well_df), dtype=int)
    for i in range(1, len(well_df)):
        # Case 1: Unplanned shut-in / trip (active yesterday, shut today)
        if onstream[i] < 12.0 and onstream[i-1] >= 20.0:
            anomalies[i] = 1
        # Case 2: Severe rate drop under constant or decreasing intake pressure
        elif liquid[i-1] > 1000.0 and liquid[i] < 0.3 * liquid[i-1]:
            anomalies[i] = 1

    well_df["IS_OPERATIONAL_UPSET"] = anomalies

    return well_df.reset_index(drop=True)
