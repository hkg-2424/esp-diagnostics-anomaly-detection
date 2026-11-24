"""
Visualization Module for ESP Performance & Diagnostics
Generates:
1. ESP Operating Envelope (H-Q Performance Curve with Thrust Regimes & Real Points)
2. Downhole Sensor Telemetry & Anomaly Tracking Timeline
3. Equipment Health Index Profile
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_esp_envelope_and_diagnostics(
    df: pd.DataFrame,
    pump_curve,
    well_name: str,
    output_path: str = None
):
    """
    Creates a comprehensive 3-panel artificial lift diagnostic figure:
    Panel 1: ESP Head vs. Capacity (H-Q) Operating Envelope
    Panel 2: Downhole Intake Pressure and Pump Delta P vs. Time
    Panel 3: Downhole Health Index and Anomaly Flagging
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1.3, 1.0, 0.9]})

    # --- Panel 1: ESP Operating Envelope (H-Q Curve) ---
    q_range = np.linspace(0, 22000, 300)
    h_theory = pump_curve.get_theoretical_head(q_range)

    ax1.plot(q_range, h_theory, color="black", linewidth=2.5, label="Manufacturer H-Q Assembly Curve (60 Hz)")
    
    # Shade Thrust Regimes
    ax1.axvspan(0, pump_curve.ror_min, color="gold", alpha=0.25, label="Downthrust Hazard Zone")
    ax1.axvspan(pump_curve.ror_min, pump_curve.ror_max, color="lightgreen", alpha=0.25, label="Recommended Operating Range (Safe ROR)")
    ax1.axvspan(pump_curve.ror_max, 22000, color="lightsalmon", alpha=0.25, label="Upthrust Hazard Zone")

    # Scatter actual operating points
    q_act = df["LIQUID_RATE_BPD"].values
    tdh_act = df["TDH_FEET"].values
    regimes = df["OPERATING_REGIME"].values

    # Color code points
    colors = []
    for r in regimes:
        if r == "Safe Operating Zone (ROR)":
            colors.append("forestgreen")
        elif r == "Downthrust Hazard":
            colors.append("darkorange")
        else:
            colors.append("crimson")

    scatter = ax1.scatter(q_act, tdh_act, c=colors, s=18, alpha=0.6, edgecolors="none")
    ax1.set_title(f"ESP Operating Envelope & Thrust Diagnostic Mapping\n{well_name}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Liquid Production Rate (BPD)", fontsize=11)
    ax1.set_ylabel("Total Dynamic Head (TDH, Feet)", fontsize=11)
    ax1.set_xlim(0, 20000)
    ax1.set_ylim(0, max(np.nanmax(tdh_act) * 1.15, 9000))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # --- Panel 2: Sensor Telemetry Tracking ---
    days = df["DAYS"].values
    ax2.plot(days, df["INTAKE_PRESSURE_PSI"].values, color="dodgerblue", linewidth=1.5, label="Intake Pressure (psi)")
    ax2.plot(days, df["PUMP_DELTA_P_PSI"].values, color="purple", linewidth=1.5, label="Pump Delta P (psi)")
    ax2.axhline(1700.0, color="crimson", linestyle="--", linewidth=1.2, label="Bubble Point Threshold (1,700 psi)")
    
    # Twin axis for Temperature
    ax2_temp = ax2.twinx()
    ax2_temp.plot(days, df["DOWNHOLE_TEMP_C"].values, color="crimson", linewidth=1.2, alpha=0.7, label="Motor Temp (°C)")
    ax2_temp.set_ylabel("Motor Temp (°C)", color="crimson", fontsize=11)
    ax2_temp.tick_params(axis='y', labelcolor='crimson')

    ax2.set_title(f"Downhole Pressures & Thermal Telemetry\n{well_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Elapsed Days on Stream", fontsize=11)
    ax2.set_ylabel("Pressure (psi)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True)

    # --- Panel 3: Equipment Health Index & Anomaly Flags ---
    health = df["DOWNHOLE_HEALTH_INDEX"].values
    ax3.plot(days, health, color="teal", linewidth=1.5, label="Downhole Health Index (0 - 100%)")
    ax3.axhline(70.0, color="green", linestyle=":", alpha=0.7, label="Healthy Threshold (70%)")
    ax3.axhline(40.0, color="red", linestyle=":", alpha=0.7, label="Critical Alarm Threshold (40%)")

    # Highlight anomalies
    anom_days = days[df["ML_IS_ANOMALY"] == 1]
    anom_health = health[df["ML_IS_ANOMALY"] == 1]
    ax3.scatter(anom_days, anom_health, color="crimson", s=25, zorder=5, label="Flagged Anomalies / Trips")

    ax3.set_title(f"Automated Predictive Downhole Health Index\n{well_name}", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Elapsed Days on Stream", fontsize=11)
    ax3.set_ylabel("Health Index (%)", fontsize=11)
    ax3.set_ylim(-5, 105)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"ESP Diagnostic Plot saved to: {output_path}")
    plt.close()
