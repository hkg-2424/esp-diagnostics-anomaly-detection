"""
Main Artificial Lift (ESP) Diagnostic & Surveillance Pipeline
Orchestrates:
1. Ingestion of real downhole telemetry
2. Physics transformation: TDH, fluid density, affinity laws
3. Operating envelope classification (Safe ROR, Downthrust, Upthrust)
4. Multi-fault physics rule diagnostics & ML Health Index estimation
5. Operational accuracy benchmarking against ground-truth field upsets
"""

import os
import pandas as pd
import numpy as np

from .data.esp_data_loader import load_esp_well_telemetry
from .models.esp_physics import calculate_mixture_specific_gravity, calculate_tdh, ESPStagePerformanceCurve
from .models.anomaly_detector import ESPAnomalyDetector
from .evaluation.lift_metrics import compute_operating_envelope_metrics, evaluate_anomaly_detection_performance
from .visualization.lift_plots import plot_esp_envelope_and_diagnostics


class ESPPerformanceAnalyzer:
    def __init__(self, well_name: str = "15/9-F-12", num_stages: int = 140):
        self.well_name = well_name
        self.pump_curve = ESPStagePerformanceCurve(num_stages=num_stages)
        self.detector = ESPAnomalyDetector()
        self.processed_data = None
        self.envelope_metrics = None
        self.anomaly_metrics = None

    def run_surveillance(self, output_plot_path: str = None) -> dict:
        """Executes full diagnostic and predictive surveillance workflow."""
        df = load_esp_well_telemetry(self.well_name)

        # 1. Physical Hydraulic Calculations
        # Calculate in-situ mixture specific gravity based on water cut
        df["WATER_CUT_PCT"] = df["WATER_CUT"].fillna(0.0)
        df["GAMMA_MIX"] = [
            calculate_mixture_specific_gravity(wc, api_gravity=35.0)
            for wc in df["WATER_CUT_PCT"]
        ]

        # Calculate Total Dynamic Head (TDH) in feet
        df["TDH_FEET"] = calculate_tdh(
            pump_delta_p_psi=df["PUMP_DELTA_P_PSI"].values,
            gamma_mix=df["GAMMA_MIX"].values
        )

        # 2. Operating Envelope & Thrust Classification
        df["OPERATING_REGIME"] = self.pump_curve.batch_classify_regimes(df["LIQUID_RATE_BPD"].values)
        self.envelope_metrics = compute_operating_envelope_metrics(df["OPERATING_REGIME"].values)

        # 3. Physics Rules Diagnostic
        physics_diag = self.detector.evaluate_physics_rules(df)
        for col in physics_diag.columns:
            df[col] = physics_diag[col]

        # 4. Machine Learning Anomaly Detection & Downhole Health Index
        ml_features = [
            "INTAKE_PRESSURE_PSI",
            "PUMP_DELTA_P_PSI",
            "DOWNHOLE_TEMP_C",
            "LIQUID_RATE_BPD",
            "CHOKE_PCT",
            "TDH_FEET"
        ]
        
        # Train ML on first 25% of timeline (presumed initial healthy baseline)
        n_train = max(int(len(df) * 0.25), 50)
        df_train = df.iloc[:n_train]
        self.detector.fit_ml_detector(df_train, ml_features)

        ml_results = self.detector.predict_health_index(df, ml_features)
        for col in ml_results.columns:
            df[col] = ml_results[col]

        # 5. Evaluate Anomaly Detection Against Ground-Truth Operational Upsets
        self.anomaly_metrics = evaluate_anomaly_detection_performance(
            y_actual=df["IS_OPERATIONAL_UPSET"].values,
            y_pred=df["ML_IS_ANOMALY"].values
        )

        # 6. Generate Publication-Quality Plots
        if output_plot_path:
            plot_esp_envelope_and_diagnostics(
                df=df,
                pump_curve=self.pump_curve,
                well_name=self.well_name,
                output_path=output_plot_path
            )

        self.processed_data = df

        return {
            "well_name": self.well_name,
            "total_records": len(df),
            "operating_envelope": self.envelope_metrics,
            "anomaly_accuracy_benchmark": self.anomaly_metrics,
            "average_health_index": round(float(df["DOWNHOLE_HEALTH_INDEX"].mean()), 2),
            "critical_days_count": int(np.sum(df["DOWNHOLE_HEALTH_INDEX"] < 40.0)),
            "gas_interference_events": int(df["FAULT_GAS_INTERFERENCE"].sum()),
            "thermal_overheat_events": int(df["FAULT_MOTOR_OVERHEAT"].sum())
        }
