"""
ESP Anomaly Detection & Diagnostic Engine
Combines physics-based rule checking with machine-learning multivariate anomaly detection:
1. Physics Rule Diagnostics (Gas lock, motor thermal overload, pump wear, severe downthrust)
2. Unsupervised Isolation Forest Anomaly Detection
3. Downhole Equipment Health Index (0 - 100%)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class ESPAnomalyDetector:
    def __init__(
        self,
        bubble_point_psi: float = 1700.0,
        temp_overheat_c: float = 115.0,
        temp_warning_c: float = 100.0,
        contamination_rate: float = 0.08
    ):
        self.p_bubble = bubble_point_psi
        self.t_overheat = temp_overheat_c
        self.t_warning = temp_warning_c
        self.contamination = contamination_rate

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=120,
            contamination=self.contamination,
            random_state=42
        )
        self.is_fitted = False

    def evaluate_physics_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies petroleum domain physical rules to diagnose specific fault mechanisms.
        """
        diag = pd.DataFrame(index=df.index)

        p_int = df["INTAKE_PRESSURE_PSI"].values
        temp = df["DOWNHOLE_TEMP_C"].values
        dp = df["PUMP_DELTA_P_PSI"].values
        rate = df["LIQUID_RATE_BPD"].values

        # Rule 1: Gas Interference / Cavitation Risk
        # Intake pressure below bubble point causes free gas breakout, reducing pump head
        gas_risk = (p_int < self.p_bubble) & (p_int > 0) & (rate > 500)
        diag["FAULT_GAS_INTERFERENCE"] = gas_risk.astype(int)

        # Rule 2: Motor Thermal Overload
        thermal_alarm = temp >= self.t_overheat
        thermal_warn = (temp >= self.t_warning) & (temp < self.t_overheat)
        diag["FAULT_MOTOR_OVERHEAT"] = thermal_alarm.astype(int)
        diag["WARN_MOTOR_HIGH_TEMP"] = thermal_warn.astype(int)

        # Rule 3: Severe Low-Flow / Downthrust Hazard
        downthrust = (rate < 3500.0) & (rate > 50.0) & (dp > 500.0)
        diag["FAULT_DOWNTHRUST_HAZARD"] = downthrust.astype(int)

        # Rule 4: Choked Wellhead / Backpressure Restriction
        choked = (df["CHOKE_PCT"].values < 15.0) & (rate > 100.0)
        diag["FAULT_CHOKED_RESTRICTION"] = choked.astype(int)

        return diag

    def fit_ml_detector(self, df_train: pd.DataFrame, feature_cols: list) -> None:
        """Fits Isolation Forest on baseline operating period."""
        X = df_train[feature_cols].dropna().values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

    def predict_health_index(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Computes anomaly labels and continuous Downhole Health Index (0 - 100%).
        """
        if not self.is_fitted:
            self.fit_ml_detector(df, feature_cols)

        X = df[feature_cols].ffill().fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Raw anomaly score: negative for anomalies, positive for normal
        raw_scores = self.model.decision_function(X_scaled)
        binary_pred = self.model.predict(X_scaled)  # -1 for anomaly, +1 for normal

        # Normalize score into Health Index: 0% (Critical Alarm) to 100% (Pristine Health)
        # Using sigmoid scaling
        min_score, max_score = np.percentile(raw_scores, 2), np.percentile(raw_scores, 98)
        norm_score = np.clip((raw_scores - min_score) / (max_score - min_score + 1e-12), 0.0, 1.0)
        health_index = np.round(norm_score * 100.0, 1)

        result = pd.DataFrame({
            "ML_IS_ANOMALY": np.where(binary_pred == -1, 1, 0),
            "RAW_ANOMALY_SCORE": np.round(raw_scores, 4),
            "DOWNHOLE_HEALTH_INDEX": health_index
        }, index=df.index)

        # Add health status category
        conditions = [
            health_index >= 70.0,
            (health_index >= 40.0) & (health_index < 70.0),
            health_index < 40.0
        ]
        choices = ["Healthy / Normal", "Degraded / Caution", "Critical / High Risk"]
        result["HEALTH_STATUS"] = np.select(conditions, choices, default="Normal")

        return result
