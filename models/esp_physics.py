"""
ESP Hydraulic Physics & Pump Performance Model
Implements:
1. Multiphase downhole fluid mixture density and gradient calculation
2. Total Dynamic Head (TDH) calculation
3. Affinity Laws frequency scaling for Variable Frequency Drives (VFD)
4. Manufacturer Head-Capacity (H-Q) curve and operating thrust regime classification
"""

import numpy as np


def calculate_mixture_specific_gravity(water_cut_pct: float, api_gravity: float = 35.0, water_sg: float = 1.05) -> float:
    """
    Computes downhole in-situ liquid mixture specific gravity based on water cut.
    """
    oil_sg = 141.5 / (131.5 + api_gravity)
    fw = np.clip(water_cut_pct / 100.0, 0.0, 1.0)
    gamma_mix = oil_sg * (1.0 - fw) + water_sg * fw
    return float(gamma_mix)


def calculate_tdh(pump_delta_p_psi: np.ndarray, gamma_mix: np.ndarray) -> np.ndarray:
    """
    Computes Total Dynamic Head (TDH) in feet of liquid column:
    TDH = Delta_P / (0.4335 * gamma_mix)
    """
    gamma_mix = np.maximum(gamma_mix, 0.60)
    gradient = 0.4335 * gamma_mix  # psi per foot
    return pump_delta_p_psi / gradient


def apply_affinity_laws(
    q_actual: np.ndarray,
    h_actual: np.ndarray,
    f_actual: float = 55.0,
    f_rated: float = 60.0
) -> tuple:
    """
    Scales operating point to rated frequency (60 Hz) using pump Affinity Laws:
    Q_rated = Q_actual * (f_rated / f_actual)
    H_rated = H_actual * (f_rated / f_actual)^2
    """
    f_ratio = f_rated / max(f_actual, 20.0)
    q_rated = q_actual * f_ratio
    h_rated = h_actual * (f_ratio ** 2)
    return q_rated, h_rated


class ESPStagePerformanceCurve:
    """
    Represents an industrial multi-stage centrifugal pump assembly (e.g., Reda / Baker Hughes
    high-volume offshore ESP stage type calibrated for North Sea Volve operations).
    """
    def __init__(
        self,
        num_stages: int = 140,
        rated_frequency: float = 60.0,
        ror_min_bpd: float = 3500.0,
        ror_max_bpd: float = 16000.0,
        bep_bpd: float = 9500.0
    ):
        self.num_stages = num_stages
        self.rated_frequency = rated_frequency
        self.ror_min = ror_min_bpd
        self.ror_max = ror_max_bpd
        self.bep = bep_bpd

        # Single stage polynomial H-Q coefficients (head in feet vs rate in BPD)
        # H_stage = H0 - a*Q - b*Q^2
        self.h0_stage = 58.5  # shutoff head per stage (ft)
        self.a_stage = 0.00065
        self.b_stage = 0.000000125

    def get_theoretical_head(self, liquid_rate_bpd: np.ndarray) -> np.ndarray:
        """Computes total expected assembly head (ft) across all stages at rated 60 Hz."""
        q = np.asarray(liquid_rate_bpd, dtype=float)
        q = np.maximum(q, 0.0)
        h_stage = self.h0_stage - (self.a_stage * q) - (self.b_stage * (q ** 2))
        h_stage = np.maximum(h_stage, 0.0)
        return h_stage * self.num_stages

    def classify_operating_regime(self, liquid_rate_bpd: float) -> str:
        """
        Classifies operating state into standard API thrust regimes:
        - Downthrust: Low rate, high axial load pushing downward on thrust bearing
        - Safe ROR: Inside the Recommended Operating Range
        - Upthrust: Excessively high rate, hydraulic lift pushes impellers into diffusers
        """
        if liquid_rate_bpd < self.ror_min:
            return "Downthrust Hazard"
        elif liquid_rate_bpd > self.ror_max:
            return "Upthrust Hazard"
        else:
            return "Safe Operating Zone (ROR)"

    def batch_classify_regimes(self, liquid_rates: np.ndarray) -> np.ndarray:
        rates = np.asarray(liquid_rates)
        regimes = np.empty(len(rates), dtype=object)
        regimes[rates < self.ror_min] = "Downthrust Hazard"
        regimes[rates > self.ror_max] = "Upthrust Hazard"
        regimes[(rates >= self.ror_min) & (rates <= self.ror_max)] = "Safe Operating Zone (ROR)"
        return regimes
