"""
Artificial Lift Evaluation Metrics Module
Evaluates operating envelope adherence and anomaly detection performance against real operational ground truth.
"""

import numpy as np
import pandas as pd


def compute_operating_envelope_metrics(regimes: np.ndarray) -> dict:
    """Computes distribution of operating time across thrust regimes."""
    total = len(regimes)
    if total == 0:
        return {}
    
    unique, counts = np.unique(regimes, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    safe_count = counts_dict.get("Safe Operating Zone (ROR)", 0)
    down_count = counts_dict.get("Downthrust Hazard", 0)
    up_count = counts_dict.get("Upthrust Hazard", 0)

    return {
        "Total_Producing_Days": total,
        "Safe_ROR_Days": safe_count,
        "Safe_ROR_Pct": round((safe_count / total) * 100.0, 2),
        "Downthrust_Hazard_Days": down_count,
        "Downthrust_Hazard_Pct": round((down_count / total) * 100.0, 2),
        "Upthrust_Hazard_Days": up_count,
        "Upthrust_Hazard_Pct": round((up_count / total) * 100.0, 2)
    }


def evaluate_anomaly_detection_performance(y_actual: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes confusion matrix and classification metrics:
    - Precision: TP / (TP + FP)
    - Recall (Sensitivity): TP / (TP + FN)
    - F1-Score: Harmonic mean of precision and recall
    - False Alarm Rate: FP / (FP + TN)
    """
    y_act = np.asarray(y_actual, dtype=int)
    y_pr = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_act == 1) & (y_pr == 1)))
    fp = int(np.sum((y_act == 0) & (y_pr == 1)))
    fn = int(np.sum((y_act == 1) & (y_pr == 0)))
    tn = int(np.sum((y_act == 0) & (y_pr == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "False_Alarm_Rate_Pct": round(false_alarm_rate * 100.0, 2)
    }
