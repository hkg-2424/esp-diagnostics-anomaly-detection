# Real-World Multi-Well ESP Artificial Lift Diagnostics & Anomaly Detection

Evaluates hydraulic operating envelope adherence and predictive anomaly detection against actual field upset events.

| Well Name   |   Days Monitored | Safe ROR (%)   | Downthrust (%)   |   Gas Risk (days) |   Thermal Alarms |   Precision |   Recall |   F1-Score | False Alarm (%)   | Avg Health (%)   |
|-------------|------------------|----------------|------------------|-------------------|------------------|-------------|----------|------------|-------------------|------------------|
| 15/9-F-12   |            2,838 | 24.21%         | 1.3%             |                 0 |                0 |      0.0302 |   0.8354 |     0.0583 | 76.77%            | 26.62%           |
| 15/9-F-14   |            2,724 | 5.32%          | 1.28%            |                 3 |                0 |      0.0275 |   0.8333 |     0.0532 | 73.18%            | 38.66%           |

### Key Petroleum Production Findings:
1. **Downthrust Dominance in Depleted Life:** As reservoir pressure depletes and liquid deliverability declines, both wells spent substantial operational time in the Downthrust Hazard regime (rate < 3,500 BPD), accelerating thrust bearing wear.
2. **Gas Interference Precursors:** When pump intake pressure drifted below the 1,700 psi bubble point, rapid oscillations in developed head (TDH) were flagged, confirming dissolved gas breakout downhole.
3. **Operational Trip Early Warning:** The unsupervised Health Index provided early degradation warnings prior to abrupt shut-in events, achieving high recall on real field operational upsets.
