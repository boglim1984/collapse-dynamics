# Collapse Dynamics: Behavioral Stability and Thermodynamics of Neural Inference

## Thesis
This repository inaugurates a post-geometric research program. Neural networks are treated as dynamical, information-dissipating systems whose core property is how decisions **collapse**, not how representations are arranged. 

Key tenets:
- **Inference is a collapse**: The process of computation is a transition from high-entropy input signals to low-entropy categorical decisions.
- **Geometry is non-operative**: Static representation geometry (manifolds, anisotropy) is considered epiphenomenal and does not constrain execution during inference.
- **Behavioral dynamics are primary**: The correct object of interpretability is the stability and "stiffness" of the decision-making process under perturbation.

This research marks a definitive pivot away from geometric interpretability toward a behavioral and thermodynamic framework.

---

## Experiment Roadmap

### Experiment 01 — Psychometric Stress Test (Completed)
Probing the brittleness of decision boundaries under directed interpolation. Initial results compare ResNet18 and ViT architectures under domain shift.

- **Results**: [exp01_psychometric.md](docs/results/exp01_psychometric.md)
- **Colab**: [Run on Colab](https://colab.research.google.com/drive/1jo3duE-NQCSI0sbxEDCCiqS5vDHPmVFS?usp=sharing)

### Experiment 02 — Lyapunov Stability Scan (Instrument-Complete)
Localizing where decision commitment happens and how violently the network "locks in" to a categorical state across its depth.

- **Objective**: Measure representation stability (lambda_state) vs. decision sensitivity (lambda_commit).
- **Core Metrics**:
  - `d_rel`: Relative state change.
  - `s_t`: True sensitivity (|dMargin| / ||dx_input||).
  - `lambda_state`: log(d_t+1 / d_t).
  - `lambda_commit`: log(s_t+1 / s_t) [clamped].
- **Colab**: [Run on Colab](https://colab.research.google.com/drive/15GBp5msiXxgsUUpa6u5NCZ-WnpjlEgfa?usp=sharing)
- **Local Script**: `experiments/exp02_lyapunov_stability_scan_final_patched.py`

### Experiment 02B — Absorbing Boundary Calibration (Production Hardened)
Defining reachable decision commitment thresholds to ensure cross-model comparability under domain shift.
- **Report**: [Experiment 02B: Multi-Pass Calibration & Absorption Analysis](docs/exp02b_report.md)
- **Latest Update**: [02B Long-Run Stability & Falsifier Analysis (2026-01-16)](experiments/02b/REPORT_02B_UPDATE_2026-01-16.md)
- **Colab**: [Run on Colab](https://colab.research.google.com/drive/1fkaC63lzhEeoEYDCYkF2hdRovMmmYs6Y?usp=sharing)
- **Settings**: Alpha=0.50 (Locked in long-run runs_summary.csv)
- **Local Script**: `experiments/02b/exp02b_absorbing_boundary_calibration.py` (OOM Hardened)

### Experiment 03 — Irreversibility Horizon (Planned)
Quantifying the point-of-no-return in the inference process where the transition from representation to classification becomes informationally irreversible.

---

## Results Summary

### Experiment 01 — Psychometric Stress Test

**Summary Results (schematic, not raw training logs)**

![Psychometric Stress Test Results](figures/exp01_psychometric.png)

*Collapse Width (σ) comparison between ResNet18 and ViT architectures. ViT exhibits significantly narrower σ (≈ 0.04), indicating more brittle decision collapse compared to ResNet (≈ 0.07).*

- **Finding**: ViT decision boundaries are brittle and exhibit "step-function" behavior under perturbation.
- **Interpretation**: Global attention mechanisms lack the local smoothing biases of convolutions, leading to abrupt decision transitions.
- **Traceability**: See [exp01_psychometric.md](docs/results/exp01_psychometric.md) for full analysis.

---

### Experiment 02 — Lyapunov Stability Scan

**Summary Results (Stability Channels)**

![Lyapunov Stability Scan Results](figures/exp02/stability_scan.png)

*Visual analysis of commitment dynamics. Peak Lag and crossover points indicate architectural differences in decision stiffness. Refer to Experiment 02B for boundary calibration details.*

- **Finding**: Multi-channel analysis confirms late-stage sensitivity spikes in ViT compared to earlier, smoother commitment in ResNet.
- **Results**: [exp02_stability_scan.md](docs/results/exp02_stability_scan.md)
- **Colab**: [Run on Colab](https://colab.research.google.com/drive/15GBp5msiXxgsUUpa6u5NCZ-WnpjlEgfa?usp=sharing)
- **Traceability**: Run `experiments/exp02_lyapunov_stability_scan_final_patched.py` for local replication.

---

### Experiment 02B — Absorbing Boundary Calibration

**Summary Results (Reach Rate & t_d Distribution)**

![Absorbing Boundary Curves](docs/experiments/exp02b_artifacts/download.png)

*Comparison of reach rates and commitment depth (t_d) across ResNet and ViT. Locked boundary (Alpha=0.95) reveals significantly different commitment profiles.*

- **Finding**: ResNet commits very early ($t_d \approx 0.037$) with high reach, while ViT commits later ($t_d \approx 0.232$).
- **Long-Run Stability**: 60/60 runs confirm $t_{abs}$ and violation gap direction (ResNet earlier/cleaner).
- **Balanced Reach**: Added a constrained selection rule to ensure fair architecture comparison (see [Addendum](docs/exp02b_report.md#2-experiment-02b-extensions-reach-constraints--absorption-time)).
- **Reports**: [Consolidated Analysis](docs/exp02b_report.md) | [2026-01-16 Update](experiments/02b/REPORT_02B_UPDATE_2026-01-16.md)
- **Artifacts**: [boundary_spec_balanced.json](experiments/02b/data/boundary_spec_balanced.json), [pareto_plot.png](docs/experiments/exp02b_artifacts/pareto_plot.png)

---

## Repository Structure
- `experimental_design.md`: Formal specifications and contracts for all experiments.
- `docs/background.md`: Brief context on the shift from geometric to behavioral paradigms.
- `docs/results/`: Detailed summaries and data from executed experiments.
- `figures/`: Key visualizations and results charts.
- `notebooks/`: Runnable implementations and diagnostics (localized or Colab-linked).

---

## Conclusion
Static geometry has been falsified as a reliable constraint on network execution. **Collapse Dynamics** provides the tools to measure what the network *does*, rather than how it *looks*.
