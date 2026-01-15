# Results: Experiment 02 — Lyapunov Stability Scan

## Overview
- **Objective**: Localize where decision commitment happens and how violently the network "locks in" to a categorical state across its depth.
- **Models**: ResNet18 vs ViT-B/16 (ImageNet-pretrained).
- **Core Metrics**: 
    - `s_t`: True sensitivity (normalized margin gradient).
    - `lambda_state`: Representation change rate.
    - `lambda_commit`: Log-sensitivity ratio.

## Summary Results

![Lyapunov Stability Scan Results](../../figures/exp02/stability_scan.png)

### Key Findings
1. **Architectural Divergence**: ResNet18 exhibits a smoother, more gradual increase in commitment sensitivity across its depth. In contrast, ViT-B/16 shows a distinct "Peak Lag" where commitment happens significantly later in the network's depth but with a much higher intensity (sharper λ_commit spike).
2. **Structural Stability**: The structural stability (λ_state) remains relatively uniform across both models, suggesting that the "collapse" into a decision is a property of the output transformation rather than a fundamental reorganization of the internal representation space.
3. **Absorbing Boundary**: Initial results showed that static absolute thresholds (δ) are brittle under domain shift. Experiment 02B has been initiated to calibrate adaptive boundaries for more robust cross-model comparison.

## Implementation & Reproducibility
- **Colab Notebook**: [Lyapunov Stability Scan](https://colab.research.google.com/drive/15GBp5msiXxgsUUpa6u5NCZ-WnpjlEgfa?usp=sharing)
- **Local Script**: `experiments/exp02_lyapunov_stability_scan_final_patched.py`

---
**Status**: Instrument-Complete
