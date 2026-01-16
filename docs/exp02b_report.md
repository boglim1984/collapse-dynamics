# Experiment 02B — Absorbing Boundary Calibration (Extensions)

**Status**: Research Complete / Production Hardened  
**Date**: 2026-01-15  
**Hardware**: CUDA (A100/L40S class / MPS)  
**Dataset**: CIFAR-10 (Resized 224, ImageNet Normalization)  
**Reproduction**: [Colab Notebook](https://colab.research.google.com/drive/1fkaC63lzhEeoEYDCYkF2hdRovMmmYs6Y?usp=sharing)

---

## 1. Context and Purpose

Experiment 02B establishes the **reachable and diagnostic dominance criterion** required for the Collapse Dynamics program. The goal is to move beyond static, brittle thresholds toward a "best boundary" that is diagnostic of cross-architectural dynamics (ResNet vs. ViT) under domain shift. 

Since the last research push, we have integrated **Reach Constraints** (Balanced/Paired) and a new **Absorption-Time** metric to harden the diagnostic against late-stage commitment jitter.

---

## 2. Experiment 02B Extensions (Reach Constraints + Absorption-Time)

### 1) Matched-Reach Selection (Relative)
We swept $\alpha$ values to find a baseline shared confidence threshold that provides maximum common coverage.

- **Selected Alpha**: 0.89
- **Implied Common Reach**: 0.430
- **ResNet Reach**: 0.950 | **ViT Reach**: 0.430 | **Gap**: 0.520
- **Mean $t_d$ ResNet**: 0.036 | **Mean $t_d$ ViT**: 0.230
- **$t_d$ Gap (R-V)**: -0.194
- **Cohen’s d ($t_d$)**: 1.071 (Strong architectural separation)

### 2) Balanced-Reach Attempt (Relative)
We attempted to select a boundary by strictly constraining the `reach_gap` to ensure fair sample sizes.
- **Result**: No candidates met strict balanced gap constraints (e.g., Gap < 0.10) at diagnostic alpha levels.
- **Fallback**: Best overall `common_reach`.
- **Selected Alpha**: 0.50
- **Common Reach**: 0.636 | **Gap**: 0.352
- **Cohen’s d ($t_d$)**: 0.998
- **Conclusion**: Reach asymmetry is an **intrinsic structural limitation** of the model/data-shift pairing.

### 3) Paired-Subset Selection (Relative)
To control for difficulty, we evaluated only the subset where BOTH architectures reached dominance.
- **Selected Alpha**: 0.50
- **Paired Reach**: 0.626 ($n=313/500$)
- **Cohen’s d (Paired $t_d$)**: 0.901

### 4) Hardened Dominance Sweep (req_consecutive=3)
We swept persistence windows to ensure commitment is not a transient artifact.
- **Constraint**: $\text{paired\_reach}$ maximization with $t_d\_gap < 0$.
- **Selected Alpha**: 0.70
- **Paired Reach**: 0.492 ($n=246/500$)
- **Global Reach**: ResNet=0.970 | ViT=0.504 | **Gap**: 0.466
- **Cohen’s d (Paired $t_d$)**: 0.879

### 5) New Metric: Absorption Time ($t_{abs}$) / Last-Violation
We define $t_{abs}$ as the earliest layer index after the **LAST violation** of the dominance criterion. This ensures that the model has truly "absorbed" into the categorical state without further jitter.

- **Definition**: $t_{abs} = 1 + \max\{ t \mid \text{dominance}[t] == \text{False} \}$, else 0 if never violated. (Normalized to $[0,1]$).
- **Result (Alpha=0.70)**:
    - **ResNet**: Mean $t_{abs} \approx 0.477$ | Median = 0.714 | Mean Violations = 0.9
    - **ViT**: Mean $t_{abs} \approx 0.836$ | Median = 1.000 | Mean Violations = 6.0
    - **Gap (R-V)**: -0.360
    - **Cohen’s d ($t_{abs}$)**: -0.934

---

## 3. Interpretation & Falsifiable Predictions

### Interpretability Summaries
- **Structural Ceiling**: Reach-mismatch is not a tunable artifact; the common_reach ceiling is fundamentally driven by ViT’s lower adaptive reach under shift.
- **Sample Comparability**: The **Paired Subset** provides the most rigorous comparison by controlling for sample-level difficulty.
- **Violent Commitment**: ViT exhibits a significantly larger "jitter window" ($t_{abs} - t_d$) and higher violation counts ($6.0$ vs $0.9$), confirming that its collapse is not only later but also less stable.

### Falsifiable Predictions
1. **Latency Hierachy**: Under matched-reach alpha settings, ViT will **always** commit later (higher $t_d$ and $t_{abs}$) than ResNet.
2. **Commitment Jitter**: Under any shared alpha $> 0.5$, ViT will exhibit at least 3x the pre-absorption violations compared to ResNet.
3. **Instrinsic Asymmetry**: If the Balanced-Reach constraint continues to fail across varied domain shifts, it confirms that "effective inference depth" is an architectural constant, falsifying any claim that depth can be normalized purely via threshold tuning.

---

## 4. Reproduction Checklist

### Python/Colab Environment
- **Script/Notebook**: [Colab link](https://colab.research.google.com/drive/1fkaC63lzhEeoEYDCYkF2hdRovMmmYs6Y?usp=sharing)
- **Dependencies**: `timm`, `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`.
- **CLI Usage**: Ensure scripts use `argparse.parse_known_args()` to avoid conflicts with Jupyter/Colab kernel flags.

### Artifacts to Verify
After running the pipeline, expect the following outputs in the `exp02b_artifacts/` directory:
- `absorption_time_metrics.csv`: Violation counts and $t_{abs}$ depth.
- `matched_alpha_sweep.csv` / `boundary_spec_matched.json`: Global Shared-Alpha results.
- `paired_alpha_sweep.csv` / `boundary_spec_paired.json`: Difficulty-matched subset results.
- `dominance_hardening_sweep.csv` / `boundary_spec_hardened.json`: Persistence-hardened selection.
- `stability_runs.csv`: Cross-run variability table.
