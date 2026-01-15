# Experiment 02B — Absorbing Boundary Calibration: Research Report

**Status**: Production Hardened / Research Complete  
**Date**: 2026-01-15  
**Hardware**: CUDA (A100/L40S class / Local MPS)  
**Dataset**: CIFAR-10 (Resized 224, ImageNet Normalization)  
**Reproduction**: [Colab Notebook](https://colab.research.google.com/drive/1fkaC63lzhEeoEYDCYkF2hdRovMmmYs6Y?usp=sharing)

---

## 1. Context and Purpose

Experiment 02B is a critical calibration phase for the **Collapse Dynamics** research program. While Experiment 02 instrumented the "Lyapunov Scan" to measure sensitivity and state change across depth, it faced a "No Dominance" artifact under domain shift: if the absorbing boundary (delta) is set too high, the model's logits may never satisfy the criterion, especially for architectures like ViT that exhibit late-stage commitment.

The purpose of this study is to define a **reachable and diagnostic dominance criterion**. By sweeping alpha-relative and percentile-absolute thresholds, we seek a boundary that is technically reachable for both ResNet and ViT architectures while preserving the discriminative power needed to measure their divergent commitment dynamics.

---

## 2. Experimental Design

### Infrastructure
- **Models**: `resnet18` (torchvision) vs. `vit_tiny_patch16_224` (timm).
- **Probes**: Per-layer linear probes (feature_dim → 1000) self-distilled from final logits using MSE loss.
- **Metric**: Logit Margin $\Delta_t = \text{logit}_{top1} - \text{logit}_{top2}$.
- **Persistence Window**: Locked at $REQ\_CONSECUTIVE = 3$ layers (Persistence Window = 2) to filter transient noise.

### Criteria Families
A) **Relative**: $\Delta_t \ge \alpha \cdot \Delta_{final}$  
B) **Matched-Reach**: Selection of $\alpha$ such that architectural reach asymmetry is minimized or at least characterized.  
C) **Absorption-Time**: Measuring the "last violation" depth before final categorical commitment.

---

## 3. Results: Research Step Additions

### Step 1: Matched-Reach Selection (Relative Alpha Sweep)
We performed a sweep of $\alpha \in [0.1, 0.99]$ to find a "shared" alpha that provides sufficient common coverage.

**Key Findings (alpha ≈ 0.89)**:
- **Common Reach**: $\approx 0.430$
- **ResNet Reach**: $\approx 0.950$
- **ViT Reach**: $\approx 0.430$ (Reach-limited by ViT performance under shift)
- **$t_d$ Gap**: Negative (ResNet commits significantly earlier)
- **Effect Size (Cohen's d)**: $\approx 1.07$

This confirms that while a shared boundary allows comparison, the "intrinsic reach" of ResNet is significantly broader than ViT's under the same relative confidence threshold.

### Step 2: Absorption-Time (Last-Violation) Metric
Beyond the first-dominance point ($t_d$), we measured the **Absorption Time** ($t_{abs}$)—the last depth at which the model violated the dominance criterion before staying committed to the final decision.

**Key Results ($\alpha = 0.7$)**:
- **ResNet mean $t_{abs}$**: $\approx 0.477$
- **ViT mean $t_{abs}$**: $\approx 0.836$
- **Mean Violations**: ResNet $\approx 0.9$ vs. ViT $\approx 6.0$
- **Effect Size**: $\approx -0.934$ (Magnitude indicates high architectural separation)

ViT not only commits later but exhibits significantly more "flicker" (violations) before the final collapse, suggesting a less stable convergence path.

---

## 4. Selection Procedures & Tradeoffs

### Balanced-Reach Attempt (Asymmetry Finding)
We attempted to select a boundary by constraining the `reach_gap` strictly to $\le 0.10$. 
- **Outcome**: Documented failure mode. In the current domain-shift regime, the model capacities are so divergent that a balanced gap is only achievable at very low alphas ($\alpha < 0.3$), where the diagnostic utility of "dominance" is lost. 
- **Finding**: Reach asymmetry is an **intrinsic property** of the architecture/data-shift interaction, not a tunable artifact.

### Paired-Subset Selection
To mitigate selection bias (where we only compare the "best" ViT samples against all ResNet samples), we evaluated a **Paired Subset** where both architectures successfully reached dominance.
- **Artifact**: `boundary_spec_paired.json`
- **Utility**: Ensures that $t_d$ comparisons are performed on the same "difficulty-matched" samples.

### Dominance Hardening
A secondary hardening sweep over `req_consecutive` was performed. 
- **Selected Combo**: `req_consecutive=3`, `alpha≈0.70`.
- **Paired Reach**: $\approx 0.492$
- **Global Reach**: ResNet $\approx 0.970$ vs. ViT $\approx 0.504$.

---

## 5. Interpretation & Falsifiable Predictions

### Interpretation
The data supports a **Divergent Collapse Hypothesis**:
1. ResNet employs a **High-Stiffness Early-Lock** strategy: it resolves decisions early ($t_d < 0.1$) and with very few violations.
2. ViT employs a **Late-Stage Abrupt Collapse** strategy: it remains unresolved for over 70% of its depth, followed by a violent and sometimes shaky (high violations) commitment phase.

### Falsifiable Predictions
1. **Prediction 1 (Latency)**: Under any matched-reach $\alpha$ settings, ViT will exhibit a higher $t_d$ and $t_{abs}$ than ResNet.
2. **Prediction 2 (Violations)**: At $\alpha=0.7$, the frequency of dominance violations in ViT will consistently exceed ResNet by at least 3x.
3. **Prediction 3 (Asymmetry)**: The failure of the Balanced-Reach constraint indicates that "Inference Depth" is not a constant that can be normalized across architectures; rather, "effective depth" is a function of the architectural bias.

---

## 6. Artifacts & Reproduction

### Repository Artifacts
The following files in `docs/experiments/exp02b_artifacts/` are required for downstream Exp03 consumption:
- [boundary_spec_matched.json](./exp02b_artifacts/boundary_spec_matched.json): Primary shared-alpha definition.
- [absorption_time_metrics.csv](./exp02b_artifacts/absorption_time_metrics.csv): Violation and $t_{abs}$ raw data.
- [boundary_spec_hardened.json](./exp02b_artifacts/boundary_spec_hardened.json): Selection used for stability-stiffness correlation.
- [paired_alpha_sweep.csv](./exp02b_artifacts/paired_alpha_sweep.csv) (if present): Statistics on difficulty-matched subsets.

### Reproduction Checklist
1. Load [Colab Notebook](https://colab.research.google.com/drive/1fkaC63lzhEeoEYDCYkF2hdRovMmmYs6Y?usp=sharing).
2. Set `MODEL_LIST = ['resnet18', 'vit_tiny_patch16_224']`.
3. Set `ALPHA_RANGE = np.linspace(0.1, 0.99, 20)`.
4. Run `sweep_calibration()` and `calculate_absorption_metrics()`.
5. Expect outputs to match the "Production Hardened" artifacts in this repo.
