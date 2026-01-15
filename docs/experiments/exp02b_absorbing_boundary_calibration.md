# Experiment 02B — Absorbing Boundary Calibration (Mode A, Per-Architecture)

**Status**: Production Hardened  
**Date**: 2026-01-15  
**Hardware**: CUDA (A100/L40S class)  
**Dataset**: CIFAR-10 (resized to 224x224, ImageNet normalization)

---

## 1. Context and Purpose

In the study of **Collapse Dynamics**, we model neural inference as a dissipative process that "collapses" toward a categorical decision. A critical operational requirement for this research is defining a **reachable dominance criterion** (or absorbing boundary). If a dominance threshold is set too high or too rigidly, models under evaluation—particularly under domain shift—may never technically "reach" the categorical state, leading to "No Dominance" artifacts that obscure the underlying dynamics.

Experiment 02B serves to calibrate these boundaries. The goal is to establish a criterion that is diagnostic across divergent architectures (ResNet vs. ViT), providing a stable machine-readable definition of the commitment point $t_d$ to be consumed by downstream investigations into hysteresis and irreversibility (Experiment 03).

---

## 2. Experimental Design

This experiment employs a comparative, falsifiable protocol to sweep candidate dominance criteria.

### Models and Data
- **Architectures**: ImageNet-pretrained **ResNet18** (torchvision) vs. **ViT Tiny patch16 224** (timm).
- **Dataset**: CIFAR-10, upsampled to 224x224 with standard ImageNet-1k normalization to induce adaptation pressure.

### Instrumentation
- **Probes**: Each layer's representation is instrumented with a linear probe (feature-dim → 1000) distilled from the final model logits using a Mean Squared Error (MSE) loss.
- **Metric**: The primary metric is the **Logit Margin** $\Delta_t$:
  $$\Delta_t = \text{logit}_{top1} - \text{logit}_{top2}$$
- **Persistence**: Dominance is defined as and when the criterion is satisfied for `REQ_CONSECUTIVE` layers. Based on Experiment 02 locking, this is set to $PERSISTENCE\_WINDOW + 1 = 3$ layers.

### Criteria Families Swept
1. **Relative**: $\Delta_t \ge \alpha \cdot \Delta_{final}$
   - Includes an `EPS_FINAL_MARGIN` (0.001) safeguard to prevent division-by-zero or instability on near-random inputs.
2. **Adaptive Absolute (Mode A)**: Per-architecture percentile thresholds on $\Delta_{final}$.

---

## 3. Outputs and Metrics

The calibration results in three primary investigative outputs:
1. **Mean Margin Trajectory**: Visualization of how logit margin accumulates across depth.
2. **Reach Rate vs. Parameter**: A sweep showing what fraction of samples reach the boundary as the threshold ($\alpha$ or percentile) is varied.
3. **$t_d$ Distribution**: The histogram of indices where dominance becomes persistent for a given boundary.

### Definitions
- **Reach Rate**: The fraction of the evaluation set that successfully satisfies the dominance + persistence criteria at any point during the forward pass.
- **$t_d$ (Commitment Point)**: The first depth index (normalized 0.0 to 1.0) where the model's logits enter a state of persistent dominance.

---

## 4. Key Results

Based on the automated selection logic in `boundary_spec.json`, the following boundary was locked:

### Selected Boundary
- **Type**: Relative
- **Setting**: $\alpha = 0.95$ (Alpha=0.95)
- **Reach Band Target**: 0.2 – 0.95

### Observed Performance
| Metric | ResNet18 | ViT Tiny |
| :--- | :--- | :--- |
| **Reach Rate** | $\approx 0.948$ | $\approx 0.406$ |
| **Mean $t_d$** | $\approx 0.037$ | $\approx 0.232$ |
| **Cohen's d** | 1.098 (Effect Size) | |

### Interpretation
The data suggests fundamentally **different commitment dynamics** between architectures. Under the $\alpha=0.95$ criterion, ResNet18 enters dominance remarkably early (mean $t_d \approx 0.037$) and very robustly across the sample set. In contrast, the ViT architecture commits significantly later ($t_d \approx 0.232$) and fails to reach this relative dominance for a large portion of the samples ($\approx 40\%$ reach).

This supports the hypothesis that ResNet employs an "early-lock" strategy with high smoothness, while ViT requires significantly more depth to resolve the categorical collapse.

---

## 5. Limitations and Failure Modes

- **Asymmetric Reach**: The high reach for ResNet vs. moderate reach for ViT indicates that the boundary is highly diagnostic but not universal; some samples in the ViT fail to "surface" into dominated space.
- **Probe Bias**: The dominance is measured in the space of distilled probes. Any bias in the distillation process (training samples for the probe) will be inherited by the dominance metric.
- **Relative Sensitivity**: The relative-to-final criterion ($\alpha$) can become sensitive when the final margin $\Delta_{final}$ is extremely small. While the `EPS_FINAL_MARGIN` mitigates this, near-boundary cases can still produce jittery results.

---

## 6. Falsifiable Predictions (Diagnostic Calibration)

To validate this as a stable diagnostic instrument, the following pass/fail tests are proposed:

1. **Subsampling Stability (Pass/Fail)**: Re-running the calibration across different random seeds and sample subsets must preserve the ordinal relationship: Reach(ResNet) > Reach(ViT) AND Mean $t_d$(ResNet) < Mean $t_d$(ViT).
2. **Effect Size Significance**: The Bootstrap 95% Confidence Interval for Cohen's $d$ ($d \approx 1.1$) must exclude zero.
3. **Event-Align Check**: If Experiment 02 instability/sensitivity curves are aligned to the detected $t_d$, there should be a measurable increase in local sensitivity around the commitment point for the "snapping" architecture.

---

## 7. Next Steps: Experiment 03 Handoff

The machine-readable configuration [boundary_spec.json](./exp02b_artifacts/boundary_spec.json) has been exported. This file will serve as the "ground truth" definition of dominance for **Experiment 03 (Irreversibility Horizon)**, ensuring that measurements of hysteresis and path-reversal are performed relative to a calibrated commitment point.

---

## Reproduction and Artifacts

### Artifacts (Local)
- [Boundary Specification](./exp02b_artifacts/boundary_spec.json)
- [Calibration Results (CSV)](./exp02b_artifacts/calibration_results.csv)
- [Mean Trajectory & Reach Plot (PNG)](./exp02b_artifacts/download.png)

### Reproduction
- **Requirements**: `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `matplotlib`
- **Source**: [Colab Reproduction Notebook](https://colab.research.google.com/drive/1fkaC63lzhEeoEYDCYkF2hdRovMmmYs6Y?usp=sharing)
