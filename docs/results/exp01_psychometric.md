# Results: Experiment 01 — Psychometric Stress Test

## Overview
- **Models**: ImageNet-pretrained ResNet18 vs Vision Transformer (ViT).
- **Dataset**: CIFAR-10 under domain shift (simulated via directed input noise).
- **Primary Metric**: Collapse Width $\sigma$.

![Psychometric Stress Test Results](../../figures/exp01_psychometric.png)

## Findings
The experiment measured the transition speed between decision states along interpolation paths.

| Model | Mean Collapse Width ($\sigma$) | Stability |
|-------|--------------------------------|-----------|
| **ResNet18** | $\approx 0.07$ | Moderate |
| **ViT** | $\approx 0.04$ | Brittle |

### Key Observations
1. **ViT Brittleness**: ViT architectures exhibit a significantly narrower Collapse Width compared to ResNet18. The transition from Class A to Class B is more abrupt, suggesting that Transformers may rely on sharper, less robust decision boundaries.
2. **Distributional Overlap**: While the means differ significantly, there is measurable overlap in individual transition samples, suggesting the effect is architecture-wide but sample-dependent.
3. **Domain Shift Impact**: Both models showed reduced $\sigma$ under domain shift, but ViT performance degraded into "step-function" behavior more rapidly.

## Interpretation
ViT shows more brittle collapse dynamics. This supports the architectural implication that the global attention mechanism, while powerful, produces decision boundaries that lack the local smoothness found in convolutional biases.

---
**Status**: Completed Result
