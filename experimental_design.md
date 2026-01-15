# Experimental Design: Experiment 01 — Psychometric Stress Test

## Motivation
Traditional metrics focus on top-line accuracy, which ignores the "stiffness" of the decision-making process. This experiment probes the **collapse behavior** of neural networks—how abruptly and brittlely they transition between class identities when presented with perturbed inputs. This "decision stiffness" is a fundamental measure of behavioral stability.

## Protocol: Directed Interpolation
We evaluate the model's behavior along a linear trajectory between two distinct input samples x_A and x_B (e.g., a "cat" and a "dog" image):

x(alpha) = (1 - alpha)x_A + alpha * x_B

where alpha in [0, 1].

We measure the Softmax probabilities P(class_A) and P(class_B) across this path.

## Metric: Collapse Width (σ)
The **Collapse Width** σ is defined as the width of the transition region where the model's confidence is not dominated by either x_A or x_B. A smaller σ indicates a more brittle and abrupt collapse from one decision to another, suggesting a lack of behavioral robustness.

## Hypotheses
- **H1 (Brittleness)**: Vision Transformers (ViT) will exhibit smaller Collapse Widths (σ) compared to convolutional architectures (ResNet) at matched accuracy, indicating more brittle decision boundaries.
- **H2 (Domain Shift)**: Under domain shift (low confidence regimes), the collapse will become more "jittery" or non-monotonic, reflecting unstable dynamics.

## Planned Mechanistic Follow-ups
1. **Stability Scan**: Correlating Collapse Width with Lyapunov exponents of internal layers.
2. **Irreversibility**: Measuring when in the interpolation path the network loses the ability to "recover" the original class identity through small perturbations.

## Results & Implementation
- **Colab Notebook**: [Psychometric Stress Test Implementation](https://colab.research.google.com/drive/1jo3duE-NQCSI0sbxEDCCiqS5vDHPmVFS?usp=sharing)
- **Results Summary**: [exp01_psychometric.md](docs/results/exp01_psychometric.md)

---
**Status**: Completed (Falsification Applied)
