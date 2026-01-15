# Experiment 02B — Absorbing Boundary Calibration

## Purpose
The primary objective of this study is to define a reachable **absorbing boundary** (decision commitment threshold) without altering the core Experiment 02 instrumentation. 

In the Lyapunov stability scan, the "commitment point" $t_d$ is sensitive to the definition of the boundary $\delta$. Under domain shift (e.g., CIFAR → ImageNet-style noise), a static $\delta$ may fail to be reached, leading to "No Dominance" results. This study aims to calibrate these boundaries to ensure robust detection across architectures.

## Proposed Criteria
We propose two complementary methods for calibration:

### 1. Relative-to-Final Margin
- **Definition**: $\Delta_t / \Delta_{final} \ge \alpha$
- **Verification**: The threshold must be maintained for at least $(k+1)$ consecutive layers to filter out transient noise.
- **Sweep**: Perform an $\alpha$ sweep to find the region where commitment trajectories stabilize.

### 2. Data-Adaptive Absolute δ
- **Definition**: Percentile sweep over $\Delta_{final}$ or max $\Delta_t$ across the dataset.
- **Goal**: Find a $\delta$ that captures the "early commitment" phase of high-confidence samples while remaining reachable for low-confidence ones.

## Expected Outputs
- **Reach-rate curves**: Percentage of samples reaching the boundary as a function of $\alpha$ or $\delta$.
- **$t_d$ distributions**: Visualization of where in the network's depth $(0, 1)$ the commitment usually occurs.
- **Recommended Boundary Selection Band**: A target range (typically 30–80% reach) for choosing stable boundaries for cross-model comparison.

---
**Status**: Scaffolding / Planned
