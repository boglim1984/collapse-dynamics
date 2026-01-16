# Experiment 02B: Research Update (2026-01-16)

**Focus**: Absorption-Time Metrics, Falsifier Refinement, and Stability Hardening  
**Status**: Stability Confirmed / Harness OOM-Hardened  
**Date**: 2026-01-16 (Data Collection)

---

## 1. New Metric: Absorption-Time ($t_{abs}$)

We have introduced a second dominance-stability metric to supplement the first-dominance depth ($t_d$). **Absorption Time** ($t_{abs}$) measures the earliest layer index after which no further violations of the dominance criterion occur.

### Definition
- **Dominance Mask**: $\text{dominance}[t] = (\Delta_t \ge \alpha \cdot \text{max}(\Delta_{final}, \epsilon))$
- **Metric**: 
  - $t_{abs} = 0$ if the model never violates dominance.
  - $t_{abs} = \frac{1 + \text{max}\{t \mid \text{dominance}[t] == \text{False}\}}{L-1}$
- **Intuition**: This acts as an "absorbing boundary" measure. It is safe against late-stage snapping (jitters) and provides a more conservative estimate of true decision stability.

### Sanity Check (Alpha=0.7, Paired N=500)
- **ResNet**: Mean $t_{abs} = 0.477$ | Mean Violations $\approx 0.9$
- **ViT**: Mean $t_{abs} = 0.836$ | Mean Violations $\approx 6.0$
- **Gap (R-V)**: -0.360
- **Effect Size (Cohen’s d)**: -0.934

This confirms that ViT not only commits later but also exhibits significantly more "flicker" before reaching a stable categorical state.

---

## 2. 1-Hour Long-Run Stability Battery

We executed a stability harness for ~1 hour on Colab to verify consistent directionality.

- **Runs Completed**: 60 seeds (Seed 123–182).
- **Control**: Fixed $\alpha=0.5$ (matched mode convergence).
- **Directional Consistency**: 
  - $t_{abs}\text{\_gap} < 0$ in **60/60** runs.
  - $\text{viol\_gap} < 0$ in **60/60** runs.
- **Performance**: Common reach stabilized between $0.55 – 0.67$.
- **Advisory**: Run terminated by CUDA OOM after run 60, prompting the OOM hardening patch in the stability harness.

---

## 3. Falsifier Status & Redesign

The falsifier battery was expanded to test the robustness of the observed architecture-level separation.

### The Falsifier Results
- **Depth-Permute Null (VALID)**: Mean $d \approx -0.099$. As expected, destroying the layer-order sequence collapses the effect toward zero.
- **Sample-Shuffle Null (INVALID/WEAK)**: Mean $d \approx -0.903$. **CRITICAL FINDING**: Shuffling ViT samples against ResNet samples did *not* collapse the effect. 

> [!WARNING]
> The **Sample-Shuffle** test is officially marked as a **flawed falsifier**. Because it preserves the architecture-specific marginal distributions (population-level properties), simply breaking the pairing between individual samples does not destroy the architectural effect. 

### Redesigned Falsifiers
To provide a more rigorous null baseline, we have replaced sample-shuffling with:
1. **Label-Swap Null**: Randomly swap ResNet and ViT trajectories within each paired sample ($p=0.5$). This should collapse both sign and magnitude toward 0.
2. **Pooled-Resample Null**: Build a pooled set of all trajectories and randomly assign them into two new groups of size $N$. This destroys the architectural identity of the samples.

> [!NOTE]
> **Implementation Update**: 
> - **Fixed probe-training autograd**: Removed global grad disable; probe updates now run under `enable_grad` for successful distillation.
> - **Shape Mismatch Fix**: `HighFidelityTracker` now freezes `layer_order` at initialization. This prevents shape mismatch errors during trajectory collection (ResNet $L=8$ vs ViT $L=12$).
> - **Cross-depth Falsifiers**: Falsifiers (`labelswap` and `pooled`) operate on scalar $t_{abs}$ outcomes, ensuring compatibility across different model architectures.
> - **Expanded Spec**: Boundary spec JSON now includes the full alpha grid, balanced-reach parameters, and artifact pointers for enhanced reproducibility.

---

## 4. Stability Harness (OOM Hardening Patch)

To prevent the CUDA OOM crashes observed in long-runs, the `exp02b_absorbing_boundary_calibration.py` harness now includes:
- **Allocator Config**: `expandable_segments:True` to reduce fragmentation.
- **Optimized Batching**: Default batch sizes reduced to **32**.
- **Autocast**: Mixed-precision (fp16) logic for trajectory projections.
- **Teardown**: Explicit per-run cleanup (`del`, `gc.collect()`, `empty_cache`).

---

## Artifacts (2026-01-16)
- **New Summary**: `experiments/02b/data/runs_summary.csv` (1hr stability results)
- **Revised Metrics**: `experiments/02b/data/absorption_time_metrics.csv`
- **Harness**: `experiments/02b/exp02b_absorbing_boundary_calibration.py`
