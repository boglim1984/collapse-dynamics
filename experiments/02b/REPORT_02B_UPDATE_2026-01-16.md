# Experiment 02B: Research Update (2026-01-16)

**Focus**: Absorption-Time Metrics, Falsifier Refinement, and Stability Hardening  
**Status**: Stability Confirmed / Production Hardened  
**Date**: 2026-01-16 (Data Collection)

## Reproduce in Colab (1 hour)
- **Environment**: Open Google Colab (GPU: T4 or better).
- **Setup**: `!pip -q install timm`
- **Run**: Execute the canonical harness at: [docs/experiments/02b/colab_harness_1hr.py](file:///Users/oflahertys/Documents/Dev/Experiments/collapse-dynamics/docs/experiments/02b/colab_harness_1hr.py)
- **Outputs**: Written to `exp02b_*` timestamped folders:
  - `runs_summary.csv`: Aggregate statistics over seeds.
  - `boundary_spec_run_###.json`: Full reproducibility parameters.
  - `absorption_metrics_run_###.csv`: Per-sample $t_{abs}$ and violation counts.

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

### Falsifiers & Interpretation
The falsifier battery was expanded to test the robustness of the architectural separation:

- **Depth-Permute Null (VALID)**: Mean $d \approx -0.099$. Destroying the layer-order sequence collapses the effect toward zero, confirming that the directional gap depends on the sequential evolution of features.
- **Label-Swap / Pooled Nulls (VALID)**: These falsifiers operate on scalar $t_{abs}$ outcomes. By destroying architectural identity, they collapse the effect toward zero.
- **Sample-Shuffle Null (INVALID/WEAK)**: Mean $d \approx -0.903$. Markedly failed to collapse the effect.

> [!CAUTION]
> **Sample-Shuffle** is currently marked as an **invalid falsifier**. It preserves the architecture-specific marginal distributions (population properties), meaning population-level differences persist even when individual pairings are broken. Only **Depth-Permute**, **Label-Swap**, and **Pooled-Resample** are considered valid stress tests for this claim.

---

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

## 5. Summary & Conclusions
- **What Changed**: We transitioned from first-dominance ($t_d$) to **Absorption Time ($t_{abs}$)** as the primary stability metric to account for late-stage "flicker" and absorbing boundaries.
- **Key Result**: Directionality is robustly consistent across 60 seeds. ResNet consistently reaches stable categorical absorption earlier and with fewer violations than ViT under balanced reach ($c \approx 0.60$).
- **Effect Sizes**: Large architectural separation ($d \approx -0.9$) persists regardless of seed or sample pairing (unless architectural identity is destroyed).
- **Limitations**: GPU OOM risk is present; the harness now includes explicit mitigations.
- **Next Step**: The current dataset is sufficient to claim architectural separation in categorical stability. Future work will focus on Experiment 03 (irreversibility horizons).

## Artifacts (2026-01-16)
- **Canonical Summary**: `experiments/02b/data/stability_runs.csv`
- **Absorption Metrics**: `experiments/02b/data/absorption_time_metrics.csv`
- **Boundary Spec**: `experiments/02b/data/boundary_spec_balanced.json`
- **Canonical Harness**: `docs/experiments/02b/colab_harness_1hr.py`
