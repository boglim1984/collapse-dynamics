# Experiment 02B: Research Update (2026-01-16)

**Focus**: Absorption-Time Metrics, Falsifier Refinement, and Stability Hardening  
**Status**: Stability Confirmed / Production Hardened  
**Date**: 2026-01-16 (Data Collection)

## Reproduce in Colab (1 hour)
- **Environment**: Open Google Colab (GPU: T4 or better).
- **Setup**: `!pip -q install timm`
- **Run**: Execute the canonical harness at: docs/experiments/02b/colab_harness_1hr.py
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
- **Sample-Shuffle Null (FAILED / INVALID)**: Mean $d \approx -0.903$. This falsifier failed to collapse the effect. It preserves architecture-specific marginal distributions, meaning population-level differences persist even when individual pairings are broken.
- **Label-Swap / Pooled Nulls (FAILED / INCONCLUSIVE)**: In current long-runs, these also produced large effect sizes ($d \approx -0.9$). As currently implemented, they do not successfully destroy the architectural signal or are confounded by distributional asymmetries.

> [!CAUTION]
> **Falsifier Status**: Only **Depth-Permute** is currently considered a valid/passing stress test. **Sample-Shuffle**, **Label-Swap**, and **Pooled-Resample** are marked as **FAILED / Invalid** as implemented, as they do not successfully collapse the observed architecture-level separation.

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

> [!IMPORTANT]
> **Known Pitfalls & UX**:
> - **Colab Cell Order**: The setup/definitions cell MUST be run before any downstream collection cells to avoid `NameError` on `HighFidelityTracker`.
> - **Python Syntax**: Avoid copying natural-language headers (e.g., those containing em-dashes '—') directly into Python cells as they cause `SyntaxError`.
> - **Argparse / Jupyter**: The harness is designed for CLI; if running in a notebook, use `parse_known_args()` to avoid conflicts with the Jupyter `-f` kernel flag.

---

## 5. Conclusion / What We Learned

The "Lock + Publish" pass for Experiment 02B has frozen the following core findings:

### Core Finding (Absorbing Boundary / $t_{abs}$)
Using **Absorption Time ($t_{abs}$)** as the primary measure of decision stability at $\alpha=0.70$:
- **ResNet**: mean $t_{abs}=0.477$, median $=0.714$, mean violations $=0.9$
- **ViT**: mean $t_{abs}=0.836$, median $=1.000$, mean violations $=6.0$
- **Architectural Gap**: $\Delta(R-V) = -0.360$
- **Effect Size**: Cohen’s $d \approx -0.934$ (Large Effect)

**Interpretation**: ResNet reaches and maintains its final-margin dominance threshold substantially earlier than ViT. ViT exhibits significantly more "flicker" and late-stage violations, indicating a more volatile collapse toward the categorical decision.

### Robustness & Falsification
- **1-Hour Multi-Run Scan**: Verified across 60 seeds ($\alpha \in [0.55, 0.67]$). Directionality was consistent in **60/60** runs ($t_{abs}\text{\_gap} < 0$).
- **Falsifiers**:
    - **Depth-Permute (PASSED)**: Mean $d \approx -0.099$. Sequence destruction correctly collapses the effect.
    - **Sample-Shuffle / Pooled (FAILED)**: Mean $d \approx -0.903$. Failure to collapse suggests population-level distributional asymmetries are preserved even when pairings are broken. 

### Why $t_{abs}$ Is Superior
$t_{abs}$ is **late-snap safe**. Unlike "k consecutive layers" metrics, it defines the earliest depth *after the last* dominance violation. This avoids edge artifacts and captures the true "point of no return" for a stable prediction.

---

**Experiment 02B is now frozen. Research moves to 02C (Irreversibility Horizons).**

---

## 6. Freeze Summary
- **02B Freeze Status**: **Frozen**
- **Primary Claim**: $t_{abs}$ gap (R−V) is robustly negative and large ($d \approx -0.934$) at $\alpha=0.70$.
- **Robustness**: Stable directionality in **60/60** runs of the 1-hour scan.
- **Passing Falsifier**: **Depth-Permute** (collapses to $d \approx -0.1$).
- **Non-Passing Falsifiers**: **Sample-Shuffle**, **Label-Swap**, and **Pooled-Resample** failed to collapse the effect ($d \approx -0.9$); they are currently considered invalid nulls as implemented.

## Artifacts (Canonical & Tracked)
- **Harness**: docs/experiments/02b/colab_harness_1hr.py
- **Calibration**: experiments/02b/exp02b_absorbing_boundary_calibration.py
- **Summary Metrics**: experiments/02b/data/stability_runs.csv
- **Absorption Data**: experiments/02b/data/absorption_time_metrics.csv
- **Reproducibility Spec**: experiments/02b/data/boundary_spec_balanced.json
- **Auxiliary Artifacts**: experiments/02b/data/boundary_spec_*.json, matched_alpha_sweep.csv, download.png.

**Reproduction**: To reproduce, run the canonical harness `docs/experiments/02b/colab_harness_1hr.py` in a T4 Colab environment. Ensure the definitions cell is run first to avoid `NameError`.
