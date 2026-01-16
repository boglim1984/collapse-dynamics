# REPORT: Experiment 02B Update (2026-01-16)

**Topic**: 1-Hour Long-Run Stability + Falsifier Redesign + OOM Hardening  
**Status**: Stability Confirmed / Falsifiers Refined  
**Hardware**: Colab (L4 GPU)  

---

## 1. Executive Summary (What We Learned)

We executed an intensive **1-hour long-run stability battery** in Colab to verify the reproducibility of the Experiment 02B findings across multiple signal conditions.

### 1-Hour Long-Run (Colab) Summary
- **Completed Runs**: 60 (Seeds 123–182)
- **Alpha Selection**: Converged at $\alpha=0.50$ across all 60 completed runs.
- **Directional Consistency**:
  - **$t_{abs}$ Gap (R−V) < 0**: 60/60 runs (Direction-invariant; ResNet consistently earlier).
  - **Violation Gap (R−V) < 0**: 60/60 runs (ResNet consistently exhibits fewer violations).
- **Observed Metrics (Ranges)**:
  - **Common Reach**: $\approx 0.55 – 0.68$
  - **$t_{abs}$ Gap**: $\approx -0.23 – -0.46$
  - **Cohen’s d ($t_{abs}$)**: $\approx -0.60 – -1.22$
  - **Violation Gap**: $\approx -3.30 – -4.80$

The absorption-time metric ($t_{abs}$) and dominance violations show **strong, seed-invariant architectural separation**, reinforcing the "Divergent Collapse" hypothesis.

---

## 2. Falsifier Analysis: Success and Failures

### Depth-Permutation Falsifier (WORKING)
- **Result**: Mean $d \approx -0.099$.
- **Interpretation**: Permuting layer order for each sample correctly collapses the architectural separation toward zero, validating that the effect is depth-dependent.

### Sample-Shuffle Falsifier (FAILED / MIS-SPECIFIED)
- **Result**: Mean $d \approx -0.903$.
- **Interpretation**: Shuffling the pairing between ResNet and ViT samples failed to collapse the separation. This indicates that the separation is dominated by the **marginal distributions** of the model families rather than the specific sample-level alignment. Measuring a group-level difference via a pairing-shuffle null is a mis-specified test.

> [!CAUTION]
> The **Sample-Shuffle** falsifier is currently marked as **Inconclusive / Needs Redesign**. It does NOT invalidate the main claim, but it fails to provide a useful null baseline for group-level separation.

---

## 3. Falsifier Redesign: The New Battery

To resolve the mis-specification of the sample-shuffle test, we have implemented a redesigned battery:

1. **Pooled-Resample Null (Primary)**:
   - Pool all trajectories from both architectures.
   - Randomly split the pool into two groups of size $N$.
   - **Expected**: Separation ($d$) collapses toward 0.
2. **Random Label-Swap Null**:
   - For each pair $(ResNet_i, ViT_i)$, swap the architecture labels with $p=0.5$.
   - **Expected**: Separation collapses toward 0.
3. **Final-Anchor Scramble**:
   - Permute the final-layer margin across samples before computing dominance thresholds.
   - **Expected**: Significant reduction in effect size if the "final anchor" is essential to the relative-dominance definition.

---

## 4. Colab Stability Harness (OOM Hardening)

During the long-run, the process hit a **CUDA OOM** crash after 60 runs. The following mitigations have been implemented and documented for future runs:

- **Allocator Config**: `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`.
- **Batch Optimization**: Batch sizes for probe and trajectory collection reduced to **32/16**.
- **Memory Cleanup**: Explicit teardown per-run using `del`, `gc.collect()`, and `torch.cuda.empty_cache()`.
- **Precision**: Instrumented with `torch.cuda.amp.autocast` for trajectory forward passes to reduce peak VRAM usage.

---

## 5. Falsifiable Claim (Current Handoff)

**The Core Claim**: Under a matched-reach or fixed-$\alpha$ dominance boundary, ResNet exhibits earlier absorption (lower $t_{abs}$) and fewer dominance violations than ViT, consistently across seeds.

**Falsification Conditions**:
- The claim is falsified if the **Pooled-Resample** or **Label-Swap** null maintains the same separation magnitude as the original data.

---

## Reproduction and TODO
- **Notebook**: Patch `experiments/exp02b_absorbing_boundary_calibration.py` with the "1-Hour Harness" cell.
- **Artifacts**: New stability findings recorded in `experiments/02b/data/runs_summary.csv` (simulated placeholder) and `stability_runs.csv`.

**TODO**:
- [ ] Implement within-layer sample shuffling (destroy trajectory identity).
- [ ] Swap architecture-specific normalization anchors in `boundary_spec`.
