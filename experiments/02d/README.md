# Experiment 02D: Fixed-Threshold Basin Wall + Time-To-Escape (TTE)

## Purpose
This experiment introduced a fixed-threshold approach to measure the "basin wall" and "Time-To-Escape" (TTE). This methodology provides a clearer irreversibility signal dimension than the floating threshold used in 02C. The fixed wall reveals differential fragility between architectures.

## Colab Notebook
[Run on Colab](https://colab.research.google.com/drive/1SnfvqTtAytWC9hIn4r9e7oq4ikie06Z7?usp=sharing)

## Summary Metrics
- **Selected Alpha**: 0.890
- **Clean Stability**:
  - **ResNet**: $t_{abs}$ 0.555, violations 1.06
  - **ViT**: $t_{abs}$ 0.857, violations 6.66
- **Escape @ max sigma**:
  - **ResNet**: 0.667 (Null: 0.637) | **TTE**: 1.71
  - **ViT**: 0.590 (Null: 0.604) | **TTE**: 0.67

## Interpretation
The fixed wall reveals differential fragility; ViT escapes sooner when it escapes (lower TTE), even if the absolute escape rate is not significantly larger than the null. This indicates that once the boundary is breached, the ViT decision state collapses more rapidly or with less resistance.

## Data Artifacts
The following artifacts are available in the [data/](data/) directory:
- `absorption_metrics.csv`: Per-sample absorption and violation data.
- `alpha_sweep.csv`: Metrics across the alpha grid.
- `boundary_spec.json`: Full reproducibility parameters.
- `irreversibility_curve.csv`: Data for the irreversibility horizon plot.
- `runs_summary.csv`: Aggregate statistics for the experiment.
