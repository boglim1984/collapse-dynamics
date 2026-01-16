# Experiment 02C: Irreversibility Horizon (Floating Threshold)

## Purpose
This experiment investigated the "Irreversibility Horizon" using a floating threshold approach. The results showed that the escape rate under this methodology largely washed out compared to the null baseline (escape $\approx$ null), which motivated the subsequent redesign into the fixed-threshold approach used in Experiment 02D.

## Colab Notebook
[Run on Colab](https://colab.research.google.com/drive/1dQ-A_j2YS4ZZFika_qQ442nVprD2vnGD?usp=sharing)

## Summary Metrics
- **Selected Alpha**: 0.89
- **Clean Stability**:
  - **ResNet**: $t_{abs}$ 0.555, violations 1.06
  - **ViT**: $t_{abs}$ 0.857, violations 6.66
- **Escape @ max sigma**:
  - **ResNet**: 0.296 (Null: 0.327)
  - **ViT**: 0.306 (Null: 0.320)

## Data Artifacts
The following artifacts are available in the [data/](data/) directory:
- `absorption_metrics.csv`: Per-sample absorption and violation data.
- `alpha_sweep.csv`: Metrics across the alpha grid.
- `boundary_spec.json`: Full reproducibility parameters.
- `irreversibility_curve.csv`: Data for the irreversibility horizon plot.
- `runs_summary.csv`: Aggregate statistics for the experiment.
