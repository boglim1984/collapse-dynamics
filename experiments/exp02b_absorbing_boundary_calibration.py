"""
RUN MANIFEST:
- Same Tracker+Probes as Experiment 02.
- Purpose: Calibrate reachable absorbing boundaries.
- Sweep Parameters:
  * Alphas: [0.5, 0.7, 0.8, 0.9, 0.95] (Relative: Δ_t >= alpha * Δ_final)
  * Percentiles: [25, 50, 75, 90] (Adaptive absolute: Δ_t >= percentile(Δ_final))
- Metrics: reach_rate, mean t_d, std t_d.
- Outputs: results/exp02b/calibration_results.csv, plots.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models
import collections
import os

# Import tracker logic (simplified for standalone use)
from exp02_lyapunov_stability_scan_final_patched import HighFidelityTracker, LinearProbe, get_representation, train_distilled_probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_TRAIN_SAMPLES = 500
PROBE_EPOCHS = 3 # Faster for calibration
SEED = 42

def collect_trajectories(model, mtype, n_samples=50):
    tracker = HighFidelityTracker(model, mtype)
    tracker.register_hooks()
    
    x_train = torch.randn(PROBE_TRAIN_SAMPLES, 3, 224, 224).to(device)
    x_test = torch.randn(n_samples, 3, 224, 224).to(device)
    
    with torch.no_grad():
        final_logits_train = model(x_train)
        final_logits_test = model(x_test)
        
    tracker.clear_features()
    _ = model(x_train)
    train_feats = {k: get_representation(v, mtype, model) for k, v in tracker.features.items()}
    
    tracker.clear_features()
    _ = model(x_test)
    test_feats = {k: get_representation(v, mtype, model) for k, v in tracker.features.items()}
    
    trajectories = [] # (n_samples, n_layers)
    
    for h in tracker.hooks:
        probe = train_distilled_probe(train_feats[h].detach(), final_logits_train.detach())
        probe.eval()
        with torch.no_grad():
            logits = probe(test_feats[h])
            vals, _ = torch.topk(logits, 2, dim=1)
            margin = vals[:, 0] - vals[:, 1]
            trajectories.append(margin.cpu().numpy())
            
    tracker.remove_hooks()
    return np.stack(trajectories, axis=1) # (samples, layers)

def analyze_calibration():
    os.makedirs("results/exp02b", exist_ok=True)
    
    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device).eval()
    vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
    
    configs = [
        (resnet, 'resnet', 'ResNet18'),
        (vit, 'vit', 'ViT-B/16')
    ]
    
    alphas = [0.5, 0.7, 0.8, 0.9, 0.95]
    percentiles = [25, 50, 75, 90]
    
    all_stats = []
    
    for model, mtype, mname in configs:
        print(f"Collecting trajectories for {mname}...")
        traj = collect_trajectories(model, mtype)
        n_samples, n_layers = traj.shape
        final_margins = traj[:, -1]
        
        # 1. Alpha Sweeps
        for alpha in alphas:
            target = alpha * final_margins[:, None]
            reached = (traj >= target)
            reach_indices = [np.where(r)[0][0] / (n_layers-1) if np.any(r) else None for r in reached]
            valid_td = [t for t in reach_indices if t is not None]
            
            reach_rate = len(valid_td) / n_samples
            mean_td = np.mean(valid_td) if valid_td else np.nan
            std_td = np.std(valid_td) if valid_td else np.nan
            
            all_stats.append({
                'model': mname, 'criterion': f'alpha_{alpha}', 
                'reach_rate': reach_rate, 'mean_td': mean_td, 'std_td': std_td
            })

        # 2. Percentile Sweeps
        for p in percentiles:
            threshold = np.percentile(final_margins, p)
            reached = (traj >= threshold)
            reach_indices = [np.where(r)[0][0] / (n_layers-1) if np.any(r) else None for r in reached]
            valid_td = [t for t in reach_indices if t is not None]
            
            reach_rate = len(valid_td) / n_samples
            mean_td = np.mean(valid_td) if valid_td else np.nan
            std_td = np.std(valid_td) if valid_td else np.nan
            
            all_stats.append({
                'model': mname, 'criterion': f'p_{p}', 
                'reach_rate': reach_rate, 'mean_td': mean_td, 'std_td': std_td
            })

    df = pd.DataFrame(all_stats)
    df.to_csv("results/exp02b/calibration_results.csv", index=False)
    print("Calibration results saved to results/exp02b/calibration_results.csv")
    
    # Recommendation Plot
    plt.figure(figsize=(10, 6))
    for mname in df['model'].unique():
        m_df = df[df['model'] == mname]
        plt.plot(m_df['criterion'], m_df['reach_rate'], marker='o', label=f"{mname} Reach")
    
    plt.axhspan(0.3, 0.8, color='green', alpha=0.1, label='Target Reach Band')
    plt.title("Reach-rate curves for Boundary Calibration")
    plt.xticks(rotation=45)
    plt.ylabel("Reach Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/exp02b/reach_curves.png")
    print("Reach curve plot saved to results/exp02b/reach_curves.png")

if __name__ == "__main__":
    analyze_calibration()
