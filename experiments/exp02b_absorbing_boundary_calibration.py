"""
01/15/2026 6:25 PM — Collapse Dynamics / Exp02B Addendum

RUN MANIFEST:
- Purpose: Calibrate reachable AND balanced absorbing boundaries.
- Selection Rule: Maximize common_reach subject to reach_gap <= DELTA_GAP.
- Outputs: 
    - matched_alpha_sweep_balanced.csv
    - boundary_spec_balanced.json
    - pareto_plot.png
    - alpha_tradeoff_plot.png
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models
import collections
import os
import json

# Import tracker logic
try:
    from exp02_lyapunov_stability_scan_final_patched import HighFidelityTracker, LinearProbe, get_representation, train_distilled_probe
except ImportError:
    # Fallback for standalone test or path issues
    from experiments.exp02_lyapunov_stability_scan_final_patched import HighFidelityTracker, LinearProbe, get_representation, train_distilled_probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_TRAIN_SAMPLES = 500
PROBE_EPOCHS = 3
SEED = 42
REQ_CONSECUTIVE = 3 # Persistence window + 1

def calculate_cohens_d(td1, td2):
    """Compute Cohen's d for two lists of t_d values."""
    td1 = [t for t in td1 if t is not None]
    td2 = [t for t in td2 if t is not None]
    if len(td1) < 2 or len(td2) < 2:
        return 0.0
    u1, u2 = np.mean(td1), np.mean(td2)
    s1, s2 = np.var(td1, ddof=1), np.var(td2, ddof=1)
    n1, n2 = len(td1), len(td2)
    pooled_std = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (u1 - u2) / pooled_std

def collect_trajectories(model, mtype, n_samples=100):
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
    
    trajectories = [] 
    
    for h in tracker.hooks:
        probe = train_distilled_probe(train_feats[h].detach(), final_logits_train.detach())
        probe.eval()
        with torch.no_grad():
            logits = probe(test_feats[h])
            vals, _ = torch.topk(logits, 2, dim=1)
            margin = vals[:, 0] - vals[:, 1]
            trajectories.append(margin.cpu().numpy())
            
    tracker.remove_hooks()
    return np.stack(trajectories, axis=1)

def perform_balanced_selection(resnet_traj, vit_traj, alphas, output_dir):
    n_samples, n_layers = resnet_traj.shape
    results = []

    for alpha in alphas:
        # ResNet
        f_resnet = resnet_traj[:, -1][:, None]
        r_resnet = (resnet_traj >= alpha * f_resnet)
        td_resnet = [np.where(r)[0][0] / (n_layers-1) if np.any(r) else None for r in r_resnet]
        reach_resnet = sum(t is not None for t in td_resnet) / n_samples
        
        # ViT
        f_vit = vit_traj[:, -1][:, None]
        r_vit = (vit_traj >= alpha * f_vit)
        td_vit = [np.where(r)[0][0] / (n_layers-1) if np.any(r) else None for r in r_vit]
        reach_vit = sum(t is not None for t in td_vit) / n_samples
        
        common_reach = min(reach_resnet, reach_vit)
        reach_gap = abs(reach_resnet - reach_vit)
        
        d = calculate_cohens_d(td_resnet, td_vit)
        
        results.append({
            'alpha': alpha,
            'reach_resnet': reach_resnet,
            'reach_vit': reach_vit,
            'common_reach': common_reach,
            'reach_gap': reach_gap,
            'mean_td_resnet': np.mean([t for t in td_resnet if t is not None]) if reach_resnet > 0 else 0,
            'mean_td_vit': np.mean([t for t in td_vit if t is not None]) if reach_vit > 0 else 0,
            'cohens_d': d
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "matched_alpha_sweep_balanced.csv"), index=False)
    
    # Selection logic
    best_spec = None
    selected_gap = None
    for gap_limit in [0.10, 0.15, 0.20, 0.30, 0.50]:
        candidates = df[df['reach_gap'] <= gap_limit]
        if not candidates.empty:
            # Maximize common_reach, then maximize cohens_d
            best_idx = candidates.sort_values(by=['common_reach', 'cohens_d'], ascending=False).index[0]
            best_spec = df.loc[best_idx].to_dict()
            selected_gap = gap_limit
            break
            
    if best_spec:
        spec_dict = {
            "criterion_type": "Relative",
            "alpha_selected": best_spec['alpha'],
            "persistence_window": 2,
            "req_consecutive": 3,
            "reach_gap_constraint_used": selected_gap,
            "reach_resnet": best_spec['reach_resnet'],
            "reach_vit": best_spec['reach_vit'],
            "reach_gap": best_spec['reach_gap'],
            "common_reach": best_spec['common_reach'],
            "mean_td_resnet": best_spec['mean_td_resnet'],
            "mean_td_vit": best_spec['mean_td_vit'],
            "td_gap": best_spec['mean_td_resnet'] - best_spec['mean_td_vit'],
            "cohens_d": best_spec['cohens_d']
        }
        with open(os.path.join(output_dir, "boundary_spec_balanced.json"), "w") as f:
            json.dump(spec_dict, f, indent=4)
        
        print(f"BALANCED-REACH SELECTED alpha={spec_dict['alpha_selected']:.3f} | common_reach={spec_dict['common_reach']:.3f} | gap={spec_dict['reach_gap']:.3f} | d={spec_dict['cohens_d']:.3f} | used_gap_constraint={spec_dict['reach_gap_constraint_used']}")

    return df

def plot_pareto_and_tradeoffs(df, output_dir):
    # 1. Pareto Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['reach_gap'], df['common_reach'], c=df['cohens_d'], cmap='viridis', s=100)
    plt.colorbar(label="Cohen's d")
    
    for i, txt in enumerate(df['alpha']):
        if i % 2 == 0: # Only label every other for clarity
            plt.annotate(f"α={txt:.2f}", (df['reach_gap'][i], df['common_reach'][i]), fontsize=8)
            
    plt.xlabel("Reach Gap (|ResNet - ViT|)")
    plt.ylabel("Common Reach (Min)")
    plt.title(r"Pareto Front: Common Reach vs. Asymmetry")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "pareto_plot.png"))
    
    # 2. Alpha Tradeoff Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['alpha'], df['reach_resnet'], label='ResNet Reach', linestyle='--')
    ax1.plot(df['alpha'], df['reach_vit'], label='ViT Reach', linestyle='--')
    ax1.plot(df['alpha'], df['common_reach'], label='Common Reach', color='black', linewidth=2)
    ax1.set_xlabel("Alpha (Relative Threshold)")
    ax1.set_ylabel("Reach Rate")
    ax1.legend(loc='lower left')
    
    ax2 = ax1.twinx()
    ax2.plot(df['alpha'], df['cohens_d'], color='red', label="Cohen's d", alpha=0.5)
    ax2.set_ylabel("Effect Size (Cohen's d)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(r"Exp02B: Alpha vs. Reach & Effect Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "alpha_tradeoff_plot.png"))
    plt.close()

def update_02b_research():
    output_dir = "docs/experiments/exp02b_artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device).eval()
    vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
    
    print("Collecting ResNet trajectories...")
    resnet_traj = collect_trajectories(resnet, 'resnet')
    print("Collecting ViT trajectories...")
    vit_traj = collect_trajectories(vit, 'vit')
    
    alphas = np.linspace(0.1, 0.99, 20)
    
    print("Performing Balanced Selection Analysis...")
    df = perform_balanced_selection(resnet_traj, vit_traj, alphas, output_dir)
    
    print("Generating Pareto Diagnostics...")
    plot_pareto_and_tradeoffs(df, output_dir)
    
    print("Exp02B Addendum update complete.")

if __name__ == "__main__":
    update_02b_research()
