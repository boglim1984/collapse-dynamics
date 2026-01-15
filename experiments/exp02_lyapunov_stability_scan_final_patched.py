"""
RUN MANIFEST:
- Models: ResNet18 (torchvision), ViT-B/16 (torchvision)
- Transforms: ImageNet-compatible (resize 224, normalize)
- Seed: 42
- EPS: 1e-8
- Persistence Window: N/A (Late-stage verdict uses peak location)
- Probe Params: dim->1000, MSELoss, Adam(lr=1e-3), self-distilled from final logits
- N_SAMPLES: PROBE_TRAIN_SAMPLES=500, PROBE_EPOCHS=5
- Channel Definitions:
    (1) d_rel: ||dh|| / (||h|| + eps)
    (2) s_t: |dMargin| / ||dx_input|| (True sensitivity)
    (3) lambda_state: log(d_{t+1} / d_t)
    (4) lambda_commit: log(s_{t+1} / s_t) [clamped floor]
    (5) cum_log_gain: sum(lambda_commit)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils_data import DataLoader, Subset
import collections
import os

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_TRAIN_SAMPLES = 500
PROBE_EPOCHS = 5
EPS = 1e-8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class HighFidelityTracker:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.hooks = []
        self.features = collections.OrderedDict()
        self.hook_handles = []

    def _get_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output
        return hook_fn

    def register_hooks(self):
        if self.model_type == 'resnet':
            # ResNet: hook each BasicBlock layerX.Y
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                layer = getattr(self.model, layer_name)
                for sub_name, block in layer.named_children():
                    name = f"{layer_name}.{sub_name}"
                    self.hook_handles.append(block.register_forward_hook(self._get_hook(name)))
                    self.hooks.append(name)
        elif self.model_type == 'vit':
            # ViT: hook each blocks.X
            for i, block in enumerate(self.model.encoder.layers):
                name = f"blocks.{i}"
                self.hook_handles.append(block.register_forward_hook(self._get_hook(name)))
                self.hooks.append(name)

    def clear_features(self):
        self.features.clear()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

def get_representation(feat, model_type, model=None):
    if model_type == 'vit':
        # Apply model.norm (encoder.ln) to token sequence before CLS extraction
        # Note: In a real run we'd pass the actual norm module
        if model and hasattr(model.encoder, 'ln'):
            feat = model.encoder.ln(feat)
        if len(feat.shape) == 3:
            return feat[:, 0] # CLS token
    return feat.view(feat.size(0), -1)

class LinearProbe(nn.Module):
    def __init__(self, input_dim, out_dim=1000):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        
    def forward(self, x):
        return self.linear(x)

def train_distilled_probe(features, target_logits):
    input_dim = features.size(1)
    probe = LinearProbe(input_dim, out_dim=target_logits.size(1)).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    probe.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        outputs = probe(features)
        loss = criterion(outputs, target_logits)
        loss.backward()
        optimizer.step()
    return probe

def run_experiment_02():
    print(f"Running Experiment 02 on {device}...")
    
    # 1. Load Models
    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device).eval()
    vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
    
    models_to_test = [
        (resnet, 'resnet', 'ResNet18'),
        (vit, 'vit', 'ViT-B/16')
    ]
    
    # 2. Mock Data for script fidelity
    x_input = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
    x_train = torch.randn(PROBE_TRAIN_SAMPLES, 3, 224, 224).to(device)
    
    all_results = {}

    for model, mtype, mname in models_to_test:
        print(f" Scanning {mname}...")
        tracker = HighFidelityTracker(model, mtype)
        tracker.register_hooks()
        
        # Get final logits for distillation
        with torch.no_grad():
            final_logits_train = model(x_train)
            final_logits_input = model(x_input)
        
        tracker.clear_features()
        _ = model(x_train)
        layer_features_train = {k: get_representation(v, mtype, model) for k, v in tracker.features.items()}
        
        tracker.clear_features()
        _ = model(x_input)
        layer_features_input = {k: get_representation(v, mtype, model) for k, v in tracker.features.items()}
        
        probes = {}
        for h in tracker.hooks:
            probes[h] = train_distilled_probe(layer_features_train[h].detach(), final_logits_train.detach())
            
        # Scan Metrics
        results = collections.defaultdict(list)
        prev_h = None
        prev_s = None
        prev_d_rel = None

        for h in tracker.hooks:
            h_t = layer_features_input[h]
            logits = probes[h](h_t)
            
            # Margin calculation
            vals, _ = torch.topk(logits, 2, dim=1)
            margin = vals[:, 0] - vals[:, 1]
            
            # s_t = |dMargin| / ||dx_input||
            model.zero_grad()
            if x_input.grad is not None: x_input.grad.zero_grad()
            margin.backward(retain_graph=True)
            s_t = x_input.grad.norm(p=2).item() / (x_input.norm(p=2).item() + EPS)
            
            # d_rel = ||dh||/(||h||+eps)
            if prev_h is not None:
                d_rel = (h_t - prev_h).norm(p=2).item() / (prev_h.norm(p=2).item() + EPS)
            else:
                d_rel = 0.0
            
            # lambda_state = log(d_{t+1}/d_t)
            if prev_d_rel is not None:
                l_state = np.log((d_rel + EPS) / (prev_d_rel + EPS))
            else:
                l_state = 0.0
                
            # lambda_commit = log(s_{t+1}/s_t) [clamped floor]
            if prev_s is not None:
                l_commit = np.log((s_t + EPS) / (prev_s + EPS))
                l_commit = max(l_commit, -10.0) # floor clamp
            else:
                l_commit = 0.0
                
            results['h_names'].append(h)
            results['s_t'].append(s_t)
            results['d_rel'].append(d_rel)
            results['l_state'].append(l_state)
            results['l_commit'].append(l_commit)
            
            prev_h = h_t.detach()
            prev_s = s_t
            prev_d_rel = d_rel

        results['cum_log_gain'] = np.cumsum(results['l_commit'])
        all_results[mname] = results
        tracker.remove_hooks()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for mname, res in all_results.items():
        x_axis = np.linspace(0, 1, len(res['s_t']))
        
        axes[0,0].plot(x_axis, res['l_state'], label=mname)
        axes[0,1].plot(x_axis, res['l_commit'], label=mname)
        axes[1,0].plot(x_axis, res['cum_log_gain'], label=mname)
        axes[1,1].semilogy(x_axis, res['s_t'], label=mname)

    axes[0,0].set_title(r"$\lambda_{state} = \log(d_{t+1}/d_t)$")
    axes[0,1].set_title(r"$\lambda_{commit} = \log(s_{t+1}/s_t)$")
    axes[1,0].set_title(r"Cumulative Log-Gain")
    axes[1,1].set_title(r"Raw Sensitivity $s_t$ (log-scale)")
    
    for ax in axes.flat:
        ax.set_xlabel("Normalized Depth")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig("results/exp02/download-1.png")
    print("Experiment 02 complete. Plot saved to results/exp02/download-1.png")

if __name__ == "__main__":
    run_experiment_02()
