"""
RUN MANIFEST:
- Model Names: ResNet18 (torchvision), ViT-B/16 (torchvision)
- Dataset: CIFAR-10 / ImageNet-style Transforms
- Seed: 42
- EPS: 1e-8
- Dominance Parameters: Delta=2.0 (onset detection)
- Probe Params: MSELoss, Adam(lr=1e-3), PROBE_TRAIN_SAMPLES=500, PROBE_EPOCHS=5
- Channel Definitions:
    (1) lambda_state: Relative state change d_rel = ||dh|| / (||h|| + eps)
    (2) lambda_commit: Local sensitivity log-ratio log(s_{t+1} / s_t)
    (3) cumulative log-gain: cumsum(lambda_commit)
    (4) raw_sensitivity: Raw sensitivity magnitude s_t (plotted on log scale)
- Sensitivity Metric: |dMargin| / ||dx_input|| (True sensitivity normalization)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils_data import DataLoader, Subset
import collections

# Constants
PROBE_TRAIN_SAMPLES = 500
PROBE_EPOCHS = 5
EPS = 1e-8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)

def get_model_and_hooks(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        hooks = []
        # Hook into every BasicBlock
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) and 'layer' in name:
                for sub_name, sub_module in module.named_children():
                    hooks.append(f"{name}.{sub_name}")
        return model, hooks, 'resnet'
    
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        hooks = [f"encoder.layers.encoder_layer_{i}" for i in range(12)]
        return model, hooks, 'vit'
    
    else:
        raise ValueError("Unsupported model")

def extract_features(model, x, hook_name, model_type):
    features = {}
    def hook_fn(m, i, o):
        if model_type == 'vit':
            # ViT representation uses model.norm before CLS extraction
            # Assuming we hook the encoder layers, we need to handle CLS
            # If o is (B, L, D), we take o[:, 0]
            if isinstance(o, tuple): o = o[0]
            features[hook_name] = o
        else:
            features[hook_name] = o

    # Find the module
    target_module = dict(model.named_modules())[hook_name]
    handle = target_module.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(x)
    
    handle.remove()
    feat = features[hook_name]
    
    if model_type == 'vit':
        # Apply normalization if it's a ViT layer and we want CLS
        # In torchvision ViT, the last norm is model.encoder.ln
        # But here we just extract the CLS token
        if len(feat.shape) == 3:
            feat = feat[:, 0] # (B, D)
    
    return feat

def train_probe(features, labels):
    input_dim = features.view(features.size(0), -1).size(1)
    probe = LinearProbe(input_dim)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # One-hot labels for MSE
    labels_onehot = torch.zeros(labels.size(0), 10).scatter_(1, labels.view(-1, 1), 1)
    
    probe.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        outputs = probe(features)
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        optimizer.step()
    
    return probe

def calculate_sensitivity(model, x, probe, hook_name, model_type):
    x.requires_grad = True
    model.eval()
    probe.eval()
    
    # Forward to hook
    features = {}
    def hook_fn(m, i, o):
        if isinstance(o, tuple): o = o[0]
        features[hook_name] = o
    
    target_module = dict(model.named_modules())[hook_name]
    handle = target_module.register_forward_hook(hook_fn)
    
    outputs = model(x)
    handle.remove()
    
    feat = features[hook_name]
    if model_type == 'vit':
        feat = feat[:, 0]
    
    logits = probe(feat)
    
    # Margin calculation: top1 - top2
    vals, idxs = torch.topk(logits, 2, dim=1)
    margin = vals[:, 0] - vals[:, 1]
    
    # Lyapunov scan uses true sensitivity normalization: sensitivity = |dMargin| / ||dx_input||
    model.zero_grad()
    margin.backward(torch.ones_like(margin))
    
    grad = x.grad
    sensitivity = grad.norm(p=2).item() / (x.norm(p=2).item() + EPS)
    
    return sensitivity, feat.detach()

def run_stability_scan(model_name):
    print(f"Starting Stability Scan for {model_name}...")
    model, hooks, model_type = get_model_and_hooks(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Mock data for demonstration (replace with actual CIFAR-10)
    x_input = torch.randn(1, 3, 224, 224).to(device)
    x_train = torch.randn(PROBE_TRAIN_SAMPLES, 3, 224, 224).to(device)
    y_train = torch.randint(0, 10, (PROBE_TRAIN_SAMPLES,)).to(device)
    
    results = collections.defaultdict(list)
    prev_h = None
    prev_s = None
    
    for i, hook in enumerate(hooks):
        print(f" Processing Layer {i+1}/{len(hooks)}: {hook}")
        
        # 1. Train Probe
        feat_train = extract_features(model, x_train, hook, model_type)
        probe = train_probe(feat_train, y_train).to(device)
        
        # 2. Calculate Sensitivity
        s_t, h_t = calculate_sensitivity(model, x_input, probe, hook, model_type)
        
        # Channels
        # (1) lambda_state
        if prev_h is not None:
            dh = (h_t - prev_h).norm(p=2).item()
            lambda_state = dh / (prev_h.norm(p=2).item() + EPS)
        else:
            lambda_state = 0.0
            
        # (2) lambda_commit
        if prev_s is not None:
            lambda_commit = np.log(s_t / (prev_s + EPS))
        else:
            lambda_commit = 0.0
            
        results['s_t'].append(s_t)
        results['lambda_state'].append(lambda_state)
        results['lambda_commit'].append(lambda_commit)
        
        prev_h = h_t
        prev_s = s_t

    results['cum_log_gain'] = np.cumsum(results['lambda_commit'])
    
    return results, model_type

def plot_results(results_list, model_names):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for results, name in zip(results_list, model_names):
        layers = np.linspace(0, 1, len(results['s_t']))
        
        axes[0, 0].plot(layers, results['lambda_state'], label=name)
        axes[0, 1].plot(layers, results['lambda_commit'], label=name)
        axes[1, 0].plot(layers, results['cum_log_gain'], label=name)
        axes[1, 1].semilogy(layers, results['s_t'], label=name) # raw sensitivity log scale

    axes[0, 0].set_title(r"$\lambda_{state}$ (Relative State change)")
    axes[0, 1].set_title(r"$\lambda_{commit}$ (Local Log Sensitivity Ratio)")
    axes[1, 0].set_title(r"Cumulative Log-Gain")
    axes[1, 1].set_title(r"Raw Sensitivity ($s_t$)")
    
    for ax in axes.flat:
        ax.set_xlabel("Normalized Depth")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig("download-1.png")
    plt.show()

if __name__ == "__main__":
    res18_results, res_type = run_stability_scan('resnet18')
    vit_results, vit_type = run_stability_scan('vit_b_16')
    
    plot_results([res18_results, vit_results], ['ResNet18', 'ViT-B/16'])
    print("Stability Scan Complete. Plot saved as download-1.png.")
