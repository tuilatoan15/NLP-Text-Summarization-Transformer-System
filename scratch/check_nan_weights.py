import torch
from safetensors.torch import load_file
import os

model_path = "models/vit5-finetuned/model.safetensors"
if not os.path.exists(model_path):
    print("Safetensors file not found at", model_path)
    exit(1)

print("Loading safetensors weights...")
weights = load_file(model_path)

has_nan = False
has_inf = False
total_params = 0

for name, tensor in weights.items():
    total_params += tensor.numel()
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    if nan_count > 0:
        print(f"Tensor {name} has {nan_count} NaN values!")
        has_nan = True
    if inf_count > 0:
        print(f"Tensor {name} has {inf_count} Inf values!")
        has_inf = True

if not has_nan and not has_inf:
    print(f"All {total_params:,} weights are healthy! No NaN or Inf values found.")
else:
    print("CRITICAL WARNING: The model weights contain NaN or Inf values. The model is mathematically corrupted!")
