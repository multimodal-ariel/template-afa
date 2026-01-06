from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import hydra as hd
from omegaconf import OmegaConf
import mylib
import mymodels
import tensordict as thd
import copy
import csv
import time


from diff_models_original import SurrogateClassifier, TemplatePolicy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate differentiable template policy.")
    parser.add_argument("--dataset", "-d", type=str, default="mnist", help="Dataset name (mnist, big5, grid, gas, cube).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (0-3).")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained policy checkpoint")
    return parser.parse_args()

OmegaConf.register_new_resolver(
    name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
)

PROJ_ROOT: str = mylib.utils.get_project_root_dir()
print("*** PROJ_ROOT:", PROJ_ROOT)
args = parse_args()
_dataset_name = args.dataset

# get configs for dataset
if _dataset_name == "mnist":
    mktmpl_run_dir = f"experiments/make_template/outputs/mnist_cnnet/20250326_003820/0"
elif _dataset_name == "big5":
    mktmpl_run_dir = f"experiments/make_template/outputs/big5_cnnet/20250318_144121/0"
elif _dataset_name == "grid":
    mktmpl_run_dir = f"experiments/make_template/outputs/grid_cnnet/20250325_213622/0"
elif _dataset_name == "gas":
    mktmpl_run_dir = f"experiments/make_template/outputs/gas_cnnet/20250324_224734/0"
elif _dataset_name == "cube":
    mktmpl_run_dir = f"experiments/make_template/outputs/cube/20250318_225416/0"
elif _dataset_name == "fashion":
    mktmpl_run_dir = f"experiments/make_template/outputs/fashion_cnnet/20250326_003859/0"
else:
    raise ValueError(f"Unknown dataset: {_dataset_name}")

print(f"Loading config from: {mktmpl_run_dir}")

# load data
_tdata_shuffle_idxs = torch.load(os.path.join(PROJ_ROOT, mktmpl_run_dir, "tdata_shuffle_idxs.pt"))
tmpls = torch.load(os.path.join(PROJ_ROOT, mktmpl_run_dir, "tmpls.pt"), weights_only=False)
mktmpl_cfg = OmegaConf.load(os.path.join(PROJ_ROOT, mktmpl_run_dir, ".hydra", "config.yaml"))
_tdata, vdata, tstdata = hd.utils.call(mktmpl_cfg.data)
n_covs = _tdata["xs"].shape[1]
n_labels = len(torch.unique(_tdata["ys"]))

# using gpu
if torch.cuda.is_available():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        target_device_idx = 0 
    else:
        target_device_idx = args.gpu
    device = torch.device(f"cuda:{target_device_idx}")
else:
    raise EnvironmentError("GPU not available")

tmpls = tmpls.to(device).float()
X_test = vdata["xs"].to(device).float()
Y_test = vdata["ys"].to(device).long()

start_dim = mktmpl_cfg.init_fidx
alpha = mktmpl_cfg.lmbda

print(f"*** Starting Evaluation: {_dataset_name} | Lambda: {alpha} | Device: {device}", flush=True)

# helpers
def get_next_feature_batch(needed_mask):
    """
    Get next feat in batch
    """
    B, d = needed_mask.shape
    
    # check which samples still need features
    has_features = (needed_mask.sum(dim=1) > 0) # [B]
    next_indices = torch.argmax(needed_mask, dim=1) # [B]
    
    # update mask
    update_mask = torch.zeros_like(needed_mask)
    update_mask.scatter_(1, next_indices.unsqueeze(1), 1.0)
    
    # remove samples that do not need features
    update_mask = update_mask * has_features.unsqueeze(1).float()
    
    return update_mask, has_features

def evaluate_policy(policy, x, y, start_dim, alpha, batch_size=1024, max_steps_eval=None):
    """
    Eval
    """
    policy.eval()
    total_ce_loss = 0
    total_acc = 0
    total_feat_count = 0
    total_samples = 0
    
    if max_steps_eval is None: 
        max_steps = x.shape[1] 
    else: 
        max_steps = max_steps_eval
    
    with torch.no_grad():
        all_hard_tmpls = policy.get_hard_templates()
        all_hard_tmpls = (all_hard_tmpls > 0.5).float() # binary

        for i in range(0, x.size(0), batch_size):
            x_b = x[i:i+batch_size]
            y_b = y[i:i+batch_size]
            
            curr_mask = torch.zeros_like(x_b)
            curr_mask[:, start_dim] = 1.0
            
            # rollout
            for _ in range(max_steps):
                logits = policy(x_b, curr_mask)
                best_tmpl_idx = torch.argmax(logits, dim=1)
                
                # next mask 
                target_mask = all_hard_tmpls[best_tmpl_idx]
                
                needed = torch.relu(target_mask - curr_mask)
                needed = (needed > 0.5).float() 
                
                update_mask, active_samples = get_next_feature_batch(needed)
                curr_mask = torch.maximum(curr_mask, update_mask)
                
                if not active_samples.any():
                    break
            
            # --- Metrics ---
            feat_count = torch.sum(curr_mask, dim=1)
            pred_logits = policy.surrogate(x_b, curr_mask)
            
            preds = torch.argmax(pred_logits, dim=1)
            acc = (preds == y_b).float().sum()
            
            total_acc += acc.item()
            total_feat_count += feat_count.sum().item()
            total_samples += x_b.size(0)
            
    avg_acc = total_acc / total_samples
    avg_cost = total_feat_count / total_samples
    
    
    return avg_acc, avg_cost


# load checkpoint
print("\n--- Initializing Model Architecture ---")
hidden_size = {"mnist": 512,"big5": 256, "grid": 256, "gas": 256, "cube": 256, "fashion": 512}[args.dataset]
surrogate = SurrogateClassifier(n_covs, n_labels, hidden_dim=hidden_size).to(device)

# Resolve model path
model_path = os.path.join(args.model_path)

print(f"Loading model weights from: {model_path}")
state_dict = torch.load(model_path, map_location=device)

policy = TemplatePolicy(tmpls, surrogate, n_covs, start_dim, hidden_dim=hidden_size, optimize_templates=True).to(device)

# Load weights
policy.load_state_dict(state_dict)
policy.eval()
print("Model loaded successfully.")

# eval
# if _dataset_name == 'mnist':
#     max_rollout_steps = 21
# elif _dataset_name == 'gas':
#     max_rollout_steps = 6
# else: 
max_rollout_steps = n_covs - 1 

print("\n--- Running Inference ---")
eval_batch_size = 16384


start_time = time.time_ns()
test_acc, test_cost = evaluate_policy(
    policy, 
    X_test, 
    Y_test, 
    start_dim, 
    alpha, 
    batch_size=eval_batch_size, 
    max_steps_eval=max_rollout_steps
)

end_time = time.time_ns()

timing = (end_time - start_time) / X_test.shape[0]  
print(f"Timing: {timing}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Cost:      {test_cost:.4f}")

header = ["test_cost", "test_acc", "time_per_sample_ns"]
row = [test_cost, test_acc, timing]

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
exp_name = "final"
saved_path = os.path.join(save_dir, f"timing_{exp_name}_{_dataset_name}.csv")

with open(saved_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(header)
        
    writer.writerow(row)