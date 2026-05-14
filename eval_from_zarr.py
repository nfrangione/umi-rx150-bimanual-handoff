#!/usr/bin/env python3
"""
Evaluate a trained UMI Diffusion Policy checkpoint against ground-truth actions
from the same zarr dataset format used during training.

FIXED VERSION:
- Uses UMI's own dataset class from the checkpoint Hydra config.
- Does NOT manually reconstruct obs from raw zarr keys.
- Avoids missing derived keys like robot0_eef_pos_wrt1.
- Compares policy output result["action"] against sample["action"].
- Fails loudly instead of silently reporting fake 0.00 cm errors.

Usage on NRP:

  cd /opt/umi
  export PYTHONPATH=/opt/umi:$PYTHONPATH
  conda run -n umi python -u /workspace/eval_from_zarr.py \
    --checkpoint /workspace/checkpoints/may326_002/checkpoints/latest.ckpt \
    --zarr /workspace/data/may326.zarr.zip \
    --num_samples 200 \
    --output_dir /workspace/eval_results
"""

import argparse
import json
import os
import random
from typing import Any, Dict

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import imagecodecs_numcodecs
    imagecodecs_numcodecs.register_codecs()
    print("Registered imagecodecs_numcodecs")
except Exception as e:
    print(f"Warning: could not register imagecodecs_numcodecs: {e}")


def to_device_batch(x: Any, device: torch.device) -> Any:
    """Convert dataset sample obs to tensors, add batch dim, and move to device."""
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0).to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).unsqueeze(0).to(device)
    if isinstance(x, dict):
        return {k: to_device_batch(v, device) for k, v in x.items()}
    raise TypeError(f"Unsupported type for batching: {type(x)}")


def tensor_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")


def load_workspace_policy(checkpoint_path: str, dataset_path: str, device: torch.device):
    print(f"Loading checkpoint: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location=device)

    print("Checkpoint keys:", list(payload.keys()))

    cfg = payload["cfg"]
    print("Workspace class:", cfg._target_)

    # Critical Hydra override: nested dataset path, not task.dataset_path.
    cfg.task.dataset.dataset_path = dataset_path

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("Using EMA model")
    else:
        policy = workspace.model
        print("Using raw model")

    policy.to(device)
    policy.eval()

    print("Policy loaded successfully")
    return cfg, workspace, policy


def load_training_dataset(cfg):
    print("Instantiating dataset from cfg.task.dataset")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))

    if "obs" not in sample:
        raise KeyError(f"Dataset sample has no 'obs'. Keys: {list(sample.keys())}")
    if "action" not in sample:
        raise KeyError(f"Dataset sample has no 'action'. Keys: {list(sample.keys())}")

    print("Obs keys:", list(sample["obs"].keys()))
    for k, v in sample["obs"].items():
        arr = tensor_to_numpy(v) if isinstance(v, torch.Tensor) else np.asarray(v)
        print(f"  obs[{k}] shape={arr.shape} dtype={arr.dtype}")

    action_arr = tensor_to_numpy(sample["action"]) if isinstance(sample["action"], torch.Tensor) else np.asarray(sample["action"])
    print(f"Action shape: {action_arr.shape} dtype={action_arr.dtype}")

    return dataset


def extract_policy_action(policy_output: Dict[str, Any]) -> np.ndarray:
    """Extract first predicted action vector from policy output."""
    if not isinstance(policy_output, dict):
        raise TypeError(f"Policy output should be dict, got {type(policy_output)}")

    if not getattr(extract_policy_action, "_printed_keys", False):
        print("Policy output keys:", list(policy_output.keys()))
        extract_policy_action._printed_keys = True

    if "action" in policy_output:
        action = policy_output["action"]
    elif "action_pred" in policy_output:
        action = policy_output["action_pred"]
    else:
        raise KeyError(f"No action key found in policy output. Keys: {list(policy_output.keys())}")

    action_np = tensor_to_numpy(action)

    if action_np.ndim == 3:
        # (B, Ta, Da) -> first batch, first action step
        return action_np[0, 0].astype(np.float32)
    if action_np.ndim == 2:
        # Usually (B, Da) or (Ta, Da); use first row
        return action_np[0].astype(np.float32)
    if action_np.ndim == 1:
        return action_np.astype(np.float32)

    raise ValueError(f"Unexpected policy action shape: {action_np.shape}")


def extract_ground_truth_action(sample: Dict[str, Any]) -> np.ndarray:
    """Extract first ground-truth action vector from dataset sample."""
    action_np = tensor_to_numpy(sample["action"])

    if action_np.ndim == 2:
        # (Ta, Da) -> first action step
        return action_np[0].astype(np.float32)
    if action_np.ndim == 1:
        return action_np.astype(np.float32)

    raise ValueError(f"Unexpected ground-truth action shape: {action_np.shape}")


def evaluate(policy, dataset, device, num_samples: int, seed: int):
    rng = random.Random(seed)
    n = len(dataset)
    indices = list(range(n))

    if num_samples is not None and num_samples > 0 and num_samples < n:
        indices = rng.sample(indices, num_samples)

    print(f"Evaluating {len(indices)} samples from dataset of length {n}")

    rows = []
    maes = []
    mses = []
    l2s = []
    max_abses = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i % 25 == 0:
                print(f"Sample {i}/{len(indices)} idx={idx}")

            sample = dataset[idx]

            obs_batch = to_device_batch(sample["obs"], device)
            gt_action = extract_ground_truth_action(sample)

            policy_output = policy.predict_action(obs_batch)
            pred_action = extract_policy_action(policy_output)

            min_len = min(len(pred_action), len(gt_action))
            pred = pred_action[:min_len]
            gt = gt_action[:min_len]

            err = pred - gt
            mae = float(np.mean(np.abs(err)))
            mse = float(np.mean(err ** 2))
            rmse = float(np.sqrt(mse))
            l2 = float(np.linalg.norm(err))
            max_abs = float(np.max(np.abs(err)))

            maes.append(mae)
            mses.append(mse)
            l2s.append(l2)
            max_abses.append(max_abs)

            rows.append({
                "sample_index": int(idx),
                "action_dim_compared": int(min_len),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "l2": l2,
                "max_abs": max_abs,
                "pred_first_12": pred[:12].tolist(),
                "gt_first_12": gt[:12].tolist(),
            })

    return {
        "num_samples": len(rows),
        "mean_mae": float(np.mean(maes)) if maes else None,
        "median_mae": float(np.median(maes)) if maes else None,
        "mean_mse": float(np.mean(mses)) if mses else None,
        "mean_rmse": float(np.sqrt(np.mean(mses))) if mses else None,
        "mean_l2": float(np.mean(l2s)) if l2s else None,
        "median_l2": float(np.median(l2s)) if l2s else None,
        "mean_max_abs": float(np.mean(max_abses)) if max_abses else None,
        "rows": rows,
    }


def save_outputs(summary: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "policy_vs_zarr_eval.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON: {json_path}")

    csv_path = os.path.join(output_dir, "policy_vs_zarr_eval.csv")
    with open(csv_path, "w") as f:
        f.write("sample_index,action_dim_compared,mae,mse,rmse,l2,max_abs\n")
        for r in summary["rows"]:
            f.write(
                f'{r["sample_index"]},{r["action_dim_compared"]},'
                f'{r["mae"]},{r["mse"]},{r["rmse"]},{r["l2"]},{r["max_abs"]}\n'
            )
    print(f"Saved CSV: {csv_path}")

    if summary["rows"]:
        xs = np.arange(len(summary["rows"]))
        maes = np.array([r["mae"] for r in summary["rows"]])
        l2s = np.array([r["l2"] for r in summary["rows"]])

        plt.figure(figsize=(12, 5))
        plt.plot(xs, maes, label="MAE")
        plt.xlabel("Evaluation sample order")
        plt.ylabel("Mean absolute error")
        plt.title("Policy vs ground-truth action MAE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        mae_plot = os.path.join(output_dir, "mae_by_sample.png")
        plt.savefig(mae_plot, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {mae_plot}")

        plt.figure(figsize=(12, 5))
        plt.plot(xs, l2s, label="L2 error")
        plt.xlabel("Evaluation sample order")
        plt.ylabel("L2 error")
        plt.title("Policy vs ground-truth action L2 error")
        plt.grid(True, alpha=0.3)
        plt.legend()
        l2_plot = os.path.join(output_dir, "l2_by_sample.png")
        plt.savefig(l2_plot, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {l2_plot}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of dataset samples to evaluate. Use -1 for all.")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="Deprecated compatibility arg; ignored.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="/workspace/eval_results")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.num_episodes is not None:
        print("Note: --num_episodes is ignored by this fixed evaluator.")
        print("      It evaluates processed UmiDataset samples, not raw episodes.")
        print("      Use --num_samples to control runtime.")

    num_samples = None if args.num_samples == -1 else args.num_samples

    cfg, workspace, policy = load_workspace_policy(args.checkpoint, args.zarr, device)
    dataset = load_training_dataset(cfg)

    summary = evaluate(policy, dataset, device, num_samples=num_samples, seed=args.seed)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        if k != "rows":
            print(f"{k}: {v}")

    save_outputs(summary, args.output_dir)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
