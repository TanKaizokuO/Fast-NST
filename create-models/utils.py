"""
Utility functions for training and evaluation
"""

import torch
import os
from pathlib import Path
import json


def save_checkpoint(model, optimizer, step, loss, save_path):
    """
    Save model checkpoint

    Args:
        model: TransformerNetwork
        optimizer: Optimizer
        step: Current training step
        loss: Current loss value
        save_path: Path to save checkpoint
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device="cpu"):
    """
    Load model checkpoint

    Args:
        model: TransformerNetwork
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load to

    Returns:
        step: Training step number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    step = checkpoint.get("step", 0)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path} (step {step})")
    return step


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config_dict, save_path):
    """Save training configuration"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {save_path}")


def load_config(config_path):
    """Load training configuration"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
