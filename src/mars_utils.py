
import json, os, random, platform, datetime
import numpy as np
import torch

def set_seed(seed: int):
    """Fix all random number generators so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN algorithms: slightly slower, but repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_run_config(config: dict, folder, run_name: str):
    """Save every setting of a run (seed, steps, guidance scale, LoRA rank...) to a JSON file."""
    import importlib.metadata as md
    config = dict(config)  # copy so we don't change the caller's dict
    config["run_name"] = run_name
    config["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    config["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    config["python"] = platform.python_version()
    # Record library versions so results can be reproduced later.
    pkgs = ["torch", "torchvision", "diffusers", "peft", "accelerate", "transformers",
            "timm", "torchmetrics", "open_clip_torch"]
    config["versions"] = {}
    for p in pkgs:
        try:
            config["versions"][p] = md.version(p)
        except md.PackageNotFoundError:
            config["versions"][p] = "not installed"
    path = os.path.join(str(folder), f"{run_name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print("Saved run config ->", path)
    return path

def phase_done(root, phase: str) -> bool:
    """True if a phase already finished (a marker file exists)."""
    return os.path.exists(os.path.join(str(root), f".done_{phase}"))

def mark_phase_done(root, phase: str):
    """Write a marker file saying this phase finished successfully."""
    open(os.path.join(str(root), f".done_{phase}"), "w").close()
