from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import time

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the versioned football model bundle sequentially."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "training" / "config" / "training-plan.yaml",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("objects", "ball", "pitch"),
        default=("objects", "ball", "pitch"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(name: str, config: dict, device: str, workers: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "training" / "scripts" / "train_yolo.py"),
        "--task",
        str(config["task"]),
        "--data",
        str(ROOT / config["data"]),
        "--model",
        str(config["base_model"]),
        "--name",
        str(config["run_name"]),
        "--epochs",
        str(config["epochs"]),
        "--imgsz",
        str(config["image_size"]),
        "--batch",
        str(config["batch"]),
        "--patience",
        str(config["patience"]),
        "--device",
        device,
        "--workers",
        str(workers),
    ]


def main() -> None:
    args = parse_args()
    plan = yaml.safe_load(args.plan.resolve().read_text(encoding="utf-8"))
    device_profile = plan["device_profile"]
    device = str(device_profile["device"])
    workers = int(device_profile["workers"])
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full bundle training")

    state_path = ROOT / "training" / "cache" / "bundle-training-state.json"
    state = {
        "plan": str(args.plan.resolve()),
        "started_at_unix": time(),
        "device": device,
        "models": {},
    }
    for name in args.only:
        config = plan["models"][name]
        command = command_for(name, config, device, workers)
        state["models"][name] = {
            "status": "planned" if args.dry_run else "running",
            "command": command,
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[{name}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        started = time()
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError:
            state["models"][name].update(
                {
                    "status": "failed",
                    "elapsed_seconds": round(time() - started, 2),
                }
            )
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            raise
        state["models"][name].update(
            {
                "status": "completed",
                "elapsed_seconds": round(time() - started, 2),
            }
        )
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    state["finished_at_unix"] = time()
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Bundle training state: {state_path}")


if __name__ == "__main__":
    main()
