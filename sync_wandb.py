"""Sync offline metrics and figures to Weights & Biases (W&B) with strictly monotonic steps."""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import wandb


def sync_run(
    output_dir: str | Path,
    project: str = "celeba-flow-dit",
    name: str | None = None,
    job_type: str = "train",
    entity: str | None = None,
):
    output_path = Path(output_dir)
    config_file = output_path / "config.json"
    metrics_file = output_path / "metrics.jsonl"
    train_log = output_path / "train.log"
    figures_dir = output_path / "figures"
    if not figures_dir.exists():
        figures_dir = output_path  # files may be directly in output_dir

    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)

    run_name = name or output_path.name
    print(f"--> Syncing {run_name} to W&B project '{project}'...")

    step_events = defaultdict(dict)

    # 1. Parse metrics.jsonl if present
    if metrics_file.exists():
        print(f"    Reading metrics from {metrics_file}...")
        with open(metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    step = payload.pop("step", None)
                    if step is not None:
                        step_events[int(step)].update(payload)
                except Exception:
                    pass
    elif train_log.exists():
        print(f"    Parsing log from {train_log}...")
        pattern = re.compile(r"epoch=(\d+)\s+step=(\d+)\s+loss=([0-9.]+)")
        with open(train_log, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    ep, st, ls = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    step_events[st].update({"epoch": ep, "loss": ls})

    # 2. Add sample images
    sample_files = sorted(figures_dir.glob("sample_*.png"))
    print(f"    Found {len(sample_files)} sample figures...")
    for sample_file in sample_files:
        try:
            # find digits in stem
            digits = re.findall(r"\d+", sample_file.stem)
            if digits:
                step = int(digits[-1])
                step_events[step]["samples/face_grid"] = wandb.Image(str(sample_file), caption=f"Sample at step {step}")
        except Exception:
            pass

    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        job_type=job_type,
        config=config,
        reinit=True,
    )

    sorted_steps = sorted(step_events.keys())
    print(f"    Logging {len(sorted_steps)} unique steps...")
    for step in sorted_steps:
        wandb.log(step_events[step], step=step)

    summary_media = {}
    for cur in figures_dir.glob("curated_*.png"):
        summary_media[f"eval/{cur.stem}"] = wandb.Image(str(cur))
    for sw in figures_dir.glob("sweep_*.png"):
        summary_media[f"eval/{sw.stem}"] = wandb.Image(str(sw))
    if (figures_dir / "loss_curve.png").exists():
        summary_media["eval/loss_curve"] = wandb.Image(str(figures_dir / "loss_curve.png"))

    if summary_media:
        wandb.log(summary_media)

    url = run.url
    run.finish()
    print(f"--> Successfully synced {run_name} to {url}\n")
    return url


def main():
    parser = argparse.ArgumentParser(description="Sync runs to W&B")
    parser.add_argument("--dir", type=str, required=True, help="Path to outputs/<exp>")
    parser.add_argument("--project", type=str, default="celeba-flow-dit")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    sync_run(args.dir, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
