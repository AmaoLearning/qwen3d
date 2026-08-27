#!/usr/bin/env python3
"""Run isolated 3B RFT ablations in parallel and compare test metrics."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = (
    ("original_ce", "original", 1.0),
    ("paper_ratio", "paper_ratio", 1.0),
    ("neg_log_phi", "neg_log_phi", 1.0),
    ("focal_g0p5", "focal", 0.5),
    ("focal_g1", "focal", 1.0),
    ("focal_g2", "focal", 2.0),
    ("one_minus_log1p", "one_minus_log1p", 1.0),
)
METRICS = ("exact_match", "cider", "bleu", "meteor", "rouge")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=6563)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-val-workers", type=int, default=2)
    parser.add_argument("--sampling-frame-num", type=int, default=3)
    parser.add_argument("--launch-delay-seconds", type=float, default=20.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_eval_metrics(log_path):
    results = {}
    current_dataset = None
    dataset_re = re.compile(r"Evaluating on (\S+)")
    exact_re = re.compile(r"exact match:\s*([0-9.eE+-]+)")
    metric_re = re.compile(r"\b(cider|bleu|meteor|rouge):\s*([0-9.eE+-]+)")
    if not log_path.is_file():
        return results
    for line in log_path.read_text(errors="replace").splitlines():
        match = dataset_re.search(line)
        if match:
            name = match.group(1)
            current_dataset = "sqa3d" if "sqa3d" in name else "scanqa"
            results.setdefault(current_dataset, {})
        if current_dataset is None:
            continue
        match = exact_re.search(line)
        if match:
            results[current_dataset]["exact_match"] = float(match.group(1))
        for name, value in metric_re.findall(line):
            results[current_dataset][name] = float(value)
    return results


def write_comparison(campaign, jobs):
    parsed = {
        name: parse_eval_metrics(Path(job["log"])) for name, job in jobs.items()
    }
    baseline = parsed.get("baseline", {})
    payload = {"generated_at": utc_now(), "jobs": jobs, "metrics": parsed}
    (campaign / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Qwen-3D 3B RFT ablation comparison",
        "",
        "Deltas are relative to the untouched Qwen-3D checkpoint.",
        "",
        "| model | dataset | metric | value | delta vs baseline |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name in ("baseline", *(name for name, _, _ in VARIANTS)):
        for dataset in ("sqa3d", "scanqa"):
            for metric in METRICS:
                value = parsed.get(model_name, {}).get(dataset, {}).get(metric)
                base = baseline.get(dataset, {}).get(metric)
                value_text = "NA" if value is None else f"{value:.6f}"
                delta_text = (
                    "NA"
                    if value is None or base is None
                    else f"{value - base:+.6f}"
                )
                lines.append(
                    f"| {model_name} | {dataset} | {metric} | "
                    f"{value_text} | {delta_text} |"
                )
    lines.extend(("", "## Job status", ""))
    for name, job in jobs.items():
        lines.append(f"- `{name}`: returncode={job.get('returncode')} log=`{job['log']}`")
    (campaign / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.max_iter < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("max-iter and accumulation steps must be positive")
    if args.num_workers < 0 or args.num_val_workers < 0:
        raise ValueError("worker counts must be non-negative")

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign = (args.output_root or root / "output/rft_ablation" / stamp).resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / "logs").mkdir()
    (campaign / "models").mkdir()
    (campaign / "evaluations").mkdir()

    jobs = {}
    processes = []
    python = sys.executable
    for gpu, (name, loss_type, gamma) in enumerate(VARIANTS):
        output_root = campaign / "models" / name
        command = [
            python,
            str(root / "repro/train_rft_lora.py"),
            "--model-size", "3b",
            "--dataset", "all",
            "--eval-split", "test",
            "--loss-type", loss_type,
            "--gamma", str(gamma),
            "--num-gpus", "1",
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--num-workers", str(args.num_workers),
            "--num-val-workers", str(args.num_val_workers),
            "--sampling-frame-num", str(args.sampling_frame_num),
            "--max-iter", str(args.max_iter),
            "--output-root", str(output_root),
            "--full-run",
        ]
        jobs[name] = {
            "gpu": gpu,
            "kind": "train_and_test",
            "command": command,
            "log": str(campaign / "logs" / f"{name}.log"),
            "output_root": str(output_root),
        }

    baseline_command = [
        python,
        str(root / "repro/eval_rft_lora.py"),
        "--checkpoint", str(root / "models/qwen3d/qwen3d_3b.pth"),
        "--name", "baseline_test",
        "--model-size", "3b",
        "--dataset", "all",
        "--eval-split", "test",
        "--num-gpus", "1",
        "--num-workers", str(args.num_val_workers),
        "--sampling-frame-num", str(args.sampling_frame_num),
        "--output-root", str(campaign / "evaluations" / "baseline"),
    ]
    jobs["baseline"] = {
        "gpu": 7,
        "kind": "baseline_test",
        "command": baseline_command,
        "log": str(campaign / "logs" / "baseline.log"),
        "output_root": str(campaign / "evaluations" / "baseline"),
    }

    manifest = campaign / "status.json"
    manifest.write_text(json.dumps({"created_at": utc_now(), "jobs": jobs}, indent=2) + "\n")
    print(f"campaign={campaign}", flush=True)
    if args.dry_run:
        for name, job in jobs.items():
            print(name, "CUDA_VISIBLE_DEVICES=" + str(job["gpu"]), *job["command"])
        return

    for name, job in jobs.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        log_handle = open(job["log"], "w")
        process = subprocess.Popen(
            job["command"],
            cwd=root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job["pid"] = process.pid
        job["started_at"] = utc_now()
        processes.append((name, process, log_handle))
        manifest.write_text(json.dumps({"created_at": utc_now(), "jobs": jobs}, indent=2) + "\n")
        print(f"started {name} gpu={job['gpu']} pid={process.pid}", flush=True)
        if args.launch_delay_seconds:
            time.sleep(args.launch_delay_seconds)

    for name, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        jobs[name]["returncode"] = returncode
        jobs[name]["finished_at"] = utc_now()
        manifest.write_text(json.dumps({"updated_at": utc_now(), "jobs": jobs}, indent=2) + "\n")
        print(f"finished {name} returncode={returncode}", flush=True)

    write_comparison(campaign, jobs)
    failed = [name for name, job in jobs.items() if job.get("returncode") != 0]
    if failed:
        raise SystemExit("Failed jobs: " + ", ".join(failed))
    print(f"comparison={campaign / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
