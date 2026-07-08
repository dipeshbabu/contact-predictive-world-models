#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

KNOWN_VARIANTS = (
    "future1_noact",
    "proprio_only",
    "ppo_proprio",
    "ppo_tactile",
    "aux_contact",
    "future1",
    "future3",
    "future5",
    "masked",
    "tactile_group",
    "contact_frontier",
    "frontier_rgb",
    "frontier",
    "contact_onset",
    "contact_change",
    "both_onset",
    "both_change",
    "aux_contact_onset",
    "aux_contact_change",
    "contact1",
    "contact3",
    "current",
    "proprio",
    "contact",
    "noact",
    "recon",
    "both",
    "base",
    "aux",
)


def read_last_metric(metrics_path: Path, candidate_keys):
    if not metrics_path.exists():
        return None
    last = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for key in candidate_keys:
                if key in obj:
                    last = obj[key]
    if last is None:
        return None
    try:
        return float(last)
    except Exception:
        return None


def infer_run_fields(run_dir: Path):
    match = re.match(r"(.+)_s(\d+)$", run_dir.name)
    if not match:
        return None
    stem, seed = match.group(1), int(match.group(2))
    for variant in sorted(KNOWN_VARIANTS, key=len, reverse=True):
        suffix = f"_{variant}"
        if stem.endswith(suffix):
            return {
                "env": stem[: -len(suffix)],
                "variant": variant,
                "seed": seed,
            }
    env, variant = stem.rsplit("_", 1)
    return {
        "env": env,
        "variant": variant,
        "seed": seed,
    }


def norm_path(value):
    if value is None:
        return ""
    text = str(value)
    if not text or text == "nan":
        return ""
    return os.path.normcase(os.path.abspath(text))


def candidate_run_dirs(results: pd.DataFrame, runs_dir: Path):
    paths = []
    if "run_dir" in results:
        for value in results["run_dir"].dropna():
            text = str(value)
            if text and text != "nan":
                paths.append(Path(text))
    if runs_dir.exists():
        paths.extend(p for p in runs_dir.iterdir() if p.is_dir())

    unique = {}
    for path in paths:
        unique.setdefault(norm_path(path), path)
    return [unique[key] for key in sorted(unique)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="outputs/runs")
    ap.add_argument("--results_csv", default="outputs/results/results.csv")
    ap.add_argument("--out_csv", default="outputs/results/tactile_error_vs_success.csv")
    ap.add_argument("--out_fig", default="outputs/figs/tactile_error_vs_success.png")
    args = ap.parse_args()

    results = pd.read_csv(args.results_csv)
    if "mass_scale" not in results:
        results["mass_scale"] = 1.0
    if "friction_scale" not in results:
        results["friction_scale"] = 1.0
    for col in ("seed", "success", "proprio_noise", "tactile_dropout", "mass_scale", "friction_scale"):
        results[col] = pd.to_numeric(results[col], errors="coerce")
    clean = results[
        (results["proprio_noise"] == 0.0)
        & (results["tactile_dropout"] == 0.0)
        & (results["mass_scale"] == 1.0)
        & (results["friction_scale"] == 1.0)
    ].copy()
    if "run_dir" in clean:
        clean["_run_dir_norm"] = clean["run_dir"].map(norm_path)
    else:
        clean["_run_dir_norm"] = ""

    rows = []
    runs_dir = Path(args.runs_dir)
    skipped = {
        "name": 0,
        "baseline": 0,
        "metric": 0,
        "success": 0,
    }

    for run_dir in candidate_run_dirs(results, runs_dir):
        if not run_dir.is_dir():
            continue
        fields = infer_run_fields(run_dir)
        if fields is None:
            skipped["name"] += 1
            continue
        env, variant, seed = fields["env"], fields["variant"], fields["seed"]
        if variant in ("base", "ppo_proprio", "ppo_tactile"):
            skipped["baseline"] += 1
            continue

        metrics_path = run_dir / "metrics.jsonl"
        tactile_error = read_last_metric(
            metrics_path,
            [
                "tactile_aux_loss",
                "tactile_aux_loss_mean",
                "train/tactile_aux_loss",
                "train/tactile_aux_loss_mean",
            ],
        )
        if tactile_error is None:
            skipped["metric"] += 1
            continue

        run_norm = norm_path(run_dir)
        success_rows = clean[clean["_run_dir_norm"] == run_norm]
        if success_rows.empty:
            success_rows = clean[
                (clean["env"] == env)
                & (clean["variant"] == variant)
                & (clean["seed"] == seed)
            ]

        success = success_rows["success"].dropna()
        if success.empty:
            skipped["success"] += 1
            continue

        rows.append(
            {
                "env": env,
                "variant": variant,
                "seed": seed,
                "tactile_prediction_error": tactile_error,
                "success": float(success.iloc[0]),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "env",
            "variant",
            "seed",
            "tactile_prediction_error",
            "success",
            "run_dir",
            "metrics_path",
        ],
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if not df.empty:
        for env, sub in df.groupby("env"):
            plt.scatter(sub["tactile_prediction_error"], sub["success"], label=env)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No tactile aux runs found", ha="center", va="center")
    plt.xlabel("Tactile prediction error")
    plt.ylabel("Clean success")
    plt.title("Tactile prediction error versus success")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print("wrote", out_csv)
    print("wrote", out_fig)
    print(f"matched tactile aux runs: {len(df)}")
    print("skipped", skipped)


if __name__ == "__main__":
    main()
