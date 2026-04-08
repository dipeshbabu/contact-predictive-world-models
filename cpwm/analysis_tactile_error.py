#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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

    rows = []
    runs_dir = Path(args.runs_dir)
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            match = re.match(r"(.+?)_([A-Za-z0-9_]+)_s(\d+)$", run_dir.name)
            if not match:
                continue
            env, variant, seed = match.group(1), match.group(2), int(match.group(3))
            if variant != "aux":
                continue

            tactile_error = read_last_metric(
                run_dir / "metrics.jsonl",
                ["tactile_aux_loss", "tactile_aux_loss_mean"],
            )
            if tactile_error is None:
                continue

            success_rows = clean[
                (clean["env"] == env)
                & (clean["variant"] == variant)
                & (clean["seed"] == seed)
            ]
            if success_rows.empty:
                continue

            success = success_rows["success"].dropna()
            if success.empty:
                continue

            rows.append(
                {
                    "env": env,
                    "variant": variant,
                    "seed": seed,
                    "tactile_prediction_error": tactile_error,
                    "success": float(success.iloc[0]),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["env", "variant", "seed", "tactile_prediction_error", "success"],
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


if __name__ == "__main__":
    main()
