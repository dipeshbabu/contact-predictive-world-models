#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import pandas as pd


def read_last_probe_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return {}
    last = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            for key, value in row.items():
                if key.startswith("report/contact_probe/"):
                    try:
                        last[key.replace("report/contact_probe/", "probe_")] = float(value)
                    except Exception:
                        pass
    return last


def infer_run_fields(run_dir: str):
    name = Path(str(run_dir)).name
    match = re.match(r"(.+?)_([A-Za-z0-9_]+)_s(\d+)$", name)
    if not match:
        return None
    return {
        "env": match.group(1),
        "variant": match.group(2),
        "seed": int(match.group(3)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results/results.csv")
    ap.add_argument("--out", default="outputs/results/contact_analysis.csv")
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "mass_scale" not in df:
        df["mass_scale"] = 1.0
    if "friction_scale" not in df:
        df["friction_scale"] = 1.0
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    df["proprio_noise"] = pd.to_numeric(df["proprio_noise"], errors="coerce").fillna(0.0)
    df["tactile_dropout"] = pd.to_numeric(df["tactile_dropout"], errors="coerce").fillna(0.0)
    df["mass_scale"] = pd.to_numeric(df["mass_scale"], errors="coerce").fillna(1.0)
    df["friction_scale"] = pd.to_numeric(df["friction_scale"], errors="coerce").fillna(1.0)
    df = df[df["success"].notna()].copy()
    df = df[(df["mass_scale"] == 1.0) & (df["friction_scale"] == 1.0)]

    clean = df[(df["proprio_noise"] == 0.0) & (df["tactile_dropout"] == 0.0)]
    pert = df[(df["proprio_noise"] > 0.0) | (df["tactile_dropout"] > 0.0)]

    clean_mean = clean.groupby(["env", "variant"])["success"].mean().reset_index().rename(columns={"success": "clean_success"})
    pert_mean = pert.groupby(["env", "variant"])["success"].mean().reset_index().rename(columns={"success": "perturbed_success"})

    merged = clean_mean.merge(pert_mean, on=["env", "variant"], how="outer")
    merged["robustness_gap"] = merged["clean_success"] - merged["perturbed_success"]
    numeric_cols = ["clean_success", "perturbed_success", "robustness_gap"]
    merged[numeric_cols] = merged[numeric_cols].round(args.decimals)

    probe_rows = []
    for run_dir in sorted(set(str(x) for x in df.get("run_dir", []) if str(x) and str(x) != "nan")):
        fields = infer_run_fields(run_dir)
        if fields is None:
            continue
        metrics = read_last_probe_metrics(Path(run_dir) / "metrics.jsonl")
        if not metrics:
            continue
        probe_rows.append({**fields, **metrics})

    if probe_rows:
        probes = pd.DataFrame(probe_rows)
        metric_cols = [c for c in probes.columns if c.startswith("probe_")]
        probes = (
            probes.groupby(["env", "variant"])[metric_cols]
            .mean()
            .reset_index()
        )
        probes[metric_cols] = probes[metric_cols].round(args.decimals)
        merged = merged.merge(probes, on=["env", "variant"], how="outer")

    merged = merged.sort_values(["env", "variant"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
